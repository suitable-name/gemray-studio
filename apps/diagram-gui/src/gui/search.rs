use crate::{
    DiagramItem, MainWindow,
    bridge::{library_source, library_source::LibrarySource},
    settings::WorkerSettings,
};
use anyhow::Result;
use diagram_catalog::{
    db::sqlite::Database,
    model::{entry::DiagramListItem, filter::RangeFilter},
};
use gemray_net::library::{DesignSummary, LibraryRequest, LibraryResponse, RangeFilterWire};
use slint::{ComponentHandle, ModelRc, VecModel};
use std::sync::{Arc, Mutex};
use tracing::error;

/// A slider value that sits exactly on its data-bounds edge means "not filtering on
/// this side" -- returns `None` so `search_diagrams` skips the predicate entirely
/// (which also means rows with no value at all for that attribute stay visible, same
/// as the unfiltered "All" state of the existing shape/gear dropdowns). Any other
/// value is an active bound.
fn active_bound(value: f32, bound_edge: f32) -> Option<f64> {
    let moved_off_edge = (value - bound_edge).abs() >= 1e-6;
    moved_off_edge.then_some(f64::from(value))
}

/// Reads the four range-filter sliders' current min/max values off `ui` and builds
/// the `RangeFilter` to hand to `search_diagrams`. See `active_bound` for what makes a
/// bound "active" versus effectively unset.
#[must_use]
pub fn read_range_filter(ui: &MainWindow) -> RangeFilter {
    RangeFilter {
        ri_min: active_bound(ui.get_ri_filter_min(), ui.get_ri_bounds_min()),
        ri_max: active_bound(ui.get_ri_filter_max(), ui.get_ri_bounds_max()),
        lw_min: active_bound(ui.get_lw_filter_min(), ui.get_lw_bounds_min()),
        lw_max: active_bound(ui.get_lw_filter_max(), ui.get_lw_bounds_max()),
        volume_min: active_bound(ui.get_volume_filter_min(), ui.get_volume_bounds_min()),
        volume_max: active_bound(ui.get_volume_filter_max(), ui.get_volume_bounds_max()),
        facets_min: active_bound(ui.get_facets_filter_min(), ui.get_facets_bounds_min())
            .map(|v| v.round() as i64),
        facets_max: active_bound(ui.get_facets_filter_max(), ui.get_facets_bounds_max())
            .map(|v| v.round() as i64),
    }
}

/// Dispatches to [`refresh_diagram_list`] or a background remote search.
///
/// LOCAL is unchanged -- see `bridge::library_source`'s module doc comment. Every one
/// of this crate's search/filter callbacks calls this instead of `refresh_diagram_list`
/// directly now, so switching sources changes what a search box or filter dropdown
/// actually queries without those callbacks themselves needing to know which source is
/// active.
pub fn refresh_diagram_list_via_source(
    ui: &MainWindow,
    db_mutex: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
    search: &str,
    shape_filter: &str,
    gear_filter: &str,
) {
    let current = source
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    match current {
        LibrarySource::Local => {
            refresh_diagram_list(ui, db_mutex, search, shape_filter, gear_filter);
        }
        LibrarySource::Remote(worker) => {
            refresh_diagram_list_remote(ui, worker, search, shape_filter, gear_filter);
        }
    }
}

/// The remote counterpart of [`refresh_diagram_list`]: one `LibraryRequest::Search`
/// against `worker`, off the UI thread. See `bridge::library_mirror`'s module doc
/// comment for the same 1000-result cap this shares with the local query (both are
/// ultimately backed by the same `Database::search_diagrams`, on whichever side of the
/// connection is running it). `pub(crate)`, not private: also called directly from
/// `gui::library_remote` right after a source switch succeeds, when the caller already
/// knows for certain the source is remote and has the `WorkerSettings` in hand (no
/// need to go through `refresh_diagram_list_via_source`'s `LibrarySource` re-check).
pub(crate) fn refresh_diagram_list_remote(
    ui: &MainWindow,
    worker: WorkerSettings,
    search: &str,
    shape_filter: &str,
    gear_filter: &str,
) {
    let clean_shape = if shape_filter == "All Shapes" {
        "All".to_string()
    } else {
        shape_filter.to_string()
    };
    let clean_gear = if gear_filter == "All Gears" {
        "All".to_string()
    } else {
        gear_filter.to_string()
    };
    let range = to_range_wire(&read_range_filter(ui));
    let request = LibraryRequest::Search {
        query: search.to_string(),
        shape_filter: clean_shape,
        gear_filter: clean_gear,
        range,
    };
    library_source::spawn_library_request(
        ui.as_weak(),
        worker,
        request,
        |ui, result| match result {
            Ok(LibraryResponse::SearchResults(items)) => {
                let total = items.len();
                let slint_items: Vec<DiagramItem> = items.iter().map(to_diagram_item).collect();
                ui.set_diagram_list(ModelRc::new(VecModel::from(slint_items)));
                ui.set_total_count(total as i32);
            }
            Ok(_) => {
                ui.set_status_message("Unexpected reply searching the remote library.".into());
            }
            Err(e) => {
                error!("Remote search failed: {e}");
                ui.set_status_message(format!("Remote search failed: {e}").into());
            }
        },
    );
}

const fn to_range_wire(range: &RangeFilter) -> RangeFilterWire {
    RangeFilterWire {
        ri_min: range.ri_min,
        ri_max: range.ri_max,
        lw_min: range.lw_min,
        lw_max: range.lw_max,
        volume_min: range.volume_min,
        volume_max: range.volume_max,
        facets_min: range.facets_min,
        facets_max: range.facets_max,
    }
}

pub(crate) fn to_diagram_item(item: &DesignSummary) -> DiagramItem {
    DiagramItem {
        id: item.entry_id as i32,
        title: item.title.clone().into(),
        shape: item.shape.clone().unwrap_or_default().into(),
        gear: item.index_gear.clone().unwrap_or_default().into(),
        facets: item.facets_count.clone().unwrap_or_default().into(),
        designer: item.designer_info.clone().unwrap_or_default().into(),
        lw_ratio: item.lw_ratio.clone().unwrap_or_default().into(),
        ri: item.refractive_index.clone().unwrap_or_default().into(),
    }
}

/// The `Database` read half of [`refresh_diagram_list`].
///
/// Has no `ui` dependency beyond the already-resolved `range` -- split out so
/// `gui::library`'s post-import / rename / delete / shape-change refresh can run this
/// on a background thread (see that module's own doc comment on
/// `refresh_after_library_change` for why: re-running an
/// unfiltered search against a several-thousand-design catalogue synchronously on the
/// UI thread is itself a perceptible freeze) and only marshal the cheap
/// [`apply_diagram_list_to_ui`] step back onto the UI thread. Every OTHER caller here
/// (search box edits, filter slider drags) stays on the synchronous
/// [`refresh_diagram_list`] below, which is what a filter that must feel immediate as
/// you type actually needs.
///
/// # Errors
///
/// Returns an error if the underlying `Database::search_diagrams` query fails (a bad
/// filter combination, or the connection itself). `get_total_count`'s own failure is
/// tolerated instead -- see its `unwrap_or(items.len())` below, matching
/// `refresh_diagram_list`'s pre-existing tolerance for that one call.
pub fn fetch_diagram_list(
    db_mutex: &Arc<Mutex<Database>>,
    search: &str,
    shape_filter: &str,
    gear_filter: &str,
    range: &RangeFilter,
) -> Result<(Vec<DiagramListItem>, usize)> {
    let clean_shape = if shape_filter == "All Shapes" {
        "All"
    } else {
        shape_filter
    };
    let clean_gear = if gear_filter == "All Gears" {
        "All"
    } else {
        gear_filter
    };

    // Scoped so the guard drops before this returns: both queries need the lock, the
    // return value needs none of it, and this runs on a background thread where any
    // extra time holding the database lock is time a UI-thread callback can block on it.
    let (items, total) = {
        let db = db_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let items = db.search_diagrams(search, clean_shape, clean_gear, range)?;
        let total = db.get_total_count().unwrap_or(items.len());
        // Explicit rather than waiting for the end of this block: moving the results
        // into the tuple needs no lock, and this runs while the UI thread may be
        // waiting on the very same mutex.
        drop(db);
        (items, total)
    };
    Ok((items, total))
}

/// The `ui.set_*` half of [`refresh_diagram_list`] -- see [`fetch_diagram_list`]'s doc
/// comment for why these are split.
pub fn apply_diagram_list_to_ui(ui: &MainWindow, items: Vec<DiagramListItem>, total: usize) {
    let slint_items: Vec<DiagramItem> = items
        .into_iter()
        .map(|item| DiagramItem {
            id: item.id as i32,
            title: item.title.into(),
            shape: item.shape.unwrap_or_default().into(),
            gear: item.index_gear.unwrap_or_default().into(),
            facets: item.facets_count.unwrap_or_default().into(),
            designer: item.designer_info.unwrap_or_default().into(),
            lw_ratio: item.lw_ratio.unwrap_or_default().into(),
            ri: item.refractive_index.unwrap_or_default().into(),
        })
        .collect();

    ui.set_diagram_list(ModelRc::new(VecModel::from(slint_items)));
    ui.set_total_count(total as i32);
}

pub fn refresh_diagram_list(
    ui: &MainWindow,
    db_mutex: &Arc<Mutex<Database>>,
    search: &str,
    shape_filter: &str,
    gear_filter: &str,
) {
    let range = read_range_filter(ui);
    match fetch_diagram_list(db_mutex, search, shape_filter, gear_filter, &range) {
        Ok((items, total)) => apply_diagram_list_to_ui(ui, items, total),
        Err(e) => {
            error!("Failed to search diagrams: {:?}", e);
            ui.set_status_message(format!("Error searching database: {e}").into());
        }
    }
}

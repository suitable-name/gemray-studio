//! Diagram-list loading, range-filter bounds, search/filter callbacks, and
//! diagram-selection/URL-open/file-export callbacks.
//!
//! Split out of `gui::mod` purely to keep that module (already sizeable) from growing
//! further -- same reasoning as `gui::detail`/`gui::search`/`gui::remote`.

use crate::{
    MainWindow,
    bridge::{library_source::LibrarySource, render_thread::RenderContext},
    gui::{
        detail::{export_diagram_file_via_source, load_diagram_detail_via_source},
        search::{refresh_diagram_list, refresh_diagram_list_via_source},
        show_toast,
    },
};
use diagram_catalog::{db::sqlite::Database, model::filter::AttributeRanges};
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::sync::{Arc, Mutex};
use tracing::info;

/// Loads the shape/gear filter dropdown options and the initial diagram list from
/// the database. Split out of `run_gui` purely to keep that function under clippy's
/// function-length lint.
pub(super) fn load_filter_options_and_initial_list(ui: &MainWindow, db: &Arc<Mutex<Database>>) {
    {
        let db_guard = db.lock().unwrap();
        let mut shape_opts = vec!["All Shapes".to_string()];
        if let Ok(shapes) = db_guard.get_unique_shapes() {
            shape_opts.extend(shapes);
        }
        let shape_model: Vec<SharedString> = shape_opts
            .into_iter()
            .map(std::convert::Into::into)
            .collect();
        ui.set_shape_options(ModelRc::new(VecModel::from(shape_model)));

        let mut gear_opts = vec!["All Gears".to_string()];
        if let Ok(gears) = db_guard.get_unique_gears() {
            gear_opts.extend(gears);
        }
        drop(db_guard);
        let gear_model: Vec<SharedString> = gear_opts
            .into_iter()
            .map(std::convert::Into::into)
            .collect();
        ui.set_gear_options(ModelRc::new(VecModel::from(gear_model)));
    }

    sync_range_bounds_to_ui(ui, db);
    refresh_diagram_list(ui, db, "", "All Shapes", "All Gears");

    let total_count = db.lock().unwrap().get_total_count().unwrap_or(0);
    ui.set_status_message(format!("Database loaded: {total_count} diagrams available.").into());
}

/// Pushes the catalogue's actual min/max (RI, L/W, volume, facet count) into the
/// range-filter sliders.
///
/// Sets both the `*_bounds_*` properties (the sliders' scale) and the `*_filter_*`
/// properties (their starting position, the full range, i.e. unfiltered). Called once
/// at startup, after the diagram list's filter dropdowns are populated and before the
/// first `refresh_diagram_list` call, so the very first query already has
/// correctly-scaled (inert) range filters rather than the sliders' `0.0..1.0`-ish
/// `.slint` placeholder defaults. Also re-called after anything that could have
/// widened the real min/max -- an import in this crate's own `gui::library`, or a
/// bulk library change made by a downstream binary -- see the `pub`-ness note below.
///
/// A query failure leaves the sliders at their placeholder bounds rather than
/// panicking -- matching `load_filter_options_and_initial_list`'s existing
/// `unwrap_or_default`/silent-failure tolerance for a database that can't be read.
/// **`pub`, not `pub(crate)`, deliberately.** A binary that reuses this crate as a
/// library (see `crate`'s own doc comment) must call this after any bulk change that
/// widens the catalogue's real min/max for a range-filterable attribute, exactly as
/// `gui::library`'s import handler in this crate does. Narrowing this to `pub(crate)`
/// would compile here and silently leave those sliders stale there.
///
/// Recovers rather than panics on a poisoned `db` mutex (`unwrap_or_else(
/// std::sync::PoisonError::into_inner)`, not a bare `.unwrap()`) -- this used to be
/// the one call site in this path that didn't, which meant a panic anywhere else
/// while holding the lock (e.g. mid-import -- see `gui::library`'s BUG 1 write-up)
/// poisoned it and then took this call down too, on the very next library-change
/// refresh. Matches the poison-recovery convention `gui::library` already uses
/// throughout.
pub fn sync_range_bounds_to_ui(ui: &MainWindow, db: &Arc<Mutex<Database>>) {
    let Some(ranges) = fetch_attribute_ranges(db) else {
        return;
    };
    apply_attribute_ranges_to_ui(ui, &ranges);
}

/// The `Database` read half of [`sync_range_bounds_to_ui`], with no `ui` dependency --
/// split out so `gui::library`'s post-import/rename/delete refresh can run this on a
/// background thread (see that module's own doc comment on `refresh_after_library_change`
/// for why: re-querying the catalogue synchronously on the UI thread is itself a
/// perceptible freeze against a several-thousand-design library) and only marshal the
/// cheap [`apply_attribute_ranges_to_ui`] step back onto the UI thread.
pub fn fetch_attribute_ranges(db: &Arc<Mutex<Database>>) -> Option<AttributeRanges> {
    db.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get_attribute_ranges()
        .ok()
}

/// The `ui.set_*` half of [`sync_range_bounds_to_ui`] -- see [`fetch_attribute_ranges`]'s
/// doc comment for why these are split. Sets both the `*_bounds_*` properties (the
/// sliders' scale) and the `*_filter_*` properties (reset to the full range, i.e.
/// unfiltered) from an already-fetched [`AttributeRanges`].
pub fn apply_attribute_ranges_to_ui(ui: &MainWindow, ranges: &AttributeRanges) {
    ui.set_ri_bounds_min(ranges.ri.0 as f32);
    ui.set_ri_bounds_max(ranges.ri.1 as f32);
    ui.set_ri_filter_min(ranges.ri.0 as f32);
    ui.set_ri_filter_max(ranges.ri.1 as f32);

    ui.set_lw_bounds_min(ranges.lw_ratio.0 as f32);
    ui.set_lw_bounds_max(ranges.lw_ratio.1 as f32);
    ui.set_lw_filter_min(ranges.lw_ratio.0 as f32);
    ui.set_lw_filter_max(ranges.lw_ratio.1 as f32);

    ui.set_volume_bounds_min(ranges.volume.0 as f32);
    ui.set_volume_bounds_max(ranges.volume.1 as f32);
    ui.set_volume_filter_min(ranges.volume.0 as f32);
    ui.set_volume_filter_max(ranges.volume.1 as f32);

    ui.set_facets_bounds_min(ranges.facets.0 as f32);
    ui.set_facets_bounds_max(ranges.facets.1 as f32);
    ui.set_facets_filter_min(ranges.facets.0 as f32);
    ui.set_facets_filter_max(ranges.facets.1 as f32);
}

/// Wires up the search-text, shape-filter, and gear-filter callbacks that re-run the
/// diagram list query. Split out of `run_gui` purely to keep that function under
/// clippy's function-length lint.
///
/// Every handler here calls [`refresh_diagram_list_via_source`], not
/// [`refresh_diagram_list`] directly -- with `source` at its default (`LibrarySource::Local`,
/// see that type's own doc comment), the dispatcher calls the exact same
/// `refresh_diagram_list` these handlers called before this dispatch existed, so nothing
/// changes here until a user actually switches sources.
pub(super) fn setup_search_and_filter_callbacks(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
) {
    let db_search = Arc::clone(db);
    let source_search = Arc::clone(source);
    let ui_weak_search = ui.as_weak();
    ui.on_search_changed(move |text: SharedString| {
        if let Some(ui) = ui_weak_search.upgrade() {
            let shape_idx = ui.get_selected_shape_index() as usize;
            let shape = ui
                .get_shape_options()
                .row_data(shape_idx)
                .unwrap_or_default();
            let gear_idx = ui.get_selected_gear_index() as usize;
            let gear = ui.get_gear_options().row_data(gear_idx).unwrap_or_default();

            refresh_diagram_list_via_source(&ui, &db_search, &source_search, &text, &shape, &gear);
        }
    });

    let db_shape = Arc::clone(db);
    let source_shape = Arc::clone(source);
    let ui_weak_shape = ui.as_weak();
    ui.on_filter_shape_changed(move |shape: SharedString| {
        if let Some(ui) = ui_weak_shape.upgrade() {
            let search = ui.get_search_text();
            let gear_idx = ui.get_selected_gear_index() as usize;
            let gear = ui.get_gear_options().row_data(gear_idx).unwrap_or_default();

            refresh_diagram_list_via_source(&ui, &db_shape, &source_shape, &search, &shape, &gear);
        }
    });

    let db_gear = Arc::clone(db);
    let source_gear = Arc::clone(source);
    let ui_weak_gear = ui.as_weak();
    ui.on_filter_gear_changed(move |gear: SharedString| {
        if let Some(ui) = ui_weak_gear.upgrade() {
            let search = ui.get_search_text();
            let shape_idx = ui.get_selected_shape_index() as usize;
            let shape = ui
                .get_shape_options()
                .row_data(shape_idx)
                .unwrap_or_default();

            refresh_diagram_list_via_source(&ui, &db_gear, &source_gear, &search, &shape, &gear);
        }
    });

    // Fired on every range-filter slider drag tick and by the panel's Reset button
    // (`header.slint`). `refresh_diagram_list` itself reads the current filter values
    // straight off `ui` (via `read_range_filter`), so this handler only needs to
    // re-supply the other three existing filters.
    let db_range = Arc::clone(db);
    let source_range = Arc::clone(source);
    let ui_weak_range = ui.as_weak();
    ui.on_filters_changed(move || {
        if let Some(ui) = ui_weak_range.upgrade() {
            let search = ui.get_search_text();
            let shape_idx = ui.get_selected_shape_index() as usize;
            let shape = ui
                .get_shape_options()
                .row_data(shape_idx)
                .unwrap_or_default();
            let gear_idx = ui.get_selected_gear_index() as usize;
            let gear = ui.get_gear_options().row_data(gear_idx).unwrap_or_default();

            refresh_diagram_list_via_source(&ui, &db_range, &source_range, &search, &shape, &gear);
        }
    });
}

/// Wires up diagram-selection, "open URL externally", and file-export callbacks.
/// Split out of `run_gui` purely to keep that function under clippy's
/// function-length lint.
///
/// `on_select_diagram`/`on_export_file` dispatch via `source` so a design
/// selected while browsing a remote library is looked up (and its attachments
/// exported) against THAT library, never the local database -- see
/// `gui::detail::load_diagram_detail_via_source`'s own doc comment on why that
/// dispatch matters (a remote entry id and a local row id are independent id spaces).
/// `on_open_diagram_url` needs no such dispatch: it only ever acts on the URL string
/// already loaded into `current_detail`, regardless of which source populated it.
pub(super) fn setup_diagram_selection_and_export_callbacks(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
    render_ctx: &Arc<Mutex<RenderContext>>,
) {
    let db_select = Arc::clone(db);
    let source_select = Arc::clone(source);
    let render_ctx_select = render_ctx.clone();
    let ui_weak_select = ui.as_weak();
    ui.on_select_diagram(move |id: i32| {
        if let Some(ui) = ui_weak_select.upgrade() {
            load_diagram_detail_via_source(
                &ui,
                &db_select,
                &source_select,
                &render_ctx_select,
                i64::from(id),
            );
        }
    });

    ui.on_open_diagram_url(move |url: SharedString| {
        let url_str = url.to_string();
        info!("Opening URL: {}", url_str);
        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open").arg(&url_str).spawn();
        #[cfg(target_os = "windows")]
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", &url_str])
            .spawn();
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg(&url_str).spawn();
    });

    let db_export = Arc::clone(db);
    let source_export = Arc::clone(source);
    let ui_weak_export = ui.as_weak();
    ui.on_export_file(move |file_name: SharedString| {
        if let Some(ui) = ui_weak_export.upgrade() {
            let entry_id = ui.get_selected_entry_id();
            if entry_id >= 0 {
                export_diagram_file_via_source(
                    &ui,
                    &db_export,
                    &source_export,
                    i64::from(entry_id),
                    &file_name,
                );
            } else {
                show_toast(&ui, "No diagram selected for export.", "error");
            }
        }
    });
}

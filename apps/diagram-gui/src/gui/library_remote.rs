//! Wiring for switching the browsed library source (`switch_library_source`) and
//! driving a pull-mirror sync (`start_mirror_sync`/`cancel_mirror_sync`).
//!
//! Everything that actually decides "local or remote" and "how to mirror" lives in
//! `bridge::library_source`/`bridge::library_client`/`bridge::library_mirror` -- this
//! module is purely the Slint-callback glue, following the same shape
//! `gui::remote::worker_callbacks` already uses for "Test connection" (a blocking call
//! on its own `std::thread::spawn` worker, result marshalled back via
//! `Weak::upgrade_in_event_loop`) and `gui::mod`'s export wiring (an `Rc<RefCell<Option<_>>>`
//! held on the UI thread so a running sync's handle can be reached by a later "Cancel"
//! click).

use crate::{
    MainWindow,
    bridge::{
        library_client,
        library_mirror::{self, MirrorHandle, MirrorOutcome, MirrorProgress},
        library_source::LibrarySource,
    },
    gui::show_toast,
    settings::{SettingsPersister, WorkerSettings},
};
use diagram_catalog::db::sqlite::Database;
use gemray_net::library::{AttributeRangesWire, LibraryRequest, LibraryResponse};
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex, PoisonError},
};

/// Wires the "Local"/"Browse library" switch (see `remote_worker_dialog.slint`).
///
/// Switching to a worker first PROBES it (handshake only, no data request -- mirrors
/// `bridge::remote_render::test_connection`'s own shape) on a background thread, and
/// only commits `source` to [`LibrarySource::Remote`] -- updating the always-visible
/// badge (`header.slint`) and loading that worker's filter options + initial list -- if
/// the probe succeeds and the worker actually advertises library capacity
/// (`ConnectionInfo::library`). A failure (unreachable, wrong certs, library-less
/// worker) shows a clear toast and leaves `source`/the badge exactly as they were --
/// the viewer stays on whatever it was already showing (local, on a fresh app with
/// nothing configured yet), never a half-switched, unusable state.
pub fn setup_library_source_callbacks(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
    settings_store: &Arc<SettingsPersister>,
) {
    let db_switch = Arc::clone(db);
    let source_switch = Arc::clone(source);
    let settings_switch = settings_store.clone();
    let ui_weak = ui.as_weak();
    ui.on_switch_library_source(move |idx: i32| {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };

        if idx < 0 {
            *source_switch
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = LibrarySource::Local;
            apply_source_badge(&ui, &LibrarySource::Local);
            ui.set_library_source_worker_index(-1);
            crate::gui::diagram_list::load_filter_options_and_initial_list(&ui, &db_switch);
            show_toast(&ui, "Switched to the local library.", "info");
            return;
        }

        let Some(worker) = settings_switch
            .snapshot()
            .settings
            .remote_workers
            .get(idx as usize)
            .cloned()
        else {
            show_toast(&ui, "No such remote worker.", "error");
            return;
        };

        ui.set_status_message(format!("Connecting to {}...", worker_display_name(&worker)).into());
        let source_probe = Arc::clone(&source_switch);
        let ui_weak_result = ui.as_weak();
        let worker_for_probe = worker;
        std::thread::spawn(move || {
            let result = library_client::probe(&worker_for_probe);
            let _ = ui_weak_result.upgrade_in_event_loop(move |ui| match result {
                Ok(info) if info.library => {
                    let new_source = LibrarySource::Remote(worker_for_probe);
                    *source_probe.lock().unwrap_or_else(PoisonError::into_inner) =
                        new_source.clone();
                    apply_source_badge(&ui, &new_source);
                    ui.set_library_source_worker_index(idx);
                    if let Some(w) = new_source.worker() {
                        load_filter_options_and_initial_list_remote(&ui, w.clone());
                    }
                    show_toast(&ui, &format!("Now browsing: {}", new_source.label()), "success");
                }
                Ok(_) => {
                    show_toast(
                        &ui,
                        "That worker is reachable but does not serve a design library -- staying on the current library.",
                        "error",
                    );
                    ui.set_status_message("Ready.".into());
                }
                Err(e) => {
                    show_toast(&ui, &format!("Could not switch library: {e}"), "error");
                    ui.set_status_message("Ready.".into());
                }
            });
        });
    });
}

/// Sets the header badge's label/remote-ness (`library_source_label`/`library_is_remote`)
/// to reflect `source`. `library_source_worker_index` is set separately at each call
/// site (the row index a click came from, or `-1` for local) -- `LibrarySource` itself
/// doesn't carry a worker-list index, only the resolved `WorkerSettings`.
fn apply_source_badge(ui: &MainWindow, source: &LibrarySource) {
    ui.set_library_source_label(source.label().into());
    ui.set_library_is_remote(source.is_remote());
}

/// Loads `worker`'s filter options (shapes/gears/attribute ranges) and initial diagram
/// list -- the remote counterpart of `gui::diagram_list::load_filter_options_and_initial_list`,
/// run when a switch to that worker just succeeded.
fn load_filter_options_and_initial_list_remote(ui: &MainWindow, worker: WorkerSettings) {
    let worker_for_search = worker.clone();
    crate::bridge::library_source::spawn_library_request(
        ui.as_weak(),
        worker,
        LibraryRequest::FilterOptions,
        move |ui, result| {
            match result {
                Ok(LibraryResponse::FilterOptions {
                    shapes,
                    gears,
                    ranges,
                }) => {
                    apply_shape_gear_options(ui, &shapes, &gears);
                    apply_range_wire_to_ui(ui, &ranges);
                }
                Ok(_) => {
                    ui.set_status_message("Unexpected reply loading remote filter options.".into());
                }
                Err(e) => ui.set_status_message(
                    format!("Could not load remote filter options: {e}").into(),
                ),
            }
            // Chained rather than parallel: both requests each pay their own connect
            // cost (see `bridge::library_client`'s module doc comment -- one connect+
            // handshake per request in this phase), and the initial list only needs to
            // exist once filter options have already populated the dropdowns/bounds
            // it's rendered alongside.
            crate::gui::search::refresh_diagram_list_remote(
                ui,
                worker_for_search,
                "",
                "All Shapes",
                "All Gears",
            );
        },
    );
}

/// A worker's display name for a status message -- its `name` if set, else its
/// `address`. The same fallback [`LibrarySource::label`] uses internally, but without
/// that method's `"Remote: "` prefix (not appropriate mid-sentence, as in "Connecting
/// to X...").
fn worker_display_name(worker: &WorkerSettings) -> &str {
    if worker.name.trim().is_empty() {
        &worker.address
    } else {
        &worker.name
    }
}

fn apply_shape_gear_options(ui: &MainWindow, shapes: &[String], gears: &[String]) {
    let mut shape_opts = vec!["All Shapes".to_string()];
    shape_opts.extend(shapes.iter().cloned());
    let shape_model: Vec<SharedString> = shape_opts.into_iter().map(Into::into).collect();
    ui.set_shape_options(ModelRc::new(VecModel::from(shape_model)));

    let mut gear_opts = vec!["All Gears".to_string()];
    gear_opts.extend(gears.iter().cloned());
    let gear_model: Vec<SharedString> = gear_opts.into_iter().map(Into::into).collect();
    ui.set_gear_options(ModelRc::new(VecModel::from(gear_model)));
    ui.set_selected_shape_index(0);
    ui.set_selected_gear_index(0);
}

fn apply_range_wire_to_ui(ui: &MainWindow, ranges: &AttributeRangesWire) {
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

/// Wires "Mirror to local" / "Cancel" (see `remote_worker_dialog.slint`).
///
/// One sync at a time, enforced by the UI itself: a second row's "Mirror to local"
/// button is disabled while `mirror_in_progress` is set (see that `.slint` file) --
/// the same division of responsibility `gui::render_export::setup_render_export_callbacks`
/// already uses for exports (the Rust side here never even inspects `handle` before
/// starting a new sync; it simply overwrites it, exactly as that module's own
/// `export_handle` does).
pub fn setup_mirror_sync_callbacks(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    settings_store: &Arc<SettingsPersister>,
) {
    let handle: Rc<RefCell<Option<MirrorHandle>>> = Rc::new(RefCell::new(None));

    let db_start = Arc::clone(db);
    let settings_start = settings_store.clone();
    let handle_start = handle.clone();
    let ui_weak_start = ui.as_weak();
    ui.on_start_mirror_sync(move |idx: i32| {
        let Some(ui) = ui_weak_start.upgrade() else {
            return;
        };
        let Some(worker) = settings_start
            .snapshot()
            .settings
            .remote_workers
            .get(idx as usize)
            .cloned()
        else {
            show_toast(&ui, "No such remote worker.", "error");
            return;
        };

        ui.set_mirror_in_progress(true);
        ui.set_mirror_worker_index(idx);
        ui.set_mirror_progress_fraction(0.0);
        ui.set_mirror_has_error(false);
        ui.set_mirror_status_text("Listing the remote library...".into());

        let new_handle = library_mirror::spawn_mirror_sync(
            ui.as_weak(),
            Arc::clone(&db_start),
            worker,
            library_mirror::MirrorOptions::default(),
            |ui, progress| on_mirror_progress(ui, &progress),
            |ui, outcome| on_mirror_done(ui, &outcome),
        );
        *handle_start.borrow_mut() = Some(new_handle);
    });

    let handle_cancel = handle;
    ui.on_cancel_mirror_sync(move || {
        if let Some(h) = handle_cancel.borrow().as_ref() {
            h.cancel();
        }
    });
}

fn on_mirror_progress(ui: &MainWindow, progress: &MirrorProgress) {
    let total = progress.counts.total_found.max(1);
    let fraction = progress.processed as f32 / total as f32;
    ui.set_mirror_progress_fraction(fraction);
    ui.set_mirror_status_text(
        format!(
            "{}/{} -- {} (new {}, updated {}, unchanged {}, failed {})",
            progress.processed,
            progress.counts.total_found,
            progress.current_title,
            progress.counts.new_count,
            progress.counts.updated_count,
            progress.counts.skipped_unchanged,
            progress.counts.failed,
        )
        .into(),
    );
}

fn on_mirror_done(ui: &MainWindow, outcome: &MirrorOutcome) {
    ui.set_mirror_in_progress(false);
    let (text, is_error, toast_kind) = match outcome {
        MirrorOutcome::Completed(c) => (
            format!(
                "Mirror complete: {} new, {} updated, {} unchanged, {} failed (of {} found).",
                c.new_count, c.updated_count, c.skipped_unchanged, c.failed, c.total_found
            ),
            c.failed > 0,
            if c.failed > 0 { "error" } else { "success" },
        ),
        MirrorOutcome::Cancelled(c) => (
            format!(
                "Mirror cancelled after {} new, {} updated, {} unchanged (of {} found so far).",
                c.new_count, c.updated_count, c.skipped_unchanged, c.total_found
            ),
            false,
            "info",
        ),
        MirrorOutcome::Failed(msg) => (format!("Mirror failed: {msg}"), true, "error"),
    };
    ui.set_mirror_has_error(is_error);
    ui.set_mirror_status_text(text.clone().into());
    show_toast(ui, &text, toast_kind);
}

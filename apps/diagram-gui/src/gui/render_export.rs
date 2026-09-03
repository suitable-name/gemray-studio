//! The high-resolution export flow: validates the request, captures a
//! `SceneSnapshot` independent of the live viewport's `RenderContext.width`/`height`
//! and accumulation buffer (see `export_thread`'s module doc comment for why that
//! separation matters), and spawns it on its own worker thread via
//! `export_thread::spawn_export`.
//!
//! Split out of `gui::mod` purely to keep that module (already sizeable) from growing
//! further -- same reasoning as `gui::detail`/`gui::search`/`gui::remote`.

use crate::{
    MainWindow,
    bridge::{
        export_thread::{self, ComputeTarget, SceneSnapshot},
        render_thread::RenderContext,
    },
    gui::{color_space_from_index, show_toast},
    settings::SettingsPersister,
};
use slint::ComponentHandle;
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

/// `export_dialog.slint`'s "Compute" pill index (0/1/2, see that property's own doc
/// comment) -> [`ComputeTarget`]. Mirrors `gui::color_space_from_index`'s own
/// int-discriminant convention right above it in this file's imports.
const fn compute_target_from_index(index: i32) -> ComputeTarget {
    match index {
        0 => ComputeTarget::LocalOnly,
        1 => ComputeTarget::RemoteOnly,
        _ => ComputeTarget::Both,
    }
}

// NOTE (found, not fixed -- pure code move only): in the pre-split `gui/mod.rs`, this
// function had no doc comment of its own. A 9-paragraph doc comment describing "the
// high-resolution export flow" sat immediately above `color_space_from_index` instead
// (no blank line between them), so rustdoc actually attached it to THAT function --
// see `color_space_from_index` in `gui/mod.rs`, which still carries it verbatim.
#[expect(
    clippy::too_many_lines,
    reason = "a flat sequence of dialog-lifecycle wiring (validate, save-as, snapshot, \
              spawn, progress/done callbacks) -- the new compute-target/workers plumbing \
              and remote-fallback toast added a few lines to an already-long function \
              rather than introducing a separable concern; splitting further would just \
              move the same line count into a wrapper, matching this crate's existing \
              convention for this class of setup function (see `bridge::render_thread::\
              spawn_render_thread`'s and `gui::mod::build_main_window`'s own identical \
              `#[expect]`)"
)]
pub(super) fn setup_render_export_callbacks(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    let export_handle: Rc<RefCell<Option<export_thread::ExportHandle>>> =
        Rc::new(RefCell::new(None));

    setup_check_remote_availability_callback(ui, settings_store);

    let render_ctx_start = render_ctx.clone();
    let export_handle_start = export_handle.clone();
    let settings_store_start = settings_store.clone();
    let ui_weak_start = ui.as_weak();
    ui.on_start_export(
        move |width: i32,
              height: i32,
              samples: i32,
              color_space_index: i32,
              compute_target_index: i32| {
            let Some(ui) = ui_weak_start.upgrade() else {
                return;
            };

            let params = match export_thread::validate_export_params(width, height, samples) {
                Ok(params) => params,
                Err(err) => {
                    ui.set_export_has_error(true);
                    ui.set_export_status_message(err.into());
                    return;
                }
            };
            let color_space = color_space_from_index(color_space_index);
            let compute_target = compute_target_from_index(compute_target_index);
            // A snapshot of whatever workers are configured RIGHT NOW -- `run_export`
            // re-probes the first one itself (see its own doc comment on why this
            // isn't trusted from whatever the dialog observed when it opened), so this
            // is just the current list, not a cached capability.
            let workers = settings_store_start.snapshot().settings.remote_workers;

            let default_path = export_thread::default_export_path(params);
            // Native Save As dialog, seeded with `default_export_path`'s filename and
            // directory, filtered to `.png` (the only format this export ever writes --
            // see `bridge::export_thread::run_export`). Blocking `rfd::FileDialog`,
            // called directly on the Slint UI/event-loop thread -- see
            // `apps/diagram-gui/Cargo.toml`'s `rfd` dependency comment for why that's
            // the supported way to call it here. Placed BEFORE `SceneSnapshot::capture`/
            // `set_is_exporting` below so cancelling genuinely starts nothing, not even
            // the (cheap but real) snapshot work.
            let mut dialog = rfd::FileDialog::new().add_filter("PNG image", &["png"]);
            if let Some(name) = default_path.file_name() {
                dialog = dialog.set_file_name(name.to_string_lossy());
            }
            if let Some(dir) = default_path.parent().filter(|p| !p.as_os_str().is_empty()) {
                dialog = dialog.set_directory(dir);
            }
            let Some(output_path) = dialog.save_file() else {
                // Cancelling must not silently fall back to `default_path` -- writing a
                // file the user just declined to name would be wrong (this task's own
                // requirement). Distinct from `ExportOutcome::Cancelled` below, which is
                // the user aborting a RUNNING export -- this is turning it down before
                // any work (or `is_exporting`) has even started.
                ui.set_export_has_error(false);
                ui.set_export_status_message("Export cancelled.".into());
                return;
            };

            let scene = SceneSnapshot::capture(&render_ctx_start);

            // Pause live-viewport tracing for the duration of this export -- see
            // `RenderContext::export_active`'s own doc comment. Set here, right as the
            // export actually starts (after the save dialog and snapshot capture, so a
            // cancelled-before-starting export never touches this), and cleared
            // unconditionally in `on_done` below, alongside `set_is_exporting(false)`, so
            // every exit path (success, error, cancellation) resumes the viewport.
            render_ctx_start
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .export_active = true;

            ui.set_is_exporting(true);
            ui.set_export_progress(0.0);
            ui.set_export_has_error(false);
            ui.set_export_status_message(String::new().into());
            // Reset the live preview too -- without this, starting a new export while
            // the previous one's thumbnail is still sitting in this property would show
            // a stale, unrelated image until the first real preview tick lands.
            ui.set_export_preview_samples_done(0);
            ui.set_export_preview_samples_total(params.samples_per_pixel as i32);
            ui.set_export_preview_image(slint::Image::default());

            let render_ctx_done = render_ctx_start.clone();
            let handle = export_thread::spawn_export(
                ui_weak_start.clone(),
                scene,
                params,
                color_space,
                output_path,
                compute_target,
                workers,
                |ui: &MainWindow, progress: export_thread::ExportProgress| {
                    ui.set_export_progress(progress.fraction);
                    ui.set_export_preview_samples_done(progress.samples_done as i32);
                    ui.set_export_preview_samples_total(progress.samples_total as i32);
                    // `None` on most ticks (rate-limited -- see `PreviewThrottle`): just
                    // leave whatever is currently displayed alone rather than clearing it.
                    if let Some(buf) = progress.preview {
                        ui.set_export_preview_image(slint::Image::from_rgba8(buf));
                    }
                    // A one-off status line about the remote half of this export (a
                    // pixel-cap/unreachable-worker fallback decided before dispatch, or
                    // a mid-export worker failure recovered by finishing locally) --
                    // see `ExportProgress::note`'s own doc comment. "info" rather than
                    // "error": both cases mean the export adapted and is still going
                    // (or already finished) successfully, not that it failed.
                    if let Some(note) = progress.note {
                        show_toast(ui, &note, "info");
                    }
                },
                move |ui: &MainWindow, outcome: export_thread::ExportOutcome| {
                    ui.set_is_exporting(false);
                    // Resume the live viewport -- unconditional, exactly like
                    // `set_is_exporting(false)` above, so success, cancellation, AND
                    // failure all resume it. An export that errors out must not leave
                    // the viewport permanently frozen (the same class of bug as an
                    // import leaving a busy flag set on its error path).
                    render_ctx_done
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .export_active = false;
                    // Whatever happens, this run's in-progress thumbnail is no longer
                    // current -- clear it rather than leaving it visible (frozen on
                    // whatever it last showed) behind the result message below, or as a
                    // stale first frame if the dialog is reopened before starting again.
                    ui.set_export_preview_image(slint::Image::default());
                    match outcome {
                        export_thread::ExportOutcome::Completed(path) => {
                            let msg = format!("Exported to {}", path.display());
                            ui.set_export_has_error(false);
                            ui.set_export_status_message(msg.clone().into());
                            show_toast(ui, &msg, "success");
                        }
                        export_thread::ExportOutcome::Cancelled => {
                            ui.set_export_has_error(false);
                            ui.set_export_status_message("Export cancelled.".into());
                        }
                        export_thread::ExportOutcome::Failed(err) => {
                            ui.set_export_has_error(true);
                            ui.set_export_status_message(err.clone().into());
                            show_toast(ui, &format!("Export failed: {err}"), "error");
                        }
                    }
                },
            );
            *export_handle_start.borrow_mut() = Some(handle);
        },
    );

    ui.on_cancel_export(move || {
        if let Some(handle) = export_handle.borrow().as_ref() {
            handle.cancel();
        }
    });
}

/// Wires `check_remote_availability` (fired by `gem_viewport.slint`'s `changed
/// export_open` handler every time the export dialog opens -- see that file's own
/// comment) to a background probe of the first configured worker, reporting the result
/// into `export_remote_available`/`export_remote_unavailable_reason` so
/// `export_dialog.slint`'s Compute pill can grey out Remote/Local+Remote WITH a reason
/// rather than leaving them silently missing. Follows `on_test_worker_connection`'s
/// exact shape (`gui::remote::worker_callbacks`): a blocking probe on its own
/// `std::thread::spawn` thread, result marshalled back via `upgrade_in_event_loop`.
fn setup_check_remote_availability_callback(
    ui: &MainWindow,
    settings_store: &Arc<SettingsPersister>,
) {
    let settings_store = settings_store.clone();
    let ui_weak = ui.as_weak();
    ui.on_check_remote_availability(move || {
        let workers = settings_store.snapshot().settings.remote_workers;
        let ui_weak_result = ui_weak.clone();
        std::thread::spawn(move || {
            let result = export_thread::probe_remote(&workers);
            let _ = ui_weak_result.upgrade_in_event_loop(move |ui| match result {
                Ok(_capability) => {
                    ui.set_export_remote_available(true);
                    ui.set_export_remote_unavailable_reason(String::new().into());
                }
                Err(reason) => {
                    ui.set_export_remote_available(false);
                    ui.set_export_remote_unavailable_reason(reason.message().into());
                }
            });
        });
    });
}

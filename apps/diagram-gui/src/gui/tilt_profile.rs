//! Off-UI-thread computation of the Tilt Performance dialog's three non-canonical
//! camera azimuths (45/90/135 deg -- see `gemray::color::metrics::
//! PROFILE_AZIMUTHS_DEG`).
//!
//! The live render loop (`bridge::render_thread`) already computes and caches the
//! azimuth-0 sweep as part of every metrics update (see `bridge::render_thread::
//! metrics::compute_or_reuse_metrics`) -- that machinery is deliberately not extended
//! to cover the other three here, since doing so would mean widening
//! `compute_or_reuse_metrics`'s cache and `spawn_render_thread`'s per-frame callback
//! signature, both outside this change's scope. Instead this module recomputes the
//! three extra azimuths independently: off the UI thread (four analytic raytracing
//! sweeps instead of one is real work -- see `evaluate_angular_profile_at_azimuth`'s
//! own cost), and only while the dialog that actually shows them is open, triggered by
//! `GemViewportView::request_tilt_profile_axes` (see that callback's doc comment).
//!
//! Reads `RenderContext::active_planes`/`material_name`/`custom_materials`/
//! `light_yaw`/`light_pitch` directly -- the same inputs `compute_or_reuse_metrics`
//! keys its own cache on, minus camera yaw/pitch (this always sweeps the full 0..90
//! tilt range at fixed azimuths, so the live camera pose is irrelevant here).

use crate::{
    MainWindow,
    bridge::render_thread::{RenderContext, hash_planes, resolve_material},
    gui::curve_path::tilt_curve_path,
};
use gemray::{
    color::metrics::{PROFILE_AZIMUTHS_DEG, evaluate_angular_profile_at_azimuth},
    optics::materials::GemMaterial,
};
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

/// Cheap identity for "would recomputing the three extra azimuths produce a different
/// result" -- the same fields `bridge::render_thread::metrics::MetricsCacheKey` keys
/// on, minus camera yaw/pitch (irrelevant here, see this module's doc comment).
#[derive(Clone, PartialEq)]
struct AxesCacheKey {
    light_yaw: f32,
    light_pitch: f32,
    material_name: String,
    planes_hash: u64,
}

/// Wires `MainWindow::request_tilt_profile_axes` to a background computation of the
/// three non-canonical tilt-elevation sweeps, pushing the results into
/// `graph_*_extra_axes`/`graph_*_extra_paths` (and `extra_axes_loading` around the
/// computation) once done. Split out of `run_gui`/`build_main_window` purely to keep
/// those functions under clippy's function-length lint, matching every other
/// `setup_*_callback` in this module's siblings.
pub(super) fn setup_tilt_profile_callback(ui: &MainWindow, render_ctx: &Arc<Mutex<RenderContext>>) {
    // `Some(key)` once a request for exactly these inputs has been launched (whether
    // or not it has finished yet) -- deduplicates re-opening the dialog against the
    // same geometry/material/light without a second background sweep, and is
    // overwritten (allowing a fresh computation) the moment any of those inputs
    // actually changes.
    let launched_key: Arc<Mutex<Option<AxesCacheKey>>> = Arc::new(Mutex::new(None));
    // Bumped on every newly-launched computation; a background thread checks its own
    // snapshot against the latest value before applying results, so a stale
    // computation (superseded by a newer request before it finished) is silently
    // dropped instead of overwriting fresher data.
    let generation = Arc::new(AtomicU64::new(0));

    let ui_weak = ui.as_weak();
    let render_ctx = render_ctx.clone();
    ui.on_request_tilt_profile_axes(move || {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };

        let (planes, material_name, custom_materials, light_yaw, light_pitch) = {
            let ctx = render_ctx.lock().unwrap();
            (
                ctx.active_planes.clone(),
                ctx.material_name.clone(),
                ctx.custom_materials.clone(),
                ctx.light_yaw,
                ctx.light_pitch,
            )
        };

        let key = AxesCacheKey {
            light_yaw,
            light_pitch,
            material_name: material_name.clone(),
            planes_hash: hash_planes(&planes),
        };
        {
            let mut launched = launched_key.lock().unwrap();
            if launched.as_ref() == Some(&key) {
                // Already computed (or currently computing) for these exact inputs --
                // nothing to do. Reopening the dialog without moving the camera/light
                // or changing the material/geometry hits this every time.
                return;
            }
            *launched = Some(key);
        }

        let my_generation = generation.fetch_add(1, Ordering::SeqCst) + 1;
        let generation = generation.clone();
        ui.set_extra_axes_loading(true);
        let ui_weak_bg = ui.as_weak();

        std::thread::spawn(move || {
            let material = resolve_material(
                &GemMaterial::all_materials(),
                &custom_materials,
                &material_name,
            );

            // PROFILE_AZIMUTHS_DEG[0] (0 deg) is the canonical sweep the render thread
            // already provides via `graph_brilliance`/etc -- only the other three are
            // computed here.
            let mut brilliance_rows: Vec<[f32; 19]> = Vec::with_capacity(3);
            let mut extinction_rows: Vec<[f32; 19]> = Vec::with_capacity(3);
            let mut windowing_rows: Vec<[f32; 19]> = Vec::with_capacity(3);
            for &azimuth_deg in &PROFILE_AZIMUTHS_DEG[1..] {
                let (b, e, w) = evaluate_angular_profile_at_azimuth(
                    &planes,
                    &material,
                    azimuth_deg.to_radians(),
                    light_yaw,
                    light_pitch,
                );
                brilliance_rows.push(b);
                extinction_rows.push(e);
                windowing_rows.push(w);
            }

            let brilliance_paths: Vec<SharedString> = brilliance_rows
                .iter()
                .map(|row| tilt_curve_path(row).into())
                .collect();
            let extinction_paths: Vec<SharedString> = extinction_rows
                .iter()
                .map(|row| tilt_curve_path(row).into())
                .collect();
            let windowing_paths: Vec<SharedString> = windowing_rows
                .iter()
                .map(|row| tilt_curve_path(row).into())
                .collect();

            let to_model_rows = |rows: Vec<[f32; 19]>| -> ModelRc<ModelRc<f32>> {
                ModelRc::new(VecModel::from(
                    rows.into_iter()
                        .map(|row| ModelRc::new(VecModel::from(row.to_vec())))
                        .collect::<Vec<_>>(),
                ))
            };

            let _ = ui_weak_bg.upgrade_in_event_loop(move |ui| {
                if generation.load(Ordering::SeqCst) != my_generation {
                    // Superseded by a newer request (camera/light/material/geometry
                    // moved again before this finished) -- drop this stale result
                    // rather than overwriting whatever the newer computation lands.
                    return;
                }
                ui.set_graph_brilliance_extra_axes(to_model_rows(brilliance_rows));
                ui.set_graph_extinction_extra_axes(to_model_rows(extinction_rows));
                ui.set_graph_windowing_extra_axes(to_model_rows(windowing_rows));
                ui.set_graph_brilliance_extra_paths(ModelRc::new(VecModel::from(brilliance_paths)));
                ui.set_graph_extinction_extra_paths(ModelRc::new(VecModel::from(extinction_paths)));
                ui.set_graph_windowing_extra_paths(ModelRc::new(VecModel::from(windowing_paths)));
                ui.set_extra_axes_loading(false);
            });
        });
    });
}

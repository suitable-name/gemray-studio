//! Camera orbit/zoom, light move/position/reset, and HDR environment-map load/clear
//! callback wiring.
//!
//! Split out of `gui::mod` purely to keep that module (already sizeable) from growing
//! further -- same reasoning as `gui::detail`/`gui::search`/`gui::remote`.

use crate::{
    MainWindow,
    bridge::render_thread::{RenderContext, load_env_map},
    gui::{env_map_status_text, show_toast},
    settings::SettingsPersister,
};
use slint::{ComponentHandle, SharedString};
use std::sync::{Arc, Mutex};

/// Wires up camera orbit/zoom, light move/position, and reset-camera callbacks. Each
/// also feeds the debounced `settings_store` so camera pose and light position survive
/// a restart. Split out of `run_gui` purely to keep that function under
/// clippy's function-length lint.
pub(super) fn setup_camera_and_lighting_callbacks(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    let render_ctx_orbit = render_ctx.clone();
    let settings_store_orbit = settings_store.clone();
    ui.on_camera_orbit(move |dx: f32, dy: f32| {
        let (yaw, pitch) = {
            let mut ctx = render_ctx_orbit.lock().unwrap();
            // Horizontal drag is INVERTED relative to vertical: dragging right turns
            // the stone's right side toward the viewer, as though the hand were on the
            // gem rather than on the camera. Vertical keeps its sign, which already
            // reads correctly -- the two axes genuinely want opposite conventions here,
            // so the asymmetry is deliberate, not a stray sign.
            ctx.yaw = dx.mul_add(-0.008, ctx.yaw);
            ctx.pitch = dy.mul_add(0.008, ctx.pitch).clamp(-1.48, 1.48);
            ctx.dirty = true;
            (ctx.yaw, ctx.pitch)
        };
        settings_store_orbit.update(|s| {
            s.settings.camera_yaw = yaw;
            s.settings.camera_pitch = pitch;
        });
    });

    let render_ctx_zoom = render_ctx.clone();
    let settings_store_zoom = settings_store.clone();
    ui.on_camera_zoom(move |delta: f32| {
        let distance = {
            let mut ctx = render_ctx_zoom.lock().unwrap();
            ctx.distance = delta.mul_add(-0.002, ctx.distance).clamp(1.2, 8.0);
            ctx.dirty = true;
            ctx.distance
        };
        settings_store_zoom.update(|s| s.settings.camera_distance = distance);
    });

    let render_ctx_light = render_ctx.clone();
    let settings_store_light = settings_store.clone();
    ui.on_light_move(move |dx: f32, dy: f32| {
        let (light_yaw, light_pitch) = {
            let mut ctx = render_ctx_light.lock().unwrap();
            ctx.light_yaw = dx.mul_add(0.01, ctx.light_yaw);
            ctx.light_pitch = dy.mul_add(0.01, ctx.light_pitch).clamp(0.15, 1.55);
            ctx.dirty = true;
            (ctx.light_yaw, ctx.light_pitch)
        };
        settings_store_light.update(|s| {
            s.settings.light_yaw_deg = light_yaw.to_degrees();
            s.settings.light_pitch_deg = light_pitch.to_degrees();
        });
    });

    let render_ctx_light_pos = render_ctx.clone();
    let settings_store_light_pos = settings_store.clone();
    ui.on_light_pos_changed(move |yaw_deg: f32, pitch_deg: f32| {
        {
            let mut ctx = render_ctx_light_pos.lock().unwrap();
            ctx.light_yaw = yaw_deg.to_radians();
            ctx.light_pitch = pitch_deg.to_radians().clamp(0.15, 1.55);
            ctx.dirty = true;
        }
        settings_store_light_pos.update(|s| {
            s.settings.light_yaw_deg = yaw_deg;
            s.settings.light_pitch_deg = pitch_deg;
        });
    });

    let render_ctx_reset = render_ctx.clone();
    let settings_store_reset = settings_store.clone();
    ui.on_reset_camera(move || {
        {
            let mut ctx = render_ctx_reset.lock().unwrap();
            ctx.yaw = 0.60;
            ctx.pitch = 0.45;
            ctx.distance = 2.4;
            ctx.light_yaw = 0.85;
            ctx.light_pitch = 0.95;
            ctx.dirty = true;
        }
        settings_store_reset.update(|s| {
            s.settings.camera_yaw = 0.60;
            s.settings.camera_pitch = 0.45;
            s.settings.camera_distance = 2.4;
            s.settings.light_yaw_deg = 0.85_f32.to_degrees();
            s.settings.light_pitch_deg = 0.95_f32.to_degrees();
        });
    });
}

/// Wires up the HDR environment-map load/clear callbacks. Path is a plain typed text
/// field (`settings_dialog.slint`'s "Environment Map (HDR)" section) with no native
/// picker of its own -- unlike `gui::library::setup_import_callback`'s `.asc`
/// pickers, which use `rfd::FileDialog` (see that module's doc comment). Split out of
/// `run_gui` purely to keep that function under clippy's function-length lint.
pub(super) fn setup_environment_map_callbacks(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    let render_ctx_load = render_ctx.clone();
    let settings_store_load = settings_store.clone();
    let ui_weak_load = ui.as_weak();
    ui.on_load_env_map(move |raw_path: SharedString| {
        let Some(ui) = ui_weak_load.upgrade() else {
            return;
        };
        let path = raw_path.to_string();
        // `load_env_map` never panics -- a missing, unreadable, or malformed file
        // returns `Err` and `RenderContext::env_map`/the settings file are left exactly
        // as they were, per this task's own requirement (see that function's doc
        // comment).
        match load_env_map(&path) {
            Ok(map) => {
                let status = env_map_status_text(&map, &path);
                {
                    let mut ctx = render_ctx_load.lock().unwrap();
                    ctx.env_map = Some(map);
                    ctx.dirty = true;
                }
                settings_store_load.update(|s| s.settings.env_map_path.clone_from(&path));
                ui.set_env_map_status(status.clone().into());
                ui.set_env_map_loaded(true);
                // Loading an environment map forces the switch to the (slower) CPU
                // tracer -- see `gemray::renderer::gpu_backend`'s module doc comment --
                // which must not leave the user wondering why rendering got slower with
                // no visible cause.
                show_toast(
                    &ui,
                    &format!("{status}. Rendering on CPU: the GPU backend has no HDR support."),
                    "info",
                );
            }
            Err(err) => {
                show_toast(
                    &ui,
                    &format!("Could not load HDR environment: {err}"),
                    "error",
                );
            }
        }
    });

    let render_ctx_clear = render_ctx.clone();
    let settings_store_clear = settings_store.clone();
    let ui_weak_clear = ui.as_weak();
    ui.on_clear_env_map(move || {
        let Some(ui) = ui_weak_clear.upgrade() else {
            return;
        };
        {
            let mut ctx = render_ctx_clear.lock().unwrap();
            ctx.env_map = None;
            ctx.dirty = true;
        }
        settings_store_clear.update(|s| s.settings.env_map_path.clear());
        ui.set_env_map_status(String::new().into());
        ui.set_env_map_loaded(false);
        show_toast(
            &ui,
            "Cleared HDR environment; back to the studio rig.",
            "info",
        );
    });

    // Native file-open picker for `env_map_path` ("Browse..." in
    // `settings_dialog.slint`) -- fills the field, doesn't load anything itself;
    // `on_load_env_map` above still does that, unchanged, whichever way the path got
    // typed in. Filtered to exactly `.hdr`: the only format `EnvironmentMap::from_hdr_file`
    // (crates/gemray/src/renderer/env_map.rs) decodes, via `image::ImageFormat::Hdr`.
    // Returns the SAME text it was given when the user cancels, so the Slint-side
    // assignment (`root.env_map_path = root.pick_hdr_file(root.env_map_path)`) is a
    // no-op on cancel.
    ui.on_pick_hdr_file(|current: SharedString| {
        let mut dialog = rfd::FileDialog::new().add_filter("Radiance HDR", &["hdr"]);
        if let Some(dir) = super::starting_dir_from_picker_field(current.as_str()) {
            dialog = dialog.set_directory(dir);
        }
        // Blocking `rfd::FileDialog`, invoked directly on the Slint UI/event-loop
        // thread -- see `apps/diagram-gui/Cargo.toml`'s `rfd` dependency comment for
        // why that's the supported way to call it here.
        dialog
            .pick_file()
            .map_or(current, |path| path.display().to_string().into())
    });
}

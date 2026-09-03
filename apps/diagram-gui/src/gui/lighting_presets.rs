//! The four named-lighting-preset callbacks: create, apply, rename, delete.
//!
//! Split out of `gui::mod` purely to keep that module (already sizeable) from growing
//! further -- same reasoning as `gui::detail`/`gui::search`/`gui::remote`.

use crate::{
    MainWindow,
    bridge::render_thread::RenderContext,
    gui::{refresh_lighting_preset_options, show_toast},
    settings::{LightingPreset as SavedLightingPreset, SettingsPersister},
};
use gemray::optics::LightingPreset;
use slint::{ComponentHandle, SharedString};
use std::sync::{Arc, Mutex};

/// Wires up the four named-lighting-preset callbacks: create, apply, rename,
/// delete. A thin orchestrator over four single-purpose helpers, each split out
/// purely to keep every individual function under clippy's function-length lint.
pub(super) fn setup_lighting_preset_callbacks(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    setup_save_lighting_preset_callback(ui, render_ctx, settings_store);
    setup_apply_lighting_preset_callback(ui, render_ctx, settings_store);
    setup_rename_lighting_preset_callback(ui, settings_store);
    setup_delete_lighting_preset_callback(ui, settings_store);
}

/// Captures the CURRENT live lighting-rig state (light yaw/pitch, exposure, rig
/// selection, camera distance -- see `settings::model::LightingPreset`'s doc comment
/// for why camera yaw/pitch are deliberately excluded) and saves it as a new preset,
/// or overwrites an existing user preset of the same name.
fn setup_save_lighting_preset_callback(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    let render_ctx_save = render_ctx.clone();
    let settings_store_save = settings_store.clone();
    let ui_weak_save = ui.as_weak();
    ui.on_save_lighting_preset(move |name: SharedString| {
        let (light_yaw_deg, light_pitch_deg, exposure, lighting_rig, camera_distance) = {
            let ctx = render_ctx_save.lock().unwrap();
            (
                ctx.light_yaw.to_degrees(),
                ctx.light_pitch.to_degrees(),
                ctx.exposure,
                ctx.lighting_preset.label().to_string(),
                ctx.distance,
            )
        };
        let preset = SavedLightingPreset {
            name: name.to_string(),
            built_in: false,
            light_yaw_deg,
            light_pitch_deg,
            exposure,
            lighting_rig,
            camera_distance,
        };

        let mut result = Ok(());
        settings_store_save.update(|s| result = s.upsert_user_preset(preset.clone()));

        let Some(ui) = ui_weak_save.upgrade() else {
            return;
        };
        match result {
            Ok(()) => {
                refresh_lighting_preset_options(&ui, &settings_store_save.snapshot().presets);
                show_toast(&ui, &format!("Saved lighting preset '{name}'"), "success");
            }
            Err(err) => show_toast(&ui, &err, "error"),
        }
    });
}

/// Applies preset `idx` (looked up in the settings store's current preset list, the
/// same list the UI's `lighting_presets` model was built from): pushes its fields into
/// `RenderContext` -- setting `ctx.dirty = true` so the render restarts, exactly as
/// the existing light controls do -- mirrors them into the UI's own slider/dropdown
/// state, and persists them as the new "current" settings so the applied look survives
/// a restart even without the user touching a slider afterward.
fn setup_apply_lighting_preset_callback(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    let render_ctx_apply = render_ctx.clone();
    let settings_store_apply = settings_store.clone();
    let ui_weak_apply = ui.as_weak();
    ui.on_apply_lighting_preset(move |idx: i32| {
        let Some(preset) = usize::try_from(idx)
            .ok()
            .and_then(|i| settings_store_apply.snapshot().presets.get(i).cloned())
        else {
            return;
        };

        let lighting_preset = LightingPreset::from_label(&preset.lighting_rig);
        {
            let mut ctx = render_ctx_apply.lock().unwrap();
            ctx.light_yaw = preset.light_yaw_deg.to_radians();
            ctx.light_pitch = preset.light_pitch_deg.to_radians().clamp(0.15, 1.55);
            ctx.exposure = preset.exposure.clamp(0.2, 5.0);
            ctx.lighting_preset = lighting_preset;
            ctx.distance = preset.camera_distance.clamp(1.2, 8.0);
            ctx.dirty = true;
        }

        settings_store_apply.update(|s| {
            s.settings.light_yaw_deg = preset.light_yaw_deg;
            s.settings.light_pitch_deg = preset.light_pitch_deg;
            s.settings.exposure = preset.exposure;
            s.settings.lighting_rig.clone_from(&preset.lighting_rig);
            s.settings.camera_distance = preset.camera_distance;
        });

        let Some(ui) = ui_weak_apply.upgrade() else {
            return;
        };
        ui.set_light_yaw_deg(preset.light_yaw_deg);
        ui.set_light_pitch_deg(preset.light_pitch_deg);
        ui.set_exposure_val(preset.exposure);
        ui.set_selected_lighting_index(lighting_preset.index());
        show_toast(
            &ui,
            &format!("Applied lighting preset '{}'", preset.name),
            "info",
        );
    });
}

fn setup_rename_lighting_preset_callback(ui: &MainWindow, settings_store: &Arc<SettingsPersister>) {
    let settings_store_rename = settings_store.clone();
    let ui_weak_rename = ui.as_weak();
    ui.on_rename_lighting_preset(move |idx: i32, new_name: SharedString| {
        let Some(old_name) = usize::try_from(idx).ok().and_then(|i| {
            settings_store_rename
                .snapshot()
                .presets
                .get(i)
                .map(|p| p.name.clone())
        }) else {
            return;
        };

        let mut result = Ok(());
        settings_store_rename.update(|s| result = s.rename_preset(&old_name, &new_name));

        let Some(ui) = ui_weak_rename.upgrade() else {
            return;
        };
        match result {
            Ok(()) => {
                refresh_lighting_preset_options(&ui, &settings_store_rename.snapshot().presets);
                show_toast(&ui, "Preset renamed.", "success");
            }
            Err(err) => show_toast(&ui, &err, "error"),
        }
    });
}

fn setup_delete_lighting_preset_callback(ui: &MainWindow, settings_store: &Arc<SettingsPersister>) {
    let settings_store_delete = settings_store.clone();
    let ui_weak_delete = ui.as_weak();
    ui.on_delete_lighting_preset(move |idx: i32| {
        let Some(name) = usize::try_from(idx).ok().and_then(|i| {
            settings_store_delete
                .snapshot()
                .presets
                .get(i)
                .map(|p| p.name.clone())
        }) else {
            return;
        };

        let mut result = Ok(());
        settings_store_delete.update(|s| result = s.delete_preset(&name));

        let Some(ui) = ui_weak_delete.upgrade() else {
            return;
        };
        match result {
            Ok(()) => {
                refresh_lighting_preset_options(&ui, &settings_store_delete.snapshot().presets);
                show_toast(&ui, &format!("Deleted preset '{name}'"), "info");
            }
            Err(err) => show_toast(&ui, &err, "error"),
        }
    });
}

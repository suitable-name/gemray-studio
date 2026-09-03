//! Material-selection, quality (lighting preset/target samples/resolution/inclusion
//! scattering/pause/bounce count/exposure), and material-effect-override (crystal-axis
//! orientation, frosted girdle, edge rounding, physical stone size) callback wiring.
//!
//! Split out of `gui::mod` purely to keep that module (already sizeable) from growing
//! further -- same reasoning as `gui::detail`/`gui::search`/`gui::remote`.

use crate::{
    MainWindow,
    bridge::render_thread::{RenderContext, resolve_material},
    gui::{
        c_axis::{angles_to_c_axis, c_axis_to_angles},
        is_c_axis_override_available, local_preview_scale_from_index,
        sample_scale::exponent_to_count,
        show_toast,
    },
    settings::SettingsPersister,
};
use gemray::optics::{LightingPreset, materials::GemMaterial};
use slint::{ComponentHandle, SharedString};
use std::sync::{Arc, Mutex};

/// Wires up the material-selection callback. Split out of
/// `setup_material_and_quality_callbacks` (rather than folded into it) purely to keep
/// that function under clippy's function-length lint -- this one callback grew a
/// second responsibility (`c_axis_override_available` re-derivation) that
/// pushed the combined function over the limit.
pub(super) fn setup_material_changed_callback(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    let render_ctx_mat = render_ctx.clone();
    let settings_store_mat = settings_store.clone();
    let ui_weak_mat = ui.as_weak();
    ui.on_material_changed(move |material: SharedString| {
        let mut ctx = render_ctx_mat.lock().unwrap();
        ctx.material_name = material.to_string();
        ctx.dirty = true;
        // Re-derive whether the crystal-axis control has any effect on the
        // NEWLY selected material -- an override left dialed in from a previous
        // (anisotropic) material stays harmless either way, thanks to
        // `apply_material_overrides`'s own isotropic guard, but the UI must still grey
        // the control out (and explain why) the moment the selection lands on an
        // isotropic stone.
        let available = is_c_axis_override_available(&resolve_material(
            &GemMaterial::all_materials(),
            &ctx.custom_materials,
            &material,
        ));
        drop(ctx);
        settings_store_mat.update(|s| s.settings.selected_material = material.to_string());
        if let Some(ui) = ui_weak_mat.upgrade() {
            ui.set_c_axis_override_available(available);
            show_toast(&ui, &format!("Material switched to {material}"), "info");
        }
    });
}

/// Wires up lighting-preset/target-samples/render-resolution/inclusion-scattering
/// changes, pause/tab-visibility, bounce count, and exposure callbacks. Each of the
/// persisted settings also feeds the debounced `settings_store`. Split out of
/// `run_gui` purely to keep that function under clippy's function-length lint.
pub(super) fn setup_material_and_quality_callbacks(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    // The UI now carries the lighting preset's enum discriminant as a plain
    // `int` (see `app.slint`/`gem_viewport.slint`'s `lighting_changed(int)` callback)
    // rather than a display string -- `LightingPreset::from_index` maps it back to the
    // enum here, the one boundary crossing from "UI index" to "physics enum", and
    // `.label()` is what actually gets persisted/toasted as text.
    let render_ctx_lit = render_ctx.clone();
    let settings_store_lit = settings_store.clone();
    let ui_weak_lit = ui.as_weak();
    ui.on_lighting_changed(move |idx: i32| {
        let preset = LightingPreset::from_index(idx);
        let mut ctx = render_ctx_lit.lock().unwrap();
        ctx.lighting_preset = preset;
        ctx.dirty = true;
        drop(ctx);
        settings_store_lit.update(|s| s.settings.lighting_rig = preset.label().to_string());
        if let Some(ui) = ui_weak_lit.upgrade() {
            show_toast(&ui, &format!("Lighting preset: {}", preset.label()), "info");
        }
    });

    // Target Samples slider, replacing the old four-tier quality preset.
    // Carries the slider's EXPONENT (3..=10), same int-discriminant treatment as
    // lighting/material above -- `exponent_to_count` (see `gui::sample_scale`'s module
    // doc comment) is the one boundary crossing from "slider position" to "the actual
    // sample count", which is what's stored/persisted/rendered.
    let render_ctx_samples = render_ctx.clone();
    let settings_store_samples = settings_store.clone();
    let ui_weak_samples = ui.as_weak();
    ui.on_target_samples_changed(move |exponent: i32| {
        let target_samples = exponent_to_count(u32::try_from(exponent).unwrap_or(0));
        let mut ctx = render_ctx_samples.lock().unwrap();
        ctx.target_samples = target_samples;
        ctx.dirty = true;
        drop(ctx);
        settings_store_samples.update(|s| s.settings.target_samples = target_samples);
        if let Some(ui) = ui_weak_samples.upgrade() {
            show_toast(&ui, &format!("Target samples: {target_samples}"), "info");
        }
    });

    // Render Resolution pill selector: carries the resolved (width, height) pair
    // directly, unlike the samples slider above -- see `RenderContext::width`'s own
    // doc comment for why this stays a fixed pill list. Setting `ctx.width`/`.height`
    // is all that's needed to reset progressive accumulation cleanly: the render loop's
    // `update_accumulation_state` (in `bridge::render_thread`) already reallocates the
    // accumulation buffer, the three denoiser guide buffers, and `FramebufferTransfer`,
    // and zeroes `accum_samples`, whenever it sees `width`/`height` differ from its own
    // `last_width`/`last_height` on the very next frame -- `ctx.dirty = true` here is
    // the same belt-and-suspenders every other setting in this function sets, not load-
    // bearing for the resize itself.
    let render_ctx_res = render_ctx.clone();
    let settings_store_res = settings_store.clone();
    let ui_weak_res = ui.as_weak();
    ui.on_resolution_changed(move |width: i32, height: i32| {
        let (width, height) = (width as u32, height as u32);
        let mut ctx = render_ctx_res.lock().unwrap();
        ctx.width = width;
        ctx.height = height;
        ctx.dirty = true;
        drop(ctx);
        settings_store_res.update(|s| {
            s.settings.render_width = width;
            s.settings.render_height = height;
        });
        if let Some(ui) = ui_weak_res.upgrade() {
            show_toast(&ui, &format!("Render resolution: {width}x{height}"), "info");
        }
    });

    // Local preview-then-settle rendering: optional resolution reduction while
    // the camera is moving -- see `RenderContext::local_preview_scale`'s own doc
    // comment for the mechanism. `Off` (index 0, the default) reproduces this crate's
    // pre-existing behaviour exactly, matching every other opt-in control's
    // off-by-default convention in this function. No `ctx.dirty`/toast: this alone
    // never changes what's on screen right now (only whether the NEXT drag renders
    // reduced), the same "takes effect on the next occurrence, not immediately"
    // treatment `on_remote_render_samples_changed` (in `gui::remote::setup_worker_callbacks`)
    // gives the remote sample budget.
    let render_ctx_preview = render_ctx.clone();
    let settings_store_preview = settings_store.clone();
    ui.on_local_preview_scale_changed(move |index: i32| {
        let scale = local_preview_scale_from_index(index);
        render_ctx_preview.lock().unwrap().local_preview_scale = scale;
        settings_store_preview.update(|s| s.settings.local_preview_scale = scale);
    });

    // Inclusion/subsurface scattering amount. Linear, unlike the samples
    // slider above -- `scattering_sigma_s`'s own doc comment (in
    // `crates/gemray/src/optics/materials.rs`) gives the 0.0-3.0 useful range
    // directly, so there's no perceptual remapping to invert here.
    let render_ctx_inc = render_ctx.clone();
    let settings_store_inc = settings_store.clone();
    ui.on_inclusion_changed(move |sigma_s: f32| {
        let clamped = sigma_s.clamp(0.0, 3.0);
        let mut ctx = render_ctx_inc.lock().unwrap();
        ctx.inclusion_sigma_s = clamped;
        ctx.dirty = true;
        drop(ctx);
        settings_store_inc.update(|s| s.settings.inclusion_sigma_s = clamped);
    });

    // Render Pause / Resume -- explicit user intent.
    // Independent of the tab-visibility auto-suspend below: this is the one the button reflects.
    let render_ctx_pause = render_ctx.clone();
    ui.on_pause_toggled(move |paused: bool| {
        let mut ctx = render_ctx_pause.lock().unwrap();
        ctx.paused = paused;
    });

    // Automatic render suspend when the 3D viewport tab isn't the visible one. Must not touch
    // `ctx.paused`, so a manual pause survives switching tabs away and back.
    let render_ctx_tab = render_ctx.clone();
    ui.on_active_tab_changed(move |tab: i32| {
        let mut ctx = render_ctx_tab.lock().unwrap();
        ctx.tab_visible = tab == 0;
    });

    let render_ctx_bnc = render_ctx.clone();
    let settings_store_bnc = settings_store.clone();
    ui.on_bounces_changed(move |bounces: i32| {
        let clamped = (bounces as u32).max(1);
        let mut ctx = render_ctx_bnc.lock().unwrap();
        ctx.max_bounces = clamped;
        ctx.dirty = true;
        drop(ctx);
        settings_store_bnc.update(|s| s.settings.max_bounces = clamped);
    });

    let render_ctx_exp = render_ctx.clone();
    let settings_store_exp = settings_store.clone();
    ui.on_exposure_changed(move |exposure: f32| {
        let clamped = exposure.clamp(0.2, 5.0);
        let mut ctx = render_ctx_exp.lock().unwrap();
        ctx.exposure = clamped;
        ctx.dirty = true;
        drop(ctx);
        settings_store_exp.update(|s| s.settings.exposure = clamped);
    });
}

/// Wires up the crystal-axis orientation override, the frosted-girdle toggle, the
/// edge-rounding slider, and the physical stone-size control. Split out of
/// `setup_material_and_quality_callbacks` purely to keep that function under clippy's
/// function-length lint -- these controls are newer additions to the same
/// settings dialog, not a functionally distinct group from what that function already
/// wires up.
pub(super) fn setup_material_effect_override_callbacks(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) {
    // Crystal-axis orientation override. One callback carries BOTH angles --
    // the moved slider's new value plus the other slider's own current value, read
    // directly off `root.*` in `settings_dialog.slint` -- the same treatment
    // `light_pos_changed` (in `setup_camera_and_lighting_callbacks`) already gives
    // light yaw/pitch, so a single-axis drag never has to guess where the other axis
    // currently sits. `angles_to_c_axis` is the one boundary crossing from these degree
    // sliders to the physical `Vec3` `RenderContext::c_axis_override` stores (see that
    // field's own doc comment).
    let render_ctx_axis_angles = render_ctx.clone();
    let settings_store_axis_angles = settings_store.clone();
    ui.on_c_axis_angles_changed(move |tilt_deg: f32, azimuth_deg: f32| {
        let tilt_deg = tilt_deg.clamp(0.0, 90.0);
        let azimuth_deg = azimuth_deg.clamp(0.0, 360.0);
        let mut ctx = render_ctx_axis_angles.lock().unwrap();
        ctx.c_axis_override = Some(angles_to_c_axis(tilt_deg, azimuth_deg));
        ctx.dirty = true;
        drop(ctx);
        settings_store_axis_angles.update(|s| {
            s.settings.c_axis_tilt_deg = tilt_deg;
            s.settings.c_axis_azimuth_deg = azimuth_deg;
        });
    });

    // Crystal-axis override on/off switch. Off ("as cut", the default) leaves the
    // resolved material's own `c_axis` untouched -- see
    // `AppSettings::c_axis_override_enabled`'s own doc comment. Turning it ON seeds the
    // two angle sliders from the CURRENTLY selected material's own `c_axis` via
    // `gui::c_axis::c_axis_to_angles` (the inverse of `angles_to_c_axis` above), so
    // enabling the override never makes the stone visibly jump.
    let render_ctx_axis_toggle = render_ctx.clone();
    let settings_store_axis_toggle = settings_store.clone();
    let ui_weak_axis_toggle = ui.as_weak();
    ui.on_c_axis_override_changed(move |enabled: bool| {
        let mut ctx = render_ctx_axis_toggle.lock().unwrap();
        if enabled {
            let base = resolve_material(
                &GemMaterial::all_materials(),
                &ctx.custom_materials,
                &ctx.material_name,
            );
            let (tilt_deg, azimuth_deg) = c_axis_to_angles(base.c_axis);
            ctx.c_axis_override = Some(angles_to_c_axis(tilt_deg, azimuth_deg));
            ctx.dirty = true;
            drop(ctx);
            settings_store_axis_toggle.update(|s| {
                s.settings.c_axis_override_enabled = true;
                s.settings.c_axis_tilt_deg = tilt_deg;
                s.settings.c_axis_azimuth_deg = azimuth_deg;
            });
            if let Some(ui) = ui_weak_axis_toggle.upgrade() {
                ui.set_c_axis_tilt_deg(tilt_deg);
                ui.set_c_axis_azimuth_deg(azimuth_deg);
            }
        } else {
            ctx.c_axis_override = None;
            ctx.dirty = true;
            drop(ctx);
            settings_store_axis_toggle.update(|s| s.settings.c_axis_override_enabled = false);
        }
    });

    // Bruted (frosted) girdle finish toggle -- a plain on/off switch, not a
    // slider. See `RenderContext::girdle_frosted`'s own doc comment.
    let render_ctx_girdle = render_ctx.clone();
    let settings_store_girdle = settings_store.clone();
    ui.on_girdle_frosted_changed(move |frosted: bool| {
        let mut ctx = render_ctx_girdle.lock().unwrap();
        ctx.girdle_frosted = frosted;
        ctx.dirty = true;
        drop(ctx);
        settings_store_girdle.update(|s| s.settings.girdle_frosted = frosted);
    });

    // Facet edge rounding radius, same opt-in-linear treatment as the
    // inclusion slider (in `setup_material_and_quality_callbacks`) -- see
    // `RenderContext::edge_rounding_radius`'s own doc comment for the `0.0`-`0.03`
    // range's sourcing.
    let render_ctx_edge = render_ctx.clone();
    let settings_store_edge = settings_store.clone();
    ui.on_edge_rounding_changed(move |radius: f32| {
        let clamped = radius.clamp(0.0, 0.03);
        let mut ctx = render_ctx_edge.lock().unwrap();
        ctx.edge_rounding_radius = clamped;
        ctx.dirty = true;
        drop(ctx);
        settings_store_edge.update(|s| s.settings.edge_rounding_radius = clamped);
    });

    // Physical stone size: girdle width in millimetres, off ("today's look",
    // unscaled) at 0.0. No upper clamp beyond staying non-negative -- unlike the other
    // sliders in this function, this one is a free-typed measurement (see the
    // settings-dialog spin box) rather than a bounded slider, and
    // `RenderContext::stone_width_mm`'s own doc comment covers what an
    // out-of-range-but-positive value does (nothing unsafe: `apply_material_overrides`
    // guards the resulting scale against non-finite/non-positive results regardless).
    let render_ctx_stone = render_ctx.clone();
    let settings_store_stone = settings_store.clone();
    ui.on_stone_width_changed(move |width_mm: f32| {
        let clamped = width_mm.max(0.0);
        let mut ctx = render_ctx_stone.lock().unwrap();
        ctx.stone_width_mm = clamped;
        ctx.dirty = true;
        drop(ctx);
        settings_store_stone.update(|s| s.settings.stone_width_mm = clamped);
    });
}

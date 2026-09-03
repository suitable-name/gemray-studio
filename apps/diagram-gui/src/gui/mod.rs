// Only `search` is `pub`: `refresh_diagram_list` is reused by a downstream binary's
// sync-complete handler (see `search::refresh_diagram_list`'s doc comment). The rest
// are wiring `build_main_window` uses internally.
mod c_axis;
mod camera_lighting;
mod clipboard;
mod crystal_optics;
mod curve_path;
mod custom_materials;
mod detail;
mod diagram_list;
mod library;
mod library_remote;
mod lighting_presets;
mod material_quality;
mod remote;
mod render_export;
mod sample_scale;
pub mod search;
mod tilt_profile;

// `sync_range_bounds_to_ui` is defined in `diagram_list` but re-exported here so its
// public path (`gui::sync_range_bounds_to_ui`) is unchanged for a downstream binary's
// sync-complete handler and this crate's own `gui::library`, both of which call it at
// that spelling.
pub use diagram_list::sync_range_bounds_to_ui;

use crate::{
    LightingPresetItem, MainWindow,
    bridge::{
        library_source::LibrarySource,
        render_thread::{RenderContext, load_env_map, resolve_material, spawn_render_thread},
    },
    gui::{
        c_axis::angles_to_c_axis,
        crystal_optics::gem_material_from_row,
        curve_path::tilt_curve_path,
        remote::{
            live_compute_target_index, refresh_worker_options, remote_samples_count_to_exponent,
            setup_remote_rendering, setup_worker_callbacks,
        },
        sample_scale::count_to_exponent,
    },
    settings::{
        self, LightingPreset as SavedLightingPreset, LocalPreviewScale, SettingsFile,
        SettingsPersister,
    },
};
use diagram_catalog::db::sqlite::Database;
use gemray::{
    color::ColorSpace,
    optics::{
        LightingPreset,
        materials::{GemMaterial, OpticalCharacter},
    },
};
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::sync::{Arc, Mutex};

const DB_PATH: &str = "facet_diagrams.sqlite";

/// This crate's binary entry point (`apps/diagram-gui/src/main.rs` calls
/// straight through to this).
///
/// # Errors
///
/// See [`run_gui`].
pub fn main() -> anyhow::Result<()> {
    run_gui()
}

/// A fully constructed and wired [`MainWindow`], not yet shown or running. Built by
/// [`build_main_window`].
///
/// Exists (rather than `run_gui` doing everything and blocking on `ui.run()`) so a
/// second binary that wants this *same*, already-wired public window -- the private
/// edition's GUI, which shows this window alongside its own separate sync window --
/// can call [`build_main_window`], `.show()` both windows, and drive a single shared
/// event loop itself, without duplicating any of this crate's rendering/settings/
/// remote-rendering wiring.
///
/// `ui` is the only public field. The rest (render context, settings persister,
/// remote-rendering poll timer) exist purely to be kept alive for the life of the
/// window -- dropping any of them early would stop the thing it drives (the render
/// thread, settings autosave, remote-rendering handoff polling) -- so the caller only
/// needs to keep the whole `MainWindowHandle` alive, never reach into its internals.
pub struct MainWindowHandle {
    pub ui: MainWindow,
    // Never read again after construction -- these three exist purely so dropping
    // `MainWindowHandle` is what stops the render thread/settings autosave/handoff
    // polling, not so anything downstream can inspect them. `#[expect(dead_code)]`
    // rather than removing them: removing a field would drop it at the end of
    // `build_main_window` instead of at the end of the caller's `MainWindowHandle`'s
    // lifetime, which is the entire point of holding onto them here.
    #[expect(
        dead_code,
        reason = "RAII guard field -- kept alive, never read; see comment above"
    )]
    render_ctx: Arc<Mutex<RenderContext>>,
    #[expect(
        dead_code,
        reason = "RAII guard field -- kept alive, never read; see comment above"
    )]
    settings_store: Arc<SettingsPersister>,
    #[expect(
        dead_code,
        reason = "RAII guard field -- kept alive, never read; see comment above"
    )]
    remote_rendering_timer: slint::Timer,
}

/// Runs this crate's window standalone: builds it (see [`build_main_window`])
/// and runs it to completion.
///
/// # Errors
///
/// Returns an error if [`build_main_window`] or the window's own event loop
/// (`MainWindow::run`) fails -- see [`build_main_window`]'s doc comment for what that
/// covers.
pub fn run_gui() -> anyhow::Result<()> {
    let handle = build_main_window()?;
    handle.ui.run()?;
    Ok(())
}

/// Builds and fully wires a `MainWindow` -- everything [`run_gui`] used to do
/// inline, up to but not including actually running the event loop. See
/// [`MainWindowHandle`]'s doc comment for why this is split out.
///
/// # Errors
///
/// Returns an error if constructing the `MainWindow` itself fails (a Slint platform
/// initialization failure). A database-open failure is handled internally (falls back
/// to an in-memory-equivalent retry, logging the error into the status bar) rather
/// than propagated, matching this function's pre-split behavior.
///
/// # Panics
///
/// Panics only in the already-degraded case where the *first* `Database::new` call
/// failed AND the fallback retry (logged as a status-bar error, see above) also
/// fails -- matching this function's pre-split behavior, which likewise never
/// propagated a database-open failure as an `Err`.
// Long by one function's worth of setup calls (every one of them already split out
// and individually under the limit) -- this is `run_gui`'s original body, unchanged
// in shape by the `MainWindowHandle` split; splitting it further would just move the
// line count into an equally long "call every setup_* function" wrapper.
#[expect(
    clippy::too_many_lines,
    reason = "a flat sequence of already-extracted setup_* calls (see comment above) -- \
              splitting further just moves the same line count into a wrapper that \
              calls them all, not a real reduction"
)]
pub fn build_main_window() -> anyhow::Result<MainWindowHandle> {
    let ui = MainWindow::new()?;

    // Connect to SQLite DB
    let db = match Database::new(Some(DB_PATH)) {
        Ok(db_inst) => Arc::new(Mutex::new(db_inst)),
        Err(e) => {
            ui.set_status_message(format!("Database error: {e}").into());
            Arc::new(Mutex::new(Database::new(Some(DB_PATH)).unwrap()))
        }
    };

    // Shared Render Context for 3D Viewport
    let render_ctx = Arc::new(Mutex::new(RenderContext::default()));

    // Which library (local database, or a remote worker) the viewer is
    // currently browsing. Starts `LibrarySource::Local` -- see that type's own doc
    // comment on why every local-only call site behaves exactly as it did before this
    // existed until a user actually switches.
    let library_source: Arc<Mutex<LibrarySource>> = Arc::new(Mutex::new(LibrarySource::default()));

    // Load custom gemstone materials from SQLite. `diagram-catalog` returns plain
    // `CustomMaterialRow`s (it must not depend on `gemray`), so convert them into
    // `gemray::optics::materials::GemMaterial` here at the boundary.
    let initial_custom_mats: Vec<GemMaterial> = db
        .lock()
        .unwrap()
        .get_custom_materials()
        .unwrap_or_default()
        .iter()
        .map(gem_material_from_row)
        .collect();
    render_ctx
        .lock()
        .unwrap()
        .custom_materials
        .clone_from(&initial_custom_mats);
    refresh_material_options(&ui, &initial_custom_mats);

    // Settings persistence + lighting presets. `load_or_default`
    // never fails/panics -- a missing, unreadable, or corrupt file just logs and
    // yields defaults, see its doc comment. Applied into `render_ctx` and the UI's
    // mirrored properties before the render thread starts, then handed to a
    // debounced background writer that every settings-changing callback below feeds.
    let settings_path = settings::store::default_settings_path();
    let loaded_settings = settings::store::load_or_default(&settings_path);
    apply_loaded_settings(&ui, &render_ctx, &loaded_settings);
    refresh_lighting_preset_options(&ui, &loaded_settings.presets);
    // Remote rendering: the worker-list panel's rows, restored from the last saved
    // configuration.
    refresh_worker_options(&ui, &loaded_settings.settings.remote_workers);
    ui.set_denoise_enabled(loaded_settings.settings.denoise_enabled);
    let settings_store = Arc::new(SettingsPersister::spawn(settings_path, loaded_settings));

    // Spawn Background Multi-Threaded Physically Based Spectral Gem Raytracer
    let ui_weak_render = ui.as_weak();
    spawn_render_thread(
        ui_weak_render,
        render_ctx.clone(),
        |ui: &MainWindow, img| {
            ui.set_render_image(slint::Image::from_rgba8(img));
            ui.set_has_render(true);
        },
        |ui: &MainWindow,
         brilliance: f32,
         fire: f32,
         scintillation: f32,
         windowing: f32,
         extinction: f32,
         graph_brilliance: [f32; 19],
         graph_extinction: [f32; 19],
         graph_windowing: [f32; 19],
         cam_pitch_deg: f32| {
            ui.set_brilliance_pct(brilliance);
            ui.set_fire_index(fire);
            ui.set_scintillation_pct(scintillation);
            ui.set_windowing_pct(windowing);
            ui.set_extinction_pct(extinction);
            ui.set_graph_brilliance(slint::ModelRc::new(slint::VecModel::from(
                graph_brilliance.to_vec(),
            )));
            ui.set_graph_extinction(slint::ModelRc::new(slint::VecModel::from(
                graph_extinction.to_vec(),
            )));
            ui.set_graph_windowing(slint::ModelRc::new(slint::VecModel::from(
                graph_windowing.to_vec(),
            )));
            // Tilt-curve `Path::commands` strings for the performance graph dialog's
            // line chart -- derived here rather than threading three more arguments
            // through this closure (see `curve_path::tilt_curve_path`'s doc comment).
            ui.set_graph_brilliance_path(tilt_curve_path(&graph_brilliance).into());
            ui.set_graph_extinction_path(tilt_curve_path(&graph_extinction).into());
            ui.set_graph_windowing_path(tilt_curve_path(&graph_windowing).into());
            ui.set_cam_pitch_deg(cam_pitch_deg);
        },
    );

    camera_lighting::setup_camera_and_lighting_callbacks(&ui, &render_ctx, &settings_store);
    material_quality::setup_material_changed_callback(&ui, &render_ctx, &settings_store);
    material_quality::setup_material_and_quality_callbacks(&ui, &render_ctx, &settings_store);
    material_quality::setup_material_effect_override_callbacks(&ui, &render_ctx, &settings_store);
    camera_lighting::setup_environment_map_callbacks(&ui, &render_ctx, &settings_store);
    lighting_presets::setup_lighting_preset_callbacks(&ui, &render_ctx, &settings_store);
    render_export::setup_render_export_callbacks(&ui, &render_ctx, &settings_store);
    custom_materials::setup_custom_material_callbacks(&ui, &render_ctx, &db);
    clipboard::setup_copy_callbacks(&ui);
    tilt_profile::setup_tilt_profile_callback(&ui, &render_ctx);
    diagram_list::load_filter_options_and_initial_list(&ui, &db);
    diagram_list::setup_search_and_filter_callbacks(&ui, &db, &library_source);
    diagram_list::setup_diagram_selection_and_export_callbacks(
        &ui,
        &db,
        &library_source,
        &render_ctx,
    );
    library::setup_import_callback(&ui, &db, &library_source);
    library::setup_rename_callback(&ui, &db, &library_source);
    library::setup_set_shape_callback(&ui, &db, &library_source);
    library::setup_delete_callback(&ui, &db, &library_source, &render_ctx);
    library::setup_export_asc_callback(&ui, &db, &library_source);
    detail::setup_save_metadata_callback(&ui, &db, &library_source, &render_ctx);
    library_remote::setup_library_source_callbacks(&ui, &db, &library_source, &settings_store);
    library_remote::setup_mirror_sync_callbacks(&ui, &db, &settings_store);
    setup_worker_callbacks(&ui, &render_ctx, &settings_store);
    // Preview-then-handoff (Task: remote rendering): a repeating `slint::Timer` that
    // polls camera/light pose and drives the handoff -- must be kept alive for the
    // life of the window (a dropped `Timer` simply stops firing), hence the binding
    // held all the way to `ui.run()` below rather than being dropped immediately.
    let remote_rendering_timer = setup_remote_rendering(&ui, &render_ctx, &settings_store);

    // Signal the render thread to exit when the window closes. `running` is the thread's
    // shutdown flag -- it's only ever set to false here, once, since the thread has no way to
    // be restarted. Without this the render thread would silently outlive the window.
    let render_ctx_close = render_ctx.clone();
    let settings_store_close = settings_store.clone();
    ui.window().on_close_requested(move || {
        render_ctx_close.lock().unwrap().running = false;
        // Bypasses the debounce window -- a change made in the last `DEBOUNCE`
        // interval before quitting must not be silently lost.
        settings_store_close.flush();
        slint::CloseRequestResponse::HideWindow
    });

    Ok(MainWindowHandle {
        ui,
        render_ctx,
        settings_store,
        remote_rendering_timer,
    })
}

/// Resolves the starting directory for a native `rfd` file/folder picker from a path
/// text field's current value, per this task's requirement: the field's own value if it
/// already names an existing directory, that path's parent if it names an existing
/// file, otherwise `None` (callers then leave `rfd`'s directory unset, which defaults to
/// the process's current working directory). Shared by every picker call site that
/// still fills a text field this way (`camera_lighting`'s HDR environment-map path,
/// `remote::worker_callbacks`'s certificate-bundle-folder field) so "what counts as a
/// path already in the field" stays one rule, not several copies of it -- `library`'s
/// `.asc` import pickers no longer need it: they pick and import in one step, with no
/// field left to seed a starting directory from.
fn starting_dir_from_picker_field(current: &str) -> Option<std::path::PathBuf> {
    if current.is_empty() {
        return None;
    }
    let path = std::path::Path::new(current);
    if path.is_dir() {
        Some(path.to_path_buf())
    } else if path.is_file() {
        path.parent().map(std::path::Path::to_path_buf)
    } else {
        None
    }
}

/// Shows a toast message with a 3.5s auto-dismiss timer.
///
/// A plain function (not a closure) since it captures nothing from `run_gui` --
/// every `ui.on_X` callback across this module's submodules that needs it just calls
/// it directly on its own `ui: &MainWindow`. `pub` because a downstream binary
/// reusing this window needs to surface its own results through the same toast.
pub fn show_toast(ui: &MainWindow, msg: &str, toast_type: &str) {
    ui.set_toast_message(msg.into());
    ui.set_toast_type(toast_type.into());
    ui.set_toast_visible(true);

    let ui_weak = ui.as_weak();
    slint::Timer::single_shot(std::time::Duration::from_millis(3500), move || {
        if let Some(ui) = ui_weak.upgrade() {
            ui.set_toast_visible(false);
        }
    });
}

/// Applies settings loaded from disk into the render context and into the UI's own
/// mirrored properties (`target_samples_exponent`, `resolution_index`, `bounce_index`,
/// `exposure_val`, `light_yaw_deg`, `light_pitch_deg`, `inclusion_sigma_s`,
/// `c_axis_override_enabled`/`c_axis_tilt_deg`/`c_axis_azimuth_deg`, `girdle_frosted`,
/// `edge_rounding_radius`, `stone_width_mm` -- hoisted onto `MainWindow` for exactly this reason, see the
/// comment beside them in `app.slint`). Called once at startup, before the render
/// thread or any callback is wired up, so there is no risk of a callback firing
/// mid-application and racing this.
fn apply_loaded_settings(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    loaded: &SettingsFile,
) {
    let s = &loaded.settings;
    // Parse the persisted lighting-rig label back into its enum at this one
    // boundary -- gracefully migrating any legacy or unrecognized label (including the
    // old, mislabelled `"D65 Daylight (5500K)"` string) via `from_label`'s own
    // fallback, rather than resetting the user's choice. See
    // `gemray::optics::LightingPreset::from_label`.
    let lighting_preset = LightingPreset::from_label(&s.lighting_rig);
    {
        let mut ctx = render_ctx.lock().unwrap();
        ctx.target_samples = s.target_samples;
        ctx.width = s.render_width;
        ctx.height = s.render_height;
        ctx.max_bounces = s.max_bounces;
        ctx.exposure = s.exposure;
        ctx.inclusion_sigma_s = s.inclusion_sigma_s;
        // The settings dialog drags degrees, `RenderContext` stores the already-
        // resolved `Vec3` -- see `RenderContext::c_axis_override`'s own doc comment for
        // why this crossing happens here rather than downstream in `bridge`.
        ctx.c_axis_override = s
            .c_axis_override_enabled
            .then(|| angles_to_c_axis(s.c_axis_tilt_deg, s.c_axis_azimuth_deg));
        ctx.girdle_frosted = s.girdle_frosted;
        ctx.edge_rounding_radius = s.edge_rounding_radius;
        ctx.stone_width_mm = s.stone_width_mm;
        ctx.light_yaw = s.light_yaw_deg.to_radians();
        ctx.light_pitch = s.light_pitch_deg.to_radians().clamp(0.15, 1.55);
        ctx.lighting_preset = lighting_preset;
        ctx.yaw = s.camera_yaw;
        ctx.pitch = s.camera_pitch.clamp(-1.48, 1.48);
        ctx.distance = s.camera_distance.clamp(1.2, 8.0);
        ctx.material_name.clone_from(&s.selected_material);
        ctx.denoise_enabled = s.denoise_enabled;
        // Local preview-then-settle rendering / remote render sample
        // budget -- both live-update `RenderContext` at startup exactly like every
        // other setting in this block, `camera_moving` deliberately left at its
        // `Default` (`false`): it's re-derived from live camera-pose polling by
        // `gui::remote::poll_tick` within the first tick after the window opens, never
        // something a settings FILE has an opinion on.
        ctx.local_preview_scale = s.local_preview_scale;
        ctx.remote_render_samples = s.remote_render_samples;
        ctx.live_compute_target = s.live_compute_target;
        ctx.dirty = true;
    }

    // The slider stores an EXPONENT, the settings file stores a COUNT -- see
    // `gui::sample_scale`'s module doc comment for why, and for this conversion's
    // inverse (`exponent_to_count`, used when the slider itself changes).
    ui.set_target_samples_exponent(count_to_exponent(s.target_samples) as f32);
    ui.set_resolution_index(resolution_index(s.render_width, s.render_height));
    ui.set_bounce_index(bounces_index(s.max_bounces));
    ui.set_exposure_val(s.exposure);
    ui.set_inclusion_sigma_s(s.inclusion_sigma_s);
    ui.set_c_axis_override_enabled(s.c_axis_override_enabled);
    ui.set_c_axis_tilt_deg(s.c_axis_tilt_deg);
    ui.set_c_axis_azimuth_deg(s.c_axis_azimuth_deg);
    ui.set_girdle_frosted(s.girdle_frosted);
    ui.set_edge_rounding_radius(s.edge_rounding_radius);
    ui.set_stone_width_mm(s.stone_width_mm);
    ui.set_local_preview_scale_index(local_preview_scale_index(s.local_preview_scale));
    ui.set_live_compute_target_index(live_compute_target_index(s.live_compute_target));
    ui.set_remote_render_samples_exponent(
        remote_samples_count_to_exponent(s.remote_render_samples) as f32,
    );
    ui.set_light_yaw_deg(s.light_yaw_deg);
    ui.set_light_pitch_deg(s.light_pitch_deg);
    ui.set_selected_lighting_index(lighting_preset.index());

    if let Some(idx) = find_option_index(&ui.get_material_options(), &s.selected_material) {
        ui.set_selected_material_index(idx);
    }
    // Whether the crystal-axis control is interactive at all depends on the
    // STARTING material -- must be set here too, not just from `on_material_changed`,
    // or a session restored on an isotropic material (e.g. the "Diamond" default) would
    // show the slider as enabled until the user touched the material dropdown once.
    ui.set_c_axis_override_available(is_c_axis_override_available(&resolve_material(
        &GemMaterial::all_materials(),
        &render_ctx.lock().unwrap().custom_materials,
        &s.selected_material,
    )));

    // Reload the last-loaded HDR environment map, if any. Mirrors
    // `on_load_env_map` (in `setup_environment_map_callbacks`) but runs before any
    // callback is wired up, so a load failure here can only toast, not race a
    // simultaneous user-initiated load. `s.env_map_path` is left untouched either way
    // -- a transient failure (file on a currently-unmounted drive, say) shouldn't
    // silently forget the path the next successful launch could still use.
    ui.set_env_map_path(s.env_map_path.clone().into());
    if s.env_map_path.is_empty() {
        ui.set_env_map_loaded(false);
        ui.set_env_map_status(String::new().into());
    } else {
        match load_env_map(&s.env_map_path) {
            Ok(map) => {
                ui.set_env_map_status(env_map_status_text(&map, &s.env_map_path).into());
                ui.set_env_map_loaded(true);
                render_ctx.lock().unwrap().env_map = Some(map);
            }
            Err(err) => {
                ui.set_env_map_loaded(false);
                ui.set_env_map_status(String::new().into());
                show_toast(
                    ui,
                    &format!("Could not reload saved HDR environment: {err}"),
                    "error",
                );
            }
        }
    }
}

/// The settings dialog's "Loaded: <file> (`WxH`)" status line for a decoded
/// [`gemray::renderer::env_map::EnvironmentMap`] -- shared by the startup
/// reload in `apply_loaded_settings` and the user-initiated load in
/// `setup_environment_map_callbacks` so both report the map identically. Shows the
/// file name alone (not the full path, which can be long and is already visible in the
/// path field above it).
fn env_map_status_text(map: &gemray::renderer::env_map::EnvironmentMap, path: &str) -> String {
    let name = std::path::Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned());
    format!("Loaded: {name} ({}\u{d7}{})", map.width(), map.height())
}

/// Whether the crystal-axis orientation override has any effect on
/// `material` -- `false` for an isotropic material (Diamond, Spinel, Cubic Zirconia,
/// and any custom material with `birefringence_delta == 0.0`), whose optic axis is
/// physically meaningless: there is no birefringence to orient. Drives
/// `MainWindow.c_axis_override_available`, which `settings_dialog.slint` uses to grey
/// out the control and explain why, rather than silently letting the user drag a
/// slider that does nothing.
fn is_c_axis_override_available(material: &GemMaterial) -> bool {
    material.optical_character != OpticalCharacter::Isotropic
}

/// Reverse of the mapping baked into `settings_dialog.slint`'s bounce-count pill
/// selector (4/8/12/24/64/128 at indices 0-5, raised from the old 4/8/12/16/24 ladder
/// per the `bounce_cost.rs` benchmark -- see that pill block's own doc comment for the
/// measurements). Exact matches map directly; anything else -- a settings file written
/// by an older build (e.g. the retired 16-bounce rung) or a hand-edited value -- picks
/// the *nearest* rung by absolute distance rather than a single hardcoded fallback like
/// `resolution_index`/`local_preview_scale_index` below use. Those get away with
/// "assume the default option" because their lists are short and evenly spaced; this
/// ladder now spans 4..128 unevenly (24 -> 64 is a 40-bounce gap), so a blanket
/// fallback would misplace a value like 40 or 96 by a wide margin. Never used to
/// reject or clamp the actual stored/rendered `max_bounces` -- see `on_bounces_changed`
/// in `material_quality.rs`, which honours the raw persisted value untouched regardless
/// of which pill this highlights.
const fn bounces_index(bounces: u32) -> i32 {
    const RUNGS: [u32; 6] = [4, 8, 12, 24, 64, 128];
    let mut best_idx = 0usize;
    let mut best_dist = u32::MAX;
    let mut i = 0usize;
    while i < RUNGS.len() {
        let dist = RUNGS[i].abs_diff(bounces);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
        i += 1;
    }
    best_idx as i32
}

/// Reverse of the mapping baked into `settings_dialog.slint`'s "Render Resolution"
/// pill selector (640x480/800x600/1280x720/1920x1080 at indices 0-3) -- a blanket
/// fixed-list-with-fallback treatment, used only to highlight the closest pill at
/// startup. Falls back to the 800x600 index (1) for any pair that isn't one of the
/// four presets (a hand-edited settings file, or one saved before this control existed
/// -- see `AppSettings::render_width`'s doc comment), matching
/// `RenderContext::default()`/`DEFAULT_RENDER_WIDTH`/`DEFAULT_RENDER_HEIGHT`. Never
/// used to reject or clamp the actual stored/rendered value -- `render_width`/
/// `render_height` themselves pass through untouched regardless of what this returns.
const fn resolution_index(width: u32, height: u32) -> i32 {
    match (width, height) {
        (640, 480) => 0,
        (1280, 720) => 2,
        (1920, 1080) => 3,
        _ => 1, // 800x600 and anything unrecognized
    }
}

/// Reverse of the mapping baked into `settings_dialog.slint`'s "Motion Preview
/// Resolution" pill selector (Off/Half/Quarter at indices 0-2) -- same fixed-list
/// treatment as `resolution_index` above, used to seed the pill at startup from a
/// persisted `LocalPreviewScale`.
const fn local_preview_scale_index(scale: LocalPreviewScale) -> i32 {
    match scale {
        LocalPreviewScale::Off => 0,
        LocalPreviewScale::Half => 1,
        LocalPreviewScale::Quarter => 2,
    }
}

/// Inverse of [`local_preview_scale_index`]: what `on_local_preview_scale_changed`
/// (in `setup_material_and_quality_callbacks`) converts the pill's clicked index back
/// into. Falls back to `Off` for anything outside `0..=2` (a value the fixed pill
/// selector itself can never actually send), matching `resolution_index`'s own
/// unrecognized-value fallback convention.
const fn local_preview_scale_from_index(index: i32) -> LocalPreviewScale {
    match index {
        1 => LocalPreviewScale::Half,
        2 => LocalPreviewScale::Quarter,
        _ => LocalPreviewScale::Off,
    }
}

/// Finds `needle`'s index in a Slint `[string]` model, for restoring a `ComboBox`
/// selection (material, lighting rig) from a persisted name. Returns `None` (leaving
/// the current selection untouched) rather than guessing if the name isn't present --
/// e.g. a lighting rig from an older options list, or a custom material that was
/// deleted since the settings file was last saved.
fn find_option_index(options: &ModelRc<SharedString>, needle: &str) -> Option<i32> {
    (0..options.row_count()).find_map(|i| {
        let matches = options.row_data(i).is_some_and(|s| s.as_str() == needle);
        matches.then_some(i as i32)
    })
}

/// Rebuilds the `MainWindow.lighting_presets` model from the settings store's current
/// preset list. Called after startup load and after every create/rename/delete so the
/// settings dialog's preset rows stay in sync with what's actually persisted.
fn refresh_lighting_preset_options(ui: &MainWindow, presets: &[SavedLightingPreset]) {
    let items: Vec<LightingPresetItem> = presets
        .iter()
        .map(|p| LightingPresetItem {
            name: p.name.clone().into(),
            built_in: p.built_in,
        })
        .collect();
    ui.set_lighting_presets(ModelRc::new(VecModel::from(items)));
}

/// Wires up the high-resolution export flow: validates the request, captures
/// a `SceneSnapshot` independent of the live viewport's `RenderContext.width`/`height`
/// and accumulation buffer (see `export_thread`'s module doc comment for why that
/// separation matters), and spawns it on its own worker thread via
/// `export_thread::spawn_export`. The returned `ExportHandle` is kept in a
/// `Rc<RefCell<Option<_>>>` -- plain UI-thread-only state, not `Arc<Mutex<_>>>`, since
/// both callbacks here only ever run on the Slint event loop -- so `cancel_export` can
/// reach the in-flight export. Split out of `run_gui` purely to keep that function
/// under clippy's function-length lint.
/// Inverse of `export_dialog.slint`'s "Colour Space" pill selector
/// (sRGB/Display P3/Rec.2020 at indices 0-2) -- same fixed-list-with-fallback treatment
/// as `local_preview_scale_from_index` above. Falls back to `ColorSpace::Srgb` (index
/// 0, the required default -- see `bridge::export_thread`'s module doc comment on why
/// that space's output must stay byte-identical to before this control existed) for
/// any value the fixed pill selector itself can never actually send.
///
/// `ColorSpace::AcesCg` has no index here at all -- it is not offered by the picker,
/// see `export_dialog.slint`'s own doc comment for why a scene-linear space doesn't
/// belong in an 8-bit PNG export.
const fn color_space_from_index(index: i32) -> ColorSpace {
    match index {
        1 => ColorSpace::DisplayP3,
        2 => ColorSpace::Rec2020,
        _ => ColorSpace::Srgb,
    }
}

fn refresh_material_options(ui: &MainWindow, custom_mats: &[GemMaterial]) {
    let mut names = vec![
        "Diamond".to_string(),
        "Sapphire".to_string(),
        "Ruby".to_string(),
        "Emerald".to_string(),
        "Zircon".to_string(),
        "Tanzanite".to_string(),
        "Synthetic Moissanite".to_string(),
        "Topaz".to_string(),
        "Spinel".to_string(),
        "Quartz".to_string(),
        "Cubic Zirconia".to_string(),
    ];
    for m in custom_mats {
        if !names.iter().any(|n| n.eq_ignore_ascii_case(&m.name)) {
            names.push(m.name.clone());
        }
    }
    let model: Vec<SharedString> = names.into_iter().map(std::convert::Into::into).collect();
    ui.set_material_options(std::rc::Rc::new(slint::VecModel::from(model)).into());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounces_index_round_trips_all_six_pill_values() {
        assert_eq!(bounces_index(4), 0);
        assert_eq!(bounces_index(8), 1);
        assert_eq!(bounces_index(12), 2);
        assert_eq!(bounces_index(24), 3);
        assert_eq!(bounces_index(64), 4);
        assert_eq!(bounces_index(128), 5);
    }

    #[test]
    fn bounces_index_picks_the_nearest_rung_for_unknown_values() {
        // The retired 16-bounce rung from the old ladder: 4 away from 12, 8 away
        // from 24, so it lands on the same index (2) the old blanket fallback used
        // to give it -- but via nearest-distance, not a hardcoded default.
        assert_eq!(bounces_index(16), 2);
        // Below the lowest rung and above the highest rung both clamp to the nearest
        // end of the ladder rather than falling back to the default (12).
        assert_eq!(bounces_index(1), 0);
        assert_eq!(bounces_index(999), 5);
        // Roughly equidistant between 24 and 64 (20 either way) -- ties resolve to
        // whichever rung is checked first in `RUNGS`, i.e. the lower one.
        assert_eq!(bounces_index(44), 3);
    }

    #[test]
    fn resolution_index_round_trips_all_four_pill_values() {
        assert_eq!(resolution_index(640, 480), 0);
        assert_eq!(resolution_index(800, 600), 1);
        assert_eq!(resolution_index(1280, 720), 2);
        assert_eq!(resolution_index(1920, 1080), 3);
    }

    #[test]
    fn resolution_index_falls_back_to_800x600_for_unknown_values() {
        assert_eq!(resolution_index(1, 1), 1);
        assert_eq!(resolution_index(3840, 2160), 1);
        // A mismatched pair (e.g. a hand-edited file with one dimension changed but not
        // the other) must not accidentally match a pill via one coordinate alone.
        assert_eq!(resolution_index(640, 600), 1);
    }

    #[test]
    fn local_preview_scale_index_round_trips_all_three_pill_values() {
        for scale in [
            LocalPreviewScale::Off,
            LocalPreviewScale::Half,
            LocalPreviewScale::Quarter,
        ] {
            let idx = local_preview_scale_index(scale);
            assert_eq!(
                local_preview_scale_from_index(idx),
                scale,
                "scale={scale:?}"
            );
        }
        assert_eq!(local_preview_scale_index(LocalPreviewScale::Off), 0);
        assert_eq!(local_preview_scale_index(LocalPreviewScale::Half), 1);
        assert_eq!(local_preview_scale_index(LocalPreviewScale::Quarter), 2);
    }

    #[test]
    fn local_preview_scale_from_index_falls_back_to_off_for_unknown_values() {
        assert_eq!(local_preview_scale_from_index(-1), LocalPreviewScale::Off);
        assert_eq!(local_preview_scale_from_index(99), LocalPreviewScale::Off);
    }

    #[test]
    fn color_space_from_index_maps_all_three_pill_values() {
        assert_eq!(color_space_from_index(0), ColorSpace::Srgb);
        assert_eq!(color_space_from_index(1), ColorSpace::DisplayP3);
        assert_eq!(color_space_from_index(2), ColorSpace::Rec2020);
    }

    #[test]
    fn color_space_from_index_falls_back_to_srgb_for_unknown_values() {
        assert_eq!(color_space_from_index(-1), ColorSpace::Srgb);
        assert_eq!(color_space_from_index(99), ColorSpace::Srgb);
    }

    #[test]
    fn find_option_index_locates_an_exact_match() {
        let options: ModelRc<SharedString> = ModelRc::new(VecModel::from(vec![
            SharedString::from("Diamond"),
            SharedString::from("Sapphire"),
            SharedString::from("Ruby"),
        ]));
        assert_eq!(find_option_index(&options, "Sapphire"), Some(1));
        assert_eq!(find_option_index(&options, "Ruby"), Some(2));
    }

    #[test]
    fn find_option_index_returns_none_when_absent_rather_than_guessing() {
        let options: ModelRc<SharedString> =
            ModelRc::new(VecModel::from(vec![SharedString::from("Diamond")]));
        assert_eq!(find_option_index(&options, "Moissanite"), None);
    }

    #[test]
    fn find_option_index_on_an_empty_model_returns_none() {
        let options: ModelRc<SharedString> =
            ModelRc::new(VecModel::from(Vec::<SharedString>::new()));
        assert_eq!(find_option_index(&options, "anything"), None);
    }
}

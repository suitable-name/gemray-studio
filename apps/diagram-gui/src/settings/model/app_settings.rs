//! [`AppSettings`]: the four originally in-memory-only settings, plus camera pose,
//! the selected material, and everything else migrated into persistent storage.
//!
//! Split out of `settings::model` purely to keep that module (already sizeable) from
//! growing further.

use super::worker::{LiveComputeTarget, LocalPreviewScale, WorkerSettings};
use serde::{Deserialize, Serialize};

// Defaults mirror `bridge::render_thread::RenderContext::default()` and
// `ui/components/settings_dialog.slint`'s property initializers. Camera yaw/pitch are
// kept in RADIANS (matching `RenderContext`, their only consumer) while light
// yaw/pitch are kept in DEGREES (matching the settings-dialog sliders, their primary
// consumer) -- each field stores whatever unit its main consumer already uses, so
// loading/applying settings never needs a silent unit conversion that could be gotten
// backwards. See `gui::mod::apply_loaded_settings` for the one place that does convert
// light degrees -> radians, exactly as the existing `on_light_pos_changed` callback does.
/// Replaces `DEFAULT_QUALITY_PRESET` now that the four-tier `QualityPreset`
/// pill selector is gone in favor of a single sample-count slider. `256` matches the
/// old "High / Quality" preset's typical converged sample count closely enough to
/// keep a fresh install's default render looking the same as before this control
/// existed -- see `bridge::render_thread::RenderContext::default()`'s own
/// `target_samples`, which mirrors this.
pub const DEFAULT_TARGET_SAMPLES: u32 = 256;
/// Live render resolution -- matches `RenderContext::default()`'s `width`/
/// `height` exactly, so an existing user's first launch after this setting was added
/// looks exactly as it did before (a settings file predating this field defaults here
/// via `#[serde(default)]` on `AppSettings`, same as every other field in this file).
pub const DEFAULT_RENDER_WIDTH: u32 = 800;
pub const DEFAULT_RENDER_HEIGHT: u32 = 600;
pub const DEFAULT_MAX_BOUNCES: u32 = 12;
pub const DEFAULT_EXPOSURE: f32 = 1.0;
/// Inclusion/subsurface scattering amount, off by default -- see
/// `AppSettings::inclusion_sigma_s`'s own doc comment.
pub const DEFAULT_INCLUSION_SIGMA_S: f32 = 0.0;
/// Crystal-axis orientation override, off ("as cut") by default -- see
/// `AppSettings::c_axis_override_enabled`'s own doc comment.
pub const DEFAULT_C_AXIS_OVERRIDE_ENABLED: bool = false;
pub const DEFAULT_C_AXIS_TILT_DEG: f32 = 0.0;
pub const DEFAULT_C_AXIS_AZIMUTH_DEG: f32 = 0.0;
/// Bruted (frosted) girdle finish, off by default -- see
/// `AppSettings::girdle_frosted`'s own doc comment.
pub const DEFAULT_GIRDLE_FROSTED: bool = false;
/// Facet edge rounding radius, off by default -- see
/// `AppSettings::edge_rounding_radius`'s own doc comment.
pub const DEFAULT_EDGE_ROUNDING_RADIUS: f32 = 0.0;
/// Physical stone size (girdle width in millimetres), off by default -- see
/// `AppSettings::stone_width_mm`'s own doc comment.
pub const DEFAULT_STONE_WIDTH_MM: f32 = 0.0;
/// Local preview-then-settle rendering, off by default -- see
/// `AppSettings::local_preview_scale`'s own doc comment.
pub const DEFAULT_LOCAL_PREVIEW_SCALE: LocalPreviewScale = LocalPreviewScale::Off;
/// Remote render sample budget: unchanged from the value this used to be
/// hardcoded to (`gui::remote::REMOTE_RENDER_SAMPLES`) -- see
/// `AppSettings::remote_render_samples`'s own doc comment for why this stays the
/// default rather than moving to the middle of the new slider's range.
pub const DEFAULT_REMOTE_RENDER_SAMPLES: u32 = 512;
/// Live rendering's Local/Remote/Local+Remote choice, off by default in the sense that
/// matters: `LiveComputeTarget::Both` is a no-op without a configured worker (`gui::
/// remote::orchestrator::poll_tick` only ever dispatches remote when a worker is BOTH
/// selected by this setting AND actually present in `remote_workers`), so a fresh
/// install with no worker configured behaves exactly as this crate always has --
/// `Both` only starts doing anything the moment a worker is added, at which point
/// combining is exactly the "defaults to Local+Remote when a worker with render
/// capacity is configured" behaviour the live-rendering task asked for, with no extra
/// bookkeeping needed to notice "a worker just became available" the way a dynamic
/// UI-only default (like `export_dialog.slint`'s own `compute_target` binding) would
/// need.
pub const DEFAULT_LIVE_COMPUTE_TARGET: LiveComputeTarget = LiveComputeTarget::Both;
pub const DEFAULT_LIGHT_YAW_DEG: f32 = 48.0;
pub const DEFAULT_LIGHT_PITCH_DEG: f32 = 54.0;
pub const DEFAULT_LIGHTING_RIG: &str = "Gem Studio Ring Lights";
pub const DEFAULT_CAMERA_YAW: f32 = 0.60;
pub const DEFAULT_CAMERA_PITCH: f32 = 0.45;
pub const DEFAULT_CAMERA_DISTANCE: f32 = 2.4;
pub const DEFAULT_MATERIAL: &str = "Diamond";

/// The four originally in-memory-only settings, plus camera pose and the selected
/// material -- everything migrated into persistent storage.
///
/// `#[serde(default)]` on the struct makes every field individually optional on
/// deserialization: a settings file that predates a field (or was hand-edited to drop
/// one) still loads successfully with that one field defaulted, rather than the whole
/// document being rejected. Full-document parse failure (bad TOML syntax) is handled
/// one level up, in `store::load_or_default`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// The user's progressive-accumulation target sample count, replacing the
    /// old four-tier `quality_preset` string. `#[serde(default)]` on the struct means
    /// a settings file saved before this field existed (still possibly carrying the
    /// now-unrecognized `quality_preset` key, which TOML silently drops -- there is no
    /// `#[serde(deny_unknown_fields)]` anywhere in this file) loads fine, defaulting
    /// this to `DEFAULT_TARGET_SAMPLES`. See
    /// `tests::a_settings_file_with_the_old_quality_preset_key_still_loads_with_target_samples_defaulted`.
    pub target_samples: u32,
    /// Live render resolution, driving `RenderContext.width`/`.height` directly
    /// (they used to be hardcoded to `800x600` in `RenderContext::default()` and never
    /// written again anywhere). Restricted in the UI to a fixed pill list -- 640x480,
    /// 800x600 (this field's default), 1280x720, 1920x1080 -- rather than tracking the
    /// viewport `Image`'s actual on-screen size, which Slint has no mechanism to report
    /// back to Rust; wiring that through would be a substantially larger change than
    /// this control. See `settings_dialog.slint`'s "Render Resolution" pill section for
    /// where that list lives. A hand-edited or foreign value outside the four pills
    /// still loads and renders fine (nothing here validates it against the list) --
    /// only `gui::mod::resolution_index`, used purely to highlight the closest pill at
    /// startup, falls back for a value that isn't an exact match.
    pub render_width: u32,
    pub render_height: u32,
    pub max_bounces: u32,
    pub exposure: f32,
    pub light_yaw_deg: f32,
    pub light_pitch_deg: f32,
    /// The lighting "rig" selection (e.g. "Gem Studio Ring Lights") -- what
    /// `RenderContext::lighting_preset` / the lighting `ComboBox` calls a "lighting
    /// preset". Named `lighting_rig` here to keep it unambiguous next to `LightingPreset`
    /// (the saveable lighting-rig bundle, which itself has a `lighting_rig` field referencing this).
    pub lighting_rig: String,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_distance: f32,
    pub selected_material: String,
    /// Configured remote render workers. Global (not
    /// per-session) -- see `add_worker`/`update_worker`/`remove_worker` for the CRUD
    /// the worker-list panel drives.
    #[serde(default)]
    pub remote_workers: Vec<WorkerSettings>,
    /// Whether the À-Trous denoiser is applied to the merged accumulation, regardless
    /// of which backend (local CPU or a remote worker) produced it -- a single toggle
    /// covering the WHOLE image, never per-source: denoising is nonlinear, so it can
    /// only ever be applied once, to the fully merged result (see
    /// `bridge::render_thread`'s own doc comment on why the raw accumulation buffer is
    /// never overwritten with filtered output).
    #[serde(default = "default_denoise_enabled")]
    pub denoise_enabled: bool,
    /// Inclusion/subsurface scattering amount: the Henyey-Greenstein
    /// `sigma_s` applied on top of the resolved material via
    /// `GemMaterial::with_scattering_amount` -- see that method's, and
    /// `scattering_sigma_s`'s own, doc comments in `crates/gemray/src/optics/materials.rs`
    /// for the physical meaning and the `0.05`-`3.0` useful range this backs the
    /// settings-dialog slider with. `0.0` (the default, and every built-in material's
    /// own stored value) means off: the exact deterministic Beer-Lambert path,
    /// unchanged from before this control existed.
    pub inclusion_sigma_s: f32,
    /// Crystal-axis orientation override -- off ("as cut") by default, leaving
    /// each material's own `GemMaterial::c_axis` untouched exactly like
    /// `inclusion_sigma_s`'s `0.0` skips `with_scattering_amount` above. When on,
    /// `c_axis_tilt_deg`/`c_axis_azimuth_deg` below replace it via
    /// `gui::c_axis::angles_to_c_axis` -- see that module's doc comment for the
    /// `(sin theta cos phi, cos theta, sin theta sin phi)` mapping. Disabled in the UI
    /// (see `gui::mod`'s `c_axis_override_available`) for an isotropic material, whose
    /// optic axis is physically meaningless.
    pub c_axis_override_enabled: bool,
    /// Tilt from the table normal (`+Y`), 0-90 degrees. `0.0` reproduces every
    /// material's own default `c_axis` (`Vec3::Y`) exactly; `90.0` (at
    /// `c_axis_azimuth_deg = 0.0`) reproduces Tourmaline's own cut-orientation override
    /// (`Vec3::X`) to within `f32` trig precision -- see `gui::c_axis`'s own tests for
    /// both endpoints.
    pub c_axis_tilt_deg: f32,
    /// Azimuth around `+Y`, 0-360 degrees.
    pub c_axis_azimuth_deg: f32,
    /// Bruted (frosted) girdle finish toggle -- off by default, every facet
    /// `FacetFinish::Polished` (today's behaviour, unchanged). On, the girdle band
    /// identified by `gemray::geometry::girdle::girdle_facet_finishes` renders
    /// `Frosted` (diffusely scattering) instead -- a measured +17.3% face-up brightness
    /// change on the standard round brilliant, the motivating measurement for this control.
    pub girdle_frosted: bool,
    /// Facet (meet-point) edge rounding radius, in the same world units as
    /// `scattering_sigma_s`'s own useful-range doc comment (girdle radius of order 1
    /// model unit) -- see `GemMaterial::edge_rounding_radius`'s own doc comment in
    /// `crates/gemray/src/optics/materials.rs`, which calls `0.01` "comfortably" inside
    /// the "throws a soft glint, does not visibly bevel the facet" range. `0.0` (the
    /// default) means off: perfectly sharp edges, bit-identical to before this control
    /// existed. The settings-dialog slider's `0.03` ceiling is the largest radius this
    /// project's own physics review has actually verified end to end -- the energy-
    /// conservation furnace anchor in
    /// `crates/gemray/src/renderer/gpu/estimator_check.rs::run_furnace_edge_rounding`
    /// uses exactly that value (its Tier 3 Diamond image-comparison test one step
    /// below it, `run_image_comparison_edge_rounding`, uses `0.02`).
    pub edge_rounding_radius: f32,
    /// Physical stone size: the width across the girdle, in millimetres, to treat
    /// the active design as measuring -- see `RenderContext::stone_width_mm`'s own doc
    /// comment in `bridge::render_thread::context` for the full mechanism
    /// (`GemMaterial::with_absorption_path_scale`, driven by the design's own measured
    /// model-unit girdle width). `0.0` (the default) means off: unscaled, exactly this
    /// crate's pre-existing behaviour, same convention as `edge_rounding_radius` and
    /// `inclusion_sigma_s` above.
    pub stone_width_mm: f32,
    /// Local preview-then-settle rendering -- while the camera is moving, trace
    /// at a fraction of `render_width`/`render_height` (this scale), then snap back to
    /// full resolution once it settles. `Off` (the default) reproduces this crate's
    /// pre-existing behaviour exactly -- see `bridge::local_preview::effective_dimensions`'s
    /// own doc comment for the mechanism, and `RenderContext::camera_moving`'s for how
    /// "settled" is decided (reusing `bridge::handoff`'s own debounce rather than a
    /// second, differently-tuned one).
    pub local_preview_scale: LocalPreviewScale,
    /// The one-shot full-quality remote render's
    /// total sample count -- global (like `denoise_enabled`), not per-worker, since
    /// every worker in a session renders the identical request. `512` by default,
    /// matching the value this used to be hardcoded to
    /// (`gui::remote::REMOTE_RENDER_SAMPLES`) before this control existed, so behaviour
    /// is unchanged for anyone who doesn't touch it. See
    /// `gui::remote::REMOTE_SAMPLES_MIN_EXPONENT`/`MAX_EXPONENT` for this control's
    /// slider range.
    pub remote_render_samples: u32,
    /// Live rendering's Local/Remote/Local+Remote choice -- see
    /// `LiveComputeTarget`'s own doc comment. `#[serde(default)]` so a settings file
    /// saved before this field existed loads with it defaulted to
    /// `DEFAULT_LIVE_COMPUTE_TARGET` (`Both`), same treatment `env_map_path` below
    /// already gets.
    #[serde(default)]
    pub live_compute_target: LiveComputeTarget,
    /// HDR environment maps: path to the last-loaded Radiance `.hdr` file, or
    /// empty for "no map loaded, use the studio rig" -- the same empty-string-means-off
    /// convention `selected_material`/`lighting_rig` already use for their own
    /// always-present `String` fields, rather than an `Option<String>`, so a settings
    /// file written by an earlier version of this app (before this field existed) needs
    /// no migration beyond `#[serde(default)]`'s usual empty-string default. A plain
    /// `String`, not `PathBuf`, matching every other user-facing path field in this file
    /// (`WorkerSettings::cert_dir`). `gui::mod::apply_loaded_settings` attempts to
    /// reload it at startup via `bridge::render_thread::load_env_map`; a failure (the
    /// file moved or was deleted since it was last loaded) surfaces as a toast and
    /// leaves the render on the studio rig -- it does not block startup or clear this
    /// field, so fixing the file and relaunching picks the map back up without the user
    /// having to re-browse for it.
    #[serde(default)]
    pub env_map_path: String,
}

const fn default_denoise_enabled() -> bool {
    true
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            target_samples: DEFAULT_TARGET_SAMPLES,
            render_width: DEFAULT_RENDER_WIDTH,
            render_height: DEFAULT_RENDER_HEIGHT,
            max_bounces: DEFAULT_MAX_BOUNCES,
            exposure: DEFAULT_EXPOSURE,
            light_yaw_deg: DEFAULT_LIGHT_YAW_DEG,
            light_pitch_deg: DEFAULT_LIGHT_PITCH_DEG,
            lighting_rig: DEFAULT_LIGHTING_RIG.to_string(),
            camera_yaw: DEFAULT_CAMERA_YAW,
            camera_pitch: DEFAULT_CAMERA_PITCH,
            camera_distance: DEFAULT_CAMERA_DISTANCE,
            selected_material: DEFAULT_MATERIAL.to_string(),
            remote_workers: Vec::new(),
            denoise_enabled: true,
            inclusion_sigma_s: DEFAULT_INCLUSION_SIGMA_S,
            c_axis_override_enabled: DEFAULT_C_AXIS_OVERRIDE_ENABLED,
            c_axis_tilt_deg: DEFAULT_C_AXIS_TILT_DEG,
            c_axis_azimuth_deg: DEFAULT_C_AXIS_AZIMUTH_DEG,
            girdle_frosted: DEFAULT_GIRDLE_FROSTED,
            edge_rounding_radius: DEFAULT_EDGE_ROUNDING_RADIUS,
            stone_width_mm: DEFAULT_STONE_WIDTH_MM,
            local_preview_scale: DEFAULT_LOCAL_PREVIEW_SCALE,
            remote_render_samples: DEFAULT_REMOTE_RENDER_SAMPLES,
            live_compute_target: DEFAULT_LIVE_COMPUTE_TARGET,
            env_map_path: String::new(),
        }
    }
}

impl AppSettings {
    /// Adds a new remote worker to the end of the list -- the worker-list panel's
    /// "add" affordance.
    pub fn add_worker(&mut self, worker: WorkerSettings) {
        self.remote_workers.push(worker);
    }

    /// Overwrites the worker at `index` -- the worker-list panel's "edit" affordance.
    ///
    /// # Errors
    ///
    /// Returns an error message if `index` is out of range.
    pub fn update_worker(&mut self, index: usize, worker: WorkerSettings) -> Result<(), String> {
        let slot = self
            .remote_workers
            .get_mut(index)
            .ok_or_else(|| format!("No worker at index {index}."))?;
        *slot = worker;
        Ok(())
    }

    /// Removes the worker at `index` -- the worker-list panel's "remove" affordance.
    ///
    /// # Errors
    ///
    /// Returns an error message if `index` is out of range.
    pub fn remove_worker(&mut self, index: usize) -> Result<(), String> {
        if index >= self.remote_workers.len() {
            return Err(format!("No worker at index {index}."));
        }
        self.remote_workers.remove(index);
        Ok(())
    }
}

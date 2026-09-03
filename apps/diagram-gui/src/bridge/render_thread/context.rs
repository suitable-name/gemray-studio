//! The `RenderContext`/`FrameInputs` state: the shared, live-mutated render
//! configuration the GUI writes into and the render loop reads a per-frame snapshot
//! from, plus material resolution and per-frame quality derivation.
//!
//! Split out of `bridge::render_thread` purely to keep that module (already sizeable)
//! from growing further.

use crate::{
    bridge::stone_width::StoneWidthCache,
    settings::model::{LiveComputeTarget, LocalPreviewScale},
};
use gemray::{
    geometry::{cuts::StandardGemCuts, plane::GpuFacetPlane},
    optics::{
        materials::{GemMaterial, OpticalCharacter},
        raytracer::LightingPreset,
    },
    renderer::env_map::{EnvMapError, EnvironmentMap},
};
use gemray_net::client::Accumulator;
use glam::Vec3;
use std::sync::{Arc, Mutex};

pub struct RenderContext {
    /// Task: live render resolution, user-controlled via `settings_dialog.slint`'s
    /// "Render Resolution" pill selector (`gui::mod`'s `on_resolution_changed` is the
    /// only place besides `Default::default` below and `apply_loaded_settings` that
    /// writes these). Restricted to a FIXED list of pill choices (640x480, 800x600,
    /// 1280x720, 1920x1080) rather than tracking the viewport `Image`'s actual
    /// on-screen size: Slint has no built-in mechanism to report a widget's rendered
    /// size back to Rust, and the `Image` here is `width: 100%; height: 100%;
    /// image-fit: contain` (stretches whatever buffer size this produces to fill the
    /// viewport) -- wiring the real size through would need a new Rust<->Slint channel
    /// entirely, which is a substantially larger change than this control. A scoping
    /// decision, not an oversight. `update_accumulation_state` (in this same file)
    /// resets progressive accumulation -- and the denoiser's guide buffers and
    /// `FramebufferTransfer`, which it also reallocates -- whenever these differ from
    /// its own `last_width`/`last_height`, so a resolution change here always starts a
    /// clean accumulation on the very next frame. `gui::remote::start_remote_render`
    /// reads these same two fields to size its request, so a remote render inherits
    /// whatever resolution is set here for free. These two fields are always the
    /// CONFIGURED resolution -- the render loop's `local_preview_scale`/`camera_moving`
    /// below never mutate them, only shadow a LOCAL, possibly-reduced copy for the
    /// duration of one frame (see `bridge::local_preview::effective_dimensions`), so
    /// `start_remote_render`/export both keep sizing off the true configured value
    /// regardless of what's currently on screen.
    pub width: u32,
    pub height: u32,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub light_yaw: f32,
    pub light_pitch: f32,
    pub material_name: String,
    pub lighting_preset: LightingPreset,
    /// User-controlled progressive-accumulation target, replacing the old
    /// four-tier `QualityPreset`. The render loop stops accumulating once
    /// `accum_samples` reaches this value; samples-per-frame is derived from it, not
    /// chosen directly -- see `resolve_material_and_quality`'s doc comment.
    pub target_samples: u32,
    pub max_bounces: u32,
    pub exposure: f32,
    /// Inclusion/subsurface scattering amount, applied on top of the
    /// resolved material in `resolve_material_and_quality` via
    /// `GemMaterial::with_scattering_amount`. `0.0` (the default) means off --
    /// see `GemMaterial::scattering_sigma_s`'s own doc comment for the useful range
    /// (`0.05` barely perceptible, `3.0` milky/heavily included) this slider covers.
    pub inclusion_sigma_s: f32,
    /// Crystal-axis orientation override: `Some(axis)` replaces the resolved
    /// material's own `c_axis` (skipped for an isotropic material -- see
    /// `apply_material_overrides`'s doc comment); `None` (the default, "as cut") leaves
    /// it untouched. Already resolved to a `Vec3` here -- `gui::mod` is the one
    /// boundary crossing from the settings dialog's tilt/azimuth-degree sliders to this
    /// physical direction, via `gui::c_axis::angles_to_c_axis`, matching how
    /// `target_samples` above is already the resolved sample count rather than the
    /// slider's own exponent.
    pub c_axis_override: Option<Vec3>,
    /// Bruted (frosted) girdle finish toggle: `true` renders the identified
    /// girdle band `FacetFinish::Frosted` instead of the default `Polished` -- see
    /// `bridge::girdle_finish::GirdleFinishCache`'s doc comment for how the band is
    /// identified and cached. `false` (the default) reproduces today's all-polished
    /// path exactly.
    pub girdle_frosted: bool,
    /// Facet edge (meet-point) rounding radius, applied on top of the resolved
    /// material via `GemMaterial::with_edge_rounding`. `0.0` (the default) means off --
    /// perfectly sharp edges, bit-identical to before this control existed. See
    /// `GemMaterial::edge_rounding_radius`'s own doc comment in
    /// `crates/gemray/src/optics/materials.rs` for units/range.
    pub edge_rounding_radius: f32,
    /// Physical stone size: the width across the girdle, in millimetres, the
    /// active design should be treated as measuring for absorption/scattering
    /// purposes -- applied on top of the resolved material via
    /// `GemMaterial::with_absorption_path_scale`. `0.0` (the default, "off") means
    /// today's behaviour: every built-in cut already renders at its own ~1-model-unit
    /// girdle radius with `absorption_path_scale = 1.0`, so a stone rendered without
    /// dialling this in is bit-identical to before this control existed. When positive,
    /// `apply_material_overrides` divides this by the active design's own measured
    /// model-unit girdle width (`gemray::geometry::stone_metrics::measure_solid`, cached
    /// per design by `bridge::stone_width::StoneWidthCache`) to get the scale factor --
    /// so typing "6.5" here renders the CURRENT cut and material as if it were physically
    /// cut from a 6.5mm-wide rough, with deeper/thicker absorption for a larger stone and
    /// lighter absorption for a smaller one, without changing a single facet angle.
    pub stone_width_mm: f32,
    pub active_planes: Vec<GpuFacetPlane>,
    pub custom_materials: Vec<GemMaterial>,
    /// Shutdown signal for the render thread. Setting this to `false` ends the thread's loop
    /// *permanently* -- there is no way to restart it. This must never be reused as a pause
    /// mechanism; see `paused` for that.
    pub running: bool,
    pub dirty: bool,
    /// User-initiated pause/stop rendering control. Independent of `tab_visible`
    /// below -- both are checked by the render loop, and either being "off" suspends rendering,
    /// but they must not be conflated: switching tabs must never silently clear an explicit
    /// user pause, and pausing must never be observable as a tab-visibility change.
    pub paused: bool,
    /// Automatic suspend when the 3D viewport tab isn't the active tab. Sourced from the UI's
    /// `active_tab` via the `tab_selected` callback. Not user-facing on its own.
    pub tab_visible: bool,
    /// Whether the À-Trous denoiser is applied to the tone-mapped output -- the remote
    /// rendering global denoise toggle, see `settings::model::AppSettings::denoise_enabled`'s
    /// doc comment. `true` by default, matching this app's behavior before the toggle
    /// existed. Independent of `remote_active` below: this setting is about WHETHER to
    /// denoise, not WHICH backend produced the samples being denoised.
    pub denoise_enabled: bool,
    /// Set by the remote-rendering orchestrator (`bridge::handoff`/`bridge::remote_render`,
    /// driven from `gui::mod`) while a remote worker owns the currently-displayed
    /// image -- suspends local tracing exactly like `paused`/`tab_visible` (see the loop
    /// below), so local CPU samples never accumulate into a buffer a remote frame is
    /// about to replace. Distinct from `paused`: this is driven by the handoff state
    /// machine, not directly by the user, and must not be observable as a user-visible
    /// pause (the pause button's own displayed state stays whatever the user set it to).
    ///
    /// Deliberately stays `true` PAST a successful completion (`RemoteUpdate::Done`),
    /// not just through `Settling`/`RemoteRendering` -- a finished remote render IS the
    /// settled, full-quality image; there is nothing for local tracing to improve, and
    /// letting it resume immediately is exactly the bug this comment used to invite (see
    /// the regression this field's history fixed: the local renderer racing back in and
    /// progressively overwriting a just-finished remote image with a rough low-spp one).
    /// `gui::remote::orchestrator`'s `RemoteUpdate::Done` handler therefore no longer
    /// clears this itself. It is only ever cleared by:
    /// - `HandoffAction::DiscardRemotePartial` (a failed/cancelled remote attempt --
    ///   local must take back over immediately), or
    /// - `render_thread::mod`'s `resolve_remote_ownership`, the SINGLE choke point that
    ///   releases a *completed* render's ownership once the scene is genuinely
    ///   invalidated for a reason other than the hand-off itself -- a fresh camera drag,
    ///   or any of the ~25 `ctx.dirty = true` call sites across `gui::*` (material,
    ///   lighting, quality, resolution, c-axis, inclusion, girdle, edge rounding, stone
    ///   width, HDR env map, lighting presets, a different design loading). See that
    ///   function's own doc comment for exactly how it tells a legitimate hand-off-start
    ///   apart from a real invalidation using only `dirty`/`remote_active`, with no
    ///   per-call-site changes -- which is what makes it impossible for a future
    ///   callback that sets `dirty = true` to silently reintroduce the freeze this field
    ///   would otherwise cause.
    ///
    /// # `live_compute_target == Both`: ownership becomes "still contributing", not
    /// # "suspends"
    ///
    /// The suspension story above is unchanged for [`LiveComputeTarget::RemoteOnly`] --
    /// this field means exactly what it always has. For [`LiveComputeTarget::Both`],
    /// `render_thread::mod`'s loop no longer suspends tracing while this is `true` (see
    /// `SuspensionFlags`): local tracing keeps running, at a sample-index offset past
    /// whatever `remote_reserved_samples` reserves for the dispatched remote render, and
    /// this field instead gates whether the render loop's own display cycle folds
    /// [`remote_accumulator`](Self::remote_accumulator)'s current running total into the
    /// image it shows (see `should_combine_remote`). `resolve_remote_ownership` stays
    /// the single choke point that releases it either way -- a genuine scene
    /// invalidation stops local from suspending (`RemoteOnly`) or combining (`Both`) with a
    /// now-stale remote contribution, exactly the same release, serving two different
    /// effects depending on the mode.
    pub remote_active: bool,
    /// The remote accumulator backing the CURRENT settle's dispatched render, when
    /// `live_compute_target == Both` -- see [`remote_active`](Self::remote_active)'s doc
    /// comment. Set (alongside `remote_active`/`dirty`/[`remote_reserved_samples`](Self::remote_reserved_samples),
    /// in ONE locked mutation) by `gui::remote::orchestrator::start_remote_render`, and
    /// cleared by the same discard paths that clear `remote_active` on a failed/
    /// cancelled attempt (`HandoffAction::DiscardRemotePartial`, `RemoteUpdate::Failed`).
    /// Deliberately NOT cleared on a successful `RemoteUpdate::Done` -- a finished remote
    /// render's contribution keeps being folded into the combined image for the rest of
    /// this settle, the same "stays set past completion" treatment `remote_active`
    /// itself already gets, for the same reason: there is nothing to discard, only more
    /// local samples to keep adding on top.
    ///
    /// `Arc<Mutex<..>>`, not owned outright: the SAME accumulator instance
    /// `gui::remote::orchestrator`'s own remote-render worker thread is concurrently
    /// summing `FRAME` deltas into (see `bridge::remote_render::run`) -- this crate never
    /// creates a second one. The render thread only ever calls
    /// [`gemray_net::client::Accumulator::buffer`]/[`gemray_net::client::Accumulator::samples_done`]
    /// on it (never `last_preview`, and never applies a `StreamEvent` itself) -- see this
    /// module's own doc comment on why a `PREVIEW` snapshot must never reach a
    /// full-resolution accumulator.
    pub remote_accumulator: Option<Arc<Mutex<Accumulator>>>,
    /// How many of the absolute sample indices `[0, remote_reserved_samples)` are
    /// reserved for the remote render dispatched for the CURRENT settle -- `0` whenever
    /// nothing is reserved (not combining, or no remote dispatch this settle), which
    /// reproduces this crate's pre-existing behaviour exactly (local's own absolute
    /// sample index starts at its own count, unshifted). Local tracing's per-frame
    /// `sample_offset` (`render_thread::mod`'s loop) is this value plus however many
    /// samples local itself has traced so far THIS epoch, so its used indices always
    /// start exactly where remote's assigned range ends -- the same
    /// `local_start == remote_first + remote_count` disjointness arithmetic
    /// `bridge::export_thread::run_export` uses, specialised to a remote budget that is
    /// always dispatched as one fixed `[0, remote_render_samples)` request rather than a
    /// calibrated split (a live remote render has no separate "local's own share of a
    /// shared total" to calibrate against -- local just resumes counting past whatever
    /// remote was given, and keeps going for as long as the user's `target_samples`
    /// still calls for more).
    pub remote_reserved_samples: u32,
    /// Set by `gui::render_export::setup_render_export_callbacks` for the duration of a
    /// high-resolution export -- suspends local tracing exactly like
    /// `paused`/`tab_visible`/`remote_active` above (see the render loop's suspend gate),
    /// so the viewport stops burning CPU cores (and, with `--features gpu`, contending
    /// for `GpuBackend`'s shared `Mutex` -- see that type's own doc comment) on a picture
    /// nobody is watching while an export runs. Distinct from `paused` for the exact same
    /// reason `remote_active` is: this is driven by the export worker's own lifecycle,
    /// not directly by the user, and must not be observable as a user-visible pause (the
    /// pause button's own displayed state stays whatever the user set it to -- flipping
    /// `paused` here would corrupt it, since restoring it afterwards would have to guess
    /// whether the user had already paused before the export started). Set the instant an
    /// export actually starts and cleared on every exit path -- success, error, and
    /// cancellation -- by the same `on_done` callback that already resets `is_exporting`,
    /// so a failed export can never leave the viewport permanently frozen.
    pub export_active: bool,
    /// The one-shot remote render's total sample budget -- read live by
    /// `gui::remote::start_remote_render` at dispatch time (the exact same treatment
    /// `width`/`height` above already get for a remote render's resolution), not
    /// captured once and cached. `512` by default, matching the value this used to be
    /// hardcoded to (`gui::remote::REMOTE_RENDER_SAMPLES`) before this control existed.
    pub remote_render_samples: u32,
    /// Live rendering's Local/Remote/Local+Remote choice -- see
    /// `settings::model::LiveComputeTarget`'s own doc comment. Read fresh by
    /// `gui::remote::orchestrator::poll_tick` at every settle (so a change takes effect
    /// on the NEXT settle, not the one already in flight) and by `render_thread::mod`'s
    /// loop every frame (to decide whether `remote_active` suspends tracing or lets it
    /// keep contributing -- see `remote_active`'s own doc comment).
    pub live_compute_target: LiveComputeTarget,
    /// Local preview-then-settle rendering -- the user's chosen resolution
    /// reduction while the camera is moving. `Off` (the default) makes
    /// `bridge::local_preview::effective_dimensions` return `width`/`height` above
    /// unchanged regardless of `camera_moving`, i.e. this crate's exact pre-existing
    /// behaviour. See that function's own doc comment for the mechanism.
    pub local_preview_scale: LocalPreviewScale,
    /// Whether the camera is CURRENTLY moving -- mirrors
    /// `bridge::handoff::HandoffState::Previewing`, written once per poll tick by
    /// `gui::remote::poll_tick` from the SAME `HandoffMachine` instance/debounce that
    /// already decides the remote-rendering handoff, so there is exactly one
    /// definition of "the camera has settled" in this app, not a second one invented
    /// for this feature. Mutually exclusive with `remote_active` above, though the
    /// invariant is no longer purely a `HandoffMachine` transition-table fact:
    /// `remote_active` can stay `true` well past `Settling`/`RemoteRendering`, all the
    /// way through `Idle` after a completed remote render (see `remote_active`'s own doc
    /// comment), and `Idle` is also the state a fresh `OrientationChanged` starts FROM
    /// when transitioning INTO `Previewing` (where `camera_moving` goes `true`). What
    /// keeps the two from ever being observed simultaneously `true` in practice is
    /// `render_thread::mod`'s `resolve_remote_ownership` -- that same camera-drag tick is
    /// itself a `ctx.dirty = true` write
    /// (`gui::camera_lighting::setup_camera_and_lighting_callbacks`'s
    /// `on_camera_orbit`/`on_camera_zoom`), so the render loop releases `remote_active`
    /// on essentially the same event that will make `camera_moving` go `true` on the
    /// orchestrator's next poll tick (up to `POLL_INTERVAL_MS` later, but the release
    /// itself is immediate). The render loop below still ANDs this with `!remote_active`
    /// before applying `local_preview_scale` (belt-and-suspenders, not load-bearing) so
    /// the two mechanisms can never both try to drive the trace resolution even in the
    /// narrow window before that poll tick lands.
    pub camera_moving: bool,
    /// HDR environment maps: `Some(map)` replaces the analytic studio rig with
    /// a loaded Radiance `.hdr` panorama as the render loop's `EnvironmentSource` --
    /// see the render loop's own `environment` binding (in `render_thread::mod`), and
    /// `gemray::renderer::env_map`'s module doc comment for where this plugs into
    /// `trace_spectral_ray`. `None` (the default) reproduces this crate's pre-existing
    /// studio-rig-only behaviour exactly, same off-by-default treatment every other
    /// opt-in render override in this struct already gets.
    ///
    /// `Arc`, not a bare `EnvironmentMap`: a decoded panorama can be tens of megabytes
    /// (the pixel buffer plus its importance-sampling `Distribution2D`), and
    /// `snapshot_frame_inputs` below clones every `RenderContext` field out under the
    /// lock every frame -- an `Arc` clone is one atomic increment regardless of map
    /// size, where cloning the map itself would re-copy that whole buffer 60 times a
    /// second. `gui::mod`'s load/clear callbacks are the only writers.
    pub env_map: Option<Arc<EnvironmentMap>>,
}

impl Default for RenderContext {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            yaw: 0.60,   // 35 degrees azimuthal
            pitch: 0.45, // 26 degrees elevation (showing crown, table, and pavilion sparkle in 3D)
            distance: 2.4,
            light_yaw: 0.85,   // ~48 degrees azimuth
            light_pitch: 0.95, // ~54 degrees elevation
            material_name: "Diamond".to_string(),
            lighting_preset: LightingPreset::RingLights,
            target_samples: 256,
            max_bounces: 12,
            exposure: 1.0,
            inclusion_sigma_s: 0.0,
            c_axis_override: None,
            girdle_frosted: false,
            edge_rounding_radius: 0.0,
            stone_width_mm: 0.0,
            active_planes: StandardGemCuts::standard_round_brilliant(),
            custom_materials: Vec::new(),
            running: true,
            dirty: true,
            paused: false,
            tab_visible: true,
            denoise_enabled: true,
            remote_active: false,
            remote_accumulator: None,
            remote_reserved_samples: 0,
            export_active: false,
            remote_render_samples: 512,
            live_compute_target: LiveComputeTarget::Both,
            local_preview_scale: LocalPreviewScale::Off,
            camera_moving: false,
            env_map: None,
        }
    }
}

/// Attempts to decode a Radiance `.hdr` file at `path` into an [`EnvironmentMap`].
/// Never panics: [`EnvironmentMap::from_hdr_file`] already returns a
/// `Result` for a missing, unreadable, or malformed file, so this just wraps the
/// success value in the `Arc` `RenderContext::env_map` stores and turns the error into
/// a plain user-facing message -- `gui::mod`'s load callback shows it via the toast
/// mechanism and, critically, never assigns into `RenderContext::env_map` on `Err`, so
/// a bad file leaves the previously active environment (loaded map or studio rig)
/// untouched.
///
/// # Errors
///
/// Returns `Err` with a human-readable message if `path` cannot be read or does not
/// decode as a valid Radiance HDR image.
pub fn load_env_map(path: &str) -> Result<Arc<EnvironmentMap>, String> {
    EnvironmentMap::from_hdr_file(path)
        .map(Arc::new)
        .map_err(|e: EnvMapError| e.to_string())
}

/// One frame's worth of inputs read out of `RenderContext` under its lock. Plain data,
/// copied/cloned out so the mutex guard can be dropped immediately -- see the comment
/// at the call site in the render loop (`render_thread::mod`'s `spawn_render_thread`).
pub(super) struct FrameInputs {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) yaw: f32,
    pub(super) pitch: f32,
    pub(super) distance: f32,
    pub(super) light_yaw: f32,
    pub(super) light_pitch: f32,
    pub(super) material_name: String,
    pub(super) lighting_preset: LightingPreset,
    pub(super) target_samples: u32,
    pub(super) max_bounces: u32,
    pub(super) exposure: f32,
    pub(super) inclusion_sigma_s: f32,
    pub(super) c_axis_override: Option<Vec3>,
    pub(super) girdle_frosted: bool,
    pub(super) edge_rounding_radius: f32,
    pub(super) stone_width_mm: f32,
    pub(super) active_planes: Vec<GpuFacetPlane>,
    pub(super) custom_materials: Vec<GemMaterial>,
    pub(super) running: bool,
    pub(super) dirty: bool,
    pub(super) paused: bool,
    pub(super) tab_visible: bool,
    pub(super) denoise_enabled: bool,
    pub(super) remote_active: bool,
    pub(super) remote_accumulator: Option<Arc<Mutex<Accumulator>>>,
    pub(super) remote_reserved_samples: u32,
    pub(super) export_active: bool,
    pub(super) live_compute_target: LiveComputeTarget,
    pub(super) local_preview_scale: LocalPreviewScale,
    pub(super) camera_moving: bool,
    pub(super) env_map: Option<Arc<EnvironmentMap>>,
}

/// Snapshots every field the render loop needs for one frame out of `RenderContext`,
/// clearing `dirty` in the same locked section (so a `dirty` set by a callback
/// between the read and the clear is never lost -- this is the whole reason the read
/// and the clear happen together under one lock acquisition rather than two). Split
/// out of `spawn_render_thread` purely to keep that function under clippy's
/// function-length lint; the lock is still acquired and released at exactly the same
/// points as before (guard drops at the end of this function, same as it dropped at
/// the end of the block it was inlined in).
pub(super) fn snapshot_frame_inputs(ctx: &Arc<Mutex<RenderContext>>) -> FrameInputs {
    let mut ctx = ctx.lock().unwrap();
    let dirty = ctx.dirty;
    ctx.dirty = false;
    FrameInputs {
        width: ctx.width,
        height: ctx.height,
        yaw: ctx.yaw,
        pitch: ctx.pitch,
        distance: ctx.distance,
        light_yaw: ctx.light_yaw,
        light_pitch: ctx.light_pitch,
        material_name: ctx.material_name.clone(),
        lighting_preset: ctx.lighting_preset,
        target_samples: ctx.target_samples,
        max_bounces: ctx.max_bounces,
        exposure: ctx.exposure,
        inclusion_sigma_s: ctx.inclusion_sigma_s,
        c_axis_override: ctx.c_axis_override,
        girdle_frosted: ctx.girdle_frosted,
        edge_rounding_radius: ctx.edge_rounding_radius,
        stone_width_mm: ctx.stone_width_mm,
        active_planes: ctx.active_planes.clone(),
        custom_materials: ctx.custom_materials.clone(),
        running: ctx.running,
        dirty,
        paused: ctx.paused,
        tab_visible: ctx.tab_visible,
        denoise_enabled: ctx.denoise_enabled,
        remote_active: ctx.remote_active,
        // `Arc::clone`, not a deep copy -- see `RenderContext::remote_accumulator`'s own
        // doc comment; this is the SAME accumulator instance the remote render's own
        // worker thread is concurrently summing into, never a snapshot of it.
        remote_accumulator: ctx.remote_accumulator.clone(),
        remote_reserved_samples: ctx.remote_reserved_samples,
        export_active: ctx.export_active,
        live_compute_target: ctx.live_compute_target,
        local_preview_scale: ctx.local_preview_scale,
        camera_moving: ctx.camera_moving,
        // `Arc::clone`, not a deep copy of the decoded panorama -- see
        // `RenderContext::env_map`'s own doc comment.
        env_map: ctx.env_map.clone(),
    }
}

/// Resolves the current gem material by name: custom materials take priority over the
/// built-in presets, falling back to `materials[0]` if `material_name` matches
/// neither. Shared by the live render loop (via `resolve_material_and_quality` below)
/// and the high-resolution export worker (`bridge::export_thread::SceneSnapshot::capture`)
/// so both pick a material exactly the same way.
pub fn resolve_material(
    materials: &[GemMaterial],
    custom_materials: &[GemMaterial],
    material_name: &str,
) -> GemMaterial {
    custom_materials
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(material_name))
        .or_else(|| {
            materials
                .iter()
                .find(|m| m.name.eq_ignore_ascii_case(material_name))
        })
        .cloned()
        .unwrap_or_else(|| materials[0].clone())
}

/// Every opt-in render-time material override bundled into one struct -- matching this
/// file's existing `BackendFrame`/`FrameOutputs` precedent for keeping call sites'
/// argument lists short rather than tripping clippy's too-many-arguments lint. Shared
/// by the live render loop (via [`resolve_material_and_quality`]) and
/// `export_thread::SceneSnapshot::capture`, which is the seam that keeps a
/// high-resolution export from silently differing from the viewport it was taken from.
#[derive(Clone, Copy)]
pub struct MaterialOverrides {
    /// Inclusion/subsurface scattering: see `RenderContext::inclusion_sigma_s`.
    pub inclusion_sigma_s: f32,
    /// Crystal-axis orientation: see `RenderContext::c_axis_override`.
    pub c_axis_override: Option<Vec3>,
    /// Facet edge rounding: see `RenderContext::edge_rounding_radius`.
    pub edge_rounding_radius: f32,
    /// Physical stone size: see `RenderContext::stone_width_mm`.
    pub stone_width_mm: f32,
}

/// Applies every [`MaterialOverrides`] field on top of a resolved base material. Each
/// one is opt-in and skips its underlying `GemMaterial::with_*` call (or field write)
/// entirely at its off position, so a material with nothing dialled in renders through
/// the exact bit-identical path it did before these controls existed -- see
/// `GemMaterial::scattering_sigma_s`/`edge_rounding_radius`'s own doc comments in
/// `crates/gemray/src/optics/materials.rs`.
///
/// `active_planes`/`width_cache` are only consulted for `stone_width_mm` (see
/// `RenderContext::stone_width_mm`'s doc comment) -- passed in rather than looked up
/// internally so the live render loop can reuse one persistent `StoneWidthCache`
/// across frames (this function runs once per frame there) while a one-shot caller
/// like `export_thread::SceneSnapshot::capture` can just hand in a fresh one.
#[must_use]
pub fn apply_material_overrides(
    material: GemMaterial,
    overrides: &MaterialOverrides,
    active_planes: &[GpuFacetPlane],
    width_cache: &mut StoneWidthCache,
) -> GemMaterial {
    // Opt-in only -- `with_scattering_amount` is skipped entirely (rather than
    // called with 0.0) whenever the slider is at its off position, so a material with
    // no inclusion haze dialed in renders through the exact same deterministic
    // Beer-Lambert path as before this control existed. See
    // `GemMaterial::scattering_sigma_s`'s own doc comment.
    let material = if overrides.inclusion_sigma_s > 0.0 {
        material.with_scattering_amount(overrides.inclusion_sigma_s)
    } else {
        material
    };

    // An isotropic material's optic axis is physically meaningless -- there is
    // no birefringence to orient (see `GemMaterial::c_axis`'s own doc comment). The
    // settings-dialog control is disabled for one (`gui::mod`'s
    // `c_axis_override_available`), but this guard is what actually stops a leftover
    // override dialed in for a previously selected anisotropic material from reaching
    // an isotropic one's `c_axis`, regardless of what the UI currently shows.
    let mut material = material;
    if let Some(axis) = overrides.c_axis_override
        && material.optical_character != OpticalCharacter::Isotropic
    {
        material.c_axis = axis;
    }

    // Same opt-in-skips-the-call idiom as inclusion above -- see
    // `GemMaterial::edge_rounding_radius`'s own doc comment.
    let material = if overrides.edge_rounding_radius > 0.0 {
        material.with_edge_rounding(overrides.edge_rounding_radius)
    } else {
        material
    };

    // Physical stone size: same opt-in-skips-the-call idiom as inclusion/edge
    // rounding above -- see `RenderContext::stone_width_mm`'s own doc comment. A
    // degenerate/unmeasurable plane arrangement (`width_cache.ensure` returns `None`,
    // or a measured width too small to divide by sanely) or a non-finite/non-positive
    // resulting scale leaves the material untouched, same as the control being off,
    // rather than risking a `NaN`/negative path-length multiplier reaching the tracer.
    if overrides.stone_width_mm > 0.0
        && let Some(model_width) = width_cache.ensure(active_planes)
        && model_width > 1e-9
    {
        let scale = (f64::from(overrides.stone_width_mm) / model_width) as f32;
        if scale.is_finite() && scale > 0.0 {
            return material.with_absorption_path_scale(scale);
        }
    }
    material
}

/// Resolves the current gem material (see `resolve_material`), applies every user
/// material override on top of it (see [`MaterialOverrides`]/[`apply_material_overrides`]),
/// and derives this frame's samples-per-frame from the user's target sample count.
/// Split out of `spawn_render_thread` purely to keep that function under
/// clippy's function-length lint.
///
/// Bounce count is no longer resolved here: it used to be clamped per quality preset
/// (`.min(6)`/`.min(8)`/`.max(16)` against the user's `max_bounces`), but `QualityPreset`
/// was deleted in favor of one knob per concept -- the settings dialog's
/// own "Max Ray Bounces" selector is now the only thing that controls bounce count,
/// so callers use `RenderContext::max_bounces` directly instead of a resolved value
/// from this function.
pub(super) fn resolve_material_and_quality(
    materials: &[GemMaterial],
    custom_materials: &[GemMaterial],
    material_name: &str,
    target_samples: u32,
    overrides: &MaterialOverrides,
    active_planes: &[GpuFacetPlane],
    width_cache: &mut StoneWidthCache,
) -> (GemMaterial, u32) {
    let current_mat = resolve_material(materials, custom_materials, material_name);
    let current_mat = apply_material_overrides(current_mat, overrides, active_planes, width_cache);

    // Samples-per-frame is derived from the target, not chosen directly: the render
    // loop (`render_thread::mod`) sleeps ~16ms per frame regardless of spp, so a large
    // target rendered at a fixed low spp would spend most of its wall-clock time
    // sleeping rather than tracing. Scaling spp with the target keeps that sleep
    // overhead roughly proportional instead of dominating at high targets.
    let spp = (target_samples / 64).clamp(1, 8);

    (current_mat, spp)
}

//! The live-viewport render thread: `RenderContext` (the shared, live-mutated render
//! configuration the GUI writes into), the progressive-accumulation loop that reads a
//! per-frame snapshot from it, and everything that loop dispatches to -- the CPU
//! scanline tracer, the GPU backend wrapper, and the gemological metrics cache.
//!
//! Split into submodules purely to keep this file from growing further: [`context`]
//! (the `RenderContext`/`FrameInputs` state, plus material resolution), [`metrics`]
//! (the gemological-metrics cache), [`scanline`] (the CPU scanline tracer),
//! [`gpu_backend`] (the GPU backend wrapper and frame dispatch), [`denoise`] (denoise +
//! tone-map, the pure functions), and [`display_thread`] (Task R6 -- the dedicated
//! thread that calls them, off the render thread's own trace loop; see that module's
//! own doc comment for the pipelining design). This file keeps the thread loop itself
//! (`spawn_render_thread`) and its two small per-frame helpers.
//!
//! # Local+Remote combined live rendering
//!
//! Once the camera settles, `settings::model::LiveComputeTarget::Both` (the default
//! whenever a worker is configured -- see that type's doc comment) lets this thread's
//! own local tracing keep running ALONGSIDE a dispatched remote render, both
//! contributing to the SAME displayed image, rather than local being suspended for the
//! image's whole lifetime the way [`LiveComputeTarget::RemoteOnly`] still works (and the
//! way every mode used to work before this existed).
//!
//! **The two engines are never made to write one buffer.** `accum_buffer` below stays
//! exactly what it always has been: a plain `Vec<Vec3>`, owned outright by this thread,
//! touched under no lock at all -- the hot per-sample trace path is completely
//! unaffected by whether a remote render happens to be in flight. `RenderContext::
//! remote_accumulator` is the SAME `gemray_net::client::Accumulator` the remote render's
//! own socket thread (`bridge::remote_render::run`) is concurrently summing `FRAME`
//! deltas into; this thread only ever reads its `buffer()`/`samples_done()` (never
//! `last_preview` -- a `PREVIEW` must never reach a full-resolution accumulator, see
//! `gemray_net::messages::stream`'s module docs), and only at the display cadence
//! ([`DENOISE_MIN_INTERVAL`], not per traced sample -- see [`should_combine_remote`]'s
//! call site): a short lock, a full-buffer read, an elementwise add into a scratch
//! buffer for THAT display cycle alone. Nothing from either engine is ever discarded or
//! overwritten by the other; both are read-only sums.
//!
//! **Disjoint sample ranges.** A live remote render is always dispatched as ONE request
//! covering exactly `[0, remote_render_samples)` (see `gui::remote::orchestrator::
//! start_remote_render`) -- fixed at dispatch time, unlike the export's calibrated
//! split, since a live remote render has no separate local total to calibrate against.
//! `RenderContext::remote_reserved_samples` records that reserved size (`0` when not
//! combining), and every frame this loop traces shifts its own absolute sample index
//! (the seed for each sample's jitter/RNG -- see `gemray::renderer::gpu_backend`'s
//! "Sample-range additivity" doc section) past it, so local's used indices always start
//! exactly where remote's assigned range ends and the two can never redraw the same
//! stratified sample.
//!
//! **What happened to the old suspend-while-remote mechanism.** `RenderContext::
//! remote_active`/`resolve_remote_ownership` are KEPT, not removed, and their pure
//! contract is UNCHANGED -- but what a resolved `remote_active == true` DOES now depends
//! on `live_compute_target`: for `RemoteOnly` it still suspends tracing outright, exactly
//! as before this feature existed (the bug `resolve_remote_ownership` was built to fix --
//! local racing back in and overwriting a finished remote image with a from-scratch
//! local one -- is still a live risk there, since `RemoteOnly`'s local buffer would
//! otherwise start fresh while the display kept showing remote's image); for `Both`, that
//! same bug is structurally impossible (local only ever ADDS into a display-time sum, it
//! never owns or overwrites the displayed buffer), so `remote_active` instead gates
//! [`should_combine_remote`] -- whether this thread's display cycle still folds
//! `remote_accumulator` in at all. A non-drag scene invalidation (material, lighting,
//! ...) still needs to stop combining with what is now a stale-scene remote contribution,
//! which is exactly what `resolve_remote_ownership`'s existing release condition already
//! detects -- one mechanism, two effects, selected by the mode.

mod context;
mod denoise;
mod display_thread;
mod gpu_backend;
mod metrics;
mod scanline;

pub use context::{
    MaterialOverrides, RenderContext, apply_material_overrides, load_env_map, resolve_material,
};
pub use denoise::{
    DenoiseScratch, FirstHitSnapshot, denoise_and_tonemap_frame, tonemap_running_average,
};
pub use metrics::hash_planes;

use crate::{
    bridge::{girdle_finish::GirdleFinishCache, local_preview, stone_width::StoneWidthCache},
    settings::model::LiveComputeTarget,
};
use context::{FrameInputs, resolve_material_and_quality, snapshot_frame_inputs};
use display_thread::{FrameMetricsSnapshot, spawn_display_thread};
use gemray::optics::{
    materials::GemMaterial,
    raytracer::{Camera, EnvironmentSource, FacetFinish},
};
use glam::Vec3;
use gpu_backend::{
    BackendFrame, FrameOutputs, HybridPacing, ViewportGpu, accumulate_frame_samples,
};
use metrics::{MetricsCache, compute_or_reuse_metrics};
use slint::{ComponentHandle, Weak};
use std::{
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

/// Minimum real-time gap between two display cycles (snapshot + hand-off to the
/// dedicated display thread -- see [`display_thread`]) being SENT while samples are
/// still accumulating (`accum_samples < target_samples`). See the loop body's
/// `denoise_due` computation for the exceptions that always send regardless of this gap
/// (first frame, `dirty`, a dimension change, denoising disabled, and the frame that
/// reaches `target_samples`). Task: R1/R2 -- the À-Trous denoiser dominates every live
/// frame (267ms/1072ms per call at 800x600/1920x1080 on the reference profiling
/// machine, even after the kernel-cost work in `renderer::denoise`), so re-running it on
/// literally every ~16ms loop iteration while still converging capped the viewport at a
/// few FPS for no visible benefit -- the filtered image a human can actually perceive
/// changing does not change every 16ms. 120ms sits in the middle of the requested
/// 100-150ms range: fast enough that the viewport still feels live while converging,
/// slow enough to cut denoiser invocations by roughly 7-8x during that phase.
///
/// Task R6 moved the actual denoise+tonemap+push work onto [`display_thread`], off this
/// gap's caller -- so unlike before, this alone is now the whole cadence story: the
/// render thread never blocks on a cycle's cost, so there is nothing left to scale a
/// duty-cycle gap against (the removed `DENOISE_DUTY_DIVISOR`/`denoise_gap` used to
/// widen the gap by a multiple of the LAST cycle's measured wall time, precisely because
/// that cost used to be paid synchronously, on this thread).
const DENOISE_MIN_INTERVAL: Duration = Duration::from_millis(120);

/// Pushes one finished frame's image and gemological metrics to the UI thread's
/// event loop. Split out of `spawn_render_thread` purely to keep that function under
/// clippy's function-length lint.
fn push_frame_to_ui<T, F, M>(
    ui_weak: &Weak<T>,
    update_image: &F,
    update_metrics: &M,
    image: slint::SharedPixelBuffer<slint::Rgba8Pixel>,
    metrics_snapshot: FrameMetricsSnapshot,
) where
    T: ComponentHandle + 'static,
    F: Fn(&T, slint::SharedPixelBuffer<slint::Rgba8Pixel>) + Send + 'static + Clone,
    M: Fn(&T, f32, f32, f32, f32, f32, [f32; 19], [f32; 19], [f32; 19], f32)
        + Send
        + 'static
        + Clone,
{
    let _ = ui_weak.upgrade_in_event_loop({
        let update_image = update_image.clone();
        let update_metrics = update_metrics.clone();
        move |ui| {
            update_image(&ui, image);
            let metrics = metrics_snapshot.metrics;
            update_metrics(
                &ui,
                metrics.brilliance_pct,
                metrics.fire_index,
                metrics.scintillation_pct,
                metrics.windowing_pct,
                metrics.extinction_pct,
                metrics_snapshot.graph_brilliance,
                metrics_snapshot.graph_extinction,
                metrics_snapshot.graph_windowing,
                metrics_snapshot.cam_pitch_deg,
            );
        }
    });
}

/// Resets the progressive-accumulation state (buffer, sample count, and the three
/// Task-2 first-hit guide buffers) whenever the output dimensions change, and
/// separately whenever the frame is marked `dirty` (camera/material/etc. moved). Split
/// out of `spawn_render_thread` purely to keep that function under clippy's
/// function-length lint; the two reset conditions and their effects are unchanged from
/// when this was inlined -- only the three added guide buffers are new. They only need
/// resizing (not re-zeroing) on `dirty`: `render_frame_scanlines` unconditionally
/// overwrites every pixel's guide values every call regardless of `dirty`, so there is
/// no stale value for a `dirty` reset to clear.
///
/// Task R6 moved the framebuffer transfer this used to also reallocate here onto
/// [`display_thread`], which reallocates its own copy on the identical
/// `width`/`height`-changed condition (see that module's `spawn_display_thread`) --
/// this function no longer touches it at all.
///
/// This is also what makes local preview-then-settle rendering (see
/// `bridge::local_preview::effective_dimensions`) work for free: its caller in
/// `spawn_render_thread`'s loop feeds `width`/`height` shadowed to a (possibly reduced)
/// EFFECTIVE resolution rather than the raw configured one, so every preview<->full
/// transition is just another `width != *last_width || height != *last_height` this
/// function already handles -- no separate reset path was added for it.
/// The render loop's own accumulation state, owned by [`spawn_render_thread`]'s loop and
/// kept alive across frames (never reallocated except on a reset -- see
/// [`update_accumulation_state`]) so steady-state rendering performs no extra per-frame
/// heap allocation. Bundled here purely to keep [`update_accumulation_state`]'s
/// parameter list under clippy's argument-count lint.
struct AccumulationBuffers<'a> {
    accum: &'a mut Vec<Vec3>,
    first_hit_depth: &'a mut Vec<f32>,
    first_hit_normal: &'a mut Vec<Vec3>,
    first_hit_facet_id: &'a mut Vec<i32>,
}

fn update_accumulation_state(
    width: u32,
    height: u32,
    dirty: bool,
    buffers: &mut AccumulationBuffers<'_>,
    accum_samples: &mut u32,
    last_width: &mut u32,
    last_height: &mut u32,
) {
    // Dimension change or camera movement resets progressive accumulation
    if width != *last_width || height != *last_height {
        let pixel_count = (width * height) as usize;
        *buffers.accum = vec![Vec3::ZERO; pixel_count];
        *buffers.first_hit_depth = vec![1.0e6; pixel_count];
        *buffers.first_hit_normal = vec![Vec3::ZERO; pixel_count];
        *buffers.first_hit_facet_id = vec![-1; pixel_count];
        *accum_samples = 0;
        *last_width = width;
        *last_height = height;
    }

    if dirty {
        buffers.accum.fill(Vec3::ZERO);
        *accum_samples = 0;
    }
}

/// Whether the render loop should skip tracing this iteration: an explicit user pause,
/// the 3D tab not being visible, a remote worker SOLELY owning the displayed image
/// (`remote_suspends` -- see its own doc comment for why this is no longer simply
/// `RenderContext::remote_active` itself), or a high-resolution export currently running
/// -- see `RenderContext::paused`/`tab_visible`/`export_active`'s own doc comments for
/// the rest. Split out purely so this decision is unit-testable on its own (see this
/// module's tests) without spinning up the whole render thread.
///
/// Taken as a named struct rather than four positional `bool`s: transposing any two of
/// them at a call site would compile and silently suspend (or fail to suspend) for the
/// wrong reason, which is exactly the mistake `clippy::fn_params_excessive_bools`
/// exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SuspensionFlags {
    pub(super) paused: bool,
    pub(super) tab_visible: bool,
    /// `RenderContext::remote_active`'s resolved value, ALREADY narrowed by the caller
    /// to only the case where it should actually suspend tracing -- see
    /// [`remote_suspends_local`]'s doc comment for exactly which case that is. Named
    /// differently from the `RenderContext` field it's derived from on purpose: this is
    /// no longer "is remote active at all", it's "does that activity mean local must
    /// stay off".
    pub(super) remote_suspends: bool,
    pub(super) export_active: bool,
}

impl SuspensionFlags {
    const fn tracing_suspended(self) -> bool {
        self.paused || !self.tab_visible || self.remote_suspends || self.export_active
    }
}

/// Whether a resolved `remote_active == true` should suspend LOCAL tracing this frame --
/// only ever true for [`LiveComputeTarget::RemoteOnly`], the one mode where local
/// contributing anything would just be immediately-discarded work (there is nowhere for
/// it to go: `should_combine_remote` never applies outside `Both`, so nothing would ever
/// read what local traced). For [`LiveComputeTarget::Both`], `remote_active` instead
/// gates combining (see [`should_combine_remote`]) and local keeps tracing regardless;
/// for [`LiveComputeTarget::LocalOnly`], `remote_active` never becomes `true` in the
/// first place (`gui::remote::orchestrator::poll_tick` never dispatches a remote render
/// for that mode), so this is unreachable in practice but still total rather than
/// `unreachable!()`, matching this crate's existing precedent for pure decision
/// functions covering every input rather than trusting a caller invariant (see
/// `bridge::export_thread::remote::split_remote_samples`'s own doc comment for the same
/// reasoning). Pure and directly unit-testable -- see this module's tests.
#[must_use]
const fn remote_suspends_local(
    remote_active: bool,
    live_compute_target: LiveComputeTarget,
) -> bool {
    remote_active && matches!(live_compute_target, LiveComputeTarget::RemoteOnly)
}

/// Whether the render loop's display cycle should fold `RenderContext::
/// remote_accumulator`'s current running total into the image it shows this cycle --
/// true only for [`LiveComputeTarget::Both`] while a resolved `remote_active` says a
/// remote render still owns part of the current image (see `RenderContext::
/// remote_active`'s own doc comment on what "still" means: this stays `true` well past a
/// successful completion, so local's own continued tracing keeps adding to remote's
/// finished contribution rather than the display ever losing it). `RemoteOnly` never
/// reaches the display cycle this gates AT ALL while `remote_active` is `true` --
/// [`remote_suspends_local`] suspends tracing (and, with it, this thread's whole display
/// step) outright in that mode -- so this function does not need to special-case it; it
/// simply always answers `false` for `RemoteOnly`/`LocalOnly`; the doc-comment
/// distinction is about which callers can ever actually observe a `true` for
/// `RemoteOnly`, not about this function's own logic. Pure and directly unit-testable --
/// see this module's tests.
#[must_use]
const fn should_combine_remote(
    remote_active: bool,
    live_compute_target: LiveComputeTarget,
) -> bool {
    remote_active && matches!(live_compute_target, LiveComputeTarget::Both)
}

/// The absolute sample index local tracing's `sample_offset` should use for a frame that
/// has already traced `local_pre_frame_count` samples this epoch -- see the module doc
/// comment's disjointness argument. `remote_reserved_samples` (`RenderContext::
/// remote_reserved_samples`) is `0` whenever nothing is reserved (not combining, or no
/// remote dispatch this epoch), which makes this an exact identity on
/// `local_pre_frame_count` -- the same `current_sample_count - spp` expression this
/// crate always used before combining existed. Pure and directly unit-testable -- see
/// this module's tests.
#[must_use]
const fn combined_sample_offset(remote_reserved_samples: u32, local_pre_frame_count: u32) -> u32 {
    remote_reserved_samples + local_pre_frame_count
}

/// The single choke point that releases stale remote-image ownership -- see
/// `RenderContext::remote_active`'s doc comment for why `remote_active` must stay set
/// all the way through a *completed* remote render (not just `Settling`/
/// `RemoteRendering`) and this function's own doc comment on the call site (below) for
/// why it can only be released here, not at each of the ~25 individual `ctx.dirty =
/// true` call sites scattered across `gui::*` (material, lighting, quality, resolution,
/// c-axis, inclusion, girdle, edge rounding, stone width, HDR env map, lighting
/// presets, a different design loading, camera orbit/zoom/light move -- literally
/// anything that can invalidate the image).
///
/// `remote_active` can only ever transition `false -> true` one way: the handoff
/// orchestrator's `DiscardLocalPreview` action (`gui::remote::orchestrator::apply_actions`)
/// sets it `true` in the SAME locked mutation as `dirty = true`, entering `Settling` --
/// and the mutual-exclusion invariant `RenderContext::camera_moving` documents means
/// `remote_active` is always `false` immediately before that write. So the FIRST frame
/// this loop ever observes `remote_active == true` after a frame where it was `false`,
/// a fresh `dirty` alongside it is that legitimate hand-off starting, not an
/// invalidation -- `was_remote_active_last_frame` is `false`, and this returns
/// `remote_active` unchanged (`true`).
///
/// Any OTHER frame where `dirty` is freshly `true` while `remote_active` was ALREADY
/// `true` one frame ago can only mean something besides that one `DiscardLocalPreview`
/// write touched the scene -- a `gui::*` callback setting `ctx.dirty = true` for a
/// reason having nothing to do with the handoff machine (this covers a completed remote
/// render just as well as one still in flight: after `RemoteUpdate::Done`,
/// `remote_active` is deliberately left `true` -- see that field's doc comment -- so
/// `was_remote_active_last_frame` is still `true` the next time ANY dirty write lands,
/// including the very next camera-drag tick). That releases ownership (`false`)
/// regardless of which callback caused it, which is exactly what makes it impossible
/// for a future callback to reintroduce the freeze by adding a new `ctx.dirty = true`
/// site: this function never enumerates call sites, it only ever looks at the two
/// fields every such site already has to touch (`dirty`) or leave alone (`remote_active`).
///
/// Pure and directly unit-testable -- see this module's tests.
const fn resolve_remote_ownership(
    dirty: bool,
    remote_active: bool,
    was_remote_active_last_frame: bool,
) -> bool {
    if dirty && remote_active && was_remote_active_last_frame {
        false
    } else {
        remote_active
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the render worker thread's full loop (resize/dirty handling, sampling, \
              progressive accumulation, UI callbacks); splitting it apart risks changing \
              GUI render-thread behaviour that can only be verified by launching the app, \
              which is out of scope for this pass"
)]
pub fn spawn_render_thread<T, F, M>(
    ui_weak: Weak<T>,
    ctx: Arc<Mutex<RenderContext>>,
    update_image: F,
    update_metrics: M,
) where
    T: ComponentHandle + 'static,
    F: Fn(&T, slint::SharedPixelBuffer<slint::Rgba8Pixel>) + Send + 'static + Clone,
    M: Fn(&T, f32, f32, f32, f32, f32, [f32; 19], [f32; 19], [f32; 19], f32)
        + Send
        + 'static
        + Clone,
{
    thread::spawn(move || {
        let mut last_width = 0;
        let mut last_height = 0;

        let materials = GemMaterial::all_materials();
        let mut accum_buffer: Vec<Vec3> = Vec::new();
        // First-hit guide buffers alongside the accumulation buffer -- see
        // `render_frame_scanlines` for how they're populated and
        // `display_thread::spawn_display_thread` for how they're consumed. Kept alive
        // across frames (rather than allocated per-frame) so steady-state rendering
        // performs no extra per-frame heap allocation on this side of the hand-off --
        // see `display_thread::DisplayWork::fill`'s own doc comment for the matching
        // reuse on the display thread's side of it.
        let mut first_hit_depth: Vec<f32> = Vec::new();
        let mut first_hit_normal: Vec<Vec3> = Vec::new();
        let mut first_hit_facet_id: Vec<i32> = Vec::new();
        // Task: Local+Remote combined live rendering -- scratch buffer for folding a
        // remote accumulator's current total into a display cycle's input; see
        // `should_combine_remote`'s call site near the bottom of this loop. Reused
        // across display cycles (never per-trace-frame) the same "allocate once, keep
        // alive" way `accum_buffer`/the guide buffers above already are; stays empty
        // (and unused) for the whole session whenever nothing ever combines.
        let mut combined_scratch: Vec<Vec3> = Vec::new();
        let mut accum_samples: u32 = 0;
        // Cadence gate for handing a display cycle off to `display` (Task R6, see
        // `display_thread`) -- see `DENOISE_MIN_INTERVAL`'s doc comment. `None` means
        // "no cycle has been sent yet since the last reset", which forces one
        // immediately (first frame, and any time `dirty`/a dimension change
        // invalidates the previous one).
        let mut last_denoise_at: Option<Instant> = None;
        let mut metrics_cache: Option<MetricsCache> = None;
        // Acquired once, off the frame loop: adapter acquisition and shader compilation
        // both take long enough to be worth doing exactly once. Declines every frame on
        // a machine with no usable GPU, which is not an error -- see `ViewportGpu`.
        let mut gpu_backend = ViewportGpu::acquire();
        // Hybrid CPU+GPU pacing state -- see `HybridPacing`.
        let mut hybrid_pacing = HybridPacing::new();
        // Frosted girdle: recomputed only when the active design's geometry
        // actually changes -- see `GirdleFinishCache`'s doc comment.
        let mut girdle_cache = GirdleFinishCache::new();
        // Physical stone size: the design's own model-unit girdle width, remeasured
        // only when its geometry changes -- see `StoneWidthCache`'s doc comment.
        let mut stone_width_cache = StoneWidthCache::new();
        // Task R6: denoise+tonemap+push runs on this dedicated thread instead of
        // blocking the trace loop below -- see `display_thread`'s own doc comment for
        // the full pipelining design. `ui_weak`/`update_image`/`update_metrics` move
        // here rather than staying available to this loop: every push to the UI now
        // happens exclusively on the display thread's side of the hand-off.
        let display = spawn_display_thread(ui_weak, update_image, update_metrics);
        // Task 2: single choke point for releasing stale remote-image ownership -- see
        // `resolve_remote_ownership`'s own doc comment. `false` matches
        // `RenderContext::remote_active`'s own `Default` value, so the very first
        // iteration never misfires as a release.
        let mut prev_remote_active = false;

        loop {
            let FrameInputs {
                width,
                height,
                yaw,
                pitch,
                distance,
                light_yaw,
                light_pitch,
                material_name,
                lighting_preset,
                target_samples,
                max_bounces,
                exposure,
                inclusion_sigma_s,
                c_axis_override,
                girdle_frosted,
                edge_rounding_radius,
                stone_width_mm,
                active_planes,
                custom_materials,
                running,
                dirty,
                paused,
                tab_visible,
                denoise_enabled,
                remote_active: remote_active_snapshot,
                remote_accumulator,
                remote_reserved_samples,
                export_active,
                live_compute_target,
                local_preview_scale,
                camera_moving,
                env_map,
            } = snapshot_frame_inputs(&ctx);

            if !running {
                break;
            }

            // Task 2: release stale remote-image ownership the instant a fresh `dirty`
            // arrives for a reason other than the handoff's own hand-off-starting write
            // -- see `resolve_remote_ownership`'s doc comment for the full reasoning.
            // Shadows `remote_active` with the resolved value for the rest of this
            // frame so a release also resumes tracing THIS iteration, not one 100ms
            // suspended-sleep later.
            let remote_active =
                resolve_remote_ownership(dirty, remote_active_snapshot, prev_remote_active);
            if remote_active != remote_active_snapshot {
                // Write back so every other reader of `RenderContext::remote_active`
                // (`gui::remote::orchestrator::poll_tick`'s own served_by reconciliation,
                // in particular) observes the release promptly rather than waiting for
                // the orchestrator's next unrelated write to it.
                ctx.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remote_active = false;
            }
            prev_remote_active = remote_active;

            // Task: Local+Remote combined live rendering -- read the remote
            // accumulator's current SAMPLE COUNT once per iteration (a cheap lock +
            // `u32` read, never the full buffer -- see `should_combine_remote`'s call
            // site further down for where the full buffer is actually read, gated to
            // the much lower display cadence). Used below both to decide whether this
            // iteration can skip tracing (the combined total, not just local's own, may
            // already have reached `target_samples`) and, later, as part of the
            // combined image's true sample count.
            let remote_samples_done_now: u32 =
                if should_combine_remote(remote_active, live_compute_target) {
                    remote_accumulator.as_ref().map_or(0, |acc| {
                        acc.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .samples_done()
                    })
                } else {
                    0
                };

            if width == 0 || height == 0 {
                thread::sleep(std::time::Duration::from_millis(16));
                continue;
            }

            // Task: local preview-then-settle rendering -- shadows `width`/`height`
            // with the (possibly reduced) dimensions to actually trace at THIS frame,
            // for the rest of the loop body; `ctx.width`/`ctx.height` themselves (the
            // user's CONFIGURED resolution, read again above via `snapshot_frame_inputs`)
            // are never mutated by this -- `gui::mod::on_resolution_changed`/export/
            // remote-render dispatch all keep reading the true configured value from
            // `RenderContext` directly, unaffected by whatever's currently on screen.
            // `&& !remote_active` is belt-and-suspenders, not load-bearing -- see
            // `RenderContext::camera_moving`'s doc comment for why the two are already
            // structurally mutually exclusive.
            let (width, height) = local_preview::effective_dimensions(
                width,
                height,
                local_preview_scale,
                camera_moving && !remote_active,
            );

            // Captured before `update_accumulation_state` overwrites `last_width`/
            // `last_height` below -- used, alongside `dirty`, to invalidate the denoise
            // cadence cache immediately on any accumulation reset rather than letting a
            // stale filtered frame from the old dimensions/camera pose sit on screen
            // for up to `DENOISE_MIN_INTERVAL` longer.
            let accumulation_reset = dirty || width != last_width || height != last_height;

            update_accumulation_state(
                width,
                height,
                dirty,
                &mut AccumulationBuffers {
                    accum: &mut accum_buffer,
                    first_hit_depth: &mut first_hit_depth,
                    first_hit_normal: &mut first_hit_normal,
                    first_hit_facet_id: &mut first_hit_facet_id,
                },
                &mut accum_samples,
                &mut last_width,
                &mut last_height,
            );

            if accumulation_reset {
                last_denoise_at = None;
                // Task R6: invalidate any display cycle already in flight (or queued)
                // from before this reset -- see `DisplayHandle::bump_generation`'s doc
                // comment for why this is what keeps a stale frame from the OLD pose
                // from ever reaching the screen after a `dirty`/dimension-change reset.
                display.bump_generation();
            }

            // Rendering is suspended by an explicit user pause, because the 3D tab isn't
            // currently visible, because a remote worker currently owns the displayed
            // image (`remote_active` -- Task: remote rendering's preview-then-handoff;
            // see `RenderContext::remote_active`'s own doc comment), or because a
            // high-resolution export is currently running (`export_active` -- see
            // `RenderContext::export_active`'s own doc comment). Every condition skips
            // the raytracing and the (cached) metrics evaluation entirely -- the
            // accumulation buffer above has already been kept in sync with `dirty`, so
            // resuming continues converging rather than starting over. The sleep is long
            // enough that a suspended render costs essentially no CPU, but short enough
            // (~100ms) that resuming feels responsive.
            let suspension = SuspensionFlags {
                paused,
                tab_visible,
                remote_suspends: remote_suspends_local(remote_active, live_compute_target),
                export_active,
            };
            if suspension.tracing_suspended() {
                thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }

            let (current_mat, spp) = resolve_material_and_quality(
                &materials,
                &custom_materials,
                &material_name,
                target_samples,
                &MaterialOverrides {
                    inclusion_sigma_s,
                    c_axis_override,
                    edge_rounding_radius,
                    stone_width_mm,
                },
                &active_planes,
                &mut stone_width_cache,
            );

            // If we have accumulated enough samples in still mode, sleep a bit to
            // conserve power. Combined against remote's contribution too (`+
            // remote_samples_done_now`, `0` whenever not combining) -- once local's own
            // samples plus whatever remote has already delivered reach the user's
            // target, further local tracing is wasted work; see this module's own doc
            // comment.
            if accum_samples + remote_samples_done_now >= target_samples && !dirty {
                thread::sleep(std::time::Duration::from_millis(60));
                continue;
            }

            accum_samples += spp;

            // Optical Metrics calculation: True 3D Analytical Raytracing from Camera PoV accounting for light source direction.
            // This is expensive (single-threaded analytical raytracing) and its result depends only on
            // (active_planes, current_mat, yaw, pitch, light_yaw, light_pitch) -- none of which change between
            // progressive accumulation samples. Recompute only when those inputs actually changed, instead of
            // on every one of the ~64 iterations it takes to reach the sample cap.
            let (metrics, graph_brilliance, graph_extinction, graph_windowing) =
                compute_or_reuse_metrics(
                    &mut metrics_cache,
                    &active_planes,
                    &current_mat,
                    yaw,
                    pitch,
                    light_yaw,
                    light_pitch,
                );
            let cam_pitch_deg = pitch.to_degrees();

            let camera = Camera::new(yaw, pitch, distance, 42.0);

            // Parallel Scanline Raytracing across CPU threads
            let current_sample_count = accum_samples;

            // A loaded HDR panorama (`gui::mod`'s load/clear callbacks, via
            // `RenderContext::env_map`) replaces the analytic studio rig as this
            // frame's environment source. `Option::as_deref` turns the `Arc` snapshot
            // above into the `&EnvironmentMap` `EnvironmentSource::HdrMap` borrows --
            // see `gemray::renderer::gpu_backend`'s module doc comment for what happens next: the
            // GPU megakernel has no `env_mode` for `HdrMap` and declines every such
            // frame (`GpuFrameError::UnsupportedEnvironment`), so `accumulate_frame_samples`
            // below transparently falls through to `render_frame_scanlines`, the CPU
            // path, for as long as a map stays loaded.
            let environment = env_map.as_deref().map_or_else(
                || lighting_preset.studio(exposure, light_yaw, light_pitch),
                EnvironmentSource::HdrMap,
            );

            // Frosted girdle: `&[]` at the off position reproduces the exact
            // pre-existing all-polished behaviour -- see `BackendFrame::facet_finishes`'s
            // doc comment.
            let facet_finishes: &[FacetFinish] = if girdle_frosted {
                girdle_cache.ensure(&active_planes)
            } else {
                &[]
            };

            accumulate_frame_samples(
                &mut gpu_backend,
                &BackendFrame {
                    width,
                    height,
                    yaw,
                    pitch,
                    distance,
                    camera: &camera,
                    planes: &active_planes,
                    facet_finishes,
                    material: &current_mat,
                    max_bounces,
                    environment,
                    spp,
                    // Task: Local+Remote combined live rendering -- shifted past
                    // whatever `remote_reserved_samples` reserves for a dispatched
                    // remote render's own `[0, remote_reserved_samples)` range (`0`
                    // whenever not combining, reproducing this expression's
                    // pre-existing value exactly). This is the disjointness guarantee:
                    // local's absolute sample index -- the seed for every sample's
                    // jitter/RNG on both backends -- never falls inside the range
                    // remote was assigned, so the two engines never redraw the same
                    // stratified sample. See this module's own doc comment.
                    sample_offset: combined_sample_offset(
                        remote_reserved_samples,
                        current_sample_count - spp,
                    ),
                },
                &mut FrameOutputs {
                    accum: &mut accum_buffer,
                    depth: &mut first_hit_depth,
                    normal: &mut first_hit_normal,
                    facet_id: &mut first_hit_facet_id,
                },
                &mut hybrid_pacing,
            );

            // Denoise on the READBACK path only -- `accum_buffer` above is left
            // as the raw running sum, never overwritten with filtered output, so the
            // progressive estimator itself stays unbiased; only the tone-mapped image
            // actually displayed is filtered. See `denoise_and_tonemap_frame`'s doc
            // comment.
            //
            // `denoise_enabled` (Task: remote rendering's global denoise toggle) gates
            // whether that filtering happens at all -- when off, this is a straight
            // tone-map of the raw running average, cheap enough (and already what a
            // user who disabled denoising expects to see) that it always runs every
            // frame, with no cadence gating below.
            //
            // Task: R1/R2 -- while accumulating (`current_sample_count <
            // target_samples`), the display cycle is rate-limited to at most once per
            // `DENOISE_MIN_INTERVAL` (see its doc comment): every loop iteration still
            // traces new samples into `accum_buffer` above regardless, so convergence
            // itself is unaffected, but the UI simply keeps showing the last frame
            // actually pushed on the iterations in between (nothing is sent at all,
            // rather than re-pushing identical bytes or falling back to an undenoised
            // `tonemap_running_average`, which would flicker between filtered and
            // unfiltered). `converged_now` -- this is the iteration whose
            // `accum_samples` first reaches `target_samples` -- and `accumulation_reset`
            // -- `dirty`/a dimension change just invalidated whatever was on screen --
            // both always attempt a cycle regardless of the gap: the loop's earlier
            // `accum_samples >= target_samples && !dirty` early-continue means this is
            // the LAST iteration that will send anything at all before the render
            // settles, so it must be the fully denoised, fully up-to-date frame, not a
            // stale cached one.
            let converged_now = accum_samples + remote_samples_done_now >= target_samples;
            let denoise_due = !denoise_enabled
                || accumulation_reset
                || converged_now
                || last_denoise_at.is_none_or(|t| t.elapsed() >= DENOISE_MIN_INTERVAL);

            if denoise_due {
                // Task R6: the render thread never blocks on a display cycle's cost --
                // it hands the frame off to `display` (see `display_thread`) and keeps
                // tracing. At most one cycle may be in flight at a time, so a cycle is
                // simply skipped (not queued) while the previous one is still running;
                // the tracer keeps accumulating regardless, and the next iteration's
                // `denoise_due` retries. The one exception is `converged_now`: the
                // frame that reaches `target_samples` must always be displayed exactly
                // once, so THAT send waits out any in-flight cycle instead of skipping.
                let can_send = if converged_now {
                    while display.busy() {
                        thread::sleep(display_thread::CONVERGENCE_WAIT_POLL);
                    }
                    true
                } else {
                    !display.busy()
                };

                if can_send {
                    // Task: Local+Remote combined live rendering -- fold the remote
                    // accumulator's CURRENT full running total into a scratch buffer for
                    // THIS display cycle only. Never merged permanently into
                    // `accum_buffer` (which stays exclusively local's own for the whole
                    // session, touched under no lock -- see this module's doc comment):
                    // this read-lock-and-add happens here, at the display cadence, not
                    // on the hot per-sample trace path above. Skipped entirely (falls
                    // through to `&accum_buffer` with no extra allocation or copy)
                    // whenever `should_combine_remote` says no -- the pre-existing
                    // local-only and suspended-RemoteOnly paths are unaffected.
                    let (display_accum, display_sample_count): (&[Vec3], u32) =
                        if should_combine_remote(remote_active, live_compute_target)
                            && let Some(remote_acc) = &remote_accumulator
                        {
                            let acc = remote_acc
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            combined_scratch.clear();
                            combined_scratch.extend(
                                accum_buffer.iter().zip(acc.buffer()).map(|(a, b)| *a + *b),
                            );
                            (&combined_scratch, current_sample_count + acc.samples_done())
                        } else {
                            (&accum_buffer, current_sample_count)
                        };

                    let mut work = display.reclaim();
                    work.fill(
                        display.current_generation(),
                        denoise_enabled,
                        FirstHitSnapshot {
                            width,
                            height,
                            current_sample_count: display_sample_count,
                            accum_buffer: display_accum,
                            first_hit_depth: &first_hit_depth,
                            first_hit_normal: &first_hit_normal,
                            first_hit_facet_id: &first_hit_facet_id,
                        },
                        FrameMetricsSnapshot {
                            metrics,
                            graph_brilliance,
                            graph_extinction,
                            graph_windowing,
                            cam_pitch_deg,
                        },
                    );
                    display.send(work);
                    last_denoise_at = Some(Instant::now());
                }
            }

            // Target smooth interactive framerate ~30-60 FPS
            thread::sleep(std::time::Duration::from_millis(16));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- tracing_suspended: the four independent suspend flags ----------------------

    #[test]
    fn tracing_runs_only_when_all_four_flags_allow_it() {
        assert!(
            !SuspensionFlags {
                paused: false,
                tab_visible: true,
                remote_suspends: false,
                export_active: false
            }
            .tracing_suspended()
        );
    }

    #[test]
    fn each_flag_alone_suspends_tracing() {
        assert!(
            SuspensionFlags {
                paused: true,
                tab_visible: true,
                remote_suspends: false,
                export_active: false
            }
            .tracing_suspended(),
            "paused"
        );
        assert!(
            SuspensionFlags {
                paused: false,
                tab_visible: false,
                remote_suspends: false,
                export_active: false
            }
            .tracing_suspended(),
            "!tab_visible"
        );
        assert!(
            SuspensionFlags {
                paused: false,
                tab_visible: true,
                remote_suspends: true,
                export_active: false
            }
            .tracing_suspended(),
            "remote_suspends"
        );
        assert!(
            SuspensionFlags {
                paused: false,
                tab_visible: true,
                remote_suspends: false,
                export_active: true
            }
            .tracing_suspended(),
            "export_active"
        );
    }

    #[test]
    fn every_flag_combination_still_suspends_if_any_one_is_set() {
        for paused in [false, true] {
            for tab_visible in [false, true] {
                for remote_suspends in [false, true] {
                    for export_active in [false, true] {
                        let expected = paused || !tab_visible || remote_suspends || export_active;
                        assert_eq!(
                            SuspensionFlags {
                                paused,
                                tab_visible,
                                remote_suspends,
                                export_active,
                            }
                            .tracing_suspended(),
                            expected,
                            "paused={paused} tab_visible={tab_visible} \
                             remote_suspends={remote_suspends} export_active={export_active}"
                        );
                    }
                }
            }
        }
    }

    // ---- remote_suspends_local / should_combine_remote: the mode-dependent split ----

    #[test]
    fn remote_only_is_the_sole_mode_where_remote_activity_suspends_local_tracing() {
        for mode in [
            LiveComputeTarget::LocalOnly,
            LiveComputeTarget::RemoteOnly,
            LiveComputeTarget::Both,
        ] {
            assert!(
                !remote_suspends_local(false, mode),
                "remote_active == false must never suspend, regardless of mode"
            );
        }
        assert!(
            !remote_suspends_local(true, LiveComputeTarget::LocalOnly),
            "LocalOnly never dispatches remote, but even if remote_active were \
             somehow true, LocalOnly must not suspend local tracing"
        );
        assert!(
            remote_suspends_local(true, LiveComputeTarget::RemoteOnly),
            "RemoteOnly must reproduce the pre-existing suspend-while-remote-active \
             behaviour exactly"
        );
        assert!(
            !remote_suspends_local(true, LiveComputeTarget::Both),
            "Both must never suspend local tracing -- it keeps contributing, it never \
             yields the buffer to remote"
        );
    }

    #[test]
    fn only_both_mode_combines_and_only_while_remote_is_still_active() {
        for mode in [
            LiveComputeTarget::LocalOnly,
            LiveComputeTarget::RemoteOnly,
            LiveComputeTarget::Both,
        ] {
            assert!(
                !should_combine_remote(false, mode),
                "remote_active == false must never combine, regardless of mode"
            );
        }
        assert!(!should_combine_remote(true, LiveComputeTarget::LocalOnly));
        assert!(
            !should_combine_remote(true, LiveComputeTarget::RemoteOnly),
            "RemoteOnly's local tracing is suspended whenever this could matter -- it \
             must never itself decide to combine"
        );
        assert!(should_combine_remote(true, LiveComputeTarget::Both));
    }

    #[test]
    fn combining_persists_past_a_successful_remote_completion() {
        // `RenderContext::remote_active` deliberately stays `true` past `RemoteUpdate::
        // Done` (see its own doc comment) -- for `Both`, this is what lets local's
        // continued tracing keep ADDING to remote's finished contribution, mirroring
        // `resolve_remote_ownership`'s own
        // `remote_rendering_survives_quiet_frames_with_no_dirty` test below for the
        // suspension side of the same field.
        assert!(should_combine_remote(true, LiveComputeTarget::Both));
    }

    /// Verify requirement: "a settled combined render is not overwritten by either
    /// engine". `resolve_remote_ownership` staying `true` through quiet frames (pinned
    /// separately, below, for the raw resolution itself) means `should_combine_remote`
    /// never flips `false` on its own just because time passed or `Done` arrived --
    /// local's growing sample count only ever ADDS to what `remote_accumulator` already
    /// holds (see `render_thread::mod`'s doc comment: neither engine's buffer is ever
    /// reset or discarded by the other), so the combined image can only ever converge
    /// further, never regress to a from-scratch restart.
    #[test]
    fn a_settled_combined_render_keeps_combining_rather_than_being_overwritten() {
        for _quiet_frame in 0..5 {
            // Every quiet frame (no fresh `dirty`) after a completed remote render
            // resolves to the SAME `remote_active` -- see
            // `remote_rendering_survives_quiet_frames_with_no_dirty` below -- so
            // `should_combine_remote` keeps answering `true` across all of them.
            let resolved = resolve_remote_ownership(false, true, true);
            assert!(resolved);
            assert!(should_combine_remote(resolved, LiveComputeTarget::Both));
        }
    }

    /// Verify requirement: "a scene change during a combined render restarts
    /// accumulation correctly". A non-drag scene change (material/lighting/etc.) is
    /// exactly the `dirty=true, remote_active=true, was_remote_active_last_frame=true`
    /// shape `resolve_remote_ownership`'s own
    /// `a_scene_change_after_a_completed_remote_render_resumes_local_tracing` test
    /// already pins for the RemoteOnly-suspension side of this field -- this pins the
    /// Both-mode COMBINING side of the identical release: it must stop folding what is
    /// now a stale-scene remote contribution into the display, the same release
    /// `update_accumulation_state`'s own (separately tested) `dirty` handling already
    /// resets `accum_buffer` for.
    #[test]
    fn a_scene_change_during_a_combined_render_stops_combining_a_now_stale_contribution() {
        let resolved = resolve_remote_ownership(true, true, true);
        assert!(
            !resolved,
            "a genuine scene invalidation must release ownership even mid-combine"
        );
        assert!(
            !should_combine_remote(resolved, LiveComputeTarget::Both),
            "combining must stop the instant ownership is released, so local's freshly \
             reset accum_buffer (see update_accumulation_state's own dirty handling) is \
             never summed with a now-stale-scene remote contribution"
        );
    }

    // ---- Disjoint sample ranges for the combined live path ---------------------------

    #[test]
    fn local_sample_offset_never_falls_inside_remotes_reserved_range() {
        // Mirrors `bridge::export_thread::remote::tests::
        // remote_and_local_ranges_never_overlap_for_any_split`, specialised to the live
        // path's fixed (not calibrated) reservation: local's absolute sample index for
        // ANY frame of a combined epoch is `combined_sample_offset(reserved, n)` for
        // some `n >= 0`, which by construction can never land inside
        // `[0, remote_reserved_samples)` -- no gap, no overlap, exactly the export's own
        // `remote_range.end == local_range.start` invariant, specialised to a fixed
        // (not calibrated) split.
        for remote_reserved_samples in [0u32, 128, 512, 4096] {
            for local_pre_frame_count in [0u32, 1, 64, 10_000] {
                let sample_offset =
                    combined_sample_offset(remote_reserved_samples, local_pre_frame_count);
                assert!(
                    sample_offset >= remote_reserved_samples,
                    "local's sample_offset must never fall inside remote's reserved \
                     range [0, {remote_reserved_samples})"
                );
            }
        }
    }

    #[test]
    fn a_reservation_of_zero_reproduces_pre_combining_behaviour_exactly() {
        // `remote_reserved_samples == 0` is what every non-combining frame (LocalOnly,
        // RemoteOnly, or Both before any remote dispatch this epoch) feeds
        // `combined_sample_offset` -- this must be a true no-op, bit-identical to the
        // `current_sample_count - spp` expression this crate always used before
        // combining existed.
        for local_pre_frame_count in [0u32, 1, 64, 10_000] {
            assert_eq!(
                combined_sample_offset(0, local_pre_frame_count),
                local_pre_frame_count
            );
        }
    }

    // ---- resolve_remote_ownership: Task 2's single choke point -----------------------

    #[test]
    fn nothing_active_and_no_dirty_stays_inactive() {
        assert!(!resolve_remote_ownership(false, false, false));
    }

    #[test]
    fn the_handoffs_own_settling_entry_does_not_release_ownership() {
        // DiscardLocalPreview sets `dirty` and `remote_active` together in the SAME
        // locked mutation, starting from `remote_active == false` (the mutual-exclusion
        // invariant `RenderContext::camera_moving` documents) -- so this is the exact
        // shape of the frame that first observes it: `dirty` freshly true, `remote_active`
        // freshly true, and it was `false` one frame ago.
        assert!(
            resolve_remote_ownership(true, true, false),
            "the hand-off starting must not immediately cancel itself"
        );
    }

    #[test]
    fn a_completed_remote_render_is_not_overwritten_by_local_tracing() {
        // The exact steady state after `RemoteUpdate::Done`: `remote_active` stays
        // `true` (see `RenderContext::remote_active`'s doc comment), and every
        // subsequent frame with nothing new touching the scene (`dirty == false`) must
        // leave it exactly alone -- this is the regression test for the reported bug
        // (local tracing restarting from scratch and progressively overwriting the
        // finished remote image).
        assert!(resolve_remote_ownership(false, true, true));
    }

    #[test]
    fn a_scene_change_after_a_completed_remote_render_resumes_local_tracing() {
        // Same steady state as above (`remote_active` left `true` after `Done`), but
        // now a `gui::*` callback -- material, lighting, quality, a fresh camera drag,
        // anything -- sets `ctx.dirty = true` for a reason that has nothing to do with
        // the handoff machine. `was_remote_active_last_frame == true` is what tells
        // this apart from the legitimate Settling-entry case above: ownership must be
        // released so local tracing actually resumes instead of freezing on a now-wrong
        // image.
        assert!(!resolve_remote_ownership(true, true, true));
    }

    #[test]
    fn a_dirty_frame_while_already_inactive_stays_inactive() {
        // Ordinary local rendering (remote hand-off never engaged, or already released)
        // -- a `dirty` frame here is just this crate's everyday "camera moved"/"material
        // changed" accumulation reset, nothing to release.
        assert!(!resolve_remote_ownership(true, false, false));
        assert!(!resolve_remote_ownership(true, false, true));
    }

    #[test]
    fn remote_rendering_survives_quiet_frames_with_no_dirty() {
        // Mid-`RemoteRendering`/`Settling`, most frames carry no fresh `dirty` at all
        // (the pose hasn't changed) -- ownership must simply persist.
        assert!(resolve_remote_ownership(false, true, true));
        assert!(resolve_remote_ownership(false, true, false));
    }

    // ---- update_accumulation_state: suspension must never discard samples -----------

    #[test]
    fn a_quiet_frame_with_unchanged_dimensions_preserves_accumulated_samples() {
        // Models what every suspended frame (paused/tab-hidden/remote_active/
        // export_active) actually feeds this function: `dirty == false`, same
        // width/height as last frame -- see the render loop's own call site, which runs
        // this UNCONDITIONALLY every iteration, suspended or not, before the
        // `tracing_suspended` gate ever short-circuits. Neither branch may fire.
        let mut accum = vec![Vec3::new(1.0, 2.0, 3.0); 4];
        let mut first_hit_depth = vec![0.5; 4];
        let mut first_hit_normal = vec![Vec3::X; 4];
        let mut first_hit_facet_id = vec![7; 4];
        let mut accum_samples = 42;
        let mut last_width = 2;
        let mut last_height = 2;

        update_accumulation_state(
            2,
            2,
            false,
            &mut AccumulationBuffers {
                accum: &mut accum,
                first_hit_depth: &mut first_hit_depth,
                first_hit_normal: &mut first_hit_normal,
                first_hit_facet_id: &mut first_hit_facet_id,
            },
            &mut accum_samples,
            &mut last_width,
            &mut last_height,
        );

        assert_eq!(
            accum_samples, 42,
            "a suspended frame must not reset progress"
        );
        assert_eq!(
            accum,
            vec![Vec3::new(1.0, 2.0, 3.0); 4],
            "the running sum must survive"
        );
    }

    #[test]
    fn a_dirty_frame_still_resets_accumulation_regardless_of_remote_ownership() {
        // `update_accumulation_state` only ever sees `dirty`/`width`/`height` -- it has
        // no idea `remote_active` exists, by design (see `AccumulationBuffers`'s call
        // site: the reset decision and the ownership-release decision are two
        // deliberately independent concerns computed from the same underlying `dirty`
        // flag, not one function doing both). A real scene change must still reset the
        // buffer even on the very frame that also releases remote ownership.
        let mut accum = vec![Vec3::new(1.0, 2.0, 3.0); 4];
        let mut first_hit_depth = vec![0.5; 4];
        let mut first_hit_normal = vec![Vec3::X; 4];
        let mut first_hit_facet_id = vec![7; 4];
        let mut accum_samples = 42;
        let mut last_width = 2;
        let mut last_height = 2;

        update_accumulation_state(
            2,
            2,
            true,
            &mut AccumulationBuffers {
                accum: &mut accum,
                first_hit_depth: &mut first_hit_depth,
                first_hit_normal: &mut first_hit_normal,
                first_hit_facet_id: &mut first_hit_facet_id,
            },
            &mut accum_samples,
            &mut last_width,
            &mut last_height,
        );

        assert_eq!(accum_samples, 0);
        assert_eq!(accum, vec![Vec3::ZERO; 4]);
    }
}

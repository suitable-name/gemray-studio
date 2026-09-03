//! The preview-then-handoff orchestrator: a repeating `slint::Timer` that polls
//! `RenderContext`'s camera/light pose, feeds `bridge::handoff::HandoffMachine`, and
//! dispatches to `bridge::remote_render` when the machine decides to hand off --
//! including the async guide-buffer and denoise generations that keep the (multi-second
//! at 4K) denoise pass off the Slint UI thread.
//!
//! Split out of `gui::remote` purely to keep that module (already sizeable) from
//! growing further -- same reasoning as `gui::detail`/`gui::search`/`gui::remote`
//! itself.
//!
//! `poll_tick` is also the one place that decides "is the camera currently moving" for
//! `bridge::local_preview` (the local-only preview-then-settle feature) -- it writes
//! `RenderContext::camera_moving` from this SAME `HandoffMachine` instance's state every
//! tick, so that feature and this one share one definition of "settled" rather than each
//! running its own, differently-tuned debounce timer.

use super::worker_callbacks::backend_label;
use crate::{
    MainWindow,
    bridge::{
        export_thread::SceneSnapshot,
        guide_pass::{GuideBuffers, GuideCache, GuideKey, generate_guide_buffers_cancellable},
        handoff::{HandoffAction, HandoffEvent, HandoffMachine, HandoffState},
        remote_render::{self, RemoteRenderHandle, RemoteUpdate},
        render_thread::{
            DenoiseScratch, FirstHitSnapshot, RenderContext, denoise_and_tonemap_frame,
            tonemap_running_average,
        },
    },
    gui::show_toast,
    settings::{LiveComputeTarget, SettingsPersister, WorkerSettings},
};
use gemray::{
    geometry::plane::GpuFacetPlane, optics::raytracer::Camera, renderer::denoise::AtrousDenoiser,
};
use gemray_net::{SceneState, client::Accumulator};
use glam::Vec3;
use slint::{ComponentHandle, Weak};
use std::{
    rc::Rc,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};

/// How long orientation must stay unchanged before a remote handoff is attempted --
/// see `bridge::handoff`'s module docs on why this is what "settled" means.
const SETTLE_DEBOUNCE: Duration = Duration::from_millis(600);
/// How often the orchestrator timer below polls `RenderContext`'s camera/light pose
/// for a change. Independent of `SETTLE_DEBOUNCE` -- this is a poll granularity, not
/// the debounce itself.
const POLL_INTERVAL_MS: u64 = 100;

/// A snapshot of everything the orchestrator watches for a change, cheap to compare
/// every poll tick.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Pose {
    yaw: f32,
    pitch: f32,
    distance: f32,
    light_yaw: f32,
    light_pitch: f32,
}

fn current_pose(ctx: &Mutex<RenderContext>) -> Pose {
    let guard = ctx
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Pose {
        yaw: guard.yaw,
        pitch: guard.pitch,
        distance: guard.distance,
        light_yaw: guard.light_yaw,
        light_pitch: guard.light_pitch,
    }
}

/// State shared by the orchestrator's timer tick (always on the Slint event loop) and
/// its `on_update` closure (constructed there, but must be `Send` to cross into
/// `bridge::remote_render`'s worker thread as the closure argument -- see
/// `spawn_remote_render`'s bound). `Arc<Mutex<_>>` rather than the `Rc<RefCell<_>>>`
/// most other UI-thread-only state in this crate uses (e.g.
/// `gui::mod::setup_render_export_callbacks`'s `export_handle`) for exactly that
/// reason; every actual access still only ever happens on the Slint event loop, so the
/// lock is never contended.
struct Orchestrator {
    handoff: HandoffMachine,
    remote_handle: Option<RemoteRenderHandle>,
    accumulator: Option<Arc<Mutex<Accumulator>>>,
    last_pose: Pose,
    last_change_at: Instant,
    /// Guide buffers (depth/normal/facet-id) for denoising a remote-sourced merged
    /// image -- see `bridge::guide_pass`'s module docs.
    guide_cache: GuideCache,
    /// The guide-buffer prepass currently running on a background thread for the pose
    /// this orchestrator last dispatched a `RenderRequest` for -- `None` before the
    /// first dispatch, or once its result has been adopted/superseded/cancelled and
    /// nothing is running any more. See [`PendingGuideGeneration`]'s own doc comment.
    pending_guide_gen: Option<PendingGuideGeneration>,
    /// The full denoise-and-tonemap pass currently running on a background thread for
    /// `last_denoised`'s (or the about-to-be-`redraw`n) pose -- `None` whenever nothing
    /// is in flight. See [`PendingDenoiseGeneration`]'s own doc comment for why this,
    /// unlike `pending_guide_gen`, gets redispatched repeatedly (once per completed
    /// generation) rather than once per `RenderRequest`.
    pending_denoise_gen: Option<PendingDenoiseGeneration>,
    /// The most recently completed background denoise, tagged with the pose key
    /// ([`GuideKey`]) it is valid for. `redraw_from_accumulator` shows this (instead of
    /// re-deriving a plain, noisy tonemap every redraw) while a fresher generation is
    /// still cooking, so the displayed image doesn't flicker between denoised and noisy
    /// on every `FRAME` event during the several seconds one background pass takes at
    /// 4K. Cleared (never shown) once the pose changes -- a stale-pose denoised image
    /// is no more valid to keep displaying than a stale-pose guide buffer would be, and
    /// that check is the same structural [`GuideKey`] equality [`adopt_ready_guides`]
    /// already uses, not a timer or generation counter.
    last_denoised: Option<(GuideKey, Vec<u8>)>,
}

/// A background primary-ray-only guide-buffer prepass in flight for one dispatched
/// `RenderRequest` -- started at dispatch time (`start_remote_render`), not on the
/// first redraw that needs guides, so the work overlaps the network round trip and the
/// remote render itself instead of stalling the first post-dispatch redraw. See
/// `bridge::guide_pass`'s module doc comment for the full story of why this used to run
/// synchronously on the UI thread.
///
/// Cancellation mirrors `bridge::export_thread::ExportHandle`'s cooperative
/// `Arc<AtomicBool>` pattern (see that module's doc comment, lines ~144-236): setting
/// `cancel` doesn't tear down the background thread forcibly, it just asks
/// `generate_guide_buffers_cancellable`'s per-row check to stop early.
///
/// A superseded generation can never overwrite a newer one's result: every call to
/// [`spawn_guide_generation`] allocates a BRAND NEW `result` `Arc`, never reuses the
/// previous generation's. So even if a cancelled (or simply slow) background thread
/// finishes after `Orchestrator::pending_guide_gen` has already moved on to a fresh
/// [`PendingGuideGeneration`], its write lands in an `Arc` that nothing still reachable
/// from `Orchestrator` ever reads from again -- there is no shared mutable slot two
/// generations could race to write into.
struct PendingGuideGeneration {
    /// The pose/geometry/resolution this generation is FOR -- the SAME [`GuideKey`]
    /// identity `bridge::guide_pass::GuideCache` keys its own cache on (see
    /// `GuideCache::key_for`), reused rather than inventing a second notion of "same
    /// pose and geometry". Compared against the CURRENT desired key on every redraw
    /// (`adopt_ready_guides`) so a result is only ever adopted when it actually matches
    /// the image currently being displayed.
    key: GuideKey,
    /// Cooperative cancellation flag: set when the pose changes again (the render this
    /// generation was for gets cancelled/superseded), so the background thread abandons
    /// a now-pointless computation instead of running it to completion.
    cancel: Arc<AtomicBool>,
    /// Filled in by the background thread once, and only once, if it finishes without
    /// observing `cancel`. `None` while generation is still in flight (or was
    /// abandoned) -- callers must treat a `None` read as "not ready yet", never block
    /// waiting for it.
    result: Arc<Mutex<Option<GuideBuffers>>>,
}

/// Kicks off the guide-buffer prepass for `key` on a background thread. Does NOT cancel
/// or replace any previous [`PendingGuideGeneration`] itself -- the caller
/// (`start_remote_render`) does that first, since only it knows whether a previous
/// generation exists at all.
fn spawn_guide_generation(
    key: GuideKey,
    camera: Camera,
    planes: Vec<GpuFacetPlane>,
    width: u32,
    height: u32,
) -> PendingGuideGeneration {
    let cancel = Arc::new(AtomicBool::new(false));
    let result = Arc::new(Mutex::new(None));
    let cancel_worker = Arc::clone(&cancel);
    let result_worker = Arc::clone(&result);

    std::thread::spawn(move || {
        if let Some(buffers) =
            generate_guide_buffers_cancellable(width, height, &camera, &planes, &cancel_worker)
        {
            *result_worker.lock().unwrap_or_else(PoisonError::into_inner) = Some(buffers);
        }
        // A `None` (cancelled) result is simply dropped: `pending_guide_gen` in
        // `Orchestrator` has already moved on to a different generation by the time
        // cancellation is ever observed here (see `apply_actions`'s
        // `DiscardRemotePartial` handling), so there is nothing left that would read
        // this result even if it were written.
    });

    PendingGuideGeneration {
        key,
        cancel,
        result,
    }
}

/// Checks whether guide buffers are ready and correct for `desired_key` -- the
/// pose/geometry identity (see `GuideCache::key_for`) of the image about to be redrawn
/// -- adopting a background generation's result into `guide_cache` when it matches, and
/// reporting "not ready" otherwise so the caller can fall back to a plain tonemap for
/// this frame rather than blocking the UI thread to regenerate guides synchronously.
///
/// Pure aside from the one `Mutex` lock on `pending`'s result slot, so it's directly
/// unit-testable with a hand-built [`PendingGuideGeneration`] and no real background
/// thread, socket, or timer -- see this module's tests.
fn adopt_ready_guides(
    desired_key: &GuideKey,
    guide_cache: &mut GuideCache,
    pending: Option<&PendingGuideGeneration>,
) -> bool {
    if guide_cache.matches_key(desired_key) {
        // Either a previous adopt already installed the right buffers, or `ensure` was
        // called for this exact key before -- either way, nothing to do.
        return true;
    }

    let Some(pending) = pending else {
        return false;
    };
    if pending.key != *desired_key {
        // The in-flight (or already-finished) generation is for a DIFFERENT pose --
        // reject it rather than denoising this frame with guides that don't match it.
        return false;
    }

    let mut slot = pending
        .result
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let Some(buffers) = slot.take() else {
        // Still in flight.
        return false;
    };
    drop(slot);
    guide_cache.adopt(desired_key.clone(), buffers);
    true
}

/// The full À-Trous denoise-and-tonemap pass (`render_thread::denoise_and_tonemap_frame`)
/// running on a background thread for one pose/buffer snapshot, mirroring
/// [`PendingGuideGeneration`] -- see that struct's doc comment for the shared
/// cancellation-by-orphaning reasoning (a brand new `result` `Arc` every dispatch, so a
/// stale generation's eventual write lands somewhere nothing still reachable ever reads).
///
/// Unlike guide generation, which is dispatched exactly once per `RenderRequest`
/// (guides depend only on pose+geometry, both already known at dispatch time), a denoise
/// generation is redispatched repeatedly -- once every time the previous one completes
/// and is adopted -- because its INPUT (the accumulation buffer) keeps growing as more
/// `FRAME` events arrive. There is deliberately no cooperative `cancel` flag here the way
/// `PendingGuideGeneration` has one: `renderer::denoise::AtrousDenoiser` has no
/// early-exit hook (and this task does not touch the denoiser's algorithm), so an
/// abandoned generation simply runs to completion on its own thread, consuming CPU but
/// touching nothing else -- correctness comes entirely from the structural key check in
/// [`adopt_ready_denoise`] below, never from timing or an assumption that a stale
/// generation stopped.
struct PendingDenoiseGeneration {
    /// The pose/geometry/resolution this generation's OUTPUT is valid for -- the buffer
    /// snapshot it was handed came from an accumulator matching this pose, and a result
    /// is only ever adopted once [`adopt_ready_denoise`] confirms this still equals the
    /// pose currently on screen. The buffer's sample COUNT is deliberately not part of
    /// the key: a denoised frame that is a few samples behind the latest accumulation is
    /// still a valid (if slightly stale) image of the SAME pose, unlike a guide buffer
    /// for the wrong pose entirely.
    key: GuideKey,
    /// Filled in by the background thread once, and only once, with the finished RGBA
    /// bytes. `None` while still in flight; callers must treat that as "not ready yet",
    /// never block waiting for it.
    result: Arc<Mutex<Option<Vec<u8>>>>,
}

/// Kicks off one full denoise-and-tonemap pass on a background thread for `key`'s pose,
/// over the given `buffer`/`samples_done` snapshot and `guides` (already known-good for
/// `key` -- the caller confirms this via [`adopt_ready_guides`] before calling, and
/// clones the buffers out of `guide_cache` since that cache is otherwise only ever
/// touched from the Slint event-loop thread; see `GuideCache`'s own doc comment).
///
/// Runs the actual pass via [`render_merged_frame`] itself (the same function this
/// module's `render_merged_frame_*` tests exercise directly), fed a throwaway
/// [`GuideCache`] pre-seeded with `guides` via [`GuideCache::adopt`] so `render_merged_frame`'s
/// own internal `guide_cache.ensure` call is a guaranteed cache hit rather than a
/// synchronous regenerate -- this background thread never calls
/// `generate_guide_buffers` itself. Builds its own fresh, throwaway [`AtrousDenoiser`]
/// and scratch buffers too, rather than sharing anything `render_thread`'s local loop or
/// a synchronous call from this module's tests would use -- this background thread owns
/// nothing anyone else touches, and one extra allocation every ~3.5s (4K) is immaterial
/// next to the pass itself.
/// Everything [`spawn_denoise_generation`] needs on its background thread, owned rather
/// than borrowed since that thread outlives this function's own call frame.
struct DenoiseGenerationJob {
    width: u32,
    height: u32,
    samples_done: u32,
    buffer: Vec<Vec3>,
    guides: GuideBuffers,
    yaw: f32,
    pitch: f32,
    distance: f32,
    planes: Vec<GpuFacetPlane>,
}

fn spawn_denoise_generation(key: GuideKey, job: DenoiseGenerationJob) -> PendingDenoiseGeneration {
    let result = Arc::new(Mutex::new(None));
    let result_worker = Arc::clone(&result);
    let key_for_thread = key.clone();

    std::thread::spawn(move || {
        let mut guide_cache = GuideCache::new();
        guide_cache.adopt(key_for_thread, job.guides);
        let mut denoiser = AtrousDenoiser::new();
        let mut avg_color_buf = Vec::new();
        let mut filtered_buf = Vec::new();
        let bytes = render_merged_frame(
            AccumSnapshot {
                width: job.width,
                height: job.height,
                samples_done: job.samples_done,
                buffer: &job.buffer,
            },
            true,
            PoseAndGeometry {
                yaw: job.yaw,
                pitch: job.pitch,
                distance: job.distance,
                planes: &job.planes,
            },
            &mut guide_cache,
            &mut DenoiseScratch {
                denoiser: &mut denoiser,
                avg_color_buf: &mut avg_color_buf,
                filtered_buf: &mut filtered_buf,
            },
        );
        *result_worker.lock().unwrap_or_else(PoisonError::into_inner) = Some(bytes);
    });

    PendingDenoiseGeneration { key, result }
}

/// Checks whether a background denoise pass is finished and still valid for
/// `desired_key`, taking (and returning) its bytes if so. `None` covers both "nothing in
/// flight" and "in flight but for a different pose" and "in flight but not done yet" --
/// the caller cannot distinguish those cases from this alone and does not need to; every
/// caller's fallback (show the last-adopted frame, or a plain tonemap if there is none)
/// is the same regardless of which.
///
/// Pure aside from the one `Mutex` lock on `pending`'s result slot, so it's directly
/// unit-testable with a hand-built [`PendingDenoiseGeneration`] and no real background
/// thread -- mirrors [`adopt_ready_guides`]'s own testability for the same reason.
fn adopt_ready_denoise(
    desired_key: &GuideKey,
    pending: Option<&PendingDenoiseGeneration>,
) -> Option<Vec<u8>> {
    let pending = pending?;
    if pending.key != *desired_key {
        return None;
    }
    let mut slot = pending
        .result
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    slot.take()
}

/// Wires up the preview-then-handoff orchestrator: a repeating `slint::Timer` polls
/// `render_ctx`'s camera/light pose, feeds `bridge::handoff::HandoffMachine`, and
/// dispatches to `bridge::remote_render` when the machine decides to hand off. Returns
/// the `slint::Timer` -- the caller (`gui::mod::run_gui`) MUST keep it alive for the
/// life of the window (a dropped `Timer` stops firing), the same requirement as any
/// other `slint::Timer` used this way.
#[must_use]
pub fn setup_remote_rendering(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
) -> slint::Timer {
    let state = Arc::new(Mutex::new(Orchestrator {
        handoff: HandoffMachine::new(),
        remote_handle: None,
        accumulator: None,
        last_pose: current_pose(render_ctx),
        last_change_at: Instant::now(),
        guide_cache: GuideCache::new(),
        pending_guide_gen: None,
        pending_denoise_gen: None,
        last_denoised: None,
    }));
    let next_request_id = Rc::new(AtomicU32::new(1));

    let timer = slint::Timer::default();
    let ui_weak = ui.as_weak();
    let render_ctx_poll = render_ctx.clone();
    let settings_store_poll = settings_store.clone();
    timer.start(
        slint::TimerMode::Repeated,
        Duration::from_millis(POLL_INTERVAL_MS),
        move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };
            poll_tick(
                &ui,
                &render_ctx_poll,
                &settings_store_poll,
                &state,
                &next_request_id,
            );
        },
    );
    timer
}

fn poll_tick(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    settings_store: &Arc<SettingsPersister>,
    state: &Arc<Mutex<Orchestrator>>,
    next_request_id: &Rc<AtomicU32>,
) {
    let pose = current_pose(render_ctx);
    let changed = pose != lock(state).last_pose;

    if changed {
        lock(state).last_pose = pose;
        lock(state).last_change_at = Instant::now();
        let actions = lock(state).handoff.handle(HandoffEvent::OrientationChanged);
        apply_actions(&actions, render_ctx, state);
        sync_served_by_to_ui(ui, state);
        sync_camera_moving_to_ctx(render_ctx, state);
        return;
    }

    let (should_check_settle, elapsed_enough) = {
        let s = lock(state);
        (
            matches!(s.handoff.state(), HandoffState::Previewing),
            s.last_change_at.elapsed() >= SETTLE_DEBOUNCE,
        )
    };
    if should_check_settle && elapsed_enough {
        // Task: Local+Remote combined live rendering -- `LiveComputeTarget::LocalOnly`
        // means "never hand off to remote at all", so it's treated as no worker being
        // configured right here, at the one place `HandoffMachine` ever learns whether a
        // worker is available. `bridge::handoff` itself stays entirely unaware this
        // choice exists -- it only ever sees `worker_available: bool`, exactly as
        // before this feature existed -- see that module's own doc comment on staying
        // pure.
        let live_compute_target = render_ctx
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .live_compute_target;
        let worker = if matches!(live_compute_target, LiveComputeTarget::LocalOnly) {
            None
        } else {
            settings_store
                .snapshot()
                .settings
                .remote_workers
                .first()
                .cloned()
        };
        let actions = lock(state).handoff.handle(HandoffEvent::SettleElapsed {
            worker_available: worker.is_some(),
        });
        apply_actions(&actions, render_ctx, state);
        if let Some(worker) = worker {
            start_remote_render(ui, render_ctx, worker, next_request_id, state);
        }
    }
    // Task: local preview-then-settle rendering -- `bridge::RenderContext::camera_moving`
    // must be kept in sync on EVERY tick, not just the branches above that changed the
    // handoff state, since `Previewing` can also simply persist unchanged tick to tick
    // while a drag continues (see the module doc comment for why this reuses the
    // `HandoffMachine` instance already driven above rather than a second debounce).
    sync_camera_moving_to_ctx(render_ctx, state);
    reconcile_served_by_after_release(ui, render_ctx, state);
}

/// Task 2: `render_thread::mod`'s `resolve_remote_ownership` can release
/// `ctx.remote_active` entirely on its own, off this orchestrator's own timer, whenever
/// a non-drag scene change (material/lighting/quality/etc.) arrives while a completed
/// remote render is still the displayed image -- see that function's doc comment. A
/// real camera drag already keeps `HandoffMachine::served_by` truthful on its own (the
/// pose-changed branch above, via `HandoffEvent::OrientationChanged`), but a non-drag
/// release has no handoff event of its own to ride along with, so this polls for it:
/// whenever `served_by() == Remote` (which, per `HandoffMachine::handle`'s own
/// bookkeeping, only ever holds while `state() == Idle`) but the render loop has since
/// cleared `ctx.remote_active`, local tracing has already silently taken back over --
/// feed `HandoffEvent::SceneInvalidated` so the "served by remote" indicator stops
/// claiming otherwise. A no-op on every other tick (the common case): one extra lock
/// acquisition on `render_ctx`, already held every tick by `current_pose` above.
fn reconcile_served_by_after_release(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    state: &Arc<Mutex<Orchestrator>>,
) {
    let still_remote_active = render_ctx
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remote_active;
    if still_remote_active {
        return;
    }
    let served_by_remote = matches!(
        lock(state).handoff.served_by(),
        crate::bridge::handoff::ImageSource::Remote
    );
    if served_by_remote {
        lock(state).handoff.handle(HandoffEvent::SceneInvalidated);
        sync_served_by_to_ui(ui, state);
    }
}

/// Mirrors `HandoffMachine::state() == Previewing` onto `RenderContext::camera_moving`
/// -- see that field's own doc comment for why this is the single source of truth for
/// "is the camera currently moving" shared by both the remote handoff and the local
/// preview-then-settle feature.
fn sync_camera_moving_to_ctx(
    render_ctx: &Arc<Mutex<RenderContext>>,
    state: &Arc<Mutex<Orchestrator>>,
) {
    let moving = matches!(lock(state).handoff.state(), HandoffState::Previewing);
    render_ctx
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .camera_moving = moving;
}

/// Locks `state`, recovering from a poisoned mutex the same way every other shared
/// state in this crate does (`std::sync::PoisonError::into_inner`) -- a panic on one
/// event's handling must not permanently wedge every future one.
fn lock(state: &Arc<Mutex<Orchestrator>>) -> std::sync::MutexGuard<'_, Orchestrator> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Mirrors `HandoffMachine::served_by` onto `MainWindow.served_by_remote` -- the single
/// source of truth for "which backend served the current image" the worker-list panel
/// displays. Called after every `HandoffMachine::handle` call that could change it
/// (a fresh drag flips it back to `Local` immediately, per `HandoffMachine`'s own doc
/// comment -- not just on a successful remote completion).
fn sync_served_by_to_ui(ui: &MainWindow, state: &Arc<Mutex<Orchestrator>>) {
    let served_by_remote = matches!(
        lock(state).handoff.served_by(),
        crate::bridge::handoff::ImageSource::Remote
    );
    ui.set_served_by_remote(served_by_remote);
}

/// Carries out the [`HandoffAction`]s a `HandoffMachine::handle` call returned, against
/// the real `RenderContext` and the orchestrator's own remote-render/accumulator state.
/// Does NOT dispatch [`HandoffAction::SendRenderRequestToWorker`] itself (that needs the
/// worker config and UI handle, supplied by [`start_remote_render`]'s caller right after
/// this returns) -- only the discard/cancel side effects.
fn apply_actions(
    actions: &[HandoffAction],
    render_ctx: &Arc<Mutex<RenderContext>>,
    state: &Arc<Mutex<Orchestrator>>,
) {
    for action in actions {
        match action {
            // `DiscardLocalPreview` and `SendRenderRequestToWorker` always fire together
            // (see `bridge::handoff`'s module doc comment) and are both deferred to the
            // SAME call site: `start_remote_render`, called by `poll_tick` right after
            // this. For `LiveComputeTarget::Both`, that's also where the
            // `remote_active`/`dirty`/shared-accumulator hand-off
            // (`RenderContext::remote_accumulator`) has to land, in ONE locked mutation
            // with the discard/dispatch -- see that field's own doc comment.
            HandoffAction::DiscardLocalPreview | HandoffAction::SendRenderRequestToWorker => {}
            HandoffAction::SendCancelToWorker => {
                if let Some(handle) = &lock(state).remote_handle {
                    handle.cancel();
                }
            }
            HandoffAction::DiscardRemotePartial => {
                let mut s = lock(state);
                s.remote_handle = None;
                s.accumulator = None;
                // The pose that made this render's guides valid is gone too -- abandon
                // whatever prepass was still running for it rather than letting a stale
                // pose's guides land on a future redraw (see `PendingGuideGeneration`'s
                // doc comment).
                if let Some(pending) = s.pending_guide_gen.take() {
                    pending.cancel.store(true, Ordering::Relaxed);
                }
                // Same idea for the background denoise pass -- there is no cooperative
                // cancel for it (see `PendingDenoiseGeneration`'s doc comment), so this
                // just drops the orchestrator's reference; the thread itself still runs
                // to completion, but nothing reachable from here will ever adopt its
                // result once a fresh `RenderRequest` overwrites this pose's state.
                s.pending_denoise_gen = None;
                s.last_denoised = None;
                drop(s);
                let mut ctx = render_ctx.lock().unwrap_or_else(PoisonError::into_inner);
                ctx.remote_active = false;
                ctx.dirty = true; // resume local previewing fresh, not from a discarded buffer
                // Task: Local+Remote combined live rendering -- the just-cancelled
                // request's shared accumulator (if any) must never keep being folded
                // into a combined display: a fresh local preview is about to start from
                // an empty buffer (the `dirty = true` above), and combining it with a
                // now-abandoned request's partial contribution would be exactly the
                // "carry a buffer from one source into the other" the handoff module's
                // own invariant forbids.
                ctx.remote_accumulator = None;
                ctx.remote_reserved_samples = 0;
            }
        }
    }
}

fn start_remote_render(
    ui: &MainWindow,
    render_ctx: &Arc<Mutex<RenderContext>>,
    worker: WorkerSettings,
    next_request_id: &Rc<AtomicU32>,
    state: &Arc<Mutex<Orchestrator>>,
) {
    // `samples`/`live_compute_target` are read live off `RenderContext` here, the exact
    // same treatment `width`/`height` already get (see this function's own doc comment)
    // -- a remote render always uses whatever sample budget/mode is CURRENTLY
    // configured, not a value captured once at some earlier time.
    let (width, height, samples, live_compute_target) = {
        let ctx = render_ctx.lock().unwrap_or_else(PoisonError::into_inner);
        (
            ctx.width,
            ctx.height,
            ctx.remote_render_samples,
            ctx.live_compute_target,
        )
    };
    if width == 0 || height == 0 {
        return;
    }
    let combining = matches!(live_compute_target, LiveComputeTarget::Both);
    let snapshot = SceneSnapshot::capture(render_ctx);
    let scene = scene_state_from_snapshot(&snapshot, width, height);

    // Kick off the guide-buffer prepass NOW, at dispatch time, rather than waiting for
    // the first `FRAME`/`PREVIEW` redraw to need it -- camera pose and gem geometry are
    // both already known here, so this overlaps the network round trip and the remote
    // render itself instead of stalling the UI thread on the first post-settle redraw
    // (see `bridge::guide_pass`'s module doc comment). Cancel whatever generation was
    // still running for a previous pose first -- a fresh dispatch always means a fresh
    // pose (`start_remote_render` only ever runs after `HandoffEvent::SettleElapsed`).
    //
    // Skipped entirely for `LiveComputeTarget::Both`: local tracing keeps running for
    // that mode (see `render_thread::mod`'s doc comment), producing its OWN first-hit
    // guide buffers for this exact pose as a side effect of its ordinary trace loop --
    // fresher and cheaper than a separate async prepass, so this would-be-redundant
    // dispatch is skipped rather than racing a second computation of the same thing.
    if !combining {
        let guide_key = GuideCache::key_for(
            width,
            height,
            snapshot.yaw,
            snapshot.pitch,
            snapshot.distance,
            &snapshot.active_planes,
        );
        let guide_camera = Camera::new(snapshot.yaw, snapshot.pitch, snapshot.distance, 42.0);
        let mut s = lock(state);
        if let Some(previous) = s.pending_guide_gen.take() {
            previous.cancel.store(true, Ordering::Relaxed);
        }
        s.pending_guide_gen = Some(spawn_guide_generation(
            guide_key,
            guide_camera,
            snapshot.active_planes, // moved: `snapshot` isn't used again after this
            width,
            height,
        ));
    }

    let accumulator = Arc::new(Mutex::new(Accumulator::new(width, height)));
    lock(state).accumulator = Some(Arc::clone(&accumulator));

    // Task: Local+Remote combined live rendering -- discards the local preview and
    // starts this settle's dispatch in ONE locked mutation, together with the
    // shared-accumulator hand-off the render thread reads (`RenderContext::
    // remote_accumulator`/`remote_reserved_samples`), so that thread can never observe
    // `remote_active`/`dirty` freshly true while still holding a STALE (previous
    // epoch's, or absent) accumulator/reservation -- see `RenderContext::
    // remote_accumulator`'s own doc comment. This is also where
    // `HandoffAction::DiscardLocalPreview`'s actual work happens -- see
    // `apply_actions`'s now-deferred arm for it.
    {
        let mut ctx = render_ctx.lock().unwrap_or_else(PoisonError::into_inner);
        ctx.remote_active = true;
        ctx.dirty = true;
        ctx.remote_accumulator = combining.then(|| Arc::clone(&accumulator));
        ctx.remote_reserved_samples = if combining { samples } else { 0 };
    }

    let request_id = next_request_id.fetch_add(1, Ordering::Relaxed);
    let ui_weak = ui.as_weak();
    let state_for_updates = Arc::clone(state);
    let render_ctx_for_updates = Arc::clone(render_ctx);
    let accumulator_for_redraw = Arc::clone(&accumulator);

    let handle = remote_render::spawn_remote_render(
        remote_render::RemoteRenderRequest {
            worker,
            request_id,
            scene,
            first_sample: 0,
            samples,
            width,
            height,
        },
        accumulator,
        move |update: RemoteUpdate| {
            handle_remote_update(
                &ui_weak,
                &render_ctx_for_updates,
                &state_for_updates,
                &accumulator_for_redraw,
                width,
                height,
                update,
            );
        },
    );
    lock(state).remote_handle = Some(handle);
}

/// Builds the fully-resolved `gemray_net::SceneState` a remote worker needs from a
/// local `SceneSnapshot` plus the session's render resolution -- see
/// `gemray_net::scene::SceneState`'s own doc comment on why every field must be a
/// resolved value, never a name/id the worker can't look up.
///
/// Frosted girdle: `snapshot.facet_finishes` is already either empty (toggle off at
/// capture time) or `girdle_facet_finishes(&snapshot.active_planes)` (toggle on) -- see
/// `export_thread::scene_snapshot::SceneSnapshot::capture`. `SceneState::girdle_frosted`
/// carries that same on/off bit rather than the resolved list itself (see that field's
/// own doc comment on why), so a non-empty `facet_finishes` here becomes `true`: the
/// remote worker re-derives the identical `Vec<FacetFinish>` from the identical
/// `planes` it already receives below, via the same deterministic
/// `girdle_facet_finishes` function.
fn scene_state_from_snapshot(snapshot: &SceneSnapshot, width: u32, height: u32) -> SceneState {
    SceneState {
        width,
        height,
        yaw: snapshot.yaw,
        pitch: snapshot.pitch,
        distance: snapshot.distance,
        light_yaw: snapshot.light_yaw,
        light_pitch: snapshot.light_pitch,
        exposure: snapshot.exposure,
        max_bounces: snapshot.max_bounces,
        lighting_preset: snapshot.lighting_preset,
        material: snapshot.material.clone(),
        planes: snapshot.active_planes.clone(),
        girdle_frosted: !snapshot.facet_finishes.is_empty(),
    }
}

/// Whether the live viewport is currently in `LiveComputeTarget::Both` -- see
/// `handle_remote_update`'s call sites for why they skip pushing a REMOTE-ONLY redraw
/// while this holds (the render thread's own display cycle is the combined image's sole
/// producer in that mode -- see `render_thread::mod`'s doc comment).
fn is_combining(render_ctx: &Arc<Mutex<RenderContext>>) -> bool {
    matches!(
        render_ctx
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .live_compute_target,
        LiveComputeTarget::Both
    )
}

fn handle_remote_update(
    ui_weak: &Weak<MainWindow>,
    render_ctx: &Arc<Mutex<RenderContext>>,
    state: &Arc<Mutex<Orchestrator>>,
    accumulator: &Arc<Mutex<Accumulator>>,
    width: u32,
    height: u32,
    update: RemoteUpdate,
) {
    let render_ctx = render_ctx.clone();
    let state = Arc::clone(state);
    let accumulator = Arc::clone(accumulator);
    let _ = ui_weak.upgrade_in_event_loop(move |ui| {
        match update {
            RemoteUpdate::Connected(info) => {
                let actions = lock(&state)
                    .handoff
                    .handle(HandoffEvent::RemoteStreamStarted);
                apply_actions(&actions, &render_ctx, &state);
                ui.set_served_by_worker_name(backend_label(info.render.as_ref()).into());
            }
            RemoteUpdate::Frame { samples_done } => {
                tracing::trace!("remote render: {samples_done} samples done");
                // Task: Local+Remote combined live rendering -- while combining, this
                // orchestrator's own denoise-and-push pipeline is skipped entirely: the
                // render thread's periodic display cycle already folds this SAME
                // accumulator's current total into the combined image it pushes (see
                // `render_thread::mod`'s doc comment), at a cadence (`DENOISE_MIN_INTERVAL`,
                // ~120ms) at least as fast as remote `FRAME` events typically arrive. A
                // redraw here would show a REMOTE-ONLY partial sum -- undercounting
                // local's own contribution -- and fight the render thread's combined
                // push for the same `render_image` property.
                if !is_combining(&render_ctx) {
                    redraw_from_accumulator(&ui, &accumulator, &render_ctx, &state, width, height);
                }
            }
            RemoteUpdate::Preview => {
                if !is_combining(&render_ctx) {
                    redraw_from_accumulator(&ui, &accumulator, &render_ctx, &state, width, height);
                }
            }
            RemoteUpdate::Progress { samples_done } => {
                tracing::trace!("remote render progress: {samples_done} samples done");
            }
            RemoteUpdate::Done { cancelled } => {
                if cancelled {
                    // A cancellation the worker confirmed after this orchestrator had
                    // already moved on (via DiscardRemotePartial) -- nothing further
                    // to do; the local preview is already back in charge.
                    return;
                }
                if !is_combining(&render_ctx) {
                    redraw_from_accumulator(&ui, &accumulator, &render_ctx, &state, width, height);
                }
                let actions = lock(&state).handoff.handle(HandoffEvent::RemoteDone);
                apply_actions(&actions, &render_ctx, &state);
                // `ctx.remote_active` is deliberately NOT cleared here -- a finished
                // remote render IS the settled, full-quality image; clearing it would
                // let local tracing race back in and progressively overwrite it with a
                // rough low-spp restart (the exact bug this comment replaces). It stays
                // set until `render_thread::mod`'s `resolve_remote_ownership` releases it
                // once the scene is genuinely invalidated -- see that function's and
                // `RenderContext::remote_active`'s doc comments.
                sync_served_by_to_ui(&ui, &state);
                let mut s = lock(&state);
                s.remote_handle = None;
                s.accumulator = None;
            }
            RemoteUpdate::Failed(message) => {
                let actions = lock(&state).handoff.handle(HandoffEvent::RemoteFailed);
                apply_actions(&actions, &render_ctx, &state);
                {
                    let mut ctx = render_ctx.lock().unwrap_or_else(PoisonError::into_inner);
                    ctx.remote_active = false;
                    ctx.dirty = true;
                    // Task: Local+Remote combined live rendering -- same reasoning as
                    // `DiscardRemotePartial` in `apply_actions`: a failed request's
                    // shared accumulator must never keep being folded into the combined
                    // display local is about to restart fresh (`dirty = true` above).
                    ctx.remote_accumulator = None;
                    ctx.remote_reserved_samples = 0;
                }
                let mut s = lock(&state);
                s.remote_handle = None;
                s.accumulator = None;
                drop(s);
                show_toast(&ui, &format!("Remote render failed: {message}"), "error");
            }
        }
    });
}

/// The remote accumulator's current running sum, as [`render_merged_frame`] receives it:
/// dimensions, sample count, and the merged radiance itself. `Copy` -- every field is a
/// shared reference or a scalar, matching `render_thread::gpu_backend::BackendFrame`'s
/// identical rationale.
#[derive(Clone, Copy)]
struct AccumSnapshot<'a> {
    width: u32,
    height: u32,
    samples_done: u32,
    buffer: &'a [Vec3],
}

/// The pose (`yaw`/`pitch`/`distance`) and geometry [`render_merged_frame`] needs to key
/// and, on a cache miss, regenerate the guide-buffer prepass -- see
/// [`GuideCache::ensure`]. `Copy` for the same reason as [`AccumSnapshot`].
#[derive(Clone, Copy)]
struct PoseAndGeometry<'a> {
    yaw: f32,
    pitch: f32,
    distance: f32,
    planes: &'a [GpuFacetPlane],
}

/// Turns a remote accumulator's current running sum into a displayable RGBA byte
/// buffer, applying the SAME single À-Trous denoise pass the local path applies to its
/// own readback -- never a per-source one (`buffer` here is already the merged,
/// summed-across-however-many-`FRAME`-events-arrived-so-far radiance; denoising is
/// nonlinear, so it must run once, after summing, on that whole merged buffer, exactly
/// like `render_thread::denoise_and_tonemap_frame`'s own doc comment requires of the
/// local accumulation buffer -- there is no per-contribution denoise anywhere in this
/// pipeline).
///
/// The depth/normal/facet-id guides a remote payload never carries (see
/// `bridge::remote_render`'s module docs on why they're not shipped over the wire) come
/// from `guide_cache`: a local primary-ray-only prepass over the CURRENT camera pose
/// and gem geometry (see `bridge::guide_pass`'s module docs for why that's valid for
/// ANY image of that pose, remote-sourced or not, and why caching on pose+geometry
/// rather than recomputing every call is what keeps this cheap across many `FRAME`
/// events from one in-progress render).
///
/// No Slint/GUI types in the signature -- exercised directly by this module's own unit
/// tests without a window, a socket, or a worker. This function itself still
/// runs the full (multi-second at 4K) denoise pass synchronously -- it is no longer
/// called directly from [`redraw_from_accumulator`] for that reason (that would block
/// the Slint UI thread, which is literally where `redraw_from_accumulator` runs, being
/// invoked from inside `handle_remote_update`'s `upgrade_in_event_loop` closure). It is
/// now called from [`spawn_denoise_generation`]'s background thread instead, with a
/// throwaway `GuideCache` pre-seeded via [`GuideCache::adopt`] so its own internal
/// `guide_cache.ensure` call is a guaranteed cache hit rather than a synchronous
/// regenerate -- see that function's doc comment.
fn render_merged_frame(
    accum: AccumSnapshot<'_>,
    denoise_enabled: bool,
    pose: PoseAndGeometry<'_>,
    guide_cache: &mut GuideCache,
    scratch: &mut DenoiseScratch<'_>,
) -> Vec<u8> {
    if !denoise_enabled {
        return tonemap_running_average(
            accum.width,
            accum.height,
            accum.samples_done,
            accum.buffer,
        );
    }
    let guides = guide_cache.ensure(
        accum.width,
        accum.height,
        pose.yaw,
        pose.pitch,
        pose.distance,
        pose.planes,
    );
    denoise_and_tonemap_frame(
        FirstHitSnapshot {
            width: accum.width,
            height: accum.height,
            current_sample_count: accum.samples_done,
            accum_buffer: accum.buffer,
            first_hit_depth: &guides.depth,
            first_hit_normal: &guides.normal,
            first_hit_facet_id: &guides.facet_id,
        },
        scratch,
    )
}

/// Reads the accumulator's current running sum plus the pose/geometry/denoise-toggle
/// state needs out of the real `Accumulator`/`RenderContext`/`Orchestrator`, decides
/// what to display, and pushes the result to the viewport image.
///
/// This runs on the Slint UI/event-loop thread (called from inside
/// `handle_remote_update`'s `upgrade_in_event_loop` closure, for every `RemoteUpdate::Frame`/
/// `Preview`/`Done` -- which, mid-render, can arrive many times per second; see
/// `bridge::guide_pass`'s module doc comment). That is exactly why the actual
/// (multi-second at 4K) denoise pass must never run inline here -- this function only
/// ever does cheap work synchronously (a plain tonemap, now itself parallel -- see
/// `gemray::renderer::tonemap` -- costs tens of milliseconds even at 4K, not seconds)
/// and defers the expensive pass to a background thread via [`spawn_denoise_generation`],
/// swapping in its result on a LATER redraw once [`adopt_ready_denoise`] confirms it is
/// ready and still valid for the pose on screen. See [`PendingDenoiseGeneration`]'s doc
/// comment for why "still valid" is a structural pose-key check, not a timing
/// assumption.
fn redraw_from_accumulator(
    ui: &MainWindow,
    accumulator: &Arc<Mutex<Accumulator>>,
    render_ctx: &Arc<Mutex<RenderContext>>,
    state: &Arc<Mutex<Orchestrator>>,
    width: u32,
    height: u32,
) {
    let (buffer, samples_done) = {
        let acc = accumulator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (acc.buffer().to_vec(), acc.samples_done().max(1))
    };
    let (yaw, pitch, distance, planes, denoise_enabled) = {
        let ctx = render_ctx.lock().unwrap_or_else(PoisonError::into_inner);
        (
            ctx.yaw,
            ctx.pitch,
            ctx.distance,
            ctx.active_planes.clone(),
            ctx.denoise_enabled,
        )
    };
    let desired_key = GuideCache::key_for(width, height, yaw, pitch, distance, &planes);

    let mut orch = lock(state);

    let bytes = if denoise_enabled {
        // 1. A fresher generation just finished -- adopt it as the new "last known
        // good" for this pose (structural key check inside `adopt_ready_denoise`; see
        // its doc comment). The generation slot is now spent either way (taken if
        // ready, otherwise still legitimately in flight), so only clear it on adoption.
        if let Some(fresh) = adopt_ready_denoise(&desired_key, orch.pending_denoise_gen.as_ref()) {
            orch.pending_denoise_gen = None;
            orch.last_denoised = Some((desired_key.clone(), fresh));
        }

        // 2. Decide what to actually show this redraw: the freshest denoised frame for
        // THIS pose if we have one (even if a newer generation is still cooking -- a
        // few-samples-stale denoised image beats flickering back to noise every redraw
        // while an ~3.5s-at-4K pass runs), otherwise the cheap plain tonemap.
        match orch.last_denoised.as_ref() {
            Some((key, bytes)) if *key == desired_key => bytes.clone(),
            _ => {
                orch.last_denoised = None;
                tonemap_running_average(width, height, samples_done, &buffer)
            }
        }
    } else {
        orch.pending_denoise_gen = None;
        orch.last_denoised = None;
        tonemap_running_average(width, height, samples_done, &buffer)
    };

    // 3. Keep exactly one background denoise generation in flight per pose: dispatch a
    // fresh one whenever denoising is on and nothing is already running for the CURRENT
    // pose (covers "nothing dispatched yet", "the one we just adopted above", and "the
    // pose changed since the last dispatch" identically -- all three look the same:
    // `pending_denoise_gen`'s key doesn't match `desired_key`). Needs guides for this
    // pose to already be ready (cheap, ~288ms at 4K, already overlapped with the
    // network round trip via `start_remote_render`'s own background prepass) --
    // otherwise there is nothing to hand the background thread yet; a later redraw,
    // once the guide prepass lands, dispatches then.
    if denoise_enabled {
        let pending_matches = orch
            .pending_denoise_gen
            .as_ref()
            .is_some_and(|p| p.key == desired_key);
        if !pending_matches {
            let Orchestrator {
                guide_cache,
                pending_guide_gen,
                ..
            } = &mut *orch;
            if adopt_ready_guides(&desired_key, guide_cache, pending_guide_gen.as_ref()) {
                let guides = guide_cache
                    .ensure(width, height, yaw, pitch, distance, &planes)
                    .clone();
                orch.pending_denoise_gen = Some(spawn_denoise_generation(
                    desired_key,
                    DenoiseGenerationJob {
                        width,
                        height,
                        samples_done,
                        buffer,
                        guides,
                        yaw,
                        pitch,
                        distance,
                        planes,
                    },
                ));
            }
        }
    }
    drop(orch);

    let mut fb = crate::bridge::pixel_buffer::FramebufferTransfer::new(width, height);
    let image = fb.copy_from_gpu_slice(&bytes);
    ui.set_render_image(slint::Image::from_rgba8(image));
    ui.set_has_render(true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::guide_pass::generate_guide_buffers;
    use gemray::geometry::cuts::StandardGemCuts;

    /// A deliberately non-uniform synthetic accumulation buffer (mirrors
    /// `render_thread`'s own `denoise_and_tonemap_frame` tests) so a bug that skips
    /// filtering, or filters the wrong data, would visibly change the output.
    fn synthetic_noisy_buffer(width: u32, height: u32) -> Vec<Vec3> {
        (0..(width * height) as usize)
            .map(|i| {
                let x = (i % width as usize) as f32;
                Vec3::new(1.0 + x, 0.5 * x, 0.1f32.mul_add(-x, 2.0))
            })
            .collect()
    }

    #[test]
    fn render_merged_frame_skips_denoising_when_disabled() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let (width, height) = (6u32, 5u32);
        let buffer = synthetic_noisy_buffer(width, height);
        let mut guide_cache = GuideCache::new();
        let mut denoiser = AtrousDenoiser::new();
        let mut avg = Vec::new();
        let mut filtered = Vec::new();

        let bytes = render_merged_frame(
            AccumSnapshot {
                width,
                height,
                samples_done: 4,
                buffer: &buffer,
            },
            false,
            PoseAndGeometry {
                yaw: 0.60,
                pitch: 0.45,
                distance: 2.4,
                planes: &planes,
            },
            &mut guide_cache,
            &mut DenoiseScratch {
                denoiser: &mut denoiser,
                avg_color_buf: &mut avg,
                filtered_buf: &mut filtered,
            },
        );

        assert_eq!(bytes, tonemap_running_average(width, height, 4, &buffer));
        assert_eq!(
            guide_cache.generation(),
            0,
            "denoising disabled must never trigger the guide prepass at all -- the \
             user's toggle governs whether the (cheap, but not free) prepass runs, \
             exactly like it already governs the local path"
        );
    }

    #[test]
    fn render_merged_frame_reuses_guides_across_repeated_calls_with_an_unchanged_pose() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let (width, height) = (6u32, 5u32);
        let buffer = synthetic_noisy_buffer(width, height);
        let mut guide_cache = GuideCache::new();
        let mut denoiser = AtrousDenoiser::new();
        let mut avg = Vec::new();
        let mut filtered = Vec::new();

        // Three redraws, as if three `FRAME` events arrived from the same
        // in-progress remote render at an unchanged camera pose.
        for samples in [1u32, 2, 3] {
            render_merged_frame(
                AccumSnapshot {
                    width,
                    height,
                    samples_done: samples,
                    buffer: &buffer,
                },
                true,
                PoseAndGeometry {
                    yaw: 0.60,
                    pitch: 0.45,
                    distance: 2.4,
                    planes: &planes,
                },
                &mut guide_cache,
                &mut DenoiseScratch {
                    denoiser: &mut denoiser,
                    avg_color_buf: &mut avg,
                    filtered_buf: &mut filtered,
                },
            );
        }

        assert_eq!(
            guide_cache.generation(),
            1,
            "repeated redraws of the same in-progress remote render at an unchanged \
             pose must reuse the cached guide buffers, not regenerate them per frame"
        );
    }

    #[test]
    fn render_merged_frame_regenerates_guides_when_the_pose_changes() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let (width, height) = (6u32, 5u32);
        let buffer = synthetic_noisy_buffer(width, height);
        let mut guide_cache = GuideCache::new();
        let mut denoiser = AtrousDenoiser::new();
        let mut avg = Vec::new();
        let mut filtered = Vec::new();

        render_merged_frame(
            AccumSnapshot {
                width,
                height,
                samples_done: 1,
                buffer: &buffer,
            },
            true,
            PoseAndGeometry {
                yaw: 0.60,
                pitch: 0.45,
                distance: 2.4,
                planes: &planes,
            },
            &mut guide_cache,
            &mut DenoiseScratch {
                denoiser: &mut denoiser,
                avg_color_buf: &mut avg,
                filtered_buf: &mut filtered,
            },
        );
        render_merged_frame(
            AccumSnapshot {
                width,
                height,
                samples_done: 1,
                buffer: &buffer,
            },
            true,
            PoseAndGeometry {
                yaw: 0.90,
                pitch: 0.45,
                distance: 2.4,
                planes: &planes,
            },
            &mut guide_cache,
            &mut DenoiseScratch {
                denoiser: &mut denoiser,
                avg_color_buf: &mut avg,
                filtered_buf: &mut filtered,
            },
        );

        assert_eq!(
            guide_cache.generation(),
            2,
            "a camera-pose change (a fresh drag settling into a new remote request) \
             must invalidate the cached guides -- reusing a stale pose's guides would \
             misalign depth/normal/facet-id against the new image's geometry"
        );
    }

    /// `render_merged_frame` takes exactly one buffer and one `samples_done` -- never a
    /// per-contribution slice with its own smaller count -- and that TRUE merged total
    /// is what must drive the denoiser's convergence taper (`renderer::denoise`'s
    /// module docs on `sigma_color_effective`). This pins that the plumbing actually
    /// threads the real total through: at a converged sample count the output must be
    /// bit-identical to a plain tonemap (matching `render_thread`'s own
    /// `denoise_and_tonemap_frame_is_identity_at_high_sample_counts` guarantee for the
    /// local path). If this pipeline instead denoised a small per-contribution slice at
    /// its own (permanently small) sample count -- the "never on a partial
    /// contribution" mistake the task explicitly calls out -- this identity would never
    /// be reachable even once the real remote accumulation had converged.
    #[test]
    fn render_merged_frame_at_a_converged_sample_count_matches_a_plain_tonemap() {
        const HIGH_SAMPLE_COUNT: u32 = 50_000; // taper(50000) << taper_identity_epsilon

        let planes = StandardGemCuts::standard_round_brilliant();
        let (width, height) = (6u32, 5u32);
        let buffer = synthetic_noisy_buffer(width, height);

        let mut guide_cache = GuideCache::new();
        let mut denoiser = AtrousDenoiser::new();
        let mut avg = Vec::new();
        let mut filtered = Vec::new();
        let high_total_bytes = render_merged_frame(
            AccumSnapshot {
                width,
                height,
                samples_done: HIGH_SAMPLE_COUNT,
                buffer: &buffer,
            },
            true,
            PoseAndGeometry {
                yaw: 0.60,
                pitch: 0.45,
                distance: 2.4,
                planes: &planes,
            },
            &mut guide_cache,
            &mut DenoiseScratch {
                denoiser: &mut denoiser,
                avg_color_buf: &mut avg,
                filtered_buf: &mut filtered,
            },
        );

        assert_eq!(
            high_total_bytes,
            tonemap_running_average(width, height, HIGH_SAMPLE_COUNT, &buffer),
            "at a converged TRUE total, denoising the merged buffer must be an exact \
             no-op, just like the local path's readback"
        );
    }

    /// At a low (unconverged) sample count, `render_merged_frame` must run the SAME
    /// `render_thread::denoise_and_tonemap_frame` the local path uses, fed the SAME
    /// guides an independent `GuideCache`/`generate_guide_buffers` call for the
    /// identical pose/geometry would produce -- not some other, silently-diverged code
    /// path. This is a direct cross-check of the actual wiring (rather than asserting
    /// the output merely "looks filtered", which for a real gem's facet geometry over a
    /// tiny test image can legitimately be a no-op if every pixel in the crop lands on
    /// a distinct facet -- the hard per-facet edge-stop is deliberately that strict, see
    /// `renderer::denoise`'s module docs).
    #[test]
    fn render_merged_frame_at_a_low_sample_count_matches_denoise_and_tonemap_frame_directly() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let (width, height) = (6u32, 5u32);
        let buffer = synthetic_noisy_buffer(width, height);
        let (yaw, pitch, distance) = (0.60, 0.45, 2.4);
        let samples_done = 1;

        let mut guide_cache = GuideCache::new();
        let mut denoiser = AtrousDenoiser::new();
        let mut avg = Vec::new();
        let mut filtered = Vec::new();
        let actual = render_merged_frame(
            AccumSnapshot {
                width,
                height,
                samples_done,
                buffer: &buffer,
            },
            true,
            PoseAndGeometry {
                yaw,
                pitch,
                distance,
                planes: &planes,
            },
            &mut guide_cache,
            &mut DenoiseScratch {
                denoiser: &mut denoiser,
                avg_color_buf: &mut avg,
                filtered_buf: &mut filtered,
            },
        );

        // Independently reproduce what `render_merged_frame` should have done: generate
        // the guides for the same pose/geometry and denoise directly, with fresh
        // scratch state so nothing is shared with the call above.
        let camera = gemray::optics::raytracer::Camera::new(yaw, pitch, distance, 42.0);
        let guides = generate_guide_buffers(width, height, &camera, &planes);
        let mut expected_denoiser = AtrousDenoiser::new();
        let mut expected_avg = Vec::new();
        let mut expected_filtered = Vec::new();
        let expected = denoise_and_tonemap_frame(
            FirstHitSnapshot {
                width,
                height,
                current_sample_count: samples_done,
                accum_buffer: &buffer,
                first_hit_depth: &guides.depth,
                first_hit_normal: &guides.normal,
                first_hit_facet_id: &guides.facet_id,
            },
            &mut DenoiseScratch {
                denoiser: &mut expected_denoiser,
                avg_color_buf: &mut expected_avg,
                filtered_buf: &mut expected_filtered,
            },
        );

        assert_eq!(
            actual, expected,
            "render_merged_frame must denoise via the exact same guides/mechanism a \
             direct generate_guide_buffers + denoise_and_tonemap_frame call would \
             produce for the identical pose and geometry"
        );
    }

    // ---- Async guide generation: adopt_ready_guides ---------------------------------

    /// Builds a [`PendingGuideGeneration`] whose result is already sitting in its slot
    /// (as if the background thread had already finished), for `key`.
    fn ready_pending(key: GuideKey, buffers: GuideBuffers) -> PendingGuideGeneration {
        PendingGuideGeneration {
            key,
            cancel: Arc::new(AtomicBool::new(false)),
            result: Arc::new(Mutex::new(Some(buffers))),
        }
    }

    // ---- Background denoise generation / adopt_ready_denoise -----------------------

    /// Builds a [`PendingDenoiseGeneration`] whose result is already sitting in its slot
    /// (as if the background thread had already finished), for `key`.
    fn ready_denoise_pending(key: GuideKey, bytes: Vec<u8>) -> PendingDenoiseGeneration {
        PendingDenoiseGeneration {
            key,
            result: Arc::new(Mutex::new(Some(bytes))),
        }
    }

    /// Builds a [`PendingDenoiseGeneration`] that is still in flight -- no result yet --
    /// for `key`.
    fn in_flight_denoise_pending(key: GuideKey) -> PendingDenoiseGeneration {
        PendingDenoiseGeneration {
            key,
            result: Arc::new(Mutex::new(None)),
        }
    }

    #[test]
    fn adopt_ready_denoise_returns_none_when_nothing_is_pending() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let key = GuideCache::key_for(6, 5, 0.60, 0.45, 2.4, &planes);
        assert_eq!(adopt_ready_denoise(&key, None), None);
    }

    #[test]
    fn adopt_ready_denoise_returns_none_while_still_in_flight() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let key = GuideCache::key_for(6, 5, 0.60, 0.45, 2.4, &planes);
        let pending = in_flight_denoise_pending(key.clone());

        assert_eq!(
            adopt_ready_denoise(&key, Some(&pending)),
            None,
            "a generation that hasn't produced a result yet is 'not ready', not an error"
        );
    }

    #[test]
    fn adopt_ready_denoise_takes_the_result_once_ready_and_matching() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let key = GuideCache::key_for(6, 5, 0.60, 0.45, 2.4, &planes);
        let bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let pending = ready_denoise_pending(key.clone(), bytes.clone());

        assert_eq!(
            adopt_ready_denoise(&key, Some(&pending)),
            Some(bytes),
            "a finished generation for the exact pose on screen must be adopted"
        );
        assert_eq!(
            adopt_ready_denoise(&key, Some(&pending)),
            None,
            "the result is TAKEN, not cloned -- a second read of the same generation \
             must not resurrect it (mirrors PendingGuideGeneration's own `.take()`)"
        );
    }

    #[test]
    fn adopt_ready_denoise_rejects_a_result_for_a_different_pose() {
        let planes = StandardGemCuts::standard_round_brilliant();
        // The background generation finished, but for a DIFFERENT yaw than the pose
        // this frame is actually being rendered for -- e.g. it settled on the previous
        // pose right as a new drag started.
        let stale_key = GuideCache::key_for(6, 5, 0.10, 0.45, 2.4, &planes);
        let pending = ready_denoise_pending(stale_key, vec![9u8, 9, 9, 9]);

        let current_key = GuideCache::key_for(6, 5, 0.60, 0.45, 2.4, &planes);

        assert_eq!(
            adopt_ready_denoise(&current_key, Some(&pending)),
            None,
            "a background result for a pose other than the one being displayed must \
             never be adopted -- that would denoise this pose's radiance with another \
             pose's edges. This rejection is structural (a key comparison), not \
             timing-dependent."
        );
    }

    #[test]
    fn a_superseded_generations_late_result_does_not_overwrite_newer_guides() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let (width, height) = (6u32, 5u32);

        // The cache already holds the CURRENT (newer) pose's guides -- as if an earlier
        // redraw had already adopted them.
        let current_key = GuideCache::key_for(width, height, 0.90, 0.45, 2.4, &planes);
        let current_camera = Camera::new(0.90, 0.45, 2.4, 42.0);
        let current_buffers = generate_guide_buffers(width, height, &current_camera, &planes);
        let mut guide_cache = GuideCache::new();
        guide_cache.adopt(current_key.clone(), current_buffers.clone());

        // A generation kicked off for an OLDER pose finally finishes late, after being
        // superseded (its cancel flag was set, but it raced past the last check and
        // still produced a result).
        let old_key = GuideCache::key_for(width, height, 0.10, 0.45, 2.4, &planes);
        let old_camera = Camera::new(0.10, 0.45, 2.4, 42.0);
        let old_buffers = generate_guide_buffers(width, height, &old_camera, &planes);
        let superseded = ready_pending(old_key, old_buffers);

        let adopted = adopt_ready_guides(&current_key, &mut guide_cache, Some(&superseded));

        assert!(
            adopted,
            "the cache already had the correct guides for the current pose"
        );
        assert!(
            guide_cache.matches_key(&current_key),
            "a superseded generation's late result for an OLDER pose must never \
             overwrite the current pose's already-adopted guides"
        );
        assert_eq!(
            guide_cache
                .ensure(width, height, 0.90, 0.45, 2.4, &planes)
                .depth,
            current_buffers.depth,
            "the buffers actually in the cache must still be the current pose's, not \
             the stale generation's"
        );
    }
}

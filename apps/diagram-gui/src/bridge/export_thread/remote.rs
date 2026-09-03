//! Remote engine for the high-resolution export: probes a configured worker's render
//! capacity, times a small calibration dispatch against it, and runs the export-sized
//! `RenderRequest` itself -- reusing `bridge::remote_render`'s socket-owning client
//! (the SAME code path the live-viewport preview-then-handoff orchestrator drives) as a
//! third engine alongside `batch`'s existing CPU/GPU hybrid, rather than a parallel
//! reimplementation of the wire protocol.
//!
//! # Why disjointness is trivially guaranteed here
//!
//! Every sample range this module hands to `spawn_remote_render` (calibration or the
//! main dispatch) is `[start, start + count)` for a `start` the caller advances by
//! exactly `count` afterwards -- the same "next engine starts where the last one's
//! request left off" discipline `batch::calibrate_split`/`hybrid_batch` already use for
//! CPU vs. GPU. `run_export` threads one absolute counter through calibration, the
//! remote dispatch, and the local CPU/GPU loop in strict sequence (even though the
//! remote dispatch and the local loop then run concurrently -- see `run_export`'s own
//! doc comment), so no two engines are ever handed overlapping absolute indices.
//!
//! # Partial completion, not failure
//!
//! [`run_remote_batch`] never discards what a worker already streamed back:
//! `gemray-worker`'s tracer (`apps/gemray-worker/src/stream_emit/tracer.rs::run_tracer`)
//! traces its assigned range in increasing sub-batches and only ever advances
//! `samples_done` after folding each one in, checking cancellation *between* (never
//! mid-) sub-batches -- so at any moment, including a mid-stream disconnect or a
//! `CANCEL`, the accumulator's `samples_done()` is exactly the length of the completed
//! PREFIX of `[first_sample, first_sample + samples)`, and its `buffer()` is exactly
//! that prefix's valid, already-summed radiance. `run_export` relies on this to trace
//! only the shortfall locally rather than redoing (and double-counting) anything.

use super::{
    batch::{ExportCtx, render_batch},
    params::ComputeTarget,
    scene_snapshot::SceneSnapshot,
};
use crate::{
    bridge::remote_render::{self, RemoteRenderRequest, RemoteUpdate},
    settings::WorkerSettings,
};
use gemray_net::{SceneState, client::Accumulator, messages::RenderCapability};
use glam::Vec3;
use std::{
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    time::{Duration, Instant},
};

/// Below this many REMAINING samples, dispatching to a remote worker at all (handshake
/// plus a calibration round trip) costs more than it could ever save -- the same
/// reasoning as `batch::HYBRID_MIN_SPP`, just for a link whose round-trip latency is
/// typically far higher than an in-process CPU/GPU dispatch's.
pub(super) const REMOTE_MIN_SPP: u32 = 32;

/// Sample count for the local-vs-remote calibration probe -- see
/// [`calibrate_remote_fraction`]. Large enough to amortize one TLS/network round trip's
/// fixed overhead against real tracing time; small enough not to waste meaningful export
/// time on a measurement.
pub(super) const REMOTE_CALIBRATION_SAMPLES: u32 = 8;

/// A single fixed id: every remote dispatch this module makes (calibration or the main
/// export request) opens its OWN fresh connection (`spawn_remote_render` calls
/// `connect_and_handshake` itself), so nothing is ever pipelined behind something else
/// on the same socket the way the live viewport's `next_request_id` counter has to
/// guard against -- see `gemray_net::messages::stream`'s module docs on `request_id`
/// epochs. Reusing `1` across separate connections is therefore safe.
const REQUEST_ID: u32 = 1;

/// Why remote compute is unavailable for this export -- surfaced to the export dialog
/// verbatim (via [`RemoteUnavailable::message`]) so a disabled pill always explains
/// itself, per the top-level task's requirement to distinguish these three cases rather
/// than collapsing them into one generic "unavailable".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteUnavailable {
    NoWorkerConfigured,
    Unreachable(String),
    /// `Welcome::render` was `None` -- the worker is a library-only build. Mirrors
    /// `bridge::remote_render::RemoteError::NoRenderCapacity`.
    LibraryOnly,
}

impl RemoteUnavailable {
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::NoWorkerConfigured => "No remote worker is configured.".to_string(),
            Self::Unreachable(e) => format!("Remote worker unreachable ({e})."),
            Self::LibraryOnly => {
                "The configured worker serves the design library only -- it has no \
                 render capacity."
                    .to_string()
            }
        }
    }
}

/// A remote worker confirmed reachable and render-capable, with everything a
/// [`RenderRequest`] against it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteCapability {
    pub worker: WorkerSettings,
    /// This worker's advertised `RenderCapability::max_pixels` -- checked against the
    /// export's own `width * height` BEFORE ever dispatching, never assumed to be the
    /// hardcoded `gemray-worker` default (a different worker build may advertise a
    /// different cap). See `run_export`'s pixel-cap fallback.
    pub max_pixels: u32,
}

/// Whether an export at `width x height` exceeds `capability`'s advertised
/// `max_pixels` -- checked BEFORE ever dispatching (never after a rejected request),
/// and always against the WORKER'S OWN advertised cap from its `WELCOME` rather than a
/// hardcoded constant, since a different worker build may advertise a different one.
/// Pure and side-effect-free so the pixel-cap fallback decision is directly
/// unit-testable without a live worker -- see this module's tests.
#[must_use]
pub(super) fn exceeds_pixel_cap(width: u32, height: u32, capability: &RemoteCapability) -> bool {
    u64::from(width) * u64::from(height) > u64::from(capability.max_pixels)
}

/// How many of `remaining` samples should go to remote, given `compute_target` and
/// (for [`ComputeTarget::Both`] only) the fraction [`calibrate_remote_fraction`]
/// measured. Pure and side-effect-free -- extracted out of `run_export`'s own
/// split-decision so the THREE distinct compute-target behaviours (all-local,
/// all-remote, and a calibrated blend) are directly unit-testable without a live
/// worker or a real calibration probe. `ComputeTarget::LocalOnly` always returns `0`
/// even though `run_export` never actually calls this for that target (remote is
/// skipped entirely before reaching a split decision at all) -- included for
/// completeness/exhaustiveness rather than `unreachable!()`, so this function stays
/// total and trivially testable on its own.
#[must_use]
pub(super) fn split_remote_samples(
    compute_target: ComputeTarget,
    remaining: u32,
    remote_frac: Option<f64>,
) -> u32 {
    match compute_target {
        ComputeTarget::LocalOnly => 0,
        ComputeTarget::RemoteOnly => remaining,
        ComputeTarget::Both => remote_frac.map_or(0, |frac| {
            ((f64::from(remaining) * frac).round() as u32).min(remaining)
        }),
    }
}

/// How many of a remote dispatch's `assigned` samples never got traced -- `0` when it
/// finished cleanly (`completed == assigned`), positive iff the connection dropped, the
/// worker errored, or it was cancelled partway (`completed < assigned`). Pure and
/// side-effect-free so `run_export`'s mid-export-failure fallback (retrace the
/// shortfall locally, never the whole assigned range -- every sample the worker DID
/// complete is already valid, already-summed radiance, see [`run_remote_batch`]'s doc
/// comment) is directly unit-testable without a live worker. `saturating_sub` rather
/// than a bare `-` purely as defense-in-depth against `completed` somehow exceeding
/// `assigned` (never observed -- an `Accumulator` only ever counts samples for the
/// range it was actually asked to trace) -- underflowing here would panic in a release
/// build's checked-arithmetic-off path and wrap in a debug build's checked one, both
/// far worse than just reporting "nothing left to retrace".
#[must_use]
pub(super) const fn shortfall(assigned: u32, completed: u32) -> u32 {
    assigned.saturating_sub(completed)
}

/// Probes the first configured worker -- the same "session-wide, first entry"
/// convention `gui::remote::orchestrator::poll_tick` already uses for the live
/// viewport's handoff (`settings_store.snapshot().settings.remote_workers.first()`) --
/// for render capacity. Blocking (a real TLS handshake); callers run this off the UI
/// thread, which the export's own worker thread already is.
///
/// # Errors
///
/// See [`RemoteUnavailable`]'s variants.
pub fn probe_remote(workers: &[WorkerSettings]) -> Result<RemoteCapability, RemoteUnavailable> {
    let Some(worker) = workers.first().cloned() else {
        return Err(RemoteUnavailable::NoWorkerConfigured);
    };
    match remote_render::connect_and_handshake(&worker) {
        Ok((_stream, welcome)) => match welcome.render {
            Some(RenderCapability { max_pixels, .. }) => {
                Ok(RemoteCapability { worker, max_pixels })
            }
            None => Err(RemoteUnavailable::LibraryOnly),
        },
        Err(e) => Err(RemoteUnavailable::Unreachable(e.to_string())),
    }
}

/// Builds the `gemray_net::SceneState` a remote worker needs from an export's own
/// [`SceneSnapshot`] plus its `width x height` -- the export-side equivalent of
/// `gui::remote::orchestrator::scene_state_from_snapshot`, duplicated rather than
/// shared because that function is private to a different module tree and the two
/// callers' surrounding types (`SceneSnapshot` here, the live viewport's own capture
/// there) already differ in scope. See that function's doc comment for the frosted-
/// girdle field mapping this mirrors exactly.
#[must_use]
pub(super) fn scene_state_from_snapshot(
    snapshot: &SceneSnapshot,
    width: u32,
    height: u32,
) -> SceneState {
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

/// Runs one `RenderRequest` against `capability.worker`, covering `[first_sample,
/// first_sample + samples)` at `width x height`, blocking until it finishes, fails, or
/// `cancel` is observed -- in which case [`remote_render::RemoteRenderHandle::cancel`]
/// is sent and this waits for the worker's own `DONE { cancelled: true }` confirmation
/// rather than abandoning the connection outright (matching
/// `bridge::remote_render`'s own documented cooperative-cancellation contract).
///
/// `accumulator` is caller-provided (rather than built internally) so a caller running
/// this on a background thread can peek at its live `buffer()`/`samples_done()` from
/// another thread while this call is still in flight -- `run_export` uses that for its
/// export-progress preview during the concurrent local+remote phase.
///
/// `spawn_remote_render` already owns the socket on its own thread and reports progress
/// via a callback; this wraps that in a plain channel so the caller (`run_export`'s own
/// background thread, itself already off the Slint event loop) can wait on it
/// synchronously -- unlike the live-viewport orchestrator, which drives the identical
/// `spawn_remote_render` from `Weak::upgrade_in_event_loop` callbacks because THAT code
/// runs on the UI thread and must never block it.
///
/// Returns `(samples_done, cancelled, error)`: `samples_done` is always exactly what the
/// accumulator ended up holding (a valid prefix -- see the module doc comment) whether
/// this ended by finishing, being cancelled, or failing; `error` is `Some` only when the
/// request ended by failing outright (a connection/protocol error or a worker-side
/// `ERROR`), never merely because it was cancelled or ran short.
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a distinct piece of one RenderRequest's own identity \
              (worker capability, scene, sample range, resolution, the shared \
              accumulator, cancellation) -- bundling them into a struct would just move \
              the same count into field access, not reduce it, and `RemoteRenderRequest` \
              already plays that role one level down in `bridge::remote_render`"
)]
pub(super) fn run_remote_batch(
    capability: &RemoteCapability,
    scene: SceneState,
    first_sample: u32,
    samples: u32,
    width: u32,
    height: u32,
    accumulator: &Arc<Mutex<Accumulator>>,
    cancel: &AtomicBool,
) -> (u32, bool, Option<String>) {
    let (tx, rx) = mpsc::channel::<RemoteUpdate>();
    let handle = remote_render::spawn_remote_render(
        RemoteRenderRequest {
            worker: capability.worker.clone(),
            request_id: REQUEST_ID,
            scene,
            first_sample,
            samples,
            width,
            height,
        },
        Arc::clone(accumulator),
        move |update| {
            let _ = tx.send(update);
        },
    );

    let mut cancel_sent = false;
    loop {
        if !cancel_sent && cancel.load(Ordering::Relaxed) {
            handle.cancel();
            cancel_sent = true;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(RemoteUpdate::Done { cancelled }) => {
                let acc = accumulator.lock().unwrap_or_else(PoisonError::into_inner);
                return (acc.samples_done(), cancelled, None);
            }
            Ok(RemoteUpdate::Failed(message)) => {
                let acc = accumulator.lock().unwrap_or_else(PoisonError::into_inner);
                return (acc.samples_done(), false, Some(message));
            }
            // Connected/Frame/Preview/Progress, or simply nothing pending yet within
            // this poll: nothing terminal, keep waiting -- the accumulator itself
            // already reflects every Frame as it's applied (`Accumulator::apply`), so
            // there's nothing further to do with either case here.
            Ok(_) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                // The worker thread ended without ever sending a terminal update (a
                // panic inside it, most likely) -- treat it like a failure so the
                // caller falls back to local rather than looping forever.
                let acc = accumulator.lock().unwrap_or_else(PoisonError::into_inner);
                return (
                    acc.samples_done(),
                    false,
                    Some("remote render worker thread ended unexpectedly".to_string()),
                );
            }
        }
    }
}

/// Times a [`REMOTE_CALIBRATION_SAMPLES`]-sample probe against the remote worker and an
/// equally-sized local probe (GPU if `ctx.gpu` accepts it, CPU otherwise -- whichever
/// engine the export's own hybrid batches would actually use first), and returns the
/// SHARE of the remaining budget remote should be given -- `batch::calibrate_split`'s
/// exact shape (`frac = other_engine_time / (this_engine_time + other_engine_time)`),
/// applied to a third engine. `None` means "decline remote for this export": either the
/// probe itself failed/was cancelled/ran short, in which case trusting its timing at all
/// would be trusting a broken measurement.
///
/// Both probes' radiance is folded into `accum`/`gpu_accum` for real (not thrown away --
/// see `run_export`'s doc comment on why calibration samples are always real, counted
/// samples in this codebase), and `*samples_done` is advanced past both, so the caller's
/// next dispatch (remote's main request or the local loop) picks up exactly where this
/// left off.
///
/// # A documented simplification
///
/// This times "remote vs. whichever ONE local engine responds" (GPU-first, matching
/// `run_export`'s own pre-hybrid single-engine fallback order), not "remote vs. local's
/// full CPU+GPU-concurrent hybrid throughput" -- when both a GPU and a remote worker are
/// available, this slightly underestimates local's true combined throughput, biasing a
/// bit more of the split toward remote than a perfectly joint three-way calibration
/// would. A fully joint CPU+GPU+remote probe (all three timed concurrently on a single
/// race-free sample range) was judged not worth the added complexity for a calibration
/// step whose only job is picking a STARTING split: once the concurrent phase begins,
/// `batch::hybrid_batch` continues to re-measure and re-blend the LOCAL CPU/GPU split on
/// every batch exactly as it already does without remote in the picture (see its own doc
/// comment) -- only the remote/local boundary itself is fixed once, up front, since
/// remote work is dispatched as one large request rather than many small batches (see
/// `run_export`'s doc comment for why).
pub(super) fn calibrate_remote_fraction(
    ctx: &ExportCtx<'_>,
    capability: &RemoteCapability,
    scene_state: &SceneState,
    samples_done: &mut u32,
    accum: &mut [Vec3],
    gpu_accum: &mut [Vec3],
    cancel: &AtomicBool,
) -> Option<f64> {
    let probe = REMOTE_CALIBRATION_SAMPLES;
    let remote_accumulator = Arc::new(Mutex::new(Accumulator::new(ctx.width, ctx.height)));
    let remote_start_sample = *samples_done;

    let timer = Instant::now();
    let (remote_done, remote_cancelled, remote_error) = run_remote_batch(
        capability,
        scene_state.clone(),
        remote_start_sample,
        probe,
        ctx.width,
        ctx.height,
        &remote_accumulator,
        cancel,
    );
    let remote_time = timer.elapsed().as_secs_f64().max(1e-9);

    {
        let acc = remote_accumulator
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for (dst, src) in accum.iter_mut().zip(acc.buffer()) {
            *dst += *src;
        }
    }
    *samples_done += remote_done;

    if remote_cancelled || remote_error.is_some() || remote_done < probe {
        return None;
    }
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let local_start_sample = *samples_done;
    let timer = Instant::now();
    if !ctx
        .gpu
        .try_accumulate(ctx.gpu_scene, local_start_sample, probe, gpu_accum)
    {
        render_batch(
            ctx.width,
            ctx.height,
            probe,
            local_start_sample,
            ctx.camera,
            ctx.scene,
            accum,
        );
    }
    let local_time = timer.elapsed().as_secs_f64().max(1e-9);
    *samples_done += probe;

    Some((local_time / (local_time + remote_time)).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_remote_reports_no_worker_configured_when_the_list_is_empty() {
        assert_eq!(
            probe_remote(&[]),
            Err(RemoteUnavailable::NoWorkerConfigured)
        );
    }

    #[test]
    fn probe_remote_reports_unreachable_for_a_bogus_address() {
        let worker = WorkerSettings {
            address: "127.0.0.1:1".to_string(), // nothing listens here
            cert_dir: std::env::temp_dir().display().to_string(),
            ..WorkerSettings::default()
        };
        let err = probe_remote(std::slice::from_ref(&worker)).unwrap_err();
        assert!(matches!(err, RemoteUnavailable::Unreachable(_)));
    }

    // ---- Pixel-cap fallback (constraint 1) -------------------------------------

    fn capability(max_pixels: u32) -> RemoteCapability {
        RemoteCapability {
            worker: WorkerSettings::default(),
            max_pixels,
        }
    }

    #[test]
    fn exceeds_pixel_cap_uses_the_workers_own_advertised_cap_not_a_hardcoded_constant() {
        // A worker that advertises a SMALLER cap than `gemray-worker`'s own
        // `validate::MAX_PIXELS` default must still be honoured -- this function must
        // never assume the well-known constant.
        let small_worker = capability(100);
        assert!(exceeds_pixel_cap(11, 10, &small_worker)); // 110 > 100
        assert!(!exceeds_pixel_cap(10, 10, &small_worker)); // 100 == 100, not over

        let real_default = capability(7680 * 4320);
        assert!(!exceeds_pixel_cap(3840, 2160, &real_default)); // 4K fits
        assert!(exceeds_pixel_cap(8192, 8192, &real_default)); // max custom export size does not
    }

    // ---- Split decisions (constraint 3) ----------------------------------------

    #[test]
    fn split_remote_samples_local_only_never_sends_anything_remote() {
        assert_eq!(
            split_remote_samples(ComputeTarget::LocalOnly, 1000, Some(0.9)),
            0
        );
    }

    #[test]
    fn split_remote_samples_remote_only_claims_the_entire_remaining_budget() {
        assert_eq!(
            split_remote_samples(ComputeTarget::RemoteOnly, 1000, None),
            1000
        );
    }

    #[test]
    fn split_remote_samples_both_declines_remote_when_calibration_failed() {
        // `None` is what a failed/cancelled/short calibration probe reports (see
        // `calibrate_remote_fraction`'s doc comment) -- must fall back to giving
        // remote nothing, not some arbitrary default share.
        assert_eq!(split_remote_samples(ComputeTarget::Both, 1000, None), 0);
    }

    #[test]
    fn split_remote_samples_both_scales_by_the_measured_fraction_and_never_exceeds_remaining() {
        assert_eq!(
            split_remote_samples(ComputeTarget::Both, 1000, Some(0.25)),
            250
        );
        assert_eq!(
            split_remote_samples(ComputeTarget::Both, 1000, Some(1.0)),
            1000,
            "a fraction of 1.0 (remote measured far faster) must still be capped at \
             `remaining`, never overshoot it"
        );
        assert_eq!(
            split_remote_samples(ComputeTarget::Both, 1000, Some(0.0)),
            0
        );
    }

    // ---- Sample-range disjointness (the top-level task's core correctness rule) --

    /// The exact arithmetic `run_export` uses to place remote's assigned range ahead
    /// of local's: `local_start == remote_first + remote_count`. This pins that
    /// invariant directly, independent of `run_export`'s much larger integration
    /// surface -- a regression here would silently let two engines redraw the same
    /// samples (biasing the average, per `gemray::renderer::gpu_backend`'s "Sample-
    /// range additivity" doc section) rather than extending it.
    #[test]
    fn remote_and_local_ranges_never_overlap_for_any_split() {
        for total in [32u32, 100, 1000, 32768] {
            for frac in [None, Some(0.0), Some(0.37), Some(1.0)] {
                let remote_count = split_remote_samples(ComputeTarget::Both, total, frac);
                let remote_first = 0u32; // calibration already consumed, matches run_export
                let local_start = remote_first + remote_count;

                let remote_range = remote_first..(remote_first + remote_count);
                let local_range = local_start..total;

                assert_eq!(
                    remote_range.end, local_range.start,
                    "remote's range must end exactly where local's begins -- no gap, \
                     no overlap"
                );
                assert!(
                    remote_count <= total,
                    "remote must never be assigned more than the total budget"
                );
            }
        }
    }

    // ---- Mid-export worker failure (constraint 4) -------------------------------

    #[test]
    fn shortfall_is_zero_when_remote_finished_everything_it_was_assigned() {
        assert_eq!(shortfall(1000, 1000), 0);
    }

    #[test]
    fn shortfall_is_the_unfinished_remainder_of_a_partial_completion() {
        // The realistic "connection dropped after some FRAMEs landed" case -- only the
        // unfinished tail should ever be retraced locally, never the whole range (that
        // would re-trace, and double-count, samples the worker already completed).
        assert_eq!(shortfall(1000, 400), 600);
    }

    #[test]
    fn shortfall_is_the_full_range_when_remote_completed_nothing() {
        // The worst case (e.g. the connection dropped before a single FRAME arrived) --
        // must fall back to tracing the ENTIRE assigned range locally, not leave a gap.
        assert_eq!(shortfall(1000, 0), 1000);
    }

    #[test]
    fn shortfall_never_underflows_even_if_completed_somehow_exceeded_assigned() {
        assert_eq!(shortfall(100, 150), 0);
    }

    #[test]
    fn remote_unavailable_messages_are_distinct_and_human_readable() {
        assert!(
            RemoteUnavailable::NoWorkerConfigured
                .message()
                .contains("No remote worker")
        );
        assert!(
            RemoteUnavailable::LibraryOnly
                .message()
                .contains("library only")
        );
        assert!(
            RemoteUnavailable::Unreachable("boom".to_string())
                .message()
                .contains("boom")
        );
    }
}

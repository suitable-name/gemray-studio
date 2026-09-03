//! The free-running tracer: [`run_tracer`] traces the requested sample range in
//! adaptively-sized sub-batches, folding each into a shared accumulation buffer --
//! never touching the socket itself. See `crate::serve`'s module docs for why that
//! separation is the whole point.

use super::{emitter::PendingDelta, sizing::next_batch_size};
use crate::render_core;
use gemray::renderer::gpu_backend::GpuBackend;
use gemray_net::SceneState;
use glam::Vec3;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Instant,
};

/// Shared between `run_tracer` (the sole writer) and `run_stream`'s emitter (the sole
/// reader) via a `Mutex` -- see `run_stream`'s doc comment for why a lock held only for
/// these brief updates/drains never stalls either side for long.
pub(super) struct SharedState {
    pub(super) pending_delta: PendingDelta,
    pub(super) running_total: Vec<Vec3>,
    pub(super) samples_done: u32,
    pub(super) finished: bool,
    /// Set instead of `finished` looking any different on its own -- a caller-supplied
    /// (but validation-passing) scene can still make `gemray`'s tracer panic on some
    /// pathological geometry; see `serve`'s own module docs on why tracing runs inside
    /// `catch_unwind` as defense in depth. `run_stream` checks this once `finished` is
    /// set and reports [`StreamOutcome::TracePanicked`] instead of a normal
    /// `FRAME`+`DONE` sequence when it's set.
    pub(super) panicked: bool,
}

/// One tracer job's fixed inputs (the whole requested sample range plus enough of the
/// request to trace it), bundled so [`run_tracer`] stays within clippy's argument-count
/// limit -- mirrors how `gemray::renderer::gpu::GpuFrameScene` bundles a GPU call's own
/// inputs for the same reason.
pub(super) struct TracerJob {
    pub(super) scene: SceneState,
    pub(super) first_sample: u32,
    pub(super) samples: u32,
    pub(super) threads: usize,
}

/// Runs the tracer side: free-runs over `[job.first_sample, job.first_sample +
/// job.samples)` in adaptively-sized sub-batches (see [`next_batch_size`]), folding each
/// one into `state.pending_delta` and `state.running_total`, checking `cancel` between
/// (never mid-) sub-batches. Never touches the socket -- see `serve`'s module docs on
/// why that separation is the whole point.
///
/// Each sub-batch traces through [`render_core::trace_samples_with_gpu`], which prefers
/// `gpu` and falls back to the CPU tracer whenever it declines -- a GPU dispatch is
/// blocking, so it is exactly one more (synchronous) way to produce a sub-batch, and the
/// same `cancel` check between batches bounds its cancellation latency exactly as it
/// already bounds the CPU tracer's.
///
/// Runs the whole loop inside `catch_unwind`: a panic anywhere in `gemray`'s tracer
/// (unanticipated pathological, but validation-passing, geometry) is caught here rather
/// than taking down this thread silently mid-request, so `state.panicked` reliably gets
/// set for `run_stream` to notice.
pub(super) fn run_tracer(
    job: TracerJob,
    gpu: &GpuBackend,
    state: &Arc<Mutex<SharedState>>,
    cancel: &Arc<AtomicBool>,
    progress_tx: &mpsc::Sender<()>,
) {
    let TracerJob {
        scene,
        first_sample,
        samples,
        threads,
    } = job;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut produced: u32 = 0;
        let mut batch_size: u32 = 1;

        while produced < samples {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let this_batch = batch_size.min(samples - produced);
            let batch_first = first_sample + produced;

            let start = Instant::now();
            let buf =
                render_core::trace_samples_with_gpu(gpu, &scene, batch_first, this_batch, threads);
            let elapsed = start.elapsed();

            {
                let mut guard = state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                guard.pending_delta.add(batch_first, this_batch, &buf);
                for (total, contribution) in guard.running_total.iter_mut().zip(&buf) {
                    *total += *contribution;
                }
                produced += this_batch;
                guard.samples_done = produced;
            }
            let _ = progress_tx.send(());

            batch_size = next_batch_size(batch_size, elapsed);
        }
    }));

    let mut guard = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.finished = true;
    guard.panicked = result.is_err();
    drop(guard);
    let _ = progress_tx.send(());
}

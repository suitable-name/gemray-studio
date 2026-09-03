//! The dedicated display thread (Task R6): owns the À-Trous denoiser, its scratch
//! buffers, and the `FramebufferTransfer`, so denoise+tonemap+push never blocks the
//! render thread's own trace loop.
//!
//! Split out of `bridge::render_thread` purely to keep that module (already sizeable)
//! from growing further. See `spawn_render_thread`'s own send site for how the loop
//! decides when to hand off a cycle.

use super::{
    denoise::{
        DenoiseScratch, FirstHitSnapshot, denoise_and_tonemap_frame, tonemap_running_average,
    },
    push_frame_to_ui,
};
use crate::bridge::pixel_buffer::FramebufferTransfer;
use gemray::{color::metrics::GemOpticalMetrics, renderer::denoise::AtrousDenoiser};
use glam::Vec3;
use slint::{ComponentHandle, Weak};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    thread,
    time::Duration,
};

/// One frame's gemological metrics, angular-profile graphs, and the camera pitch its
/// polar plot needs -- bundled because [`DisplayWork::fill`] and [`push_frame_to_ui`]
/// both just copy these five values straight through from `compute_or_reuse_metrics`'s
/// result to the UI update callback, never touching them individually. `Copy`, like
/// every field it bundles, so reading `work.metrics_snapshot` out of a `&DisplayWork`
/// (e.g. at its `push_frame_to_ui` call site) copies rather than partially moving out of
/// `DisplayWork`, which is sent back to the render loop's pool right after.
#[derive(Clone, Copy)]
pub(super) struct FrameMetricsSnapshot {
    pub(super) metrics: GemOpticalMetrics,
    pub(super) graph_brilliance: [f32; 19],
    pub(super) graph_extinction: [f32; 19],
    pub(super) graph_windowing: [f32; 19],
    pub(super) cam_pitch_deg: f32,
}

/// One display cycle's worth of inputs, copied out of the render loop's own buffers
/// under no lock (the render loop only ever hands off a [`DisplayWork`] it isn't
/// touching anymore -- see [`DisplayHandle::send`]). Reused across cycles via
/// [`DisplayHandle::reclaim`] instead of reallocated every time.
pub(super) struct DisplayWork {
    /// Stamped with [`DisplayHandle::current_generation`]'s value at snapshot time.
    /// The display thread drops (never pushes) a result whose `generation` no longer
    /// matches the shared counter's CURRENT value -- see `spawn_display_thread`'s loop
    /// body -- which is what keeps a `dirty`/resize reset from ever letting a stale
    /// frame from the old pose reach the screen after the reset.
    generation: u64,
    width: u32,
    height: u32,
    current_sample_count: u32,
    denoise_enabled: bool,
    accum: Vec<Vec3>,
    depth: Vec<f32>,
    normal: Vec<Vec3>,
    facet_id: Vec<i32>,
    metrics_snapshot: FrameMetricsSnapshot,
}

impl DisplayWork {
    /// A fresh, empty work buffer -- allocates nothing until [`Self::fill`] actually
    /// populates it; `fill`'s `clear` + `extend_from_slice` then reuses whatever
    /// capacity a previous cycle already grew each `Vec` to, rather than dropping and
    /// reallocating every cycle.
    const fn empty() -> Self {
        Self {
            generation: 0,
            width: 0,
            height: 0,
            current_sample_count: 0,
            denoise_enabled: true,
            accum: Vec::new(),
            depth: Vec::new(),
            normal: Vec::new(),
            facet_id: Vec::new(),
            metrics_snapshot: FrameMetricsSnapshot {
                metrics: GemOpticalMetrics {
                    brilliance_pct: 0.0,
                    fire_index: 0.0,
                    scintillation_pct: 0.0,
                    windowing_pct: 0.0,
                    extinction_pct: 0.0,
                },
                graph_brilliance: [0.0; 19],
                graph_extinction: [0.0; 19],
                graph_windowing: [0.0; 19],
                cam_pitch_deg: 0.0,
            },
        }
    }

    /// Overwrites every field from this frame's live render-loop state. See
    /// [`Self::empty`]'s doc comment for why this reuses -- rather than reallocates --
    /// `accum`/`depth`/`normal`/`facet_id`'s heap buffers.
    pub(super) fn fill(
        &mut self,
        generation: u64,
        denoise_enabled: bool,
        frame: FirstHitSnapshot<'_>,
        metrics_snapshot: FrameMetricsSnapshot,
    ) {
        self.generation = generation;
        self.width = frame.width;
        self.height = frame.height;
        self.current_sample_count = frame.current_sample_count;
        self.denoise_enabled = denoise_enabled;
        self.accum.clear();
        self.accum.extend_from_slice(frame.accum_buffer);
        self.depth.clear();
        self.depth.extend_from_slice(frame.first_hit_depth);
        self.normal.clear();
        self.normal.extend_from_slice(frame.first_hit_normal);
        self.facet_id.clear();
        self.facet_id.extend_from_slice(frame.first_hit_facet_id);
        self.metrics_snapshot = metrics_snapshot;
    }
}

/// The render loop's handle onto the dedicated display thread: everything it needs to
/// hand off a cycle, reclaim a pooled buffer, and coordinate the generation/in-flight
/// state that keeps at most one cycle running at a time and a reset from ever letting a
/// stale frame through. See [`spawn_display_thread`].
pub(super) struct DisplayHandle {
    work_tx: Sender<DisplayWork>,
    pool_rx: Receiver<DisplayWork>,
    in_flight: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
}

impl DisplayHandle {
    /// `true` while the display thread is still processing a previously sent cycle.
    /// The render loop must not send another cycle while this holds -- with one
    /// exception: the frame that reaches `target_samples`, which must always be
    /// displayed exactly once, so its caller waits this out instead of skipping (see
    /// `spawn_render_thread`'s send site).
    pub(super) fn busy(&self) -> bool {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Invalidates any currently in-flight (or already queued) display result -- call
    /// whenever the render loop resets progressive accumulation (`dirty` or a dimension
    /// change), so a cycle snapshotted from the OLD pose is dropped by the display
    /// thread rather than displayed after the reset.
    pub(super) fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Reclaims a pooled [`DisplayWork`] freed by a completed cycle (its heap buffers
    /// still at whatever capacity they last grew to), or allocates a fresh empty one if
    /// none is available yet (only possible for the first cycle or two, before the
    /// display thread has completed and returned any).
    pub(super) fn reclaim(&self) -> DisplayWork {
        self.pool_rx
            .try_recv()
            .unwrap_or_else(|_| DisplayWork::empty())
    }

    /// Hands `work` off to the display thread and marks a cycle in flight. Callers must
    /// have already confirmed [`Self::busy`] is `false` (directly, or by spin-waiting
    /// it out for the one must-not-skip convergence case).
    pub(super) fn send(&self, work: DisplayWork) {
        self.in_flight.store(true, Ordering::Release);
        // A send only fails once the display thread's receiver is gone, which only
        // happens after this very `DisplayHandle` (and the render loop that owns it)
        // has already been dropped -- nothing left to hand work to, and nothing left
        // to observe the error either.
        let _ = self.work_tx.send(work);
    }
}

/// Spawns the dedicated display thread and returns the render loop's [`DisplayHandle`]
/// onto it.
///
/// The display thread owns the `AtrousDenoiser`, its scratch buffers (`avg_color_buf`/
/// `filtered_buf`), and the `FramebufferTransfer` (reallocated here, exactly like the
/// render loop's own accumulation buffers, whenever a cycle's `width`/`height` differ
/// from the last one this thread processed) -- denoising, tone-mapping, and pushing to
/// the UI all happen off the render thread's own trace loop.
///
/// Runs until `work_tx` (the render loop's send half, owned by the returned
/// [`DisplayHandle`]) is dropped -- which happens when `spawn_render_thread`'s loop
/// exits on `running == false` and the `DisplayHandle` local goes out of scope with it
/// -- at which point `work_rx.recv()` returns `Err` and this thread ends cleanly.
pub(super) fn spawn_display_thread<T, F, M>(
    ui_weak: Weak<T>,
    update_image: F,
    update_metrics: M,
) -> DisplayHandle
where
    T: ComponentHandle + 'static,
    F: Fn(&T, slint::SharedPixelBuffer<slint::Rgba8Pixel>) + Send + 'static + Clone,
    M: Fn(&T, f32, f32, f32, f32, f32, [f32; 19], [f32; 19], [f32; 19], f32)
        + Send
        + 'static
        + Clone,
{
    let (work_tx, work_rx) = channel::<DisplayWork>();
    let (pool_tx, pool_rx) = channel::<DisplayWork>();
    let in_flight = Arc::new(AtomicBool::new(false));
    let generation = Arc::new(AtomicU64::new(0));
    let in_flight_thread = Arc::clone(&in_flight);
    let generation_thread = Arc::clone(&generation);

    thread::spawn(move || {
        let mut denoiser = AtrousDenoiser::new();
        let mut avg_color_buf: Vec<Vec3> = Vec::new();
        let mut filtered_buf: Vec<Vec3> = Vec::new();
        let mut fb_transfer = FramebufferTransfer::new(1, 1);
        let mut last_width = 0u32;
        let mut last_height = 0u32;

        while let Ok(work) = work_rx.recv() {
            // A reset that landed strictly AFTER this work item was snapshotted (its
            // own `generation` tag no longer matches the shared counter's CURRENT
            // value) means the pose/accumulation it was traced from no longer matches
            // what belongs on screen -- drop it rather than displaying a frame from
            // the old camera position or a since-cleared accumulation.
            let stale = work.generation != generation_thread.load(Ordering::Acquire);

            if !stale {
                if work.width != last_width || work.height != last_height {
                    fb_transfer = FramebufferTransfer::new(work.width, work.height);
                    last_width = work.width;
                    last_height = work.height;
                }

                let output_bytes = if work.denoise_enabled {
                    denoise_and_tonemap_frame(
                        FirstHitSnapshot {
                            width: work.width,
                            height: work.height,
                            current_sample_count: work.current_sample_count,
                            accum_buffer: &work.accum,
                            first_hit_depth: &work.depth,
                            first_hit_normal: &work.normal,
                            first_hit_facet_id: &work.facet_id,
                        },
                        &mut DenoiseScratch {
                            denoiser: &mut denoiser,
                            avg_color_buf: &mut avg_color_buf,
                            filtered_buf: &mut filtered_buf,
                        },
                    )
                } else {
                    tonemap_running_average(
                        work.width,
                        work.height,
                        work.current_sample_count,
                        &work.accum,
                    )
                };

                let image = fb_transfer.copy_from_gpu_slice(&output_bytes);
                push_frame_to_ui(
                    &ui_weak,
                    &update_image,
                    &update_metrics,
                    image,
                    work.metrics_snapshot,
                );
            }

            in_flight_thread.store(false, Ordering::Release);
            // Return the buffers for the render loop to reuse next cycle. A failed send
            // here just means the render loop (and its `pool_rx`) is already gone --
            // this thread is about to exit too, the next time `work_rx.recv()` returns
            // `Err`.
            let _ = pool_tx.send(work);
        }
    });

    DisplayHandle {
        work_tx,
        pool_rx,
        in_flight,
        generation,
    }
}

/// Convenience for [`spawn_render_thread`]'s convergence wait: the frame that reaches
/// `target_samples` must always be displayed exactly once, so its send site spins on
/// [`DisplayHandle::busy`] with this short sleep between checks rather than skipping.
/// À-Trous cycles run in the 100ms-1s range (see `DENOISE_MIN_INTERVAL`'s doc comment),
/// so a 1ms poll adds negligible latency while staying cheap to spin.
pub(super) const CONVERGENCE_WAIT_POLL: Duration = Duration::from_millis(1);

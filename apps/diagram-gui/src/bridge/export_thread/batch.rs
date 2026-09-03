//! The batch render loop: hybrid CPU+GPU calibration and per-batch dispatch, plus the
//! CPU scanline batch tracer itself.
//!
//! Split out of `bridge::export_thread` purely to keep that module (already sizeable)
//! from growing further.

use super::scene_snapshot::SceneSnapshot;
use gemray::{
    optics::raytracer::{Camera, pixel_rotations, sample_draws, trace_spectral_ray_with_finish},
    renderer::gpu_backend::{GpuBackend, GpuSceneRef},
};
use glam::Vec3;
use std::{
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
};

/// Below this many samples per pixel, hybrid calibration costs more than it
/// saves; the export takes the single-engine path instead.
pub(super) const HYBRID_MIN_SPP: u32 = 8;

/// How many batches the local hybrid loop targets across the FULL export (not just
/// whatever sub-range it ends up covering -- see [`run_local_batches`]'s doc comment on
/// why `total_samples` rather than the local range's own length drives this).
const TARGET_BATCHES: u32 = 40;

/// Everything one export batch needs, bundled so the hybrid helpers stay
/// within clippy's argument-count limit.
pub(super) struct ExportCtx<'a> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) camera: &'a Camera,
    pub(super) scene: &'a SceneSnapshot,
    pub(super) gpu: &'a GpuBackend,
    pub(super) gpu_scene: &'a GpuSceneRef<'a>,
}

/// Times one real export sample per pixel on each engine (GPU, then CPU -- both counted
/// toward the export, not thrown away) and returns the GPU's throughput share for
/// hybrid batches, or `None` when the GPU declines (no adapter, no `gpu` feature, or an
/// unsupported material such as the biaxial stones) -- the export then proceeds exactly
/// as it did before hybrid existed.
///
/// The GPU side actually dispatches TWO 1-spp samples, not one: the first sample of any
/// dispatch includes output-buffer allocation and driver warm-up (measured cold 102ms vs
/// warm 73ms at 800x600), which would otherwise inflate `gpu_time` into a pessimistic,
/// non-representative estimate of the GPU's steady-state throughput for the ~40 batches
/// that follow. Timing the SECOND sample instead means that cost lands on the (untimed,
/// but still real and counted) first one. Checks `cancel` once, between the GPU and CPU
/// measurements -- the two calibration samples this function times -- so a
/// cancel-during-calibration export doesn't also pay for the CPU calibration sample
/// before `run_export`'s own top-of-loop check reports it Cancelled.
pub(super) fn calibrate_split(
    ctx: &ExportCtx<'_>,
    samples_done: &mut u32,
    accum: &mut [Vec3],
    gpu_accum: &mut [Vec3],
    cancel: &AtomicBool,
) -> Option<f64> {
    // Untimed warm-up sample: a real export sample (counted below), just not the one
    // whose wall-clock cost feeds the split.
    if !ctx
        .gpu
        .try_accumulate(ctx.gpu_scene, *samples_done, 1, gpu_accum)
    {
        return None;
    }
    *samples_done += 1;

    let start = std::time::Instant::now();
    if !ctx
        .gpu
        .try_accumulate(ctx.gpu_scene, *samples_done, 1, gpu_accum)
    {
        return None;
    }
    let gpu_time = start.elapsed().as_secs_f64().max(1e-9);
    *samples_done += 1;

    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let start = std::time::Instant::now();
    render_batch(
        ctx.width,
        ctx.height,
        1,
        *samples_done,
        ctx.camera,
        ctx.scene,
        accum,
    );
    let cpu_time = start.elapsed().as_secs_f64().max(1e-9);
    *samples_done += 1;

    // GPU share proportional to measured throughput (1/time per engine).
    let frac = cpu_time / (gpu_time + cpu_time);
    Some(frac.clamp(0.0, 1.0))
}

/// One hybrid batch: the GPU traces the batch's lower sample range into its
/// own buffer on a scoped thread while every CPU core traces the upper range
/// into `accum`; the ranges are disjoint, so the per-sample stratification
/// each engine derives from the absolute sample index stays consistent. If
/// the GPU declines mid-export (e.g. device loss), its share is retraced on
/// the CPU so the sample count stays exact, and `gpu_frac` is cleared so
/// later batches stop offering it work.
///
/// # Adapting the split as the export runs
///
/// `calibrate_split` only measures one cold(ish) dispatch pair before any real batch
/// runs; freezing that fraction for all ~40 batches ignores thermal drift and other load
/// changes over what can be a multi-minute export. So every batch that actually measures
/// both engines (`gpu_share > 0 && cpu_share > 0`) times its own GPU and CPU shares and
/// blends this batch's own measured split into `gpu_frac` via the same 0.7-old/0.3-new
/// exponential moving average `render_thread::gpu_backend::HybridPacing::blend` uses for
/// the live viewport -- so the split keeps tracking the machine's actual throughput
/// instead of staying pinned to whatever `calibrate_split` happened to measure once.
/// Batches that only exercise one engine (`gpu_share == 0`, or `cpu_share == 0` once the
/// GPU is carrying nearly the whole frame) leave `gpu_frac` exactly as it was: there is
/// no second engine's timing to compare against for those.
pub(super) fn hybrid_batch(
    ctx: &ExportCtx<'_>,
    samples_done: u32,
    this_batch: u32,
    gpu_frac: &mut Option<f64>,
    accum: &mut [Vec3],
    gpu_accum: &mut [Vec3],
) {
    let frac = gpu_frac.unwrap_or(0.0);
    let gpu_share = (f64::from(this_batch) * frac).round() as u32;
    let gpu_share = gpu_share.min(this_batch);
    let cpu_share = this_batch - gpu_share;

    if gpu_share == 0 {
        render_batch(
            ctx.width,
            ctx.height,
            this_batch,
            samples_done,
            ctx.camera,
            ctx.scene,
            accum,
        );
        return;
    }

    let (gpu_ok, gpu_elapsed, cpu_elapsed) = if cpu_share == 0 {
        let start = std::time::Instant::now();
        let ok = ctx
            .gpu
            .try_accumulate(ctx.gpu_scene, samples_done, gpu_share, gpu_accum);
        (ok, start.elapsed(), std::time::Duration::ZERO)
    } else {
        thread::scope(|s| {
            let gpu_task = s.spawn(|| {
                let start = std::time::Instant::now();
                let ok = ctx
                    .gpu
                    .try_accumulate(ctx.gpu_scene, samples_done, gpu_share, gpu_accum);
                (ok, start.elapsed())
            });
            let cpu_start = std::time::Instant::now();
            render_batch(
                ctx.width,
                ctx.height,
                cpu_share,
                samples_done + gpu_share,
                ctx.camera,
                ctx.scene,
                accum,
            );
            let cpu_elapsed = cpu_start.elapsed();
            let (gpu_ok, gpu_elapsed) = gpu_task
                .join()
                .unwrap_or((false, std::time::Duration::ZERO));
            (gpu_ok, gpu_elapsed, cpu_elapsed)
        })
    };

    if !gpu_ok {
        render_batch(
            ctx.width,
            ctx.height,
            gpu_share,
            samples_done,
            ctx.camera,
            ctx.scene,
            accum,
        );
        *gpu_frac = None;
        return;
    }

    if cpu_share > 0 {
        let gpu_rate = f64::from(gpu_share) / gpu_elapsed.as_secs_f64().max(1e-9);
        let cpu_rate = f64::from(cpu_share) / cpu_elapsed.as_secs_f64().max(1e-9);
        let measured_frac = gpu_rate / (gpu_rate + cpu_rate);
        let updated = frac.mul_add(0.7, measured_frac * 0.3);
        *gpu_frac = Some(updated.clamp(0.0, 1.0));
    }
}

/// Runs the local CPU+GPU hybrid loop over `[*samples_done, total_samples)` in small
/// batches (so cancellation and progress reporting stay fine-grained, same reasoning as
/// `run_export`'s own pre-remote loop always used), calling `on_batch` with the updated
/// `*samples_done` plus read-only reborrows of `accum`/`gpu_accum` after each one -- a
/// reborrow, not a second independent capture, because `accum`/`gpu_accum` are already
/// borrowed mutably by this function's own parameters; threading them through `on_batch`
/// this way is what lets `run_export`'s progress-reporting closure read the buffers it
/// needs without the borrow checker seeing two conflicting borrows.
///
/// Extracted out of `export_thread::run_export` so that function can run this on the
/// SAME thread it's already executing on while a concurrently-dispatched remote engine
/// (`export_thread::remote::run_remote_batch`) runs on another, inside one
/// `thread::scope` -- see `run_export`'s own doc comment for why local and remote need
/// to overlap in wall-clock time rather than run one after the other. `*samples_done`'s
/// STARTING value is whatever the caller set it to (typically past calibration, and past
/// however many samples were handed to remote) -- this function only ever advances it,
/// never resets it, so it stays a correct absolute sample-range cursor regardless of how
/// much of `[0, total_samples)` other engines already own.
///
/// `batch_size` is derived from `total_samples` (the export's OVERALL target), not from
/// `total_samples - *samples_done` (the local engine's own, possibly much smaller,
/// remaining share) -- this keeps a local-only export's batch granularity, and hence its
/// progress-reporting/cancellation latency, byte-for-byte identical to before remote
/// existed. When remote is carrying most of the frame, local simply ends up running
/// fewer, same-sized batches over its smaller share, which is a fine trade against
/// reporting progress at a resolution that depends on how the split came out.
///
/// Returns `true` iff cancellation was observed (mirrors `run_export`'s own early-return
/// convention) -- callers own deciding what a cancellation means once they've also
/// learned what the concurrently-running remote engine did, since a cancelled export
/// must NOT be reported as done just because the local half stopped.
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a distinct piece of the local hybrid loop's own state \
              (scene/GPU context, the CPU/GPU split estimate, the two accumulation \
              buffers, the absolute sample cursor, cancellation, and the progress \
              callback) -- bundling them into a struct would just move the same count \
              into field access, not reduce it"
)]
pub(super) fn run_local_batches(
    ctx: &ExportCtx<'_>,
    total_samples: u32,
    gpu_frac: &mut Option<f64>,
    samples_done: &mut u32,
    accum: &mut [Vec3],
    gpu_accum: &mut [Vec3],
    cancel: &AtomicBool,
    mut on_batch: impl FnMut(u32, &[Vec3], &[Vec3]),
) -> bool {
    let batch_size = (total_samples / TARGET_BATCHES).max(1);

    while *samples_done < total_samples {
        if cancel.load(Ordering::Relaxed) {
            return true;
        }

        let this_batch = batch_size.min(total_samples - *samples_done);

        if gpu_frac.is_some() {
            hybrid_batch(ctx, *samples_done, this_batch, gpu_frac, accum, gpu_accum);
        } else if !ctx
            .gpu
            .try_accumulate(ctx.gpu_scene, *samples_done, this_batch, gpu_accum)
        {
            render_batch(
                ctx.width,
                ctx.height,
                this_batch,
                *samples_done,
                ctx.camera,
                ctx.scene,
                accum,
            );
        }
        *samples_done += this_batch;
        on_batch(*samples_done, accum, gpu_accum);
    }

    cancel.load(Ordering::Relaxed)
}

/// Traces `batch_spp` additional samples per pixel across `thread::available_parallelism`
/// CPU threads, adding them into `accum` -- the export-worker analog of
/// `render_thread::render_frame_scanlines`, minus the per-frame tone-mapping (the
/// export only tone-maps once, at the very end, in `tonemap_to_rgba`) and minus the
/// progressive-frame bookkeeping (an export runs to completion in one shot rather than
/// forever refining a still-visible frame).
///
/// # Work distribution: a shared atomic row counter, not contiguous row bands
///
/// Rows through the stone cost far more than background rows, so this claims rows
/// dynamically through a shared `AtomicUsize` counter (`fetch_add` per row) rather than
/// splitting the image into `num_threads` contiguous bands -- exactly the same fix, for
/// the same reason, as `render_thread::scanline::render_frame_scanlines`; see that
/// function's doc comment for the measured wall-time/utilization numbers. `accum` is
/// pre-split into per-row slices (`chunks_mut(width)`) behind a `Mutex<Vec<Option<&mut
/// [Vec3]>>>`; a thread claims row `y`, locks just long enough to `Option::take` that
/// row's slice (the lock is per ROW, not per pixel), then accumulates the whole row
/// without the lock held. Every row is claimed by exactly one thread (`fetch_add` hands
/// out each index once), so per-pixel sums stay bit-identical to the old contiguous
/// split.
pub(super) fn render_batch(
    width: u32,
    height: u32,
    batch_spp: u32,
    samples_already_done: u32,
    camera: &Camera,
    scene: &SceneSnapshot,
    accum: &mut [Vec3],
) {
    let num_threads = thread::available_parallelism().map_or(8, std::num::NonZero::get);
    let width_usize = width as usize;

    let rows: Vec<Option<&mut [Vec3]>> = accum.chunks_mut(width_usize).map(Some).collect();
    let rows = Mutex::new(rows);
    let next_row = AtomicUsize::new(0);

    thread::scope(|s| {
        for _ in 0..num_threads {
            let rows = &rows;
            let next_row = &next_row;

            s.spawn(move || {
                loop {
                    let y = next_row.fetch_add(1, Ordering::Relaxed);
                    if y >= height as usize {
                        break;
                    }

                    let row = rows
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)[y]
                        .take()
                        .expect("each row index is claimed by exactly one thread via fetch_add");

                    for (x, pixel) in row.iter_mut().enumerate() {
                        let global_pixel_idx = (y * width_usize + x) as u32;
                        let mut sample_sum = Vec3::ZERO;

                        // Per-pixel Cranley-Patterson rotations for the
                        // stratified pixel-jitter/hero-wavelength draws below -- see
                        // `apps/gemray-worker/src/render_core.rs::trace_into` for
                        // the formula this must stay in sync with (both now compute
                        // through the shared `gemray::optics::raytracer::sampling`
                        // functions rather than a hand-copy).
                        let rot = pixel_rotations(global_pixel_idx);

                        for s_idx in 0..batch_spp {
                            let sample_num = samples_already_done + s_idx;
                            let draws = sample_draws(global_pixel_idx, sample_num, &rot);

                            let ray = camera.generate_ray(
                                x as f32,
                                y as f32,
                                width as f32,
                                height as f32,
                                draws.jitter_x,
                                draws.jitter_y,
                            );

                            // Export always uses the analytic studio rig -- see
                            // the matching note at `render_thread.rs`'s call site for
                            // where an HDR-map UI hook would plug in for both paths.
                            let environment = scene.lighting_preset.studio(
                                scene.exposure,
                                scene.light_yaw,
                                scene.light_pitch,
                            );
                            // Frosted girdle: `scene.facet_finishes` is empty
                            // whenever the toggle was off at capture time, which
                            // `trace_spectral_ray_with_finish`'s own doc comment
                            // documents as exactly equivalent to `trace_spectral_ray`.
                            sample_sum += trace_spectral_ray_with_finish(
                                ray,
                                &scene.active_planes,
                                &scene.facet_finishes,
                                &scene.material,
                                scene.max_bounces,
                                environment,
                                draws.seed,
                                draws.hero_rand,
                                None,
                            );
                        }

                        *pixel += sample_sum;
                    }
                }
            });
        }
    });
}

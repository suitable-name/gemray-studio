//! The CPU scanline tracer: renders one full frame in parallel across
//! `thread::available_parallelism` CPU threads.
//!
//! Split out of `bridge::render_thread` purely to keep that module (already sizeable)
//! from growing further.

use super::gpu_backend::{BackendFrame, FrameOutputs};
use gemray::optics::raytracer::{
    HitRecord, pixel_rotations, sample_draws, trace_spectral_ray_with_finish,
};
use glam::Vec3;
use std::{
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

/// One row's worth of the four output buffers, handed out to whichever thread's atomic
/// counter claims that row -- see [`render_frame_scanlines`]'s doc comment.
struct RowSlices<'a> {
    acc: &'a mut [Vec3],
    depth: &'a mut [f32],
    normal: &'a mut [Vec3],
    facet: &'a mut [i32],
}

/// Renders one full frame's scanlines in parallel across `thread::available_parallelism`
/// CPU threads, accumulating each pixel's radiance into `accum_buffer` and each pixel's
/// PRIMARY-ray first-hit depth/normal/facet-index into the three `first_hit_*` guide
/// buffers (these feed the À-Trous denoiser -- see `denoise_and_tonemap_frame`,
/// the next stage in the pipeline, and `renderer::denoise`'s module docs for why they're
/// needed). Split out of `spawn_render_thread` purely to keep that function under
/// clippy's function-length lint; the chunking and jitter logic are unchanged from when
/// this was inlined -- only the addition of the guide-buffer capture, and the removal of
/// the per-pixel tone-mapping (now `denoise_and_tonemap_frame`'s job, run once over the
/// WHOLE frame after this accumulation pass finishes, since the denoiser needs every
/// pixel's guide data available at once) are new.
///
/// # Work distribution: a shared atomic row counter, not contiguous row bands
///
/// Rows that pass through the stone cost far more than background rows, so splitting the
/// image into `num_threads` contiguous row bands (one per thread, as this function used
/// to) starves whichever threads land entirely on cheap background rows while the
/// thread(s) covering the stone are still working -- measured on a 480x360/2spp/16-thread
/// render: 256ms wall time at 49% thread utilization for contiguous bands, 145ms at 99%
/// for the dynamic counter below. Each thread instead repeatedly claims "the next
/// unclaimed row" via `next_row.fetch_add(1, ..)` until every row is taken, so a thread
/// that finishes a cheap row picks up the next available one immediately rather than
/// sitting idle inside its own pre-assigned band.
///
/// Each row's four output slices are handed out exactly once: `rows` pre-splits every
/// buffer into per-row slices via `chunks_mut(width)` and wraps the resulting
/// `Vec<Option<RowSlices>>` in a [`Mutex`]. A thread that claims row `y` locks `rows`
/// just long enough to `Option::take` that row's slices (the lock is held per ROW, not
/// per pixel), then processes the whole row -- writing every pixel's accumulation,
/// depth, normal and facet-id -- without the lock held. Because `next_row.fetch_add`
/// hands out each index exactly once, no two threads ever contend for the same `Option`,
/// and every pixel is still written by exactly one thread, so per-pixel results stay
/// bit-identical to the old contiguous-band split -- only which thread happens to render
/// which row (never the seed, jitter, or per-pixel summation order within a row) changes
/// run to run.
pub(super) fn render_frame_scanlines(
    frame: &BackendFrame<'_>,
    spp: u32,
    current_sample_count: u32,
    outputs: &mut FrameOutputs<'_>,
) {
    let width = frame.width;
    let height = frame.height;
    let num_threads = thread::available_parallelism().map_or(8, std::num::NonZero::get);
    let width_usize = width as usize;

    // Reborrowed as four disjoint fields up front (rather than chained off `outputs`
    // directly) so the simultaneous `zip` below borrows each independently.
    let FrameOutputs {
        accum,
        depth,
        normal,
        facet_id,
    } = outputs;
    let rows: Vec<Option<RowSlices<'_>>> = accum
        .chunks_mut(width_usize)
        .zip(depth.chunks_mut(width_usize))
        .zip(normal.chunks_mut(width_usize))
        .zip(facet_id.chunks_mut(width_usize))
        .map(|(((acc, depth), normal), facet)| {
            Some(RowSlices {
                acc,
                depth,
                normal,
                facet,
            })
        })
        .collect();
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

                    let RowSlices {
                        acc: acc_row,
                        depth: depth_row,
                        normal: normal_row,
                        facet: facet_row,
                    } = rows
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)[y]
                        .take()
                        .expect("each row index is claimed by exactly one thread via fetch_add");

                    for x in 0..width_usize {
                        let global_pixel_idx = (y * width_usize + x) as u32;
                        let mut sample_sum = Vec3::ZERO;
                        let mut primary_hit: Option<HitRecord> = None;

                        // Per-pixel Cranley-Patterson rotations for the stratified
                        // pixel-jitter/hero-wavelength draws below -- pure functions of
                        // `global_pixel_idx` alone. See `sample_draws`/`pixel_rotations`'s
                        // own doc comments (`gemray::optics::raytracer::sampling`) for the
                        // mechanism, and `apps/gemray-worker/src/render_core.rs::trace_into`
                        // for the formula this must stay in sync with -- both now compute
                        // through those same shared functions rather than a hand-copy.
                        let rot = pixel_rotations(global_pixel_idx);

                        for s_idx in 0..spp {
                            let sample_num = current_sample_count - spp + s_idx;
                            let draws = sample_draws(global_pixel_idx, sample_num, &rot);

                            let ray = frame.camera.generate_ray(
                                x as f32,
                                y as f32,
                                width as f32,
                                height as f32,
                                draws.jitter_x,
                                draws.jitter_y,
                            );

                            // Frosted girdle: `facet_finishes` is `&[]` whenever
                            // `RenderContext::girdle_frosted` is off (see the call site
                            // below), which `trace_spectral_ray_with_finish`'s own doc
                            // comment documents as exactly equivalent to
                            // `trace_spectral_ray` -- every facet index looks up
                            // `FacetFinish::default() == Polished`.
                            let sample_xyz = trace_spectral_ray_with_finish(
                                ray,
                                frame.planes,
                                frame.facet_finishes,
                                frame.material,
                                frame.max_bounces,
                                frame.environment,
                                draws.seed,
                                draws.hero_rand,
                                Some(&mut primary_hit),
                            );

                            sample_sum += sample_xyz;
                        }

                        acc_row[x] += sample_sum;
                        // Facet-id design decision: `primary_hit` above is
                        // captured from the LAST sample traced this call, which -- like
                        // every other sample this pixel could have drawn -- is the HERO
                        // channel's own first hit (see `trace_spectral_ray`'s bounce-0
                        // capture comment). Anti-aliasing jitter means consecutive
                        // samples can occasionally land on different facets right at a
                        // silhouette edge; the denoiser's own facet-identity guide term
                        // is a hard Kronecker delta specifically so a stale-by-one-sample
                        // id at such an edge costs at most a slightly conservative (not
                        // incorrect) edge-stop there, never a wrong-but-confident blend.
                        depth_row[x] = primary_hit.map_or(1.0e6, |h| h.t);
                        normal_row[x] = primary_hit.map_or(Vec3::ZERO, |h| h.normal);
                        facet_row[x] = primary_hit.map_or(-1, |h| h.facet_idx as i32);
                    }
                }
            });
        }
    });
}

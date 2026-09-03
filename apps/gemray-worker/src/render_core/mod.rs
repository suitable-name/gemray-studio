//! The one function both subcommands use to actually run the tracer: [`trace_samples`].
//!
//! It traces an explicit `[first_sample, first_sample + samples)` range across a whole
//! frame and returns the SUMMED (never averaged) per-pixel XYZ radiance -- exactly what
//! a caller accumulating contributions from multiple workers needs to add straight into
//! its own running total. See `gemray_net`'s crate docs for why sample-index
//! partitioning (not screen-space tiling) is what makes that additive.
//!
//! # Keeping the seed formula in sync with the viewer
//!
//! The per-sample RNG seed and pixel-jitter derivation below is copied verbatim from
//! `apps/diagram-gui/src/bridge/render_thread.rs`'s `render_frame_scanlines` (and
//! `crates/gemray-net/tests/partition_correctness.rs`, which reproduces the same
//! formula to test additivity against the real tracer end to end). It is NOT
//! re-derived here: the formula is a function of `(global_pixel_idx, sample_num)`
//! alone, `sample_num` being the ABSOLUTE sample index (`first_sample + s_idx`), never
//! a batch-relative offset -- that is precisely the property that makes tracing a
//! sample range here, remotely, produce the identical numbers the viewer would have
//! produced tracing that same range itself. If this formula and the viewer's ever
//! drift apart, a remote worker's contribution silently composes into a wrong (but
//! entirely plausible-looking) image -- see `gemray_net::handshake`'s doc comment for
//! why the build-hash check exists to catch the analogous drift on the *physics* side;
//! this formula is the wire-protocol-level analogue of that same risk.

use gemray::optics::raytracer::{
    Camera, EnvironmentSource, FacetFinish, pixel_rotations, sample_draws,
    trace_spectral_ray_with_finish,
};
use gemray_net::SceneState;
use glam::Vec3;
use std::{
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use gemray::renderer::gpu_backend::{GpuBackend, GpuSceneRef};

/// Resolves the per-facet finish `scene.girdle_frosted` implies.
///
/// `gemray_net::SceneState` carries no `Vec<FacetFinish>` of its own -- the wire-format
/// encoding is the single `girdle_frosted` bool (see that field's doc comment for why):
/// the worker re-derives the exact same `Vec<FacetFinish>` the viewer would have used
/// locally via `gemray::geometry::girdle_facet_finishes`, a pure, deterministic function
/// of `scene.planes` alone (see that function's own doc comment on why the same planes
/// always yield the same finishes, bit-for-bit). `false` returns an empty slice, which
/// `trace_spectral_ray_with_finish`'s own doc comment documents as exactly equivalent to
/// `trace_spectral_ray` -- every facet looks up `FacetFinish::default() == Polished`.
#[must_use]
fn resolve_facet_finishes(scene: &SceneState) -> Vec<FacetFinish> {
    if scene.girdle_frosted {
        gemray::geometry::girdle_facet_finishes(&scene.planes)
    } else {
        Vec::new()
    }
}

/// Fixed FOV matching the viewer's own hard-coded value. `SceneState` carries no FOV
/// field of its own (see that struct's doc comment on what it deliberately does and
/// doesn't carry), so this must match `apps/diagram-gui/src/bridge/render_thread.rs`
/// and `export_thread.rs`'s `42.0` exactly for a remote worker's camera rays to line up
/// with the viewer's.
const VIEWER_FOV_DEG: f32 = 42.0;

/// Resolves a `--threads`-style argument (`0` meaning "let the OS decide") to an actual
/// thread count.
///
/// Shared by [`trace_samples`]'s own chunking and by `serve`'s `Welcome` message (which
/// reports the thread count it actually renders with, not the literal `0` sentinel).
#[must_use]
pub fn effective_thread_count(threads: usize) -> usize {
    if threads == 0 {
        thread::available_parallelism().map_or(8, std::num::NonZero::get)
    } else {
        threads
    }
}

/// Traces samples `[first_sample, first_sample + samples)` across the whole frame.
///
/// Parallel across `threads` CPU threads (`0` for "all available cores" -- see
/// [`effective_thread_count`]), over `scene.width x scene.height`.
///
/// Returns the summed (not averaged) per-pixel XYZ radiance, one [`Vec3`] per pixel in
/// row-major order. Callers that want a displayable/exportable image (e.g. `render_cmd`)
/// divide by their own total sample count when tone-mapping; callers accumulating
/// contributions from multiple workers (e.g. a viewer talking to several `serve`
/// instances) add these sums directly into their own running total -- either way, this
/// function itself never divides, so it can't bake in an assumption about how many
/// total samples the caller ultimately wants (see the module docs and `gemray_net`'s
/// crate docs on why summing, not averaging, is what remote offload requires).
///
/// Returns an all-[`Vec3::ZERO`] buffer (still correctly sized) if `samples`, `width`,
/// or `height` is zero -- callers are expected to have already rejected those via
/// [`crate::validate`] if zero is actually invalid for their use case; this function
/// itself has no opinion and just traces nothing.
#[must_use]
pub fn trace_samples(
    scene: &SceneState,
    first_sample: u32,
    samples: u32,
    threads: usize,
) -> Vec<Vec3> {
    let width = scene.width;
    let height = scene.height;
    let mut buffer = vec![Vec3::ZERO; width as usize * height as usize];
    if samples == 0 || width == 0 || height == 0 {
        return buffer;
    }

    let camera = Camera::new(scene.yaw, scene.pitch, scene.distance, VIEWER_FOV_DEG);
    let environment =
        scene
            .lighting_preset
            .studio(scene.exposure, scene.light_yaw, scene.light_pitch);

    trace_into(
        scene,
        first_sample,
        samples,
        threads,
        &camera,
        environment,
        &mut buffer,
    );
    buffer
}

/// Traces samples `[first_sample, first_sample + samples)` across the whole frame,
/// preferring `gpu`.
///
/// Falls back to the CPU tracer (via [`trace_into`], the exact same code
/// [`trace_samples`] itself runs) whenever `gpu` declines -- see
/// `gemray::renderer::gpu_backend::GpuBackend`'s doc comment for why a decline is normal, not an error.
///
/// A drop-in replacement for [`trace_samples`] at every call site: same signature (with
/// `gpu` prepended), same return contract (a summed, never averaged, per-pixel XYZ
/// buffer), same disjoint-and-additive sample-range property both `render_cmd` and
/// `serve` depend on -- both backends ADD into a `Vec3::ZERO`-initialized buffer with
/// identical semantics (see `gemray::renderer::gpu_backend`'s own doc comment on why that's what keeps a
/// GPU worker's sample ranges mergeable with a CPU viewer's).
#[must_use]
pub fn trace_samples_with_gpu(
    gpu: &GpuBackend,
    scene: &SceneState,
    first_sample: u32,
    samples: u32,
    threads: usize,
) -> Vec<Vec3> {
    let width = scene.width;
    let height = scene.height;
    let mut buffer = vec![Vec3::ZERO; width as usize * height as usize];
    if samples == 0 || width == 0 || height == 0 {
        return buffer;
    }

    let camera = Camera::new(scene.yaw, scene.pitch, scene.distance, VIEWER_FOV_DEG);
    let environment =
        scene
            .lighting_preset
            .studio(scene.exposure, scene.light_yaw, scene.light_pitch);

    // Frosted girdle: `scene.girdle_frosted` is the wire-format encoding of the
    // viewer's toggle -- see `resolve_facet_finishes`'s doc comment for why a bool
    // (re-expanded here into the actual `Vec<FacetFinish>`) rather than shipping the
    // list itself. The CPU fallback below (`trace_into`) resolves the identical
    // finishes from the identical `scene.girdle_frosted`/`scene.planes`, so a GPU
    // decline mid-trace can never silently switch a scene between polished and frosted.
    let facet_finishes = resolve_facet_finishes(scene);
    let gpu_scene = GpuSceneRef {
        camera: &camera,
        width,
        height,
        planes: &scene.planes,
        facet_finishes: &facet_finishes,
        material: &scene.material,
        max_bounces: scene.max_bounces,
        environment,
    };
    if gpu.try_accumulate(&gpu_scene, first_sample, samples, &mut buffer) {
        return buffer;
    }

    // The GPU declined (no adapter, --no-gpu, or an unsupported material -- see
    // `gemray::renderer::gpu_backend::GpuBackend`'s doc comment). `buffer` is guaranteed still all-zero
    // here -- a decline never partially writes -- so falling back is exactly
    // `trace_samples`' own zero-initialized start.
    trace_into(
        scene,
        first_sample,
        samples,
        threads,
        &camera,
        environment,
        &mut buffer,
    );
    buffer
}

/// The CPU tracing loop itself, ADDING into an already-sized `buffer` (`width x height`
/// entries, row-major). Shared by [`trace_samples`] and [`trace_samples_with_gpu`]'s CPU
/// fallback so both go through the exact same seed formula and thread-chunking -- see
/// this module's own doc comment on why that formula must never drift from the viewer's.
///
/// # Work distribution: a shared atomic row counter, not contiguous row bands
///
/// Rows through the stone cost far more than background rows, so this claims rows
/// dynamically through a shared `AtomicUsize` counter (`fetch_add` per row) rather than
/// splitting the frame into `num_threads` contiguous bands -- the same fix, for the same
/// reason, as `apps/diagram-gui/src/bridge/render_thread/scanline.rs::render_frame_scanlines`
/// and `export_thread/batch.rs::render_batch`; see the former's doc comment for the
/// measured wall-time/utilization numbers. `buffer` is pre-split into per-row slices
/// (`chunks_mut(width)`) behind a `Mutex<Vec<Option<&mut [Vec3]>>>`; a thread claims row
/// `y`, locks just long enough to `Option::take` that row's slice (the lock is per ROW,
/// not per pixel), then traces the whole row without the lock held. Every row is claimed
/// by exactly one thread (`fetch_add` hands out each index once), so per-pixel sums stay
/// bit-identical to the old contiguous split -- see `single_and_multi_threaded_traces_agree`.
fn trace_into(
    scene: &SceneState,
    first_sample: u32,
    samples: u32,
    threads: usize,
    camera: &Camera,
    environment: EnvironmentSource<'_>,
    buffer: &mut [Vec3],
) {
    let width = scene.width;
    let height = scene.height;
    let width_usize = width as usize;

    let num_threads = effective_thread_count(threads).max(1);
    let planes = &scene.planes;
    let material = &scene.material;
    let max_bounces = scene.max_bounces;
    // Frosted girdle -- see `resolve_facet_finishes`'s doc comment.
    let facet_finishes = resolve_facet_finishes(scene);
    let facet_finishes = facet_finishes.as_slice();

    let rows: Vec<Option<&mut [Vec3]>> = buffer.chunks_mut(width_usize).map(Some).collect();
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

                        // Per-pixel Cranley-Patterson rotations for the stratified
                        // pixel-jitter/hero-wavelength draws below -- pure functions of
                        // `global_pixel_idx` alone, hoisted out of the sample loop since
                        // they don't vary per sample. See `gemray::optics::raytracer::sampling`'s
                        // `pixel_rotations`/`sample_draws` doc comments for the mechanism
                        // (shared with every other production call site now, rather than
                        // a hand-copy) and why jx/jy/hero_rand each use a DIFFERENT prime
                        // base (2, 3, 5) rather than the same base rotated three ways --
                        // using one base for all three measurably made variance WORSE,
                        // not better, for exactly the highest-variance pixels this
                        // targets.
                        let rot = pixel_rotations(global_pixel_idx);

                        for s_idx in 0..samples {
                            let sample_num = first_sample.wrapping_add(s_idx);
                            // Stratified, not an unstratified hash-uniform -- each of
                            // these is a pure function of the ABSOLUTE sample index
                            // alone (never a batch-relative counter), which is exactly
                            // what lets disjoint worker sample ranges compose correctly
                            // -- see this function's own doc comment.
                            let draws = sample_draws(global_pixel_idx, sample_num, &rot);

                            let ray = camera.generate_ray(
                                x as f32,
                                y as f32,
                                width as f32,
                                height as f32,
                                draws.jitter_x,
                                draws.jitter_y,
                            );

                            sample_sum += trace_spectral_ray_with_finish(
                                ray,
                                planes,
                                facet_finishes,
                                material,
                                max_bounces,
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

#[cfg(test)]
mod tests {
    use super::*;
    use gemray::{
        geometry::cuts::StandardGemCuts,
        optics::{materials::GemMaterial, raytracer::LightingPreset},
    };

    fn tiny_scene() -> SceneState {
        SceneState {
            width: 8,
            height: 8,
            yaw: 0.4,
            pitch: 0.3,
            distance: 3.0,
            light_yaw: 0.85,
            light_pitch: 0.95,
            exposure: 1.0,
            max_bounces: 4,
            lighting_preset: LightingPreset::Daylight,
            material: GemMaterial::diamond(),
            planes: StandardGemCuts::standard_round_brilliant(),
            girdle_frosted: false,
        }
    }

    #[test]
    fn returns_a_buffer_sized_to_the_scene() {
        let scene = tiny_scene();
        let buf = trace_samples(&scene, 0, 4, 1);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn zero_samples_returns_an_all_zero_buffer_of_the_right_size() {
        let scene = tiny_scene();
        let buf = trace_samples(&scene, 0, 0, 1);
        assert_eq!(buf.len(), 64);
        assert!(buf.iter().all(|v| *v == Vec3::ZERO));
    }

    #[test]
    fn single_and_multi_threaded_traces_agree() {
        let scene = tiny_scene();
        let single = trace_samples(&scene, 0, 4, 1);
        let multi = trace_samples(&scene, 0, 4, 4);
        for (a, b) in single.iter().zip(multi.iter()) {
            // Thread count only changes how work is chunked, never which (pixel,
            // sample) pairs are traced or in what order they're summed WITHIN a
            // pixel -- so this must be bit-exact, not just approximately equal.
            assert_eq!(a, b);
        }
    }

    #[test]
    fn splitting_a_sample_range_across_two_calls_sums_to_the_same_result_as_one_call() {
        // The additivity property this whole crate exists to preserve, exercised
        // through `trace_samples` itself rather than the lower-level formula
        // `gemray-net`'s own partition test already covers. Relative tolerance, not
        // bit-exact -- float addition is not associative, so summing in a different
        // grouping can differ in the last bit or two of an `f32` even for a correct
        // implementation. See `gemray-net/tests/partition_correctness.rs`'s doc comment.
        let scene = tiny_scene();
        let whole = trace_samples(&scene, 0, 8, 2);
        let first_half = trace_samples(&scene, 0, 4, 2);
        let second_half = trace_samples(&scene, 4, 4, 2);

        for i in 0..whole.len() {
            let split_sum = first_half[i] + second_half[i];
            let diff = (whole[i] - split_sum).abs();
            let scale = whole[i].abs().max(split_sum.abs()).max(Vec3::splat(1e-6));
            let rel = diff / scale;
            assert!(
                rel.max_element() < 1e-3,
                "pixel {i}: whole={:?} split_sum={:?} rel={:?}",
                whole[i],
                split_sum,
                rel
            );
        }
    }
}

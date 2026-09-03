//! CPU+GPU hybrid frame rendering: the GPU and every CPU core trace disjoint sample
//! ranges of the SAME frame at the same time, and their per-pixel radiance sums are
//! merged into one image.
//!
//! # Why this is safe to build on top of [`frame`], not a new estimator
//!
//! This module ports no physics of its own. The GPU side is exactly
//! [`GpuFrameRenderer::accumulate`], used as-is; the CPU side is exactly the
//! `trace_spectral_ray_with_finish` reference construction `estimator_check`'s own CPU
//! reference loop uses to stay bit-comparable with the GPU shader (same seed hash, same
//! low-discrepancy stratified jitter, the same Cranley-Patterson rotation, the same
//! hero-wavelength draw). The seed/jitter/hero construction itself now comes from
//! `optics::raytracer::sampling::{pixel_rotations, sample_draws}` -- the ONE place that
//! formula is written down (see that module's doc comment) -- rather than being
//! hand-copied inline here; `estimator_check`'s own CPU reference loop still carries its
//! own independent hand-copy of the same formula (self-test scaffolding this module
//! deliberately doesn't couple to -- see below), so any future change to the formula
//! still needs to land in both places.
//!
//! # Why the split is disjoint sample RANGES, not disjoint pixels
//!
//! `GpuFrameRenderer::accumulate`'s `sample_offset` parameter exists precisely so
//! different workers can own disjoint sample ranges of one frame -- see its own doc
//! comment ("`sample_offset` must be the number of samples already in `accum` for these
//! pixels: the shader derives each thread's seed and its stratified jitter from the
//! absolute sample index"). `estimator_check::run_image_comparison`'s Tier 3 check
//! already exercises exactly this pattern (CPU and GPU on disjoint sample ranges of the
//! same scene, "as production renders would split work" per that module's own doc
//! comment) as a statistical equivalence harness; this module is the production version
//! of that pattern: the GPU renders samples `[0, gpu_samples)` (`sample_offset = 0`,
//! `GpuFrameRenderer::accumulate`'s existing untouched path) while the CPU renders
//! `[gpu_samples, gpu_samples + cpu_samples)` (`sample_offset = gpu_samples` baked into
//! the CPU sample-index arithmetic below), and BOTH run at the same time rather than one
//! after the other.
//!
//! # Merge convention: sum, never average
//!
//! Every accumulation buffer in this crate (`GpuFrameRenderer::accumulate`'s own
//! `accum`, the CPU reference loop's per-pixel totals) is a running SUM of per-sample
//! radiance, divided by the sample count only once, by the caller, at display time --
//! never inside the renderer. [`render_hybrid`] keeps that convention: it adds the GPU
//! engine's per-pixel sum and the CPU engine's per-pixel sum together, and the caller
//! divides the result by `split.total_spp()` exactly as it would for a single-engine
//! render.
//!
//! # Biaxial routing
//!
//! A genuinely biaxial material used to have no WGSL indicatrix -- see
//! [`GpuFrameScene`]'s own doc comment on `material` and
//! `GpuFrameRenderer::accumulate`'s `UnsupportedMaterial` handling. [`render_hybrid`]
//! still carries the rule at the split level (for any material with
//! `GemMaterial::gpu_supported() == false` it silently overrides whatever split was
//! requested to an all-CPU one, `gpu_samples = 0`, rather than ever calling `accumulate`
//! with a material it would reject, and reports that override via
//! [`HybridStats::forced_cpu_only`]) but the eigenvector-conditioning fix to
//! `optics::birefringence::BiaxialIndicatrix` means `GemMaterial::gpu_supported()` now
//! returns `true` UNCONDITIONALLY (see that method's own doc comment) -- so
//! `forced_cpu_only` can no longer actually become `true` for any material this crate
//! ships. The field and the check both stay: `gpu_supported()` remains a real per-scene
//! predicate a future biaxial-incompatible material (or a regression) could once again
//! return `false` from, and this crate's `examples/hybrid_bench.rs` still reads
//! `forced_cpu_only` to confirm the current, un-overridden behaviour.
//!
//! # Determinism
//!
//! Two [`render_hybrid`] calls with the same explicit [`HybridSplit`] against the same
//! scene produce byte-identical summed buffers: the GPU side already has this property
//! (`estimator_check::run_determinism`, `frame::run_chunk_equivalence`), and the CPU side
//! has it by construction -- every `(pixel, sample)` tuple's seed and jitter are pure
//! functions of the pixel and absolute sample index, nothing else, so which thread
//! happened to trace it changes nothing about the result. [`HybridSplit::calibrated`]
//! itself is NOT required to be reproducible run-to-run (it measures wall-clock
//! throughput, which is inherently noisy) -- only a render given an already-chosen,
//! explicit split is.

use std::time::{Duration, Instant};

use glam::Vec3;

use crate::optics::raytracer::{pixel_rotations, sample_draws, trace_spectral_ray_with_finish};

use super::frame::{GpuFrameError, GpuFrameRenderer, GpuFrameScene};

/// A deterministic, explicit static split of one frame's total samples-per-pixel budget
/// between the GPU and the CPU. Total spp is `gpu_samples + cpu_samples`.
///
/// Deliberately NOT chosen inside [`render_hybrid`] itself -- see the module doc comment
/// on why the render always takes an explicit split, even when that split came from
/// [`Self::calibrated`] a moment earlier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridSplit {
    pub gpu_samples: u32,
    pub cpu_samples: u32,
}

impl HybridSplit {
    /// Total samples per pixel this split covers.
    #[must_use]
    pub const fn total_spp(&self) -> u32 {
        self.gpu_samples + self.cpu_samples
    }

    /// A split that sends every sample to the CPU -- what [`render_hybrid`] falls back
    /// to for a biaxial material, and a reasonable default when no GPU adapter exists at
    /// all.
    #[must_use]
    pub const fn cpu_only(total_spp: u32) -> Self {
        Self {
            gpu_samples: 0,
            cpu_samples: total_spp,
        }
    }

    /// Measures a small warmup frame's GPU-only and CPU-only throughput (samples/sec, at
    /// `warmup_spp` each, against `scene` at its real resolution) and picks a
    /// deterministic `total_spp`-sample split proportional to the two measured rates.
    ///
    /// The warmup renders are thrown away -- only their wall-clock cost is used. This
    /// exists purely to pick a reasonable starting split; the render it informs still
    /// takes that split explicitly, so nothing about [`render_hybrid`]'s own determinism
    /// depends on this function's measurement being stable across runs.
    ///
    /// A biaxial `scene.material` short-circuits straight to [`Self::cpu_only`] without
    /// touching the GPU at all, since a real render of this scene could not use it
    /// either. `total_spp == 0` or `warmup_spp == 0` also short-circuits (an all-CPU
    /// split with zero throughput measurement) rather than dividing by a zero-sample
    /// warmup.
    ///
    /// # Errors
    ///
    /// Whatever [`GpuFrameRenderer::accumulate`] returns for the GPU warmup dispatch --
    /// only reachable once `scene.material` is confirmed GPU-supported and both sample
    /// counts are nonzero, so in practice this is infallible for any scene that would
    /// also succeed under [`render_hybrid`].
    pub fn calibrated(
        renderer: &mut GpuFrameRenderer,
        scene: &GpuFrameScene<'_>,
        total_spp: u32,
        warmup_spp: u32,
    ) -> Result<Self, GpuFrameError> {
        if !scene.material.gpu_supported() {
            return Ok(Self::cpu_only(total_spp));
        }
        if total_spp == 0 || warmup_spp == 0 {
            return Ok(Self {
                gpu_samples: 0,
                cpu_samples: total_spp,
            });
        }

        let num_pixels = scene.width as usize * scene.height as usize;

        // Throwaway warm-up dispatch, BEFORE the timed one: a GPU dispatch's first call
        // includes output-buffer allocation and driver warm-up (measured cold 102ms vs
        // warm 73ms at 800x600 in the export path's own calibration -- see
        // `apps/diagram-gui/src/bridge/export_thread/batch.rs::calibrate_split`), which
        // would otherwise inflate `gpu_rate` below into an unrealistically pessimistic
        // (cold) throughput estimate every time calibration runs. This dispatch's result
        // is discarded -- only the real, now-warm dispatch's wall-clock cost feeds the
        // split.
        let mut gpu_throwaway = vec![Vec3::ZERO; num_pixels];
        renderer.accumulate(scene, 0, warmup_spp, &mut gpu_throwaway)?;

        let mut gpu_warmup = vec![Vec3::ZERO; num_pixels];
        let gpu_start = Instant::now();
        renderer.accumulate(scene, 0, warmup_spp, &mut gpu_warmup)?;
        let gpu_rate = samples_per_sec(num_pixels, warmup_spp, gpu_start.elapsed());

        let cpu_start = Instant::now();
        let _cpu_warmup = cpu_trace_range(scene, 0, warmup_spp);
        let cpu_rate = samples_per_sec(num_pixels, warmup_spp, cpu_start.elapsed());

        Ok(Self::from_rates(total_spp, gpu_rate, cpu_rate))
    }

    /// Deterministic split (round-half-up on the measured share, given fixed rates) --
    /// pulled out of [`Self::calibrated`] purely so the arithmetic is independently
    /// testable without a GPU adapter.
    fn from_rates(total_spp: u32, gpu_rate: f64, cpu_rate: f64) -> Self {
        let total_rate = gpu_rate + cpu_rate;
        if total_rate <= 0.0 {
            // Neither engine produced a measurable rate (a warmup so fast the clock read
            // zero elapsed time both ways) -- split down the middle rather than divide by
            // zero.
            let gpu_samples = total_spp / 2;
            return Self {
                gpu_samples,
                cpu_samples: total_spp - gpu_samples,
            };
        }
        let gpu_share = gpu_rate / total_rate;
        let gpu_samples = f64::from(total_spp)
            .mul_add(gpu_share, 0.5)
            .floor()
            .min(f64::from(total_spp)) as u32;
        Self {
            gpu_samples,
            cpu_samples: total_spp - gpu_samples,
        }
    }
}

/// What one [`render_hybrid`] call measured. Purely informational -- correctness never
/// depends on any field here, only on the merged buffer `render_hybrid` writes into.
#[derive(Debug, Clone, Copy)]
pub struct HybridStats {
    /// Wall-clock time for the whole hybrid render, from just before the GPU dispatch
    /// and CPU tracing both start to just after their results are merged.
    pub wall_time: Duration,
    pub gpu_samples: u32,
    pub cpu_samples: u32,
    /// `0.0` when `gpu_samples == 0` (no dispatch happened, so no rate was measured).
    pub gpu_samples_per_sec: f64,
    /// `0.0` when `cpu_samples == 0`.
    pub cpu_samples_per_sec: f64,
    /// `true` when the requested split's `gpu_samples` was overridden to `0` because
    /// `scene.material.gpu_supported()` returned `false` -- see the module doc comment's
    /// "Biaxial routing" section. Always `false` today: `GemMaterial::gpu_supported()`
    /// is unconditional now (every built-in material, biaxial ones included, returns
    /// `true`), so this override path is currently unreachable in practice. The field
    /// stays -- it still reports the truth for whatever `gpu_supported()` decides, and
    /// would flip back to observable the moment that predicate ever again returns
    /// `false` for some material.
    pub forced_cpu_only: bool,
}

/// Renders one frame with the GPU and CPU tracing disjoint sample ranges at the same
/// time, then sums their per-pixel radiance into `accum`.
///
/// `split.gpu_samples` samples per pixel run on the GPU (`sample_offset = 0`, via the
/// existing [`GpuFrameRenderer::accumulate`] machinery) while `split.cpu_samples` run on
/// the CPU (`sample_offset = split.gpu_samples`, spread across every available core).
///
/// Concurrency: the GPU dispatch runs on its own thread (`GpuFrameRenderer::accumulate`
/// needs `&mut renderer` exclusively) while the calling thread and its own worker threads
/// trace the CPU's disjoint sample range -- see [`cpu_trace_range`]. The two engines'
/// outputs land in separate buffers and are only combined after both finish, so there is
/// no shared mutable state between them during the overlap.
///
/// A biaxial `scene.material` (`GemMaterial::gpu_supported() == false`) routes EVERY
/// sample to the CPU regardless of what `requested_split` asked for -- see the module doc
/// comment's "Biaxial routing" section and [`HybridStats::forced_cpu_only`].
///
/// `accum` is ADDED into, exactly like [`GpuFrameRenderer::accumulate`]'s own `accum` --
/// a caller re-rendering into the same buffer accumulates further samples rather than
/// overwriting.
///
/// # Errors
///
/// Whatever [`GpuFrameRenderer::accumulate`] returns for the GPU dispatch. Never reached
/// when `scene.material` is biaxial, since that case never calls `accumulate` at all.
///
/// # Panics
///
/// Panics if `accum.len()` is not `scene.width * scene.height`, and if the internal GPU
/// dispatch thread itself panics (propagated via `JoinHandle::join`).
pub fn render_hybrid(
    renderer: &mut GpuFrameRenderer,
    scene: &GpuFrameScene<'_>,
    requested_split: HybridSplit,
    accum: &mut [Vec3],
) -> Result<HybridStats, GpuFrameError> {
    let num_pixels = scene.width as usize * scene.height as usize;
    assert_eq!(
        accum.len(),
        num_pixels,
        "accumulation buffer must have one entry per pixel"
    );

    let forced_cpu_only = !scene.material.gpu_supported();
    let split = if forced_cpu_only {
        HybridSplit::cpu_only(requested_split.total_spp())
    } else {
        requested_split
    };

    let wall_start = Instant::now();

    // The GPU dispatch (when there is one) runs on its own thread for the whole
    // duration of the CPU tracing below -- `renderer.accumulate` blocks on GPU readback,
    // so overlapping it with CPU work is the entire point of "hybrid". The CPU tracing
    // happens on THIS thread plus whatever worker threads `cpu_trace_range` itself
    // spawns, all still nested inside this one `scope` call.
    let (gpu_result, cpu_buf, cpu_elapsed) = std::thread::scope(|s| {
        // Proving this closure is `Send` requires the solver to walk all the way
        // through `wgpu`'s internal buffer/registry nesting -- see the crate root's
        // `#![recursion_limit = "256"]` for why that overflows the default limit.
        let gpu_handle = s.spawn(move || -> Result<(Vec<Vec3>, Duration), GpuFrameError> {
            let mut buf = vec![Vec3::ZERO; num_pixels];
            if split.gpu_samples == 0 {
                return Ok((buf, Duration::ZERO));
            }
            let start = Instant::now();
            renderer.accumulate(scene, 0, split.gpu_samples, &mut buf)?;
            Ok((buf, start.elapsed()))
        });

        let cpu_start = Instant::now();
        let cpu_buf = if split.cpu_samples == 0 {
            vec![Vec3::ZERO; num_pixels]
        } else {
            cpu_trace_range(scene, split.gpu_samples, split.cpu_samples)
        };
        let cpu_elapsed = cpu_start.elapsed();

        let gpu_result = gpu_handle
            .join()
            .expect("hybrid GPU dispatch thread panicked");
        (gpu_result, cpu_buf, cpu_elapsed)
    });

    let (gpu_buf, gpu_elapsed) = gpu_result?;

    for ((dst, gpu_sample), cpu_sample) in accum.iter_mut().zip(gpu_buf).zip(cpu_buf) {
        *dst += gpu_sample + cpu_sample;
    }

    Ok(HybridStats {
        wall_time: wall_start.elapsed(),
        gpu_samples: split.gpu_samples,
        cpu_samples: split.cpu_samples,
        gpu_samples_per_sec: samples_per_sec(num_pixels, split.gpu_samples, gpu_elapsed),
        cpu_samples_per_sec: samples_per_sec(num_pixels, split.cpu_samples, cpu_elapsed),
        forced_cpu_only,
    })
}

/// Samples/sec for one engine, `0.0` if it traced no samples at all (rather than
/// reporting a meaningless rate for zero work).
fn samples_per_sec(num_pixels: usize, samples: u32, elapsed: Duration) -> f64 {
    if samples == 0 {
        return 0.0;
    }
    let total_samples = num_pixels as f64 * f64::from(samples);
    total_samples / elapsed.as_secs_f64().max(f64::EPSILON)
}

/// The CPU-engine mirror of [`GpuFrameRenderer::accumulate`].
///
/// Traces `spp` samples per pixel starting at sample index `sample_offset`, using the
/// exact per-sample seed/jitter construction the GPU shader draws from for the same
/// indices (see the module doc comment), and ADDS each pixel's summed radiance into
/// `accum` -- same accumulate-not-overwrite convention, same `sample_offset` meaning, as
/// its GPU counterpart.
///
/// [`render_hybrid`]'s own CPU tracing, and its biaxial "route everything to CPU" case,
/// both go through this exact function (via [`cpu_trace_range`]) -- so a caller building
/// an isolated CPU-only measurement to check against a hybrid render (as
/// `examples/hybrid_bench.rs`'s correctness check does) exercises the same code path
/// `render_hybrid` itself uses, not a lookalike that could silently drift from it.
///
/// # Panics
///
/// Panics if `accum.len()` is not `scene.width * scene.height`.
pub fn cpu_accumulate(scene: &GpuFrameScene<'_>, sample_offset: u32, spp: u32, accum: &mut [Vec3]) {
    let num_pixels = scene.width as usize * scene.height as usize;
    assert_eq!(
        accum.len(),
        num_pixels,
        "accumulation buffer must have one entry per pixel"
    );
    let sums = cpu_trace_range(scene, sample_offset, spp);
    for (dst, src) in accum.iter_mut().zip(sums) {
        *dst += src;
    }
}

// ---------------------------------------------------------------------------------
// CPU sample construction -- mirrors `renderer::gpu::estimator_check`'s own CPU
// reference loop (`cpu_sample_xyz`/`cpu_samples`) exactly: same seed hash, same
// low-discrepancy stratified jitter, same Cranley-Patterson rotation, same hero-
// wavelength draw, now both drawn from `optics::raytracer::sampling::{pixel_rotations,
// sample_draws}` rather than each hand-copying the arithmetic inline. See the module
// doc comment for why `estimator_check`'s own copy of this construction still isn't
// imported from here.
// ---------------------------------------------------------------------------------

/// One `(pixel, sample_num)` sample, traced through the real
/// `optics::raytracer::trace_spectral_ray_with_finish` -- never a reimplementation of the
/// estimator, only of the per-sample seed/jitter construction around it.
fn cpu_sample_xyz(scene: &GpuFrameScene<'_>, pixel: u32, sample_num: u32) -> Vec3 {
    let width = scene.width;
    let x = pixel % width;
    let y = pixel / width;

    let rot = pixel_rotations(pixel);
    let draws = sample_draws(pixel, sample_num, &rot);

    let ray = scene.camera.generate_ray(
        x as f32,
        y as f32,
        width as f32,
        scene.height as f32,
        draws.jitter_x,
        draws.jitter_y,
    );
    trace_spectral_ray_with_finish(
        ray,
        scene.planes,
        scene.facet_finishes,
        scene.material,
        scene.max_bounces,
        scene.environment,
        draws.seed,
        draws.hero_rand,
        None,
    )
}

/// Traces `spp` samples per pixel on the CPU, sample indices `[sample_offset,
/// sample_offset + spp)`, across every available core, and returns one SUMMED `Vec3` per
/// pixel (never an average -- see the module doc comment's "Merge convention" section).
///
/// Pixels are partitioned across threads INTERLEAVED (thread `t` owns pixels `t`, `t +
/// num_threads`, `t + 2*num_threads`, ...) rather than in contiguous blocks, so that a
/// spatially clustered cost difference (a facet that scatters more, a region that misses
/// the gem entirely) does not pile all of its extra work onto whichever thread happened
/// to own that block.
fn cpu_trace_range(scene: &GpuFrameScene<'_>, sample_offset: u32, spp: u32) -> Vec<Vec3> {
    let num_pixels = scene.width as usize * scene.height as usize;
    let mut out = vec![Vec3::ZERO; num_pixels];
    if spp == 0 || num_pixels == 0 {
        return out;
    }

    let num_threads = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .min(num_pixels);

    let partials: Vec<Vec<(usize, Vec3)>> = std::thread::scope(|s| {
        // Collected into a `Vec` deliberately: every thread must be SPAWNED before any
        // is joined, or the "concurrent" trace would silently serialize into spawn,
        // join, spawn, join, ... one worker at a time.
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_idx| {
                s.spawn(move || {
                    let mut local = Vec::with_capacity(num_pixels / num_threads + 1);
                    let mut pixel = thread_idx;
                    while pixel < num_pixels {
                        let mut sum = Vec3::ZERO;
                        for local_sample in 0..spp {
                            let sample_num = sample_offset + local_sample;
                            sum += cpu_sample_xyz(scene, pixel as u32, sample_num);
                        }
                        local.push((pixel, sum));
                        pixel += num_threads;
                    }
                    local
                })
            })
            .collect();

        let mut partials = Vec::with_capacity(handles.len());
        for handle in handles {
            partials.push(
                handle
                    .join()
                    .expect("hybrid CPU trace worker thread panicked"),
            );
        }
        partials
    });

    for part in partials {
        for (pixel, sum) in part {
            out[pixel] = sum;
        }
    }
    out
}

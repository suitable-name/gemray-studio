//! Phase 2: end-to-end spectral estimator self-tests -- driven by
//! `shaders/spectral_transport.wgsl`'s `transport_main` entry point.
//!
//! - [`run_determinism`]: two dispatches against identical input, byte-for-byte.
//! - [`run_furnace`]: the energy-conservation furnace anchor -- a colourless,
//!   non-dispersive, non-absorbing cubic gem (the real 57-facet Standard Round
//!   Brilliant, `geometry::cuts::StandardGemCuts::standard_round_brilliant`) inside a
//!   uniform (direction-independent) environment must return exactly that uniform
//!   radiance in expectation. This exercises Fresnel/TIR/Russian-roulette/spectral-MIS
//!   against a TRUTH anchor (not merely CPU-vs-GPU), the same way Phase 1's furnace
//!   anchor did for the geometry/environment machinery.
//! - [`run_image_comparison`]: Tier 3 statistical image equivalence -- Welford mean/M2
//!   on CPU and GPU DISJOINT sample ranges (as production renders would split work),
//!   per-pixel z-score, and connected-component clustering of failing pixels (a
//!   structured bias clusters; noise salts-and-peppers) -- against a real dispersive,
//!   absorbing cubic gem (Spinel) under the analytic studio lighting rig.
//! - [`run_spectral_debug`]: see that function's own doc comment for the honest scope
//!   limit on what this can and cannot verify given the CPU visibility-only constraint.
//!
//! This module is split into one file per topic; see each submodule's own doc comment
//! for what it owns. This file keeps everything shared across them: the test
//! scenes/materials, the shared GPU dispatch, the CPU reference sample generation, and
//! the Welford/z-score statistics.

use crate::{
    geometry::{
        GpuFacetPlane,
        cuts::{STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS, StandardGemCuts},
    },
    optics::{
        absorption::AbsorptionTensor,
        dispersion::DispersionModel,
        materials::{CrystalSystem, GemMaterial, OpticalCharacter},
        raytracer::{
            Camera, EnvironmentSource, FacetFinish, HERO_WAVELENGTH_ROTATION_STREAM,
            PIXEL_JITTER_X_ROTATION_STREAM, PIXEL_JITTER_Y_ROTATION_STREAM,
            cranley_patterson_rotate, hash_u32, low_discrepancy_base2, radical_inverse_base,
            trace_spectral_ray_with_finish,
        },
    },
    renderer::{
        buffers::{GpuCameraParams, GpuGemMaterial, GpuTransportParams, encode_facet_finishes},
        gpu::compute,
    },
};
use glam::Vec3;

mod determinism;
mod furnace;
mod image_comparison;
mod spectral_debug;

pub use determinism::{DeterminismResult, run_determinism};
pub use furnace::{
    FurnaceResult, run_furnace, run_furnace_edge_rounding, run_furnace_frosted_girdle,
    run_furnace_scattering,
};
pub use image_comparison::{
    ImageComparisonResult, run_image_comparison, run_image_comparison_absorption_path_scale,
    run_image_comparison_alexandrite, run_image_comparison_biaxial_scattering,
    run_image_comparison_edge_rounding, run_image_comparison_frosted_girdle,
    run_image_comparison_scattering, run_image_comparison_tanzanite, run_image_comparison_topaz,
    run_image_comparison_tourmaline, run_image_comparison_zircon,
    run_specialisation_image_comparison,
};
pub use spectral_debug::{SpectralDebugResult, run_spectral_debug};

// `spectral_transport.wgsl` alone is not valid WGSL any more -- it assumes
// `shaders/transport_physics.wgsl`'s functions are already in scope. `build.rs`
// concatenates the two into `$OUT_DIR/spectral_transport.generated.wgsl`, which is what
// gets compiled here; see `transport_physics.wgsl`'s header comment for why. Re-exported
// from `frame` rather than `include_str!`d a second time, so this check and the renderer
// provably compile the same source text.
use crate::renderer::gpu::frame::{self, SHADER_SRC};

// ---------------------------------------------------------------------------------
// Test scenes.
// ---------------------------------------------------------------------------------

/// The furnace anchor's gem.
///
/// Colourless (empty band set -> zero absorption at every wavelength), non-dispersive
/// (`Cauchy { a, b: 0, c: 0 }` is a wavelength-independent constant index --
/// `DispersionModel::evaluate`'s `Cauchy` branch reduces to bare `a` when `b == c ==
/// 0.0`), cubic (`birefringence_delta: 0.0`, `crystal_system: Cubic`). `n = 1.5` is
/// chosen well above 1.0 (so Fresnel reflection/TIR are genuinely exercised, not
/// trivially zero) and comfortably below any real gem's critical-angle edge cases this
/// module's Tier 2 sibling already probes directly.
#[must_use]
pub fn furnace_material() -> GemMaterial {
    GemMaterial {
        name: "Phase2 furnace anchor (colourless n=1.5 cubic)".to_string(),
        crystal_system: CrystalSystem::Cubic,
        optical_character: OpticalCharacter::Isotropic,
        dispersion: DispersionModel::Cauchy {
            a: 1.5,
            b: 0.0,
            c: 0.0,
        },
        birefringence_delta: 0.0,
        absorption: AbsorptionTensor::isotropic(vec![]),
        c_axis: Vec3::Y,
        biaxial_delta_beta_alpha: None,
        scattering_sigma_s: 0.0,
        scattering_g: 0.0,
        edge_rounding_radius: 0.0,
        absorption_path_scale: 1.0,
    }
}

/// Tier 3's gem.
///
/// The real built-in "Spinel" (cubic, genuinely dispersive Sellmeier3 fit, genuinely
/// absorbing via `legacy_rgb_bands`) -- exercises dispersion-driven spectral MIS
/// ("fire") and pleochroic-formula absorption together, not just the furnace's
/// colourless/non-dispersive degenerate case.
///
/// # Panics
///
/// Panics if `"Spinel"` is ever removed from `GemMaterial::all_materials()` -- this is
/// self-test scaffolding, not a code path a real caller can reach with a name that
/// might legitimately be missing.
#[must_use]
pub fn tier3_material() -> GemMaterial {
    GemMaterial::all_materials()
        .into_iter()
        .find(|m| m.name == "Spinel")
        .expect("\"Spinel\" is a built-in cubic material in GemMaterial::all_materials()")
}

#[must_use]
pub fn test_camera() -> Camera {
    // distance=5.0 against the ~1-unit-scale Standard Round Brilliant (girdle radius
    // ~1.0, table facet at y=0.32 -- see `polyhedron_check`'s own case bank), narrow
    // enough fov that most of the frame's rays actually hit the gem rather than testing
    // only the trivial miss branch.
    Camera::new(0.35, 0.28, 5.0, 18.0)
}

fn round_brilliant_planes() -> Vec<GpuFacetPlane> {
    StandardGemCuts::standard_round_brilliant()
}

/// GPU port (frosted girdle finish): an all-`Polished` `facet_finish::*`-valued
/// buffer sized to `planes.len()` -- what every existing (pre-Task-2) GPU self-test in
/// this module dispatches with, so the megakernel's `finish == FACET_FINISH_FROSTED`
/// branch is never taken and every one of those checks keeps exercising EXACTLY its
/// pre-Task-2 code path.
#[must_use]
fn all_polished_finishes(num_planes: usize) -> Vec<u32> {
    encode_facet_finishes(&[], num_planes)
}

/// GPU port (frosted girdle finish): `FacetFinish::Polished` everywhere except
/// the girdle band (`STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS`), which is
/// `FacetFinish::Frosted` -- the single CPU-side source of truth this module's frosted
/// GPU self-tests build both their CPU reference (`trace_spectral_ray_with_finish`) and
/// their GPU-encoded (`encode_facet_finishes`) buffer from, mirroring
/// `tests/raytracer_tests.rs`'s own identically-named, identically-constructed helper.
#[must_use]
fn bruted_girdle_finishes(num_planes: usize) -> Vec<FacetFinish> {
    let mut finishes = vec![FacetFinish::Polished; num_planes];
    for i in STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS {
        finishes[i] = FacetFinish::Frosted;
    }
    finishes
}

/// Phase 3: Zircon.
///
/// The largest birefringence in the built-in material set (`birefringence_delta =
/// +0.0590`), positive uniaxial. Exercises the `theta_c` iteration, the 50/50
/// ordinary/extraordinary eigenmode split, and `extraordinary_poynting_dir`'s walk-off,
/// all at the strongest birefringence this crate ships.
///
/// # Panics
///
/// Panics if `"Zircon"` is ever removed from `GemMaterial::all_materials()` -- see
/// [`tier3_material`]'s doc comment for the same self-test-scaffolding rationale.
#[must_use]
pub fn zircon_material() -> GemMaterial {
    GemMaterial::all_materials()
        .into_iter()
        .find(|m| m.name == "Zircon")
        .expect("\"Zircon\" is a built-in uniaxial material in GemMaterial::all_materials()")
}

/// Phase 3: Tourmaline.
///
/// Strongly NEGATIVE uniaxial (`birefringence_delta = -0.0210`), the sign opposite
/// Zircon's, so the two together exercise both branches of
/// `effective_extraordinary_index`'s `n_e > n_o` / `n_e < n_o` behaviour.
///
/// # Panics
///
/// Panics if `"Tourmaline"` is ever removed from `GemMaterial::all_materials()` -- see
/// [`tier3_material`]'s doc comment for the same self-test-scaffolding rationale.
#[must_use]
pub fn tourmaline_material() -> GemMaterial {
    GemMaterial::all_materials()
        .into_iter()
        .find(|m| m.name == "Tourmaline")
        .expect("\"Tourmaline\" is a built-in uniaxial material in GemMaterial::all_materials()")
}

/// Phase 4 (biaxial GPU port): Alexandrite.
///
/// Genuinely biaxial with a full three-independent-band-set (`beta_ray`) absorption
/// tensor -- exercises `biaxial_wave_indices`/`biaxial_eigen_polarizations`/
/// `biaxial_mode_poynting_dir`/`biaxial_resolve_entry_mode` end to end through the real
/// megakernel dispatch, AND `pleochroic_channel_alpha_biaxial`'s three-coefficient
/// absorption path.
///
/// # Panics
///
/// Panics if `"Alexandrite"` is ever removed from `GemMaterial::all_materials()` -- see
/// [`tier3_material`]'s doc comment for the same self-test-scaffolding rationale.
#[must_use]
pub fn alexandrite_material() -> GemMaterial {
    GemMaterial::all_materials()
        .into_iter()
        .find(|m| m.name == "Alexandrite")
        .expect("\"Alexandrite\" is a built-in biaxial material in GemMaterial::all_materials()")
}

/// Phase 4 (biaxial GPU port): Topaz.
///
/// Genuinely biaxial (three distinct principal indices) but STILL using the two-band-set
/// (`o_ray`/`e_ray`) absorption approximation (`beta_ray = None`) -- exercises the
/// biaxial INDEX path (mode-A/mode-B directions, walk-off) with the material's own
/// eigenmode directions feeding the pre-existing uniaxial two-coefficient
/// `pleochroic_channel_alpha`, the other combination the megakernel's `is_biaxial &&
/// has_beta_ray` branch has to get right (see `spectral_transport.wgsl`'s absorption
/// block).
///
/// # Panics
///
/// Panics if `"Topaz"` is ever removed from `GemMaterial::all_materials()` -- see
/// [`tier3_material`]'s doc comment for the same self-test-scaffolding rationale.
#[must_use]
pub fn topaz_material() -> GemMaterial {
    GemMaterial::all_materials()
        .into_iter()
        .find(|m| m.name == "Topaz")
        .expect("\"Topaz\" is a built-in biaxial material in GemMaterial::all_materials()")
}

/// Phase 4 (biaxial GPU port): Tanzanite.
///
/// Genuinely biaxial with its own full three-band-set absorption tensor (distinct
/// numbers, distinct `birefringence_delta`/`biaxial_delta_beta_alpha` sign/magnitude
/// from Alexandrite) -- a second real trichroic material exercising the same code path
/// as Alexandrite with different data, the same "two materials on both sides of a sign"
/// coverage principle Zircon/Tourmaline apply to uniaxial birefringence.
///
/// # Panics
///
/// Panics if `"Tanzanite"` is ever removed from `GemMaterial::all_materials()` -- see
/// [`tier3_material`]'s doc comment for the same self-test-scaffolding rationale.
#[must_use]
pub fn tanzanite_material() -> GemMaterial {
    GemMaterial::all_materials()
        .into_iter()
        .find(|m| m.name == "Tanzanite")
        .expect("\"Tanzanite\" is a built-in biaxial material in GemMaterial::all_materials()")
}

// ---------------------------------------------------------------------------------
// Shared GPU dispatch.
// ---------------------------------------------------------------------------------

struct TransportDispatch {
    xyz: Vec<f32>,
    radiance: Vec<f32>,
    lambdas: Vec<f32>,
    path_pdf: Vec<f32>,
}

/// Dispatches `transport_main` once over `total_tuples` (pixel, sample) threads and
/// reads back all four output buffers (final XYZ plus the pre-integration per-channel
/// debug arrays -- see `shaders/spectral_transport.wgsl`'s doc comment for why a single
/// entry point always writes all four rather than two entry points with different
/// binding subsets).
///
/// Always the GENERIC (`MATERIAL_CLASS = 0`) pipeline -- see
/// [`dispatch_transport_for_class`] for the specialised-pipeline variant Tier 3's image
/// comparisons use instead. Every OTHER caller here (furnace anchors, determinism,
/// spectral debug) deliberately keeps exercising the general runtime-dispatch kernel,
/// not any one material class's specialised pipeline.
fn dispatch_transport(
    ctx: &crate::renderer::gpu::GpuContext,
    camera_params: &GpuCameraParams,
    params: &GpuTransportParams,
    material: &GpuGemMaterial,
    planes: &[GpuFacetPlane],
    facet_finishes: &[u32],
    total_tuples: usize,
) -> TransportDispatch {
    dispatch_transport_for_class(
        ctx,
        camera_params,
        params,
        MaterialForDispatch {
            encoded: material,
            class: frame::material_class::GENERIC,
        },
        planes,
        facet_finishes,
        total_tuples,
    )
}

/// Bundles an encoded material with the `MATERIAL_CLASS` pipeline-overridable constant
/// its dispatch should use -- purely to keep [`dispatch_transport_for_class`]'s argument
/// count within clippy's `too_many_arguments` limit (the same reason
/// `renderer::gpu::frame::GpuFrameScene` bundles a scene's fields), grouping two values
/// that are conceptually paired anyway: which material, and which specialised pipeline
/// [`frame::classify_material`] (or, for [`dispatch_transport`], the fixed GENERIC
/// value) says to render it through.
///
/// `Copy`: a reference plus a `u32`, so passing it by value (as
/// [`dispatch_transport_for_class`] does) is exactly as cheap as passing a reference
/// to it, and marking it `Copy` is clippy's own suggested fix for that -- taking it
/// by value keeps the call sites reading as "here is the material to dispatch",
/// rather than a reference to a reference.
#[derive(Clone, Copy)]
struct MaterialForDispatch<'a> {
    encoded: &'a GpuGemMaterial,
    class: u32,
}

/// Like [`dispatch_transport`], but compiles the pipeline with `spectral_transport.wgsl`'s
/// `MATERIAL_CLASS` override fixed to `material.class` (one of `frame::material_class`'s
/// values) rather than left at its GENERIC default -- [`dispatch_transport`] itself is
/// exactly this function called with `frame::material_class::GENERIC`, so every
/// pre-existing caller is bit-for-bit unaffected.
///
/// [`image_comparison::run_image_comparison_for`] is this function's only caller with a
/// non-GENERIC class: routing Tier 3's statistical image comparisons through the SAME
/// specialised pipeline `GpuFrameRenderer::accumulate` would pick for that material
/// (via `frame::classify_material`) is what makes those checks exercise the production
/// path, not a lookalike GENERIC dispatch -- see `frame`'s module doc comment.
fn dispatch_transport_for_class(
    ctx: &crate::renderer::gpu::GpuContext,
    camera_params: &GpuCameraParams,
    params: &GpuTransportParams,
    material: MaterialForDispatch<'_>,
    planes: &[GpuFacetPlane],
    facet_finishes: &[u32],
    total_tuples: usize,
) -> TransportDispatch {
    let pipeline = compute::create_compute_pipeline_with_constants(
        &ctx.device,
        "transport_main",
        SHADER_SRC,
        "transport_main",
        &[("MATERIAL_CLASS", f64::from(material.class))],
    );
    let material = material.encoded;
    let outputs = frame::TransportOutputs::new(&ctx.device, total_tuples);
    frame::encode_and_dispatch(
        &frame::TransportDispatchArgs {
            ctx,
            pipeline: &pipeline,
            camera_params,
            params,
            material,
            planes,
            facet_finishes,
            outputs: &outputs,
        },
        total_tuples,
    );
    TransportDispatch {
        xyz: compute::readback(&ctx.device, &ctx.queue, outputs.xyz(), total_tuples * 3),
        radiance: compute::readback(
            &ctx.device,
            &ctx.queue,
            outputs.radiance(),
            total_tuples * 8,
        ),
        lambdas: compute::readback(&ctx.device, &ctx.queue, outputs.lambdas(), total_tuples * 8),
        path_pdf: compute::readback(
            &ctx.device,
            &ctx.queue,
            outputs.path_pdf(),
            total_tuples * 8,
        ),
    }
}

const fn camera_params_for(
    camera: &Camera,
    width: u32,
    height: u32,
    samples: u32,
) -> GpuCameraParams {
    GpuCameraParams {
        origin: camera.origin.to_array(),
        fov_tan: camera.fov_tan,
        forward: camera.forward.to_array(),
        width: width as f32,
        right: camera.right.to_array(),
        height: height as f32,
        up: camera.up.to_array(),
        num_samples: samples,
    }
}

// ---------------------------------------------------------------------------------
// CPU reference sample generation -- mirrors
// `apps/gemray-worker/src/render_core.rs`'s exact seed/jitter/hero-wavelength
// derivation (stratified via `low_discrepancy_base2` +
// `cranley_patterson_rotate`, not a plain `hash_u32` uniform), calling the real
// `trace_spectral_ray` directly (never a reimplementation of the estimator).
// ---------------------------------------------------------------------------------

/// Bundles the fixed-for-a-whole-run scene inputs [`cpu_sample_xyz`] needs, purely to
/// keep that function's argument count within clippy's `too_many_arguments` limit --
/// every field here is set once by the caller and never changes across the many
/// `(pixel, sample)` calls a threaded CPU reference sweep makes.
#[derive(Clone, Copy)]
struct CpuScene<'a> {
    camera: &'a Camera,
    width: u32,
    height: u32,
    planes: &'a [GpuFacetPlane],
    /// GPU port (frosted girdle finish): `&[]` (every existing caller) is
    /// exactly equivalent to the pre-Task-2 all-`Polished` `trace_spectral_ray` this
    /// used to call directly (see [`trace_spectral_ray_with_finish`]'s own doc
    /// comment) -- so switching every CPU reference sample in this module to go
    /// through it unconditionally is a no-op for every check except the new frosted
    /// girdle ones, while also making "all-Polished is bit-identical" an load-bearing
    /// property of the shared test harness itself, not just a CPU-only regression test.
    facet_finishes: &'a [FacetFinish],
    material: &'a GemMaterial,
    max_bounces: u32,
}

fn cpu_sample_xyz(
    scene: &CpuScene<'_>,
    pixel: u32,
    sample_num: u32,
    environment: EnvironmentSource<'_>,
) -> Vec3 {
    let CpuScene {
        camera,
        width,
        height,
        planes,
        facet_finishes,
        material,
        max_bounces,
    } = *scene;
    let x = pixel % width;
    let y = pixel / width;
    let seed = hash_u32(pixel.wrapping_mul(0x9e37_79b9) ^ sample_num.wrapping_mul(0x85eb_ca6b));

    // Stratified pixel jitter and hero wavelength -- see
    // `apps/gemray-worker/src/render_core.rs::trace_samples` (the production
    // reference this mirrors) for the full rationale.
    let rot_jx = low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_X_ROTATION_STREAM));
    let rot_jy = low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_Y_ROTATION_STREAM));
    let rot_hero = low_discrepancy_base2(hash_u32(pixel ^ HERO_WAVELENGTH_ROTATION_STREAM));
    let jx = cranley_patterson_rotate(low_discrepancy_base2(sample_num), rot_jx) - 0.5;
    let jy = cranley_patterson_rotate(radical_inverse_base(sample_num, 3), rot_jy) - 0.5;
    let hero_rand = cranley_patterson_rotate(radical_inverse_base(sample_num, 5), rot_hero);

    let ray = camera.generate_ray(x as f32, y as f32, width as f32, height as f32, jx, jy);
    trace_spectral_ray_with_finish(
        ray,
        planes,
        facet_finishes,
        material,
        max_bounces,
        environment,
        seed,
        hero_rand,
        None,
    )
}

/// Threaded CPU reference: `num_pixels * samples_per_pixel` samples, pixel-major order
/// (matching the GPU output buffer's `idx = pixel * samples_per_dispatch + local_sample`
/// layout), sample indices `[sample_offset, sample_offset + samples_per_pixel)`.
fn cpu_samples<F>(
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    sample_offset: u32,
    trace_one: F,
) -> Vec<Vec3>
where
    F: Fn(u32, u32) -> Vec3 + Sync,
{
    let num_pixels = width * height;
    let num_threads = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .min(16);
    let mut out = vec![Vec3::ZERO; (num_pixels as usize) * (samples_per_pixel as usize)];
    let rows_per_chunk = (num_pixels as usize).div_ceil(num_threads);
    std::thread::scope(|s| {
        let chunks: Vec<&mut [Vec3]> = out
            .chunks_mut(rows_per_chunk * samples_per_pixel as usize)
            .collect();
        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let start_pixel = (chunk_idx * rows_per_chunk) as u32;
            let trace_one = &trace_one;
            s.spawn(move || {
                for (local_pixel, pixel_slot) in
                    chunk.chunks_mut(samples_per_pixel as usize).enumerate()
                {
                    let pixel = start_pixel + local_pixel as u32;
                    if pixel >= num_pixels {
                        break;
                    }
                    for (local_sample, slot) in pixel_slot.iter_mut().enumerate() {
                        let sample_num = sample_offset + local_sample as u32;
                        *slot = trace_one(pixel, sample_num);
                    }
                }
            });
        }
    });
    out
}

// ---------------------------------------------------------------------------------
// Welford online mean/M2, in f64 for accumulator stability (this is an analysis tool
// applied AFTER the f32 estimator produced its samples -- it does not change what is
// being measured).
// ---------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 = delta.mul_add(delta2, self.m2);
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n as f64 - 1.0)
        }
    }

    fn standard_error_sq(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.variance() / self.n as f64
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct WelfordXyz {
    x: Welford,
    y: Welford,
    z: Welford,
}

impl WelfordXyz {
    fn update(&mut self, v: Vec3) {
        self.x.update(f64::from(v.x));
        self.y.update(f64::from(v.y));
        self.z.update(f64::from(v.z));
    }

    const fn mean(&self) -> Vec3 {
        Vec3::new(self.x.mean as f32, self.y.mean as f32, self.z.mean as f32)
    }
}

fn z_score(cpu: &Welford, gpu: &Welford) -> f64 {
    let se2 = cpu.standard_error_sq() + gpu.standard_error_sq();
    if se2 <= 0.0 {
        return 0.0;
    }
    (gpu.mean - cpu.mean) / se2.sqrt()
}

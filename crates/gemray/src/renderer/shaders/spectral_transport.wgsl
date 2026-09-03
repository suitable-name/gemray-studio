// Phase 2/3/4: the full isotropic + uniaxial-birefringent + biaxial-birefringent
// spectral estimator -- driven by `renderer::gpu::estimator_check` (Tier 3 statistical
// image comparison, the energy-conservation furnace anchor, and the spectral-space
// debug comparison) and `renderer::gpu::transport_check` (Tier 2 per-function ULP
// budgets on the small pieces below, exercised standalone via
// `shaders/transport_functions.wgsl`).
//
// A fresh translation of `optics::raytracer::trace_spectral_ray` as it exists today.
// Phase 4 ports `hero_biaxial_wave_dirs`/`per_channel_biaxial_indices`/
// `resolve_entry_mode`/`mode_poynting_dir`/`eigen_polarizations` (the
// `BiaxialIndicatrix`-only machinery, see `shaders/transport_physics.wgsl`'s own Phase 4
// section) and wires them into this kernel's mode-A/mode-B dispatch below, generalizing
// the uniaxial ordinary/extraordinary split established by Phase 3. Whether this port
// is actually TRUSTED for a real render is a SEPARATE decision governed entirely by
// `optics::materials::GemMaterial::gpu_supported` -- that routing predicate, not
// anything in this file, is what a caller must consult before ever handing a biaxial
// scene to this kernel; see that function's own doc comment for the current state of
// that decision (verified-and-enabled, or still CPU-only pending verification -- read
// its doc comment, don't infer either way from this shader existing). For a CUBIC
// (isotropic) material `is_anisotropic` is additionally
// always `false` (`crystal_system != Cubic && |birefringence_delta| > 1e-4`), which
// further collapses `theta_c_for_bounce` to its else branch (`n_eff_ch[k] == n_o_ch[k]`
// unconditionally), `entering_anisotropic` to always `false` (so the ordinary/
// extraordinary eigenmode split and walk-off never occur -- the `entering_anisotropic`
// guard around the mode-selection draw is simply never entered, not merely a no-op
// division, now that the surviving mode's throughput is left unscaled rather than
// divided by a `split_pdf` that could be a reachable non-1.0 constant), and
// `is_extraordinary` to always-irrelevant -- reducing this kernel back to exactly Phase
// 2's isotropic-only behaviour, bit-for-bit; see this file's git history / Phase 3 diff
// for the by-construction argument. What is exercised
// regardless of `is_anisotropic` (see `optics::birefringence::effective_pleochroic_alpha`'s
// own doc comment: "An isotropic tensor ... returns the same value regardless of e_hat/
// eigenmode ... so isotropic materials are automatically azimuth-independent with no
// special case needed"): the full pleochroic Beer-Lambert absorption path
// (`electric_field_direction`, `ordinary_eigen_polarization`,
// `extraordinary_eigen_polarization`, the uniaxial `AbsorptionTensor3` quadratic form,
// `effective_pleochroic_alpha`) is ported faithfully rather than shortcut to a bare
// `spectral_absorption(lambda)`, so this kernel reproduces the CPU's actual rounding on
// that path, not merely its mathematically-equal isotropic limit.
//
// Phase 3 additions (uniaxial only): the `theta_c` fixed-point iteration
// (`theta_c_for_bounce`), the per-channel ordinary/effective-extraordinary index pair
// (`per_channel_uniaxial_index`), the 50/50 ordinary/extraordinary eigenmode split on
// air->crystal entry (drawn from `BIREFRINGENT_SPLIT_STREAM`; each mode already
// carries only its own ~0.5 share of the incident energy, and that share is drawn
// with the SAME 0.5 probability it is weighted by, so the surviving branch is left
// UNSCALED -- unlike the Fresnel reflect/transmit split below, whose two branches are
// mutually exclusive fates for the FULL incident beam and so genuinely do need
// dividing back out by their own selection probability; see `apply_refract_bounce`'s
// doc comment on the CPU side for the full reasoning), and the extraordinary
// ray's walk-off direction (`extraordinary_poynting_dir`). Chromatic termination (a
// companion channel k's own refracted/walk-off direction failing to match the shared
// hero-driven direction to within `DIRECTION_MATCH_COS_TOL`) is now also reachable via a
// dispersive WALK-OFF mismatch, not just a dispersive Snell-angle mismatch -- see
// `apply_refract_channel`'s doc comment on the CPU side for why the direction compared
// against must be the STORED `final_refr_dir`/`final_dir_k`, never a value recomputed a
// second time (the "direction-match identity trap": a few-ULP-different recomputation of
// the SAME direction fails its own match against itself, chromatically self-terminating
// every channel).
//
// Phase 4 additions (biaxial, `is_biaxial := material.dispersion.has_biaxial_delta !=
// 0u`): "mode A" and "mode B" generalize "ordinary" and "extraordinary" to a crystal
// with no single optic axis -- mode A is the FASTER (lower-index) root of
// `biaxial_wave_indices`, mode B the SLOWER (higher-index) root, at whichever
// wave-normal direction is relevant; `is_extraordinary` (`false` selects mode A, `true`
// selects mode B) is the exact same per-ray local Phase 3 already threads through TIR/
// reflect/internal-mode-coupling, reused unchanged. Unlike the uniaxial ordinary ray,
// NEITHER biaxial mode has a direction-independent index, so an air->crystal entry
// resolves BOTH modes' wave-normal directions via `biaxial_resolve_entry_mode`'s
// two-iteration fixed point (once each), evaluated once from the HERO channel and
// shared by every companion channel (exactly mirroring `theta_c`'s "one shared
// direction, per-channel index magnitude" structure) -- see `n_alpha_hero`/
// `n_beta_hero`/`n_gamma_hero`/`biax_ax0`/`biax_ax1`/`biax_ax2`/`wave_dir_a_hero`/
// `wave_dir_b_hero` below. Both biaxial modes walk off (`biaxial_mode_poynting_dir`),
// unlike uniaxial's ordinary ray, which never does. The pleochroic absorption path
// additionally consults the material's third (`beta_ray`) band set, when present, via
// `pleochroic_channel_alpha_biaxial`'s three-independent-coefficient quadratic form,
// with the two eigenmode directions sourced from `biaxial_eigen_polarizations` instead
// of the uniaxial ordinary/extraordinary approximation whenever `is_biaxial`.
//
// Every stochastic decision (the Fresnel reflect/transmit branch, Russian roulette)
// uses its own LOCALLY COMPUTED probability for both the branch comparison and the
// compensating division -- see the module-level task brief's "structural rule that
// makes float divergence harmless". Precision is `f32` throughout; `f32::mul_add`
// mirrors WGSL `fma`, and `powi` (none needed in the isotropic path) would be ported as
// a multiplication chain, never `pow()`.
//
// A single entry point (`transport_main`) always writes all four output buffers (final
// XYZ, and -- Requirement: "expose per-wavelength-channel radiance before XYZ
// integration in a debug mode" -- the pre-integration per-channel radiance/lambda/
// path_pdf arrays) rather than two entry points with different binding subsets: WGSL's
// per-entry-point auto bind-group-layout inference is based on static reachability, not
// runtime branching, so a `write_debug: bool` parameter to a shared helper function
// would still force BOTH entry points to bind all 8 buffers anyway. `estimator_check`'s
// large statistical dispatches simply allocate (and never read back) the debug buffers.
//
// Per-thread state (8 `vec4<f32>` Stokes vectors + 8 `f32` path_pdf + 8 `f32` lambdas +
// ray origin/dir + a handful of loop scalars, ~60-70 floats) lives entirely in this
// function's local variables -- no cross-thread communication, no atomics, one thread
// per (pixel, sample) tuple writing only its own output slot. This is what makes two
// dispatches against identical input byte-identical (see `estimator_check::run_determinism`):
// GPU scheduling order can never matter when no thread ever reads another thread's
// output.

// Kernel specialisation (perf task, 2026-09-02): a pipeline-overridable constant --
// see WGSL's `override` declarations -- resolved at PIPELINE creation time via
// `wgpu::PipelineCompilationOptions::constants` (`renderer::gpu::compute::
// create_compute_pipeline_with_constants`), not at shader-module-parse time like a
// plain `const`. `renderer::gpu::frame::GpuFrameRenderer` compiles one specialised
// pipeline per class (see that module's doc comment) plus the GENERIC pipeline every
// self-test in this crate keeps using unmodified (`MATERIAL_CLASS`'s declared default,
// 0u, is exactly the GENERIC value, so every dispatch that never sets this override --
// which is every dispatch before this task -- is bit-for-bit unchanged).
//
// The override feeds ONLY the `is_anisotropic`/`is_biaxial` derivations below, never a
// duplicate copy of any formula: for MATERIAL_CLASS_ISOTROPIC, `is_anisotropic` is
// forced `false` regardless of the material buffer's own flag, which is EXACTLY the
// condition this file's header comment already proves collapses this kernel to Phase
// 2's isotropic-only behaviour bit-for-bit -- so the isotropic pipeline doesn't need a
// second proof, it reuses that one. For MATERIAL_CLASS_UNIAXIAL and
// MATERIAL_CLASS_ISOTROPIC, `is_biaxial` is forced `false`, which is exactly the
// condition every biaxial-only per-ray array below (`biax_ax0`/`ax1`/`ax2`,
// `n_alpha_hero`/`n_gamma_hero`, `wave_dir_a_hero`/`wave_dir_b_hero`,
// `n_biax_a_ch`/`n_biax_b_ch`/`n_alpha_ch`/`n_gamma_ch`, `alpha_beta_hoisted`) is
// populated under and read back under -- once the override substitutes a compile-time
// `false`, every one of those reads is unreachable, so naga/the driver's own dead-code
// elimination can drop the writes, the arrays themselves, and the register space they
// would otherwise hold live across the whole bounce loop, for every render dispatched
// through that specialised pipeline -- a plain diamond no longer carries a biaxial
// crystal's state, and an isotropic render carries neither the biaxial NOR the uniaxial
// (`theta_c`/`n_eff_ch`/eigenmode-split/walk-off) state. Both derivations pass the
// runtime buffer value through UNCHANGED for MATERIAL_CLASS_GENERIC and (for
// `is_biaxial`) MATERIAL_CLASS_BIAXIAL, so the generic pipeline's behaviour -- and every
// self-test that dispatches it -- is provably identical to before this task.
override MATERIAL_CLASS: u32 = 0u;
const MATERIAL_CLASS_GENERIC: u32 = 0u;
const MATERIAL_CLASS_ISOTROPIC: u32 = 1u;
const MATERIAL_CLASS_UNIAXIAL: u32 = 2u;
const MATERIAL_CLASS_BIAXIAL: u32 = 3u;

const NUM_CHANNELS: u32 = 8u;
const SPECTRUM_MIN: f32 = 380.0;
const SPECTRUM_SPAN: f32 = 400.0;
const NORM_FACTOR: f32 = (400.0 / 8.0) / 106.856;
const DIRECTION_MATCH_COS_TOL: f32 = 1.0 - 1e-6;
const RR_FLOOR: f32 = 0.05;
// RAY_EPS, R_UNPOL_MIN, R_UNPOL_MAX, FRESNEL_BRANCH_STREAM, RUSSIAN_ROULETTE_STREAM,
// BIREFRINGENT_SPLIT_STREAM, MODE_COUPLING_STREAM, FROSTED_DIR_U_STREAM,
// FROSTED_DIR_V_STREAM, and hash_u32 now live in `transport_physics.wgsl` (Task 2 GPU
// port moved them there so `apply_frosted_bounce` -- shared verbatim with Tier 2's
// `transport_functions.wgsl` -- has them in scope without a duplicate copy). See that
// file's own comment at their new location.

// optics::raytracer::{BRADFORD_XYZ_TO_LMS, BRADFORD_LMS_TO_XYZ, apply_von_kries_white_balance}
// (Fix 3). `params.white_balance` is `optics::raytracer::compute_illuminant_white_balance`'s
// precomputed Bradford-LMS-space scale (see `renderer::buffers::GpuTransportParams`'s doc
// comment) -- applying it correctly means transforming to this SAME Bradford LMS basis,
// scaling, and transforming back, not multiplying it into XYZ directly. Local to this file
// (not `transport_physics.wgsl`) since `transport_functions.wgsl`'s Tier 2 kernels never
// apply a white balance -- only `compute_illuminant_white_balance` itself is Tier 2 tested,
// via `shaders/environment.wgsl`'s own (separately defined) copy of these same constants.
const BRADFORD_XYZ_TO_LMS = mat3x3<f32>(
    vec3<f32>(0.8951, -0.7502, 0.0389),
    vec3<f32>(0.2664, 1.7135, -0.0685),
    vec3<f32>(-0.1614, 0.0367, 1.0296),
);
const BRADFORD_LMS_TO_XYZ = mat3x3<f32>(
    vec3<f32>(0.986993, 0.432305, -0.008529),
    vec3<f32>(-0.147054, 0.518360, 0.040043),
    vec3<f32>(0.159963, 0.049291, 0.968487),
);

fn apply_von_kries_white_balance(xyz: vec3<f32>, lms_scale: vec3<f32>) -> vec3<f32> {
    let lms = BRADFORD_XYZ_TO_LMS * xyz;
    return BRADFORD_LMS_TO_XYZ * (lms * lms_scale);
}

// ---------------------------------------------------------------------------------
// Struct layouts -- must match `renderer::buffers` field-for-field (see that module's
// doc comment on why a hand-derived offset is never trusted without the echo test).
// ---------------------------------------------------------------------------------

struct GpuCameraParams {
    origin: vec3<f32>,
    fov_tan: f32,
    forward: vec3<f32>,
    width: f32,
    right: vec3<f32>,
    height: f32,
    up: vec3<f32>,
    num_samples: u32,
}

struct GpuTransportParams {
    num_pixels: u32,
    max_bounces: u32,
    sample_offset: u32,
    env_mode: u32,
    l0: f32,
    studio_temp_k: f32,
    studio_spot_mult: f32,
    studio_exposure: f32,
    studio_light_yaw: f32,
    studio_light_pitch: f32,
    pixel_offset: u32,
    // R4: was `_pad1: f32` -- reused (like `pixel_offset` reused `_pad0`) as a flag
    // gating `transport_main`'s three per-channel debug writes; see
    // `renderer::buffers::GpuTransportParams::write_debug_buffers`'s doc comment.
    write_debug_buffers: u32,
    white_balance: vec3<f32>,
}

struct DispersionParams {
    model_type: u32,
    param_a: vec4<f32>,
    param_b: vec4<f32>,
    param_c: vec4<f32>,
    c_axis_and_birefringence: vec4<f32>,
    is_anisotropic: u32,
    biaxial_delta_beta_alpha: f32,
    has_biaxial_delta: u32,
}

struct GpuGemMaterial {
    dispersion: DispersionParams,
    crystal_system: u32,
    optical_character: u32,
    is_pleochroic: u32,
    o_ray_band_count: u32,
    e_ray_band_count: u32,
    o_ray_bands: array<AbsorptionBand, 8>,
    e_ray_bands: array<AbsorptionBand, 8>,
    scattering_sigma_s: f32,
    scattering_g: f32,
    edge_rounding_radius: f32,
    // Phase 4 (biaxial GPU port): see `renderer::buffers::GpuGemMaterial`'s own doc
    // comment for why these are appended here rather than inserted alongside
    // `o_ray_band_count`/`e_ray_band_count`.
    has_beta_ray: u32,
    beta_ray_band_count: u32,
    beta_ray_bands: array<AbsorptionBand, 8>,
    // P1 (absorption path scale): see `renderer::buffers::GpuGemMaterial`'s own doc
    // comment -- appended after `beta_ray_bands` for the same "no shift for any
    // earlier field" reason.
    absorption_path_scale: f32,
}

struct FacetPlane {
    normal: vec3<f32>,
    d: f32,
}

@group(0) @binding(0) var<uniform> camera: GpuCameraParams;
@group(0) @binding(1) var<uniform> params: GpuTransportParams;
@group(0) @binding(2) var<storage, read> material: GpuGemMaterial;
@group(0) @binding(3) var<storage, read> planes: array<FacetPlane>;
@group(0) @binding(4) var<storage, read_write> out_xyz: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_radiance: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_lambdas: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_path_pdf: array<f32>;
// Task 2 GPU port (frosted girdle finish): `optics::raytracer::FacetFinish`, one entry
// per `planes[i]`, PARALLEL to `planes` (a separate binding, not a widened
// `FacetPlane`) -- see `renderer::buffers::facet_finish`'s module doc comment for why. A
// value of `facet_finish::FROSTED` (1u) routes that facet's bounce through
// `apply_frosted_bounce` in `transport_physics.wgsl` instead of the polished TIR/
// reflect/refract dispatch below; any other value (including an out-of-bounds index --
// see the bounds-checked lookup at its use site) is `facet_finish::POLISHED` (0u).
@group(0) @binding(8) var<storage, read> facet_finishes: array<u32>;

const FACET_FINISH_FROSTED: u32 = 1u;

// ---------------------------------------------------------------------------------
// hash_u32 -- optics::raytracer::hash_u32, bit-exact per Phase 0. Now defined in
// `transport_physics.wgsl` (see this file's header note above) -- look there, not here.
// ---------------------------------------------------------------------------------
// Fix 4 -- optics::raytracer::{low_discrepancy_base2, radical_inverse_base,
// cranley_patterson_rotate, PIXEL_JITTER_X_ROTATION_STREAM,
// PIXEL_JITTER_Y_ROTATION_STREAM, HERO_WAVELENGTH_ROTATION_STREAM}, bit-exact per Tier 1
// (see shaders/rng_equivalence.wgsl, which mirrors this same construction for the
// dedicated GPU/CPU RNG bit-exactness self-test).
//
// jx/jy/hero_rand use three DIFFERENT prime bases (2, 3, 5), not the same base rotated
// three ways -- see the matching Rust doc comment (`optics::raytracer`'s Fix 4 section)
// for the measurement showing why: same-base pairing made variance WORSE, not better,
// for exactly the highest-variance pixels this fix targets.
// ---------------------------------------------------------------------------------

const PIXEL_JITTER_X_ROTATION_STREAM: u32 = 0xA511E9B3u;
const PIXEL_JITTER_Y_ROTATION_STREAM: u32 = 0x63D81B23u;
const HERO_WAVELENGTH_ROTATION_STREAM: u32 = 0x1B873593u;

fn low_discrepancy_base2(n: u32) -> f32 {
    return f32(reverseBits(n)) / 4294967296.0;
}

// optics::raytracer::radical_inverse_base -- general prime-base radical inverse (base 2
// uses the faster bit-reversal path above instead). Pure integer digit extraction
// feeding a running float accumulation with plain `+`/`*`//` (no `fma()`), matching the
// CPU side's non-fused `+=`/`/=` exactly.
fn radical_inverse_base(n_in: u32, base: u32) -> f32 {
    var n = n_in;
    var val: f32 = 0.0;
    var inv_base: f32 = 1.0 / f32(base);
    loop {
        if (n == 0u) {
            break;
        }
        let digit = n % base;
        val = fma(f32(digit), inv_base, val);
        inv_base = inv_base / f32(base);
        n = n / base;
    }
    return val;
}

fn cranley_patterson_rotate(x: f32, offset: f32) -> f32 {
    let sum = x + offset;
    return sum - floor(sum);
}

// ---------------------------------------------------------------------------------
// cie_1931_cmf -- color::cie1931::cie_1931_cmf, ported identically to
// shaders/environment.wgsl / shaders/furnace.wgsl (Phase 1).
// ---------------------------------------------------------------------------------

fn cmf_lobe(x: f32, mu: f32, sigma_lo: f32, sigma_hi: f32) -> f32 {
    var sigma: f32;
    if (x < mu) {
        sigma = sigma_lo;
    } else {
        sigma = sigma_hi;
    }
    let t = (x - mu) / sigma;
    return exp(-0.5 * t * t);
}

fn cie_1931_cmf(l: f32) -> vec3<f32> {
    let x_mid = fma(1.056, cmf_lobe(l, 599.8, 37.9, 31.0), 0.362 * cmf_lobe(l, 442.0, 16.0, 26.7));
    let x = fma(0.065, -cmf_lobe(l, 501.1, 20.4, 26.2), x_mid);
    let y = fma(0.286, cmf_lobe(l, 530.9, 16.3, 31.1), 0.821 * cmf_lobe(l, 568.8, 46.9, 40.5));
    let z = fma(0.681, cmf_lobe(l, 459.0, 26.0, 13.8), 1.217 * cmf_lobe(l, 437.0, 11.8, 36.0));
    return vec3<f32>(max(x, 0.0), max(y, 0.0), max(z, 0.0));
}

// ---------------------------------------------------------------------------------
// optics::renderer::env_map_spectrum::rgb_to_spectral_radiance -- used for the
// direction-independent "uniform furnace" environment (env_mode == 0u): an
// `EnvironmentMap::uniform(w, h, [l0, l0, l0])` (grey, hence direction-independent
// AND, per this reconstruction, still wavelength-DEPENDENT -- see the module doc
// comment there; this is deliberately NOT flattened to a bare `l0` return here since
// that is not what the real CPU environment this mirrors would evaluate).
// ---------------------------------------------------------------------------------

fn asymmetric_gaussian(x: f32, mu: f32, sigma_lo: f32, sigma_hi: f32) -> f32 {
    var sigma: f32;
    if (x < mu) {
        sigma = sigma_lo;
    } else {
        sigma = sigma_hi;
    }
    let t = (x - mu) / sigma;
    return exp(-0.5 * t * t);
}

fn rgb_to_spectral_radiance(r: f32, g: f32, b: f32, lambda_nm: f32) -> f32 {
    let rc = max(r, 0.0);
    let gc = max(g, 0.0);
    let bc = max(b, 0.0);
    return fma(
        rc, asymmetric_gaussian(lambda_nm, 615.0, 45.0, 65.0),
        fma(
            gc, asymmetric_gaussian(lambda_nm, 545.0, 45.0, 45.0),
            bc * asymmetric_gaussian(lambda_nm, 465.0, 40.0, 45.0),
        ),
    );
}

// ---------------------------------------------------------------------------------
// optics::raytracer::sample_studio_environment (+ optics::studio_rig::StudioRig) --
// ported identically to shaders/environment.wgsl (Phase 1).
// ---------------------------------------------------------------------------------

fn powi_u(base: f32, exp: u32) -> f32 {
    var result: f32 = 1.0;
    var b: f32 = base;
    var e: u32 = exp;
    loop {
        if (e == 0u) {
            break;
        }
        if ((e & 1u) == 1u) {
            result = result * b;
        }
        b = b * b;
        e = e >> 1u;
    }
    return result;
}

fn blackbody_spectrum(lambda_nm: f32, temp_k: f32) -> f32 {
    let t_k = max(temp_k, 1000.0);
    let h_c_k: f32 = 14388000.0;
    let exp_val = exp(min(h_c_k / (lambda_nm * t_k), 80.0));
    let exp_560 = exp(min(h_c_k / (560.0 * t_k), 80.0));
    let denom = max(exp_val - 1.0, 1e-6);
    let denom_560 = max(exp_560 - 1.0, 1e-6);
    let ratio = denom_560 / denom;
    return clamp(powi_u(560.0 / lambda_nm, 5u) * ratio, 0.01, 20.0);
}

const RING_LIGHT_COUNT: u32 = 16u;

fn studio_rig_key_dir(light_yaw: f32, light_pitch: f32) -> vec3<f32> {
    let cos_lp = cos(light_pitch);
    let sin_lp = sin(light_pitch);
    let cos_ly = cos(light_yaw);
    let sin_ly = sin(light_yaw);
    return normalize(vec3<f32>(cos_lp * sin_ly, sin_lp, cos_lp * cos_ly));
}

fn studio_rig_fill_dir(light_yaw: f32, light_pitch: f32) -> vec3<f32> {
    let fill_yaw = fma(PI, 0.78, light_yaw);
    let fill_pitch = clamp(light_pitch * 0.65, 0.15, 1.2);
    return normalize(vec3<f32>(cos(fill_pitch) * sin(fill_yaw), sin(fill_pitch), cos(fill_pitch) * cos(fill_yaw)));
}

fn studio_rig_ring_dir(i: u32, light_yaw: f32, sin_lp: f32) -> vec3<f32> {
    let angle = fma(f32(i), PI * 2.0 / f32(RING_LIGHT_COUNT), light_yaw);
    return normalize(vec3<f32>(cos(angle) * 0.75, sin_lp * 0.8, sin(angle) * 0.75));
}

// Fix 4: `key_dir`/`fill_dir`/`sin_lp` (the `StudioRig`-equivalent quantities, ~40
// sin/cos plus normalizes worth of work) are constant across an entire ray -- the CPU
// side's `optics::raytracer::accumulate_miss_radiance` builds them once per ray rather
// than once per spectral channel, and this mirrors that: the caller (`transport_main`'s
// miss branch) now computes them ONCE before its `NUM_CHANNELS` loop and passes them in,
// instead of this function recomputing them on every one of the (up to 8) per-channel
// calls it used to make internally.
fn sample_studio_environment_with_rig(
    dir_in: vec3<f32>,
    lambda_nm: f32,
    key_dir: vec3<f32>,
    fill_dir: vec3<f32>,
    sin_lp: f32,
) -> f32 {
    let d = normalize(dir_in);
    let spec_power = blackbody_spectrum(lambda_nm, params.studio_temp_k);

    let bg_val = max(fma(0.012, fma(d.y, 0.5, 0.5), 0.015), 0.005) * params.studio_exposure;
    var radiance = bg_val * spec_power;

    let key_dot = max(dot(d, key_dir), 0.0);
    if (key_dot > 0.0) {
        let softbox = powi_u(key_dot, 28u) * 12.0 * params.studio_spot_mult * params.studio_exposure;
        radiance = fma(softbox, spec_power, radiance);
    }

    let fill_dot = max(dot(d, fill_dir), 0.0);
    if (fill_dot > 0.0) {
        let fill = powi_u(fill_dot, 18u) * 4.5 * params.studio_exposure;
        radiance = fma(fill, spec_power, radiance);
    }

    for (var i: u32 = 0u; i < RING_LIGHT_COUNT; i = i + 1u) {
        let ring_dir = studio_rig_ring_dir(i, params.studio_light_yaw, sin_lp);
        let ring_dot = max(dot(d, ring_dir), 0.0);
        if (ring_dot > 0.96) {
            let spark = (ring_dot - 0.96) / 0.04;
            let intensity = powi_u(spark, 6u) * 22.0 * params.studio_spot_mult * params.studio_exposure;
            radiance = fma(intensity, spec_power, radiance);
        }
    }

    return radiance;
}

fn sample_environment_with_rig(
    dir: vec3<f32>,
    lambda_nm: f32,
    key_dir: vec3<f32>,
    fill_dir: vec3<f32>,
    sin_lp: f32,
) -> f32 {
    if (params.env_mode == 0u) {
        return rgb_to_spectral_radiance(params.l0, params.l0, params.l0, lambda_nm);
    }
    return sample_studio_environment_with_rig(dir, lambda_nm, key_dir, fill_dir, sin_lp);
}

// ---------------------------------------------------------------------------------
// `dispersion_evaluate`, `spectral_absorption`, the four `mueller_*` Mueller-matrix
// constructors, `tir_phase_delta`, `normalize_or_zero`, `signed_frame_rotation_psi`,
// `degree_of_polarization`, `polarization_azimuth`, `arbitrary_perpendicular`,
// `electric_field_direction`, `stable_orthonormal_basis_t`,
// `ordinary_eigen_polarization`, `extraordinary_eigen_polarization`,
// `quadratic_form`, and `pleochroic_channel_alpha` all now live in
// `shaders/transport_physics.wgsl`, the single shared source `build.rs` concatenates
// ahead of this file (and of `transport_functions.wgsl`) into the generated shader --
// see that file's header comment. Look there, not here.
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// optics::raytracer::intersect_polyhedron
// ---------------------------------------------------------------------------------

struct HitInfo {
    hit: bool,
    t: f32,
    normal: vec3<f32>,
    // Task 2 GPU port: optics::raytracer::HitRecord::facet_idx, ported here (this
    // kernel's own `intersect_ray` never tracked it before -- unlike the standalone
    // Phase 1 `intersect_polyhedron.wgsl` kernel/`GpuHitRecord`, which already did) so
    // the frosted-finish lookup below has something to index `facet_finishes` with.
    // Mirrors `near_facet`/`far_facet`'s own `.unwrap_or(0)` fallback: `0u` whenever
    // `t_near`/`t_far` never left their sentinel values, i.e. exactly when a hit is
    // never reported either.
    facet_idx: u32,
}

fn intersect_ray(origin: vec3<f32>, dir: vec3<f32>) -> HitInfo {
    var t_near: f32 = -1e30;
    var t_far: f32 = 1e30;
    var near_normal = vec3<f32>(0.0, 0.0, 0.0);
    var far_normal = vec3<f32>(0.0, 0.0, 0.0);
    var near_idx: u32 = 0u;
    var far_idx: u32 = 0u;
    var result: HitInfo;
    let num_planes = arrayLength(&planes);
    for (var i: u32 = 0u; i < num_planes; i = i + 1u) {
        let p = planes[i];
        let n = p.normal;
        let denom = dot(n, dir);
        let side = p.d + dot(n, origin);
        let numer = -side;
        if (abs(denom) > 1e-7) {
            let t = numer / denom;
            if (denom < 0.0) {
                if (t > t_near) {
                    t_near = t;
                    near_normal = n;
                    near_idx = i;
                }
            } else if (t < t_far) {
                t_far = t;
                far_normal = n;
                far_idx = i;
            }
        } else if (side > 0.0) {
            // Fix 3: ray (near-)parallel to this plane, origin already outside its
            // half-space -- the polyhedron intersection is empty for this ray. See
            // optics::raytracer::intersect_polyhedron's matching comment.
            result.hit = false;
            result.t = 0.0;
            result.normal = vec3<f32>(0.0, 0.0, 0.0);
            result.facet_idx = 0u;
            return result;
        }
    }
    if (t_near > t_far) {
        result.hit = false;
        result.t = 0.0;
        result.normal = vec3<f32>(0.0, 0.0, 0.0);
        result.facet_idx = 0u;
    } else if (t_near > 1e-4) {
        result.hit = true;
        result.t = t_near;
        result.normal = near_normal;
        result.facet_idx = near_idx;
    } else if (t_far > 1e-4) {
        result.hit = true;
        result.t = t_far;
        result.normal = far_normal;
        result.facet_idx = far_idx;
    } else {
        result.hit = false;
        result.t = 0.0;
        result.normal = vec3<f32>(0.0, 0.0, 0.0);
        result.facet_idx = 0u;
    }
    return result;
}

// ---------------------------------------------------------------------------------
// Physics review, Task 2 GPU port: optics::raytracer::shading_normal_near_edge -- reads
// the `planes` storage binding directly (like `intersect_ray` above), so this stays a
// megakernel-local function rather than living in the shared `transport_physics.wgsl`
// prelude -- see that file's own doc comment for why a binding-touching function is
// necessarily per-file. `shaders/shading_normal.wgsl`'s own Tier 2 self-test
// (`renderer::gpu::transport_check::run_shading_normal_near_edge`) is a SEPARATE,
// standalone file with its OWN `planes` binding (bound to the SAME real round-brilliant
// case bank `polyhedron_check` already uses) and its own copy of this identical
// function body, compared against the REAL CPU `optics::raytracer::shading_normal_near_edge`
// -- both WGSL copies are direct, unmodified line-for-line translations of that one CPU
// function, so they cannot drift from each other in substance, only in which binding
// group supplies `planes`.
// ---------------------------------------------------------------------------------

fn shading_normal_near_edge(hit_point: vec3<f32>, hit_facet_idx: u32, hit_normal: vec3<f32>, rounding_radius: f32) -> vec3<f32> {
    if (rounding_radius <= 0.0) {
        return hit_normal;
    }
    var nearest_dist: f32 = 1e30;
    var nearest_normal = hit_normal;
    let num_planes = arrayLength(&planes);
    for (var i: u32 = 0u; i < num_planes; i = i + 1u) {
        if (i == hit_facet_idx) {
            continue;
        }
        let p = planes[i];
        let dist = -(p.d + dot(p.normal, hit_point));
        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest_normal = p.normal;
        }
    }
    if (nearest_dist >= rounding_radius) {
        return hit_normal;
    }
    let t = clamp(1.0 - nearest_dist / rounding_radius, 0.0, 1.0);
    let smooth_t = t * t * fma(-2.0, t, 3.0);
    let bisector = normalize_or_zero(hit_normal + nearest_normal);
    return normalize_or_zero(hit_normal * (1.0 - smooth_t) + bisector * smooth_t);
}

// ---------------------------------------------------------------------------------
// optics::raytracer::Camera::generate_ray -- bit-exact per Phase 1's camera_check.
// ---------------------------------------------------------------------------------

struct RayGen {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

fn generate_camera_ray(pixel: u32, jx: f32, jy: f32) -> RayGen {
    let x = f32(pixel % u32(camera.width));
    let y = f32(pixel / u32(camera.width));
    let aspect = camera.width / camera.height;
    let u = ((x + jx) / camera.width - 0.5) * 2.0 * aspect * camera.fov_tan;
    let v = (0.5 - (y + jy) / camera.height) * 2.0 * camera.fov_tan;
    let dir = normalize(camera.forward + camera.right * u + camera.up * v);
    var result: RayGen;
    result.origin = camera.origin;
    result.dir = dir;
    return result;
}

// ---------------------------------------------------------------------------------
// Task 1: optics::raytracer::apply_internal_mode_coupling -- stochastic o<->e
// re-coupling at an INTERNAL reflection inside an anisotropic crystal. See that Rust
// function's doc comment for the full physics/unbiasedness rationale, in particular
// "This is a RELABELING, not a SPLIT -- and neither draw needs a `1/p` division":
// unlike the entry split (BIREFRINGENT_SPLIT_STREAM, where one incident ray genuinely
// becomes two physically distinct rays, each carrying only its own ~0.5 energy share
// -- a share already matched by its own 0.5 selection probability, so estimating
// their sum from one traced sample needs no scaling at all, not a `1/0.5`), this draw
// only re-rolls which eigenmode LABEL governs the index used for the path's NEXT
// bounce -- no new ray, no energy split, nothing for `stokes`/`path_pdf` to do.
// Split into the pure coin-flip draw here (mirroring the CPU's own draw exactly, and
// matching this file's existing style of every other per-bounce helper returning a
// value the caller assigns) with NO stokes/path_pdf scaling at any call site below.
// ---------------------------------------------------------------------------------

fn internal_mode_coupling_draw(seed0: u32, bounce: u32) -> bool {
    let split_rand = f32(hash_u32(seed0 ^ hash_u32(bounce ^ MODE_COUPLING_STREAM))) / 4294967295.0;
    return split_rand < 0.5;
}

// ---------------------------------------------------------------------------------
// optics::raytracer::trace_spectral_ray -- the isotropic-only estimator.
// ---------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn transport_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.num_pixels * camera.num_samples;
    if (idx >= total) {
        return;
    }

    // GLOBAL pixel index: `params.pixel_offset` is 0 for a dispatch covering the whole
    // frame (every self-test in `renderer::gpu`, so their outputs are bit-unchanged by
    // this), and non-zero only for `renderer::gpu::frame`'s chunked dispatches. It is
    // added HERE, not to `idx`, precisely so it reaches camera-ray generation and the
    // per-pixel Cranley-Patterson rotations below -- which must be a function of the
    // pixel's place in the frame -- while the output writes at the end of this function
    // keep indexing by the dispatch-local `idx`.
    let pixel = idx / camera.num_samples + params.pixel_offset;
    let local_sample = idx % camera.num_samples;
    let sample_num = local_sample + params.sample_offset;

    let seed0 = hash_u32((pixel * 0x9e3779b9u) ^ (sample_num * 0x85ebca6bu));

    // Fix 4 (optics::raytracer::{low_discrepancy_base2, cranley_patterson_rotate}):
    // stratified pixel jitter and hero wavelength, not an unstratified hash-uniform.
    // `seed0` above still seeds every per-bounce draw below (Fresnel branch, Russian
    // roulette, birefringent split) exactly as before -- only jx/jy/hero_rand come
    // from this construction now.
    let rot_jx = low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_X_ROTATION_STREAM));
    let rot_jy = low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_Y_ROTATION_STREAM));
    let rot_hero = low_discrepancy_base2(hash_u32(pixel ^ HERO_WAVELENGTH_ROTATION_STREAM));
    let jx = cranley_patterson_rotate(low_discrepancy_base2(sample_num), rot_jx) - 0.5;
    let jy = cranley_patterson_rotate(radical_inverse_base(sample_num, 3u), rot_jy) - 0.5;
    let hero_rand = cranley_patterson_rotate(radical_inverse_base(sample_num, 5u), rot_hero);
    let raygen = generate_camera_ray(pixel, jx, jy);

    let channel_width = SPECTRUM_SPAN / f32(NUM_CHANNELS);
    let lambda_hero = fma(hero_rand, SPECTRUM_SPAN, SPECTRUM_MIN);
    var lambdas: array<f32, 8>;
    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
        let offset = fma(f32(k), channel_width, lambda_hero - SPECTRUM_MIN);
        lambdas[k] = SPECTRUM_MIN + (offset % SPECTRUM_SPAN);
    }

    var stokes: array<vec4<f32>, 8>;
    var radiance: array<f32, 8>;
    var path_pdf: array<f32, 8>;
    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
        stokes[k] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
        radiance[k] = 0.0;
        path_pdf[k] = 1.0;
    }

    let c_axis = material.dispersion.c_axis_and_birefringence.xyz;
    let birefringence_delta = material.dispersion.c_axis_and_birefringence.w;
    // Kernel specialisation: ANDed with the pipeline-overridable MATERIAL_CLASS (see
    // this file's header comment at that override's declaration) -- a no-op for
    // MATERIAL_CLASS_GENERIC (every self-test), forced `false` for
    // MATERIAL_CLASS_ISOTROPIC regardless of the material buffer's own flag.
    let is_anisotropic = (material.dispersion.is_anisotropic != 0u)
        && (MATERIAL_CLASS != MATERIAL_CLASS_ISOTROPIC);
    // Phase 4: whether this material's anisotropy is genuinely biaxial (three distinct
    // principal indices) rather than the uniaxial ordinary/extraordinary approximation
    // -- optics::raytracer::compute_bounce_refraction_geometry's `is_biaxial` local,
    // hoisted per-ray (constant across every bounce, exactly like `c_axis` above).
    // Kernel specialisation: ANDed with MATERIAL_CLASS the same way `is_anisotropic` is
    // above -- a no-op for MATERIAL_CLASS_GENERIC and MATERIAL_CLASS_BIAXIAL, forced
    // `false` for MATERIAL_CLASS_ISOTROPIC and MATERIAL_CLASS_UNIAXIAL regardless of the
    // material buffer's own flag (never true for either class's routed materials
    // anyway -- see `renderer::gpu::frame::classify_material` -- so this only removes
    // dead-by-construction reachability, it never changes what a correctly-routed
    // dispatch computes).
    let is_biaxial = (material.dispersion.has_biaxial_delta != 0u)
        && (MATERIAL_CLASS == MATERIAL_CLASS_GENERIC || MATERIAL_CLASS == MATERIAL_CLASS_BIAXIAL);
    // The biaxial principal-axis frame (alpha, beta, gamma world directions) depends
    // only on `c_axis` (see `biaxial_axes_from_gamma`'s own doc comment), so -- exactly
    // like `c_axis` itself -- it is computed once per ray, not rebuilt every bounce or
    // every channel; bit-identical either way since it is a pure function of `c_axis`.
    let biax_axes = biaxial_axes_from_gamma(c_axis);
    let biax_ax0 = biax_axes.ax0;
    let biax_ax1 = biax_axes.ax1;
    let biax_ax2 = biax_axes.ax2;
    // Phase 4: the hero channel's base dispersion value and, from it, the hero
    // indicatrix's three principal indices (optics::materials::GemMaterial::
    // biaxial_indicatrix's `n_beta := dispersion.evaluate(lambda)`, `n_alpha := n_beta -
    // biaxial_delta_beta_alpha`, `n_gamma := n_alpha + birefringence_delta` convention).
    // Hoisted per-ray (bit-identical to recomputing every bounce, since `lambdas` is
    // itself fixed for the whole ray -- see optics::raytracer::compute_bounce_refraction_geometry's
    // own per-bounce `n_o_hero_seed`, a pure function of unchanging inputs computed
    // fresh each bounce on the CPU side too). Computed unconditionally (cheap, and
    // provably harmless when `!is_biaxial` -- `biaxial_delta_beta_alpha` is `0.0` for
    // every non-biaxial material's encoding, so `n_alpha_hero == n_beta_hero ==
    // n_o_hero_seed` in that case, and no `biaxial_*` function is ever CALLED with
    // these values unless `is_biaxial` guards it).
    let n_o_hero_seed = dispersion_evaluate(material.dispersion.model_type, material.dispersion.param_a, material.dispersion.param_b, lambdas[0]);
    let n_beta_hero = n_o_hero_seed;
    let n_alpha_hero = n_beta_hero - material.dispersion.biaxial_delta_beta_alpha;
    let n_gamma_hero = n_alpha_hero + birefringence_delta;

    // R3-GPU: `n_o_ch[k]` (per_channel_uniaxial_index's `dispersion_evaluate` result)
    // and the per-channel pleochroic absorption coefficients (`spectral_absorption` on
    // the o/e/beta band sets) depend ONLY on `lambdas[k]` (fixed for this whole ray, see
    // above) and the material's own dispersion/band data -- never on anything that
    // varies per bounce (`theta_c`, `current_dir`, current Stokes state, ...), exactly
    // like `n_o_hero_seed` above. The pre-hoist code recomputed all of these on EVERY
    // bounce (inside the scattering block, the absorption block, and the
    // per_channel_uniaxial_index loop below); hoisting them here to per-ray arrays,
    // computed ONCE, means every per-bounce use below reads an already-computed value
    // instead of re-deriving it -- the exact same `dispersion_evaluate`/
    // `spectral_absorption` calls on the exact same inputs, so every value is
    // bit-identical to what the per-bounce recomputation produced. `alpha_beta_hoisted`
    // is left zero (matching `spectral_absorption`'s own zero-band-count result) unless
    // `is_biaxial`, mirroring the original code's "only ever computed when needed" guard
    // rather than doing pointless band-summation work for the common non-biaxial case.
    //
    // KEEP/REVERT: measured (best-of-3, 800x600 Zircon and Tourmaline, 24 bounces)
    // consistently and substantially faster than recomputing per-bounce -- see this
    // task's report for the numbers and the noisy-machine caveat on their exact
    // magnitude; the qualitative direction (large win, every run) is robust to that
    // noise. Kept.
    var n_o_hoisted: array<f32, 8>;
    var alpha_o_hoisted: array<f32, 8>;
    var alpha_e_hoisted: array<f32, 8>;
    var alpha_beta_hoisted: array<f32, 8>;
    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
        n_o_hoisted[k] = dispersion_evaluate(material.dispersion.model_type, material.dispersion.param_a, material.dispersion.param_b, lambdas[k]);
        alpha_o_hoisted[k] = spectral_absorption(material.o_ray_bands, material.o_ray_band_count, lambdas[k]);
        alpha_e_hoisted[k] = spectral_absorption(material.e_ray_bands, material.e_ray_band_count, lambdas[k]);
        alpha_beta_hoisted[k] = 0.0;
    }
    if (is_biaxial) {
        for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
            alpha_beta_hoisted[k] = spectral_absorption(material.beta_ray_bands, material.beta_ray_band_count, lambdas[k]);
        }
    }

    var current_origin = raygen.origin;
    var current_dir = raygen.dir;
    // P2: the wave normal `k`, tracked alongside `current_dir` (the Poynting/energy
    // direction `S`) -- mirrors optics::raytracer::transport::trace_spectral_ray_inner's
    // `current_k` exactly. Starts equal to the ray's direction: outside the gem (air)
    // `k == S` always (isotropic medium, no walk-off).
    var current_k = raygen.dir;
    var inside_gem = false;
    // Phase 3: which eigenmode the ray currently inside the crystal was stochastically
    // assigned to at its most recent air->crystal entry -- optics::raytracer::
    // trace_spectral_ray's `is_extraordinary` local. Meaningless while `!inside_gem`;
    // carried across internal bounces exactly as the CPU carries it.
    var is_extraordinary = false;
    var prev_plane_normal = vec3<f32>(0.0, 0.0, 0.0);
    var have_prev_plane_normal = false;

    for (var bounce: u32 = 0u; bounce < params.max_bounces; bounce = bounce + 1u) {
        let hit = intersect_ray(current_origin, current_dir);
        if (!hit.hit) {
            // Fix 4: build the studio rig's key/fill/ring directions ONCE per ray,
            // outside the per-channel loop, rather than once per channel -- see
            // `sample_studio_environment_with_rig`'s doc comment. Harmless (and unused
            // by `sample_environment_with_rig`) when `params.env_mode == 0u` (HDR/flat
            // environment).
            let studio_key_dir = studio_rig_key_dir(params.studio_light_yaw, params.studio_light_pitch);
            let studio_fill_dir = studio_rig_fill_dir(params.studio_light_yaw, params.studio_light_pitch);
            let studio_sin_lp = sin(params.studio_light_pitch);
            for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                let env = sample_environment_with_rig(current_dir, lambdas[k], studio_key_dir, studio_fill_dir, studio_sin_lp);
                radiance[k] = fma(max(stokes[k].x, 0.0), env, radiance[k]);
            }
            break;
        }

        // Physics review, Task 1 (inclusion/subsurface scattering): mirrors
        // optics::raytracer::trace_spectral_ray_inner's restructured bounce loop --
        // attempt a Henyey-Greenstein scattering event somewhere along this segment
        // BEFORE the plane-of-incidence rotation / facet processing below, exactly
        // like the CPU reference (see that function's own doc comment for why the
        // scatter check runs first). Gated on `material.scattering_sigma_s > 0.0` --
        // every existing scene (`<= 0.0`) skips this block entirely and reaches the
        // exact pre-Task-1 code below unconditionally, matching the CPU's
        // default-off bit-identity guarantee (the new branch never runs, not "reduces
        // to the same value").
        if (inside_gem && material.scattering_sigma_s > 0.0) {
            var s_axis = vec3<f32>(0.0, 0.0, 0.0);
            if (have_prev_plane_normal) {
                s_axis = prev_plane_normal;
            }
            // L4: mirror the absorption block's `is_biaxial` branching below (~line
            // 837) -- optics::raytracer::scattering::try_scatter_step feeds its
            // maybe_scatter_or_extinguish call with the SAME channel_absorption_alphas
            // the absorption block uses, so a biaxial material's scattering alphas
            // must come from the biaxial indicatrix's own eigen_polarizations
            // (evaluated at the hero wavelength, exactly like the absorption block),
            // not the uniaxial ordinary/extraordinary approximation. Guarded on
            // `is_biaxial` (false for every existing non-biaxial scene), so this is
            // bit-identical to the previous code whenever it isn't true.
            // P2: eigen-polarizations and the propagation direction fed to
            // pleochroic_channel_alpha(_biaxial) are properties of the WAVE NORMAL `k`,
            // not the Poynting direction `S` -- current_k, not current_dir. See
            // optics::raytracer::scattering::try_scatter_step's own doc comment, rule 6.
            var eigen_a: vec3<f32>;
            var eigen_b: vec3<f32>;
            if (is_biaxial) {
                let eig = biaxial_eigen_polarizations(n_alpha_hero, n_beta_hero, n_gamma_hero, biax_ax0, biax_ax1, biax_ax2, current_k);
                eigen_a = eig.d_slow;
                eigen_b = eig.d_fast;
            } else {
                eigen_a = ordinary_eigen_polarization(current_k, c_axis);
                eigen_b = extraordinary_eigen_polarization(current_k, c_axis);
            }
            var alphas: array<f32, 8>;
            for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                // R3-GPU: alpha_o/alpha_e/alpha_beta hoisted above (per-ray, not
                // per-bounce) -- see that hoist's own doc comment.
                if (is_biaxial && material.has_beta_ray != 0u) {
                    alphas[k] = pleochroic_channel_alpha_biaxial(
                        alpha_o_hoisted[k], alpha_beta_hoisted[k], alpha_e_hoisted[k], c_axis, s_axis, current_k, eigen_a, eigen_b, stokes[k],
                    );
                } else {
                    alphas[k] = pleochroic_channel_alpha(
                        alpha_o_hoisted[k], alpha_e_hoisted[k], c_axis, s_axis, current_k, eigen_a, eigen_b, stokes[k],
                    );
                }
            }
            let sc = maybe_scatter_or_extinguish(
                alphas, material.scattering_sigma_s, material.scattering_g, current_dir, hit.t,
                material.absorption_path_scale, seed0, bounce, &stokes, &path_pdf,
            );
            if (sc.scattered != 0u) {
                current_origin = current_origin + sc.t_free * current_dir;
                current_dir = sc.new_dir;
                // P2: a scattering event depolarizes (see the comment below), so `k`
                // collapses to `S` going forward -- mirrors
                // optics::raytracer::scattering::try_scatter_step's identical
                // `*current_k = new_dir` treatment.
                current_k = sc.new_dir;
                // Scattered Stokes vectors are already depolarized (see
                // maybe_scatter_or_extinguish), so the previous plane of incidence is
                // no longer physically meaningful -- reset it exactly like the
                // pre-first-bounce state, mirroring the CPU's `prev_plane_normal =
                // None`.
                have_prev_plane_normal = false;

                if (bounce > 4u) {
                    var max_intensity: f32 = 0.0;
                    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                        max_intensity = max(max_intensity, max(stokes[k].x, 0.0));
                    }
                    let q = clamp(max_intensity, RR_FLOOR, 1.0);
                    let rr_rand = f32(hash_u32(seed0 ^ hash_u32(bounce ^ RUSSIAN_ROULETTE_STREAM))) / 4294967295.0;
                    if (rr_rand > q) {
                        break;
                    }
                    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                        stokes[k] = stokes[k] * (1.0 / q);
                    }
                }
                continue;
            }
            // No scatter event fired: the path survived to the facet boundary, and
            // `maybe_scatter_or_extinguish` already applied this segment's full
            // extinction weight (absorption AND the scattering removal probability)
            // to `stokes`/`path_pdf`. Fall through to the facet-dispatch code below,
            // but see the (skipped) absorption block further down for why it is not
            // applied a second time.
        }

        let hit_point = current_origin + hit.t * current_dir;
        // Task 2 (facet edge rounding): see shading_normal_near_edge's own doc comment.
        var normal = shading_normal_near_edge(hit_point, hit.facet_idx, hit.normal, material.edge_rounding_radius);
        // P2: `wave_dir_at_bounce` is the wave normal `k` (== current_k), used for
        // EVERY index lookup, cos_i/sin_i, Snell/Fresnel evaluation, TIR decision, and
        // the Stokes plane-of-incidence frame below -- see
        // optics::raytracer::refraction's "wave normal vs Poynting direction" design
        // note. The Poynting/energy direction `S` (== current_dir) is used directly
        // (geometric origin-advance/intersection only, unaffected by P2) where still
        // needed. `wave_dir_at_bounce == current_dir` trivially outside the crystal and
        // for the uniaxial ordinary eigenmode, so every such case is bit-identical to
        // the pre-P2 all-`current_dir` code.
        let wave_dir_at_bounce = current_k;

        // Plane-of-incidence frame rotation (Fix 1: signed psi via atan2).
        let cpn_raw = cross(wave_dir_at_bounce, normal);
        let cpn_len2 = dot(cpn_raw, cpn_raw);
        var current_plane_normal: vec3<f32>;
        if (cpn_len2 > 0.0) {
            current_plane_normal = cpn_raw / sqrt(cpn_len2);
        } else {
            current_plane_normal = vec3<f32>(0.0, 0.0, 0.0);
        }
        if (have_prev_plane_normal && cpn_len2 > 1e-6 && dot(prev_plane_normal, prev_plane_normal) > 1e-6) {
            let psi = signed_frame_rotation_psi(prev_plane_normal, current_plane_normal, wave_dir_at_bounce);
            let rot = mueller_frame_rotation(psi);
            for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                stokes[k] = rot * stokes[k];
            }
        }
        prev_plane_normal = current_plane_normal;
        have_prev_plane_normal = true;

        if (inside_gem) {
            normal = -normal;
            // Task 1: a scattering-active material's extinction for this segment was
            // already applied above (in the `maybe_scatter_or_extinguish` no-scatter
            // branch) -- applying the plain absorption loop here too would charge
            // this segment's absorption TWICE. `material.scattering_sigma_s <= 0.0`
            // is exactly the pre-Task-1 case (including every existing scene), where
            // the block above never ran and this is the ONLY absorption
            // application, matching the CPU's identical guard.
            if (material.scattering_sigma_s <= 0.0) {
                // Phase 4: optics::raytracer::channel_absorption_alphas -- for a
                // biaxial material the two eigenmode directions come from the
                // material's OWN indicatrix (`biaxial_eigen_polarizations`, evaluated
                // at the hero wavelength -- the direction-only quantities this feeds
                // don't vary per channel, exactly like the uniaxial ordinary/
                // extraordinary pair they generalize), not the uniaxial
                // ordinary/extraordinary approximation.
                var eigen_a: vec3<f32>;
                var eigen_b: vec3<f32>;
                if (is_biaxial) {
                    let eig = biaxial_eigen_polarizations(n_alpha_hero, n_beta_hero, n_gamma_hero, biax_ax0, biax_ax1, biax_ax2, wave_dir_at_bounce);
                    eigen_a = eig.d_slow;
                    eigen_b = eig.d_fast;
                } else {
                    eigen_a = ordinary_eigen_polarization(wave_dir_at_bounce, c_axis);
                    eigen_b = extraordinary_eigen_polarization(wave_dir_at_bounce, c_axis);
                }
                for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                    // R3-GPU: alpha_o/alpha_e/alpha_beta hoisted above (per-ray, not
                    // per-bounce) -- see that hoist's own doc comment.
                    var alpha_eff: f32;
                    if (is_biaxial && material.has_beta_ray != 0u) {
                        alpha_eff = pleochroic_channel_alpha_biaxial(
                            alpha_o_hoisted[k], alpha_beta_hoisted[k], alpha_e_hoisted[k], c_axis, current_plane_normal, wave_dir_at_bounce, eigen_a, eigen_b, stokes[k],
                        );
                    } else {
                        alpha_eff = pleochroic_channel_alpha(
                            alpha_o_hoisted[k], alpha_e_hoisted[k], c_axis, current_plane_normal, wave_dir_at_bounce, eigen_a, eigen_b, stokes[k],
                        );
                    }
                    // P1 (absorption path scale): mirrors
                    // optics::raytracer::absorption::apply_absorption's own
                    // `path_len * ctx.material.absorption_path_scale` multiply exactly.
                    let scaled_hit_t = hit.t * material.absorption_path_scale;
                    let trans_factor = exp(-alpha_eff * scaled_hit_t);
                    stokes[k] = stokes[k] * trans_factor;
                }
            }
        }

        // P2: angle of incidence measured against the WAVE NORMAL `k`
        // (`wave_dir_at_bounce`), not the Poynting/energy direction `S` -- see this
        // block's own design note above.
        let cos_i = clamp(dot(-wave_dir_at_bounce, normal), 0.0, 1.0);
        let sin_i = sqrt(max(fma(-cos_i, cos_i, 1.0), 0.0));

        // Phase 3: theta_c fixed-point iteration (optics::raytracer::theta_c_for_bounce)
        // plus the per-channel ordinary/effective-extraordinary index pair
        // (optics::raytracer::per_channel_uniaxial_indices, called once per channel via
        // per_channel_uniaxial_index -- see transport_physics.wgsl). For a cubic
        // material (is_anisotropic == false) this reduces to n_eff_ch[k] == n_o_ch[k]
        // for every k, exactly Phase 2's isotropic-only computation. For a biaxial
        // material this is a DEAD (unused) computation -- see transport_physics.wgsl's
        // `theta_c_for_bounce` doc comment. P2: fed `wave_dir_at_bounce` (`k`), not
        // `current_dir` (`S`).
        let theta_c = theta_c_for_bounce(normal, wave_dir_at_bounce, cos_i, inside_gem, is_anisotropic, c_axis, n_o_hero_seed, birefringence_delta);

        // R3-GPU: `n_o_ch[k]` is exactly `n_o_hoisted[k]` (both `dispersion_evaluate` on
        // the same `lambdas[k]`, hoisted above) -- only the theta_c-dependent
        // `n_eff_k` half of `per_channel_uniaxial_index` still needs to run every
        // bounce, so that's all this loop does now, reproducing that function's own
        // `n_e_k`/`effective_extraordinary_index` body (see transport_physics.wgsl) on
        // the hoisted `n_o_hoisted[k]` rather than calling the full function (which
        // would redundantly re-evaluate dispersion) -- bit-identical either way.
        var n_eff_ch: array<f32, 8>;
        for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
            let n_e_k = n_o_hoisted[k] + birefringence_delta;
            var n_eff_k = n_o_hoisted[k];
            if (is_anisotropic) {
                n_eff_k = effective_extraordinary_index(n_o_hoisted[k], n_e_k, theta_c);
            }
            n_eff_ch[k] = n_eff_k;
        }
        let n_o_hero = n_o_hoisted[0];
        let n_e_hero = n_o_hero + birefringence_delta;

        // Phase 4: optics::raytracer::{hero_biaxial_wave_dirs, per_channel_biaxial_indices}.
        // "mode A" is the faster (lower-index) root, "mode B" the slower -- see this
        // file's header comment. Zero-initialized and never consulted downstream unless
        // `is_biaxial` guards the read, exactly mirroring the CPU arrays' own
        // "computed unconditionally, only ever POPULATED when is_biaxial" contract.
        var wave_dir_a_hero = vec3<f32>(0.0, 0.0, 0.0);
        var wave_dir_b_hero = vec3<f32>(0.0, 0.0, 0.0);
        var n_biax_a_ch: array<f32, 8>;
        var n_biax_b_ch: array<f32, 8>;
        var n_alpha_ch: array<f32, 8>;
        var n_gamma_ch: array<f32, 8>;
        for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
            n_biax_a_ch[k] = 0.0;
            n_biax_b_ch[k] = 0.0;
            n_alpha_ch[k] = 0.0;
            n_gamma_ch[k] = 0.0;
        }
        if (is_biaxial) {
            if (inside_gem) {
                // P2: both biaxial modes walk off (see optics::raytracer::refraction's
                // hero_biaxial_wave_dirs doc comment) -- k, not S.
                wave_dir_a_hero = wave_dir_at_bounce;
                wave_dir_b_hero = wave_dir_at_bounce;
            } else {
                let res_a = biaxial_resolve_entry_mode(n_alpha_hero, n_beta_hero, n_gamma_hero, biax_ax0, biax_ax1, biax_ax2, wave_dir_at_bounce, normal, cos_i, n_o_hero_seed, false);
                let res_b = biaxial_resolve_entry_mode(n_alpha_hero, n_beta_hero, n_gamma_hero, biax_ax0, biax_ax1, biax_ax2, wave_dir_at_bounce, normal, cos_i, n_o_hero_seed, true);
                wave_dir_a_hero = res_a.wave_dir;
                wave_dir_b_hero = res_b.wave_dir;
            }
            for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                // optics::materials::GemMaterial::biaxial_indicatrix's per-channel
                // n_beta/n_alpha/n_gamma -- n_beta_k is bit-identical to n_o_hoisted[k]
                // (both `dispersion.evaluate(lambdas[k])`, same inputs), so reused
                // rather than recomputed.
                let n_beta_k = n_o_hoisted[k];
                let n_alpha_k = n_beta_k - material.dispersion.biaxial_delta_beta_alpha;
                let n_gamma_k = n_alpha_k + birefringence_delta;
                n_alpha_ch[k] = n_alpha_k;
                n_gamma_ch[k] = n_gamma_k;
                let ni_a = biaxial_wave_indices(n_alpha_k, n_beta_k, n_gamma_k, biax_ax0, biax_ax1, biax_ax2, wave_dir_a_hero);
                let ni_b = biaxial_wave_indices(n_alpha_k, n_beta_k, n_gamma_k, biax_ax0, biax_ax1, biax_ax2, wave_dir_b_hero);
                n_biax_a_ch[k] = ni_a.y;
                n_biax_b_ch[k] = ni_b.x;
            }
        }
        let n_biax_a_hero = n_biax_a_ch[0];
        let n_biax_b_hero = n_biax_b_ch[0];

        // Fix 3c (CPU side) / Phase 3/4: while inside an anisotropic crystal, the medium
        // index this ray is currently in is mode A or mode B depending on which
        // eigenmode `is_extraordinary` selected at the most recent entry; outside the
        // crystal (or for an isotropic material) it is always the mode-B array, which
        // for a cubic material equals n_o_hoisted exactly (see per_channel_uniaxial_index).
        // `is_biaxial` selects which pair of arrays ("mode A"/"mode B") is consulted --
        // the biaxial ones computed just above, or the uniaxial n_o_hoisted/n_eff_ch pair.
        let use_mode_a_medium = is_anisotropic && inside_gem && !is_extraordinary;
        var n_medium_ch: array<f32, 8>;
        for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
            var mode_a_k: f32;
            var mode_b_k: f32;
            if (is_biaxial) {
                mode_a_k = n_biax_a_ch[k];
                mode_b_k = n_biax_b_ch[k];
            } else {
                mode_a_k = n_o_hoisted[k];
                mode_b_k = n_eff_ch[k];
            }
            n_medium_ch[k] = select(mode_b_k, mode_a_k, use_mode_a_medium);
        }
        let n_medium_hero = n_medium_ch[0];
        let n1 = select(1.0, n_medium_hero, inside_gem);
        let n2 = select(n_medium_hero, 1.0, inside_gem);
        let eta = n1 / n2;
        let sin2_t = eta * eta * fma(-cos_i, cos_i, 1.0);

        // Task 2 GPU port (frosted girdle finish): which specular/diffuse treatment
        // this facet gets. Bounds-checked (mirrors optics::raytracer::trace_spectral_ray_inner's
        // `facet_finishes.get(hit_rec.facet_idx).copied().unwrap_or_default()`): an
        // index past the end of a shorter-than-`planes` buffer defaults to
        // `facet_finish::POLISHED`, exactly as the CPU's `Default` does.
        // `select`'s two value arguments are BOTH evaluated in WGSL (no short-circuit),
        // so an `if` guard is used instead of `select` here -- indexing
        // `facet_finishes[hit.facet_idx]` unconditionally would attempt an
        // out-of-bounds storage-buffer read whenever `facet_idx >= arrayLength(...)`
        // (harmless per WGSL's robustness guarantees, but its RESULT is
        // implementation-defined, so this must never be allowed to influence `finish`).
        var finish: u32 = 0u;
        if (hit.facet_idx < arrayLength(&facet_finishes)) {
            finish = facet_finishes[hit.facet_idx];
        }
        // Captured before any of the three bounce-dispatch arms below run -- see
        // optics::raytracer::dispatch_bounce's doc comment for why `was_internal_reflection`
        // is always computed against the PRE-bounce `inside_gem`. The TIR/reflect/refract
        // arms below never need this explicitly (they either never touch `inside_gem` at
        // all, or only flip it once at their very end, after which nothing else in that
        // arm reads it this bounce) -- only the frosted arm, which can update `inside_gem`
        // itself, needs the pre-bounce value spelled out separately.
        let pre_bounce_inside_gem = inside_gem;

        if (finish == FACET_FINISH_FROSTED) {
            // optics::raytracer::apply_frosted_bounce (via dispatch_bounce): the
            // REPLACEMENT for the TIR/partial-reflect/refract dispatch below, not an
            // addition to it -- see transport_physics.wgsl's own doc comment for the
            // achromatic-by-design physics this must preserve exactly.
            let fb = apply_frosted_bounce(
                is_anisotropic, sin2_t, n1, n2, cos_i, normal, inside_gem, is_extraordinary,
                seed0, bounce, &stokes, &path_pdf,
            );
            current_origin = hit_point + fb.new_dir * RAY_EPS;
            current_dir = fb.new_dir;
            // P2: a frosted (diffuse) bounce already depolarizes -- k collapses to S,
            // mirroring optics::raytracer::transport::dispatch_bounce's identical
            // treatment of FacetFinish::Frosted.
            current_k = fb.new_dir;
            inside_gem = fb.new_inside_gem != 0u;
            if (fb.has_extraordinary_update != 0u) {
                is_extraordinary = fb.extraordinary_update != 0u;
            }
            // Task 1: same was_internal_reflection formula optics::raytracer::dispatch_bounce
            // uses for every bounce kind -- true for the frosted TIR-forced and reflect
            // arms (inside_gem unchanged, no extraordinary update reported), false for
            // the transmit arm (inside_gem always flips there).
            let was_internal_reflection = pre_bounce_inside_gem
                && (fb.has_extraordinary_update == 0u)
                && (inside_gem == pre_bounce_inside_gem);
            if (is_anisotropic && was_internal_reflection) {
                // No stokes/path_pdf scaling here -- see internal_mode_coupling_draw's
                // doc comment: this is a RELABELING of which eigenmode governs the
                // NEXT bounce, not a SPLIT into two rays, so the matching unbiased
                // scale factor is 1.0 (no-op) -- the same 1.0 conclusion the entry
                // split itself reaches, for a different reason (see this file's header
                // comment / `apply_refract_bounce`'s doc comment on the CPU side).
                is_extraordinary = internal_mode_coupling_draw(seed0, bounce);
            }
        } else if (sin2_t > 1.0) {
            // Hero forced TIR (probability 1, no pdf division needed).
            for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                let n1k = select(1.0, n_medium_ch[k], inside_gem);
                let n2k = select(n_medium_ch[k], 1.0, inside_gem);
                let etak = n1k / n2k;
                let sin2_t_k = etak * etak * fma(-cos_i, cos_i, 1.0);
                if (sin2_t_k > 1.0) {
                    let delta_k = tir_phase_delta(n1k, cos_i, sin_i);
                    stokes[k] = mueller_tir_retardation(delta_k) * stokes[k];
                } else {
                    let cos_t_k = sqrt(max(1.0 - sin2_t_k, 0.0));
                    let r_s_k = fma(n2k, -cos_t_k, n1k * cos_i) / fma(n2k, cos_t_k, n1k * cos_i);
                    let r_p_k = fma(n1k, -cos_t_k, n2k * cos_i) / fma(n1k, cos_t_k, n2k * cos_i);
                    stokes[k] = mueller_fresnel_reflection(r_s_k, r_p_k) * stokes[k];
                    let r_unpol_k = clamp(0.5 * fma(r_p_k, r_p_k, r_s_k * r_s_k), R_UNPOL_MIN, R_UNPOL_MAX);
                    path_pdf[k] = path_pdf[k] * r_unpol_k;
                }
            }
            // P2: reflects the WAVE NORMAL `k` (not `S`), then re-derives `S'` for the
            // reflected `k'` via `poynting_dir_for_mode` -- see this file's own design
            // note above `wave_dir_at_bounce`.
            let k_prime = wave_dir_at_bounce - 2.0 * dot(wave_dir_at_bounce, normal) * normal;
            let s_prime = poynting_dir_for_mode(
                is_anisotropic, is_biaxial, inside_gem, is_extraordinary, k_prime, c_axis,
                n_o_hero, n_e_hero, n_alpha_hero, n_beta_hero, n_gamma_hero, biax_ax0, biax_ax1, biax_ax2,
            );
            current_origin = hit_point + s_prime * RAY_EPS;
            current_dir = s_prime;
            current_k = k_prime;

            // Task 1: TIR is always an internal reflection (`n1 > n2` for the hero
            // channel implies `inside_gem`, exactly as on the CPU side).
            if (is_anisotropic) {
                // No stokes/path_pdf scaling -- relabeling, not a split; see
                // internal_mode_coupling_draw's doc comment.
                is_extraordinary = internal_mode_coupling_draw(seed0, bounce);
            }
        } else {
            let cos_t = sqrt(1.0 - sin2_t);
            let r_s = fma(n2, -cos_t, n1 * cos_i) / fma(n2, cos_t, n1 * cos_i);
            let r_p = fma(n1, -cos_t, n2 * cos_i) / fma(n1, cos_t, n2 * cos_i);
            let r_unpol_raw = 0.5 * fma(r_p, r_p, r_s * r_s);
            let r_unpol = clamp(r_unpol_raw, R_UNPOL_MIN, R_UNPOL_MAX);
            let rng_bounce = f32(hash_u32(seed0 ^ hash_u32(bounce ^ FRESNEL_BRANCH_STREAM))) / 4294967295.0;

            if (rng_bounce < r_unpol) {
                // REFLECT: each channel applies its OWN Fresnel/TIR matrix, divided by
                // the SAME hero selection probability r_unpol (the PDF-division
                // coupling that makes GPU/CPU float divergence harmless -- see this
                // file's header comment).
                for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                    let n1k = select(1.0, n_medium_ch[k], inside_gem);
                    let n2k = select(n_medium_ch[k], 1.0, inside_gem);
                    let etak = n1k / n2k;
                    let sin2_t_k = etak * etak * fma(-cos_i, cos_i, 1.0);
                    var refl: mat4x4<f32>;
                    if (sin2_t_k > 1.0) {
                        let delta_k = tir_phase_delta(n1k, cos_i, sin_i);
                        refl = mueller_tir_retardation(delta_k);
                    } else {
                        let cos_t_k = sqrt(max(1.0 - sin2_t_k, 0.0));
                        let r_s_k = fma(n2k, -cos_t_k, n1k * cos_i) / fma(n2k, cos_t_k, n1k * cos_i);
                        let r_p_k = fma(n1k, -cos_t_k, n2k * cos_i) / fma(n1k, cos_t_k, n2k * cos_i);
                        let r_unpol_k = clamp(0.5 * fma(r_p_k, r_p_k, r_s_k * r_s_k), R_UNPOL_MIN, R_UNPOL_MAX);
                        path_pdf[k] = path_pdf[k] * r_unpol_k;
                        refl = mueller_fresnel_reflection(r_s_k, r_p_k);
                    }
                    stokes[k] = (refl * stokes[k]) * (1.0 / r_unpol);
                }
                // P2: reflects the WAVE NORMAL `k` (not `S`), then re-derives `S'` for
                // the reflected `k'` via `poynting_dir_for_mode` -- see this file's own
                // design note above `wave_dir_at_bounce`.
                let k_prime = wave_dir_at_bounce - 2.0 * dot(wave_dir_at_bounce, normal) * normal;
                let s_prime = poynting_dir_for_mode(
                    is_anisotropic, is_biaxial, inside_gem, is_extraordinary, k_prime, c_axis,
                    n_o_hero, n_e_hero, n_alpha_hero, n_beta_hero, n_gamma_hero, biax_ax0, biax_ax1, biax_ax2,
                );
                current_origin = hit_point + s_prime * RAY_EPS;
                current_dir = s_prime;
                current_k = k_prime;

                // Task 1: this arm never changes `inside_gem` (unlike the refract arm
                // below, which always flips it), so `inside_gem` here still holds its
                // pre-bounce value -- an internal reflection iff it was already true.
                if (is_anisotropic && inside_gem) {
                    // No stokes/path_pdf scaling -- relabeling, not a split; see
                    // internal_mode_coupling_draw's doc comment.
                    is_extraordinary = internal_mode_coupling_draw(seed0, bounce);
                }
            } else {
                // REFRACT -- optics::raytracer::apply_refract_bounce. Phase 3: on an
                // air->crystal entry into an anisotropic material (entering_anisotropic),
                // unpolarized light couples 50/50 into the ordinary/extraordinary
                // eigenmodes, drawn from BIREFRINGENT_SPLIT_STREAM -- NOT divided by
                // the 0.5 selection probability compared against, unlike r_unpol above:
                // each mode already carries only its own ~0.5 share of the incident
                // energy, and that share is drawn with the SAME 0.5 probability it is
                // weighted by, so the surviving mode's throughput is left unscaled --
                // see this file's header comment and `apply_refract_bounce`'s doc
                // comment on the CPU side. For a cubic material (or any bounce that is
                // not an anisotropic entry) entering_anisotropic is false and
                // use_extraordinary keeps whatever is_extraordinary already was --
                // bit-for-bit Phase 2 behaviour.
                let entering_anisotropic = !inside_gem && is_anisotropic;
                var use_extraordinary = is_extraordinary;
                if (entering_anisotropic) {
                    let split_rand = f32(hash_u32(seed0 ^ hash_u32(bounce ^ BIREFRINGENT_SPLIT_STREAM))) / 4294967295.0;
                    use_extraordinary = split_rand < 0.5;
                }

                // Direction: the mode-A eigenmode uses n_mode_a and (uniaxial only) is
                // never walked off; the mode-B eigenmode's ENERGY (Poynting) direction
                // is displaced by the walk-off angle -- computed BEFORE the per-channel
                // loop below so each companion channel's own hypothetical direction can
                // be compared against this SAME hero-driven direction (Fix G / Part 2 on
                // the CPU side). Phase 4: for a biaxial material entering the crystal,
                // NEITHER mode is a plain constant-index Snell refraction -- BOTH modes
                // walk off via `biaxial_mode_poynting_dir`, using `n_biax_a_hero`/
                // `n_biax_b_hero` (the SAME looked-up scalars the per-channel loop's own
                // `k == hero_idx` iteration uses) for self-consistency.
                // P2: `refr_wave_dir` is the SNELL-REFRACTED WAVE NORMAL `k'` (fed from
                // `wave_dir_at_bounce`, not `current_dir`/`S` -- see this file's
                // own design note above `wave_dir_at_bounce`, and rule 4 in particular:
                // "at exit into air, refract k (not S)"). Captured into `new_k` alongside
                // the Poynting-converted `final_refr_dir` (`S'`), since the caller needs
                // BOTH from here on.
                var new_k: vec3<f32>;
                var final_refr_dir: vec3<f32>;
                if (entering_anisotropic && is_biaxial) {
                    let n2_hero_dir = select(n_biax_a_hero, n_biax_b_hero, use_extraordinary);
                    let eta_dir = n1 / n2_hero_dir;
                    let sin2_t_dir = min(eta_dir * eta_dir * fma(-cos_i, cos_i, 1.0), 1.0);
                    let cos_t_dir = sqrt(max(1.0 - sin2_t_dir, 0.0));
                    let refr_wave_dir = normalize(
                        eta_dir * wave_dir_at_bounce + fma(eta_dir, cos_i, -cos_t_dir) * normal,
                    );
                    new_k = refr_wave_dir;
                    final_refr_dir = biaxial_mode_poynting_dir(n_alpha_hero, n_beta_hero, n_gamma_hero, biax_ax0, biax_ax1, biax_ax2, refr_wave_dir, use_extraordinary);
                } else {
                    let n2_hero_dir = select(n2, n_o_hero, entering_anisotropic && !use_extraordinary);
                    let eta_dir = n1 / n2_hero_dir;
                    let sin2_t_dir = min(eta_dir * eta_dir * fma(-cos_i, cos_i, 1.0), 1.0);
                    let cos_t_dir = sqrt(max(1.0 - sin2_t_dir, 0.0));
                    let refr_wave_dir = normalize(
                        eta_dir * wave_dir_at_bounce + fma(eta_dir, cos_i, -cos_t_dir) * normal,
                    );
                    new_k = refr_wave_dir;
                    final_refr_dir = refr_wave_dir;
                    if (entering_anisotropic && use_extraordinary) {
                        final_refr_dir = extraordinary_poynting_dir(refr_wave_dir, c_axis, n_o_hero, n_e_hero);
                    }
                }

                for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                    let n1k = select(1.0, n_medium_ch[k], inside_gem);
                    var n2k = select(n_medium_ch[k], 1.0, inside_gem);
                    if (entering_anisotropic && !use_extraordinary) {
                        if (is_biaxial) {
                            n2k = n_biax_a_ch[k];
                        } else {
                            n2k = n_o_hoisted[k];
                        }
                    }
                    let ratio_k = n1k / n2k;
                    let sin2_t_k = (ratio_k * ratio_k) * fma(-cos_i, cos_i, 1.0);
                    if (sin2_t_k > 1.0) {
                        // Chromatic termination: channel k cannot transmit at this
                        // angle even though the hero-driven path did. Both the Stokes
                        // contribution AND the path_pdf are dropped to exactly 0 --
                        // never just down-weighted (see this file's header comment).
                        stokes[k] = stokes[k] * 0.0;
                        path_pdf[k] = 0.0;
                        continue;
                    }
                    let cos_t_k = sqrt(max(1.0 - sin2_t_k, 0.0));
                    let refr_wave_dir_k = normalize(
                        ratio_k * wave_dir_at_bounce + fma(ratio_k, cos_i, -cos_t_k) * normal,
                    );
                    // Fix G (Part 2) / the direction-match identity trap: channel k's own
                    // walk-off, using k's own per-channel indices, compared against the
                    // STORED `final_refr_dir` above -- never a second recomputation of
                    // the hero's own direction (which would be a few ULP different and
                    // chromatically self-terminate the hero channel against itself).
                    // Phase 4: channel k's own biaxial walk-off, using k's own
                    // (n_alpha_ch[k], n_beta_ch := n_o_hoisted[k], n_gamma_ch[k]) evaluated
                    // at k's own single-shot refracted wave direction -- the direct
                    // per-channel generalization of the uniaxial extraordinary_poynting_dir
                    // call below.
                    var final_dir_k = refr_wave_dir_k;
                    if (entering_anisotropic && is_biaxial) {
                        final_dir_k = biaxial_mode_poynting_dir(n_alpha_ch[k], n_o_hoisted[k], n_gamma_ch[k], biax_ax0, biax_ax1, biax_ax2, refr_wave_dir_k, use_extraordinary);
                    } else if (entering_anisotropic && use_extraordinary) {
                        let n_e_k = n_o_hoisted[k] + birefringence_delta;
                        final_dir_k = extraordinary_poynting_dir(refr_wave_dir_k, c_axis, n_o_hoisted[k], n_e_k);
                    }
                    let direction_matches = dot(final_dir_k, final_refr_dir) >= DIRECTION_MATCH_COS_TOL;
                    if (direction_matches) {
                        let t_s_k = (2.0 * n1k * cos_i) / fma(n2k, cos_t_k, n1k * cos_i);
                        let t_p_k = (2.0 * n1k * cos_i) / fma(n1k, cos_t_k, n2k * cos_i);
                        let trans = mueller_fresnel_transmission(n1k, n2k, cos_i, cos_t_k, t_s_k, t_p_k);
                        // No `/ split_pdf` -- see the entering_anisotropic comment above.
                        stokes[k] = (trans * stokes[k]) * (1.0 / (1.0 - r_unpol));

                        let r_s_k = fma(n2k, -cos_t_k, n1k * cos_i) / fma(n2k, cos_t_k, n1k * cos_i);
                        let r_p_k = fma(n1k, -cos_t_k, n2k * cos_i) / fma(n1k, cos_t_k, n2k * cos_i);
                        let r_unpol_k = clamp(0.5 * fma(r_p_k, r_p_k, r_s_k * r_s_k), R_UNPOL_MIN, R_UNPOL_MAX);
                        // No `* split_pdf` -- scale-invariant under a uniform per-channel
                        // factor, was a pure no-op on the MIS weight; see refraction.rs.
                        path_pdf[k] = path_pdf[k] * (1.0 - r_unpol_k);
                    } else {
                        stokes[k] = stokes[k] * 0.0;
                        path_pdf[k] = 0.0;
                    }
                }
                current_origin = hit_point + final_refr_dir * RAY_EPS;
                current_dir = final_refr_dir;
                current_k = new_k;
                inside_gem = !inside_gem;
                if (entering_anisotropic) {
                    is_extraordinary = use_extraordinary;
                }
            }
        }

        if (bounce > 4u) {
            var max_intensity: f32 = 0.0;
            for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                max_intensity = max(max_intensity, max(stokes[k].x, 0.0));
            }
            let q = clamp(max_intensity, RR_FLOOR, 1.0);
            let rr_rand = f32(hash_u32(seed0 ^ hash_u32(bounce ^ RUSSIAN_ROULETTE_STREAM))) / 4294967295.0;
            if (rr_rand > q) {
                break;
            }
            for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
                stokes[k] = stokes[k] * (1.0 / q);
            }
        }
    }

    var sum_pdf: f32 = 0.0;
    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
        sum_pdf = sum_pdf + path_pdf[k];
    }
    var mis_weight: f32 = 1.0;
    if (sum_pdf > 1e-12) {
        mis_weight = f32(NUM_CHANNELS) * path_pdf[0] / sum_pdf;
    }

    var xyz = vec3<f32>(0.0, 0.0, 0.0);
    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
        let cmf = cie_1931_cmf(lambdas[k]);
        let weighted = radiance[k] * mis_weight;
        xyz = xyz + cmf * (weighted * NORM_FACTOR);
    }
    if (params.env_mode == 1u) {
        // Fix 3: Bradford-LMS-space von Kries adaptation, not a raw XYZ scale -- see
        // `apply_von_kries_white_balance`'s doc comment above and
        // `optics::raytracer::apply_von_kries_white_balance` on the CPU side.
        xyz = apply_von_kries_white_balance(xyz, params.white_balance);
    }

    out_xyz[idx * 3u + 0u] = xyz.x;
    out_xyz[idx * 3u + 1u] = xyz.y;
    out_xyz[idx * 3u + 2u] = xyz.z;
    // R4: `out_xyz` above is the only buffer a production dispatch
    // (`renderer::gpu::frame::GpuFrameRenderer::accumulate`) ever reads back -- the three
    // per-channel debug buffers below exist for Tier 2/spectral-debug self-tests only.
    // Guarding their writes on `params.write_debug_buffers` (nonzero for every self-test,
    // via `GpuTransportParams::new`'s default) lets a production dispatch bind tiny
    // fixed-size dummy buffers for them instead of buffers sized like `out_xyz` -- 9x
    // less write traffic and 9x more samples per chunk-budget dispatch. Every self-test's
    // own behaviour is UNCHANGED: `write_debug_buffers != 0u` takes the exact same branch
    // that ran unconditionally before this guard was added.
    if (params.write_debug_buffers != 0u) {
        for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
            out_radiance[idx * 8u + k] = radiance[k];
            out_lambdas[idx * 8u + k] = lambdas[k];
            out_path_pdf[idx * 8u + k] = path_pdf[k];
        }
    }
}

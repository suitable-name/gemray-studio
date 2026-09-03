// Phase 2 shared physics prelude -- the SINGLE definition of every ported function
// consumed by BOTH `spectral_transport.wgsl` (the megakernel, the actually-shipped
// path) and `transport_functions.wgsl` (the standalone kernels Tier 2's per-function
// ULP checks exercise, driven by `renderer::gpu::transport_check`).
//
// # Why this file exists
//
// Before it existed, this physics was hand-copied into both shader files (WGSL has no
// `#include`). The copies could drift -- and once, demonstrably, did: an injected fault
// in `transport_functions.wgsl`'s copy was caught precisely by Tier 2 with an exact
// argmax diagnostic, while the SAME fault in the megakernel's own copy was caught only
// marginally by Tier 3 (a handful of isolated-singleton pixels that would have
// independently passed at that sample budget). Tier 2 was validating a duplicate, not
// the shipped code. See `renderer::gpu::transport_check`'s module doc comment for the
// full story, and the fault-injection re-run recorded there proving this file closes
// the gap.
//
// # How it's wired in
//
// `build.rs` concatenates this file's text ahead of `spectral_transport.wgsl` and
// `transport_functions.wgsl` at build time into `$OUT_DIR/*.generated.wgsl` -- those
// generated files, not the checked-in `.wgsl` files directly, are what
// `renderer::gpu::estimator_check` and `renderer::gpu::transport_check` `include_str!`.
// Neither `spectral_transport.wgsl` nor `transport_functions.wgsl` is valid WGSL in
// isolation any more: both assume every symbol defined here is already in scope, the
// same way a Rust module assumes its `use` imports resolve.
//
// `build.rs`'s `GEMRAY_BUILD_ID` content hash walks `src/**/*.wgsl` on disk (this file
// included, since it lives under `src/renderer/shaders/`) -- NOT the generated
// `$OUT_DIR` output -- so the hash still fingerprints exactly the physics text that
// ships, just spread across one fewer duplicate copy than before.
//
// # Rules for editing this file
//
// Only functions/types genuinely shared VERBATIM between the megakernel and the Tier 2
// kernels belong here. Do not add bindings (`@group`/`@binding`) or entry points
// (`@compute`) -- those are necessarily per-file (the megakernel's bindings and the
// Tier 2 kernels' per-case bindings are entirely different shapes). A function that
// only one of the two files needs stays local to that file.

const PI: f32 = 3.14159265358979323846;

// ---------------------------------------------------------------------------------
// optics::raytracer::hash_u32 plus the per-bounce stream salts
// (FRESNEL_BRANCH_STREAM/RUSSIAN_ROULETTE_STREAM/BIREFRINGENT_SPLIT_STREAM/
// MODE_COUPLING_STREAM/FROSTED_DIR_U_STREAM/FROSTED_DIR_V_STREAM) and the frosted
// r_unpol clamp bounds (R_UNPOL_MIN/R_UNPOL_MAX).
//
// Task 2 GPU port (frosted girdle finish) moved these here from
// `spectral_transport.wgsl` (which used to define its own copy, the only consumer
// until now) so `apply_frosted_bounce` below -- shared verbatim between the megakernel
// and Tier 2's `transport_functions.wgsl` -- has them in scope in EITHER concatenated
// file without a second copy of the constants themselves. A pure move, not a value
// change: `spectral_transport.wgsl` no longer defines these (see that file's own
// comment at the old location) so there is exactly one definition per concatenated
// module, never two (WGSL rejects a duplicate top-level identifier).
const FRESNEL_BRANCH_STREAM: u32 = 0x9e3779b1u;
const RUSSIAN_ROULETTE_STREAM: u32 = 0x517cc1b7u;
const BIREFRINGENT_SPLIT_STREAM: u32 = 0x2545f491u;
const MODE_COUPLING_STREAM: u32 = 0xcc9e2d51u;
// Task 2 (girdle finish): the 2D cosine-weighted-hemisphere direction draw at a frosted
// bounce -- two independent streams for (u, v), mirroring
// optics::raytracer::{FROSTED_DIR_U_STREAM, FROSTED_DIR_V_STREAM}.
const FROSTED_DIR_U_STREAM: u32 = 0x27d4eb2fu;
const FROSTED_DIR_V_STREAM: u32 = 0x165667b1u;

const R_UNPOL_MIN: f32 = 1e-4;
const R_UNPOL_MAX: f32 = 1.0 - 1e-4;
const RAY_EPS: f32 = 1e-4;

fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x * 0x85ebca6bu;
    x = x ^ (x >> 13u);
    x = x * 0xc2b2ae35u;
    x = x ^ (x >> 16u);
    return x;
}

// ---------------------------------------------------------------------------------
// optics::polarization -- Stokes/Mueller matrix constructors. `StokesVector::
// apply_matrix` itself has no separate function to share: every call site below (in
// both consuming files) applies a matrix via the WGSL builtin `mat4x4<f32> *
// vec4<f32>` operator, so there is nothing hand-written to duplicate or dedupe there.
// ---------------------------------------------------------------------------------

// Fix 2 (see optics::polarization::MuellerMatrix::frame_rotation's doc comment): the
// CPU array was written row-major but fed to a column-major constructor, realizing
// R(-psi) instead of R(psi). WGSL's `mat4x4<f32>(col0, col1, col2, col3)` constructor
// is likewise column-major, so this mirrors the same fix -- column 1 gets `-s2` and
// column 2 gets `s2` (swapped from before) to build the textbook R(psi).
fn mueller_frame_rotation(psi: f32) -> mat4x4<f32> {
    let c2 = cos(2.0 * psi);
    let s2 = sin(2.0 * psi);
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, c2, -s2, 0.0),
        vec4<f32>(0.0, s2, c2, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}

fn mueller_fresnel_reflection(r_s: f32, r_p: f32) -> mat4x4<f32> {
    let rs2 = r_s * r_s;
    let rp2 = r_p * r_p;
    let a = 0.5 * (rs2 + rp2);
    let b = 0.5 * (rs2 - rp2);
    let c = r_s * r_p;
    return mat4x4<f32>(
        vec4<f32>(a, b, 0.0, 0.0),
        vec4<f32>(b, a, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, c, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, c),
    );
}

fn mueller_fresnel_transmission(n1: f32, n2: f32, cos_i: f32, cos_t: f32, t_s: f32, t_p: f32) -> mat4x4<f32> {
    let factor = (n2 * cos_t) / max(n1 * cos_i, 1e-6);
    let ts2 = t_s * t_s * factor;
    let tp2 = t_p * t_p * factor;
    let a = 0.5 * (ts2 + tp2);
    let b = 0.5 * (ts2 - tp2);
    let c = t_s * t_p * factor;
    return mat4x4<f32>(
        vec4<f32>(a, b, 0.0, 0.0),
        vec4<f32>(b, a, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, c, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, c),
    );
}

fn mueller_tir_retardation(delta: f32) -> mat4x4<f32> {
    let cos_d = cos(delta);
    let sin_d = sin(delta);
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, cos_d, -sin_d),
        vec4<f32>(0.0, 0.0, sin_d, cos_d),
    );
}

// optics::raytracer::tir_phase_delta -- TIR phase retardation delta = delta_p - delta_s.
fn tir_phase_delta(n1k: f32, cos_i: f32, sin_i: f32) -> f32 {
    let a = n1k * n1k * sin_i;
    let inner = max(fma(a, sin_i, -1.0), 0.0);
    let tan_half_delta_k = (cos_i * sqrt(inner)) / max(n1k * sin_i * sin_i, 1e-6);
    return 2.0 * atan(tan_half_delta_k);
}

fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (l2 > 1e-30) {
        return v / sqrt(l2);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// optics::raytracer::signed_frame_rotation_psi -- signed plane-of-incidence rotation
// angle between consecutive bounces, via atan2 (Fix 1 in the megakernel's own doc
// comment).
fn signed_frame_rotation_psi(prev: vec3<f32>, curr: vec3<f32>, axis: vec3<f32>) -> f32 {
    let cos_psi = clamp(dot(prev, curr), -1.0, 1.0);
    let sin_psi = dot(cross(prev, curr), normalize_or_zero(axis));
    return atan2(sin_psi, cos_psi);
}

fn degree_of_polarization(s: vec4<f32>) -> f32 {
    if (s.x <= 1e-7) {
        return 0.0;
    }
    let mag = sqrt(fma(s.w, s.w, fma(s.z, s.z, s.y * s.y)));
    return clamp(mag / s.x, 0.0, 1.0);
}

fn polarization_azimuth(s: vec4<f32>) -> f32 {
    return 0.5 * atan2(s.z, s.y);
}

fn arbitrary_perpendicular(n: vec3<f32>) -> vec3<f32> {
    var a: vec3<f32>;
    if (abs(n.x) > 0.9) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    return normalize_or_zero(a - n * dot(n, a));
}

// optics::polarization::electric_field_direction
fn electric_field_direction(s: vec4<f32>, s_axis: vec3<f32>, propagation_dir: vec3<f32>) -> vec3<f32> {
    let k_hat = normalize_or_zero(propagation_dir);
    let s_raw = s_axis - k_hat * dot(k_hat, s_axis);
    var s_hat: vec3<f32>;
    if (dot(s_raw, s_raw) > 1e-8) {
        s_hat = normalize(s_raw);
    } else {
        s_hat = arbitrary_perpendicular(k_hat);
    }
    let p_hat = cross(k_hat, s_hat);
    let psi = polarization_azimuth(s);
    let e = cos(psi) * s_hat + sin(psi) * p_hat;
    if (dot(e, e) > 1e-8) {
        return normalize(e);
    }
    return s_hat;
}

// ---------------------------------------------------------------------------------
// optics::birefringence -- the uniaxial pleochroic absorption path (exercised even
// for an isotropic material -- see `spectral_transport.wgsl`'s header comment).
// ---------------------------------------------------------------------------------

// `arbitrary_perpendicular` and this are two CPU-side names for the identical
// construction (a stable orthonormal vector perpendicular to `n`); kept as distinct
// WGSL functions here, as in both files before this dedup, to mirror the CPU naming
// each call site is ported from.
fn stable_orthonormal_basis_t(n: vec3<f32>) -> vec3<f32> {
    var a: vec3<f32>;
    if (abs(n.x) > 0.9) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    return normalize_or_zero(a - n * dot(n, a));
}

fn ordinary_eigen_polarization(wave_normal: vec3<f32>, c_axis: vec3<f32>) -> vec3<f32> {
    let crs = cross(wave_normal, c_axis);
    if (dot(crs, crs) > 1e-8) {
        return normalize(crs);
    }
    return stable_orthonormal_basis_t(normalize_or_zero(wave_normal));
}

fn extraordinary_eigen_polarization(wave_normal: vec3<f32>, c_axis: vec3<f32>) -> vec3<f32> {
    let o_hat = ordinary_eigen_polarization(wave_normal, c_axis);
    return normalize_or_zero(cross(wave_normal, o_hat));
}

fn quadratic_form(alpha_o: f32, alpha_e: f32, a1: vec3<f32>, a2: vec3<f32>, c: vec3<f32>, e_hat: vec3<f32>) -> f32 {
    let l0 = dot(a1, e_hat);
    let l1 = dot(a2, e_hat);
    let l2 = dot(c, e_hat);
    return alpha_o * (l0 * l0) + alpha_o * (l1 * l1) + alpha_e * (l2 * l2);
}

fn pleochroic_channel_alpha(
    alpha_o: f32,
    alpha_e: f32,
    c_axis: vec3<f32>,
    s_axis: vec3<f32>,
    propagation_dir: vec3<f32>,
    eigen_a: vec3<f32>,
    eigen_b: vec3<f32>,
    s: vec4<f32>,
) -> f32 {
    let c = normalize_or_zero(c_axis);
    let a1 = stable_orthonormal_basis_t(c);
    let a2 = cross(c, a1);
    let e_hat = electric_field_direction(s, s_axis, propagation_dir);
    let alpha_polarized = quadratic_form(alpha_o, alpha_e, a1, a2, c, e_hat);
    let alpha_unpolarized = 0.5 * (quadratic_form(alpha_o, alpha_e, a1, a2, c, eigen_a) + quadratic_form(alpha_o, alpha_e, a1, a2, c, eigen_b));
    let p = clamp(degree_of_polarization(s), 0.0, 1.0);
    return fma(p, alpha_polarized - alpha_unpolarized, alpha_unpolarized);
}

// ---------------------------------------------------------------------------------
// Phase 3 -- optics::birefringence::BirefringenceParams::{effective_extraordinary_index,
// walk_off_angle, extraordinary_poynting_dir} plus
// optics::raytracer::{theta_c_for_bounce, per_channel_uniaxial_indices} (the latter as a
// PER-CHANNEL function, `per_channel_uniaxial_index`, called once per channel by the
// caller -- see this file's own doc comment: the CPU original loops over
// `NUM_CHANNELS` internally, WGSL callers do that looping themselves and call this once
// per iteration, so the shared body stays the single source of truth for one channel's
// computation either way).
//
// `theta_c_for_bounce` below still omits the CPU function's `is_biaxial` parameter and
// always takes the `!inside_gem && is_anisotropic` branch: for a genuinely biaxial
// material (see the Phase 4 section further down for `BiaxialIndicatrix`'s own port)
// its output -- and `per_channel_uniaxial_index`'s below -- is a provably DEAD value.
// `spectral_transport.wgsl`'s `transport_main` selects the medium index for a bounce
// from the biaxial mode-A/mode-B arrays instead whenever `is_biaxial` is true, so the
// uniaxial `n_o_ch`/`n_eff_ch` this function's result feeds into is computed but never
// read for a biaxial material -- exactly the same "harmless, unused, still cheap"
// pattern this crate's CPU side documents elsewhere (see e.g.
// `BounceRefractionGeometry`'s uniaxial/biaxial field pairs). Taking the CPU's
// `!is_biaxial`-aware branch here would compute a DIFFERENT (also-unused) theta, but
// never a NaN/Inf one -- `n_o_hero_seed`/`birefringence_delta` are always finite,
// well-scaled index values regardless of crystal system -- so this simplification costs
// nothing observable in rendered output while avoiding a signature change to an
// already-verified, already-ULP-checked function.
// ---------------------------------------------------------------------------------

fn effective_extraordinary_index(n_o: f32, n_e: f32, theta: f32) -> f32 {
    if (abs(n_o - n_e) < 1e-5) {
        return n_o;
    }
    let sin_t = sin(theta);
    let cos_t = cos(theta);
    let ne_cos = n_e * cos_t;
    let no_sin = n_o * sin_t;
    let denom_sq = fma(ne_cos, ne_cos, no_sin * no_sin);
    if (denom_sq <= 1e-8) {
        return n_o;
    }
    return (n_o * n_e) / sqrt(denom_sq);
}

fn walk_off_angle(n_o: f32, n_e: f32, theta: f32) -> f32 {
    if (abs(n_o - n_e) < 1e-5) {
        return 0.0;
    }
    let n_o2 = n_o * n_o;
    let n_e2 = n_e * n_e;
    let sin_t = sin(theta);
    let cos_t = cos(theta);
    let numer = (n_o2 - n_e2) * sin_t * cos_t;
    let denom = max(fma(n_e2 * cos_t, cos_t, n_o2 * sin_t * sin_t), 1e-6);
    let tan_rho = numer / denom;
    return atan(tan_rho);
}

// Fix 1 (see optics::birefringence::BirefringenceParams::extraordinary_poynting_dir's
// doc comment): the optic axis is a director (S(c) == S(-c)), so fold `c_axis` onto the
// wave normal's own hemisphere via `sign` and negate the tilt on that branch to
// compensate -- `theta`/`delta` already use the unsigned `|cos_theta|` and so are
// branch-independent, but `c_proj` is built from the SIGNED `cos_theta` and flips sign
// under `c_axis -> -c_axis` while `delta` does not.
fn extraordinary_poynting_dir(wave_normal: vec3<f32>, c_axis: vec3<f32>, n_o: f32, n_e: f32) -> vec3<f32> {
    let cos_theta = clamp(dot(wave_normal, c_axis), -1.0, 1.0);
    let theta = acos(abs(cos_theta));
    let delta = walk_off_angle(n_o, n_e, theta);

    if (abs(delta) < 1e-5) {
        return wave_normal;
    }

    let c_proj = normalize_or_zero(c_axis - cos_theta * wave_normal);
    if (dot(c_proj, c_proj) < 1e-6) {
        return wave_normal;
    }
    var sign: f32 = 1.0;
    if (cos_theta < 0.0) {
        sign = -1.0;
    }

    return normalize(wave_normal * cos(delta) - sign * c_proj * sin(delta));
}

// optics::raytracer::theta_c_for_bounce -- see this section's header comment for why
// `is_biaxial` is omitted (always false for any material the GPU ever sees).
fn theta_c_for_bounce(
    normal: vec3<f32>,
    ray_dir: vec3<f32>,
    cos_i: f32,
    inside_gem: bool,
    is_anisotropic: bool,
    c_axis: vec3<f32>,
    n_o_hero_seed: f32,
    birefringence_delta: f32,
) -> f32 {
    if (!inside_gem && is_anisotropic) {
        let n_e_hero_seed = n_o_hero_seed + birefringence_delta;
        var n_guess = n_o_hero_seed;
        var theta: f32 = 0.0;
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let eta_guess = 1.0 / n_guess;
            let sin2_t_guess = eta_guess * eta_guess * fma(-cos_i, cos_i, 1.0);
            if (sin2_t_guess > 1.0) {
                break;
            }
            let cos_t_guess = sqrt(max(1.0 - sin2_t_guess, 0.0));
            let wave_dir_guess = normalize(eta_guess * ray_dir + fma(eta_guess, cos_i, -cos_t_guess) * normal);
            let cos_theta_wave = abs(clamp(dot(wave_dir_guess, c_axis), -1.0, 1.0));
            theta = acos(cos_theta_wave);
            n_guess = effective_extraordinary_index(n_o_hero_seed, n_e_hero_seed, theta);
        }
        return theta;
    } else {
        return acos(abs(clamp(dot(ray_dir, c_axis), -1.0, 1.0)));
    }
}

// ---------------------------------------------------------------------------------
// Phase 4: optics::birefringence::BiaxialIndicatrix -- the genuinely biaxial
// (three-distinct-principal-index) generalization of the uniaxial machinery above.
// Every function here is a direct, line-for-line port of the corresponding
// `BiaxialIndicatrix` method or free function, taking the indicatrix's three fields
// (`n_alpha`, `n_beta`, `n_gamma`, `axes` -- flattened to `ax0`/`ax1`/`ax2`, the three
// world-space principal-axis columns, alpha/beta/gamma respectively, matching
// `Mat3::from_cols(a1, a2, g)` in `BiaxialIndicatrix::from_gamma_axis`) as explicit
// parameters rather than reading a struct binding, mirroring `dispersion_evaluate`'s
// "one shared body, two different binding shapes" convention above -- the megakernel
// builds `ax0`/`ax1`/`ax2` once per ray via `biaxial_axes_from_gamma` (since they depend
// only on `c_axis`, constant across a ray's bounces) and Tier 2's per-case kernels build
// them once per case the same way.
// ---------------------------------------------------------------------------------

struct BiaxialAxes {
    ax0: vec3<f32>,
    ax1: vec3<f32>,
    ax2: vec3<f32>,
}

// optics::birefringence::BiaxialIndicatrix::from_gamma_axis's axis-frame construction
// (`stable_orthonormal_basis(gamma_axis)` completed to a right-handed orthonormal
// triple) -- pulled out on its own since every ported function below needs it and it
// depends only on `c_axis`/`gamma_axis`, never on wavelength or wave normal.
fn biaxial_axes_from_gamma(gamma_axis: vec3<f32>) -> BiaxialAxes {
    let g = normalize_or_zero(gamma_axis);
    let a1 = stable_orthonormal_basis_t(g);
    let a2 = cross(g, a1);
    var result: BiaxialAxes;
    result.ax0 = a1;
    result.ax1 = a2;
    result.ax2 = g;
    return result;
}

fn biaxial_b_coeffs(n_alpha: f32, n_beta: f32, n_gamma: f32) -> vec3<f32> {
    return vec3<f32>(1.0 / (n_alpha * n_alpha), 1.0 / (n_beta * n_beta), 1.0 / (n_gamma * n_gamma));
}

// optics::birefringence::BiaxialIndicatrix::indices_are_degenerate
fn biaxial_indices_degenerate(a: f32, b: f32) -> bool {
    let scale = max(max(abs(a), abs(b)), 1.0);
    return abs(a - b) <= sqrt(1.1920929e-7) * scale;
}

// optics::birefringence::BiaxialIndicatrix::uniaxial_wave_indices
fn biaxial_uniaxial_wave_indices(k_hat: vec3<f32>, axis: vec3<f32>, n_o: f32, n_e: f32) -> vec2<f32> {
    let theta = acos(abs(clamp(dot(k_hat, axis), -1.0, 1.0)));
    let n_eff = effective_extraordinary_index(n_o, n_e, theta);
    return vec2<f32>(max(n_o, n_eff), min(n_o, n_eff));
}

// optics::birefringence::BiaxialIndicatrix::wave_indices -- returns (n_slow, n_fast).
fn biaxial_wave_indices(
    n_alpha: f32, n_beta: f32, n_gamma: f32,
    ax0: vec3<f32>, ax1: vec3<f32>, ax2: vec3<f32>,
    wave_normal: vec3<f32>,
) -> vec2<f32> {
    let k = normalize_or_zero(wave_normal);

    let ab_degen = biaxial_indices_degenerate(n_alpha, n_beta);
    let bg_degen = biaxial_indices_degenerate(n_beta, n_gamma);

    if (ab_degen && bg_degen) {
        let n = (n_alpha + n_beta + n_gamma) / 3.0;
        return vec2<f32>(n, n);
    }
    if (ab_degen) {
        return biaxial_uniaxial_wave_indices(k, ax2, 0.5 * (n_alpha + n_beta), n_gamma);
    }
    if (bg_degen) {
        return biaxial_uniaxial_wave_indices(k, ax0, 0.5 * (n_beta + n_gamma), n_alpha);
    }

    let local = vec3<f32>(dot(ax0, k), dot(ax1, k), dot(ax2, k));
    let a2 = local.x * local.x;
    let b2 = local.y * local.y;
    let g2 = local.z * local.z;
    let bc = biaxial_b_coeffs(n_alpha, n_beta, n_gamma);

    let big_b = fma(a2, bc.y + bc.z, fma(b2, bc.x + bc.z, g2 * (bc.x + bc.y)));
    let big_c = fma(a2, bc.y * bc.z, fma(b2, bc.x * bc.z, g2 * bc.x * bc.y));
    let disc = sqrt(max(fma(big_b, big_b, -4.0 * big_c), 0.0));

    let x_lo = 0.5 * (big_b - disc);
    let x_hi = 0.5 * (big_b + disc);

    let n_slow = 1.0 / sqrt(max(x_lo, 1e-12));
    let n_fast = 1.0 / sqrt(max(x_hi, 1e-12));
    return vec2<f32>(n_slow, n_fast);
}

// Mirrors optics::birefringence::cross_fma / dot_fma op-for-op: explicit `fma` (not
// plain `*`/`-`/`dot`/`cross`) so this rounds identically to the CPU side -- see
// BiaxialIndicatrix::eigenvector_world's doc comment for why bit-parity matters here.
fn cross_fma(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        fma(a.y, b.z, -(a.z * b.y)),
        fma(a.z, b.x, -(a.x * b.z)),
        fma(a.x, b.y, -(a.y * b.x)),
    );
}

fn dot_fma(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
}

// Mirrors optics::birefringence::canonicalize_eigenvector_sign op-for-op -- see that
// function's doc comment for the tie-break convention (x, then y, then z priority).
fn canonicalize_eigenvector_sign(v: vec3<f32>) -> vec3<f32> {
    let ax = abs(v.x);
    let ay = abs(v.y);
    let az = abs(v.z);
    var largest: f32;
    if (ax >= ay && ax >= az) {
        largest = v.x;
    } else if (ay >= az) {
        largest = v.y;
    } else {
        largest = v.z;
    }
    if (largest < 0.0) {
        return -v;
    }
    return v;
}

// optics::birefringence::BiaxialIndicatrix::eigenvector_world -- see that Rust
// function's doc comment for the "transverse impermeability" (Gamma = P.B.P) matrix
// construction and the sign-aligned row-pair-cross-product null-vector extraction this
// mirrors op-for-op (reformulated 2026-09-02 for numerical conditioning; replaces the
// previous "cleared-denominator" polynomial form, and then again replaces a first
// largest-of-three-magnitude version of this construction that still measured up to
// ~640K ULP against the CPU side -- see the Rust doc comment for why the discrete
// argmax branch itself was the remaining problem).
//
// Mirrors optics::birefringence::BiaxialIndicatrix::precise_root_near op-for-op: an
// algebraically-exact discriminant reformulation that replaces `B^2 - 4C` (a
// subtraction of two ~1-magnitude sums) with direct, Sterbenz-exact differences of the
// principal `1/n^2` values -- see the Rust doc comment for the full derivation.
fn precise_root_near(local: vec3<f32>, b: vec3<f32>, x: f32) -> f32 {
    let a = local.x * local.x;
    let bb = local.y * local.y;
    let cc = local.z * local.z;
    let big_b = fma(a, b.y + b.z, fma(bb, b.x + b.z, cc * (b.x + b.y)));

    let xdiff = b.x - b.y;
    let ydiff = b.z - b.x;

    let a_plus_c = a + cc;
    let a_plus_bb = a + bb;
    let two_a_minus_bc = 2.0 * fma(bb, -cc, a);

    let disc_sq = fma(
        a_plus_c * a_plus_c,
        xdiff * xdiff,
        fma(a_plus_bb * a_plus_bb, ydiff * ydiff, two_a_minus_bc * xdiff * ydiff),
    );
    let disc = sqrt(max(disc_sq, 0.0));

    let x_lo = 0.5 * (big_b - disc);
    let x_hi = 0.5 * (big_b + disc);

    if (abs(x - x_lo) <= abs(x - x_hi)) {
        return x_lo;
    }
    return x_hi;
}

fn biaxial_eigenvector_world(
    ax0: vec3<f32>, ax1: vec3<f32>, ax2: vec3<f32>,
    local: vec3<f32>, b: vec3<f32>, x_in: f32, k_hat: vec3<f32>,
) -> vec3<f32> {
    let x = precise_root_near(local, b, x_in);

    let s = fma(b.x, local.x * local.x, fma(b.y, local.y * local.y, b.z * local.z * local.z));

    let m00 = fma(local.x * local.x, fma(-2.0, b.x, s), b.x - x);
    let m11 = fma(local.y * local.y, fma(-2.0, b.y, s), b.y - x);
    let m22 = fma(local.z * local.z, fma(-2.0, b.z, s), b.z - x);
    let m01 = local.x * local.y * (s - b.x - b.y);
    let m02 = local.x * local.z * (s - b.x - b.z);
    let m12 = local.y * local.z * (s - b.y - b.z);

    let row0 = vec3<f32>(m00, m01, m02);
    let row1 = vec3<f32>(m01, m11, m12);
    let row2 = vec3<f32>(m02, m12, m22);

    let c01 = cross_fma(row0, row1);
    let c02 = cross_fma(row0, row2);
    let c12 = cross_fma(row1, row2);

    var sign02 = 1.0;
    if (dot_fma(c01, c02) < 0.0) {
        sign02 = -1.0;
    }
    var sign12 = 1.0;
    if (dot_fma(c01, c12) < 0.0) {
        sign12 = -1.0;
    }
    let v_local = c01 + c02 * sign02 + c12 * sign12;

    let v_world = ax0 * v_local.x + ax1 * v_local.y + ax2 * v_local.z;
    if (dot(v_world, v_world) > 1e-12) {
        return normalize(canonicalize_eigenvector_sign(v_world));
    }
    return stable_orthonormal_basis_t(k_hat);
}

struct BiaxialEigenResult {
    d_slow: vec3<f32>,
    d_fast: vec3<f32>,
}

// optics::birefringence::BiaxialIndicatrix::eigen_polarizations
fn biaxial_eigen_polarizations(
    n_alpha: f32, n_beta: f32, n_gamma: f32,
    ax0: vec3<f32>, ax1: vec3<f32>, ax2: vec3<f32>,
    wave_normal: vec3<f32>,
) -> BiaxialEigenResult {
    let k = normalize_or_zero(wave_normal);
    let local = vec3<f32>(dot(ax0, k), dot(ax1, k), dot(ax2, k));
    let bc = biaxial_b_coeffs(n_alpha, n_beta, n_gamma);
    let ni = biaxial_wave_indices(n_alpha, n_beta, n_gamma, ax0, ax1, ax2, wave_normal);
    let x_slow = 1.0 / (ni.x * ni.x);
    let x_fast = 1.0 / (ni.y * ni.y);

    var result: BiaxialEigenResult;
    result.d_slow = biaxial_eigenvector_world(ax0, ax1, ax2, local, bc, x_slow, k);
    result.d_fast = biaxial_eigenvector_world(ax0, ax1, ax2, local, bc, x_fast, k);
    return result;
}

// optics::birefringence::BiaxialIndicatrix::d_to_e_direction
fn biaxial_d_to_e_direction(
    n_alpha: f32, n_beta: f32, n_gamma: f32,
    ax0: vec3<f32>, ax1: vec3<f32>, ax2: vec3<f32>,
    d_hat: vec3<f32>,
) -> vec3<f32> {
    let d_local = vec3<f32>(dot(ax0, d_hat), dot(ax1, d_hat), dot(ax2, d_hat));
    let e_local = vec3<f32>(
        d_local.x / (n_alpha * n_alpha),
        d_local.y / (n_beta * n_beta),
        d_local.z / (n_gamma * n_gamma),
    );
    return normalize_or_zero(ax0 * e_local.x + ax1 * e_local.y + ax2 * e_local.z);
}

// optics::birefringence::poynting_direction -- the general (uniaxial-or-biaxial)
// Poynting/walk-off direction from a wave normal and world-space E-field direction.
fn poynting_direction(wave_normal: vec3<f32>, e_field_hat: vec3<f32>) -> vec3<f32> {
    let k_hat = normalize_or_zero(wave_normal);
    let e_hat = normalize_or_zero(e_field_hat);
    let perp = k_hat - e_hat * dot(e_hat, k_hat);
    let len2 = dot(perp, perp);
    if (len2 > 1e-10) {
        return perp / sqrt(len2);
    }
    return k_hat;
}

// optics::birefringence::BiaxialIndicatrix::mode_poynting_dir
fn biaxial_mode_poynting_dir(
    n_alpha: f32, n_beta: f32, n_gamma: f32,
    ax0: vec3<f32>, ax1: vec3<f32>, ax2: vec3<f32>,
    wave_normal: vec3<f32>, want_slow: bool,
) -> vec3<f32> {
    let eig = biaxial_eigen_polarizations(n_alpha, n_beta, n_gamma, ax0, ax1, ax2, wave_normal);
    var d_hat: vec3<f32>;
    if (want_slow) {
        d_hat = eig.d_slow;
    } else {
        d_hat = eig.d_fast;
    }
    let e_hat = biaxial_d_to_e_direction(n_alpha, n_beta, n_gamma, ax0, ax1, ax2, d_hat);
    return poynting_direction(wave_normal, e_hat);
}

// optics::raytracer::refraction::poynting_dir_for_mode -- recovers the mode's Poynting
// (energy/ray) direction S for a freshly-reflected wave normal k. Returns k unchanged
// (S == k) outside the crystal, for an isotropic material, and for the uniaxial
// ORDINARY eigenmode -- see that function's own doc comment for the full rationale
// (bit-identity in every one of those cases). Hero-level scalars
// (n_alpha_hero/n_beta_hero/n_gamma_hero/biax_ax0/biax_ax1/biax_ax2, n_o_hero/n_e_hero,
// c_axis) are passed explicitly rather than read from a struct binding, mirroring this
// file's established "one shared body, explicit scalar parameters" convention.
fn poynting_dir_for_mode(
    is_anisotropic: bool,
    is_biaxial: bool,
    inside_gem: bool,
    is_extraordinary: bool,
    k: vec3<f32>,
    c_axis: vec3<f32>,
    n_o_hero: f32,
    n_e_hero: f32,
    n_alpha_hero: f32,
    n_beta_hero: f32,
    n_gamma_hero: f32,
    biax_ax0: vec3<f32>,
    biax_ax1: vec3<f32>,
    biax_ax2: vec3<f32>,
) -> vec3<f32> {
    if (!inside_gem || !is_anisotropic) {
        return k;
    }
    if (is_biaxial) {
        return biaxial_mode_poynting_dir(n_alpha_hero, n_beta_hero, n_gamma_hero, biax_ax0, biax_ax1, biax_ax2, k, is_extraordinary);
    }
    if (is_extraordinary) {
        return extraordinary_poynting_dir(k, c_axis, n_o_hero, n_e_hero);
    }
    return k;
}

struct BiaxialResolveResult {
    n: f32,
    wave_dir: vec3<f32>,
}

// optics::birefringence::BiaxialIndicatrix::resolve_entry_mode -- the two-iteration
// fixed point resolving a biaxial mode's refracted wave-normal direction at an
// air->crystal entry.
fn biaxial_resolve_entry_mode(
    n_alpha: f32, n_beta: f32, n_gamma: f32,
    ax0: vec3<f32>, ax1: vec3<f32>, ax2: vec3<f32>,
    incident_dir: vec3<f32>, normal: vec3<f32>, cos_i: f32, n_seed: f32, want_slow: bool,
) -> BiaxialResolveResult {
    var n_guess = n_seed;
    var wave_dir = incident_dir;
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let eta_guess = 1.0 / n_guess;
        let sin2_t_guess = eta_guess * eta_guess * fma(-cos_i, cos_i, 1.0);
        if (sin2_t_guess > 1.0) {
            break;
        }
        let cos_t_guess = sqrt(max(1.0 - sin2_t_guess, 0.0));
        wave_dir = normalize(eta_guess * incident_dir + fma(eta_guess, cos_i, -cos_t_guess) * normal);
        let ni = biaxial_wave_indices(n_alpha, n_beta, n_gamma, ax0, ax1, ax2, wave_dir);
        if (want_slow) {
            n_guess = ni.x;
        } else {
            n_guess = ni.y;
        }
    }
    var result: BiaxialResolveResult;
    result.n = n_guess;
    result.wave_dir = wave_dir;
    return result;
}

// optics::birefringence::AbsorptionTensor3::biaxial + AbsorptionTensor3::quadratic_form
// -- the three-independent-principal-coefficient generalization of `quadratic_form`
// above. Reconstructs its axis frame fresh from `c_axis` (never reuses a caller's
// `ax0`/`ax1`/`ax2`), exactly mirroring how the uniaxial `quadratic_form`/
// `pleochroic_channel_alpha` pair above independently rebuild theirs -- both
// constructions are bit-identical to `biaxial_axes_from_gamma` on the same input (see
// `birefringence::biaxial_reduction_tests::absorption_frame_is_bit_identical_to_index_frame`
// on the CPU side), so which call site does the rebuilding is a style choice, not a
// correctness one.
fn quadratic_form3(alpha: f32, beta: f32, gamma: f32, a1: vec3<f32>, a2: vec3<f32>, c: vec3<f32>, e_hat: vec3<f32>) -> f32 {
    let l0 = dot(a1, e_hat);
    let l1 = dot(a2, e_hat);
    let l2 = dot(c, e_hat);
    return alpha * (l0 * l0) + beta * (l1 * l1) + gamma * (l2 * l2);
}

// optics::birefringence::pleochroic_channel_alpha with `alpha_beta = Some(alpha_beta)`
// -- the genuinely biaxial (trichroic) three-coefficient absorption path.
fn pleochroic_channel_alpha_biaxial(
    alpha_o: f32,
    alpha_beta: f32,
    alpha_e: f32,
    c_axis: vec3<f32>,
    s_axis: vec3<f32>,
    propagation_dir: vec3<f32>,
    eigen_a: vec3<f32>,
    eigen_b: vec3<f32>,
    s: vec4<f32>,
) -> f32 {
    let c = normalize_or_zero(c_axis);
    let a1 = stable_orthonormal_basis_t(c);
    let a2 = cross(c, a1);
    let e_hat = electric_field_direction(s, s_axis, propagation_dir);
    let alpha_polarized = quadratic_form3(alpha_o, alpha_beta, alpha_e, a1, a2, c, e_hat);
    let alpha_unpolarized = 0.5 * (quadratic_form3(alpha_o, alpha_beta, alpha_e, a1, a2, c, eigen_a) + quadratic_form3(alpha_o, alpha_beta, alpha_e, a1, a2, c, eigen_b));
    let p = clamp(degree_of_polarization(s), 0.0, 1.0);
    return fma(p, alpha_polarized - alpha_unpolarized, alpha_unpolarized);
}

// ---------------------------------------------------------------------------------
// optics::dispersion::DispersionModel::evaluate -- takes the dispersion params as
// explicit arguments (rather than reading a `material: GpuGemMaterial` binding
// directly) so the exact same function body is callable both from the megakernel
// (which has that binding) and from Tier 2's `dispersion_main` (which reads per-case
// values out of its own `DispersionCase` storage buffer instead).
// ---------------------------------------------------------------------------------

fn dispersion_evaluate(model_type: u32, param_a: vec4<f32>, param_b: vec4<f32>, lambda_nm: f32) -> f32 {
    let lambda_um = lambda_nm * 1e-3;
    let l2 = lambda_um * lambda_um;
    if (model_type == 0u) {
        let n2 = 1.0 + (param_a.x * l2) / (l2 - param_b.x);
        return sqrt(max(n2, 1.0));
    } else if (model_type == 1u) {
        var n2: f32 = 1.0;
        n2 = n2 + (param_a.x * l2) / (l2 - param_b.x);
        n2 = n2 + (param_a.y * l2) / (l2 - param_b.y);
        n2 = n2 + (param_a.z * l2) / (l2 - param_b.z);
        return sqrt(max(n2, 1.0));
    } else {
        let l4 = l2 * l2;
        return param_a.x + (param_a.y / l2) + (param_a.z / l4);
    }
}

// optics::raytracer::per_channel_uniaxial_indices -- one channel's (n_o, n_eff) pair;
// see the Phase 3 section header comment above for why the CPU's internal
// NUM_CHANNELS loop is the WGSL caller's responsibility instead of this function's.
// Placed after `dispersion_evaluate` (which it calls) rather than up in the Phase 3
// section above, purely so every function here is defined after everything it calls.
fn per_channel_uniaxial_index(
    model_type: u32,
    param_a: vec4<f32>,
    param_b: vec4<f32>,
    lambda_nm: f32,
    birefringence_delta: f32,
    is_anisotropic: bool,
    theta_c: f32,
) -> vec2<f32> {
    let n_o_k = dispersion_evaluate(model_type, param_a, param_b, lambda_nm);
    let n_e_k = n_o_k + birefringence_delta;
    var n_eff_k = n_o_k;
    if (is_anisotropic) {
        n_eff_k = effective_extraordinary_index(n_o_k, n_e_k, theta_c);
    }
    return vec2<f32>(n_o_k, n_eff_k);
}

// ---------------------------------------------------------------------------------
// optics::raytracer::spectral_absorption -- takes the band array/count as explicit
// arguments (rather than reading `material.o_ray_bands`/`material.e_ray_bands`
// directly) for the same reason as `dispersion_evaluate` above: one shared body, two
// different binding shapes at the call sites. The megakernel calls this once per
// eigenmode (`material.o_ray_bands`/`o_ray_band_count`, then `e_ray_bands`/
// `e_ray_band_count`) where it previously had two near-identical `_o`/`_e` copies of
// this function.
// ---------------------------------------------------------------------------------

struct AbsorptionBand {
    center_nm: f32,
    width_nm: f32,
    peak: f32,
}

fn spectral_absorption(bands: array<AbsorptionBand, 8>, band_count: u32, lambda_nm: f32) -> f32 {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < band_count; i = i + 1u) {
        let band = bands[i];
        let t = (lambda_nm - band.center_nm) / band.width_nm;
        sum = sum + band.peak * exp(-0.5 * t * t);
    }
    return sum;
}

// ---------------------------------------------------------------------------------
// Task 2 GPU port: optics::raytracer::{frosted_orthonormal_basis,
// cosine_weighted_hemisphere, apply_frosted_bounce} -- the diffuse (bruted/frosted
// girdle facet) bounce, ported here (not into `spectral_transport.wgsl` or
// `transport_functions.wgsl` separately) so the shipped megakernel and Tier 2's
// standalone `frosted_bounce_main`/`cosine_hemisphere_main` kernels call the exact same
// function object, never two texts that could drift -- see this file's own header
// comment for why that property matters (a duplicate-vs-shipped-code fault was
// previously caught only by luck, see `renderer::gpu::transport_check`'s module doc
// comment).
//
// # The one deliberate simplification, preserved exactly
//
// `apply_frosted_bounce` is achromatic BY DESIGN: every spectral channel shares the ONE
// direction drawn below (not a per-channel direction) and the ONE broadband
// reflect/transmit split `r_unpol` (computed from the HERO channel's `n1`/`n2`/`cos_i`
// only -- never a per-channel `r_unpol_k`). That is what lets a frosted bounce compose
// with the existing per-channel `path_pdf` bookkeeping and the final
// `spectral_mis_weight`/MIS combination with NO chromatic-termination guard: a smooth,
// finite-support hemisphere BSDF assigns strictly positive density to the realized
// direction under every channel's own hypothetical hero-driven technique (unlike a
// delta BSDF, whose density is exactly zero off its one wavelength-dependent
// direction), so there is no measure-zero mismatch to drop to zero. See
// `optics::raytracer::apply_frosted_bounce`'s own doc comment for the full derivation
// -- this WGSL translation must never diverge from it: no per-channel direction, no
// per-channel `r_unpol_k`, no extra `path_pdf` division (the cosine-weighted-hemisphere
// pdf already exactly cancels the assumed Lambertian `albedo = 1.0` BRDF/BTDF, folded
// into the `1.0 / r_unpol` / `1.0 / t_unpol` throughput scale below).
// ---------------------------------------------------------------------------------

struct FrostedBasis {
    t: vec3<f32>,
    b: vec3<f32>,
}

// optics::raytracer::frosted_orthonormal_basis
fn frosted_orthonormal_basis(n: vec3<f32>) -> FrostedBasis {
    var a: vec3<f32>;
    if (abs(n.x) > 0.9) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize_or_zero(a - n * dot(n, a));
    let b = cross(n, t);
    var result: FrostedBasis;
    result.t = t;
    result.b = b;
    return result;
}

// optics::raytracer::cosine_weighted_hemisphere -- Malley's method (polar mapping, not
// the concentric-disk variant), so this is a direct line-for-line translation.
fn cosine_weighted_hemisphere(u1: f32, u2: f32, n: vec3<f32>) -> vec3<f32> {
    let r = sqrt(u1);
    let theta = 2.0 * PI * u2;
    let sin_t = sin(theta);
    let cos_t = cos(theta);
    let basis = frosted_orthonormal_basis(n);
    let dir = basis.t * (r * cos_t) + basis.b * (r * sin_t) + n * sqrt(max(1.0 - u1, 0.0));
    return normalize_or_zero(dir);
}

// The (new_dir, new_inside_gem, has_extraordinary_update, extraordinary_update) tuple
// `apply_frosted_bounce` returns, encoded for WGSL (which has no `Option<bool>`):
// `has_extraordinary_update == 0u` is the CPU's `None` (the TIR-forced and reflect
// arms); `!= 0u` is `Some(extraordinary_update != 0u)` (only reachable from the
// transmit arm's `entering_anisotropic` branch, mirroring
// optics::raytracer::apply_frosted_bounce's `entering_anisotropic.then_some(..)`).
struct FrostedBounceResult {
    new_dir: vec3<f32>,
    new_inside_gem: u32,
    has_extraordinary_update: u32,
    extraordinary_update: u32,
}

// optics::raytracer::apply_frosted_bounce -- the CPU signature takes `&RayMaterialContext`
// / `&BounceRefractionGeometry`; this WGSL translation flattens exactly the fields that
// function actually reads out of them (`ctx.is_anisotropic` and
// `geo.{sin2_t,n1,n2,cos_i}` -- see the CPU function's own doc comment) into explicit
// scalar/vector parameters, the same flattening convention every other ported function
// in this file already uses for its CPU struct-based counterpart (e.g.
// `theta_c_for_bounce` above). `stokes`/`path_pdf` are `ptr<function, ...>` so this
// mutates the caller's own local arrays in place, mirroring the CPU's `&mut
// [StokesVector; NUM_CHANNELS]` / `&mut [f32; NUM_CHANNELS]` out-parameters exactly.
fn apply_frosted_bounce(
    is_anisotropic: bool,
    sin2_t: f32,
    n1: f32,
    n2: f32,
    cos_i: f32,
    normal: vec3<f32>,
    inside_gem: bool,
    is_extraordinary: bool,
    rng_seed: u32,
    bounce: u32,
    stokes: ptr<function, array<vec4<f32>, 8>>,
    path_pdf: ptr<function, array<f32, 8>>,
) -> FrostedBounceResult {
    let u1 = f32(hash_u32(rng_seed ^ hash_u32(bounce ^ FROSTED_DIR_U_STREAM))) / 4294967295.0;
    let u2 = f32(hash_u32(rng_seed ^ hash_u32(bounce ^ FROSTED_DIR_V_STREAM))) / 4294967295.0;

    var result: FrostedBounceResult;

    if (sin2_t > 1.0) {
        // Forced reflect (TIR), probability 1 -- no draw, no pdf division, mirroring
        // optics::raytracer::apply_tir_bounce's identical reasoning for the polished
        // path.
        let new_dir = cosine_weighted_hemisphere(u1, u2, normal);
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let intensity = max((*stokes)[k].x, 0.0);
            (*stokes)[k] = vec4<f32>(intensity, 0.0, 0.0, 0.0);
        }
        result.new_dir = new_dir;
        result.new_inside_gem = select(0u, 1u, inside_gem);
        result.has_extraordinary_update = 0u;
        result.extraordinary_update = 0u;
        return result;
    }

    let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
    let r_s = fma(n2, -cos_t, n1 * cos_i) / fma(n2, cos_t, n1 * cos_i);
    let r_p = fma(n1, -cos_t, n2 * cos_i) / fma(n1, cos_t, n2 * cos_i);
    let r_unpol = clamp(0.5 * fma(r_p, r_p, r_s * r_s), R_UNPOL_MIN, R_UNPOL_MAX);
    let rng_bounce = f32(hash_u32(rng_seed ^ hash_u32(bounce ^ FRESNEL_BRANCH_STREAM))) / 4294967295.0;

    if (rng_bounce < r_unpol) {
        let new_dir = cosine_weighted_hemisphere(u1, u2, normal);
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let intensity = max((*stokes)[k].x, 0.0) / r_unpol;
            (*stokes)[k] = vec4<f32>(intensity, 0.0, 0.0, 0.0);
            (*path_pdf)[k] = (*path_pdf)[k] * r_unpol;
        }
        result.new_dir = new_dir;
        result.new_inside_gem = select(0u, 1u, inside_gem);
        result.has_extraordinary_update = 0u;
        result.extraordinary_update = 0u;
        return result;
    }

    let new_dir = cosine_weighted_hemisphere(u1, u2, -normal);
    let entering_anisotropic = (!inside_gem) && is_anisotropic;
    // Mode SELECTION is still a stochastic 50/50 draw -- only the throughput weighting
    // that used to accompany it (a `split_pdf` divisor/multiplier) is gone, since it
    // estimated twice the transmitted energy no interface can deliver. See
    // optics::raytracer::apply_frosted_bounce's doc comment for the full energy-share
    // reasoning (same shape as the polished path's entry split in refraction.rs).
    var use_extraordinary = is_extraordinary;
    if (entering_anisotropic) {
        let split_rand = f32(hash_u32(rng_seed ^ hash_u32(bounce ^ BIREFRINGENT_SPLIT_STREAM))) / 4294967295.0;
        use_extraordinary = split_rand < 0.5;
    }
    let t_unpol = 1.0 - r_unpol;
    // No `/ split_pdf` -- see the entering_anisotropic comment above.
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let intensity = max((*stokes)[k].x, 0.0) / t_unpol;
        (*stokes)[k] = vec4<f32>(intensity, 0.0, 0.0, 0.0);
        // No `* split_pdf` -- scale-invariant under a uniform per-channel factor, was a
        // pure no-op on the MIS weight; see refraction.rs.
        (*path_pdf)[k] = (*path_pdf)[k] * t_unpol;
    }
    result.new_dir = new_dir;
    result.new_inside_gem = select(1u, 0u, inside_gem);
    if (entering_anisotropic) {
        result.has_extraordinary_update = 1u;
        result.extraordinary_update = select(0u, 1u, use_extraordinary);
    } else {
        result.has_extraordinary_update = 0u;
        result.extraordinary_update = 0u;
    }
    return result;
}

// ---------------------------------------------------------------------------------
// Physics review, Task 1 GPU port: inclusion/subsurface scattering
// (optics::raytracer::{henyey_greenstein_phase, sample_henyey_greenstein_direction,
// maybe_scatter_or_extinguish}) -- ported here (not into `spectral_transport.wgsl` or
// `transport_functions.wgsl` separately) for the exact same reason `apply_frosted_bounce`
// lives here: the shipped megakernel and Tier 2's standalone kernels must call the SAME
// function object, never two texts that could drift (see this file's own header
// comment).
//
// `maybe_scatter_or_extinguish` takes the per-channel absorption coefficients
// (`alphas`) as an explicit `array<f32, 8>` argument rather than recomputing them from
// band data itself, mirroring `dispersion_evaluate`'s "one shared body, two different
// binding shapes at the call sites" convention above: the megakernel already computes
// this exact per-channel array inline (the pre-Task-1 absorption block,
// `spectral_absorption` + `pleochroic_channel_alpha`), so this function starts from
// that array rather than re-deriving it -- see `optics::raytracer::maybe_scatter_or_extinguish`'s
// doc comment for the full estimator derivation (hazards 1-5); this is a line-for-line
// translation of that function's body, `channel_absorption_alphas`'s work already done
// by the caller.
// ---------------------------------------------------------------------------------

// optics::raytracer::{DISTANCE_SAMPLE_STREAM, PHASE_DIR_U_STREAM, PHASE_DIR_V_STREAM}.
const DISTANCE_SAMPLE_STREAM: u32 = 0xa24baed4u;
const PHASE_DIR_U_STREAM: u32 = 0x9fb21c65u;
const PHASE_DIR_V_STREAM: u32 = 0x1ce4e5b9u;

// optics::raytracer::henyey_greenstein_phase
fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = pow(max(fma(2.0 * g, -cos_theta, 1.0 + g2), 1e-6), 1.5);
    return (1.0 - g2) / (4.0 * PI * denom);
}

// optics::raytracer::sample_henyey_greenstein_direction
fn sample_henyey_greenstein_direction(u1: f32, u2: f32, g: f32, forward: vec3<f32>) -> vec3<f32> {
    var cos_theta: f32;
    if (abs(g) < 1e-3) {
        cos_theta = fma(-2.0, u1, 1.0);
    } else {
        let one_minus_g2 = fma(-g, g, 1.0);
        let denom = fma(2.0 * g, u1, 1.0 - g);
        let sq = one_minus_g2 / denom;
        cos_theta = fma(-sq, sq, fma(g, g, 1.0)) / (2.0 * g);
    }
    cos_theta = clamp(cos_theta, -1.0, 1.0);
    let sin_theta = sqrt(max(fma(cos_theta, -cos_theta, 1.0), 0.0));
    let phi = 2.0 * PI * u2;
    let sin_p = sin(phi);
    let cos_p = cos(phi);
    let basis = frosted_orthonormal_basis(forward);
    let dir = basis.t * (sin_theta * cos_p) + basis.b * (sin_theta * sin_p) + forward * cos_theta;
    return normalize_or_zero(dir);
}

// The `Option<(f32, Vec3)>` `maybe_scatter_or_extinguish` returns, encoded for WGSL:
// `scattered == 0u` is the CPU's `None` (`t_free`/`new_dir` unset, ignored by the
// caller); `!= 0u` is `Some((t_free, new_dir))`.
struct ScatterOrExtinguishResult {
    scattered: u32,
    t_free: f32,
    new_dir: vec3<f32>,
}

// optics::raytracer::maybe_scatter_or_extinguish. `stokes`/`path_pdf` are
// `ptr<function, ...>`, mirroring `apply_frosted_bounce`'s identical in-place-mutation
// convention above.
fn maybe_scatter_or_extinguish(
    alphas: array<f32, 8>,
    sigma_s: f32,
    g: f32,
    ray_dir: vec3<f32>,
    hit_t: f32,
    path_scale: f32,
    rng_seed: u32,
    bounce: u32,
    stokes: ptr<function, array<vec4<f32>, 8>>,
    path_pdf: ptr<function, array<f32, 8>>,
) -> ScatterOrExtinguishResult {
    let sigma_t_hero = alphas[0] + sigma_s;

    let dist_rand = f32(hash_u32(rng_seed ^ hash_u32(bounce ^ DISTANCE_SAMPLE_STREAM))) / 4294967295.0;
    let one_minus_u = max(1.0 - dist_rand, 1e-7);
    let t_free = -(log(one_minus_u)) / sigma_t_hero;

    // P1 (absorption path scale): model units -> absorption-length units. See
    // optics::materials::GemMaterial::absorption_path_scale and the CPU
    // maybe_scatter_or_extinguish's own doc comment. `path_scale == 1.0` (every
    // built-in) is an exact no-op.
    let hit_t_scaled = hit_t * path_scale;

    var result: ScatterOrExtinguishResult;

    if (t_free < hit_t_scaled) {
        let pdf_hero = sigma_t_hero * one_minus_u;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let sigma_t_k = alphas[k] + sigma_s;
            let tr_k = exp(-sigma_t_k * t_free);
            let weight = tr_k * sigma_s / pdf_hero;
            let intensity = max((*stokes)[k].x, 0.0) * weight;
            (*stokes)[k] = vec4<f32>(intensity, 0.0, 0.0, 0.0);
            (*path_pdf)[k] = (*path_pdf)[k] * (sigma_t_k * tr_k);
        }
        let u1 = f32(hash_u32(rng_seed ^ hash_u32(bounce ^ PHASE_DIR_U_STREAM))) / 4294967295.0;
        let u2 = f32(hash_u32(rng_seed ^ hash_u32(bounce ^ PHASE_DIR_V_STREAM))) / 4294967295.0;
        result.scattered = 1u;
        // Convert the sampled free-path distance back to MODEL units -- see the CPU
        // function's own doc comment for why this unit conversion preserves the
        // estimator's unbiasedness. `path_scale == 1.0` is an exact no-op division.
        result.t_free = t_free / path_scale;
        result.new_dir = sample_henyey_greenstein_direction(u1, u2, g, ray_dir);
        return result;
    }

    let survive_hero = exp(-sigma_t_hero * hit_t_scaled);
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let sigma_t_k = alphas[k] + sigma_s;
        let survive_k = exp(-sigma_t_k * hit_t_scaled);
        (*stokes)[k] = (*stokes)[k] * (survive_k / max(survive_hero, 1e-30));
        (*path_pdf)[k] = (*path_pdf)[k] * survive_k;
    }
    result.scattered = 0u;
    result.t_free = 0.0;
    result.new_dir = vec3<f32>(0.0, 0.0, 0.0);
    return result;
}

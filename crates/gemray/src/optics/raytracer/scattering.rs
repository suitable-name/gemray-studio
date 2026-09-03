//! Henyey-Greenstein volumetric scattering and the frosted-facet BSDF.
//!
//! [`maybe_scatter_or_extinguish`] per-bounce extinction/
//! scatter estimator, the HG phase function and its importance sampler, and //! [`apply_frosted_bounce`] diffuse reflect/transmit dispatch.

use super::{
    NUM_CHANNELS,
    absorption::channel_absorption_alphas,
    camera::Ray,
    refraction::{BounceRefractionGeometry, RayMaterialContext, RayWavelengthCache},
    sampling::{
        BIREFRINGENT_SPLIT_STREAM, DISTANCE_SAMPLE_STREAM, FRESNEL_BRANCH_STREAM,
        FROSTED_DIR_U_STREAM, FROSTED_DIR_V_STREAM, PHASE_DIR_U_STREAM, PHASE_DIR_V_STREAM,
        hash_u32,
    },
    transport::apply_russian_roulette,
};
use crate::optics::{materials::GemMaterial, polarization::StokesVector};
use glam::Vec3;

/// Homogeneous
/// Henyey-Greenstein volumetric scattering, decided/sampled once for the shared
/// hero-driven geometric path and reweighted per channel -- the SAME structural pattern
/// [`apply_partial_fresnel_bounce`]/[`apply_refract_channel`] already establish for
/// every other stochastic decision in this module tree.
///
/// `alphas` is the caller's precomputed [`channel_absorption_alphas`] result (the
/// caller in `trace_spectral_ray_inner` computes it once, immediately before calling
/// this function) rather than derived internally from a `RayMaterialContext` -- this
/// function only needs the resulting per-channel scalars, never the geometry/eigenmode
/// machinery that produces them, and taking them as an explicit parameter is what lets
/// `shaders/transport_physics.wgsl`'s WGSL translation (and its Tier 2 GPU self-test,
/// `renderer::gpu::transport_check::run_scatter_or_extinguish`) share the exact same
/// function signature shape as the megakernel's own inline alpha computation, the same
/// "one shared body, two different binding shapes at the call sites" convention
/// `dispersion_evaluate` already establishes on the WGSL side.
///
/// # The estimator, hazard by hazard
///
/// 1. **Extinction, not absorption.** `sigma_t = sigma_a + sigma_s` (`sigma_a` is this
///    channel's own pleochroic [`channel_absorption_alphas`] value -- unchanged,
///    chromatic, exactly what [`apply_absorption`] already used; `sigma_s` is the
///    material's single achromatic [`GemMaterial::scattering_sigma_s`]). A scattering
///    event REDIRECTS the path rather than destroying its energy: the `sigma_s / sigma_t`
///    single-scattering albedo below is a redirection weight, never treated as loss on
///    its own -- loss only happens through `sigma_a`, which is already what
///    `apply_absorption` charged.
/// 2. **Every stochastic decision divides by the SAME value that drove it.** This
///    function draws exactly ONE `[0, 1)` random (`DISTANCE_SAMPLE_STREAM`) to invert
///    the HERO channel's own exponential CDF (`sigma_t_hero`) into a free-path distance
///    `t_free` -- the classic homogeneous-medium analog sampler. Both branches below
///    divide by the density THAT draw actually has under the hero's own technique
///    (`pdf_hero` in the scatter branch; `survive_hero` in the no-scatter branch) --
///    never a per-channel value in the denominator -- exactly mirroring how
///    `apply_partial_reflect_bounce` divides every channel's Stokes by the HERO's
///    `r_unpol`, not `r_unpol_k`. Each companion channel k's OWN physical weight
///    (`tr_k`/`survive_k`, built from k's own `sigma_t_k`) sits in the numerator, and
///    `path_pdf[k]` is separately updated with k's OWN technique's density
///    (`sigma_t_k * tr_k` / `survive_k`) for the final `spectral_mis_weight` combination
///    -- the same two-quantity split (throughput correction vs. MIS bookkeeping) every
///    other bounce dispatch in this module tree already makes.
///
///    For `k == hero_idx` both branches collapse to a value-independent identity: the
///    scatter branch's `tr_hero * sigma_s / pdf_hero` reduces algebraically to
///    `sigma_s / sigma_t_hero` (the single-scattering albedo, the textbook analog-sampler
///    result), and the no-scatter branch's `survive_hero / survive_hero` is exactly
///    `1.0` -- the well-known property of unbounded free-path sampling that makes the
///    "reached the boundary" case carry NO extra dimming of its own: the medium's
///    extinction is entirely accounted for by the STATISTICS of how often a path
///    scatters away before reaching the boundary, not by a per-sample multiplicative
///    factor. This is a genuinely different estimator SHAPE than `apply_absorption`'s
///    deterministic `exp(-alpha*path_len)` -- unbiased in expectation over many samples,
///    not identical sample-by-sample -- which is exactly why this function is gated
///    behind `scattering_sigma_s > 0.0` and never runs when it is `0.0` (see this
///    function's caller in `trace_spectral_ray_inner`): the two estimators are NOT
///    required to agree bit-for-bit, only in the limit `sigma_s -> 0`, and the
///    default-off bit-identity guarantee instead comes from `apply_absorption`'s
///    original code path being taken completely unconditionally rather than "reduced to"
///    from this one.
/// 3. **HG phase/pdf cancellation.** [`sample_henyey_greenstein_direction`]
///    importance-samples EXACTLY the Henyey-Greenstein phase function's own normalized
///    distribution (see that function's doc comment), so `phase(cos_theta) /
///    pdf_direction(cos_theta) == 1.0` identically -- no separate direction-sampling
///    factor appears in the per-channel loop below, mirroring
///    `cosine_weighted_hemisphere`'s identical cancellation for the frosted BSDF.
/// 4. **Achromatic direction and distance.** Exactly ONE `t_free` and ONE scattered
///    direction are sampled (from the hero channel's own `sigma_t_hero` and the
///    material's single achromatic `scattering_g`) and reused for every channel -- never
///    a per-channel resample. Since `scattering_g` does not vary by channel, every
///    channel's own hypothetical phase/pdf ratio is ALSO exactly `1.0` (not just the
///    hero's), so -- unlike the polished refraction path's `direction_matches` guard --
///    there is no chromatic-termination check to apply here at all: a companion channel
///    never has its Stokes/`path_pdf` dropped to zero for a direction mismatch, because
///    there is only ever one direction to match against.
/// 5. **New, decorrelated RNG streams** ([`DISTANCE_SAMPLE_STREAM`],
///    [`PHASE_DIR_U_STREAM`], [`PHASE_DIR_V_STREAM`]), salted in the exact same
///    `hash_u32(rng_seed ^ hash_u32(bounce ^ STREAM))` style as every other per-bounce
///    draw in this module tree, mirrored bit-for-bit in `shaders/transport_physics.wgsl`.
///
/// # Depolarization
///
/// Like [`apply_frosted_bounce`], every channel's Stokes vector collapses to
/// `StokesVector::unpolarized` at a scatter event: multiple incoherent scattering off an
/// inclusion's internal structure scrambles the coherent phase relationship polarization
/// depends on, so there is no meaningful Mueller-matrix structure left to carry forward.
///
/// # Return value
///
/// `Some((t_free, new_dir))` if a scatter event fired strictly before `hit_t` (the
/// caller must redirect `current_ray` from `t_free` along its OLD direction, not advance
/// all the way to the facet); `None` if the path survived to the facet boundary
/// (`stokes`/`path_pdf` already carry that survival's extinction weight -- the caller
/// proceeds to the facet dispatch as usual, WITHOUT also calling [`apply_absorption`]
/// for this same segment).
///
/// `pub(crate)`, not private: `renderer::gpu::transport_check`'s Tier 2 ULP check
/// (`run_scatter_or_extinguish`) calls this REAL function directly (never a
/// reimplementation), comparing against `shaders/transport_physics.wgsl`'s bit-for-bit
/// WGSL translation, which is also what the shipped megakernel
/// (`shaders/spectral_transport.wgsl`) calls for a scattering-active material.
/// Visibility only -- no numerical change.
#[expect(
    clippy::too_many_arguments,
    reason = "argument list mirrors transport_physics.wgsl's own \
              maybe_scatter_or_extinguish (alphas, sigma_s, g, ray_dir, hit_t, \
              path_scale, rng_seed, bounce, stokes, path_pdf) one-for-one -- this Rust \
              function IS the reference `renderer::gpu::transport_check`'s Tier 2 ULP \
              check compares that WGSL translation against, so bundling its parameters \
              into a struct here would break the direct correspondence a reviewer needs \
              when checking the two side by side"
)]
pub(crate) fn maybe_scatter_or_extinguish(
    alphas: &[f32; NUM_CHANNELS],
    sigma_s: f32,
    g: f32,
    hero_idx: usize,
    ray_dir: Vec3,
    hit_t: f32,
    path_scale: f32,
    rng_seed: u32,
    bounce: u32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
) -> Option<(f32, Vec3)> {
    let sigma_t_hero = alphas[hero_idx] + sigma_s;

    let dist_rand =
        (hash_u32(rng_seed ^ hash_u32(bounce ^ DISTANCE_SAMPLE_STREAM)) as f32) / 4_294_967_295.0;
    // Unbounded exponential free-path sample (the standard homogeneous-medium analog
    // sampler -- see this function's doc comment, hazard 2). `one_minus_u` is reused
    // directly as `pdf_hero` below in the scatter branch (`exp(-sigma_t_hero * t_free)
    // == one_minus_u` algebraically, by construction of `t_free` from it), rather than
    // recomputing the same exponential a second time. `t_free` comes out in
    // ABSORPTION-LENGTH units here (the same units `alphas`/`sigma_s` are defined in),
    // since `sigma_t_hero` is an absorption-length-unit rate -- see below for the
    // model-unit conversions this implies.
    let one_minus_u = (1.0 - dist_rand).max(1e-7);
    let t_free = -(one_minus_u.ln()) / sigma_t_hero;

    // P1 (absorption path scale): `hit_t` arrives in MODEL units (the caller's ray
    // parameter); every comparison/weight below needs it in the same absorption-length
    // units `t_free`/`alphas`/`sigma_s` are already in -- see
    // `GemMaterial::absorption_path_scale`'s doc comment. `path_scale == 1.0` (every
    // built-in) makes this multiply an exact IEEE 754 no-op, so `hit_t_scaled` is
    // bit-identical to the pre-P1 `hit_t` for every existing material/scene, and every
    // computation below stays byte-for-byte unchanged too.
    let hit_t_scaled = hit_t * path_scale;

    if t_free < hit_t_scaled {
        let pdf_hero = sigma_t_hero * one_minus_u;
        // Vectorized Beer-Lambert transmittance; exp_f32x8 is a few-ULP polynomial
        // exponential, NOT bit-identical to f32::exp -- see apply_absorption's comment
        // and src/simd.rs module docs.
        let mut tr_args = [0f32; NUM_CHANNELS];
        for k in 0..NUM_CHANNELS {
            tr_args[k] = -(alphas[k] + sigma_s) * t_free;
        }
        let tr = crate::simd::exp_f32x8(tr_args);
        for k in 0..NUM_CHANNELS {
            let sigma_t_k = alphas[k] + sigma_s;
            let tr_k = tr[k];
            let weight = tr_k * sigma_s / pdf_hero;
            stokes[k] = StokesVector::unpolarized(stokes[k].intensity() * weight);
            path_pdf[k] *= sigma_t_k * tr_k;
        }
        let u1 =
            (hash_u32(rng_seed ^ hash_u32(bounce ^ PHASE_DIR_U_STREAM)) as f32) / 4_294_967_295.0;
        let u2 =
            (hash_u32(rng_seed ^ hash_u32(bounce ^ PHASE_DIR_V_STREAM)) as f32) / 4_294_967_295.0;
        let new_dir = sample_henyey_greenstein_direction(u1, u2, g, ray_dir);
        // Convert the sampled free-path distance back to MODEL units before the caller
        // advances `current_ray.origin` along `ray_dir` by it -- `ray_dir` is a unit
        // vector in MODEL space, so a length used to advance along it must be in model
        // units too. `path_scale == 1.0` makes this division an exact IEEE 754 no-op
        // (`t_free / 1.0 == t_free` bit-for-bit), preserving this function's own
        // unbiasedness argument (this is a unit conversion of an already-correctly-
        // distributed sample, not a resampling): the free-path distance was drawn from
        // the true absorption-length-unit exponential distribution above, and dividing
        // by the same constant `path_scale` used to build `hit_t_scaled` recovers the
        // model-unit distance to that exact same physical point -- the estimator's
        // weights (`weight`, `path_pdf[k]`) are entirely computed in absorption-length
        // units above and untouched by this conversion.
        let t_free_model = t_free / path_scale;
        Some((t_free_model, new_dir))
    } else {
        // Vectorized Beer-Lambert transmittance (see the scatter branch above and
        // apply_absorption's comment). Hero consistency: survive_hero is read from this
        // same batched result at hero_idx rather than its own separate scalar .exp()
        // call, so the hero lane's ratio always comes from the same exponential its
        // companions used.
        let mut survive_args = [0f32; NUM_CHANNELS];
        for k in 0..NUM_CHANNELS {
            survive_args[k] = -(alphas[k] + sigma_s) * hit_t_scaled;
        }
        let survive = crate::simd::exp_f32x8(survive_args);
        let survive_hero = survive[hero_idx];
        for k in 0..NUM_CHANNELS {
            let survive_k = survive[k];
            stokes[k] = stokes[k].scale(survive_k / survive_hero.max(1e-30));
            path_pdf[k] *= survive_k;
        }
        None
    }
}

/// What [`try_scatter_step`] found, and what `trace_spectral_ray_inner`'s bounce loop
/// should do about it.
pub(super) enum ScatterStepOutcome {
    /// `material.scattering_sigma_s <= 0.0` -- the block was skipped entirely (no RNG
    /// draw, no extinction applied). The caller falls through to the exact pre-Task-1
    /// facet-dispatch code, including its own `apply_absorption` call.
    NotApplicable,
    /// No scatter event fired: the path survived to the facet boundary, and
    /// `maybe_scatter_or_extinguish` already applied this segment's full extinction
    /// weight to `stokes`/`path_pdf`. The caller falls through to the facet-dispatch
    /// code, but must NOT also call `apply_absorption` for this same segment.
    ReachedBoundary,
    /// A scatter event fired and the path survived Russian roulette (or wasn't yet
    /// eligible for it). The caller should `continue` the bounce loop.
    ScatteredAndSurvived,
    /// A scatter event fired and Russian roulette terminated the path. The caller
    /// should `break` the bounce loop.
    ScatteredAndTerminated,
}

/// Attempts a Henyey-Greenstein scattering event on the segment
/// from `current_ray`'s current origin to the facet just hit (`hit_t` away), mutating
/// `current_ray`/`stokes`/`path_pdf` in place and returning what happened -- extracted
/// out of `trace_spectral_ray_inner`'s bounce loop purely to keep that already-long
/// function under clippy's line-count lint (this is a "direct extraction, same
/// floating-point operations in the same order" per this file's established precedent,
/// not a new code path).
///
/// Gated on `material.scattering_sigma_s > 0.0` -- `<= 0.0` (every built-in material's
/// own stored value, and every material not explicitly opted in via
/// `GemMaterial::with_scattering`) returns [`ScatterStepOutcome::NotApplicable`]
/// immediately, before drawing from any new RNG stream: the default-off bit-identity
/// guarantee comes from this branch never running, not from the new estimator "reducing
/// to" the old one at `sigma_s == 0`. See [`maybe_scatter_or_extinguish`]'s doc comment
/// for the estimator itself (hazards 1-5).
///
/// Applies Russian roulette (via the SAME [`apply_russian_roulette`] the polished/frosted
/// facet-dispatch path already calls at the end of every bounce, same stream, same
/// `bounce > 4` gate) when a scatter event fires, since that event otherwise never
/// reaches the bounce loop's own trailing Russian-roulette call -- a scatter-continue
/// bounce still gets exactly one Russian-roulette opportunity per bounce, the same
/// invariant every other bounce kind maintains.
///
/// P2: `current_k` is the wave normal `k` (see `refraction`'s own design note) --
/// [`channel_absorption_alphas`]' eigen-polarizations are derived from it (rule 6: `k`,
/// not `S`), while [`maybe_scatter_or_extinguish`]'s own `ray_dir` argument (the
/// Henyey-Greenstein phase function's "forward" direction) stays `current_ray.dir`
/// (`S`, the energy-propagation direction a scattering event actually redirects) --
/// these are genuinely different roles, not a copy-paste of the same value. A fired
/// scatter event collapses `k == S` going forward (`*current_k = new_dir`, same as
/// `current_ray.dir`): a Henyey-Greenstein scattering event already depolarizes the
/// Stokes vector (see this function's own module doc comment), so there is no coherent
/// wave-normal-vs-Poynting distinction left to carry past it -- a documented
/// simplification, not something the P2 design note's rules cover (those are about the
/// polished TIR/reflect/refract dispatch specifically).
#[expect(
    clippy::too_many_arguments,
    reason = "already bundles the two established fixed-for-the-trace contexts \
              (mat_ctx, cache); the rest are this bounce's own scalar/state inputs \
              (material, prev_plane_normal), the RNG stream identity (rng_seed, bounce), \
              and the four separate pieces of per-ray state a scatter event can mutate \
              (current_ray, current_k, stokes, path_pdf) -- the same shape \
              dispatch_bounce's own reason explains in transport.rs, since this function \
              is that same bounce dispatch for the scattering branch specifically"
)]
pub(super) fn try_scatter_step(
    mat_ctx: &RayMaterialContext,
    cache: &RayWavelengthCache,
    material: &GemMaterial,
    prev_plane_normal: Option<Vec3>,
    current_ray: &mut Ray,
    current_k: &mut Vec3,
    hit_t: f32,
    rng_seed: u32,
    bounce: u32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
) -> ScatterStepOutcome {
    if material.scattering_sigma_s <= 0.0 {
        return ScatterStepOutcome::NotApplicable;
    }
    let s_axis = prev_plane_normal.unwrap_or(Vec3::ZERO);
    let alphas = channel_absorption_alphas(mat_ctx, cache, s_axis, *current_k, stokes);
    let Some((t_free, new_dir)) = maybe_scatter_or_extinguish(
        &alphas,
        material.scattering_sigma_s,
        material.scattering_g,
        mat_ctx.hero_idx,
        current_ray.dir,
        hit_t,
        material.absorption_path_scale,
        rng_seed,
        bounce,
        stokes,
        path_pdf,
    ) else {
        return ScatterStepOutcome::ReachedBoundary;
    };
    current_ray.origin += t_free * current_ray.dir;
    current_ray.dir = new_dir;
    *current_k = new_dir;
    if bounce > 4 && !apply_russian_roulette(bounce, rng_seed, stokes) {
        return ScatterStepOutcome::ScatteredAndTerminated;
    }
    ScatterStepOutcome::ScatteredAndSurvived
}

/// A stable orthonormal basis `(t, b)` perpendicular to unit vector `n` -- the same
/// branch-minimal construction `birefringence::stable_orthonormal_basis` uses, kept as
/// its own local copy (rather than exposing that private helper across modules) the same
/// way `transport_physics.wgsl` already keeps `arbitrary_perpendicular`/
/// `stable_orthonormal_basis_t` as distinct-but-identical WGSL constructions: each call
/// site mirrors the specific origin it was written for.
fn frosted_orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
    let t = (a - n * n.dot(a)).normalize_or_zero();
    let b = n.cross(t);
    (t, b)
}

/// Cosine-weighted hemisphere direction about `n`, from two independent uniform
/// `[0, 1)` randoms (Malley's method -- polar mapping, not the concentric-disk variant,
/// so the WGSL port is a direct translation of the same handful of trig calls). See
/// `apply_frosted_bounce`'s doc comment for why its call sites need no explicit pdf
/// division.
///
/// GPU port: `pub(crate)`, not private -- `renderer::gpu::transport_check`'s Tier 2 ULP
/// check dispatches the SAME shared WGSL translation of this exact function (see
/// `shaders/transport_physics.wgsl`) and compares against calling this real CPU
/// function directly, never a reimplementation. Visibility only.
pub(crate) fn cosine_weighted_hemisphere(u1: f32, u2: f32, n: Vec3) -> Vec3 {
    let r = u1.sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    let (sin_t, cos_t) = theta.sin_cos();
    let (t, b) = frosted_orthonormal_basis(n);
    (t * (r * cos_t) + b * (r * sin_t) + n * (1.0 - u1).max(0.0).sqrt()).normalize_or_zero()
}

/// The Henyey-Greenstein phase
/// function, normalized to integrate to `1.0` over the full sphere (4*pi steradians).
///
/// `cos_theta` is the cosine of the angle between the SCATTERED direction and the ray's
/// ORIGINAL propagation direction (not the "look back toward the source" convention some
/// renderer texts use) -- so under this convention, `g > 0` means "phase mass
/// concentrated near `cos_theta = +1`", i.e. forward scattering, matching
/// [`crate::optics::materials::GemMaterial::scattering_g`]'s own doc comment exactly.
/// `g == 0.0` reduces to the isotropic constant `1 / (4*pi)` for every `cos_theta`.
///
/// Not called by [`maybe_scatter_or_extinguish`]'s own estimator (see that function's
/// doc comment, hazard 3: [`sample_henyey_greenstein_direction`] importance-samples this
/// exact distribution, so `phase / pdf` cancels to `1.0` and never needs to be evaluated
/// explicitly at a real scattering event) -- exposed as its own function purely so a
/// Tier 2 GPU self-test can pin `phase(cos_theta, g) / pdf(cos_theta, g)` at `1.0`
/// directly, which is the actual correctness claim hazard 3 makes.
#[must_use]
// Only ever evaluated by tests and by the GPU Tier 2 equivalence check. The
// transport path never calls it: `sample_henyey_greenstein_direction` samples this
// exact distribution, so `phase / pdf` cancels to a constant and the phase value is
// never needed at runtime. That cancellation is the point -- keep this as the
// reference the sampler is checked against, not as something to "wire up".
#[cfg(any(test, feature = "gpu"))]
pub(crate) fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = (2.0 * g).mul_add(-cos_theta, 1.0 + g2).max(1e-6).powf(1.5);
    (1.0 - g2) / (4.0 * std::f32::consts::PI * denom)
}

/// Importance-samples a direction from the Henyey-Greenstein
/// phase function about `forward` (the ray's current propagation direction), from two
/// independent uniform `[0, 1)` randoms.
///
/// # Derivation
///
/// The marginal density in `mu = cos_theta` (integrating the azimuthal angle out of
/// [`henyey_greenstein_phase`]'s solid-angle density) is `p(mu) = (1 - g^2) / (2 * (1 +
/// g^2 - 2*g*mu)^1.5)`. Inverting its CDF in closed form (standard Henyey-Greenstein
/// result, re-derived here under THIS function's own `cos_theta` sign convention -- see
/// [`henyey_greenstein_phase`]'s doc comment) gives, for `g != 0`:
///
/// `mu = (1 + g^2 - ((1 - g^2) / (1 - g + 2*g*u1))^2) / (2*g)`
///
/// and the isotropic special case `mu = 1 - 2*u1` for `g` close to `0` (both the exact
/// `g == 0` limit and, per [`GemMaterial::scattering_g`]'s doc comment, the practically
/// negligible range right around it -- the general formula is numerically unstable as
/// `g -> 0` since it divides by `g` twice).
///
/// The azimuthal angle `phi` is uniform (the phase function has no azimuthal
/// dependence), and the direction is built via the SAME "stable orthonormal basis about
/// an axis, polar mapping" construction [`cosine_weighted_hemisphere`] uses for the
/// frosted BSDF -- reusing [`frosted_orthonormal_basis`] directly (that function's own
/// name predates this caller; it is a generic "stable basis about a unit vector"
/// construction, not girdle-finish-specific) so the WGSL port is a direct line-for-line
/// translation of the same handful of trig calls, mirroring how `spectral_transport.wgsl`
/// already shares that exact function between the frosted bounce and (after this task)
/// this sampler.
///
/// Because this samples EXACTLY [`henyey_greenstein_phase`]'s own normalized
/// distribution, `henyey_greenstein_phase(dot(result, forward), g) / pdf(result)` is
/// `1.0` for every `(u1, u2, g)` -- see [`maybe_scatter_or_extinguish`]'s doc comment,
/// hazard 3, for why that cancellation is what lets a scattering event's per-channel
/// weight omit a separate direction-sampling factor.
#[must_use]
pub(crate) fn sample_henyey_greenstein_direction(u1: f32, u2: f32, g: f32, forward: Vec3) -> Vec3 {
    let cos_theta = if g.abs() < 1e-3 {
        2.0f32.mul_add(-u1, 1.0)
    } else {
        let one_minus_g2 = g.mul_add(-g, 1.0);
        let denom = (2.0 * g).mul_add(u1, 1.0 - g);
        let sq = one_minus_g2 / denom;
        sq.mul_add(-sq, g.mul_add(g, 1.0)) / (2.0 * g)
    };
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    let sin_theta = f32::mul_add(cos_theta, -cos_theta, 1.0).max(0.0).sqrt();
    let phi = 2.0 * std::f32::consts::PI * u2;
    let (sin_p, cos_p) = phi.sin_cos();
    let (t, b) = frosted_orthonormal_basis(forward);
    (t * (sin_theta * cos_p) + b * (sin_theta * sin_p) + forward * cos_theta).normalize_or_zero()
}

/// The REPLACEMENT for the specular
/// TIR/partial-reflect/refract dispatch at a `FacetFinish::Frosted` facet.
///
/// # The model, honestly
///
/// A real bruted/ground surface's reflectance does not carry the polished interface's
/// sharp per-wavelength Fresnel structure -- roughness at a scale comparable to the
/// facet averages over a spread of local micro-facet angles, washing out the smooth
/// formula's fine structure -- so this uses ONE broadband reflect/transmit fraction
/// (`r_unpol`, from the HERO channel's angle and index only) for EVERY spectral channel,
/// unlike the polished path's per-channel `r_unpol_k`. That is the one deliberate
/// simplification this function makes beyond "diffuse instead of specular": every
/// channel shares not just the sampled direction (physically correct -- roughness
/// scattering is not meaningfully dispersive) but also the reflect/transmit split
/// probability itself (a modeling simplification, not a measured property of any real
/// bruted surface).
///
/// Direction is drawn from the cosine-weighted hemisphere about the correct macroscopic
/// normal (`normal` for reflect, `-normal` for transmit) -- the textbook importance-
/// sampling technique for a Lambertian BRDF/BTDF (`f = albedo / PI`, `pdf =
/// cos(theta) / PI`), chosen so `f * cos(theta) / pdf` is the constant `albedo` (taken as
/// `1.0`: the interface's total reflectance/transmittance is already fully spent on
/// `r_unpol` / `1 - r_unpol` below) -- which is why no separate pdf division appears in
/// the per-channel loop below: it is already folded into the `1.0 / r_unpol` (or
/// `1.0 / t_unpol`) throughput scale by that exact cancellation, not omitted.
///
/// Depolarizes every channel (`StokesVector::unpolarized`): a ground surface's multiple
/// internal micro-scattering events scramble the coherent phase relationship
/// polarization depends on, so there is no meaningful Mueller-matrix structure left to
/// carry forward.
///
/// # Estimator composition -- why this needs no chromatic termination
///
/// The polished path drops a companion channel's contribution to EXACTLY zero when that
/// channel's own specular direction fails to match the hero-driven direction
/// (`apply_refract_channel`'s `direction_matches` check) -- necessary there because a
/// delta BSDF has zero density everywhere except the one direction Snell's law picks,
/// and that direction is wavelength-dependent. Neither premise holds here: the direction
/// drawn above is the SAME for every channel (drawn once, not per-channel) and the BSDF
/// is a smooth, finite-support hemisphere, not a delta -- so every channel's own
/// hypothetical technique assigns manifestly positive density to the realized direction,
/// with no measure-zero mismatch to guard against. That is what makes a frosted bounce
/// composable with the existing per-channel `path_pdf` bookkeeping and the final
/// `spectral_mis_weight` combination without restructuring either: `path_pdf[k] *=
/// r_unpol` (or `t_unpol`) uses the SAME factor for every channel, exactly as a plain
/// specular reflect off a non-dispersive material already does -- an event that cannot
/// discriminate between hero-wavelength techniques is not a new category this framework
/// has to learn, it is the SAME category the isotropic case already exercises.
///
/// Reuses `FRESNEL_BRANCH_STREAM` for the reflect-vs-transmit branch decision (same
/// stochastic decision, same stream, a different consequence once taken) -- that split
/// DOES divide by its own selection probability (`r_unpol` / `t_unpol` above), because
/// reflect and transmit partition disjoint, non-overlapping energy. `BIREFRINGENT_SPLIT_STREAM`
/// on an air->crystal entry into an anisotropic material (unpolarized light entering a
/// birefringent crystal still couples into both eigenmodes regardless of whether the
/// entry facet is smooth or rough) is a DIFFERENT shape: it picks which ~0.5 energy
/// SHARE this path becomes, not a 1-of-N disjoint alternative, so it is NOT divided by
/// its own 0.5 selection probability -- exactly the same energy-share reasoning
/// `apply_refract_bounce`'s doc comment in `refraction.rs` gives for the polished path's
/// identical entry split. Internal mode re-coupling is applied by the CALLER, via the
/// same `maybe_apply_internal_mode_coupling` a polished internal reflection uses.
// GPU port (frosted girdle finish): `pub(crate)`, not private -- see this function's
// own physics doc comment above. `renderer::gpu::transport_check`'s Tier 2 ULP check
// (`run_frosted_bounce`) calls this REAL function directly (never a reimplementation),
// comparing against `shaders/transport_physics.wgsl`'s bit-for-bit WGSL translation,
// which is also what the shipped megakernel (`shaders/spectral_transport.wgsl`) calls
// for a `FacetFinish::Frosted` facet. Visibility only -- no numerical change.
#[expect(
    clippy::too_many_arguments,
    reason = "already bundles ctx/geo, the same established contexts \
              apply_partial_fresnel_bounce and apply_refract_bounce use in refraction.rs; \
              the rest matches transport_physics.wgsl's own apply_frosted_bounce \
              parameter-for-parameter (normal, inside_gem, is_extraordinary, rng_seed, \
              bounce, stokes, path_pdf against geo's flattened is_anisotropic/sin2_t/n1/ \
              n2/cos_i) -- see this function's own doc comment on the Tier 2 ULP check \
              that compares this REAL function against that WGSL translation"
)]
pub(crate) fn apply_frosted_bounce(
    ctx: &RayMaterialContext,
    geo: &BounceRefractionGeometry,
    normal: Vec3,
    inside_gem: bool,
    is_extraordinary: bool,
    rng_seed: u32,
    bounce: u32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
) -> (Vec3, bool, Option<bool>) {
    let u1 =
        (hash_u32(rng_seed ^ hash_u32(bounce ^ FROSTED_DIR_U_STREAM)) as f32) / 4_294_967_295.0;
    let u2 =
        (hash_u32(rng_seed ^ hash_u32(bounce ^ FROSTED_DIR_V_STREAM)) as f32) / 4_294_967_295.0;

    if geo.sin2_t > 1.0 {
        // Forced reflect (TIR), probability 1 -- no draw, no pdf division, mirroring
        // `apply_tir_bounce`'s identical reasoning for the polished path.
        let new_dir = cosine_weighted_hemisphere(u1, u2, normal);
        for s in &mut *stokes {
            *s = StokesVector::unpolarized(s.intensity());
        }
        return (new_dir, inside_gem, None);
    }

    let cos_t = (1.0 - geo.sin2_t).max(0.0).sqrt();
    let r_s = f32::mul_add(geo.n2, -cos_t, geo.n1 * geo.cos_i)
        / f32::mul_add(geo.n2, cos_t, geo.n1 * geo.cos_i);
    let r_p = f32::mul_add(geo.n1, -cos_t, geo.n2 * geo.cos_i)
        / f32::mul_add(geo.n1, cos_t, geo.n2 * geo.cos_i);
    let r_unpol = (0.5 * r_p.mul_add(r_p, r_s * r_s)).clamp(1e-4, 1.0 - 1e-4);
    let rng_bounce =
        (hash_u32(rng_seed ^ hash_u32(bounce ^ FRESNEL_BRANCH_STREAM)) as f32) / 4_294_967_295.0;

    if rng_bounce < r_unpol {
        let new_dir = cosine_weighted_hemisphere(u1, u2, normal);
        for k in 0..NUM_CHANNELS {
            stokes[k] = StokesVector::unpolarized(stokes[k].intensity() / r_unpol);
            path_pdf[k] *= r_unpol;
        }
        (new_dir, inside_gem, None)
    } else {
        let new_dir = cosine_weighted_hemisphere(u1, u2, -normal);
        let entering_anisotropic = !inside_gem && ctx.is_anisotropic;
        // Mode SELECTION is still a stochastic 50/50 draw (see this function's doc
        // comment) -- only the throughput weighting that used to accompany it (a
        // `split_pdf` divisor/multiplier) is gone, since it estimated twice the
        // transmitted energy no interface can deliver -- see `apply_refract_bounce`'s
        // doc comment in `refraction.rs` for the full energy-share reasoning.
        let use_extraordinary = if entering_anisotropic {
            let split_rand = (hash_u32(rng_seed ^ hash_u32(bounce ^ BIREFRINGENT_SPLIT_STREAM))
                as f32)
                / 4_294_967_295.0;
            split_rand < 0.5
        } else {
            is_extraordinary
        };
        let t_unpol = 1.0 - r_unpol;
        for k in 0..NUM_CHANNELS {
            // No `/ split_pdf` here (see `apply_refract_channel`'s doc comment in
            // `refraction.rs`): the diffuse-transmitted intensity above is already the
            // full transmitted intensity for the SELECTED mode's own (broadband,
            // hero-driven) index, and that mode carries only its ~0.5 energy share of the
            // incident light -- dividing by the 0.5 selection probability on top of that
            // would estimate twice the energy the interface actually transmits.
            stokes[k] = StokesVector::unpolarized(stokes[k].intensity() / t_unpol);
            // No `* split_pdf` here either -- `path_pdf`'s role in `spectral_mis_weight`
            // is scale-invariant under multiplying every channel's `path_pdf` by the SAME
            // uniform factor (0.5 here, identical for every k), so it was a pure no-op on
            // the actual MIS weight even before removal, just one that risked underflow
            // for no benefit.
            path_pdf[k] *= t_unpol;
        }
        (
            new_dir,
            !inside_gem,
            entering_anisotropic.then_some(use_extraordinary),
        )
    }
}

/// Tests for
/// [`henyey_greenstein_phase`], [`sample_henyey_greenstein_direction`],
/// [`maybe_scatter_or_extinguish`], and the `GemMaterial::scattering_sigma_s` gate in
/// `trace_spectral_ray_inner`'s bounce loop.
#[cfg(test)]
mod scattering_tests {
    use super::{
        super::{
            camera::Camera,
            color::cie_1931_cmf,
            environment::{EnvironmentSource, LightingPreset},
            transport::trace_spectral_ray,
        },
        *,
    };
    use crate::{
        geometry::cuts::StandardGemCuts,
        renderer::env_map::{EnvironmentMap, rgb_to_spectral_radiance},
    };

    /// [`henyey_greenstein_phase`] must integrate to `1.0` over the full sphere for any
    /// `g` -- a basic normalization sanity check, via dense quadrature over `cos_theta`
    /// (the phase function has no azimuthal dependence, so the solid-angle integral
    /// reduces to `2*pi * integral_{-1}^{1} phase(mu) dmu`).
    #[test]
    fn hg_phase_integrates_to_one_over_the_sphere() {
        for g in [-0.8f32, -0.3, 0.0, 0.3, 0.7, 0.95] {
            let steps = 200_000u32;
            let dmu = 2.0 / f64::from(steps);
            let mut integral = 0.0f64;
            for i in 0..steps {
                let mu = f64::mul_add(f64::from(i) + 0.5, dmu, -1.0);
                integral = f64::mul_add(
                    f64::from(henyey_greenstein_phase(mu as f32, g)),
                    dmu,
                    integral,
                );
            }
            integral *= 2.0 * std::f64::consts::PI;
            assert!(
                (integral - 1.0).abs() < 1e-3,
                "HG phase function should integrate to 1.0 over the sphere for g={g}, got {integral}"
            );
        }
    }

    /// `henyey_greenstein_phase(1.0, g)` (exact forward direction) must strictly
    /// increase with `g` for `g > 0` -- pins down this file's sign convention
    /// ("positive g forward-scatters", per both `henyey_greenstein_phase`'s and
    /// `GemMaterial::scattering_g`'s doc comments) against the phase function itself,
    /// independent of the sampler.
    #[test]
    fn hg_phase_is_more_forward_peaked_for_larger_positive_g() {
        let p0 = henyey_greenstein_phase(1.0, 0.0);
        let p5 = henyey_greenstein_phase(1.0, 0.5);
        let p9 = henyey_greenstein_phase(1.0, 0.9);
        assert!(
            p0 < p5 && p5 < p9,
            "forward-direction phase value should strictly increase with g: p(g=0)={p0}, \
             p(g=0.5)={p5}, p(g=0.9)={p9}"
        );
    }

    /// The decisive check for hazard 3 ("HG phase function sampling must cancel
    /// correctly"): [`sample_henyey_greenstein_direction`]'s mean `cos_theta` (angle
    /// between the sampled direction and `forward`) must converge to `g` exactly --
    /// the well-known closed-form mean of the Henyey-Greenstein distribution. This is a
    /// property of the SAMPLER matching the PHASE FUNCTION'S OWN distribution (not
    /// merely "produces some distribution"); a sampler with a sign error, a swapped
    /// `u1`/`u2`, or a wrong CDF inversion would converge to the wrong mean (or the
    /// right mean with the wrong sign) and be caught here.
    #[test]
    fn hg_sampling_mean_cos_theta_converges_to_g() {
        let forward = Vec3::new(0.2, 0.6, 0.77).normalize();
        for g in [-0.7f32, -0.2, 0.0, 0.4, 0.85] {
            let trials = 400_000u32;
            let mut sum = 0.0f64;
            for i in 0..trials {
                let u1 = (hash_u32(i ^ 0x1111_1111) as f32) / 4_294_967_295.0;
                let u2 = (hash_u32(i ^ 0x2222_2222) as f32) / 4_294_967_295.0;
                let dir = sample_henyey_greenstein_direction(u1, u2, g, forward);
                sum += f64::from(dir.dot(forward));
            }
            let mean = (sum / f64::from(trials)) as f32;
            assert!(
                (mean - g).abs() < 0.01,
                "mean cos_theta for g={g} should converge to g itself, got {mean} over {trials} trials"
            );
        }
    }

    /// Sampled directions must stay unit-length and finite across the full `g` range,
    /// including right at the isotropic branch threshold this function switches
    /// formulas at.
    #[test]
    fn hg_sampling_always_produces_finite_unit_directions() {
        let forward = Vec3::new(-0.3, 0.1, 0.94).normalize();
        for g in [-0.999f32, -0.5, -1e-3, 0.0, 1e-3, 0.5, 0.999] {
            for i in 0..2000u32 {
                let u1 = (hash_u32(i ^ 0x3333_3333) as f32) / 4_294_967_295.0;
                let u2 = (hash_u32(i ^ 0x4444_4444) as f32) / 4_294_967_295.0;
                let dir = sample_henyey_greenstein_direction(u1, u2, g, forward);
                assert!(
                    dir.is_finite(),
                    "non-finite direction for g={g}, i={i}: {dir:?}"
                );
                assert!(
                    (dir.length() - 1.0).abs() < 1e-4,
                    "non-unit direction for g={g}, i={i}: {dir:?} (len={})",
                    dir.length()
                );
            }
        }
    }

    fn round_brilliant_colourless_scattering_material(sigma_s: f32, g: f32) -> GemMaterial {
        // Colourless (zero absorption bands -> sigma_a == 0 everywhere), non-dispersive,
        // cubic -- isolates the scattering estimator from the pre-existing chromatic
        // absorption/dispersion machinery, mirroring
        // `renderer::gpu::estimator_check::furnace_material`'s construction.
        GemMaterial::new_custom("scattering furnace probe", 1.5, 0.0, 0.0, [0.0, 0.0, 0.0])
            .with_scattering(sigma_s, g)
    }

    /// Default-off bit-identity (non-negotiable regression guard, the same one
    /// `frosted_finish_all_polished_is_bit_identical_to_trace_spectral_ray` establishes
    ///): a material with `scattering_sigma_s == 0.0` reached via
    /// `GemMaterial`'s plain default (every built-in, `new_custom`) must trace BIT
    /// IDENTICALLY to the same material with scattering explicitly set to `(0.0, g)` for
    /// several `g` -- proving the `scattering_sigma_s > 0.0` gate in
    /// `trace_spectral_ray_inner`, not merely "the estimator happens to reduce to the
    /// same value at `sigma_s=0`", is what disables the new code path (that reduction is
    /// NOT guaranteed algebraically -- see `maybe_scatter_or_extinguish`'s doc comment,
    /// hazard 2 -- so this test is the only thing standing between a future refactor and
    /// a silently biased default).
    #[test]
    fn default_off_scattering_is_bit_identical_regardless_of_g() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let material_default = GemMaterial::ruby();
        assert!(
            material_default.scattering_sigma_s <= 0.0,
            "test premise: Ruby's own built-in scattering_sigma_s must be 0.0"
        );
        let camera = Camera::new(0.35, 0.28, 5.0, 18.0);
        let env = || LightingPreset::Daylight.studio(1.0, 0.4, 0.35);

        for g in [-0.6f32, 0.0, 0.4, 0.95] {
            let material_explicit_zero = material_default.clone().with_scattering(0.0, g);
            for iy in 0..6usize {
                for ix in 0..6usize {
                    let ray = camera.generate_ray(ix as f32, iy as f32, 6.0, 6.0, 0.5, 0.5);
                    for s in 0..4u32 {
                        let pixel_id = (iy as u32) * 6 + (ix as u32);
                        let seed = hash_u32(pixel_id ^ hash_u32(s ^ 0x1357_9BDF));
                        let hero_rand = (hash_u32(seed) as f32) / 4_294_967_295.0;
                        let a = trace_spectral_ray(
                            ray,
                            &planes,
                            &material_default,
                            10,
                            env(),
                            seed,
                            hero_rand,
                            None,
                        );
                        let b = trace_spectral_ray(
                            ray,
                            &planes,
                            &material_explicit_zero,
                            10,
                            env(),
                            seed,
                            hero_rand,
                            None,
                        );
                        assert_eq!(
                            a.to_array(),
                            b.to_array(),
                            "sigma_s=0.0 (default) vs sigma_s=0.0 (explicit, g={g}) must be \
                             BIT identical at pixel ({ix},{iy}) sample {s}"
                        );
                    }
                }
            }
        }
    }

    /// The decisive check (the physics review's own words: "the white furnace is
    /// the decisive check"): a LOSSLESS scattering medium (`sigma_a == 0`, `sigma_s >
    /// 0`) immersed in a spatially uniform environment must still render at exactly that
    /// environment's own radiance. Scattering redirects energy; it must not create or
    /// destroy it. Mirrors `tests/raytracer_tests.rs`'s
    /// `frosted_girdle_white_furnace_energy_conservation_still_holds` exactly, with a
    /// scattering-active colourless material (`sigma_a=0`, `sigma_s=1.2`, forward-biased
    /// g=0.4) instead of a frosted girdle.
    #[test]
    fn lossless_scattering_white_furnace_energy_conservation_holds() {
        const L0: f32 = 2.5;
        const SAMPLES_PER_PIXEL: u32 = 96;
        const GRID: usize = 12;
        const TOLERANCE: f32 = 0.08; // generous: CPU-only unit test sample budget

        let planes = StandardGemCuts::standard_round_brilliant();
        let material = round_brilliant_colourless_scattering_material(1.2, 0.4);
        let env_map = EnvironmentMap::uniform(1, 1, [L0, L0, L0]);

        let camera = Camera::new(0.35, 0.28, 5.0, 18.0);
        let mut sum = Vec3::ZERO;
        let mut count = 0u32;
        for iy in 0..GRID {
            for ix in 0..GRID {
                let ray =
                    camera.generate_ray(ix as f32, iy as f32, GRID as f32, GRID as f32, 0.5, 0.5);
                for s in 0..SAMPLES_PER_PIXEL {
                    let pixel_id = (iy as u32) * (GRID as u32) + (ix as u32);
                    let seed = hash_u32(pixel_id ^ hash_u32(s ^ 0x5CA1_AB1E));
                    sum += trace_spectral_ray(
                        ray,
                        &planes,
                        &material,
                        16,
                        EnvironmentSource::HdrMap(&env_map),
                        seed,
                        (hash_u32(seed) as f32) / 4_294_967_295.0,
                        None,
                    );
                    count += 1;
                }
            }
        }
        let mean = sum / count as f32;

        let mut target = Vec3::ZERO;
        for step in 0..=(780 - 380) {
            let lambda = 380.0f32 + step as f32;
            let spec = rgb_to_spectral_radiance([L0, L0, L0], lambda);
            target += cie_1931_cmf(lambda) * spec;
        }
        target /= 106.856;

        let rel_err = |v: f32, t: f32| (v - t).abs() / t.abs().max(1e-6);
        let (ex, ey, ez) = (
            rel_err(mean.x, target.x),
            rel_err(mean.y, target.y),
            rel_err(mean.z, target.z),
        );
        println!(
            "[lossless-scattering furnace] mean={mean:?} target={target:?} rel_err=({ex:.4}, {ey:.4}, {ez:.4}) over {count} samples"
        );
        assert!(
            ex <= TOLERANCE && ey <= TOLERANCE && ez <= TOLERANCE,
            "a lossless scattering medium should still converge to the uniform \
             environment's own radiance (mean={mean:?}, target={target:?}, \
             rel_err=({ex}, {ey}, {ez}), tolerance={TOLERANCE})"
        );
    }

    /// The scattering estimator must actually redistribute energy directionally, not
    /// just pass an energy-conservation check by coincidence: a strongly forward-biased
    /// scattering medium (`g` close to `1.0`) inside the gem should measurably change a
    /// transmissive scene's face-up appearance relative to `sigma_s = 0`, the same
    /// "decisive measurement" standard `frosted_girdle_changes_face_up_appearance_measurably`
    /// sets for girdle finish.
    #[test]
    fn scattering_measurably_changes_face_up_appearance() {
        const SAMPLES_PER_PIXEL: u32 = 96;
        const GRID: usize = 14;

        let planes = StandardGemCuts::standard_round_brilliant();
        let clear = GemMaterial::diamond();
        assert!(clear.scattering_sigma_s <= 0.0);
        let hazy = clear.clone().with_scattering(1.5, 0.3);
        let camera = Camera::new(0.35, 0.28, 5.0, 18.0);
        let env = || LightingPreset::RingLights.studio(1.0, 0.85, 0.95);

        let mut sum_clear = Vec3::ZERO;
        let mut sum_hazy = Vec3::ZERO;
        let mut count = 0u32;
        for iy in 0..GRID {
            for ix in 0..GRID {
                let ray =
                    camera.generate_ray(ix as f32, iy as f32, GRID as f32, GRID as f32, 0.5, 0.5);
                for s in 0..SAMPLES_PER_PIXEL {
                    let pixel_id = (iy as u32) * (GRID as u32) + (ix as u32);
                    let seed = hash_u32(pixel_id ^ hash_u32(s ^ 0x0BAD_F00D));
                    let hero_rand = (hash_u32(seed) as f32) / 4_294_967_295.0;
                    sum_clear +=
                        trace_spectral_ray(ray, &planes, &clear, 12, env(), seed, hero_rand, None);
                    sum_hazy +=
                        trace_spectral_ray(ray, &planes, &hazy, 12, env(), seed, hero_rand, None);
                    count += 1;
                }
            }
        }
        let mean_clear = sum_clear / count as f32;
        let mean_hazy = sum_hazy / count as f32;
        let delta_y = (mean_hazy.y - mean_clear.y).abs();
        let relative_change = delta_y / mean_clear.y.max(1e-6);
        println!(
            "[scattering face-up] clear Y={:.5} hazy Y={:.5} delta_y={:.5} ({:.2}%) over {count} samples",
            mean_clear.y,
            mean_hazy.y,
            delta_y,
            100.0 * relative_change
        );
        assert!(
            relative_change > 0.01,
            "a forward-scattering inclusion medium should measurably change face-up \
             brightness (>1%), not render identically to a clear stone -- got {:.4}% \
             (clear Y={:.5}, hazy Y={:.5})",
            100.0 * relative_change,
            mean_clear.y,
            mean_hazy.y
        );
    }

    /// [`maybe_scatter_or_extinguish`]'s hero channel must reproduce the textbook
    /// single-scattering-albedo/unity-survival identities exactly (hazard 2's doc
    /// comment derivation), checked directly rather than only inferred from the furnace
    /// test above.
    #[test]
    fn hero_channel_weight_matches_albedo_and_unity_survival_identities() {
        let material = round_brilliant_colourless_scattering_material(0.8, 0.0);
        let alphas = [0.0f32; NUM_CHANNELS]; // sigma_a == 0 for this material, every channel
        let sigma_t_hero = material.scattering_sigma_s; // sigma_a == 0 for this material
        let albedo = material.scattering_sigma_s / sigma_t_hero; // == 1.0 (lossless)

        let mut scatter_trials = 0u32;
        let mut survive_trials = 0u32;
        for seed in 0..4000u32 {
            let mut stokes = [StokesVector::unpolarized(1.0); NUM_CHANNELS];
            let mut path_pdf = [1.0f32; NUM_CHANNELS];
            let hit_t = 1.0f32; // fixed boundary distance
            let outcome = maybe_scatter_or_extinguish(
                &alphas,
                material.scattering_sigma_s,
                material.scattering_g,
                0,
                Vec3::Z,
                hit_t,
                1.0,
                seed,
                0,
                &mut stokes,
                &mut path_pdf,
            );
            if outcome.is_some() {
                scatter_trials += 1;
                assert!(
                    (stokes[0].intensity() - albedo).abs() < 1e-4,
                    "hero channel's scatter-branch weight should equal the single-scattering \
                     albedo sigma_s/sigma_t = {albedo} exactly (lossless), got {}",
                    stokes[0].intensity()
                );
            } else {
                survive_trials += 1;
                assert!(
                    (stokes[0].intensity() - 1.0).abs() < 1e-5,
                    "hero channel's no-scatter-branch weight should be exactly 1.0 \
                     (unity-survival identity), got {}",
                    stokes[0].intensity()
                );
            }
        }
        assert!(
            scatter_trials > 100 && survive_trials > 100,
            "test premise: both branches should fire many times at sigma_t*hit_t = {} \
             (scatter={scatter_trials}, survive={survive_trials})",
            sigma_t_hero * 1.0
        );
    }

    /// A COMPANION (non-hero) channel's weight, in BOTH branches, against an
    /// independently-derived closed form -- not merely re-checking the implementation
    /// against itself. Uses two channels with deliberately DIFFERENT chromatic
    /// absorption (Tourmaline's o-ray vs. e-ray at 550nm/430nm, the same pair
    /// `scatter_event_path_pdf_is_genuinely_per_channel_when_absorption_is_chromatic`
    /// uses) so the hero-vs-companion divergence this closes actually exercises the
    /// chromatic term, unlike [`hero_channel_weight_matches_albedo_and_unity_survival_identities`]'s
    /// colourless material (where `sigma_t_k == sigma_t_hero` for every k, so a bug
    /// that swaps the shared hero-pdf denominator for a per-channel one is invisible --
    /// this exact finding was measured as a negative control).
    ///
    /// Closed forms (independently re-derived, not copied from the implementation):
    /// - no-scatter branch: `weight_k = exp(-(sigma_t_k - sigma_t_hero) * hit_t)`
    /// - scatter branch: `weight_k = (sigma_s / sigma_t_hero) * exp(-(sigma_t_k -
    ///   sigma_t_hero) * t_free)`, recovered by re-deriving `t_free` from the SAME
    ///   `dist_rand` the function drew (`t_free = -ln(1 - dist_rand) / sigma_t_hero`)
    ///   using a fixed `rng_seed`/`bounce` so the test can predict the exact draw.
    #[test]
    fn companion_channel_weight_matches_independently_derived_closed_form() {
        let material = GemMaterial::by_name("Tourmaline")
            .expect("\"Tourmaline\" must be a built-in material")
            .with_scattering(0.8, 0.0);
        let lambda_hero = 550.0f32;
        let lambda_companion = 430.0f32; // strongly dichroic vs. 550nm, see module doc comment
        let mat_ctx = RayMaterialContext {
            material: &material,
            lambdas: [
                lambda_hero,
                lambda_companion,
                lambda_hero,
                lambda_hero,
                lambda_hero,
                lambda_hero,
                lambda_hero,
                lambda_hero,
            ],
            hero_idx: 0,
            c_axis: material.c_axis,
            is_anisotropic: true,
            enable_internal_mode_coupling: true,
        };
        let cache = super::super::refraction::build_ray_wavelength_cache(&mat_ctx);
        let ray_dir = Vec3::new(0.3, 0.9, 0.3).normalize();
        let sigma_s = material.scattering_sigma_s;

        // Independently compute this scenario's own alpha_hero/alpha_companion via the
        // real (shared, not reimplemented) `channel_absorption_alphas` -- using the SAME
        // building block the function under test uses is legitimate here since this
        // test's whole point is checking the DOWNSTREAM weight formula, not
        // re-deriving pleochroic absorption itself.
        let stokes_probe = [StokesVector::unpolarized(1.0); NUM_CHANNELS];
        let alphas =
            channel_absorption_alphas(&mat_ctx, &cache, Vec3::ZERO, ray_dir, &stokes_probe);
        assert!(
            (alphas[0] - alphas[1]).abs() > 1e-4,
            "test premise: hero (550nm) and companion (430nm) must have genuinely \
             different alpha (got {} vs {})",
            alphas[0],
            alphas[1]
        );
        let sigma_t_hero = alphas[0] + sigma_s;
        let sigma_t_companion = alphas[1] + sigma_s;

        let mut no_scatter_checked = false;
        let mut scatter_checked = false;
        for seed in 0..3000u32 {
            let hit_t = 0.9f32;
            let mut stokes = [StokesVector::unpolarized(1.0); NUM_CHANNELS];
            let mut path_pdf = [1.0f32; NUM_CHANNELS];
            let outcome = maybe_scatter_or_extinguish(
                &alphas,
                sigma_s,
                material.scattering_g,
                0,
                ray_dir,
                hit_t,
                1.0,
                seed,
                0,
                &mut stokes,
                &mut path_pdf,
            );
            let dist_rand =
                (hash_u32(seed ^ hash_u32(DISTANCE_SAMPLE_STREAM)) as f32) / 4_294_967_295.0;
            let one_minus_u = (1.0 - dist_rand).max(1e-7);
            let t_free = -(one_minus_u.ln()) / sigma_t_hero;

            if outcome.is_some() {
                if t_free >= hit_t {
                    continue; // outcome disagreed with our recomputed t_free; skip (guard only)
                }
                scatter_checked = true;
                let expected =
                    (sigma_s / sigma_t_hero) * (-(sigma_t_companion - sigma_t_hero) * t_free).exp();
                let actual = stokes[1].intensity();
                assert!(
                    (actual - expected).abs() < 1e-3 * expected.abs().max(1.0),
                    "scatter-branch companion weight mismatch at seed={seed}: expected \
                     {expected} (closed form), got {actual}"
                );
            } else {
                no_scatter_checked = true;
                let expected = (-(sigma_t_companion - sigma_t_hero) * hit_t).exp();
                let actual = stokes[1].intensity();
                assert!(
                    (actual - expected).abs() < 1e-3 * expected.abs().max(1.0),
                    "no-scatter-branch companion weight mismatch at seed={seed}: expected \
                     {expected} (closed form), got {actual}"
                );
            }
        }
        assert!(
            no_scatter_checked && scatter_checked,
            "test premise: both branches should be exercised and checked \
             (no_scatter_checked={no_scatter_checked}, scatter_checked={scatter_checked})"
        );
    }

    /// Negative control: if a companion channel's `path_pdf` update used the HERO's own
    /// technique density instead of its own (i.e. dropped the chromatic differentiation
    /// hazard 2 requires), `spectral_mis_weight` would silently collapse toward `N` for
    /// every ray instead of reflecting genuine per-channel agreement/disagreement. This
    /// test pins the CORRECT behaviour directly: two channels with different absorption
    /// (hence different `sigma_t_k`) must end up with DIFFERENT `path_pdf` values after
    /// a scatter event, proving the per-channel `sigma_t_k * tr_k` term is actually
    /// channel-dependent and not a hero broadcast.
    #[test]
    fn scatter_event_path_pdf_is_genuinely_per_channel_when_absorption_is_chromatic() {
        // Real built-in Tourmaline (strongly dichroic o-ray vs. e-ray absorption --
        // see `polarized_probe_matches_band_sums_for_real_tourmaline_data`, which
        // isolates that same chromatic difference), opted into scattering.
        let material = GemMaterial::by_name("Tourmaline")
            .expect("\"Tourmaline\" must be a built-in material")
            .with_scattering(0.8, 0.0);
        let mat_ctx = RayMaterialContext {
            material: &material,
            lambdas: [550.0, 430.0, 550.0, 550.0, 550.0, 550.0, 550.0, 550.0],
            hero_idx: 0,
            c_axis: material.c_axis,
            is_anisotropic: true,
            enable_internal_mode_coupling: true,
        };
        let cache = super::super::refraction::build_ray_wavelength_cache(&mat_ctx);
        let ray_dir = Vec3::new(0.3, 0.9, 0.3).normalize();
        let probe_stokes = [StokesVector::unpolarized(1.0); NUM_CHANNELS];
        let alphas =
            channel_absorption_alphas(&mat_ctx, &cache, Vec3::ZERO, ray_dir, &probe_stokes);
        let mut found_scatter = false;
        for seed in 0..2000u32 {
            let mut stokes = [StokesVector::unpolarized(1.0); NUM_CHANNELS];
            let mut path_pdf = [1.0f32; NUM_CHANNELS];
            let outcome = maybe_scatter_or_extinguish(
                &alphas,
                material.scattering_sigma_s,
                material.scattering_g,
                0,
                ray_dir,
                1.0,
                1.0,
                seed,
                0,
                &mut stokes,
                &mut path_pdf,
            );
            if outcome.is_some() {
                found_scatter = true;
                assert!(
                    (path_pdf[0] - path_pdf[1]).abs() > 1e-6,
                    "channel 0 (550nm) and channel 1 (430nm, strongly dichroic) must get \
                     DIFFERENT path_pdf values from a scatter event when the material's \
                     absorption is genuinely chromatic -- got path_pdf[0]={}, path_pdf[1]={}",
                    path_pdf[0],
                    path_pdf[1]
                );
            }
        }
        assert!(
            found_scatter,
            "test premise: at least one trial should hit the scatter branch"
        );
    }
}

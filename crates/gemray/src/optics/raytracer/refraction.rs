//! Fresnel reflection/transmission and Total Internal Reflection.
//!
//! The per-bounce refractive-index/incidence-angle geometry ([`RayMaterialContext`],
//! [`BounceRefractionGeometry`]) and the TIR/partial-reflect/refract bounce appliers
//! built on top of it.

use super::{
    NUM_CHANNELS,
    absorption::spectral_absorption,
    sampling::{BIREFRINGENT_SPLIT_STREAM, FRESNEL_BRANCH_STREAM, hash_u32},
};
use crate::optics::{
    birefringence::{AbsorptionTensor3, BiaxialIndicatrix, BirefringenceParams},
    materials::GemMaterial,
    polarization::{MuellerMatrix, StokesVector},
};
use glam::Vec3;

/// P2: the wave normal `k` and the Poynting (energy-propagation) direction `S` coincide
/// EXACTLY in an isotropic medium (air, or a cubic material) and for the uniaxial
/// ORDINARY eigenmode (no walk-off by definition) -- they diverge only for the
/// extraordinary/mode-B eigenmode of a genuinely anisotropic material while inside it,
/// by the walk-off angle (order 1-2 degrees for zircon-class birefringence). Everywhere
/// in this module tree that used to advance/refract/reflect using `ray_dir` as if it
/// were `k` while inside an anisotropic crystal was actually feeding it `S` (the walked-
/// off Poynting direction stored as `Ray::dir`/`current_k`'s sibling) -- Snell's law and
/// Fresnel phase matching apply to `k`, not `S`. This module now threads BOTH through the
/// bounce loop: `S` (== `Ray::dir` / `current_ray.dir` in `transport.rs`) for
/// intersection/advancement and path length, `k` (== `current_k` in `transport.rs`) for
/// every index lookup, `cos_i`/`sin_i`, Snell refraction, Fresnel coefficients, the TIR
/// decision and phase, and the Stokes plane-of-incidence frame -- see each function's own
/// doc comment below for its specific role. [`poynting_dir_for_mode`] is the one new
/// primitive this design needs: recovering `S` from a freshly-reflected/refracted `k`.
///
/// Total Internal Reflection phase retardation delta = `delta_p` - `delta_s` (Fresnel-Fresnel
/// rhomb style formula) for a wave whose channel index `n1k` puts it past its own
/// critical angle at this interface (`n1k * sin_i > 1`). Shared by the two sites in
/// `trace_spectral_ray` that need it: the branch where the HERO channel is itself past
/// critical angle (deterministic reflect), and the partial-reflection branch's
/// per-channel loop, where an individual channel k can be past ITS OWN critical angle
/// even though the hero (which drives the shared reflect/refract decision) isn't.
/// Previously only the first site applied this retardation; the second used
/// `MuellerMatrix::fresnel_reflection(1.0, 1.0)`, which has the right |r| = 1 magnitude
/// but silently drops the TIR phase shift, making TIR at that site polarization-inert.
#[inline]
pub(crate) fn tir_phase_delta(n1k: f32, cos_i: f32, sin_i: f32) -> f32 {
    let tan_half_delta_k = (cos_i * (n1k * n1k * sin_i).mul_add(sin_i, -1.0).max(0.0).sqrt())
        / (n1k * sin_i * sin_i).max(1e-6);
    2.0 * tan_half_delta_k.atan()
}

/// Per-ray context that stays fixed across every bounce of `trace_spectral_ray`'s main
/// loop: the material, the ray's 8 hero-wavelength comb, which slot drives the shared
/// geometric path, the optical c-axis, and whether this material is anisotropic at
/// all. Bundled into one struct purely to keep [`compute_bounce_refraction_geometry`]'s
/// argument count within clippy's `too_many_arguments` limit -- every field here is set
/// once before the bounce loop starts and never changes.
pub(crate) struct RayMaterialContext<'a> {
    pub(crate) material: &'a GemMaterial,
    pub(crate) lambdas: [f32; NUM_CHANNELS],
    pub(crate) hero_idx: usize,
    pub(crate) c_axis: Vec3,
    pub(crate) is_anisotropic: bool,
    /// Whether `maybe_apply_internal_mode_coupling` is active for this path.
    /// Bundled here (rather than a bare extra parameter threaded through every bounce
    /// call) since it is, like every other field here, fixed for the whole trace -- see
    /// `trace_spectral_ray_inner`'s `enable_internal_mode_coupling` parameter for what
    /// sets it.
    pub(crate) enable_internal_mode_coupling: bool,
}

/// R3: precomputed per-sample cache of quantities that depend only on the ray's fixed
/// 8-wavelength comb (`ctx.lambdas`) and the material -- both fixed for the whole trace,
/// same as every `RayMaterialContext` field -- but were previously recomputed from
/// scratch inside the bounce loop, once (sometimes several times) per bounce. Built once
/// via [`build_ray_wavelength_cache`], immediately after `RayMaterialContext` itself, and
/// read back every bounce instead.
///
/// Kept as its own struct rather than new fields on [`RayMaterialContext`]: that struct
/// is built via a bare struct literal (all fields, no `..Default::default()`) at several
/// `renderer::gpu::transport_check` Tier 2 GPU self-test call sites outside this task's
/// files, which must keep compiling unchanged -- adding required fields there would break
/// them, and `RayMaterialContext` cannot derive `Default` (its `&'a GemMaterial` field
/// isn't `Default`).
///
/// Every field here is produced by the exact SAME function, on the exact SAME input, that
/// the pre-hoist code called inline every bounce (see each field's doc comment for which
/// call it replaces) -- so a cached read-back is bit-identical to recomputing it: both are
/// the same deterministic pure function of `(material, lambdas)`, neither of which changes
/// across a trace.
pub(crate) struct RayWavelengthCache {
    /// `material.dispersion.evaluate(lambdas[k])` per channel -- was recomputed by
    /// `per_channel_uniaxial_indices` every bounce (only `n_eff_ch`, the other half of
    /// that function's return value, actually depends on the per-bounce `theta_c`; this
    /// half never did). See [`build_ray_wavelength_cache`] for why that function is still
    /// the one that computes this array (reused once per sample, not duplicated) and
    /// [`per_channel_effective_extraordinary_indices`] for the per-bounce `n_eff_ch` half
    /// that reads it back.
    pub(crate) n_o_ch: [f32; NUM_CHANNELS],
    /// `material.biaxial_indicatrix(lambdas[hero_idx])` -- was recomputed inline in both
    /// `hero_biaxial_wave_dirs` and `channel_absorption_alphas` every bounce. Always
    /// exactly `biaxial_ch[hero_idx]` (same call, same input); kept as its own field
    /// purely so call sites that only need the hero's own indicatrix don't have to index
    /// `biaxial_ch` themselves.
    pub(crate) hero_indicatrix: Option<BiaxialIndicatrix>,
    /// `material.biaxial_indicatrix(lambdas[k])` per channel -- was recomputed inline in
    /// `per_channel_biaxial_indices` and `apply_refract_channel` every bounce (up to
    /// ~17-25 calls per bounce for a biaxial material, each rebuilding the axis frame).
    pub(crate) biaxial_ch: [Option<BiaxialIndicatrix>; NUM_CHANNELS],
    /// Per-channel [`AbsorptionTensor3`] (uniaxial or biaxial, matching whether
    /// `hero_indicatrix.is_some()`), built from this channel's own `alpha_o`/`alpha_e`/
    /// `alpha_beta` (each `spectral_absorption` at `lambdas[k]`) and the material's fixed
    /// `c_axis`. `channel_absorption_alphas`'s per-bounce, per-channel call into
    /// `birefringence::pleochroic_channel_alpha` used to rebuild exactly this tensor (a
    /// fresh `stable_orthonormal_basis` + `Mat3`) from scratch every time, even though
    /// none of its inputs vary by bounce. `channel_absorption_alphas` now calls
    /// `birefringence::effective_pleochroic_alpha` directly against this cached tensor
    /// instead -- the same downstream function `pleochroic_channel_alpha` itself calls,
    /// just fed a cached tensor rather than a freshly-built one (this avoids restructuring
    /// `birefringence.rs`'s own `pleochroic_channel_alpha`/`AbsorptionTensor3`, which is
    /// not among this task's owned files).
    pub(crate) tensor_ch: [AbsorptionTensor3; NUM_CHANNELS],
}

/// Builds [`RayWavelengthCache`] once per sample -- see that struct's doc comment for why
/// each field is bit-identical to what the pre-hoist code recomputed inline every bounce.
pub(super) fn build_ray_wavelength_cache(ctx: &RayMaterialContext) -> RayWavelengthCache {
    let material = ctx.material;

    // Same expression `per_channel_uniaxial_indices` already evaluates for its own
    // `n_o_ch` return value -- called here once per SAMPLE (not once per bounce, as the
    // pre-hoist bounce loop did) purely to reuse that exact call rather than duplicate its
    // formula. `n_o_ch` never depends on `theta_c` (only the discarded `n_eff_ch` half
    // does), so the `0.0` argument here is an arbitrary placeholder -- the real per-bounce
    // `n_eff_ch`, which DOES need each bounce's own `theta_c`, is computed fresh every
    // bounce by `per_channel_effective_extraordinary_indices` from this cached `n_o_ch`.
    let (n_o_ch, _n_eff_ch_unused_theta_c_independent_half) =
        per_channel_uniaxial_indices(ctx, 0.0);

    let biaxial_ch: [Option<BiaxialIndicatrix>; NUM_CHANNELS] =
        std::array::from_fn(|k| material.biaxial_indicatrix(ctx.lambdas[k]));
    let hero_indicatrix = biaxial_ch[ctx.hero_idx];
    // Same condition `channel_absorption_alphas` used to derive locally from its own
    // freshly-called `material.biaxial_indicatrix(lambdas[hero_idx])`.
    let is_biaxial = hero_indicatrix.is_some();

    let abs_o = &material.absorption.o_ray;
    let abs_e = &material.absorption.e_ray;
    let abs_beta = material.absorption.beta_ray.as_deref();
    let alpha_o_ch: [f32; NUM_CHANNELS] =
        std::array::from_fn(|k| spectral_absorption(abs_o, ctx.lambdas[k]));
    let alpha_e_ch: [f32; NUM_CHANNELS] =
        std::array::from_fn(|k| spectral_absorption(abs_e, ctx.lambdas[k]));
    let alpha_beta_ch: [Option<f32>; NUM_CHANNELS] = std::array::from_fn(|k| {
        if is_biaxial {
            abs_beta.map(|bands| spectral_absorption(bands, ctx.lambdas[k]))
        } else {
            None
        }
    });
    let tensor_ch: [AbsorptionTensor3; NUM_CHANNELS] = std::array::from_fn(|k| {
        alpha_beta_ch[k].map_or_else(
            || AbsorptionTensor3::uniaxial(alpha_o_ch[k], alpha_e_ch[k], ctx.c_axis),
            |beta| AbsorptionTensor3::biaxial(alpha_o_ch[k], beta, alpha_e_ch[k], ctx.c_axis),
        )
    });

    RayWavelengthCache {
        n_o_ch,
        hero_indicatrix,
        biaxial_ch,
        tensor_ch,
    }
}

/// Per-bounce refractive-index and incidence-angle quantities, computed once at the top
/// of each bounce iteration before the TIR / partial-reflect / refract branches decide
/// what to do with them.
///
/// GPU port (frosted girdle finish): `pub(crate)` (struct and every field) plus
/// `#[derive(Clone, Copy, Default)]` so `renderer::gpu::transport_check`'s Tier 2 ULP
/// check for `apply_frosted_bounce` can build one directly -- that function reads only
/// `cos_i`/`n1`/`n2`/`sin2_t` from this struct (see its own doc comment), so the test
/// harness constructs a `Default::default()` instance and overrides just those four
/// fields, rather than hand-deriving the full biaxial/per-channel machinery real bounce
/// dispatch would populate. `Default` is derivable regardless of `BiaxialIndicatrix`'s
/// own `Default`-ness since `Option<T>` always implements `Default` (`None`). Visibility
/// only -- no field, type, or computation here changed.
#[derive(Clone, Copy, Default)]
pub(crate) struct BounceRefractionGeometry {
    pub(crate) cos_i: f32,
    pub(crate) sin_i: f32,
    pub(crate) is_biaxial: bool,
    pub(crate) n_o_hero: f32,
    pub(crate) n_e_hero: f32,
    pub(crate) hero_indicatrix: Option<BiaxialIndicatrix>,
    pub(crate) n_biax_a_hero: f32,
    pub(crate) n_biax_b_hero: f32,
    pub(crate) n_o_ch: [f32; NUM_CHANNELS],
    pub(crate) n_biax_a_ch: [f32; NUM_CHANNELS],
    pub(crate) n1: f32,
    pub(crate) n2: f32,
    pub(crate) sin2_t: f32,
    pub(crate) n1_ch: [f32; NUM_CHANNELS],
    pub(crate) n2_ch: [f32; NUM_CHANNELS],
    pub(crate) sin2_t_ch: [f32; NUM_CHANNELS],
}

/// Builds [`BounceRefractionGeometry`] for the current bounce. A direct extraction of
/// `trace_spectral_ray`'s pre-extraction inline block: the exact same sequence of
/// floating-point operations, in the exact same order, just packaged as a function so
/// that function does not have to spell it out inline. Touches no accumulator (`stokes`,
/// `path_pdf`, `radiance`) and mutates no loop state -- a pure "compute from this
/// bounce's inputs and return" step, so extracting it cannot perturb the bit-exact
/// result the rest of the bounce loop goes on to compute.
///
/// `theta_c` (the angle against the c-axis used to evaluate the
/// direction-dependent extraordinary index) must be measured against the WAVE NORMAL
/// `k`, not the Poynting/energy direction `S` -- see this module's own "wave normal vs
/// Poynting direction" design note above. On an air->crystal entry (`!inside_gem`) this
/// is mildly circular: the refracted wave normal depends on `n2`, which (for the
/// extraordinary index) depends on `theta_c`, which depends on the refracted wave
/// normal. Resolved with 2 fixed-point iterations, seeded from the ordinary index `n_o`
/// (an isotropic first guess -- and exactly correct if the path turns out to be the
/// ordinary eigenmode); `k_hat == S` trivially outside the crystal (isotropic air), so
/// this branch is unaffected by the P2 k/S split either way. While already inside the
/// crystal, `k_hat` is the CALLER's own tracked wave normal (`current_k` in
/// `transport.rs`, carried and re-evaluated across bounces -- see
/// [`poynting_dir_for_mode`]), so the angle is read directly from it.
///
/// Only applies to the uniaxial ordinary/extraordinary approximation -- a biaxial
/// material never consults its result (see the biaxial per-channel/per-mode block in
/// [`compute_bounce_refraction_geometry`] for its own, separate iteration via
/// `BiaxialIndicatrix::resolve_entry_mode`); callers guard `is_biaxial` themselves.
pub(crate) fn theta_c_for_bounce(
    ctx: &RayMaterialContext,
    normal: Vec3,
    k_hat: Vec3,
    cos_i: f32,
    inside_gem: bool,
    is_biaxial: bool,
    n_o_hero_seed: f32,
) -> f32 {
    let material = ctx.material;
    let c_axis = ctx.c_axis;
    let is_anisotropic = ctx.is_anisotropic;

    if !inside_gem && is_anisotropic && !is_biaxial {
        let n_e_hero_seed = n_o_hero_seed + material.birefringence_delta;
        let mut n_guess = n_o_hero_seed;
        let mut theta = 0.0f32;
        for _ in 0..2 {
            let eta_guess = 1.0 / n_guess;
            let sin2_t_guess = eta_guess * eta_guess * cos_i.mul_add(-cos_i, 1.0);
            if sin2_t_guess > 1.0 {
                break;
            }
            let cos_t_guess = (1.0 - sin2_t_guess).max(0.0).sqrt();
            let wave_dir_guess =
                (eta_guess * k_hat + eta_guess.mul_add(cos_i, -cos_t_guess) * normal).normalize();
            let cos_theta_wave = wave_dir_guess.dot(c_axis).clamp(-1.0, 1.0).abs();
            theta = cos_theta_wave.acos();
            n_guess = BirefringenceParams::effective_extraordinary_index(
                n_o_hero_seed,
                n_e_hero_seed,
                theta,
            );
        }
        theta
    } else {
        k_hat.dot(c_axis).clamp(-1.0, 1.0).abs().acos()
    }
}

/// Per-channel uniaxial ordinary (`n_o_ch`) and effective-extraordinary (`n_eff_ch`)
/// indices, each channel evaluated at its OWN wavelength (Fix F) against the shared
/// `theta_c` (see [`theta_c_for_bounce`]).
pub(crate) fn per_channel_uniaxial_indices(
    ctx: &RayMaterialContext,
    theta_c: f32,
) -> ([f32; NUM_CHANNELS], [f32; NUM_CHANNELS]) {
    let material = ctx.material;
    let is_anisotropic = ctx.is_anisotropic;
    let mut n_o_ch = [0.0f32; NUM_CHANNELS];
    let mut n_eff_ch = [0.0f32; NUM_CHANNELS];
    for k in 0..NUM_CHANNELS {
        let n_o_k = material.dispersion.evaluate(ctx.lambdas[k]);
        let n_e_k = n_o_k + material.birefringence_delta;
        n_o_ch[k] = n_o_k;
        n_eff_ch[k] = if is_anisotropic {
            BirefringenceParams::effective_extraordinary_index(n_o_k, n_e_k, theta_c)
        } else {
            n_o_k
        };
    }
    (n_o_ch, n_eff_ch)
}

/// The per-bounce (`theta_c`-dependent) half of what `per_channel_uniaxial_indices` used
/// to compute from scratch every bounce, reading the theta_c-INDEPENDENT half
/// (`n_o_ch`) back from [`RayWavelengthCache`] instead of recomputing it. Per-element,
/// this is exactly `per_channel_uniaxial_indices`'s own `n_eff_ch[k]` formula, fed the
/// SAME `n_o_k` value (`cache.n_o_ch[k]`, bit-identical to a fresh
/// `material.dispersion.evaluate(lambdas[k])` call -- see that struct's doc comment) --
/// so this is bit-identical to the pre-hoist per-bounce recomputation.
fn per_channel_effective_extraordinary_indices(
    ctx: &RayMaterialContext,
    n_o_ch: &[f32; NUM_CHANNELS],
    theta_c: f32,
) -> [f32; NUM_CHANNELS] {
    let material = ctx.material;
    let is_anisotropic = ctx.is_anisotropic;
    let mut n_eff_ch = [0.0f32; NUM_CHANNELS];
    for k in 0..NUM_CHANNELS {
        let n_o_k = n_o_ch[k];
        let n_e_k = n_o_k + material.birefringence_delta;
        n_eff_ch[k] = if is_anisotropic {
            BirefringenceParams::effective_extraordinary_index(n_o_k, n_e_k, theta_c)
        } else {
            n_o_k
        };
    }
    n_eff_ch
}

/// Genuinely biaxial per-channel mode indices, and -- at an air->crystal
/// entry -- the per-mode wave-normal directions those indices were resolved at. Unlike
/// the uniaxial arrays, NEITHER eigenmode here has a direction-independent index: "mode
/// A" names the FASTER (lower-index) root of `wave_indices` and "mode B" the SLOWER
/// (higher-index) root at the relevant wave-normal direction -- an arbitrary but
/// self-consistent relabelling of `is_extraordinary`'s two slots, not a claim that
/// "mode B" is always what was "extraordinary" in the uniaxial arrays.
///
/// The wave-normal direction feeding each channel's index lookup is resolved ONCE from
/// the HERO channel's own indicatrix -- exactly mirroring `theta_c` (one shared
/// geometric direction reused for every channel; only the index MAGNITUDE varies per
/// channel). Entering the crystal this needs `resolve_entry_mode`'s fixed-point
/// iteration (the direction depends on the index, which depends on the direction) run
/// once per mode, since neither mode has a constant index to seed the OTHER from.
/// Already inside the crystal, `k_hat` is the CALLER's own tracked wave normal
/// (`current_k` in `transport.rs`), so both modes share it directly with no iteration --
/// P2: both biaxial modes walk off (see [`poynting_dir_for_mode`]'s doc comment), so
/// `k_hat` here is genuinely NOT the same as `Ray::dir`/`S` while inside the crystal in
/// either mode, unlike the uniaxial ordinary eigenmode.
fn hero_biaxial_wave_dirs(
    cache: &RayWavelengthCache,
    normal: Vec3,
    k_hat: Vec3,
    cos_i: f32,
    inside_gem: bool,
    is_biaxial: bool,
    n_o_hero_seed: f32,
) -> (Option<BiaxialIndicatrix>, Vec3, Vec3) {
    // Bit-identical to a fresh `ctx.material.biaxial_indicatrix(ctx.lambdas[ctx.hero_idx])`
    // call -- see `RayWavelengthCache::hero_indicatrix`'s doc comment. `is_biaxial` and
    // `cache.hero_indicatrix.is_some()` are the same condition, so the `is_biaxial` guard
    // here is redundant with the cached value but kept to match the pre-hoist code's own
    // explicit gate exactly.
    let hero_indicatrix = if is_biaxial {
        cache.hero_indicatrix
    } else {
        None
    };
    let (wave_dir_a_hero, wave_dir_b_hero) =
        hero_indicatrix.map_or((Vec3::ZERO, Vec3::ZERO), |ind| {
            if inside_gem {
                (k_hat, k_hat)
            } else {
                let (_, dir_a) = ind.resolve_entry_mode(k_hat, normal, cos_i, n_o_hero_seed, false);
                let (_, dir_b) = ind.resolve_entry_mode(k_hat, normal, cos_i, n_o_hero_seed, true);
                (dir_a, dir_b)
            }
        });
    (hero_indicatrix, wave_dir_a_hero, wave_dir_b_hero)
}

/// Per-channel biaxial mode-A/mode-B indices, each channel's own indicatrix evaluated
/// at the shared hero wave-normal directions from [`hero_biaxial_wave_dirs`]. Zero
/// (the array default) for a non-biaxial material or a channel whose indicatrix is
/// somehow unavailable, matching the pre-extraction code's behaviour exactly.
fn per_channel_biaxial_indices(
    cache: &RayWavelengthCache,
    is_biaxial: bool,
    wave_dir_a_hero: Vec3,
    wave_dir_b_hero: Vec3,
) -> ([f32; NUM_CHANNELS], [f32; NUM_CHANNELS]) {
    let mut n_biax_a_ch = [0.0f32; NUM_CHANNELS];
    let mut n_biax_b_ch = [0.0f32; NUM_CHANNELS];
    if is_biaxial {
        for k in 0..NUM_CHANNELS {
            // Bit-identical to a fresh `ctx.material.biaxial_indicatrix(ctx.lambdas[k])`
            // call -- see `RayWavelengthCache::biaxial_ch`'s doc comment.
            if let Some(ind_k) = cache.biaxial_ch[k] {
                n_biax_a_ch[k] = ind_k.wave_indices(wave_dir_a_hero).1;
                n_biax_b_ch[k] = ind_k.wave_indices(wave_dir_b_hero).0;
            }
        }
    }
    (n_biax_a_ch, n_biax_b_ch)
}

/// Per-channel `n1`/`n2`/`sin2(theta_t)`, evaluated against the SAME shared incidence
/// geometry (`cos_i`, the facet normal) as the hero -- only the index (`n_medium_ch`)
/// varies by channel.
fn per_channel_medium_indices(
    inside_gem: bool,
    n_medium_ch: [f32; NUM_CHANNELS],
    cos_i: f32,
) -> (
    [f32; NUM_CHANNELS],
    [f32; NUM_CHANNELS],
    [f32; NUM_CHANNELS],
) {
    let mut n1_ch = [0.0f32; NUM_CHANNELS];
    let mut n2_ch = [0.0f32; NUM_CHANNELS];
    let mut sin2_t_ch = [0.0f32; NUM_CHANNELS];
    for k in 0..NUM_CHANNELS {
        let n1k = if inside_gem { n_medium_ch[k] } else { 1.0 };
        let n2k = if inside_gem { 1.0 } else { n_medium_ch[k] };
        let etak = n1k / n2k;
        n1_ch[k] = n1k;
        n2_ch[k] = n2k;
        sin2_t_ch[k] = etak * etak * cos_i.mul_add(-cos_i, 1.0);
    }
    (n1_ch, n2_ch, sin2_t_ch)
}

pub(super) fn compute_bounce_refraction_geometry(
    ctx: &RayMaterialContext,
    cache: &RayWavelengthCache,
    normal: Vec3,
    k_hat: Vec3,
    inside_gem: bool,
    is_extraordinary: bool,
) -> BounceRefractionGeometry {
    let material = ctx.material;
    let hero_idx = ctx.hero_idx;
    let is_anisotropic = ctx.is_anisotropic;

    // P2: angle of incidence measured against the WAVE NORMAL `k_hat`, not the
    // Poynting/energy direction `S` -- see this module's own "wave normal vs Poynting
    // direction" design note above. `k_hat == S` trivially outside the crystal
    // (isotropic air) and for the uniaxial ordinary eigenmode, so this is bit-identical
    // to the pre-P2 `(-ray_dir).dot(normal)` in every case this task's bit-identity
    // requirement covers. Computed up front (it's purely geometric -- normal and wave
    // direction, no index involved) since the fixed-point iteration below needs it to
    // seed a trial refraction.
    let cos_i = (-k_hat).dot(normal).clamp(0.0, 1.0);
    let sin_i = cos_i.mul_add(-cos_i, 1.0).max(0.0).sqrt();

    // Refractive index calculation. Fix F: every channel now evaluates the material's
    // dispersion at its OWN wavelength instead of all eight channels sharing the
    // hero's index. The single traced GEOMETRIC path is still driven entirely by the
    // hero channel, so this remains one ray per bounce; only the per-channel
    // Stokes/radiometric bookkeeping is wavelength-correct.
    // Is this material's anisotropy genuinely biaxial (three distinct
    // principal indices) rather than the uniaxial ordinary/extraordinary
    // approximation every other anisotropic built-in uses?
    let is_biaxial = material.biaxial_delta_beta_alpha.is_some();

    // Bit-identical to a fresh `material.dispersion.evaluate(ctx.lambdas[hero_idx])` call
    // -- see `RayWavelengthCache::n_o_ch`'s doc comment. This is the exact "two textually
    // different expressions for the same quantity" case R3 calls out (this local seed vs.
    // the `n_o_ch[hero_idx]` lookup a few lines below): both now read the SAME cached
    // array element, so they stay trivially equal exactly as they always were.
    let n_o_hero_seed = cache.n_o_ch[hero_idx];
    let theta_c = theta_c_for_bounce(
        ctx,
        normal,
        k_hat,
        cos_i,
        inside_gem,
        is_biaxial,
        n_o_hero_seed,
    );

    let n_o_ch = cache.n_o_ch;
    let n_eff_ch = per_channel_effective_extraordinary_indices(ctx, &n_o_ch, theta_c);
    let n_o_hero = n_o_ch[hero_idx];
    let n_e_hero = n_o_hero + material.birefringence_delta;

    let (hero_indicatrix, wave_dir_a_hero, wave_dir_b_hero) = hero_biaxial_wave_dirs(
        cache,
        normal,
        k_hat,
        cos_i,
        inside_gem,
        is_biaxial,
        n_o_hero_seed,
    );
    let (n_biax_a_ch, n_biax_b_ch) =
        per_channel_biaxial_indices(cache, is_biaxial, wave_dir_a_hero, wave_dir_b_hero);
    // Fix G (Part 2) self-consistency: the hero's own per-mode index, read back out of
    // the per-channel arrays above (a LOOKUP, not an independent recomputation) --
    // exactly mirroring how the uniaxial branch defines `n_o_hero := n_o_ch[hero_idx]`.
    // Using the SAME array element (rather than a fresh `wave_indices` call) at both
    // the hero-level refraction below and the per-channel loop's own `k == hero_idx`
    // iteration guarantees they compute bit-identical directions from bit-identical
    // inputs -- otherwise the hero channel could spuriously fail its OWN
    // direction-match check against itself, chromatically self-terminating on every
    // biaxial entry.
    let n_biax_a_hero = n_biax_a_ch[hero_idx];
    let n_biax_b_hero = n_biax_b_ch[hero_idx];

    // Which per-channel index represents "the medium this ray is currently
    // in". While inside an anisotropic crystal, this is mode A or mode B depending on
    // which eigenmode the path was stochastically assigned to at its most recent entry
    // (`is_extraordinary`, set in the refract branch below). Outside the crystal this
    // keeps using the mode B array, matching the pre-existing entering-interface
    // approximation. `n_mode_a_ch`/`n_mode_b_ch` select between the
    // uniaxial and biaxial arrays computed above; for a non-biaxial material this is
    // exactly the previous `n_o_ch`/`n_eff_ch` pair, unchanged.
    let n_mode_a_ch = if is_biaxial { n_biax_a_ch } else { n_o_ch };
    let n_mode_b_ch = if is_biaxial { n_biax_b_ch } else { n_eff_ch };
    let n_medium_ch: [f32; NUM_CHANNELS] = if is_anisotropic && inside_gem && !is_extraordinary {
        n_mode_a_ch
    } else {
        n_mode_b_ch
    };
    let n_medium_hero = n_medium_ch[hero_idx];

    let n1 = if inside_gem { n_medium_hero } else { 1.0 };
    let n2 = if inside_gem { 1.0 } else { n_medium_hero };
    let eta = n1 / n2;
    let sin2_t = eta * eta * cos_i.mul_add(-cos_i, 1.0);

    let (n1_ch, n2_ch, sin2_t_ch) = per_channel_medium_indices(inside_gem, n_medium_ch, cos_i);

    BounceRefractionGeometry {
        cos_i,
        sin_i,
        is_biaxial,
        n_o_hero,
        n_e_hero,
        hero_indicatrix,
        n_biax_a_hero,
        n_biax_b_hero,
        n_o_ch,
        n_biax_a_ch,
        n1,
        n2,
        sin2_t,
        n1_ch,
        n2_ch,
        sin2_t_ch,
    }
}

/// P2: recovers the mode's Poynting (energy/ray) direction `S` for a freshly-computed
/// wave normal `k` -- needed after a reflection event, where the reflection law
/// (`k' = reflect(k, normal)`) acts on `k` directly and `S'` must be RE-EVALUATED (not
/// carried over from before the reflection), since the mode's own index and walk-off
/// angle both depend on the wave-normal direction, which the reflection just changed.
///
/// Returns `k` completely unchanged (`S == k`, no walk-off, no extra arithmetic) in
/// every case this task's bit-identity requirement covers: outside the crystal
/// (`!inside_gem`, isotropic air), an isotropic material (`!ctx.is_anisotropic`), and the
/// uniaxial ORDINARY eigenmode (`!is_extraordinary` while `!geo.is_biaxial`). A biaxial
/// material's two modes ("mode A"/"mode B", `is_extraordinary` selecting `want_slow`
/// exactly as [`apply_refract_bounce`]'s own entry-split already does) BOTH walk off --
/// see `hero_biaxial_wave_dirs`'s doc comment -- so [`BiaxialIndicatrix::mode_poynting_dir`]
/// is called unconditionally whenever `geo.is_biaxial`.
#[must_use]
pub(crate) fn poynting_dir_for_mode(
    ctx: &RayMaterialContext,
    geo: &BounceRefractionGeometry,
    k: Vec3,
    inside_gem: bool,
    is_extraordinary: bool,
) -> Vec3 {
    if !inside_gem || !ctx.is_anisotropic {
        return k;
    }
    if geo.is_biaxial {
        geo.hero_indicatrix
            .map_or(k, |ind| ind.mode_poynting_dir(k, is_extraordinary))
    } else if is_extraordinary {
        BirefringenceParams::extraordinary_poynting_dir(k, ctx.c_axis, geo.n_o_hero, geo.n_e_hero)
    } else {
        k
    }
}

/// Total Internal Reflection for the hero channel forces a deterministic reflect for
/// the shared path (probability 1, no pdf division needed). Each channel still gets its
/// OWN physically-correct outcome for that reflect event: if channel k is itself past
/// its own (wavelength-dependent) critical angle it gets the exact TIR phase
/// retardation at its own index; otherwise -- since the critical angle itself depends
/// on n(lambda), a channel can be below the hero's critical angle even though the hero
/// isn't -- it gets its own ordinary partial-reflectance Fresnel matrix. No probability
/// division is needed here: the hero's own selection probability for this action is 1
/// (forced), and each channel's Stokes value already carries the correct
/// importance-sampling division from whichever EARLIER decision actually had a
/// nontrivial hero probability. P2: the reflection law acts on the WAVE NORMAL `k_hat`,
/// not `S` -- returns the reflected wave normal `k'`; the caller derives the reflected
/// Poynting direction `S'` via [`poynting_dir_for_mode`] (re-evaluated at `k'`, per this
/// module's own design note above) and sets `current_ray.origin` itself (needs
/// `hit_point`, which this function has no reason to take).
pub(super) fn apply_tir_bounce(
    geo: &BounceRefractionGeometry,
    k_hat: Vec3,
    normal: Vec3,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
) -> Vec3 {
    for k in 0..NUM_CHANNELS {
        let n1k = geo.n1_ch[k];
        let n2k = geo.n2_ch[k];
        if geo.sin2_t_ch[k] > 1.0 {
            let delta_k = tir_phase_delta(n1k, geo.cos_i, geo.sin_i);
            let tir_matrix_k = MuellerMatrix::tir_retardation(delta_k);
            stokes[k] = stokes[k].apply_matrix(&tir_matrix_k);
            // Fix G (Part 2): channel k is ALSO past its own critical angle here, so
            // under k's own technique this reflect is ALSO forced (probability 1) --
            // and reflection direction never depends on wavelength, so channel k's
            // path-pdf factor here is exactly 1 -- a no-op on the running product,
            // left unwritten.
        } else {
            let cos_t_k = (1.0 - geo.sin2_t_ch[k]).max(0.0).sqrt();
            let r_s_k = f32::mul_add(n2k, -cos_t_k, n1k * geo.cos_i)
                / f32::mul_add(n2k, cos_t_k, n1k * geo.cos_i);
            let r_p_k = f32::mul_add(n1k, -cos_t_k, n2k * geo.cos_i)
                / f32::mul_add(n1k, cos_t_k, n2k * geo.cos_i);
            let refl_matrix_k = MuellerMatrix::fresnel_reflection(r_s_k, r_p_k);
            stokes[k] = stokes[k].apply_matrix(&refl_matrix_k);
            // Fix G (Part 2): channel k is genuinely below ITS OWN critical angle
            // here even though the hero forced a reflect -- under k's own technique,
            // reflecting (the observed outcome) has probability equal to k's own
            // unpolarized reflectance. Direction still matches trivially (reflection
            // is never dispersive), so no chromatic-termination check applies at a
            // reflect event.
            let r_unpol_k = (0.5 * r_p_k.mul_add(r_p_k, r_s_k * r_s_k)).clamp(1e-4, 1.0 - 1e-4);
            path_pdf[k] *= r_unpol_k;
        }
    }
    k_hat - 2.0 * k_hat.dot(normal) * normal
}

/// Partial Fresnel Reflection & Refraction via Stokes-Mueller Polarized Wave Transport
/// -- the REFLECT half. Which branch is taken (reflect vs. transmit) is decided once,
/// from the hero's `r_unpol`, driving the single shared geometric path; each channel
/// then applies ITS OWN Fresnel reflection matrix, divided by the SAME hero selection
/// probability (`r_unpol` -- the actual probability with which "reflect" was sampled).
/// This is the direct per-channel generalization of Fix B: when the material is
/// non-dispersive, every channel's matrix is identical to the hero's and this reduces
/// exactly to Fix B's corrected unweighted-sum result. P2: reflects the WAVE NORMAL
/// `k_hat` (see [`apply_tir_bounce`]'s identical doc-comment note) and returns the
/// reflected wave normal `k'`; the caller derives `S'` via [`poynting_dir_for_mode`] and
/// sets `current_ray.origin` itself.
fn apply_partial_reflect_bounce(
    geo: &BounceRefractionGeometry,
    r_unpol: f32,
    k_hat: Vec3,
    normal: Vec3,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
) -> Vec3 {
    for k in 0..NUM_CHANNELS {
        let n1k = geo.n1_ch[k];
        let n2k = geo.n2_ch[k];
        let refl_matrix_k = if geo.sin2_t_ch[k] > 1.0 {
            // Channel k is past ITS OWN critical angle here even though the hero
            // isn't. magnitude alone is not enough -- a wave beyond its own
            // critical angle also picks up the TIR phase retardation delta = delta_p
            // - delta_s, exactly as `apply_tir_bounce` applies for the hero. Fix G
            // (Part 2): channel k is forced to TIR here regardless of the hero's own
            // (partial-reflectance) physics -- under k's own technique this reflect
            // ALSO happens with probability 1, so no path-pdf factor is needed (a
            // no-op on the running product).
            let delta_k = tir_phase_delta(n1k, geo.cos_i, geo.sin_i);
            MuellerMatrix::tir_retardation(delta_k)
        } else {
            let cos_t_k = (1.0 - geo.sin2_t_ch[k]).max(0.0).sqrt();
            let r_s_k = f32::mul_add(n2k, -cos_t_k, n1k * geo.cos_i)
                / f32::mul_add(n2k, cos_t_k, n1k * geo.cos_i);
            let r_p_k = f32::mul_add(n1k, -cos_t_k, n2k * geo.cos_i)
                / f32::mul_add(n1k, cos_t_k, n2k * geo.cos_i);
            // Fix G (Part 2): channel k's own probability of choosing reflect here,
            // using k's OWN unpolarized reflectance (not the hero's r_unpol).
            // Reflection direction never depends on wavelength, so no
            // chromatic-termination direction check applies at a reflect event (only
            // refract events disperse).
            let r_unpol_k = (0.5 * r_p_k.mul_add(r_p_k, r_s_k * r_s_k)).clamp(1e-4, 1.0 - 1e-4);
            path_pdf[k] *= r_unpol_k;
            MuellerMatrix::fresnel_reflection(r_s_k, r_p_k)
        };
        stokes[k] = stokes[k].apply_matrix(&refl_matrix_k).scale(1.0 / r_unpol);
    }
    k_hat - 2.0 * k_hat.dot(normal) * normal
}

/// What [`apply_refract_bounce`] changes about the traced path beyond `stokes` and
/// `path_pdf` (which it mutates directly): the new wave normal `k'` and Poynting
/// direction `S'` (P2: no longer a single "new ray direction" -- see this module's own
/// "wave normal vs Poynting direction" design note above), and -- only on an
/// air->crystal entry into an anisotropic material -- the eigenmode this path was
/// stochastically assigned to. `None` means "leave `is_extraordinary` exactly as it
/// was", mirroring the pre-extraction code's `if entering_anisotropic { is_extraordinary
/// = use_extraordinary; }` guard exactly (not a `Some(is_extraordinary)` no-op, which
/// would be observably identical here but would misstate the condition for a reader).
struct RefractBounceOutcome {
    new_k: Vec3,
    new_s: Vec3,
    is_extraordinary_update: Option<bool>,
}

/// Partial Fresnel Reflection & Refraction via Stokes-Mueller Polarized Wave Transport
/// -- the REFRACT half (transmit branch, taken when the hero's `rng_bounce >=
/// r_unpol`). A companion channel's refracted direction must match the shared
/// hero-driven direction to within `DIRECTION_MATCH_COS_TOL` (not exact float equality
/// -- two independent evaluations of the SAME refraction formula with the SAME index
/// are bit-identical or differ by a handful of ULPs, but two DIFFERENT indices
/// generically produce a direction difference many orders of magnitude larger).
///
/// At an air->crystal entry into an anisotropic material, unpolarized incident
/// light couples into TWO orthogonally polarized eigenmodes -- ordinary (no walk-off)
/// and extraordinary (Poynting direction displaced by the walk-off angle), each
/// carrying roughly HALF the incident energy. Only one geometric path is traced per
/// sample, so which eigenmode this path becomes is chosen stochastically 50/50 --
/// but this is an energy SHARE, not a 1-of-N selection among equally-sized
/// alternatives: `trans_matrix_k` below already computes the full transmitted
/// intensity for a beam AT the selected mode's own index, so weighting it by that
/// mode's ~0.5 energy share gives an unbiased estimator of `0.5*T_o + 0.5*T_e` with a
/// factor of exactly 1 -- not the `1 / 0.5 = 2.0` a naive "divide by the selection
/// probability" rule (correct for the reflect/refract split's `r_unpol` /
/// `1 - r_unpol`, where the two branches partition disjoint, non-overlapping energy)
/// would suggest. The mode selection is still stochastic 50/50 -- only its throughput
/// weighting differs from a same-shaped disjoint split.
///
/// Chromatic termination (Fix G, Part 2): when channel k's own specular refraction
/// direction genuinely diverges from the direction the hero-driven path actually took
/// (or channel k cannot transmit at this angle at all), channel k's path pdf -- AND its
/// Stokes/radiance contribution -- are dropped to exactly 0, not merely down-weighted.
/// A two-channel Fresnel Monte Carlo cross-check confirms this is required for
/// unbiasedness, not optional extra realism -- see
/// `two_channel_dispersive_termination_monte_carlo_is_unbiased_under_alternating_hero`.
#[expect(
    clippy::too_many_arguments,
    reason = "already bundles every value fixed for the whole trace (ctx) or for this \
              bounce (cache, geo) into the established context structs; the rest are \
              this call's own scalar inputs (r_unpol, k_hat, normal, inside_gem, \
              is_extraordinary), the RNG stream identity (rng_seed, bounce), and the two \
              mutable per-channel accumulators (stokes, path_pdf) every other \
              bounce-dispatch function in this module tree takes the same way -- folding \
              those last four into one more struct wouldn't reduce what the caller has \
              to supply, only relabel it"
)]
fn apply_refract_bounce(
    ctx: &RayMaterialContext,
    cache: &RayWavelengthCache,
    geo: &BounceRefractionGeometry,
    r_unpol: f32,
    k_hat: Vec3,
    normal: Vec3,
    inside_gem: bool,
    is_extraordinary: bool,
    rng_seed: u32,
    bounce: u32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
) -> RefractBounceOutcome {
    let c_axis = ctx.c_axis;

    // The choice is recorded in the returned `is_extraordinary_update` so subsequent
    // internal bounces keep using the same eigenmode's index (via `n_medium_ch` in
    // `compute_bounce_refraction_geometry`). Exiting the crystal, and any refraction in
    // an isotropic material, leaves `entering_anisotropic` false and reduces exactly to
    // the pre-existing single-index behaviour.
    let entering_anisotropic = !inside_gem && ctx.is_anisotropic;
    // Mode SELECTION is still a stochastic 50/50 draw (see this function's doc
    // comment) -- only the throughput weighting that used to accompany it (a
    // `split_pdf` divisor/multiplier) is gone, since it estimated twice the
    // transmitted energy no interface can deliver.
    let use_extraordinary = if entering_anisotropic {
        let split_rand = (hash_u32(rng_seed ^ hash_u32(bounce ^ BIREFRINGENT_SPLIT_STREAM)) as f32)
            / 4_294_967_295.0;
        split_rand < 0.5
    } else {
        is_extraordinary
    };

    // Direction: the ordinary eigenmode's wave normal uses n_o and is never walked
    // off; the extraordinary eigenmode uses n_eff and its ENERGY (Poynting) direction
    // is displaced by the walk-off angle. Computed BEFORE the per-channel loop below
    // (Fix G / Part 2) because each companion channel's own hypothetical refracted
    // direction must be compared against this SAME hero-driven direction to detect a
    // dispersive mismatch. for a biaxial material entering the crystal,
    // NEITHER mode is a plain constant-index Snell refraction -- BOTH modes walk off
    // via `mode_poynting_dir` -- using `geo.n_biax_a_hero`/`geo.n_biax_b_hero` (the
    // SAME looked-up scalars the per-channel loop's own `k == hero_idx` iteration
    // uses) for self-consistency; see those fields' doc comment.
    // P2: `refr_wave_dir` here is the SNELL-REFRACTED WAVE NORMAL `k'` (Snell's law acts
    // on `k`, incident-side fed from this function's own `k_hat` parameter, not `S`) --
    // captured alongside the Poynting-converted `S'` in every branch below, since the
    // caller needs BOTH from here on (see [`RefractBounceOutcome`]'s own doc comment).
    // At an air->crystal ENTRY, `k_hat == S` trivially (isotropic air), so this is
    // bit-identical to the pre-P2 `ray_dir`-fed formula in every entry case. At an EXIT
    // (leaving an anisotropic crystal) or any isotropic refraction, using `k_hat`
    // instead of the old `ray_dir`(=S) is the actual P2 fix: Snell's law at the exit
    // interface must refract the wave normal, not the walked-off Poynting direction --
    // this is rule 4 ("at exit into air, refract k (not S)") from this module's design
    // note above.
    let (new_k, final_refr_dir) = if let (true, Some(ind)) =
        (entering_anisotropic && geo.is_biaxial, geo.hero_indicatrix)
    {
        let n2_hero_dir = if use_extraordinary {
            geo.n_biax_b_hero
        } else {
            geo.n_biax_a_hero
        };
        let eta_dir = geo.n1 / n2_hero_dir;
        let sin2_t_dir = (eta_dir * eta_dir * geo.cos_i.mul_add(-geo.cos_i, 1.0)).min(1.0);
        let cos_t_dir = (1.0 - sin2_t_dir).max(0.0).sqrt();
        let refr_wave_dir =
            (eta_dir * k_hat + f32::mul_add(eta_dir, geo.cos_i, -cos_t_dir) * normal).normalize();
        (
            refr_wave_dir,
            ind.mode_poynting_dir(refr_wave_dir, use_extraordinary),
        )
    } else {
        let n2_hero_dir = if entering_anisotropic && !use_extraordinary {
            geo.n_o_hero
        } else {
            geo.n2
        };
        let eta_dir = geo.n1 / n2_hero_dir;
        let sin2_t_dir = (eta_dir * eta_dir * geo.cos_i.mul_add(-geo.cos_i, 1.0)).min(1.0);
        let cos_t_dir = (1.0 - sin2_t_dir).max(0.0).sqrt();
        let refr_wave_dir =
            (eta_dir * k_hat + f32::mul_add(eta_dir, geo.cos_i, -cos_t_dir) * normal).normalize();

        let s = if entering_anisotropic && use_extraordinary {
            BirefringenceParams::extraordinary_poynting_dir(
                refr_wave_dir,
                c_axis,
                geo.n_o_hero,
                geo.n_e_hero,
            )
        } else {
            refr_wave_dir
        };
        (refr_wave_dir, s)
    };

    for k in 0..NUM_CHANNELS {
        apply_refract_channel(
            ctx,
            cache,
            geo,
            k,
            k_hat,
            normal,
            entering_anisotropic,
            use_extraordinary,
            final_refr_dir,
            r_unpol,
            stokes,
            path_pdf,
        );
    }

    RefractBounceOutcome {
        new_k,
        new_s: final_refr_dir,
        is_extraordinary_update: entering_anisotropic.then_some(use_extraordinary),
    }
}

/// One channel's share of [`apply_refract_bounce`]'s per-channel loop -- see that
/// function's doc comment for the full rationale (chromatic termination, Fix G Part 2
/// per-channel path-pdf bookkeeping). Each channel `k` is fully independent of every
/// other (no accumulator threaded across `k`), so extracting this loop body changes
/// nothing about the floating-point operations performed for any channel.
#[expect(
    clippy::too_many_arguments,
    reason = "already bundles ctx/cache/geo, same as apply_refract_bounce (this \
              function's only caller); the rest are the per-channel index `k` plus that \
              call's own scalar inputs and the two mutable accumulators every \
              bounce-dispatch function in this file takes -- see apply_refract_bounce's \
              own reason for why one more struct wouldn't reduce what's actually \
              threaded through here"
)]
fn apply_refract_channel(
    ctx: &RayMaterialContext,
    cache: &RayWavelengthCache,
    geo: &BounceRefractionGeometry,
    k: usize,
    k_hat: Vec3,
    normal: Vec3,
    entering_anisotropic: bool,
    use_extraordinary: bool,
    final_refr_dir: Vec3,
    r_unpol: f32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
) {
    const DIRECTION_MATCH_COS_TOL: f32 = 1.0 - 1e-6;
    let material = ctx.material;
    let c_axis = ctx.c_axis;

    let n1k = geo.n1_ch[k];
    // geo.n2_ch[k] (== n_eff_ch[k] here) is the extraordinary-biased index used to
    // decide reflect vs. refract; only correct for this channel's transmission if the
    // extraordinary mode was actually selected above. If the ordinary mode was
    // selected instead, this channel transmits at its own ordinary index
    // geo.n_o_ch[k]. for a biaxial material, use this channel's own
    // biaxial mode-A index (already resolved at the shared hero direction) instead of
    // the uniaxial index.
    let n2k = if entering_anisotropic && !use_extraordinary {
        if geo.is_biaxial {
            geo.n_biax_a_ch[k]
        } else {
            geo.n_o_ch[k]
        }
    } else {
        geo.n2_ch[k]
    };
    let sin2_t_k = (n1k / n2k).powi(2) * geo.cos_i.mul_add(-geo.cos_i, 1.0);
    if sin2_t_k > 1.0 {
        // Channel k cannot physically transmit at this angle even though the
        // hero-driven path did. Correct and unbiased, not a bias to be corrected: its
        // reflect-branch contributions elsewhere are already correctly weighted by
        // the hero's own selection probability.
        stokes[k] = stokes[k].scale(0.0);
        path_pdf[k] = 0.0;
        return;
    }

    let cos_t_k = (1.0 - sin2_t_k).max(0.0).sqrt();

    // Fix G (Part 2): the direction channel k's OWN technique would have taken
    // through this same interface, using k's OWN index -- including k's own
    // ordinary/extraordinary Poynting-direction walk-off, since that is ALSO
    // wavelength-dependent. Refraction is specular, so channel k's technique has
    // positive density of having produced the realized path only where this
    // direction coincides with `final_refr_dir`.
    let eta_dir_k = n1k / n2k;
    let refr_wave_dir_k =
        (eta_dir_k * k_hat + f32::mul_add(eta_dir_k, geo.cos_i, -cos_t_k) * normal).normalize();
    // Channel k's own biaxial walk-off, using k's own indicatrix
    // evaluated at k's own single-shot refracted wave direction -- the direct
    // per-channel generalization of the uniaxial `extraordinary_poynting_dir` call
    // below.
    // Bit-identical to a fresh `material.biaxial_indicatrix(lambdas[k])` call -- see
    // `RayWavelengthCache::biaxial_ch`'s doc comment.
    let final_dir_k = if let (true, Some(ind_k)) =
        (entering_anisotropic && geo.is_biaxial, cache.biaxial_ch[k])
    {
        ind_k.mode_poynting_dir(refr_wave_dir_k, use_extraordinary)
    } else if entering_anisotropic && use_extraordinary {
        let n_e_k = geo.n_o_ch[k] + material.birefringence_delta;
        BirefringenceParams::extraordinary_poynting_dir(
            refr_wave_dir_k,
            c_axis,
            geo.n_o_ch[k],
            n_e_k,
        )
    } else {
        refr_wave_dir_k
    };
    let direction_matches = final_dir_k.dot(final_refr_dir) >= DIRECTION_MATCH_COS_TOL;

    if direction_matches {
        let t_s_k = (2.0 * n1k * geo.cos_i) / f32::mul_add(n2k, cos_t_k, n1k * geo.cos_i);
        let t_p_k = (2.0 * n1k * geo.cos_i) / f32::mul_add(n1k, cos_t_k, n2k * geo.cos_i);
        let trans_matrix_k =
            MuellerMatrix::fresnel_transmission(n1k, n2k, geo.cos_i, cos_t_k, t_s_k, t_p_k);
        // No `/ split_pdf` here (see `apply_refract_bounce`'s doc comment): at an
        // anisotropic entry, `trans_matrix_k` is already the full transmitted
        // intensity for a beam at the SELECTED mode's own index, and that mode
        // carries only its ~0.5 energy share of the incident light -- dividing by
        // the 0.5 selection probability on top of that would estimate twice the
        // energy the interface actually transmits.
        stokes[k] = stokes[k]
            .apply_matrix(&trans_matrix_k)
            .scale(1.0 / (1.0 - r_unpol));

        // Fix G (Part 2): channel k's OWN probability of choosing "transmit" at this
        // interface, using k's OWN unpolarized reflectance. For k == hero_idx this
        // reproduces r_unpol exactly ONLY when `n2k` here is computed from the SAME
        // index the branch decision in `apply_partial_fresnel_bounce` used (`geo.n2`,
        // built from `geo.sin2_t`/`geo.n1`/`geo.n2`, i.e. `n_eff`/mode-B for an
        // anisotropic path) -- true for an isotropic material, and true at an
        // anisotropic entry when the extraordinary mode ends up selected (`n2k ==
        // geo.n2_ch[hero_idx] == geo.n2`). At an anisotropic entry where the ORDINARY
        // mode is selected instead, `n2k` above is `geo.n_o_ch[hero_idx]`
        // (uniaxial) or `geo.n_biax_a_ch[hero_idx]` (biaxial) -- genuinely different
        // from the `n_eff`-based index the branch decision was made with -- so
        // `r_unpol_k` for k == hero_idx is NOT `r_unpol` in that case; it is still the
        // physically correct probability for the mode actually selected, just not
        // numerically identical to the value that drove the earlier reflect/transmit
        // coin flip.
        let r_s_k = f32::mul_add(n2k, -cos_t_k, n1k * geo.cos_i)
            / f32::mul_add(n2k, cos_t_k, n1k * geo.cos_i);
        let r_p_k = f32::mul_add(n1k, -cos_t_k, n2k * geo.cos_i)
            / f32::mul_add(n1k, cos_t_k, n2k * geo.cos_i);
        let r_unpol_k = (0.5 * r_p_k.mul_add(r_p_k, r_s_k * r_s_k)).clamp(1e-4, 1.0 - 1e-4);
        // No `* split_pdf` here either: `path_pdf`'s role is `spectral_mis_weight`'s
        // per-channel weight, `N * path_pdf[hero] / sum(path_pdf)`, which is
        // scale-invariant under multiplying every channel's `path_pdf` by the SAME
        // uniform factor (0.5 here, identical for every k) -- so it was a pure no-op
        // on the actual MIS weight even before removal, just one that risked
        // underflow across many bounces for no benefit. Dropping it is a robustness
        // win, not a behavioural change.
        path_pdf[k] *= 1.0 - r_unpol_k;
    } else {
        // Chromatic termination -- see `apply_refract_bounce`'s doc comment.
        stokes[k] = stokes[k].scale(0.0);
        path_pdf[k] = 0.0;
    }
}

/// Partial Fresnel Reflection & Refraction via Stokes-Mueller Polarized Wave Transport:
/// decides reflect vs. transmit for the shared hero-driven path from the hero's own
/// `r_unpol` (a well-mixed hash of `(rng_seed, bounce)` -- Fix E, replacing the old
/// deterministic `(rng_seed + bounce*7919) % 1000` arithmetic progression, decorrelated
/// from the Russian-roulette draw via a distinct salt), then applies whichever event
/// was sampled via [`apply_partial_reflect_bounce`] or [`apply_refract_bounce`]. A
/// direct extraction of the pre-extraction inline branch dispatch: the same
/// floating-point operations, in the same order, driven by the same random draw.
/// Returns the new wave normal `k'`, the new Poynting direction `S'` (P2: reflection's
/// `k'` is re-converted to `S'` via [`poynting_dir_for_mode`] here, using the mode still
/// in effect -- reflection alone never changes `is_extraordinary`, see
/// `maybe_apply_internal_mode_coupling`'s own doc comment for the SEPARATE relabeling
/// step that may reassign it for the NEXT bounce), the new `inside_gem` state, and
/// (mirroring [`RefractBounceOutcome`]) the `is_extraordinary` update, if any.
#[expect(
    clippy::too_many_arguments,
    reason = "already bundles ctx/cache/geo, same as apply_refract_bounce/ \
              apply_refract_channel; the rest are this call's own scalar inputs and the \
              RNG stream identity plus the two mutable per-channel accumulators every \
              bounce-dispatch function in this file takes -- see apply_refract_bounce's \
              own reason for why one more struct wouldn't reduce what's actually \
              threaded through here"
)]
pub(super) fn apply_partial_fresnel_bounce(
    ctx: &RayMaterialContext,
    cache: &RayWavelengthCache,
    geo: &BounceRefractionGeometry,
    k_hat: Vec3,
    normal: Vec3,
    inside_gem: bool,
    is_extraordinary: bool,
    rng_seed: u32,
    bounce: u32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
) -> (Vec3, Vec3, bool, Option<bool>) {
    let cos_t = (1.0 - geo.sin2_t).sqrt();
    let r_s = f32::mul_add(geo.n2, -cos_t, geo.n1 * geo.cos_i)
        / f32::mul_add(geo.n2, cos_t, geo.n1 * geo.cos_i);
    let r_p = f32::mul_add(geo.n1, -cos_t, geo.n2 * geo.cos_i)
        / f32::mul_add(geo.n1, cos_t, geo.n2 * geo.cos_i);

    let r_unpol_raw = 0.5 * r_p.mul_add(r_p, r_s * r_s);
    let r_unpol = r_unpol_raw.clamp(1e-4, 1.0 - 1e-4);
    let rng_bounce =
        (hash_u32(rng_seed ^ hash_u32(bounce ^ FRESNEL_BRANCH_STREAM)) as f32) / 4_294_967_295.0;

    if rng_bounce < r_unpol {
        let new_k = apply_partial_reflect_bounce(geo, r_unpol, k_hat, normal, stokes, path_pdf);
        let new_s = poynting_dir_for_mode(ctx, geo, new_k, inside_gem, is_extraordinary);
        (new_k, new_s, inside_gem, None)
    } else {
        let outcome = apply_refract_bounce(
            ctx,
            cache,
            geo,
            r_unpol,
            k_hat,
            normal,
            inside_gem,
            is_extraordinary,
            rng_seed,
            bounce,
            stokes,
            path_pdf,
        );
        (
            outcome.new_k,
            outcome.new_s,
            !inside_gem,
            outcome.is_extraordinary_update,
        )
    }
}

#[cfg(test)]
mod tir_phase_retardation_tests {
    use super::*;

    /// The partial-reflection branch's "channel k is past its own critical
    /// angle" case now applies `tir_phase_delta` (via `MuellerMatrix::tir_retardation`)
    /// instead of the phase-blind `fresnel_reflection(1.0, 1.0)`. This checks the
    /// shared helper reproduces the same delta the dedicated (hero-is-past-critical)
    /// TIR branch computes inline, for a channel genuinely past its critical angle --
    /// i.e. the two sites are guaranteed to agree because they now share one formula.
    #[test]
    fn tir_phase_delta_is_nonzero_past_critical_angle() {
        // n1 = 2.42 (diamond-like), incidence well past the ~24.4 degree critical angle.
        let n1k = 2.42f32;
        let cos_i = 60.0f32.to_radians().cos();
        let sin_i = 60.0f32.to_radians().sin();
        assert!(
            n1k * sin_i > 1.0,
            "test setup must be past the critical angle"
        );

        let delta = tir_phase_delta(n1k, cos_i, sin_i);
        assert!(
            delta.abs() > 1e-3,
            "TIR phase retardation should be nonzero past the critical angle (got {delta})"
        );

        let tir_matrix = MuellerMatrix::tir_retardation(delta);
        let linear_45 = StokesVector::new(1.0, 0.0, 1.0, 0.0);
        let out = linear_45.apply_matrix(&tir_matrix);
        assert!(
            out.v.abs() > 1e-3,
            "a nonzero TIR phase retardation must convert some linear polarization to circular (V) (got V={})",
            out.v
        );
    }

    /// At exactly grazing incidence the retardation formula must stay finite (no NaN /
    /// Inf) even though several terms blow up in the naive algebra.
    #[test]
    fn tir_phase_delta_is_finite_near_grazing_incidence() {
        let n1k = 2.42f32;
        let cos_i = 0.001f32;
        let sin_i = cos_i.mul_add(-cos_i, 1.0).sqrt();
        let delta = tir_phase_delta(n1k, cos_i, sin_i);
        assert!(
            delta.is_finite(),
            "delta must remain finite near grazing incidence (got {delta})"
        );
    }
}

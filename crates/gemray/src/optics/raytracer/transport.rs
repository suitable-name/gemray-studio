//! The spectral transport loop itself.
//!
//! [`trace_spectral_ray`] and [`trace_spectral_ray_with_finish`] (the public entry
//! points) and [`trace_spectral_ray_inner`] (their shared bounce loop), plus the
//! per-bounce facet dispatch and Russian-roulette/mode-coupling machinery they drive.

use super::{
    NUM_CHANNELS,
    absorption::{apply_absorption, rotate_stokes_to_plane_of_incidence},
    camera::{FacetFinish, HitRecord, Ray},
    color::{apply_von_kries_white_balance, integrate_channels_to_xyz},
    environment::{EnvironmentSource, environment_white_balance, sample_environment_channel},
    intersect::{build_plane_soa, intersect_polyhedron_soa, shading_normal_near_edge},
    refraction::{
        BounceRefractionGeometry, RayMaterialContext, RayWavelengthCache,
        apply_partial_fresnel_bounce, apply_tir_bounce, build_ray_wavelength_cache,
        compute_bounce_refraction_geometry, poynting_dir_for_mode,
    },
    sampling::{MODE_COUPLING_STREAM, RUSSIAN_ROULETTE_STREAM, hash_u32},
    scattering::{ScatterStepOutcome, apply_frosted_bounce, try_scatter_step},
};
use crate::{
    geometry::plane::GpuFacetPlane,
    optics::{
        materials::{CrystalSystem, GemMaterial},
        polarization::StokesVector,
    },
};
use glam::Vec3;

/// Fix G (Part 1): builds the `N`-member wrapped hero-wavelength comb from a single
/// `hero_rand` in `[0, 1)`.
///
/// Per `GEMSTONE_RENDERING_BLUEPRINT.md`'s Algorithm 2, line 5
/// (`wavelengths[i] = 380 + fmod((lambda_hero - 380) + i*(400/N), 400)`, with
/// `lambda_hero = 380 + hero_rand*400` drawn over the FULL visible range rather than
/// just the first sub-band). Pulled out to a standalone, directly-testable function
/// (rather than left inlined in `trace_spectral_ray`) so the wraparound arithmetic
/// itself -- that every generated wavelength stays within `[380, 780]`, and that the
/// hero always lands at index 0 by construction -- can be unit tested in isolation,
/// without needing a full ray/material/planes setup. Uses a const-generic array
/// (`std::array::from_fn`) rather than returning a `Vec`, so calling it from
/// `trace_spectral_ray`'s hot per-sample path costs no heap allocation.
#[must_use]
pub fn wrapped_hero_wavelengths<const N: usize>(hero_rand: f32) -> [f32; N] {
    const SPECTRUM_SPAN: f32 = 780.0 - 380.0;
    let channel_width = SPECTRUM_SPAN / N as f32;
    let lambda_hero = hero_rand.mul_add(SPECTRUM_SPAN, 380.0);
    std::array::from_fn(|k| {
        let offset = (k as f32).mul_add(channel_width, lambda_hero - 380.0);
        380.0 + offset.rem_euclid(SPECTRUM_SPAN)
    })
}

/// Accumulates the environment radiance for a ray that exited or missed the gemstone
/// entirely, terminating `trace_spectral_ray`'s bounce loop. A direct extraction of the
/// pre-extraction inline block: the same per-channel `f32::mul_add` chain in the same
/// order, called exactly once (this is the loop's break path), so extracting it changes
/// nothing about the computed bits.
///
/// `StudioRig::new` (~40 sin/cos plus 16 vector normalizes) is constant across
/// an entire ray -- `light_yaw`/`light_pitch` don't vary per spectral channel -- so it
/// is built ONCE here, before the `NUM_CHANNELS` loop, rather than once per channel
/// inside `sample_studio_environment` as it used to be (measured: 204ns per
/// `StudioRig::new`, x8 redundant rebuilds per ray ~= the 1373ns/ray gap this fix
/// closes). Bit-identical: same rig values, computed once instead of eight times.
fn accumulate_miss_radiance(
    environment: EnvironmentSource<'_>,
    ray_dir: Vec3,
    lambdas: &[f32; NUM_CHANNELS],
    stokes: &[StokesVector; NUM_CHANNELS],
    radiance: &mut [f32; NUM_CHANNELS],
) {
    let studio_rig = match environment {
        EnvironmentSource::Studio {
            light_yaw,
            light_pitch,
            ..
        } => Some(crate::optics::studio_rig::StudioRig::new(
            light_yaw,
            light_pitch,
        )),
        EnvironmentSource::HdrMap(_) => None,
    };
    for k in 0..NUM_CHANNELS {
        let env_spectral =
            sample_environment_channel(environment, ray_dir, lambdas[k], studio_rig.as_ref());
        radiance[k] = f32::mul_add(stokes[k].intensity(), env_spectral, radiance[k]);
    }
}

/// Dispatches ONE bounce's TIR / partial-reflect / refract event -- or, for a
/// `FacetFinish::Frosted` facet, `apply_frosted_bounce` replacement -- and then
/// applies internal mode re-coupling if that event turned out to be an internal
/// reflection. This is `trace_spectral_ray_inner`'s entire per-bounce dispatch, extracted
/// into one function (mutating `current_ray`/`inside_gem`/`is_extraordinary` through
/// their `&mut` references) purely to keep that loop -- and clippy's function-length
/// lint -- manageable; every floating-point operation below is exactly what a bare
/// inline three-way dispatch would perform, just written once instead of at two call
/// sites. `was_internal_reflection`'s formula is the SAME one already established
/// for the polished reflect arm (`inside_gem && is_extraordinary_update.is_none() &&
/// new_inside_gem == inside_gem`, evaluated on the PRE-bounce `inside_gem`): it is
/// unconditionally `true` whenever a TIR bounce (polished or frosted) fires, since TIR is
/// only ever geometrically reachable with `inside_gem == true` to begin with (`n1 > n2`
/// for the hero channel requires it -- see `compute_bounce_refraction_geometry`), and it
/// is `false` for any transmit/refract event, since those always flip `inside_gem`.
///
/// P2: `current_k` is the wave normal `k` (see `refraction`'s own "wave normal vs
/// Poynting direction" design note) -- read here as the INCOMING `k` for this bounce's
/// TIR/Fresnel dispatch, and updated in place to the OUTGOING `k'` alongside
/// `current_ray.dir` (`S'`). A `FacetFinish::Frosted` bounce collapses `k' == S'`: a
/// diffuse (cosine-weighted-hemisphere) bounce already depolarizes exactly like a
/// Henyey-Greenstein scatter event (see `try_scatter_step`'s own identical treatment),
/// so there is no coherent wave-normal-vs-Poynting distinction left to track past it --
/// this is a documented simplification, not something rule 3 of the P2 design note
/// covers (that rule is specifically about the POLISHED TIR/reflect/refract dispatch).
#[expect(
    clippy::too_many_arguments,
    reason = "already bundles every value fixed for the trace (ctx) or the bounce \
              (cache, geo) into the established context structs; the rest are this \
              bounce's own scalar inputs (hit_point, normal, finish), the RNG stream \
              identity (rng_seed, bounce), and the FOUR separate pieces of per-ray state \
              a bounce can mutate (stokes, path_pdf, current_ray, current_k, inside_gem, \
              is_extraordinary) -- collapsing those into one struct would hide which of \
              six independently-updated fields each bounce kind actually touches behind \
              a single opaque &mut, which is a worse trade than the argument count"
)]
fn dispatch_bounce(
    ctx: &RayMaterialContext,
    cache: &RayWavelengthCache,
    geo: &BounceRefractionGeometry,
    hit_point: Vec3,
    normal: Vec3,
    finish: FacetFinish,
    rng_seed: u32,
    bounce: u32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
    path_pdf: &mut [f32; NUM_CHANNELS],
    current_ray: &mut Ray,
    current_k: &mut Vec3,
    inside_gem: &mut bool,
    is_extraordinary: &mut bool,
) {
    let (new_k, new_s, new_inside_gem, is_extraordinary_update) = if finish == FacetFinish::Frosted
    {
        let (new_dir, new_inside_gem, is_extraordinary_update) = apply_frosted_bounce(
            ctx,
            geo,
            normal,
            *inside_gem,
            *is_extraordinary,
            rng_seed,
            bounce,
            stokes,
            path_pdf,
        );
        (new_dir, new_dir, new_inside_gem, is_extraordinary_update)
    } else if geo.sin2_t > 1.0 {
        let k_prime = apply_tir_bounce(geo, *current_k, normal, stokes, path_pdf);
        let s_prime = poynting_dir_for_mode(ctx, geo, k_prime, *inside_gem, *is_extraordinary);
        (k_prime, s_prime, *inside_gem, None)
    } else {
        apply_partial_fresnel_bounce(
            ctx,
            cache,
            geo,
            *current_k,
            normal,
            *inside_gem,
            *is_extraordinary,
            rng_seed,
            bounce,
            stokes,
            path_pdf,
        )
    };
    let was_internal_reflection =
        *inside_gem && is_extraordinary_update.is_none() && new_inside_gem == *inside_gem;
    current_ray.dir = new_s;
    current_ray.origin = hit_point + new_s * 1e-4;
    *current_k = new_k;
    *inside_gem = new_inside_gem;
    if let Some(updated) = is_extraordinary_update {
        *is_extraordinary = updated;
    }
    maybe_apply_internal_mode_coupling(
        ctx,
        was_internal_reflection,
        rng_seed,
        bounce,
        is_extraordinary,
    );
}

/// Russian Roulette termination with weighted survival (Fix D). The previous hard
/// cutoff (`if max_intensity < 0.02 { break }`) simply discarded the remaining energy
/// outright once a path got dim, which biases the estimator dark -- worst on the long
/// internal bounce trains inside high-index stones. Instead, survive with probability
/// `q` (derived from the path's maximum spectral throughput) and, on survival, divide
/// every Stokes vector by `q` so the estimator stays unbiased in expectation
/// (`E[survive] * (1/q) = 1`). Returns `false` if the path should terminate (the
/// pre-extraction code's `break`), `true` if it survives (Stokes vectors already
/// rescaled in place, exactly as before).
pub(super) fn apply_russian_roulette(
    bounce: u32,
    rng_seed: u32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
) -> bool {
    let max_intensity = stokes.iter().fold(0.0f32, |a, s| a.max(s.intensity()));
    let q = max_intensity.clamp(0.05, 1.0);
    let rr_rand =
        (hash_u32(rng_seed ^ hash_u32(bounce ^ RUSSIAN_ROULETTE_STREAM)) as f32) / 4_294_967_295.0;
    if rr_rand > q {
        return false;
    }
    for s in &mut *stokes {
        *s = s.scale(1.0 / q);
    }
    true
}

/// Stochastic o<->e (uniaxial) / mode-A<->mode-B (biaxial)
/// re-coupling at an INTERNAL reflection inside an anisotropic crystal.
///
/// # What this models, and what it does not
///
/// Today's model assigns a path's eigenmode ONCE, at the air->crystal entry
/// (`BIREFRINGENT_SPLIT_STREAM`), and every subsequent internal bounce reuses that same
/// mode's scalar index (`n_medium_ch` in `compute_bounce_refraction_geometry`) via an
/// ISOTROPIC Fresnel calculation. That is wrong in a specific way: a uniaxial crystal's
/// o/e eigenbasis is defined relative to the LOCAL wave-normal direction at each facet
/// (`ordinary_eigen_polarization`/`extraordinary_eigen_polarization`, both functions of
/// `wave_normal`), so a facet with a different orientation relative to `c_axis` has a
/// DIFFERENT eigenbasis than the facet the path last bounced off. A polarization state
/// that was purely in the old facet's o (or e) eigenmode is therefore, in general, a
/// SUPERPOSITION of the new facet's o' and e' eigenmodes -- real light partially converts
/// between the two labels at every internal bounce, which is a genuine contributor to
/// the "doubling" complexity high-birefringence stones (zircon, moissanite) show.
///
/// This function models that conversion as a fresh, UNCONDITIONAL 50/50 coin flip of
/// which mode's index governs the path from here on -- deliberately the same blanket
/// 50/50 the entry split already uses, not a per-bounce probability derived from the
/// actual angle between the old and new eigenbases (that would require projecting the
/// carried Stokes/eigenmode state onto the new facet's eigenbasis, i.e. genuine
/// anisotropic Fresnel coefficients at an anisotropic interface -- explicitly out of
/// scope, see this module's callers' doc comments). What this buys, honestly: internal
/// bounces stop being frozen to whichever mode the path entered as, so a
/// many-internal-bounce path (exactly the pavilion TIR trains responsible for a cut
/// stone's brilliance) now samples a genuine MIX of o-like and e-like propagation across
/// its life, instead of committing to one for its entire interior traversal. What it does
/// NOT buy: a physically-derived conversion FRACTION per bounce, or coherent
/// superposition -- this is still scalar Fresnel with an effective index at an
/// anisotropic interface, exactly as the entry split already was, just re-rolled more
/// often.
///
/// # This is a RELABELING, not a SPLIT -- and neither draw needs a `1/p` division
///
/// It is tempting to reach for a `1 / selection_probability` reweighting for this
/// draw, since it is keyed off an unconditional 50/50 coin flip just like the entry
/// split's own mode-selection draw in `apply_refract_bounce`. Neither draw actually
/// needs one, though -- for two structurally different reasons.
///
/// At the entry split, one incident ray genuinely becomes TWO physically distinct rays
/// (the o-ray and the e-ray), each carrying only its own ~0.5 share of the incident
/// energy. The selected mode's own full Fresnel transmittance is evaluated against the
/// FULL incident Stokes vector -- i.e. it always computes "what if ALL the incident
/// energy took this mode's index," which overstates that mode's true contribution by
/// exactly the 2x its ~0.5 energy share corrects for. Since the mode is drawn with
/// probability 0.5 -- the SAME 0.5 as its true energy share -- the plain, unscaled
/// sample is already an unbiased estimator of the average `0.5*T_o + 0.5*T_e`:
/// `E[g(X)] = 0.5*g(o) + 0.5*g(e)` IS that average directly, no `1/p` correction
/// needed, precisely because the target quantity is itself a probability-weighted
/// average and the sampling distribution already matches those weights. (A `1/p`
/// correction is the right tool when the target is instead a SUM of disjoint
/// contributions sampled at non-matching probabilities -- e.g. the reflect/refract
/// branch's `r_unpol`/`1 - r_unpol` split, where the two branches are mutually
/// exclusive fates for the FULL incident beam, not simultaneous energy-sharing
/// alternatives, so recovering the full per-branch magnitude genuinely does require
/// dividing back out by the branch's own selection probability.)
///
/// Here, nothing new is created. The path already has exactly one Stokes vector and one
/// `path_pdf` -- this function does not touch either of them beyond the label they are
/// filed under. All it does is re-roll which of the two eigenmode LABELS ("o" or "e")
/// governs the refractive index used for the path's NEXT bounce (see
/// `compute_bounce_refraction_geometry`'s `n_medium_ch` lookup). That is a
/// redistribution of an existing quantity's bookkeeping, not an estimate of a sum of two
/// quantities. The energy carried by the path before this call and after it is, and must
/// be, identical -- otherwise a lossless closed system (a furnace) would gain or lose
/// energy on every single internal reflection, purely from re-labeling which index
/// governs the next bounce, which is nonsensical: the physical light did not get
/// brighter or dimmer because this code changed its mind about which eigenmode name to
/// call it.
///
/// The reason the entry split's unscaled draw and this function's unscaled draw are
/// BOTH correct despite looking so similar: the entry split's `0.5` selection
/// probability is deliberately set to match the SAME `0.5` blanket energy-fraction
/// assumption used for each mode's true contribution, so the direct sample already
/// IS the true per-mode contribution to the average sum -- see the section above, no
/// division needed. Here there is no sum and no per-mode energy fraction to recover --
/// the `0.5` is purely which label applies to the one quantity already in hand, so the
/// matching, unbiased scale factor is `1.0`, i.e. no scaling at all -- the same `1.0`
/// conclusion as the entry split, reached for a different reason.
///
/// # Unbiasedness
///
/// The draw is independent of every channel (one shared coin flip drives the one shared
/// geometric path, exactly like `BIREFRINGENT_SPLIT_STREAM`/`FRESNEL_BRANCH_STREAM`), so
/// every channel's own hypothetical technique would draw the identical outcome from this
/// same `(rng_seed, bounce)`-keyed stream -- but since this is a relabeling and not a
/// split (see above), neither `stokes` (the actual radiometric quantity) nor `path_pdf`
/// (each channel's own technique-density bookkeeping for the final spectral MIS weight)
/// needs any adjustment: neither `stokes` (the actual radiometric quantity) nor
/// `path_pdf` (each channel's own technique-density bookkeeping for the final spectral
/// MIS weight) is touched here -- unlike the entry split, this function takes no
/// `stokes`/`path_pdf` parameters at all, since it has nothing to do to them. Leaving
/// `path_pdf` untouched is also a robustness win over the old `*= 0.5`: across the long
/// internal-bounce trains a TIR-heavy pavilion produces (tens to ~100 internal
/// reflections is not unusual), `0.5^n` underflows to exactly `0.0` well before `n`
/// reaches 100, which would have driven `sum(path_pdf) == 0` and tripped
/// `spectral_mis_weight`'s "should not happen" guard. Returns the freshly-selected
/// `is_extraordinary` value; the caller applies it only when the internal reflection it
/// just dispatched actually happened while `inside_gem && is_anisotropic`.
fn apply_internal_mode_coupling(rng_seed: u32, bounce: u32) -> bool {
    const SPLIT_PDF: f32 = 0.5;
    let split_rand =
        (hash_u32(rng_seed ^ hash_u32(bounce ^ MODE_COUPLING_STREAM)) as f32) / 4_294_967_295.0;
    split_rand < SPLIT_PDF
}

/// Applies [`apply_internal_mode_coupling`] in place on `is_extraordinary` exactly when
/// the bounce dispatch that just ran was an internal reflection inside an anisotropic
/// crystal; a no-op otherwise. `was_internal_reflection` is `true` unconditionally for a
/// caller reporting a TIR bounce (`apply_tir_bounce` is reachable only with `n1 > n2` for
/// the hero channel, i.e. only from inside the gem -- see
/// `compute_bounce_refraction_geometry`: `n1 == 1.0` outside it), and
/// `is_extraordinary_update.is_none() && new_inside_gem == inside_gem_before_bounce` for
/// a caller reporting the outcome of `apply_partial_fresnel_bounce` (its reflect arm
/// returns `is_extraordinary_update == None` and never changes `inside_gem`, unlike its
/// refract arm, which always flips `inside_gem` and always returns `Some`). Extracted --
/// and made void/in-place rather than returning `Option<bool>` -- purely to keep both
/// this function's own call sites and `trace_spectral_ray_inner` itself compact enough
/// for clippy's function-length lint.
fn maybe_apply_internal_mode_coupling(
    ctx: &RayMaterialContext,
    was_internal_reflection: bool,
    rng_seed: u32,
    bounce: u32,
    is_extraordinary: &mut bool,
) {
    if ctx.enable_internal_mode_coupling && ctx.is_anisotropic && was_internal_reflection {
        *is_extraordinary = apply_internal_mode_coupling(rng_seed, bounce);
    }
}

/// Why a traced path's bounce loop stopped.
///
/// Bookkeeping the `bounce_cost` benchmark harness (`examples/bounce_cost.rs`) needs to
/// answer "what fraction of paths hit the bounce cap", not something any existing caller
/// of [`trace_spectral_ray_inner`] cares about. Maps 1:1 onto the loop's four
/// `break`/fall-through sites in [`trace_spectral_ray_inner`]:
/// - [`Self::Escaped`]: the ray missed every facet (or exited the polyhedron) and
///   [`accumulate_miss_radiance`] sampled the environment.
/// - [`Self::ScatterAbsorbed`]: [`try_scatter_step`] returned
///   [`ScatterStepOutcome::ScatteredAndTerminated`] -- only reachable for a material with
///   nonzero `scattering_sigma_s`; plain Beer-Lambert absorption in a non-scattering
///   material never terminates a path on its own; it only dims `stokes` until
///   [`apply_russian_roulette`] eventually kills it (recorded as [`Self::RussianRoulette`]
///   below, not this variant).
/// - [`Self::RussianRoulette`]: [`apply_russian_roulette`] drew a kill.
/// - [`Self::HitCap`]: none of the above fired before the `for bounce in 0..max_bounces`
///   loop ran out of iterations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathTermination {
    /// The ray missed the gemstone (or exited it) and the environment was sampled.
    Escaped,
    /// A Henyey-Greenstein scattering event extinguished the path (absorbed).
    ScatterAbsorbed,
    /// Russian roulette's stochastic kill fired.
    RussianRoulette,
    /// The loop ran out of bounces (`max_bounces`) before any of the above happened.
    HitCap,
}

/// Traces a single spectral ray through the 3D gemstone polyhedron with:
/// 1. 8-Channel Stratified Hero Wavelength Spectral Sampling (HWSS)
/// 2. Full 4D Stokes-Mueller Polarized Wave Tracking (with TIR phase shift and Brewster extinction)
/// 3. Birefringent Extraordinary Ray Walk-Off & Optical Doubling
/// 4. Directional Pleochroic Beer-Lambert Absorption Tensors
///
/// # `hero_rand`
///
/// The hero-wavelength draw is now an explicit `[0, 1)` parameter rather than derived
/// internally from `rng_seed` via `hash_u32` -- callers that want it STRATIFIED across a
/// pixel's own sample sequence (every real production call site: the worker, the
/// viewer, and the exporter) compute it via `low_discrepancy_base2` +
/// `cranley_patterson_rotate` from the absolute sample index and pixel index
/// separately, information a single opaque `rng_seed` cannot carry (a hash's whole job
/// is destroying input structure, so a caller cannot recover a stratifiable value from
/// `rng_seed` alone once pixel and sample have already been mixed together into it).
/// Callers that don't care about stratification (most tests, and the Tier 2/3 GPU
/// self-tests that intentionally reproduce the OLD unstratified formula to stay
/// comparable to their own historical baselines) can simply pass
/// `(hash_u32(rng_seed) as f32) / 4_294_967_295.0`, reproducing the pre-Fix-4 behaviour
/// exactly. `rng_seed` itself is unchanged: it still seeds every per-bounce draw
/// (Fresnel branch, Russian roulette, birefringent split) exactly as before
/// only touches the hero-wavelength and (in its callers) pixel-jitter draws.
#[expect(
    clippy::too_many_arguments,
    reason = "the crate's public raytracing entry point: each parameter is one \
              independent, named input a caller supplies once per traced ray (which \
              ray, which geometry, which material, the bounce cap, the lighting \
              environment, the RNG seed, the hero-wavelength draw, an optional debug \
              output) -- unlike the per-bounce internal functions in this module tree, \
              this is called once per sample, not threaded through every bounce, and \
              wrapping it in a context struct would only make every existing caller \
              (apps, examples, GPU/CPU parity tests) build a struct instead of calling a \
              function for no reduction in what they actually supply"
)]
#[must_use]
pub fn trace_spectral_ray(
    initial_ray: Ray,
    planes: &[GpuFacetPlane],
    material: &GemMaterial,
    max_bounces: u32,
    environment: EnvironmentSource<'_>,
    rng_seed: u32,
    hero_rand: f32,
    primary_hit_out: Option<&mut Option<HitRecord>>,
) -> Vec3 {
    // Internal o<->e mode re-coupling is always on for the
    // public entry point -- see `trace_spectral_ray_inner`'s doc comment for the
    // mechanism. The `bool` parameter exists so this file's own `#[cfg(test)]` module
    // can render the SAME material/ray/seed with it forced off, as a same-crate,
    // same-code-path A/B comparison proving the effect is real (see
    // `mode_coupling_tests::internal_mode_coupling_changes_zircon_render`) -- it is
    // deliberately not part of the public signature above it. Girdle finish:
    // `&[]` for `facet_finishes` -- every facet index looks up
    // `FacetFinish::default() == Polished` (see that lookup at the top of the bounce
    // loop below), so this reproduces the exact pre-Task-2 all-polished behaviour for
    // every one of this function's existing callers, unchanged.
    trace_spectral_ray_inner(
        initial_ray,
        planes,
        &[],
        material,
        max_bounces,
        environment,
        rng_seed,
        hero_rand,
        primary_hit_out,
        true,
        None,
    )
}

/// [`trace_spectral_ray`] with an explicit
/// per-facet `finish`.
///
/// See [`FacetFinish`]'s doc comment for why this is a separate function (not a new
/// parameter on `trace_spectral_ray` itself) and for its CPU-only status.
/// `facet_finishes[i]` is looked up for `planes[i]`; a facet index with no corresponding
/// entry (a shorter slice, or `&[]`) defaults to `FacetFinish::Polished`, so passing
/// `&[]` here is exactly equivalent to calling `trace_spectral_ray`.
#[expect(
    clippy::too_many_arguments,
    reason = "see trace_spectral_ray's own reason -- same public entry point, plus the \
              one additional per-facet finish slice this variant exists for"
)]
#[must_use]
pub fn trace_spectral_ray_with_finish(
    initial_ray: Ray,
    planes: &[GpuFacetPlane],
    facet_finishes: &[FacetFinish],
    material: &GemMaterial,
    max_bounces: u32,
    environment: EnvironmentSource<'_>,
    rng_seed: u32,
    hero_rand: f32,
    primary_hit_out: Option<&mut Option<HitRecord>>,
) -> Vec3 {
    trace_spectral_ray_inner(
        initial_ray,
        planes,
        facet_finishes,
        material,
        max_bounces,
        environment,
        rng_seed,
        hero_rand,
        primary_hit_out,
        true,
        None,
    )
}

/// Instrumented variant of [`trace_spectral_ray_with_finish`] for the `bounce_cost`
/// benchmark harness.
///
/// Identical computation (same code path, `enable_internal_mode_coupling = true`, same
/// call to [`trace_spectral_ray_inner`]), plus how many bounces the path actually took
/// and why it stopped -- see [`PathTermination`]'s doc comment for the four possible
/// reasons and how "bounces taken" is counted (the `for bounce in 0..max_bounces` loop's
/// iteration index at the point of termination; `max_bounces` itself for
/// [`PathTermination::HitCap`]).
#[expect(
    clippy::too_many_arguments,
    reason = "see trace_spectral_ray's own reason -- same public entry point, plus the \
              per-facet finish slice and the `bounce_cost` harness's own instrumentation \
              this variant exists for"
)]
#[must_use]
pub fn trace_spectral_ray_with_finish_instrumented(
    initial_ray: Ray,
    planes: &[GpuFacetPlane],
    facet_finishes: &[FacetFinish],
    material: &GemMaterial,
    max_bounces: u32,
    environment: EnvironmentSource<'_>,
    rng_seed: u32,
    hero_rand: f32,
) -> (Vec3, u32, PathTermination) {
    let mut termination = (max_bounces, PathTermination::HitCap);
    let radiance = trace_spectral_ray_inner(
        initial_ray,
        planes,
        facet_finishes,
        material,
        max_bounces,
        environment,
        rng_seed,
        hero_rand,
        None,
        true,
        Some(&mut termination),
    );
    (radiance, termination.0, termination.1)
}

/// Records the primary ray's first hit for the denoiser's guide buffers -- a
/// direct extraction from `trace_spectral_ray_inner` (same operations, same
/// order; out of line purely for clippy's line-count lint). Only bounce 0
/// writes: the ray at that point is still exactly the camera ray, and the
/// single traced geometric path is always driven by the hero channel, so this
/// unambiguously IS the hero channel's own first hit.
fn capture_primary_hit(
    primary_hit_out: &mut Option<&mut Option<HitRecord>>,
    bounce: u32,
    hit: Option<HitRecord>,
) {
    if bounce == 0
        && let Some(slot) = primary_hit_out.as_deref_mut()
    {
        *slot = hit;
    }
}

/// The facet's finish, with `Polished` for any index `facet_finishes` doesn't
/// cover -- a direct extraction from `trace_spectral_ray_inner` (same
/// operations, same order; out of line purely for clippy's line-count lint).
fn facet_finish_for(facet_finishes: &[FacetFinish], facet_idx: usize) -> FacetFinish {
    facet_finishes.get(facet_idx).copied().unwrap_or_default()
}

/// Interior-side handling of one facet hit -- a direct extraction from
/// `trace_spectral_ray_inner` (same operations in the same order, out of line
/// purely for clippy's line-count lint): flips the geometric normal to face
/// the interior ray, then applies the segment's plain Beer-Lambert absorption.
///
/// A scattering-active material's extinction for this segment was
/// already applied by `try_scatter_step`'s no-scatter branch -- calling the
/// plain `apply_absorption` here too would charge this segment's absorption
/// TWICE. `scattering_sigma_s <= 0.0` is exactly the pre-Task-1 case
/// (including every existing scene), where that branch never ran and this is
/// the ONLY absorption application, byte for byte the same call this module tree
/// always made. Does nothing while the ray is outside the gem.
///
/// `ray_ctx` bundles `(&RayMaterialContext, &RayWavelengthCache)` into a single
/// parameter (R3: rather than two separate ones) purely to keep this function's
/// argument count within clippy's `too_many_arguments` limit without a new `#[allow]`.
///
/// P2: `k_hat` is the WAVE NORMAL `k` (`current_k` in the caller), not the Poynting
/// direction `S` -- `apply_absorption`'s `channel_absorption_alphas` call derives its
/// eigen-polarizations from this direction, which is a property of `k`, not `S`; `hit_t`
/// (the path LENGTH) stays geometric, unaffected. See `refraction`'s own design note,
/// rule 6.
fn apply_interior_segment(
    ray_ctx: (&RayMaterialContext, &RayWavelengthCache),
    current_plane_normal: Vec3,
    k_hat: Vec3,
    hit_t: f32,
    inside_gem: bool,
    normal: &mut Vec3,
    stokes: &mut [StokesVector; NUM_CHANNELS],
) {
    let (ctx, cache) = ray_ctx;
    if !inside_gem {
        return;
    }
    *normal = -*normal;
    if ctx.material.scattering_sigma_s <= 0.0 {
        // See `apply_absorption`'s doc comment: a direct extraction of this
        // same block, same floating-point operations in the same order.
        apply_absorption(ctx, cache, current_plane_normal, k_hat, hit_t, stokes);
    }
}

/// Builds the per-sample [`RayMaterialContext`] -- a direct extraction from
/// `trace_spectral_ray_inner` (same operations in the same order, kept out of
/// line purely for clippy's line-count lint). The anisotropy gate and c-axis
/// lookup are byte-for-byte the code that used to sit inline.
fn build_ray_material_context(
    material: &GemMaterial,
    lambdas: [f32; NUM_CHANNELS],
    hero_idx: usize,
    enable_internal_mode_coupling: bool,
) -> RayMaterialContext<'_> {
    // Optical c-axis for anisotropic birefringence, per-material (
    // previously hard-coded to Vec3::Y for every material).
    let c_axis = material.c_axis;
    let is_anisotropic = material.crystal_system != CrystalSystem::Cubic
        && material.birefringence_delta.abs() > 1e-4;
    RayMaterialContext {
        material,
        lambdas,
        hero_idx,
        c_axis,
        is_anisotropic,
        enable_internal_mode_coupling,
    }
}

/// R3: bundles [`build_ray_material_context`] and [`build_ray_wavelength_cache`] into one
/// call (out of line purely for clippy's line-count lint on
/// `trace_spectral_ray_inner`) -- `RayWavelengthCache`'s own doc comment covers what it
/// caches and why that is bit-identical to the pre-hoist per-bounce recomputation.
fn build_ray_context(
    material: &GemMaterial,
    lambdas: [f32; NUM_CHANNELS],
    hero_idx: usize,
    enable_internal_mode_coupling: bool,
) -> (RayMaterialContext<'_>, RayWavelengthCache) {
    let mat_ctx =
        build_ray_material_context(material, lambdas, hero_idx, enable_internal_mode_coupling);
    let wavelength_cache = build_ray_wavelength_cache(&mat_ctx);
    (mat_ctx, wavelength_cache)
}

/// Records why/where the bounce loop stopped into the caller's `termination_out` slot, a
/// no-op if it is `None` -- extracted purely to keep each of
/// [`trace_spectral_ray_inner`]'s three break sites down to one line, for clippy's
/// function-length lint.
fn record_termination(
    termination_out: &mut Option<&mut (u32, PathTermination)>,
    bounce: u32,
    reason: PathTermination,
) {
    if let Some(out) = termination_out.as_deref_mut() {
        *out = (bounce, reason);
    }
}

/// The actual body of [`trace_spectral_ray`]/[`trace_spectral_ray_with_finish`] -- see
/// those functions' doc comments for the full parameter list.
/// `enable_internal_mode_coupling` is addition: `true` (what both public
/// wrappers always pass) applies `apply_internal_mode_coupling` at every internal
/// reflection inside an anisotropic crystal; `false` reproduces the exact pre-Task-1
/// behaviour (mode fixed at entry for the whole interior traversal), used only by this
/// file's own tests to measure the difference the mechanism makes.
#[expect(
    clippy::too_many_arguments,
    reason = "the shared bounce loop every public entry point above delegates to -- see \
              trace_spectral_ray's own reason for the public-facing parameters, plus the \
              two debug/instrumentation output hooks (primary_hit_out, termination_out) \
              only `bounce_cost` and this file's own tests populate"
)]
#[expect(
    clippy::too_many_lines,
    reason = "already at the pedantic line-length threshold pre-instrumentation; the \
              `bounce_cost` harness's termination bookkeeping (one `record_termination` \
              call per break site) pushes it a few lines over -- see `record_termination`'s \
              own doc comment for why that extraction, not further splitting this already \
              heavily-decomposed loop, is the right amount of surgery"
)]
fn trace_spectral_ray_inner(
    initial_ray: Ray,
    planes: &[GpuFacetPlane],
    facet_finishes: &[FacetFinish],
    material: &GemMaterial,
    max_bounces: u32,
    environment: EnvironmentSource<'_>,
    rng_seed: u32,
    hero_rand: f32,
    mut primary_hit_out: Option<&mut Option<HitRecord>>,
    enable_internal_mode_coupling: bool,
    mut termination_out: Option<&mut (u32, PathTermination)>,
) -> Vec3 {
    // Fix G (Part 1): hero is now drawn over the FULL visible range [380, 780), not
    // just the first channel_width sub-band, and the remaining channels wrap around
    // instead of running off the top past 780nm -- see `wrapped_hero_wavelengths`'s
    // doc comment for the formula (Algorithm 2, line 5 of
    // GEMSTONE_RENDERING_BLUEPRINT.md) and why it matters for MIS.
    //
    // The property this buys, that the old construction (hero confined to
    // [380, 430), no wraparound) did NOT have: a hero draw `h` and `h + channel_width`
    // now generate the exact SAME 8-member wavelength comb, just cyclically rotated --
    // so for a FIXED comb, each of its N members is equally likely (probability
    // channel_width / 400 = 1/N) to be the one drawn as hero on any given invocation.
    // That is the premise spectral MIS (`spectral_mis_weight` below) requires: across
    // the ensemble of independent ray samples a render accumulates, every wavelength
    // must be reachable as the driving ("hero") channel at positive, uniform
    // probability. Under the OLD construction it was not (channel 0 was *always* the
    // hero, i.e. there was exactly one technique, which is why `mis_weighted_radiance`
    // used to be a hard-coded identity function).
    let lambdas: [f32; NUM_CHANNELS] = wrapped_hero_wavelengths(hero_rand);

    let mut stokes = [StokesVector::unpolarized(1.0); NUM_CHANNELS];
    let mut radiance = [0.0f32; NUM_CHANNELS];

    // `hero_idx` names which slot of `lambdas` (and every other per-channel array
    // below) holds the wavelength actually driving the shared geometric path, so call
    // sites read `lambdas[hero_idx]` etc. instead of a bare magic-number `[0]`. Under
    // this exact construction (`lambdas[k] = wrap(lambda_hero + k*channel_width)`, see
    // `wrapped_hero_wavelengths`), substituting k=0 shows `lambdas[0] == lambda_hero`
    // identically on every call -- so `hero_idx` is provably 0 for every invocation,
    // not something that varies at runtime; what varies from call to call (Part 1's
    // whole point) is which physical WAVELENGTH VALUE lambda_hero itself is, now
    // spanning the full spectrum instead of being confined to the first sub-band.
    // `hero_idx` is still threaded through explicitly (rather than hardcoding `[0]` at
    // each of the several call sites below) so the code documents *which* channel is
    // playing the hero role, and so a future change to this construction (e.g.
    // rotating which array slot the hero lands in) has one place to update instead of
    // several silently-wrong `[0]`s.
    let hero_idx: usize = 0;

    // Fix G (Part 2): per-channel running density of "technique k (channel k as hero)
    // would have generated this exact realized path" -- see the TIR/reflect/refract
    // branches below for how each factor is derived, and `spectral_mis_weight`'s doc
    // comment for the final combination. Starts at 1.0 (the multiplicative identity)
    // for every channel: before any interaction, every technique is equally capable
    // of having produced the (as yet nonexistent) path.
    let mut path_pdf = [1.0f32; NUM_CHANNELS];

    let mut current_ray = initial_ray;
    // P2: the wave normal `k`, tracked ALONGSIDE `current_ray.dir` (the Poynting/energy
    // direction `S`) -- see `refraction`'s own "wave normal vs Poynting direction"
    // design note. Starts equal to the initial ray's direction: outside the gem (air)
    // `k == S` always (isotropic medium, no walk-off), so this is exactly `current_ray.dir`
    // until the first air->crystal entry.
    let mut current_k = initial_ray.dir;
    let mut inside_gem = false;
    // `None` until the first well-defined plane of incidence is recorded, so no
    // spurious frame rotation is applied at the very first surface (there was
    // previously no real "previous" plane of incidence to rotate from).
    let mut prev_plane_normal: Option<Vec3> = None;
    // Which eigenmode the ray currently inside the crystal was stochastically
    // assigned to at its most recent air->crystal entry. Meaningless while
    // `!inside_gem`; carried across internal bounces so a path that entered as (say)
    // the extraordinary ray keeps using the extraordinary index for every subsequent
    // internal reflection until it exits.
    //
    // For a UNIAXIAL material this is exactly the ordinary/extraordinary
    // split it always was. For a genuinely BIAXIAL material (Alexandrite, Topaz,
    // Tanzanite) there is no "ordinary" ray -- both eigenmodes are direction-dependent
    // and both walk off -- so this flag is generalized to a plain two-valued mode
    // selector: `false` selects "mode A" (the faster, lower-index root of
    // `BiaxialIndicatrix::wave_indices`), `true` selects "mode B" (the slower,
    // higher-index root). The name is kept (rather than renamed throughout) since the
    // uniaxial case, which is still the overwhelming majority of anisotropic built-ins,
    // keeps its exact original meaning and every read site below already documents the
    // biaxial generalization locally.
    let mut is_extraordinary = false;

    // Fixed across every bounce below -- see `RayMaterialContext`/`RayWavelengthCache`'s
    // own doc comments for why each is bundled rather than passed field by field.
    let (mat_ctx, wavelength_cache) =
        build_ray_context(material, lambdas, hero_idx, enable_internal_mode_coupling);

    // Built once per sample, scanned every bounce.
    let plane_soa = build_plane_soa(planes);

    for bounce in 0..max_bounces {
        let hit = intersect_polyhedron_soa(current_ray, &plane_soa);

        // Denoiser wiring: the primary ray's first-hit depth/normal/facet
        // index feed the À-Trous denoiser's guide buffers (see `renderer::denoise`).
        // Captured only at bounce 0 -- `current_ray` at this point is still exactly
        // `initial_ray`, and since the single traced geometric path is always driven by
        // the HERO channel (see `hero_idx`'s doc comment above), this unambiguously *is*
        // the hero channel's own first hit, not an average or a guess across the 8
        // dispersed channels a companion wavelength might have refracted toward.
        capture_primary_hit(&mut primary_hit_out, bounce, hit);

        let Some(hit_rec) = hit else {
            // Ray exited or missed the gemstone -> sample the environment source. See
            // `accumulate_miss_radiance`'s doc comment.
            accumulate_miss_radiance(
                environment,
                current_ray.dir,
                &lambdas,
                &stokes,
                &mut radiance,
            );
            record_termination(&mut termination_out, bounce, PathTermination::Escaped);
            break;
        };
        // Attempt a
        // Henyey-Greenstein scattering event somewhere along this segment (from the
        // current ray origin to the facet just hit, `hit_rec.t` away) BEFORE processing
        // that facet at all -- see `try_scatter_step`'s doc comment for the full
        // estimator/control-flow rationale (extracted into its own function purely to
        // keep this already-long function under clippy's line-count lint; every
        // floating-point operation is exactly what the inline version performed).
        if inside_gem {
            match try_scatter_step(
                &mat_ctx,
                &wavelength_cache,
                material,
                prev_plane_normal,
                &mut current_ray,
                &mut current_k,
                hit_rec.t,
                rng_seed,
                bounce,
                &mut stokes,
                &mut path_pdf,
            ) {
                ScatterStepOutcome::NotApplicable | ScatterStepOutcome::ReachedBoundary => {}
                ScatterStepOutcome::ScatteredAndSurvived => {
                    // The scattered Stokes vectors are already depolarized (see
                    // `maybe_scatter_or_extinguish`'s doc comment), so the previous
                    // plane of incidence is no longer physically meaningful -- reset it
                    // exactly like the pre-first-bounce `None`, so the NEXT real facet
                    // hit applies no spurious frame rotation (mirroring
                    // `prev_plane_normal`'s own doc comment above).
                    prev_plane_normal = None;
                    continue;
                }
                ScatterStepOutcome::ScatteredAndTerminated => {
                    record_termination(
                        &mut termination_out,
                        bounce,
                        PathTermination::ScatterAbsorbed,
                    );
                    break;
                }
            }
        }

        let hit_point = current_ray.origin + hit_rec.t * current_ray.dir;
        // Facet edge rounding: see `shading_normal_near_edge`'s doc comment.
        let mut normal = shading_normal_near_edge(
            planes,
            hit_point,
            hit_rec.facet_idx,
            hit_rec.normal,
            material.edge_rounding_radius,
        );

        // See `rotate_stokes_to_plane_of_incidence`'s doc comment: a direct extraction
        // of this same block, same floating-point operations in the same order. P2: the
        // Stokes plane-of-incidence frame is defined by the WAVE NORMAL `k`, not the
        // Poynting direction `S` -- see `refraction`'s own design note.
        let current_plane_normal =
            rotate_stokes_to_plane_of_incidence(current_k, normal, prev_plane_normal, &mut stokes);
        prev_plane_normal = Some(current_plane_normal);

        // See `apply_interior_segment`'s doc comment: a direct extraction of
        // this same block, same floating-point operations in the same order. P2: the
        // propagation direction feeding pleochroic absorption's eigen-polarizations is
        // `k`, not `S` -- `hit_rec.t` (the path LENGTH) stays geometric/`S`-based,
        // unaffected -- see `refraction`'s own design note, rule 6.
        apply_interior_segment(
            (&mat_ctx, &wavelength_cache),
            current_plane_normal,
            current_k,
            hit_rec.t,
            inside_gem,
            &mut normal,
            &mut stokes,
        );

        // See `compute_bounce_refraction_geometry`'s doc comment: a direct
        // extraction of this same block, same floating-point operations in the same
        // order, just packaged as a function. P2: index lookups (`theta_c`, biaxial
        // `wave_indices`), `cos_i`/`sin_i`, and every downstream Snell/Fresnel
        // evaluation use the WAVE NORMAL `k`, not `S` -- see `refraction`'s own design
        // note.
        let geo = compute_bounce_refraction_geometry(
            &mat_ctx,
            &wavelength_cache,
            normal,
            current_k,
            inside_gem,
            is_extraordinary,
        );

        // Girdle finish: which specular/diffuse treatment this facet gets --
        // `Polished` (the default for any index `facet_finishes` doesn't cover) takes
        // the exact pre-Task-2 dispatch; `Frosted` takes `apply_frosted_bounce` instead.
        // See `dispatch_bounce`'s doc comment for the unified call this feeds.
        let finish = facet_finish_for(facet_finishes, hit_rec.facet_idx);
        dispatch_bounce(
            &mat_ctx,
            &wavelength_cache,
            &geo,
            hit_point,
            normal,
            finish,
            rng_seed,
            bounce,
            &mut stokes,
            &mut path_pdf,
            &mut current_ray,
            &mut current_k,
            &mut inside_gem,
            &mut is_extraordinary,
        );

        // See `apply_russian_roulette`'s doc comment: a direct extraction of this same
        // block, same floating-point operations in the same order.
        if bounce > 4 && !apply_russian_roulette(bounce, rng_seed, &mut stokes) {
            record_termination(
                &mut termination_out,
                bounce,
                PathTermination::RussianRoulette,
            );
            break;
        }
    }

    // See `integrate_channels_to_xyz`'s doc comment: a direct extraction of this same
    // block, same floating-point operations in the same order.
    let xyz = integrate_channels_to_xyz(&radiance, &lambdas, &path_pdf, hero_idx);

    // Von Kries white-balance (diagonalised in Bradford LMS, not raw XYZ -- see
    // `compute_illuminant_white_balance`'s doc comment) so the chosen illuminant itself
    // renders as neutral white, rather than baking its colour temperature into every
    // pixel.
    apply_von_kries_white_balance(xyz, environment_white_balance(environment))
}

/// O<->e mode re-coupling at internal reflections.
#[cfg(test)]
mod mode_coupling_tests {
    use super::{super::environment::LightingPreset, *};
    use crate::geometry::cuts::StandardGemCuts;

    /// Mechanism-level check: `apply_internal_mode_coupling`'s draw genuinely produces
    /// BOTH outcomes at roughly the 50/50 its doc comment claims, across many
    /// `(rng_seed, bounce)` pairs -- not stuck at one constant value, which would
    /// silently turn re-coupling into a no-op.
    #[test]
    fn internal_mode_coupling_draw_is_close_to_50_50() {
        let mut true_count = 0u32;
        let trials = 20_000u32;
        for seed in 0..trials {
            if apply_internal_mode_coupling(seed, 3) {
                true_count += 1;
            }
        }
        let frac = f64::from(true_count) / f64::from(trials);
        assert!(
            (frac - 0.5).abs() < 0.02,
            "internal mode-coupling draw should split close to 50/50, got {frac} \
             extraordinary over {trials} trials"
        );
    }

    /// `apply_internal_mode_coupling` must leave `stokes` and `path_pdf` EXACTLY
    /// unchanged while still re-rolling `is_extraordinary` -- the relabeling-not-a-split
    /// rule this whole task turns on (see the function's own doc comment, "This is a
    /// RELABELING, not a SPLIT"). This replaces a prior version of this test that
    /// asserted the opposite (a `stokes *= 2.0` / `path_pdf *= 0.5` division) -- that was
    /// the bug: an inert `path_pdf` halving paired with an uncompensated `stokes`
    /// doubling on every single internal reflection inside a birefringent crystal,
    /// which compounds without bound across a TIR-heavy pavilion (see
    /// `examples/bounce_cost.rs`'s Quartz measurement in this task's own history for the
    /// magnitude: mean pixel luminance reaching ~5e16 at a 128-bounce cap).
    #[test]
    fn internal_mode_coupling_preserves_stokes_and_path_pdf() {
        let mut stokes = [StokesVector::unpolarized(1.0); NUM_CHANNELS];
        let mut path_pdf = [1.0f32; NUM_CHANNELS];
        // Distinct, non-1.0 values per channel so an accidental scale, swap, or zeroing
        // of any single channel would be caught, not just a uniform bug.
        for k in 0..NUM_CHANNELS {
            stokes[k] = StokesVector::unpolarized(1.0 + k as f32);
            path_pdf[k] = 0.1f32.mul_add(k as f32, 0.3);
        }
        let stokes_before = stokes;
        let path_pdf_before = path_pdf;
        let new_is_extraordinary = apply_internal_mode_coupling(7, 2);
        assert_eq!(
            stokes, stokes_before,
            "apply_internal_mode_coupling must not touch stokes at all -- it only \
             relabels which eigenmode governs the NEXT bounce, it does not split \
             energy the way the entry split does"
        );
        assert_eq!(
            path_pdf, path_pdf_before,
            "apply_internal_mode_coupling must not touch path_pdf at all -- see \
             stokes assertion above for why"
        );
        // Sanity: the draw itself still runs and returns a real bool (pinned against
        // the known (7, 2) hash outcome exercised by the old version of this test).
        assert!(
            new_is_extraordinary,
            "seed=7, bounce=2 is expected to draw extraordinary -- if this flips, \
             `internal_mode_coupling_draw_is_close_to_50_50` above still guards the \
             aggregate 50/50 split"
        );
    }

    /// The decisive measurement: render the SAME real Zircon material (real
    /// `birefringence_delta = +0.0590`, the largest in the built-in catalogue) at the
    /// SAME ray and seeds, averaged over many samples to beat down unrelated sampling
    /// noise, with internal mode coupling forced on vs. forced off
    /// (`trace_spectral_ray_inner`'s `enable_internal_mode_coupling` flag). Any nonzero,
    /// reproducible difference here is caused by exactly one thing: whether the eigenmode
    /// governing internal TIR/reflect bounces is re-rolled after entry -- not by any
    /// other change, since every other code path and RNG draw is identical between the
    /// two calls (same seeds, same ray, same material, same `max_bounces`).
    #[test]
    fn internal_mode_coupling_changes_zircon_render() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let zircon = GemMaterial::by_name("Zircon").expect("Zircon must be a built-in material");
        assert!(
            zircon.birefringence_delta.abs() > 0.01,
            "test assumes Zircon is strongly birefringent"
        );

        // Oblique ray (not aligned with c_axis == Y), 12 max_bounces -- the same ray
        // shape `non_biaxial_materials_render_bit_identical_to_pre_chapter_04_golden_values`
        // uses, chosen there specifically because it produces real internal TIR trains.
        let ray = Ray {
            origin: Vec3::new(0.0, 2.5, 0.0),
            dir: Vec3::new(0.18, -1.0, 0.07).normalize(),
        };
        let env = || LightingPreset::RingLights.studio(1.0, 0.85, 0.95);
        let samples = 256u32;

        let mut sum_with = Vec3::ZERO;
        let mut sum_without = Vec3::ZERO;
        for i in 0..samples {
            let seed = 1000 + i;
            let hero_rand = (hash_u32(seed) as f32) / 4_294_967_295.0;
            sum_with += trace_spectral_ray_inner(
                ray,
                &planes,
                &[],
                &zircon,
                12,
                env(),
                seed,
                hero_rand,
                None,
                true,
                None,
            );
            sum_without += trace_spectral_ray_inner(
                ray,
                &planes,
                &[],
                &zircon,
                12,
                env(),
                seed,
                hero_rand,
                None,
                false,
                None,
            );
        }
        let mean_with = sum_with / samples as f32;
        let mean_without = sum_without / samples as f32;
        let diff = (mean_with - mean_without).length();
        assert!(
            diff > 1e-4,
            "internal mode coupling should measurably change Zircon's rendered XYZ \
             (mean_with={mean_with:?}, mean_without={mean_without:?}, diff={diff})"
        );

        // Null check: the identical comparison for a CUBIC (isotropic, `is_anisotropic
        // == false`) material must show EXACTLY zero difference -- proving this is the
        // `is_anisotropic` gate doing its job, not sampling noise leaking through a loose
        // threshold above.
        let diamond = GemMaterial::by_name("Diamond").expect("Diamond must be a built-in material");
        let mut sum_with_diamond = Vec3::ZERO;
        let mut sum_without_diamond = Vec3::ZERO;
        for i in 0..samples {
            let seed = 1000 + i;
            let hero_rand = (hash_u32(seed) as f32) / 4_294_967_295.0;
            sum_with_diamond += trace_spectral_ray_inner(
                ray,
                &planes,
                &[],
                &diamond,
                12,
                env(),
                seed,
                hero_rand,
                None,
                true,
                None,
            );
            sum_without_diamond += trace_spectral_ray_inner(
                ray,
                &planes,
                &[],
                &diamond,
                12,
                env(),
                seed,
                hero_rand,
                None,
                false,
                None,
            );
        }
        assert_eq!(
            sum_with_diamond, sum_without_diamond,
            "Diamond (cubic, is_anisotropic == false) must be bit-identical with the flag \
             either way -- the mechanism must not leak into isotropic materials"
        );
    }
}

/// P2 (wave normal vs Poynting direction): a plane-parallel uniaxial slab, e-mode
/// forced. Drives `compute_bounce_refraction_geometry`/`apply_partial_fresnel_bounce`
/// directly (bypassing the full stochastic bounce loop, which only returns integrated
/// XYZ radiance with no directional output) at exactly two facets -- the slab's entry
/// and exit faces -- so the resulting `k`/`S` at each step can be inspected directly.
#[cfg(test)]
mod p2_wave_normal_tests {
    use super::{super::intersect::intersect_polyhedron, *};
    use crate::optics::materials::CrystalSystem;

    /// Two real slab faces (normal `+-Y`, `y` in `[-HALF_THICKNESS, HALF_THICKNESS]`)
    /// plus four "far blank" planes (`+-X`/`+-Z` at `+-1000`) purely so
    /// `intersect_polyhedron` sees a legitimate bounded polyhedron -- the walk-off
    /// lateral displacement this test measures is many orders of magnitude smaller than
    /// 1000 model units, so the blanks are never actually reached by the traced ray.
    const HALF_THICKNESS: f32 = 1.0;
    fn slab_planes() -> Vec<GpuFacetPlane> {
        vec![
            GpuFacetPlane::new(Vec3::Y, -HALF_THICKNESS),
            GpuFacetPlane::new(Vec3::NEG_Y, -HALF_THICKNESS),
            GpuFacetPlane::new(Vec3::X, -1000.0),
            GpuFacetPlane::new(Vec3::NEG_X, -1000.0),
            GpuFacetPlane::new(Vec3::Z, -1000.0),
            GpuFacetPlane::new(Vec3::NEG_Z, -1000.0),
        ]
    }

    /// Drives the entry facet (bounce 0, `!inside_gem`) and the exit facet (bounce 1,
    /// `inside_gem`) directly via [`compute_bounce_refraction_geometry`]/
    /// [`apply_partial_fresnel_bounce`], searching `rng_seed` for the first value that
    /// (a) selects the EXTRAORDINARY eigenmode at entry and (b) transmits (never
    /// reflects/TIRs) at BOTH facets -- the "plane-parallel slab, straight through"
    /// scenario this test needs. Returns `(entry_k, entry_s, exit_k, exit_s,
    /// lateral_displacement)`.
    ///
    /// # Panics
    ///
    /// Panics if no seed in `0..SEED_SEARCH_LIMIT` satisfies every condition -- a test
    /// premise, not an expected runtime failure (each condition individually has a
    /// probability of a large fraction of 1, so the combined search is expected to
    /// succeed within a handful of trials).
    fn trace_forced_extraordinary_slab(
        material: &GemMaterial,
        incident_origin: Vec3,
        incident_dir: Vec3,
    ) -> (Vec3, Vec3, Vec3, Vec3, f32) {
        const SEED_SEARCH_LIMIT: u32 = 20_000;
        let planes = slab_planes();
        let lambdas = [589.3f32; NUM_CHANNELS]; // sodium D line, every channel (direction-only test)
        let mat_ctx = build_ray_material_context(material, lambdas, 0, false);
        let cache = build_ray_wavelength_cache(&mat_ctx);

        // The entry hit point depends only on the fixed incident ray/geometry, not on
        // `seed` -- computed once, outside the search loop.
        let entry_hit = intersect_polyhedron(
            Ray {
                origin: incident_origin,
                dir: incident_dir,
            },
            &planes,
        )
        .expect("incident ray must hit the slab's top face");
        let entry_point = incident_origin + entry_hit.t * incident_dir;

        for seed in 0..SEED_SEARCH_LIMIT {
            let mut stokes = [StokesVector::unpolarized(1.0); NUM_CHANNELS];
            let mut path_pdf = [1.0f32; NUM_CHANNELS];

            // Entry facet: normal +Y (top of the slab), current_k == current_ray.dir ==
            // incident_dir (air, isotropic -- k == S trivially before any interface).
            let entry_normal = Vec3::Y;
            let geo0 = compute_bounce_refraction_geometry(
                &mat_ctx,
                &cache,
                entry_normal,
                incident_dir,
                false,
                false,
            );
            if geo0.sin2_t > 1.0 {
                continue; // TIR is geometrically unreachable entering from air (n1==1), but stay defensive.
            }
            let (k1, s1, inside_gem_1, is_extraordinary_update) = apply_partial_fresnel_bounce(
                &mat_ctx,
                &cache,
                &geo0,
                incident_dir,
                entry_normal,
                false,
                false,
                seed,
                0,
                &mut stokes,
                &mut path_pdf,
            );
            let entered_extraordinary = is_extraordinary_update == Some(true);
            if !inside_gem_1 || !entered_extraordinary {
                continue; // reflected off the top face, or entered the ORDINARY mode -- keep searching.
            }

            // Advance geometrically along S (k1/s1 as computed) to the exit facet.
            let post_entry_origin = entry_point + s1 * 1e-4;
            let exit_ray = Ray {
                origin: post_entry_origin,
                dir: s1,
            };
            let Some(exit_hit) = intersect_polyhedron(exit_ray, &planes) else {
                continue;
            };
            let exit_point = post_entry_origin + exit_hit.t * s1;
            // `intersect_polyhedron`'s hit normal is the OUTWARD-facing plane normal
            // (here, -Y, the bottom face's own defining normal); `apply_interior_segment`
            // flips it to face the interior ray before geometry/dispatch -- see
            // `trace_spectral_ray_inner`'s own `*normal = -*normal` under `inside_gem`.
            let exit_normal = -exit_hit.normal;

            let geo1 =
                compute_bounce_refraction_geometry(&mat_ctx, &cache, exit_normal, k1, true, true);
            if geo1.sin2_t > 1.0 {
                continue; // Would TIR back into the slab -- keep searching for a seed that transmits.
            }
            let (k2, s2, inside_gem_2, _) = apply_partial_fresnel_bounce(
                &mat_ctx,
                &cache,
                &geo1,
                k1,
                exit_normal,
                true,
                true,
                seed,
                1,
                &mut stokes,
                &mut path_pdf,
            );
            if inside_gem_2 {
                continue; // Reflected back into the slab instead of transmitting out -- keep searching.
            }

            // Lateral displacement: the perpendicular distance from `exit_point` to the
            // infinite line through `entry_point` along `incident_dir` -- nonzero iff the
            // walk-off genuinely displaced the exit point sideways relative to where a
            // straight (non-birefringent) path would have exited.
            let to_exit = exit_point - entry_point;
            let along = to_exit.dot(incident_dir);
            let perp = to_exit - along * incident_dir;
            let lateral_displacement = perp.length();

            return (k1, s1, k2, s2, lateral_displacement);
        }
        panic!(
            "no seed in 0..{SEED_SEARCH_LIMIT} entered the extraordinary mode and \
             transmitted cleanly through both slab faces -- test premise violated"
        );
    }

    /// The decisive P2 correctness check: for a plane-parallel uniaxial slab with a
    /// tilted c-axis, the EXTRAORDINARY ray's exit wave normal `k` (and, once back in
    /// isotropic air, `S == k`) must come out parallel to the incident ray within 1e-5
    /// -- exactly the classical "parallel slab" result Snell's law gives for ANY single
    /// constant index, applied twice (once per interface) to the SAME `k` throughout the
    /// slab's interior (see `refraction.rs`'s design note, rule 4: exit refraction uses
    /// `k`, not the walked-off `S`). The exit point must ALSO be laterally displaced by a
    /// nonzero amount from the straight-through path -- the walk-off did something real,
    /// it just didn't change the OUTGOING direction.
    #[test]
    fn plane_parallel_uniaxial_slab_extraordinary_ray_exits_parallel_and_displaced() {
        let mut material = GemMaterial::by_name("Zircon")
            .expect("\"Zircon\" is a built-in uniaxial material in GemMaterial::all_materials()");
        assert_eq!(material.crystal_system, CrystalSystem::Tetragonal);
        assert!(
            material.birefringence_delta.abs() > 0.01,
            "test premise: Zircon must be strongly birefringent"
        );
        // c-axis deliberately tilted away from BOTH the slab normal (Y) and the
        // incidence plane (XY) -- genuine 3D walk-off, not a coincidental in-plane one.
        material.c_axis = Vec3::new(0.3, 0.8, 0.5).normalize();

        let incident_origin = Vec3::new(0.0, 5.0, 0.0);
        // ~16.7 degrees off normal incidence -- comfortably sub-critical for n~1.9, and
        // oblique enough that Snell's law genuinely bends k (unlike normal incidence,
        // where k passes straight through regardless of index and the k/S distinction
        // this task fixes could never show up in the exit direction at all).
        let incident_dir = Vec3::new(0.3, -1.0, 0.0).normalize();

        let (entry_k, entry_s, exit_k, exit_s, lateral_displacement) =
            trace_forced_extraordinary_slab(&material, incident_origin, incident_dir);

        // Sanity: the extraordinary mode's walk-off must have actually fired (entry_k
        // != entry_s) -- otherwise this test would be silently checking the degenerate
        // ordinary-mode-equivalent case instead of what it claims to.
        assert!(
            (entry_k - entry_s).length() > 1e-4,
            "test premise: the extraordinary mode's walk-off should visibly separate k \
             from S at entry (entry_k={entry_k:?}, entry_s={entry_s:?})"
        );

        let cos_parallel_k = exit_k.dot(incident_dir).clamp(-1.0, 1.0);
        let cos_parallel_s = exit_s.dot(incident_dir).clamp(-1.0, 1.0);
        assert!(
            (1.0 - cos_parallel_k).abs() < 1e-5,
            "exit wave normal k must be parallel to the incident ray within 1e-5 \
             (exit_k={exit_k:?}, incident_dir={incident_dir:?}, 1-cos={})",
            1.0 - cos_parallel_k
        );
        // Once back in air, S == k exactly (isotropic medium) -- both must agree.
        assert!(
            (1.0 - cos_parallel_s).abs() < 1e-5,
            "exit Poynting direction S (== k in air) must be parallel to the incident ray \
             within 1e-5 (exit_s={exit_s:?}, incident_dir={incident_dir:?}, 1-cos={})",
            1.0 - cos_parallel_s
        );
        assert!(
            (exit_k - exit_s).length() < 1e-6,
            "S must equal k exactly once back in isotropic air (exit_k={exit_k:?}, \
             exit_s={exit_s:?})"
        );
        assert!(
            lateral_displacement > 1e-4,
            "the exit point must be laterally displaced from the straight-through path \
             by a nonzero amount (the walk-off's real, physical effect) -- got {lateral_displacement}"
        );
    }
}

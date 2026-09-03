//! Deterministic PRNG hashing, per-bounce RNG stream salts, and the low-discrepancy
//! (stratified) sampling helpers used for pixel jitter and hero-wavelength selection.

/// Decorrelated hash salts for `trace_spectral_ray`'s independent per-bounce random
/// draws. Each draw is `hash_u32(rng_seed ^ hash_u32(bounce ^ SALT))`, so the streams
/// are independent well-mixed sequences instead of the same value (or a trivially
/// correlated arithmetic progression) being reused for multiple decisions.
// `pub(crate)`, not private: `renderer::gpu::rng_check`'s GPU/CPU RNG equivalence
// self-test (Phase 0) needs to hash against these EXACT values, not a hand-copied
// duplicate that could silently drift from whatever this file defines -- the entire
// point of that self-test is catching drift, so it has to read the real constant.
pub(crate) const FRESNEL_BRANCH_STREAM: u32 = 0x9E37_79B1;
pub(crate) const RUSSIAN_ROULETTE_STREAM: u32 = 0x517C_C1B7;
/// Stream for the ordinary/extraordinary eigenmode split at an air->crystal
/// entry into an anisotropic material. Distinct from the two streams above so all
/// three per-bounce stochastic decisions are decorrelated from one another.
pub(crate) const BIREFRINGENT_SPLIT_STREAM: u32 = 0x2545_F491;
/// o<->e (uniaxial) / mode-A<->mode-B (biaxial) re-coupling stream for an INTERNAL
/// reflection inside an anisotropic crystal -- see `apply_internal_mode_coupling`'s doc
/// comment for the physics this approximates. Distinct from `BIREFRINGENT_SPLIT_STREAM`
/// (used only at the initial air->crystal entry) so the two decisions are decorrelated;
/// distinct from every other stream for the same reason.
pub(crate) const MODE_COUPLING_STREAM: u32 = 0xCC9E_2D51;
/// Girdle finish: the 2D cosine-weighted-hemisphere direction draw at a
/// `FacetFinish::Frosted` bounce -- two independent streams for `(u, v)`, mirroring how
/// `PIXEL_JITTER_X_ROTATION_STREAM`/`PIXEL_JITTER_Y_ROTATION_STREAM` use two streams for
/// their own 2D draw. Splitmix64-derived constants, distinct from every stream above.
pub(crate) const FROSTED_DIR_U_STREAM: u32 = 0x27D4_EB2F;
pub(crate) const FROSTED_DIR_V_STREAM: u32 = 0x1656_67B1;
/// The free-path distance
/// draw for [`maybe_scatter_or_extinguish`]'s homogeneous-medium exponential sampler.
/// Distinct from every stream above so a scattering-active material's per-bounce
/// stochastic decisions stay decorrelated from the existing Fresnel/Russian-roulette/
/// birefringent-split/frosted-direction draws.
pub(crate) const DISTANCE_SAMPLE_STREAM: u32 = 0xA24B_AED4;
/// The 2D Henyey-Greenstein direction draw at a scattering
/// event -- two independent streams for `(u, v)`, mirroring how
/// `FROSTED_DIR_U_STREAM`/`FROSTED_DIR_V_STREAM` use two streams for the frosted BSDF's
/// own 2D direction draw.
pub(crate) const PHASE_DIR_U_STREAM: u32 = 0x9FB2_1C65;
pub(crate) const PHASE_DIR_V_STREAM: u32 = 0x1CE4_E5B9;

/// Fast integer hash for high-quality spatial/temporal PRNG
#[must_use]
pub const fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    x = x.wrapping_mul(0xc2b2_ae35);
    x ^= x >> 16;
    x
}

// ---------------------------------------------------------------------------------
// Stratified pixel jitter and hero-wavelength sampling.
//
// See [`low_discrepancy_base2`], [`radical_inverse_base`], and
// [`cranley_patterson_rotate`]'s own doc comments for the mechanism. Callers:
// `apps/gemray-worker/src/render_core.rs::trace_samples`,
// `apps/diagram-gui/src/bridge/render_thread.rs::render_frame_scanlines`, and
// `apps/diagram-gui/src/bridge/export_thread.rs::render_batch` all compute pixel jitter
// and the hero draw this way now, then pass the hero value into [`trace_spectral_ray`]
// as `hero_rand` (a new explicit parameter -- it used to derive `hero_rand` internally
// from `rng_seed` via `hash_u32`, which cannot be stratified: a hash's whole job is
// destroying input structure, so feeding it a stratified value un-stratifies it right
// back into pseudo-random). Mirrored bit-for-bit in `shaders/spectral_transport.wgsl`'s
// `transport_main` (the shipped megakernel) and, for the Tier 1 GPU/CPU RNG
// bit-exactness self-test, in `shaders/rng_equivalence.wgsl` +
// `renderer::gpu::rng_check::cpu_record`.
//
// # Why THREE DIFFERENT PRIME BASES (2, 3, 5), not one base rotated three ways
//
// An earlier version of this fix used the SAME base-2 sequence for pixel-jitter-X,
// pixel-jitter-Y, AND the hero wavelength, decorrelated only by an additive
// Cranley-Patterson rotation per quantity. Measured directly (many independent
// replicates of the same ray, comparing the sample-variance of the N-sample estimator
// against the old unstratified scheme): that construction made variance WORSE, not
// better, for exactly the highest-variance ("fire") pixels this fix targets --
// consistently by 4-7x at a realistic sample count. Isolating which stratified
// quantity was responsible (stratifying only the hero draw; stratifying only the
// pixel jitter, in each case leaving the other unstratified) showed the pixel-jitter
// pairing was the dominant cause: two base-2 van der Corput sequences, even with
// independent rotations, pair up (jx, jy) into a systematically STRUCTURED 2D point
// set (samples cluster along a limited set of lines rather than covering the unit
// square), rather than genuinely improving 2D coverage -- the classic same-base
// lattice-alignment pitfall that is exactly why real Halton sequences use a DIFFERENT
// PRIME per dimension rather than the same base repeated. Switching pixel-jitter-Y and
// the hero draw to base 3 and base 5 respectively (keeping pixel-jitter-X on the fast
// bit-reversal base-2 path) removes that structure: measured on the same test, the
// worst regressed pixels went from ~0.14x-0.18x (variance 5-7x WORSE) to ~0.76x-1.6x
// (roughly neutral to a genuine improvement), and the aggregate (noisiest-quartile)
// variance ratio went from a 0.62x REGRESSION to a 1.6x IMPROVEMENT. See this crate's
// task report for the full measurement.
// ---------------------------------------------------------------------------------

/// Stream salts for three independent Cranley-Patterson rotations (one per
/// stratified quantity: pixel jitter X, pixel jitter Y, hero wavelength).
///
/// Each pixel's rotation for a given quantity is
/// `low_discrepancy_base2(hash_u32(pixel_index ^ SALT))` -- a per-pixel phase shift
/// applied to that quantity's own [`radical_inverse_base`] sequence (see this section's
/// header comment for why the three quantities use three DIFFERENT bases, not just
/// three rotations of the same one). This also finally decorrelates the pixel-jitter-X
/// draw from the hero draw: both used to read off the SAME `hash_u32(seed)` call (low
/// decimal digits for jitter, the full magnitude for hero) in the pre-Fix-4 code --
/// correlated in principle, harmless in practice, but free to fix while this file is
/// already being touched.
// `pub`, not `pub(crate)`: the production callers that must compute these rotations
// themselves (`apps/gemray-worker/src/render_core.rs`,
// `apps/diagram-gui/src/bridge/render_thread.rs` and `export_thread.rs`) live outside
// this crate.
pub const PIXEL_JITTER_X_ROTATION_STREAM: u32 = 0xA511_E9B3;
pub const PIXEL_JITTER_Y_ROTATION_STREAM: u32 = 0x63D8_1B23;
pub const HERO_WAVELENGTH_ROTATION_STREAM: u32 = 0x1B87_3593;

/// Base-2 van der Corput radical-inverse sequence.
///
/// `n` with its bits reversed, reinterpreted as a fraction of `2^32` in `[0, 1)`. Used
/// for pixel-jitter-X (the fast bit-reversal path for base 2 -- see
/// [`radical_inverse_base`] for the other two stratified quantities' bases, 3 and 5,
/// which need the general loop instead).
///
/// # Why this, not `hash_u32(n) as f32 / 4_294_967_295.0`
///
/// A hash is a good source of INDEPENDENT-looking randomness but a poor source of
/// STRATIFICATION: `N` independent uniform draws leave gaps and clumps, which is
/// exactly what shows up as speckle noise when `N` is the sample count of a single
/// pixel and the quantity being drawn dominates that pixel's variance. The van der
/// Corput sequence's first `N` terms, for ANY `N` (not just a power of two), are
/// provably far more evenly spread over `[0, 1)` than `N` independent uniform draws.
/// Crucially, "first `N` terms" needs no advance knowledge of `N`: term `n` depends only
/// on `n` itself, so this stays a pure function of the absolute sample index alone --
/// see this section's own header comment for why that purity is load-bearing for
/// distributed rendering, where a worker only ever sees an arbitrary, disjoint SLICE of
/// the absolute sample-index space and must still produce exactly the terms that
/// slice's absolute indices imply, regardless of which other slices other workers are
/// computing or in what order results arrive.
#[must_use]
#[inline]
pub fn low_discrepancy_base2(n: u32) -> f32 {
    // `2^32`, not `2^32 - 1`: the canonical van der Corput normalization (`bits /
    // b^digits`). No requirement to match the unrelated `hash_u32(..) /
    // 4_294_967_295.0` convention used elsewhere in this file for ordinary uniform
    // draws -- the two are different constructions, and `2^32` is exactly
    // representable so this loses no more precision than the `u32 -> f32` conversion
    // itself already does.
    (n.reverse_bits() as f32) / 4_294_967_296.0
}

/// Radical-inverse (van der Corput) sequence in an arbitrary prime `base`.
///
/// Writes `n` in base `base`, reflects its digits around the "radix point", giving a
/// fraction in `[0, 1)`. `base = 2` is handled faster (and identically) by
/// [`low_discrepancy_base2`] via bit reversal; this general loop is for `base = 3`
/// (pixel-jitter-Y) and `base = 5` (the hero wavelength) -- see this section's header
/// comment for why using different bases per stratified quantity, rather than one base
/// rotated three ways, is what actually delivers a variance win instead of a
/// regression. Same "pure function of `n` alone, no advance knowledge of the eventual
/// sample count needed" property as [`low_discrepancy_base2`], for the same reason.
///
/// Pure integer digit extraction (`n % base`, `n / base` -- exact, no rounding
/// ambiguity) feeding a running float accumulation via `f32::mul_add` (a true fused
/// multiply-add), matching `shaders/{spectral_transport,rng_equivalence}.wgsl`'s
/// identical port, which uses WGSL's `fma()` at the same call site -- the same
/// single-rounding contract this crate already relies on for `wrapped_hero_wavelengths`
/// and the CIE CMF fit, and (per `renderer::gpu::rng_check`'s own doc comment) already
/// budgeted for at Tier 2 rather than held to Tier 1's zero tolerance, since not every
/// GPU/driver fuses `fma()` into true hardware FMA.
#[must_use]
#[inline]
pub fn radical_inverse_base(n: u32, base: u32) -> f32 {
    let mut remaining = n;
    let mut val = 0.0f32;
    let mut inv_base = 1.0f32 / base as f32;
    while remaining > 0 {
        let digit = remaining % base;
        val = (digit as f32).mul_add(inv_base, val);
        inv_base /= base as f32;
        remaining /= base;
    }
    val
}

/// Cranley-Patterson rotation: shifts a `[0, 1)` low-discrepancy sample `x` by `offset`
/// (also `[0, 1)`), wrapping around the unit interval (`x + offset` reduced mod 1).
///
/// A toroidal shift of a low-discrepancy sequence is still low-discrepancy, so this is
/// the standard way to decorrelate several independent uses of the SAME sequence -- here,
/// one rotation per PIXEL, so neighbouring pixels don't all draw the identical
/// wavelength/jitter on their first sample. (Decorrelating the three DIFFERENT
/// stratified quantities from each other is [`radical_inverse_base`]'s job, via
/// distinct bases -- see this section's header comment.)
#[must_use]
#[inline]
pub fn cranley_patterson_rotate(x: f32, offset: f32) -> f32 {
    let sum = x + offset;
    sum - sum.floor()
}

// ---------------------------------------------------------------------------------
// Shared per-sample seed/jitter/hero formula.
//
// The exact construction below used to be hand-copied at four production call sites
// (`crates/gemray/src/renderer/gpu/hybrid.rs::cpu_sample_xyz`,
// `apps/diagram-gui/src/bridge/export_thread/batch.rs::render_batch`,
// `apps/diagram-gui/src/bridge/render_thread/scanline.rs::render_frame_scanlines`, and
// `apps/gemray-worker/src/render_core/mod.rs::trace_into`) -- each copy free to drift
// from the others with no compiler help. [`pixel_rotations`] and [`sample_draws`] are
// the ONE place this formula is written down now; every production call site computes
// its per-pixel rotations and per-sample draws through these two functions instead of
// re-deriving the arithmetic. Bit-identical to the pre-extraction formula by
// construction (same operations, same order) -- see `sample_draws_tests` below, which
// checks that literally, and see `gemray-net`'s own `partition_correctness.rs` for the
// same formula checked end-to-end against the real tracer.
// ---------------------------------------------------------------------------------

/// A pixel's three Cranley-Patterson rotation offsets.
///
/// One per stratified quantity (pixel-jitter-X, pixel-jitter-Y, hero wavelength) --
/// pure functions of the pixel index alone, hoisted out of the per-sample loop since
/// they don't vary per sample. Computed once per pixel via [`pixel_rotations`] and
/// reused for every sample of that pixel via [`sample_draws`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelRotations {
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub hero: f32,
}

/// Computes `pixel`'s three [`PixelRotations`] -- see that type's doc comment.
#[must_use]
#[inline]
pub fn pixel_rotations(pixel: u32) -> PixelRotations {
    PixelRotations {
        jitter_x: low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_X_ROTATION_STREAM)),
        jitter_y: low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_Y_ROTATION_STREAM)),
        hero: low_discrepancy_base2(hash_u32(pixel ^ HERO_WAVELENGTH_ROTATION_STREAM)),
    }
}

/// One `(pixel, sample_num)` draw: the RNG seed plus the stratified pixel-jitter-X/Y and
/// hero-wavelength values, ready to feed a `Camera::generate_ray` + `trace_spectral_ray*`
/// call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleDraws {
    pub seed: u32,
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub hero_rand: f32,
}

/// Computes the RNG seed and stratified draws for one `(pixel, sample_num)` pair.
///
/// `sample_num` is the ABSOLUTE sample index, never a batch-relative offset -- see this
/// module's "partition correctness" note at each call site for why that purity is
/// load-bearing for distributed rendering. `rot` is `pixel`'s already-computed
/// [`PixelRotations`] (see [`pixel_rotations`]).
#[must_use]
#[inline]
pub fn sample_draws(pixel: u32, sample_num: u32, rot: &PixelRotations) -> SampleDraws {
    let seed = hash_u32(pixel.wrapping_mul(0x9e37_79b9) ^ sample_num.wrapping_mul(0x85eb_ca6b));
    let jitter_x = cranley_patterson_rotate(low_discrepancy_base2(sample_num), rot.jitter_x) - 0.5;
    let jitter_y =
        cranley_patterson_rotate(radical_inverse_base(sample_num, 3), rot.jitter_y) - 0.5;
    let hero_rand = cranley_patterson_rotate(radical_inverse_base(sample_num, 5), rot.hero);
    SampleDraws {
        seed,
        jitter_x,
        jitter_y,
        hero_rand,
    }
}

#[cfg(test)]
mod sample_draws_tests {
    use super::*;

    /// The literal pre-extraction formula, copied verbatim (not calling
    /// [`pixel_rotations`]/[`sample_draws`] at all) so this test can prove the new
    /// functions reproduce it bit-for-bit rather than merely agreeing with themselves.
    fn old_formula(pixel: u32, sample_num: u32) -> (u32, f32, f32, f32) {
        let seed = hash_u32(pixel.wrapping_mul(0x9e37_79b9) ^ sample_num.wrapping_mul(0x85eb_ca6b));
        let rot_jx = low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_X_ROTATION_STREAM));
        let rot_jy = low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_Y_ROTATION_STREAM));
        let rot_hero = low_discrepancy_base2(hash_u32(pixel ^ HERO_WAVELENGTH_ROTATION_STREAM));
        let jx = cranley_patterson_rotate(low_discrepancy_base2(sample_num), rot_jx) - 0.5;
        let jy = cranley_patterson_rotate(radical_inverse_base(sample_num, 3), rot_jy) - 0.5;
        let hero_rand = cranley_patterson_rotate(radical_inverse_base(sample_num, 5), rot_hero);
        (seed, jx, jy, hero_rand)
    }

    /// Checked over a few thousand `(pixel, sample_num)` pairs: [`pixel_rotations`] +
    /// [`sample_draws`] must reproduce [`old_formula`] bit-for-bit, every time -- not
    /// approximately, since every production call site depends on this being an exact
    /// drop-in replacement (see this section's header comment).
    #[test]
    fn sample_draws_matches_the_literal_old_formula_bit_for_bit() {
        let mut checked = 0u32;
        for pixel in 0..4000u32 {
            let rot = pixel_rotations(pixel);
            for sample_num in 0..3u32 {
                let (seed, jx, jy, hero) = old_formula(pixel, sample_num);
                let draws = sample_draws(pixel, sample_num, &rot);
                assert_eq!(draws.seed, seed, "pixel={pixel} sample={sample_num}");
                assert_eq!(draws.jitter_x, jx, "pixel={pixel} sample={sample_num}");
                assert_eq!(draws.jitter_y, jy, "pixel={pixel} sample={sample_num}");
                assert_eq!(draws.hero_rand, hero, "pixel={pixel} sample={sample_num}");
                checked += 1;
            }
        }
        assert!(
            checked >= 3000,
            "sanity: should have checked a few thousand pairs"
        );
    }
}

#[cfg(test)]
mod rng_decorrelation_tests {
    use super::*;

    /// Fix E: the old `(rng_seed + bounce*7919) % 1000` progression only ever produced
    /// 1000 distinct quantized values and stepped by a fixed +919 (== -81 mod 1000)
    /// every bounce -- i.e. every draw along a path was a deterministic function of the
    /// previous one, not an independent sample. The hash-based replacement should
    /// produce a much larger spread of distinct values across many seeds.
    #[test]
    fn hashed_branch_draw_is_not_quantized_to_1000_values() {
        // Mirrors the production draw at the first bounce.
        let bounce = 0u32;
        let mut values = std::collections::HashSet::new();
        for seed in 0..5000u32 {
            let v = hash_u32(seed ^ hash_u32(bounce ^ FRESNEL_BRANCH_STREAM));
            values.insert(v);
        }
        assert!(
            values.len() > 4900,
            "hashed draw should yield close to 5000 distinct values across 5000 seeds (got {})",
            values.len()
        );
    }

    /// Fix D and Fix E specify distinct decorrelated streams for the Fresnel branch
    /// decision and the Russian-roulette draw, rather than reusing the same random
    /// value for both. Confirm the two salted streams diverge for the same
    /// (seed, bounce) pair across many samples.
    #[test]
    fn fresnel_and_russian_roulette_streams_are_decorrelated() {
        let mut agreements = 0u32;
        let trials = 2000u32;
        for seed in 0..trials {
            let fresnel = hash_u32(seed ^ hash_u32(3u32 ^ FRESNEL_BRANCH_STREAM));
            let rr = hash_u32(seed ^ hash_u32(3u32 ^ RUSSIAN_ROULETTE_STREAM));
            if fresnel == rr {
                agreements += 1;
            }
        }
        assert!(
            agreements == 0,
            "the Fresnel-branch and Russian-roulette streams must not collide across {trials} trials (got {agreements} collisions)"
        );
    }

    /// Fix D: after bounce 4, survival must be a proper weighted Russian-roulette test
    /// (survive w.p. q, then divide by q) rather than a hard cutoff that silently
    /// discards energy. This exercises the exact q/draw construction used in
    /// `trace_spectral_ray` and checks the compensated estimator is unbiased in
    /// expectation: E[`survive_indicator` / q] == 1 for a fixed throughput level.
    #[test]
    fn russian_roulette_survival_is_unbiased_in_expectation() {
        let q = 0.3f32; // an arbitrary throughput level within the [0.05, 1.0] clamp range
        let trials = 200_000u32;
        let mut total = 0.0f32;
        for bounce in 0..trials {
            let rr_rand = (hash_u32(bounce ^ hash_u32(bounce ^ RUSSIAN_ROULETTE_STREAM)) as f32)
                / 4_294_967_295.0;
            if rr_rand <= q {
                total += 1.0 / q;
            }
        }
        let mean = total / trials as f32;
        assert!(
            (mean - 1.0).abs() < 0.02,
            "weighted Russian-roulette survival should be unbiased (E[indicator/q] ~= 1.0), got {mean}"
        );
    }
}

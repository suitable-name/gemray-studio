//! Shared ULP-distance helper for every Phase-1 GPU self-test.
//!
//! Originally written to mirror `renderer::gpu::rng_check`'s private `ulp_distance`
//! (Phase 0) exactly, duplicated rather than factored out from that already-shipped
//! module so this Phase-1 work touched no Phase-0 file. That Phase-0 helper does a bare
//! signed difference of the raw bit patterns with no sign handling at all, which is only
//! correct for same-sign pairs -- see `to_ordered`'s doc comment below for why this
//! module no longer mirrors it. See `rng_check`'s own doc comment for the general
//! rationale on why ULP (not a relative epsilon) is the right unit here: it is "how
//! many representable `f32` values apart are these two results", which stays meaningful
//! near zero and near large magnitudes alike, unlike a relative tolerance.

/// Distance, in ULP (representable `f32` steps), between two floats of any sign,
/// including a pair that straddles zero. Several Phase-1 quantities this is used on
/// (ray directions and normal components in `[-1, 1]`) legitimately cross zero, so
/// sign and negative-zero are handled explicitly via [`to_ordered`] rather than assumed
/// away.
#[must_use]
pub fn ulp_distance(a: f32, b: f32) -> u32 {
    // Bit-pattern equality, not `==`: this is meant as "skip the ordering work when
    // the two are already identical," and `to_bits()` states that literally, but the
    // two operators disagree on NaN and on +0.0/-0.0, so the substitution is only
    // sound because `to_ordered` below is checked to already agree with `to_bits`
    // equality on both:
    //   - `+0.0`/`-0.0`: `to_bits()` treats them as distinct, so this fast path no
    //     longer fires for that pair -- but `to_ordered` maps both to the same
    //     ordered value by construction (see its doc comment), so falling through
    //     still yields 0, matching `==`'s answer for this pair.
    //   - NaN: `to_bits()` (unlike `==`) is true for a bit-identical NaN pair, so this
    //     fast path now fires where `==` never would -- but `to_ordered` is a pure
    //     function of the bit pattern, so identical bits already forced `ai == bi`
    //     (0 ULP) on the old fall-through path too. Two *different* NaN bit patterns
    //     still fall through exactly as before.
    // So the two implementations agree on every input; this one just says "compare
    // the bits" where the old one said "compare the floats" for a function whose
    // entire job is comparing bits.
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    let ai = to_ordered(a);
    let bi = to_ordered(b);
    ai.abs_diff(bi) as u32
}

/// Whether `cpu`/`gpu` agree closely enough to not be a bug, under a hybrid rule: EITHER
/// their ULP distance is within `budget`, OR their absolute difference is under
/// `abs_floor`.
///
/// The second clause exists because ULP is a poor metric exactly where a value legitimately
/// crosses (or nearly crosses) zero, or where its magnitude is so small it is
/// photometrically/physically meaningless. Two concrete cases this workspace's own
/// measurements hit: `sin(x)` evaluated at `x` within 1 ULP of a multiple of π, where
/// the CPU and GPU trig implementations round the argument-reduced result to opposite
/// sides of exactly `0.0` -- an absolute difference of ~1e-7 registers as billions of
/// ULP simply because ULP distance through `0.0` is astronomically large relative to a
/// value's own magnitude near zero; and `cie_1931_cmf` evaluated deep in a Gaussian
/// lobe's tail (e.g. ~1e-24), where the function's *relative* precision is inherently
/// poor (a tiny absolute difference in the exponent term is amplified by the
/// exponential itself) but the *absolute* value is many orders of magnitude below
/// anything a renderer could ever visibly distinguish. `abs_floor` must be chosen well
/// below the magnitude any real algebra bug would produce (Phase 0's calibration: a
/// deliberately wrong formula measured at 8,552,444 ULP and, correspondingly, an
/// absolute/relative error many orders of magnitude above a sensible `abs_floor`) --
/// each call site documents its own choice.
#[must_use]
pub fn within_tolerance(cpu: f32, gpu: f32, budget: u32, abs_floor: f32) -> bool {
    ulp_distance(cpu, gpu) <= budget || (cpu - gpu).abs() < abs_floor
}

/// Maps an `f32`'s bit pattern to a monotonically-ordered `i64` (standard trick: for
/// non-negative floats the raw bit pattern, reinterpreted as signed, is already
/// monotonic; for negative floats, reflecting through `i32::MIN` restores monotonicity
/// across the sign boundary). Handles negative operands and negative zero correctly,
/// unlike a bare `to_bits()` difference.
///
/// Adopts the convention that `+0.0` and `-0.0` map to the *same* ordered value (0 ULP
/// apart), not adjacent values (1 ULP apart): both are "zero", and `a == b` already
/// special-cases them equal in [`ulp_distance`] as plain floats, so the ordering below
/// is chosen to agree with that rather than introduce a seam exactly at zero.
///
/// # The bug this replaced
///
/// The previous implementation was `let bits = i64::from(x.to_bits());` followed by
/// `if bits >= 0 { bits } else { bits ^ 0x7FFF_FFFF }`. Because `x.to_bits()` is a
/// `u32`, `i64::from(u32)` zero-extends -- the result is *always* in `0..=u32::MAX` and
/// therefore always `>= 0` as an `i64`, no matter the float's sign. The negative branch
/// was unreachable dead code, so every call silently degenerated to the raw unsigned
/// bit pattern with no sign handling at all -- identical (bit-for-bit) to
/// `rng_check`'s private `ulp_distance`, which never attempted the flip in the first
/// place. That raw-bit-pattern order is monotonic with the float's value *within*
/// same-sign ranges (positive bit patterns increase with the float; negative bit
/// patterns move opposite the float, but still step 1-for-1, so an abs-diff over two
/// same-sign floats still counts the right number of representable steps). It breaks
/// only when the pair straddles the sign boundary: the raw patterns jump from
/// `0x7FFF_FFFF` (largest finite positive) to `0x8000_0000` (negative zero), a gap of
/// about 2^31 that has nothing to do with how many representable floats actually lie
/// between the two values.
fn to_ordered(x: f32) -> i64 {
    let bits = x.to_bits() as i32;
    let ordered = if bits < 0 {
        i32::MIN.wrapping_sub(bits)
    } else {
        bits
    };
    i64::from(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_values_are_zero_ulp_apart() {
        assert_eq!(ulp_distance(1.0, 1.0), 0);
        assert_eq!(ulp_distance(0.0, 0.0), 0);
        assert_eq!(ulp_distance(0.0, -0.0), 0);
    }

    #[test]
    fn adjacent_representable_floats_are_one_ulp_apart() {
        let a = 1.0f32;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance(a, b), 1);
        assert_eq!(ulp_distance(b, a), 1);
    }

    #[test]
    fn crossing_zero_counts_the_full_step_distance() {
        let a = f32::from_bits(1); // smallest positive subnormal
        let b = -a;
        // one step to +0.0 in each direction -> 2 total (matches monotonic ordering
        // through 0.0 rather than treating +0 and -0 as unrelated bit patterns).
        assert_eq!(ulp_distance(a, b), 2);
    }

    #[test]
    fn positive_zero_and_negative_zero_are_zero_ulp_apart() {
        // Convention: +0.0 and -0.0 are the same point (0 ULP), not adjacent (1 ULP).
        // This matches `a == b`'s own treatment of the pair (IEEE-754 defines
        // +0.0 == -0.0), and it's what makes `crossing_zero_counts_the_full_step_distance`
        // above land on exactly 2 rather than 3 -- a seam at zero would double-count the
        // boundary itself as an extra step.
        assert_eq!(ulp_distance(0.0f32, -0.0f32), 0);
        assert_eq!(ulp_distance(-0.0f32, 0.0f32), 0);
    }

    #[test]
    fn same_sign_positive_pairs_are_unaffected_by_the_fix() {
        let a = 100.0f32;
        let b = f32::from_bits(a.to_bits() + 5);
        assert_eq!(ulp_distance(a, b), 5);
        assert_eq!(ulp_distance(b, a), 5);
    }

    #[test]
    fn same_sign_negative_pairs_are_unaffected_by_the_fix() {
        let a = -100.0f32;
        let b = f32::from_bits(a.to_bits() + 5); // more negative than `a` by 5 steps
        assert_eq!(ulp_distance(a, b), 5);
        assert_eq!(ulp_distance(b, a), 5);
    }

    #[test]
    fn opposite_sign_pairs_away_from_zero_count_the_full_path_through_zero() {
        // -1.0 to 1.0: every representable step from -1.0 up through -0.0/+0.0 to 1.0.
        // f32 has 2^23 mantissa steps per binade, and [-1,1) covers the 0.5..1 binade
        // (exponent -1) plus the subnormal+normal ramp down to zero; easiest to state
        // this in terms of to_ordered-equivalent step counts via a known-good pair:
        // count the two contiguous ranges [0, 1.0] and [-1.0, -0.0] and add them.
        let steps_zero_to_one = ulp_distance(0.0f32, 1.0f32);
        let steps_neg_one_to_zero = ulp_distance(-1.0f32, -0.0f32);
        let combined = steps_zero_to_one + steps_neg_one_to_zero;
        assert_eq!(ulp_distance(-1.0f32, 1.0f32), combined);
        assert_eq!(ulp_distance(1.0f32, -1.0f32), combined);
        // And independently, against the raw bit patterns directly: both 1.0 and -1.0
        // have the same magnitude bit pattern (0x3F800000), so the ordered distance is
        // exactly twice that value.
        assert_eq!(combined, 2 * 1.0f32.to_bits());
    }

    #[test]
    fn denormals_straddling_zero_step_correctly() {
        // A few steps into the subnormal range on each side.
        let a = f32::from_bits(3); // 3rd smallest positive subnormal
        let b = f32::from_bits(0x8000_0002); // 2nd smallest negative subnormal
        // 3 steps from a down to +0.0, 2 steps from -0.0 down to b -> 5 total.
        assert_eq!(ulp_distance(a, b), 5);
        assert_eq!(ulp_distance(b, a), 5);
    }

    #[test]
    fn denormals_same_sign_step_correctly() {
        let a = f32::from_bits(2);
        let b = f32::from_bits(9);
        assert_eq!(ulp_distance(a, b), 7);
        let na = f32::from_bits(0x8000_0002);
        let nb = f32::from_bits(0x8000_0009);
        assert_eq!(ulp_distance(na, nb), 7);
    }

    #[test]
    fn values_near_f32_max_step_correctly_same_sign() {
        let max = f32::MAX;
        let one_below = f32::from_bits(max.to_bits() - 1);
        assert_eq!(ulp_distance(max, one_below), 1);

        let neg_max = -f32::MAX;
        let one_above = f32::from_bits(neg_max.to_bits() - 1); // one step toward zero
        assert_eq!(ulp_distance(neg_max, one_above), 1);
    }

    #[test]
    fn values_near_f32_max_step_correctly_opposite_sign() {
        // f32::MAX and -f32::MAX straddle zero; distance is the full span (twice
        // f32::MAX's bit pattern, since +0.0/-0.0 coincide at the ordered origin), which
        // must not overflow the u32 return value -- comfortably under u32::MAX here.
        let d = ulp_distance(f32::MAX, -f32::MAX);
        let expected = 2u32 * f32::MAX.to_bits();
        assert_eq!(d, expected);
        assert_eq!(ulp_distance(-f32::MAX, f32::MAX), d);
    }
}

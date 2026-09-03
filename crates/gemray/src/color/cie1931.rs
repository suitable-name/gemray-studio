//! CIE 1931 2° Standard Observer Color Matching Functions (CMFs), as an analytic fit.
//!
//! This is the single source of truth for the CMF fit: [`optics::raytracer`](crate::optics::raytracer)
//! delegates to [`cie_1931_cmf`] rather than keeping its own copy, so there is exactly
//! one place where the lobe constants live.

/// Piecewise-asymmetric Gaussian lobe: a different sigma is used on the rising
/// (x < mu) side of the peak than on the falling (x >= mu) side. This asymmetry is
/// what the Wyman/Sloan/Shirley fit actually relies on; collapsing it to a single
/// symmetric sigma (as a naive re-implementation might) distorts the lobe shapes,
/// most severely the z-bar 437 nm lobe's falling side.
#[inline]
fn g(x: f32, mu: f32, sigma_lo: f32, sigma_hi: f32) -> f32 {
    let sigma = if x < mu { sigma_lo } else { sigma_hi };
    let t = (x - mu) / sigma;
    (-0.5 * t * t).exp()
}

/// Wyman, Sloan, Shirley (2013) multi-lobe analytic fit to the CIE 1931 2° Standard
/// Observer Color Matching Functions.
///
/// Uses piecewise-asymmetric Gaussian lobes (separate sigma below and above each
/// lobe's peak) -- deliberately, per [`g`]'s doc comment. Do NOT "simplify" this back
/// to a single symmetric sigma per lobe; that was a previous (now-fixed) bug here that
/// visibly skewed computed chromaticities -- e.g. equal-energy white should normalize
/// to chromaticity ~= (0.33311, 0.33359), and the old symmetric fit instead produced
/// ~= (0.3413, 0.3644).
#[must_use]
pub fn cie_1931_cmf(lambda_nm: f32) -> [f32; 3] {
    let l = lambda_nm;

    let x = 0.065f32.mul_add(
        -g(l, 501.1, 20.4, 26.2),
        1.056f32.mul_add(g(l, 599.8, 37.9, 31.0), 0.362 * g(l, 442.0, 16.0, 26.7)),
    );

    let y = 0.286f32.mul_add(g(l, 530.9, 16.3, 31.1), 0.821 * g(l, 568.8, 46.9, 40.5));

    let z = 0.681f32.mul_add(g(l, 459.0, 26.0, 13.8), 1.217 * g(l, 437.0, 11.8, 36.0));

    [x.max(0.0), y.max(0.0), z.max(0.0)]
}

/// Batched form of [`g`] for one lobe (fixed `mu`/`sigma_lo`/`sigma_hi`) over 8
/// wavelengths at once: computes all 8 lanes' `-0.5*t*t` exponent arguments with the
/// exact per-lane arithmetic [`g`] uses, then evaluates all 8 exponentials in a single
/// [`crate::simd::exp_f32x8`] call instead of 8 separate scalar `f32::exp` calls.
#[inline]
fn g_x8(lambdas: &[f32; 8], mu: f32, sigma_lo: f32, sigma_hi: f32) -> [f32; 8] {
    let mut args = [0f32; 8];
    for (k, &l) in lambdas.iter().enumerate() {
        let sigma = if l < mu { sigma_lo } else { sigma_hi };
        let t = (l - mu) / sigma;
        args[k] = -0.5 * t * t;
    }
    crate::simd::exp_f32x8(args)
}

/// Batched form of [`cie_1931_cmf`] over 8 wavelengths at once.
///
/// Uses [`crate::simd::exp_f32x8`] to batch each of the fit's 7 lobes' exponentials
/// across all 8 channels (7 vector `exp_f32x8` calls total, versus 56 scalar `f32::exp`
/// calls -- one per lobe per channel -- for 8 separate [`cie_1931_cmf`] calls).
///
/// # Deliberate ULP-level re-baseline
///
/// This is **not** bit-identical to calling [`cie_1931_cmf`] 8 times: every
/// non-exponential operation (lobe-side sigma selection, the `t` and `-0.5*t*t`
/// exponent-argument computation, the `mul_add` lobe combination, and the final
/// `.max(0.0)`) follows [`cie_1931_cmf`]'s exact per-wavelength operation order, so the
/// ONLY numeric difference is [`crate::simd::exp_f32x8`] vs libm's `f32::exp` -- a few
/// ULP, same as every other `exp_f32x8` call site (see `crate::simd`'s module docs and
/// [`crate::simd::exp_f32x8`]'s own docs for the determinism contract this follows).
///
/// [`cie_1931_cmf`] itself is untouched and remains the CPU-vs-GPU
/// ULP-equivalence-checked primitive; this function is purely additive.
#[must_use]
pub fn cie_1931_cmf_x8(lambdas: &[f32; 8]) -> [[f32; 3]; 8] {
    let lobe_x0 = g_x8(lambdas, 501.1, 20.4, 26.2);
    let lobe_x1 = g_x8(lambdas, 599.8, 37.9, 31.0);
    let lobe_x2 = g_x8(lambdas, 442.0, 16.0, 26.7);
    let lobe_y0 = g_x8(lambdas, 530.9, 16.3, 31.1);
    let lobe_y1 = g_x8(lambdas, 568.8, 46.9, 40.5);
    let lobe_z0 = g_x8(lambdas, 459.0, 26.0, 13.8);
    let lobe_z1 = g_x8(lambdas, 437.0, 11.8, 36.0);

    let mut out = [[0f32; 3]; 8];
    for k in 0..8 {
        let x = 0.065f32.mul_add(
            -lobe_x0[k],
            1.056f32.mul_add(lobe_x1[k], 0.362 * lobe_x2[k]),
        );
        let y = 0.286f32.mul_add(lobe_y0[k], 0.821 * lobe_y1[k]);
        let z = 0.681f32.mul_add(lobe_z0[k], 1.217 * lobe_z1[k]);
        out[k] = [x.max(0.0), y.max(0.0), z.max(0.0)];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`cie_1931_cmf_x8`] must agree with 8 separate [`cie_1931_cmf`] calls to a tight
    /// relative tolerance -- the documented ULP-level `exp_f32x8`-vs-libm gap, not more.
    #[test]
    fn cmf_x8_matches_scalar_closely() {
        let mut state = 99u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (((state >> 33) as f32) / (u32::MAX as f32)).mul_add(700.0 - 380.0, 380.0)
        };
        for _round in 0..200 {
            let lambdas = [
                next(),
                next(),
                next(),
                next(),
                next(),
                next(),
                next(),
                next(),
            ];
            let batched = cie_1931_cmf_x8(&lambdas);
            for (k, &l) in lambdas.iter().enumerate() {
                let scalar = cie_1931_cmf(l);
                for c in 0..3 {
                    let a = f64::from(batched[k][c]);
                    let b = f64::from(scalar[c]);
                    let tol = 1e-5 * b.abs().max(1e-6);
                    assert!(
                        (a - b).abs() <= tol,
                        "lambda={l} channel={c}: batched={a} scalar={b} diff={}",
                        (a - b).abs()
                    );
                }
            }
        }
    }
}

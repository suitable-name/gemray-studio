//! Piecewise-constant 2D importance sampling (the `Distribution1D`/`Distribution2D`
//! machinery popularised by PBRT), used by [`super::EnvironmentMap`] to draw
//! directions proportional to an equirectangular image's (solid-angle-weighted)
//! luminance.
//!
//! This module is pure math over `f32` slices -- it knows nothing about images, HDR
//! decoding, or 3D directions. That separation is deliberate: it is the piece most
//! worth unit-testing in isolation (a 1D/2D piecewise-constant sampler is a
//! self-contained, easily-checked algorithm), independent of the equirectangular
//! direction mapping and `sin(theta)` weighting that [`super::EnvironmentMap`] layers on
//! top.

/// A piecewise-constant probability distribution over `[0, 1)`, built from `n` bucket
/// weights, supporting inverse-CDF sampling and pdf lookup.
///
/// # Degenerate input
///
/// If every weight is exactly `0.0` (or the slice is empty), there is no
/// luminance-proportional distribution to speak of -- [`Self::new`] falls back to a
/// **uniform** distribution over `[0, 1)` rather than dividing by a zero integral. This
/// is what keeps an all-black environment map sampling well-defined (uniform directions,
/// pdf `1.0` in this measure) instead of producing NaN.
#[derive(Debug, Clone)]
pub struct Distribution1D {
    /// The bucket weights as supplied (length `n`, `n >= 1`).
    func: Vec<f32>,
    /// Cumulative distribution function, length `n + 1`, `cdf[0] == 0.0`,
    /// `cdf[n] == 1.0` (after normalization, or after the uniform fallback).
    cdf: Vec<f32>,
    /// Mean bucket weight (`sum(func) / n`) *before* normalization. Zero exactly when
    /// the uniform fallback was taken.
    func_int: f32,
}

impl Distribution1D {
    /// Builds a distribution from bucket weights. `func` must be non-empty; a
    /// single-bucket slice is valid (see the module docs' degenerate-input coverage).
    /// Negative weights are treated as `0.0` (radiance should never be negative, but a
    /// caller-supplied HDR texel with a decoding artifact should not panic or poison the
    /// whole distribution).
    pub fn new(func: Vec<f32>) -> Self {
        let n = func.len().max(1);
        let mut cdf = vec![0.0f32; n + 1];
        for i in 1..=n {
            let w = func.get(i - 1).copied().unwrap_or(0.0).max(0.0);
            cdf[i] = w.mul_add(1.0 / n as f32, cdf[i - 1]);
        }
        let func_int = cdf[n];
        if func_int > 0.0 {
            for c in &mut cdf[1..=n] {
                *c /= func_int;
            }
            // Guard against float roundoff leaving the top short of 1.0 -- FindInterval
            // below relies on `cdf[n] >= u` for every `u` drawn from `[0, 1)`.
            cdf[n] = 1.0;
        } else {
            for (i, c) in cdf.iter_mut().enumerate().take(n + 1).skip(1) {
                *c = i as f32 / n as f32;
            }
        }
        Self {
            func,
            cdf,
            func_int,
        }
    }

    fn n(&self) -> usize {
        self.func.len().max(1)
    }

    /// Largest `i` such that `self.cdf[i] <= u`, clamped to a valid bucket index
    /// `[0, n-1]`. Standard binary search over a CDF (PBRT's `FindInterval`).
    fn find_bucket(&self, u: f32) -> usize {
        let cdf = &self.cdf;
        let mut first = 0usize;
        let mut len = cdf.len();
        while len > 0 {
            let half = len / 2;
            let middle = first + half;
            if cdf[middle] <= u {
                first = middle + 1;
                len -= half + 1;
            } else {
                len = half;
            }
        }
        first.saturating_sub(1).min(self.n() - 1)
    }

    /// The pdf (w.r.t. the `[0, 1)` measure, i.e. `integral of pdf over [0,1) == 1`) of
    /// bucket `offset`.
    fn bucket_pdf(&self, offset: usize) -> f32 {
        if self.func_int > 0.0 {
            self.func[offset].max(0.0) / self.func_int
        } else {
            // Uniform fallback: every bucket has width 1/n and equal probability, so
            // the density is exactly 1.0 everywhere.
            1.0
        }
    }

    /// Draws a continuous sample in `[0, 1)` via inverse-CDF search on `u`. Returns
    /// `(sample, pdf, bucket_index)`; `pdf` is w.r.t. the `[0, 1)` measure.
    pub fn sample_continuous(&self, u: f32) -> (f32, f32, usize) {
        let u = u.clamp(0.0, 0.999_999_94);
        let offset = self.find_bucket(u);
        let span = self.cdf[offset + 1] - self.cdf[offset];
        let du = if span > 0.0 {
            (u - self.cdf[offset]) / span
        } else {
            0.0
        };
        let n = self.n();
        let sample = ((offset as f32 + du) / n as f32).clamp(0.0, 0.999_999_94);
        (sample, self.bucket_pdf(offset), offset)
    }

    /// The pdf (w.r.t. the `[0, 1)` measure) of a given point `x` in `[0, 1)`, computed
    /// independently of [`Self::sample_continuous`] -- used by callers that need to
    /// evaluate the density of an externally-supplied direction (e.g. for MIS) rather
    /// than one this distribution itself drew.
    pub fn pdf(&self, x: f32) -> f32 {
        let n = self.n();
        let offset = ((x.clamp(0.0, 0.999_999_94)) * n as f32) as usize;
        self.bucket_pdf(offset.min(n - 1))
    }
}

/// A piecewise-constant probability distribution over the unit square `[0,1) x [0,1)`,
/// built as a marginal distribution over rows plus one conditional distribution per row
/// (the standard two-stage construction: pick a row proportional to its total weight,
/// then pick a column within that row proportional to the row's weights).
///
/// The caller (`env_map`) is responsible for whatever weighting the `func` values
/// already encode -- this type has no notion of "row" beyond array layout, so the
/// `sin(theta)` solid-angle correction for an equirectangular image must already be
/// baked into `func` before it reaches [`Self::new`].
#[derive(Debug, Clone)]
pub struct Distribution2D {
    conditional: Vec<Distribution1D>,
    marginal: Distribution1D,
    height: usize,
}

impl Distribution2D {
    /// `func` is row-major, length `width * height`. Both dimensions must be `>= 1`.
    pub fn new(func: &[f32], width: usize, height: usize) -> Self {
        debug_assert_eq!(func.len(), width * height);
        let height = height.max(1);
        let width = width.max(1);
        let mut conditional = Vec::with_capacity(height);
        let mut marginal_func = Vec::with_capacity(height);
        for y in 0..height {
            let start = y * width;
            let row: Vec<f32> = func
                .get(start..start + width)
                .map_or_else(|| vec![0.0; width], <[f32]>::to_vec);
            let dist = Distribution1D::new(row);
            marginal_func.push(dist.func_int);
            conditional.push(dist);
        }
        let marginal = Distribution1D::new(marginal_func);
        Self {
            conditional,
            marginal,
            height,
        }
    }

    /// Draws `(u, v, pdf)` where `u` picks the column and `v` the row, both in `[0, 1)`,
    /// and `pdf` is w.r.t. the unit-square `du dv` measure (`integral of pdf over the
    /// unit square == 1`). `u1` drives the row (marginal) choice, `u0` the column
    /// (conditional) choice -- matching the PBRT convention this is modelled on.
    pub fn sample(&self, u0: f32, u1: f32) -> (f32, f32, f32) {
        let (v, pdf_v, row) = self.marginal.sample_continuous(u1);
        let (u, pdf_u, _col) = self.conditional[row].sample_continuous(u0);
        (u, v, pdf_u * pdf_v)
    }

    /// The pdf (w.r.t. the unit-square measure) at an arbitrary `(u, v)`, independent of
    /// [`Self::sample`].
    pub fn pdf(&self, u: f32, v: f32) -> f32 {
        let row = ((v.clamp(0.0, 0.999_999_94)) * self.height as f32) as usize;
        let row = row.min(self.height - 1);
        self.marginal.pdf(v) * self.conditional[row].pdf(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_weights_give_uniform_pdf_and_samples() {
        let dist = Distribution1D::new(vec![1.0; 8]);
        for i in 0..=20 {
            let u = i as f32 / 20.0;
            let (_, pdf, _) = dist.sample_continuous(u.min(0.999_999));
            assert!(
                (pdf - 1.0).abs() < 1e-5,
                "uniform bucket weights must give pdf==1, got {pdf}"
            );
        }
    }

    #[test]
    fn all_zero_weights_fall_back_to_uniform_without_nan() {
        let dist = Distribution1D::new(vec![0.0; 4]);
        for i in 0..10 {
            let u = i as f32 / 10.0;
            let (sample, pdf, _) = dist.sample_continuous(u);
            assert!(sample.is_finite() && !sample.is_nan());
            assert!((pdf - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn single_bucket_distribution_does_not_panic() {
        let dist = Distribution1D::new(vec![3.5]);
        let (sample, pdf, offset) = dist.sample_continuous(0.5);
        assert_eq!(offset, 0);
        assert!((0.0..1.0).contains(&sample));
        assert!((pdf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dominant_bucket_is_sampled_almost_always() {
        let mut weights = vec![0.0f32; 16];
        weights[10] = 1000.0;
        let dist = Distribution1D::new(weights);
        let hits = (0..1000)
            .filter(|&i| {
                let u = (i as f32 + 0.5) / 1000.0;
                dist.sample_continuous(u).2 == 10
            })
            .count();
        assert!(
            hits >= 990,
            "expected the dominant bucket to be hit almost every time, got {hits}/1000"
        );
    }

    #[test]
    fn distribution_2d_pdf_matches_sample_pdf() {
        let width = 8;
        let height = 4;
        let func: Vec<f32> = (0..width * height).map(|i| i as f32 + 1.0).collect();
        let dist2d = Distribution2D::new(&func, width, height);
        let (u, v, pdf_sampled) = dist2d.sample(0.37, 0.81);
        let pdf_looked_up = dist2d.pdf(u, v);
        assert!(
            (pdf_sampled - pdf_looked_up).abs() < 1e-4,
            "sample()'s pdf and pdf() must agree at the sampled point"
        );
    }
}

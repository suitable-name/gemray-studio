//! RGB -> spectral radiance reconstruction.
//!
//! `gemray`'s tracer is a hero-wavelength spectral renderer: at every ray-environment
//! interaction it needs radiance at a single continuous wavelength per channel (see
//! `optics::raytracer::sample_studio_environment`'s `lambda_nm` parameter), not an RGB
//! triple. An HDR environment map, however, is stored as RGB texels. Something has to
//! bridge the two -- this module is that bridge, kept to one small, explicitly
//! replaceable function per the task's request.
//!
//! # Method chosen: three overlapping Gaussian "primary" bumps
//!
//! [`rgb_to_spectral_radiance`] treats the RGB triple as weights on three fixed,
//! smooth, strictly non-negative basis functions of wavelength -- asymmetric Gaussian
//! bumps centred near a red, green, and blue primary (615 nm / 545 nm / 465 nm), each
//! with a different rise/fall width so the composite stays smooth without becoming a
//! flat plateau. The spectral radiance at `lambda_nm` is just the weighted sum
//! `r*R(lambda) + g*G(lambda) + b*B(lambda)`.
//!
//! This is deliberately the simplest defensible option, not the most accurate one:
//!
//! - **It is not a real spectral upsampling algorithm.** The physically-grounded
//!   approach (Jakob & Hanika 2019, "A Low-Dimensional Function Space for Efficient
//!   Spectral Upsampling") fits a smooth sigmoid-of-quadratic per RGB triple against the
//!   working colour space's primaries and a chosen illuminant, guaranteeing the
//!   reconstructed spectrum re-integrates (via the CIE CMFs) back to very close to the
//!   original RGB. That requires either a per-query numerical solve or a precomputed 3D
//!   lookup table -- real engineering effort disproportionate to what an *environment
//!   background* term needs relative to the gemstone's own dispersion/absorption
//!   physics, which is where spectral accuracy actually matters for this renderer.
//! - **No round-trip guarantee.** Integrating this reconstructed spectrum against the
//!   CIE 1931 CMFs and converting back to RGB will not exactly reproduce the input RGB
//!   -- expect visible desaturation on highly saturated inputs (pure spectral colours,
//!   e.g. a saturated laser-like green) and a mild colour shift on strongly chromatic
//!   HDR content.
//! - **No metamerism.** Two different real-world spectra that happen to photograph to
//!   the same RGB will reconstruct to the *same* approximate spectrum here, whereas a
//!   real environment might have (say) a fluorescent light with sharp emission lines
//!   invisible to this model. For a gemstone dispersion renderer this specifically means
//!   the fire produced by an HDR-sourced environment will be smoother/less spiky than a
//!   photographically identical scene lit by a narrow-band real source.
//! - **Not designed for RGB components above 1.0** (HDR highlights): the basis functions
//!   scale linearly with each component, so an extreme highlight just scales the bump
//!   height proportionally -- reasonable for a smooth stand-in, but it will not reproduce
//!   a genuinely narrow-band bright source (e.g. a small sun disc) as anything other than
//!   a broad, smooth peak.
//!
//! Because every call funnels through this one function, swapping in a real
//! Jakob-Hanika upsampler later (most naturally: precompute per-texel polynomial
//! coefficients once at load time in [`super::EnvironmentMap`], then evaluate
//! the sigmoid here) is a localized change.

/// Converts a linear RGB radiance (or reflectance-like) triple into an approximate
/// spectral radiance value at `lambda_nm` (nanometres). See the module docs for the
/// method and its limitations.
///
/// Negative input components are treated as `0.0` (radiance should never be negative,
/// but a caller should not have to pre-clamp). The result is always finite and
/// non-negative for finite, non-negative-clamped input.
#[must_use]
pub fn rgb_to_spectral_radiance(rgb: [f32; 3], lambda_nm: f32) -> f32 {
    let r = rgb[0].max(0.0);
    let g = rgb[1].max(0.0);
    let b = rgb[2].max(0.0);

    r.mul_add(
        asymmetric_gaussian(lambda_nm, 615.0, 45.0, 65.0),
        g.mul_add(
            asymmetric_gaussian(lambda_nm, 545.0, 45.0, 45.0),
            b * asymmetric_gaussian(lambda_nm, 465.0, 40.0, 45.0),
        ),
    )
}

/// A Gaussian bump centred at `mu`, using `sigma_lo` below the peak and `sigma_hi` above
/// it -- the asymmetry is what keeps three overlapping bumps from collapsing into an
/// almost-flat sum across the visible range while still individually staying smooth.
#[inline]
fn asymmetric_gaussian(x: f32, mu: f32, sigma_lo: f32, sigma_hi: f32) -> f32 {
    let sigma = if x < mu { sigma_lo } else { sigma_hi };
    let t = (x - mu) / sigma;
    (-0.5 * t * t).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rgb_gives_zero_spectrum() {
        for lambda in [380.0, 500.0, 560.0, 650.0, 780.0] {
            assert_eq!(rgb_to_spectral_radiance([0.0, 0.0, 0.0], lambda), 0.0);
        }
    }

    #[test]
    fn negative_components_are_clamped_not_propagated() {
        let v = rgb_to_spectral_radiance([-1.0, -2.0, -3.0], 550.0);
        assert_eq!(v, 0.0);
        assert!(!v.is_nan());
    }

    #[test]
    fn result_is_always_finite_and_non_negative() {
        for lambda in [300.0, 380.0, 450.0, 550.0, 650.0, 780.0, 900.0] {
            let v = rgb_to_spectral_radiance([1.0, 2.5, 100.0], lambda);
            assert!(v.is_finite());
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn red_weight_dominates_the_red_end_of_the_spectrum() {
        let red_only = rgb_to_spectral_radiance([1.0, 0.0, 0.0], 615.0);
        let blue_only = rgb_to_spectral_radiance([0.0, 0.0, 1.0], 615.0);
        assert!(
            red_only > blue_only,
            "at 615nm the red basis peak should dominate the blue basis's tail"
        );
    }
}

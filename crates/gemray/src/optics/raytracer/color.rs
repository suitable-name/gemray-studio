//! Spectral-to-tristimulus colour conversion.
//!
//! The CIE 1931 colour-matching functions, the spectral-MIS combination weight
//! applied at XYZ integration, von Kries white balance, and the final XYZ -> sRGB
//! gamut/gamma mapping.

use super::{
    NUM_CHANNELS,
    environment::{LightingPreset, blackbody_spectrum},
};
use glam::Vec3;

/// Wyman, Sloan, Shirley (2013) multi-lobe analytic fit to the CIE 1931 2° Standard
/// Observer Color Matching Functions.
///
/// Uses piecewise-asymmetric Gaussian lobes (separate sigma below and above each
/// lobe's peak) -- see [`crate::color::cie1931::cie_1931_cmf`], the single source of
/// truth for the fit's constants, which this delegates to. Kept here as a thin
/// `Vec3`-returning wrapper since the raytracer's hot paths want a `glam::Vec3`
/// rather than `[f32; 3]`.
#[must_use]
pub fn cie_1931_cmf(lambda_nm: f32) -> Vec3 {
    Vec3::from_array(crate::color::cie1931::cie_1931_cmf(lambda_nm))
}

/// Batched, `Vec3`-returning wrapper around
/// [`crate::color::cie1931::cie_1931_cmf_x8`] -- see that function's doc comment for
/// the deliberate ULP-level re-baseline this introduces versus 8 calls to
/// [`cie_1931_cmf`] (measured 2026-09-02: constant-folding the CMF call out of
/// `integrate_channels_to_xyz` cut tracer time by ~10.5%, above the 8% vectorization
/// threshold, hence this batched path).
#[must_use]
fn cie_1931_cmf_x8(lambdas: &[f32; NUM_CHANNELS]) -> [Vec3; NUM_CHANNELS] {
    crate::color::cie1931::cie_1931_cmf_x8(lambdas).map(Vec3::from_array)
}

/// Identity pass-through for a channel's already-unbiased radiance estimate.
///
/// A previous pass multiplied `radiance` by a PER-CHANNEL "balance heuristic" weight
/// `own_pdf / sum_pdf * num_channels`, on the premise that the 8 spectral channels
/// were N alternative *techniques* for estimating a single quantity, so Veach-style
/// MIS combination applied directly per channel. That premise was a category error
/// and remains one even after Fix G (Part 1/2, see `spectral_mis_weight` below) made
/// genuine spectral MIS valid in this function:
///
/// The channels are N different *integrands* (different wavelengths' radiance), not N
/// techniques for one integrand. Veach's single-sample balance-heuristic estimate for
/// technique `i`'s contribution to *its own* integral is `w_i * f_i / (c_i * p_i)`,
/// and after combination the `p_i` cancels -- the pdf of a *different* integrand never
/// enters the weight. Using `own_pdf / sum_pdf` (each channel's own physics pdf,
/// divided by a sum across channels) as a PER-CHANNEL multiplier on `radiance[k]`
/// double-counts: `radiance[k]` already has the hero's `1/p_hero` importance-sampling
/// correction baked in (`stokes[k].scale(1.0 / r_unpol)` etc.), so multiplying by
/// `p_k / sum_pdf` on top applies the WRONG channel's pdf as if it were a combination
/// weight. That is what a discriminating Monte Carlo test
/// (`spectral_mis_tests::two_channel_fresnel_monte_carlo_discriminates_correct_from_biased_weighting`)
/// pins down: it is biased by roughly +17% on both channels of a two-channel Fresnel
/// analogue with unequal per-channel reflectance.
///
/// What Fix G adds instead (`spectral_mis_weight`, applied once as a single shared
/// scalar at the final XYZ integration below, identically to every channel) is
/// mathematically a different thing: the weight numerator uses `path_pdf[hero_idx]`
/// -- the density of the technique that was ACTUALLY sampled -- never a companion
/// channel's own pdf. Per Veach's balance-heuristic proof, that same shared weight
/// preserves unbiasedness for every channel's own integral simultaneously, precisely
/// because it does not depend on which channel's integrand `radiance[k]` happens to
/// be. A per-channel `p_k/sum_pdf` multiplier is not a lesser form of that same idea;
/// it is answering a different, wrong question, which is why `mis_weighted_radiance`
/// itself stays the identity function -- see `spectral_mis_weight`'s doc comment for
/// where the real combination now happens.
#[inline]
const fn mis_weighted_radiance(radiance: f32) -> f32 {
    radiance
}

/// Fix G (Part 2): the spectral-MIS balance-heuristic weight `N * p_hero(x) / sum_k
/// p_k(x)`, where `p_k(x)` is `path_pdf[k]` -- channel k's own running density of
/// having produced the EXACT realized path `x` (the sequence of reflect/transmit/
/// Russian-roulette outcomes and, at each dispersive refraction, the specific
/// refracted direction taken), tracked incrementally through `trace_spectral_ray`'s
/// bounce loop (see the TIR / partial-reflect / refract branches there for how each
/// factor is derived).
///
/// This is Veach's balance heuristic for the "one-sample MIS" model, combining N
/// stochastic techniques -- here, "which channel's index drives the shared geometric
/// path" -- each chosen with probability `1/N` (Fix G / Part 1's wrapped hero
/// construction is what makes this uniform-1/N premise hold: every physical
/// wavelength is equally likely to land in the driving slot across the ensemble of
/// independent ray samples a render accumulates). Per Veach's proof this weight is
/// valid for ANY integrand multiplied by it -- including each channel's own
/// `f_k(x)/p_hero(x)` estimator (`radiance[k]`) independently -- which is why the SAME
/// scalar weight is applied uniformly to every channel's radiance at the final XYZ
/// integration, rather than a per-channel `p_k(x)/sum_pdf` weight (that per-channel
/// form is exactly the REJECTED `own_pdf/sum_pdf*N` formula -- see
/// `mis_weighted_radiance`'s doc comment -- because it silently substitutes a
/// DIFFERENT channel's density in place of the hero's, double-counting).
///
/// A companion channel's `path_pdf[k]` (and, critically, its `stokes[k]`/`radiance[k]`
/// -- see the "chromatic termination" comments in `trace_spectral_ray`'s refract
/// branch) collapses to exactly 0 the moment its own specular refraction direction
/// diverges from the direction the hero-driven path actually took. `sum_pdf` then
/// degenerates toward `path_pdf[hero_idx]` alone, pushing the weight up toward `N` --
/// concentrating that sample's contribution onto the hero's own (correctly refracted)
/// colour, which is the mechanism that produces dispersion "fire" at the image level
/// (Requirement 3): different independent samples have different hero wavelengths, so
/// they concentrate onto different colours.
///
/// `sum_pdf` degenerates to exactly `NUM_CHANNELS * path_pdf[hero_idx]` whenever every
/// channel's technique agrees at every decision (a non-dispersive material: identical
/// index for every channel, hence identical Fresnel probabilities AND identical
/// refracted directions for every channel), which makes this collapse to exactly 1.0
/// -- Requirement 2's regression test pins that down directly.
///
/// Both the unbiasedness of this combination and the necessity of also zeroing a
/// chromatically-terminated channel's OWN radiance (not merely its `path_pdf`) were
/// verified by hand-deriving exact expectations for a two-channel analogue (see
/// `two_channel_dispersive_termination_monte_carlo_is_unbiased_under_alternating_hero`
/// below and the accompanying fix report) -- a variant that leaves a terminated
/// channel's own-coefficient radiance un-zeroed is measurably biased (E[.] != 1.0) on
/// that same analogue, even though its `path_pdf` alone is correctly zeroed.
#[inline]
pub(crate) fn spectral_mis_weight(path_pdf: &[f32; 8], hero_idx: usize) -> f32 {
    let sum_pdf: f32 = path_pdf.iter().sum();
    if sum_pdf <= 1e-12 {
        // Should not happen in practice (the hero's own path_pdf factor is always
        // bounded away from 0 by the r_unpol clamps), but fall back to the safe,
        // already-proven-unbiased weight=1 rather than risk a NaN from 0/0.
        return 1.0;
    }
    (path_pdf.len() as f32) * path_pdf[hero_idx] / sum_pdf
}

/// Numerical Integration: Spectral Radiance -> CIE XYZ Tristimulus (normalized by the
/// integral of `y_bar` = 106.856). `radiance[k]` is already an unbiased per-channel
/// estimator on its own -- see [`mis_weighted_radiance`]'s doc comment for why a
/// PER-CHANNEL reweighting on top of this would be a category error even after Fix G.
/// Fix G (Part 1/2) layers a single SHARED scalar MIS weight on top, computed from the
/// fully-accumulated `path_pdf` -- see [`spectral_mis_weight`]'s doc comment for the
/// full derivation.
// `pub(crate)`, not private: Phase 1's GPU furnace-anchor self-test
// (`renderer::gpu::furnace_check`) needs to assemble a CPU-side reference estimator out
// of the SAME building blocks (`cie_1931_cmf`, `spectral_mis_weight`) the GPU port was
// translated from, rather than re-deriving the `norm_factor`/MIS-weight formula by hand
// in test code -- exactly the "never a parallel reimplementation" precedent Phase 0's
// `rng_check::cpu_record` set. Pure function, no side effects; exposing it changes no
// behavior.
pub(crate) fn integrate_channels_to_xyz(
    radiance: &[f32; NUM_CHANNELS],
    lambdas: &[f32; NUM_CHANNELS],
    path_pdf: &[f32; NUM_CHANNELS],
    hero_idx: usize,
) -> Vec3 {
    let mis_weight = spectral_mis_weight(path_pdf, hero_idx);

    let mut xyz = Vec3::ZERO;
    let norm_factor = (400.0 / NUM_CHANNELS as f32) / 106.856;
    let cmfs = cie_1931_cmf_x8(lambdas);
    for k in 0..NUM_CHANNELS {
        let cmf = cmfs[k];
        let weighted_radiance = mis_weighted_radiance(radiance[k]) * mis_weight;
        xyz += cmf * (weighted_radiance * norm_factor);
    }
    xyz
}

/// Colour temperature (Kelvin) associated with each named lighting preset. A thin
/// pass-through to [`LightingPreset::params`] -- the single source of truth both this
/// and `sample_studio_environment` read from, since the white balance must be derived
/// from the same illuminant that lit the scene.
// `pub(crate)`: Phase 1's GPU white-balance self-test needs the exact same
// preset-to-temperature mapping `sample_studio_environment` derives its lighting from
// (see this fn's own doc comment on why that single source of truth matters).
pub(crate) const fn illuminant_temperature_k(lighting_preset: LightingPreset) -> f32 {
    lighting_preset.params().temp_k
}

/// CIE Standard Illuminant D65 chromaticity -- matching
/// `color::space::ColorSpace::Srgb::white_point_xy()` (and every other D65-referenced
/// space this crate defines). This is the reference white
/// [`compute_illuminant_white_balance`] adapts every studio illuminant TOWARD: the
/// output of `trace_spectral_ray` ultimately reaches the screen through an sRGB-family
/// encode step built around this same white point (see `color::space`), so adapting to
/// it here -- rather than to some other neutral point -- is what actually makes the
/// "renders as neutral" promise in [`illuminant_white_balance`]'s doc comment true post
/// gamut-mapping, not just true in raw XYZ.
const D65_WHITE_X: f32 = 0.3127;
const D65_WHITE_Y: f32 = 0.3290;

/// Bradford chromatic-adaptation cone-response matrix (XYZ -> LMS; Lam 1985), the
/// standard basis proper von Kries adaptation diagonalises in -- used by ICC v4 and
/// most colour-management pipelines. Row-major, same convention as
/// `color::space::ColorSpace::xyz_to_rgb_matrix`. See
/// [`compute_illuminant_white_balance`]'s doc comment for why this basis
/// matters: scaling X and Z directly (the previous implementation) is NOT von Kries
/// adaptation at all, just a coincidentally similarly-shaped operation in the wrong
/// space -- XYZ tristimulus values are not cone responses, so a diagonal scale there
/// distorts the hue of any non-neutral colour under a non-D65 illuminant.
const BRADFORD_XYZ_TO_LMS: [[f32; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Exact matrix inverse of [`BRADFORD_XYZ_TO_LMS`] (LMS -> XYZ), the standard published
/// Bradford inverse (cross-checked: `BRADFORD_XYZ_TO_LMS * BRADFORD_LMS_TO_XYZ` is the
/// identity to ~1e-7, comfortably within `f32` rounding).
const BRADFORD_LMS_TO_XYZ: [[f32; 3]; 3] = [
    [0.986_993, -0.147_054, 0.159_963],
    [0.432_305, 0.518_360, 0.049_291],
    [-0.008_529, 0.040_043, 0.968_487],
];

/// Converts CIE XYZ to Bradford-space cone responses (LMS). See
/// [`BRADFORD_XYZ_TO_LMS`]'s doc comment.
#[inline]
fn xyz_to_lms_bradford(xyz: Vec3) -> Vec3 {
    let m = BRADFORD_XYZ_TO_LMS;
    Vec3::new(
        m[0][0].mul_add(xyz.x, m[0][1].mul_add(xyz.y, m[0][2] * xyz.z)),
        m[1][0].mul_add(xyz.x, m[1][1].mul_add(xyz.y, m[1][2] * xyz.z)),
        m[2][0].mul_add(xyz.x, m[2][1].mul_add(xyz.y, m[2][2] * xyz.z)),
    )
}

/// Converts Bradford-space cone responses (LMS) back to CIE XYZ. See
/// [`BRADFORD_LMS_TO_XYZ`]'s doc comment.
#[inline]
fn lms_to_xyz_bradford(lms: Vec3) -> Vec3 {
    let m = BRADFORD_LMS_TO_XYZ;
    Vec3::new(
        m[0][0].mul_add(lms.x, m[0][1].mul_add(lms.y, m[0][2] * lms.z)),
        m[1][0].mul_add(lms.x, m[1][1].mul_add(lms.y, m[1][2] * lms.z)),
        m[2][0].mul_add(lms.x, m[2][1].mul_add(lms.y, m[2][2] * lms.z)),
    )
}

/// Applies a von Kries white-balance scale (as returned by
/// [`compute_illuminant_white_balance`]) to `xyz`: transforms to Bradford LMS, scales
/// each cone response independently, transforms back. This -- not a direct per-channel
/// scale of X and Z -- is what "diagonalise the adaptation in cone space" means (Fix
/// 3). Mirrored bit-for-bit-in-spirit (same matrices, same operation order) by
/// `shaders/spectral_transport.wgsl`'s own application of `params.white_balance`.
// `pub(crate)`: `renderer::gpu::estimator_check::run_spectral_debug` reapplies this
// SAME scale, the SAME way, to its own CPU-side recombination of the GPU kernel's raw
// per-channel radiance -- it must match the megakernel's own `params.white_balance`
// application (`apply_von_kries_white_balance` in `spectral_transport.wgsl`) exactly,
// not the old bare `xyz * white_balance` convention, or that self-consistency check
// compares two different white-balance conventions against each other.
pub(crate) fn apply_von_kries_white_balance(xyz: Vec3, lms_scale: Vec3) -> Vec3 {
    lms_to_xyz_bradford(xyz_to_lms_bradford(xyz) * lms_scale)
}

/// Integrates a blackbody spectrum at `temp_k` against the CIE 1931 CMFs over
/// 380..=780 nm at 1 nm steps to obtain the per-channel von Kries white-balance scale,
/// in Bradford LMS space, that adapts that illuminant's own white point toward
/// [`D65_WHITE_X`]/[`D65_WHITE_Y`]. Applying this scale via
/// [`apply_von_kries_white_balance`] to a rendered XYZ value neutralizes the
/// illuminant's own colour cast without altering the light sources' colour
/// temperatures or the `blackbody_spectrum` clamp. Pulled out of `illuminant_white_balance`
/// unchanged -- only the caching around it differs.
///
/// # Diagonalised in LMS, not XYZ
///
/// The previous implementation returned `[Y_w/X_w, 1.0, Y_w/Z_w]` and the caller
/// multiplied it directly into XYZ -- a diagonal scale of raw X and Z tristimulus
/// values. That is not von Kries adaptation: the human visual system's chromatic
/// adaptation happens per cone class, and X/Y/Z are not cone responses (each mixes all
/// three cone types). Scaling XYZ directly is only correct for a colour that is exactly
/// the illuminant white to begin with; every other colour picks up a hue shift,
/// worst for saturated non-neutral colours under a strongly non-D65 illuminant (e.g.
/// the 3200K incandescent preset). This now integrates the source illuminant's white in
/// XYZ exactly as before, but converts BOTH that source white and the [`D65_WHITE_X`]/
/// [`D65_WHITE_Y`] reference white (at the same luminance, for a directly comparable
/// ratio) to Bradford LMS via [`xyz_to_lms_bradford`] before taking the per-component
/// ratio -- the textbook von Kries construction, just with the diagonalisation
/// happening in the basis where it is actually valid.
// `pub(crate)`: this is the exact 401-point (380..=780nm, 1nm step) quadrature Phase
// 1's GPU white-balance self-test (`renderer::gpu::environment_check`) must reproduce
// on the GPU and compare ULP against -- calling the real function rather than
// re-deriving the loop in test code.
pub(crate) fn compute_illuminant_white_balance(temp_k: f32) -> Vec3 {
    let mut xyz_w = Vec3::ZERO;
    for step in 0..=(780 - 380) {
        let lambda = 380.0f32 + step as f32;
        xyz_w += cie_1931_cmf(lambda) * blackbody_spectrum(lambda, temp_k);
    }

    let target_y = xyz_w.y.max(1e-6);
    let xyz_target = Vec3::new(
        (D65_WHITE_X / D65_WHITE_Y) * target_y,
        target_y,
        ((1.0 - D65_WHITE_X - D65_WHITE_Y) / D65_WHITE_Y) * target_y,
    );

    let lms_source = xyz_to_lms_bradford(xyz_w);
    let lms_target = xyz_to_lms_bradford(xyz_target);

    Vec3::new(
        if lms_source.x > 1e-6 {
            lms_target.x / lms_source.x
        } else {
            1.0
        },
        if lms_source.y > 1e-6 {
            lms_target.y / lms_source.y
        } else {
            1.0
        },
        if lms_source.z > 1e-6 {
            lms_target.z / lms_source.z
        } else {
            1.0
        },
    )
}

/// Returns the (lazily-computed, cached) per-channel von Kries white-balance scale for
/// a given lighting preset.
///
/// This is called from `trace_spectral_ray` once per ray sample -- i.e. up to
/// millions of times per frame across every render worker thread. The previous
/// implementation cached results in a `Mutex<HashMap<..>>`, so every single ray
/// sample serialized on one lock, which was enough contention to collapse the
/// parallel render loop back to effectively single-threaded. Since the full set of
/// lighting presets is small and known ahead of time (mirrored from
/// `illuminant_temperature_k`), each preset instead gets its own `OnceLock<Vec3>`
/// static, selected with a `match`. After the first call per preset this is a lock-free
/// atomic read with no allocation and no `HashMap` involved.
pub(super) fn illuminant_white_balance(lighting_preset: LightingPreset) -> Vec3 {
    static INCANDESCENT: std::sync::OnceLock<Vec3> = std::sync::OnceLock::new();
    static RING_LIGHTS: std::sync::OnceLock<Vec3> = std::sync::OnceLock::new();
    static DARK_SPOTLIGHT: std::sync::OnceLock<Vec3> = std::sync::OnceLock::new();
    static DAYLIGHT_DEFAULT: std::sync::OnceLock<Vec3> = std::sync::OnceLock::new();

    let temp_k = illuminant_temperature_k(lighting_preset);
    match lighting_preset {
        LightingPreset::Incandescent => {
            *INCANDESCENT.get_or_init(|| compute_illuminant_white_balance(temp_k))
        }
        LightingPreset::RingLights => {
            *RING_LIGHTS.get_or_init(|| compute_illuminant_white_balance(temp_k))
        }
        LightingPreset::DarkSpotlight => {
            *DARK_SPOTLIGHT.get_or_init(|| compute_illuminant_white_balance(temp_k))
        }
        LightingPreset::Daylight => {
            *DAYLIGHT_DEFAULT.get_or_init(|| compute_illuminant_white_balance(temp_k))
        }
    }
}

/// ACES Filmic Tone Mapping Curve, applied to a scalar luminance value.
///
/// Reduced from the old per-channel-Vec3 form: applying this curve independently per
/// RGB channel is a hue-shifting operator, which is exactly wrong for saturated
/// dispersion "fire" colours. `xyz_to_srgb_gamma` now applies it to luminance only.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    reason = "ACES filmic tonemap (Narkowicz 2015) fit constants, named a..e to match \
              the canonical y*(a*y+b) / (y*(c*y+d)+e) formula as published everywhere \
              it's referenced; renaming them would only obscure the connection to the \
              reference formula for anyone checking this against the source"
)]
pub fn aces_tonemap(y: f32) -> f32 {
    let a = 2.51f32;
    let b = 0.03f32;
    let c = 2.43f32;
    let d = 0.59f32;
    let e = 0.14f32;
    (y * a.mul_add(y, b)) / y.mul_add(c.mul_add(y, d), e)
}

/// Converts a CIE XYZ radiance sample to encoded 8-bit RGBA in an arbitrary wide-gamut
/// [`crate::color::ColorSpace`].
///
/// Space-aware chromaticity-preserving gamut mapping (radially compressing
/// out-of-gamut colours toward the space's own white point in CIE xyY at constant
/// luminance, rather than desaturating/hue-shifting via naive per-channel clamping),
/// ACES filmic tone mapping applied to luminance only, and finally that space's own
/// transfer function. See `crate::color::space` and `crate::color::gamut` for the
/// implementation this delegates to.
///
/// NOTE: no caller currently lets the user pick `space` -- every existing call site
/// still goes through [`xyz_to_srgb_gamma`] below. The render-export path (not yet
/// built) is the natural place to expose `space` as a user-facing choice (sRGB /
/// Display P3 / Rec.2020 / `ACEScg`); this function is the entry point it should call.
#[must_use]
pub fn xyz_to_rgb_in_space(xyz: Vec3, space: crate::color::ColorSpace) -> [u8; 4] {
    space.encode(xyz, crate::color::ToneMap::AcesFilmic { exposure: 1.0 })
}

/// Converts CIE XYZ to sRGB via chromaticity-preserving gamut mapping (Procedure 3).
///
/// Thin wrapper around [`xyz_to_rgb_in_space`] targeting [`crate::color::ColorSpace::Srgb`].
///
/// This used to perform the full pipeline inline, applying a flat `1/2.2` gamma as an
/// approximation of the true sRGB transfer curve. `ColorSpace::Srgb` uses the true
/// piecewise curve (`12.92 * x` below the breakpoint, `1.055 * x^(1/2.4) - 0.055`
/// above) instead, which is a real -- if small -- behavioural change. Measured directly
/// (dense sweep of the encoded-value difference between the two curves over linear
/// `x` in `[0, 1]`, not just at a handful of samples): the two curves diverge most in
/// deep shadow, peaking at an absolute difference of ~0.0335 (about 8.5 of 255 levels)
/// at linear x ~= 0.00216, and stay under 0.006 (about 1.5 of 255 levels) across the
/// 0.1-0.9 midtone/highlight range -- they do NOT converge to sub-1-level agreement
/// throughout that range (e.g. still ~1.5 levels apart at x ~= 0.41). Switched anyway
/// (rather than kept as a bug-compatible approximation) because the true curve is the
/// physically correct one and the deviation is confined to shadow detail well below
/// the range most gemstone renders spend their dynamic range in; see
/// `tests/color_tests.rs::
/// srgb_encode_matches_xyz_to_srgb_gamma_reference_within_the_known_gamma_curve_difference`
/// for the regression pinning this tolerance (10-level-per-channel bound) against a
/// handful of representative in-gamut samples.
#[must_use]
pub fn xyz_to_srgb_gamma(xyz: Vec3) -> [u8; 4] {
    xyz_to_rgb_in_space(xyz, crate::color::ColorSpace::Srgb)
}

#[cfg(test)]
mod white_balance_cache_tests {
    use super::*;

    /// Fix C: the white-balance lookup used to cache in a `Mutex<HashMap<..>>` that
    /// serialized every ray sample across every render thread. The replacement caches
    /// each known preset behind its own lock-free `OnceLock<Vec3>`, selected by a
    /// `match`, while the actual integration math (`compute_illuminant_white_balance`)
    /// is untouched. This asserts the cached value for every preset (the three named
    /// ones plus the default fallback arm) exactly matches a fresh, uncached
    /// recomputation, i.e. the visible values are unchanged by the caching rewrite.
    #[test]
    fn illuminant_white_balance_matches_direct_computation_for_all_presets() {
        for preset in LightingPreset::ALL {
            let cached = illuminant_white_balance(preset);
            let direct = compute_illuminant_white_balance(illuminant_temperature_k(preset));
            assert!(
                (cached - direct).length() < 1e-5,
                "cached white balance for {preset:?} should exactly match direct integration (cached={cached:?}, direct={direct:?})"
            );
        }
    }

    /// Every unrecognized preset LABEL must parse (via `LightingPreset::from_label`) and
    /// fall through to the same default (D65 6500K) preset -- including the legacy,
    /// mislabelled `"D65 Daylight (5500K)"` string a settings file saved before the
    /// label fix may still contain (see `LightingPreset::from_label`'s doc comment for
    /// why that is exactly the graceful-migration behaviour wanted here).
    #[test]
    fn illuminant_white_balance_default_arm_is_shared() {
        let a = illuminant_white_balance(LightingPreset::from_label("Totally Unknown Preset A"));
        let b = illuminant_white_balance(LightingPreset::from_label("Totally Unknown Preset B"));
        let legacy = illuminant_white_balance(LightingPreset::from_label("D65 Daylight (5500K)"));
        assert!(
            (a - b).length() < 1e-6,
            "distinct unrecognized presets must share the default D65 white balance"
        );
        assert!(
            (a - legacy).length() < 1e-6,
            "the legacy mislabelled D65 string must still migrate to the default D65 white balance"
        );
    }

    /// The whole point of Fix C is that concurrent lookups from many render threads no
    /// longer serialize on a shared mutex. This doesn't measure throughput directly,
    /// but it does confirm the lock-free OnceLock-per-preset statics are race-free:
    /// many threads racing to initialize the same preset's `OnceLock` must all observe
    /// the identical value.
    #[test]
    fn illuminant_white_balance_is_stable_across_concurrent_threads() {
        let presets = LightingPreset::ALL;

        let handles: Vec<_> = (0..32)
            .map(|i| {
                std::thread::spawn(move || {
                    let preset = presets[i % presets.len()];
                    (preset, illuminant_white_balance(preset))
                })
            })
            .collect();

        let mut by_preset: std::collections::HashMap<LightingPreset, Vec3> =
            std::collections::HashMap::new();
        for h in handles {
            let (preset, v) = h.join().unwrap();
            if let Some(existing) = by_preset.get(&preset) {
                assert!(
                    (*existing - v).length() < 1e-6,
                    "value for {preset:?} differs across threads"
                );
            } else {
                by_preset.insert(preset, v);
            }
        }
    }
}

#[cfg(test)]
mod spectral_mis_tests {
    use super::{
        super::{sampling::hash_u32, transport::wrapped_hero_wavelengths},
        *,
    };

    /// `mis_weighted_radiance` is now the identity function -- see its doc comment for
    /// why no reweighting is valid under `trace_spectral_ray`'s current wavelength
    /// stratification. This just pins that contract down directly.
    #[test]
    fn mis_weighted_radiance_is_identity() {
        for &r in &[0.0f32, 1.0, 4.2, 1.0e6, -3.5] {
            assert_eq!(
                mis_weighted_radiance(r),
                r,
                "mis_weighted_radiance must return its input unchanged (r={r})"
            );
        }
    }

    /// Deterministic unit-interval draw built from the same `hash_u32` PRNG
    /// `trace_spectral_ray` itself uses, so this test's Monte Carlo trials are
    /// reproducible without pulling in an external `rand` dependency.
    fn unit_rand(seed: u32) -> f32 {
        (hash_u32(seed) as f32) / 4_294_967_295.0
    }

    /// The OLD, buggy `own_pdf / sum_pdf * num_channels` balance-heuristic weight a
    /// previous pass wired into `mis_weighted_radiance`. Reproduced here directly
    /// (rather than calling into the raytracer module, which no longer contains it) purely so
    /// this regression test can demonstrate, and permanently guard against
    /// reintroducing, the bias it causes.
    fn shipped_biased_weight(
        radiance: f32,
        own_pdf: f32,
        sum_pdf: f32,
        num_channels: usize,
    ) -> f32 {
        let weight = (own_pdf / sum_pdf.max(1e-8)) * num_channels as f32;
        radiance * weight
    }

    /// Discriminating Monte Carlo regression test for the spectral-MIS bias bug.
    ///
    /// `equal_pdfs_reduce_to_unweighted_radiance` (the test this replaces) could never
    /// have caught the bug: when every channel's pdf is equal, `own_pdf/sum_pdf * N`
    /// and the constant weight 1 are algebraically IDENTICAL, so that test passed for
    /// both the correct and the buggy formula. This test instead uses UNEQUAL
    /// per-channel pdfs with a known closed-form ground truth, which the two formulas
    /// answer differently.
    ///
    /// Scenario: a minimal two-channel analogue of one Fresnel interface. Channel 0 is
    /// the hero with reflectance R0 = 0.2; channel 1 is a companion channel with a
    /// DIFFERENT reflectance R1 = 0.6 (mimicking a dispersive material, where a
    /// companion channel's own physics differs from the hero's). Exactly one branch
    /// decision (reflect vs. transmit) is made per trial, drawn from the HERO's own
    /// reflectance -- mirroring `trace_spectral_ray`'s real branch selection -- and
    /// each channel's raw per-trial value is its OWN reflectance/transmittance divided
    /// by the HERO's own branch-selection probability, exactly mirroring
    /// `stokes[k].apply_matrix(..).scale(1.0 / r_unpol)` in `trace_spectral_ray`. The
    /// closed-form ground truth for both channels is `L_k = R_k + (1 - R_k) = 1.0`
    /// exactly (a Fresnel interface reflects or transmits with unit total probability).
    ///
    /// This asserts:
    /// 1. The FIXED estimator (weight = 1, i.e. `mis_weighted_radiance` as shipped)
    ///    converges to the known truth (1.0) within a few percent for BOTH channels,
    ///    including channel 1 -- whose own reflectance differs from the hero's, so its
    ///    pdf is genuinely unequal to the hero's pdf, unlike every case
    ///    `equal_pdfs_reduce_to_unweighted_radiance` could exercise.
    /// 2. The OLD shipped `own_pdf/sum_pdf * N` weight (reproduced locally via
    ///    `shipped_biased_weight`, since it no longer exists in the raytracer module) is
    ///    biased by roughly +17% on BOTH channels on this same scenario -- i.e. this
    ///    test's assertion (1) would have FAILED against the formula this fix replaces.
    ///    Verified by hand: temporarily substituting `shipped_biased_weight(radiance[k],
    ///    path_pdf[k], sum_pdf, 2)` for `mis_weighted_radiance(radiance[k])` in
    ///    assertion (1) makes it fail with `plain_avg = [1.1662, 1.1690]` against the
    ///    3% tolerance (expected ~1.0) -- see the accompanying fix report for the full
    ///    captured numbers -- before being reverted back to the fixed formula below.
    #[test]
    fn two_channel_fresnel_monte_carlo_discriminates_correct_from_biased_weighting() {
        const R0: f32 = 0.2; // hero (channel 0) reflectance
        const R1: f32 = 0.6; // companion (channel 1) reflectance, deliberately different
        const TRIALS: u32 = 400_000;
        const GROUND_TRUTH: f32 = 1.0;

        let mut plain_sum = [0.0f64; 2];
        let mut biased_sum = [0.0f64; 2];

        for trial in 0..TRIALS {
            let xi = unit_rand(trial ^ 0xA5A5_5A5A);

            // radiance[k] and path_pdf[k] for the branch actually taken this trial,
            // mirroring trace_spectral_ray's own per-channel bookkeeping.
            let (radiance, path_pdf) = if xi < R0 {
                // Reflect branch, selected with the HERO's own probability R0.
                ([1.0f32, R1 / R0], [R0, R1])
            } else {
                // Transmit branch, selected with the HERO's own probability (1 - R0).
                ([1.0f32, (1.0 - R1) / (1.0 - R0)], [1.0 - R0, 1.0 - R1])
            };

            let sum_pdf = path_pdf[0] + path_pdf[1];
            for k in 0..2 {
                plain_sum[k] += f64::from(mis_weighted_radiance(radiance[k]));
                biased_sum[k] +=
                    f64::from(shipped_biased_weight(radiance[k], path_pdf[k], sum_pdf, 2));
            }
        }

        let plain_avg: Vec<f32> = plain_sum
            .iter()
            .map(|s| (*s / f64::from(TRIALS)) as f32)
            .collect();
        let biased_avg: Vec<f32> = biased_sum
            .iter()
            .map(|s| (*s / f64::from(TRIALS)) as f32)
            .collect();

        for (k, &avg) in plain_avg.iter().enumerate() {
            let err = (avg - GROUND_TRUTH).abs() / GROUND_TRUTH;
            assert!(
                err < 0.03,
                "FIXED (weight=1) estimator for channel {} should converge to the ground truth {} within 3% over {} trials (got {}, {:.2}% error)",
                k,
                GROUND_TRUTH,
                TRIALS,
                avg,
                err * 100.0
            );
        }

        // The old shipped formula must be clearly, substantially biased on this same
        // scenario -- this is what proves the test actually discriminates between the
        // two formulas, rather than accidentally passing for both the way
        // `equal_pdfs_reduce_to_unweighted_radiance` used to.
        for (k, &avg) in biased_avg.iter().enumerate() {
            let err = (avg - GROUND_TRUTH).abs() / GROUND_TRUTH;
            assert!(
                err > 0.10,
                "the OLD shipped own_pdf/sum_pdf*N weight is expected to be substantially biased (>10%) on this scenario for channel {} (got {}, {:.2}% error) -- if this assertion fails, this regression test has lost its discriminating power",
                k,
                avg,
                err * 100.0
            );
        }
    }

    /// Fix G (Part 1): the wrapped hero-wavelength construction must (a) keep every
    /// generated wavelength within the visible range [380, 780] regardless of the
    /// hero draw, including right at the wraparound boundary, and (b) always place the
    /// hero (`lambda_hero` itself) at array index 0 -- see `wrapped_hero_wavelengths`'s
    /// and `trace_spectral_ray`'s `hero_idx` doc comments for why that is provably the
    /// case under this exact formula.
    #[test]
    fn wrapped_hero_wavelengths_stay_in_visible_range_and_hero_is_always_index_0() {
        for seed in 0..20_000u32 {
            let hero_rand = unit_rand(seed);
            let lambdas: [f32; 8] = wrapped_hero_wavelengths(hero_rand);
            let lambda_hero = hero_rand.mul_add(780.0 - 380.0, 380.0);

            for (k, &l) in lambdas.iter().enumerate() {
                assert!(
                    (380.0..=780.0).contains(&l),
                    "wavelength at channel {k} must stay within [380, 780] (seed={seed}, hero_rand={hero_rand}, got {l})"
                );
            }
            assert!(
                (lambdas[0] - lambda_hero).abs() < 1e-3,
                "hero must always land at array index 0 (seed={}, hero_rand={}, lambdas[0]={}, lambda_hero={})",
                seed,
                hero_rand,
                lambdas[0],
                lambda_hero
            );
        }

        // Boundary check: a hero_rand right at the top of its range wraps the highest
        // companion channels back down past 380nm rather than running off past 780nm.
        let lambdas_top: [f32; 8] = wrapped_hero_wavelengths(0.999_999);
        for &l in &lambdas_top {
            assert!(
                (380.0..=780.0).contains(&l),
                "boundary hero draw produced an out-of-range wavelength: {l}"
            );
        }
    }

    /// Fix G (Part 1): confirms the key statistical property the wrapped construction
    /// buys over the old one -- every one of the N channel SLOTS is, across many
    /// draws, uniformly distributed over the full comb-relative rotation, i.e. no
    /// channel index is structurally privileged. Concretely: the fractional position
    /// of `lambdas[k]` within its own 50nm sub-band should be uniform on [0, 1) for
    /// every k, including k=7 (the last channel), which the OLD construction could
    /// never populate with a value below 380 + 7*50 = 730nm.
    #[test]
    fn wrapped_hero_wavelengths_cover_every_channel_slot_uniformly() {
        let mut min_seen = [1000.0f32; 8];
        let mut max_seen = [0.0f32; 8];
        for seed in 0..20_000u32 {
            let hero_rand = unit_rand(seed ^ 0xDEAD_BEEF);
            let lambdas: [f32; 8] = wrapped_hero_wavelengths(hero_rand);
            for k in 0..8 {
                min_seen[k] = min_seen[k].min(lambdas[k]);
                max_seen[k] = max_seen[k].max(lambdas[k]);
            }
        }
        for k in 0..8 {
            // Each channel should, across enough draws, range across nearly the
            // entire [380, 780] spectrum (not just its "home" 50nm sub-band) -- this
            // is exactly what was NOT true under the old construction.
            assert!(
                max_seen[k] - min_seen[k] > 350.0,
                "channel {} should range across nearly the full spectrum over many hero draws (got min={}, max={}, span={})",
                k,
                min_seen[k],
                max_seen[k],
                max_seen[k] - min_seen[k]
            );
        }
    }

    /// `spectral_mis_weight` must reduce to EXACTLY 1.0 whenever every channel's
    /// `path_pdf` is identical -- the case a non-dispersive material forces (see
    /// `trace_spectral_ray`'s refract/reflect branches: identical n(lambda) for every
    /// channel makes every per-channel Fresnel probability, and hence every
    /// `path_pdf` factor at every bounce, identical across channels). This is the
    /// direct mathematical guarantee behind Requirement 2 (non-dispersive materials
    /// are bit-for-bit unaffected by Fix G): `sum_pdf` collapses to exactly
    /// `N * path_pdf[hero_idx]`, so the weight is `N * p / (N * p) == 1.0` for ANY
    /// common value `p`, checked here across several different hero indices and common
    /// values (including very small ones, near the `r_unpol` clamp floor).
    #[test]
    fn spectral_mis_weight_is_exactly_unity_when_all_channels_agree() {
        for &p in &[1.0f32, 0.5, 1e-4, 1e-3, 0.999_9] {
            for hero_idx in 0..8 {
                let path_pdf = [p; 8];
                let w = spectral_mis_weight(&path_pdf, hero_idx);
                // `sum_pdf` is computed via an iterative float sum of 8 equal values,
                // which need not be bit-identical to `8.0 * p` -- so this checks
                // "exactly 1.0 up to a couple ULPs", not literal `f32` equality. This
                // ULP-level tolerance is irrelevant at the u8 pixel output the render
                // pipeline ultimately produces; the mathematically-exact statement
                // ("weight is exactly 1.0 in the real number sense whenever every
                // channel agrees") is what Requirement 2 actually depends on.
                assert!(
                    (w - 1.0).abs() < 1e-6,
                    "weight must be 1.0 (up to float rounding) when every channel's path_pdf is identical (p={p}, hero_idx={hero_idx}, got {w})"
                );
            }
        }
    }

    /// `spectral_mis_weight` must depart from 1.0 once channels disagree, and must
    /// approach `N` (here 8) as the non-hero channels' `path_pdf` collapses toward 0 --
    /// i.e. once chromatic termination kills off every companion, the surviving
    /// hero's own sample gets the full weight. This is the mechanism
    /// `spectral_mis_weight`'s doc comment describes as concentrating a sample's
    /// contribution onto the hero's own colour, producing image-level dispersion
    /// "fire" (Requirement 3).
    #[test]
    fn spectral_mis_weight_approaches_n_as_companions_are_chromatically_terminated() {
        let hero_idx = 0usize;
        let mut path_pdf = [0.3f32; 8];
        path_pdf[hero_idx] = 0.3;
        let w_all_alive = spectral_mis_weight(&path_pdf, hero_idx);
        assert!(
            (w_all_alive - 1.0).abs() < 1e-4,
            "all channels agreeing should give weight ~= 1.0 (got {w_all_alive})"
        );

        // Terminate every companion (path_pdf -> 0), leaving only the hero alive.
        for (k, p) in path_pdf.iter_mut().enumerate() {
            if k != hero_idx {
                *p = 0.0;
            }
        }
        let w_hero_only = spectral_mis_weight(&path_pdf, hero_idx);
        assert!(
            (w_hero_only - 8.0).abs() < 1e-4,
            "with every companion terminated, weight should approach N=8 (got {w_hero_only})"
        );
    }

    /// Fix G (Part 2): discriminating Monte Carlo regression test for the NEW
    /// spectral-MIS weight `N * path_pdf[hero_idx] / sum_pdf`, extended to a scenario
    /// with a genuine dispersive-refraction "chromatic termination" event -- the case
    /// the OLD, reverted `own_pdf/sum_pdf*N` weight never modelled at all (it tracked
    /// only discrete branch probabilities, never a directional match/mismatch).
    ///
    /// Unlike `two_channel_fresnel_monte_carlo_discriminates_correct_from_biased_weighting`
    /// above (which fixes the hero at channel 0 for every trial, mirroring a SINGLE
    /// `trace_spectral_ray` call in isolation), this harness additionally simulates
    /// the wrapped hero construction's key statistical property (Part 1): across MANY
    /// INDEPENDENT invocations, each of the two channels is equally likely (p=1/2) to
    /// be the one actually driving (`hero_idx`). This is not optional set-dressing --
    /// a hand derivation, cross-checked with a standalone verification script,
    /// confirms the combined weight is provably biased if tested with a single fixed
    /// hero (E[`F_A`]=1.7, E[`F_B`]=1.1 for `R_A=0.2`, `R_B=0.6` with hero fixed at A -- the
    /// SAME two-technique combination `two_channel_fresnel_monte_carlo_...` above uses
    /// weight=1 for, which is exactly why THAT test does not alternate hero), and only
    /// becomes unbiased once the ensemble genuinely alternates which channel drives --
    /// matching how a real render accumulates many independent ray samples, each with
    /// its own wrapped hero draw.
    ///
    /// Scenario: a single Fresnel interface, exactly as the sibling test above (hero
    /// reflectance vs. a companion with a DIFFERENT reflectance), but the companion's
    /// transmission direction is modelled as GENUINELY DISPERSIVE -- it never
    /// coincides with whichever channel is driving. Per `trace_spectral_ray`'s actual
    /// Part 2 logic, this means: (a) the non-driving channel's running `path_pdf`
    /// collapses to 0 at that transmit event, and (b) its Stokes/radiance value is
    /// ALSO zeroed there (chromatic termination), not merely down-weighted --
    /// mirroring the existing "cannot transmit at this angle" branch's
    /// `stokes[k].scale(0.0)`. A cross-check (same hand derivation) confirms that
    /// skipping step (b) -- i.e. continuing to accumulate the companion's
    /// own-coefficient throughput past a genuine directional mismatch while still
    /// applying the combined weight -- reintroduces bias even with hero alternating.
    /// The closed-form ground truth for EACH channel's own combined estimator is
    /// still exactly 1.0 (reflectance + transmittance = 1, Fresnel unitarity), by
    /// Veach's balance-heuristic unbiasedness theorem applied per-channel with the
    /// shared weight (see `spectral_mis_weight`'s doc comment).
    #[test]
    fn two_channel_dispersive_termination_monte_carlo_is_unbiased_under_alternating_hero() {
        const R_A: f32 = 0.2;
        const R_B: f32 = 0.6;
        const TRIALS: u32 = 400_000;
        const GROUND_TRUTH: f32 = 1.0;

        // One Fresnel-interface trial. `hero_is_a` selects which channel's own
        // reflectance drives the shared branch decision this trial, mirroring which
        // physical wavelength happened to land in `hero_idx` this invocation. Returns
        // (F_A, F_B): this trial's combined (weighted) estimate of channel A's and
        // channel B's own integral.
        fn trial(xi: f32, hero_is_a: bool) -> (f32, f32) {
            let (r_hero, r_other) = if hero_is_a { (R_A, R_B) } else { (R_B, R_A) };

            let (rad_hero, rad_other, pdf_hero, pdf_other) = if xi < r_hero {
                // Reflect: never dispersive -- both channels' directions coincide, so
                // the companion's own reflectance is a valid, nonzero path_pdf
                // contribution (mirrors the TIR/reflect branches' `r_unpol_k` factor).
                (1.0f32, r_other / r_hero, r_hero, r_other)
            } else {
                // Transmit: genuinely dispersive -- the companion's own refracted
                // direction never coincides with whichever channel is driving, so its
                // path_pdf AND its Stokes/radiance both collapse to 0 (chromatic
                // termination), exactly as `trace_spectral_ray`'s refract branch now
                // does at a direction mismatch.
                (1.0f32, 0.0f32, 1.0 - r_hero, 0.0f32)
            };

            let sum_pdf = pdf_hero + pdf_other;
            let weight = 2.0 * pdf_hero / sum_pdf.max(1e-8);

            if hero_is_a {
                (rad_hero * weight, rad_other * weight)
            } else {
                (rad_other * weight, rad_hero * weight)
            }
        }

        let mut sum_a = 0.0f64;
        let mut sum_b = 0.0f64;
        for trial_idx in 0..TRIALS {
            // Independent draws: which channel is hero this trial (mirrors Part 1's
            // wrapped construction making every channel equally likely to drive
            // across the sample ensemble), and the branch decision xi.
            let hero_is_a = unit_rand(trial_idx ^ 0x1234_5678) < 0.5;
            let xi = unit_rand(trial_idx ^ 0xA5A5_5A5A);
            let (f_a, f_b) = trial(xi, hero_is_a);
            sum_a += f64::from(f_a);
            sum_b += f64::from(f_b);
        }

        let avg_a = (sum_a / f64::from(TRIALS)) as f32;
        let avg_b = (sum_b / f64::from(TRIALS)) as f32;

        for (label, avg) in [("A", avg_a), ("B", avg_b)] {
            let err = (avg - GROUND_TRUTH).abs() / GROUND_TRUTH;
            assert!(
                err < 0.03,
                "channel {} combined estimator should converge to ground truth {} within 3% over {} trials under alternating hero (got {}, {:.2}% error)",
                label,
                GROUND_TRUTH,
                TRIALS,
                avg,
                err * 100.0
            );
        }
    }
}

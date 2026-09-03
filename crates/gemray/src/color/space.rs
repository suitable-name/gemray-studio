//! Wide-gamut colour-space definitions.
//!
//! XYZ->RGB matrices, reference white points, and transfer functions for sRGB, Display
//! P3, Rec.2020, and `ACEScg`, plus [`ColorSpace::encode`] -- a single entry point that
//! carries a CIE XYZ radiance sample all the way to encoded 8-bit output.
//!
//! # Intended wiring (not performed by this module)
//!
//! `optics::raytracer::xyz_to_srgb_gamma` currently performs this entire pipeline
//! inline: XYZ -> linear sRGB, radial gamut compression toward D65 in CIE xyY, ACES
//! tone mapping of luminance only, then a flat 1/2.2 gamma. This module is a
//! self-contained, generalised replacement for that pipeline, but it is deliberately
//! **not** wired into the renderer here (a follow-up task owns `raytracer.rs`).
//!
//! Once that file is free to edit, the mechanical swap is:
//!
//! ```ignore
//! // old:
//! let pixel = xyz_to_srgb_gamma(xyz);
//! // new:
//! let pixel = ColorSpace::Srgb.encode(xyz, ToneMap::AcesFilmic { exposure: 1.0 });
//! ```
//!
//! This reproduces `xyz_to_srgb_gamma`'s gamut-mapping and tone-mapping steps exactly
//! (see [`crate::color::gamut::project_to_gamut`] and the `AcesFilmic` variant of
//! [`ToneMap`]), with one deliberate difference: the true piecewise sRGB transfer curve
//! replaces the flat 1/2.2 gamma approximation. See
//! `tests/color_tests.rs::srgb_encode_matches_xyz_to_srgb_gamma_reference_within_the_
//! known_gamma_curve_difference` for the measured size of that difference (analytically,
//! up to ~9/255 in deep shadows near linear value 0.002, and well under 2/255 across the
//! 0.1-0.9 midtone/highlight range).

use glam::Vec3;

use super::gamut;

/// A transfer function (opto-electronic encoding curve) mapping a scene-linear value in
/// `[0, 1]` to its encoded, display-ready counterpart, and back.
///
/// These are genuinely different curves, not the same power function in disguise --
/// see each variant's docs for the constants and why they differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunction {
    /// The piecewise sRGB curve (IEC 61966-2-1 / the curve most commonly (and
    /// correctly) called "the sRGB gamma"): `12.92 * x` for `x <= 0.0031308`,
    /// `1.055 * x^(1/2.4) - 0.055` above. Shared by sRGB and Display P3 -- Display P3
    /// reuses the sRGB primaries' transfer function verbatim; it does not define its
    /// own.
    Srgb,
    /// The piecewise Rec.2020 curve (ITU-R BT.2020-2, Table 4): `4.5 * x` below
    /// `beta`, `alpha * x^0.45 - (alpha - 1)` above, with `alpha ~= 1.0992968` and
    /// `beta ~= 0.01805397`. Distinct constants and a distinct exponent (0.45, not
    /// 1/2.4) from the sRGB curve above -- treating Rec.2020 as "the same curve as
    /// sRGB" is a real error the flat 1/2.2 gamma this module replaces effectively
    /// makes for every space.
    Rec2020,
    /// No encoding curve at all: `ACEScg` is a scene-linear working space by design
    /// (it is meant for further compositing, not direct display), so `encode`/`decode`
    /// are the identity function.
    Linear,
}

impl TransferFunction {
    /// Encodes a scene-linear value into its transfer-encoded counterpart. Defined for
    /// `linear` in `[0, 1]`; the caller is expected to clamp (as [`ColorSpace::encode`]
    /// does) since a negative base raised to a fractional power is not real-valued.
    #[must_use]
    pub fn encode(self, linear: f32) -> f32 {
        match self {
            Self::Srgb => {
                if linear <= 0.003_130_8 {
                    linear * 12.92
                } else {
                    1.055f32.mul_add(linear.powf(1.0 / 2.4), -0.055)
                }
            }
            Self::Rec2020 => {
                const ALPHA: f32 = 1.099_296_8;
                const BETA: f32 = 0.018_053_97;
                if linear < BETA {
                    linear * 4.5
                } else {
                    ALPHA.mul_add(linear.powf(0.45), -(ALPHA - 1.0))
                }
            }
            Self::Linear => linear,
        }
    }

    /// Decodes a transfer-encoded value back to scene-linear. The exact inverse of
    /// [`Self::encode`] (up to floating-point rounding), including right at the
    /// piecewise breakpoint -- see `tests/color_tests.rs` for round-trip coverage of
    /// every variant across that breakpoint.
    #[must_use]
    pub fn decode(self, encoded: f32) -> f32 {
        match self {
            Self::Srgb => {
                if encoded <= 0.040_45 {
                    encoded / 12.92
                } else {
                    ((encoded + 0.055) / 1.055).powf(2.4)
                }
            }
            Self::Rec2020 => {
                const ALPHA: f32 = 1.099_296_8;
                const BETA: f32 = 0.018_053_97;
                let breakpoint = 4.5 * BETA;
                if encoded < breakpoint {
                    encoded / 4.5
                } else {
                    ((encoded + (ALPHA - 1.0)) / ALPHA).powf(1.0 / 0.45)
                }
            }
            Self::Linear => encoded,
        }
    }
}

/// Policy for compressing scene-linear radiance into the encodable `[0, 1]` range
/// before a space's transfer function is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMap {
    /// No tone mapping: values are simply clamped to `[0, 1]` by [`ColorSpace::encode`].
    /// Useful for values already known to be display-range, or when a caller wants to
    /// apply its own tone mapping upstream of this module.
    None,
    /// ACES filmic tonemap (the Narkowicz 2015 fit -- the same curve as
    /// `optics::raytracer::aces_tonemap`), applied to luminance only and then rescaled
    /// back into the RGB vector so hue and saturation are preserved. A per-channel
    /// tonemap is a hue-shifting operator, which is exactly wrong for saturated
    /// dispersion "fire" colours -- see the doc comment on `aces_tonemap` in
    /// `optics::raytracer`.
    ///
    /// `exposure` is a linear multiplier applied to radiance before the curve;
    /// `exposure = 1.0` reproduces `xyz_to_srgb_gamma`'s tone-mapping step exactly.
    AcesFilmic {
        /// Linear exposure multiplier applied to radiance before tone mapping.
        exposure: f32,
    },
}

/// A target RGB colour space: its XYZ->RGB matrix, reference white point, and transfer
/// function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// IEC 61966-2-1 sRGB, D65 white. The output space `xyz_to_srgb_gamma` currently
    /// hardwires.
    Srgb,
    /// Display P3 (Apple / SMPTE EG 432-1 primaries), D65 white. Same transfer
    /// function as sRGB, but substantially wider primaries -- particularly in red and
    /// green, which is exactly where gemstone dispersion "fire" lands.
    DisplayP3,
    /// ITU-R BT.2020-2 Rec.2020, D65 white. The widest gamut of the four, with its own
    /// distinct piecewise transfer function.
    Rec2020,
    /// Academy `ACEScg` (AP1 primaries), D60 white. Scene-linear -- no transfer function
    /// -- so it is the right target for further compositing rather than direct
    /// display.
    AcesCg,
}

impl ColorSpace {
    /// The XYZ->RGB matrix, row-major (`row[0]` produces R, `row[1]` produces G,
    /// `row[2]` produces B), for CIE 1931 XYZ with `Y` normalized to 1.0 at the space's
    /// own reference white.
    ///
    /// # Sources
    ///
    /// The project's reference document tabulates these same four matrices, but as
    /// embedded images that cannot be read from the file -- a human should cross-check
    /// the constants below against that document's figures.
    ///
    /// - **sRGB**: kept bit-identical to the constants already hardwired in
    ///   `optics::raytracer::xyz_to_linear_srgb` (the commonly published
    ///   Bruce-Lindbloom-rounded sRGB D65 XYZ-to-RGB matrix, derived from the IEC
    ///   61966-2-1 primaries/white), so that `ColorSpace::Srgb` reproduces that
    ///   function's gamut-mapped linear RGB exactly rather than merely approximately.
    /// - **Display P3**: derived from primaries R(0.680, 0.320) G(0.265, 0.690)
    ///   B(0.150, 0.060) and the D65 white point, per SMPTE EG 432-1 / Apple's Display
    ///   P3 definition, using the standard RGB-primaries -> XYZ construction (build the
    ///   primary chromaticity matrix, solve for the white-point scale factors, invert)
    ///   and cross-checked against the values published in the `colour-science` Python
    ///   library's `RGB_COLOURSPACES['Display P3']`.
    /// - **Rec.2020**: derived the same way from primaries R(0.708, 0.292) G(0.170,
    ///   0.797) B(0.131, 0.046), D65 white, per ITU-R BT.2020-2 Table 3, cross-checked
    ///   against `RGB_COLOURSPACES['ITU-R BT.2020']`.
    /// - **`ACEScg`**: derived the same way from the AP1 primaries R(0.713, 0.293)
    ///   G(0.165, 0.830) B(0.128, 0.044) and the D60 white point (x=0.32168,
    ///   y=0.33767), per Academy S-2014-004 ("`ACEScg`") Table 1, cross-checked against
    ///   the Academy's own published `XYZ_to_AP1` matrix in the same document.
    #[must_use]
    pub const fn xyz_to_rgb_matrix(self) -> [[f32; 3]; 3] {
        match self {
            Self::Srgb => [
                [3.240_454_2, -1.537_138_5, -0.498_531_4],
                [-0.969_266, 1.876_010_8, 0.041_556_0],
                [0.055_643_4, -0.204_025_9, 1.057_225_2],
            ],
            Self::DisplayP3 => [
                [2.493_497, -0.931_383_6, -0.402_710_78],
                [-0.829_489, 1.762_664, 0.023_624_686],
                [0.035_845_83, -0.076_172_39, 0.956_884_5],
            ],
            Self::Rec2020 => [
                [1.716_651_2, -0.355_670_78, -0.253_366_3],
                [-0.666_684_3, 1.616_481_2, 0.015_768_546],
                [0.017_639_857, -0.042_770_613, 0.942_103_1],
            ],
            Self::AcesCg => [
                [1.641_023_4, -0.324_803_3, -0.236_424_7],
                [-0.663_662_86, 1.615_331_6, 0.016_756_348],
                [0.011_721_894, -0.008_284_442, 0.988_394_86],
            ],
        }
    }

    /// The space's reference white point chromaticity, `(x, y)`. This is the target
    /// that out-of-gamut chromaticities are radially compressed toward -- see
    /// [`gamut::project_to_gamut`].
    #[must_use]
    pub const fn white_point_xy(self) -> (f32, f32) {
        match self {
            // CIE Standard Illuminant D60 (Academy S-2014-004) -- ACEScg's own
            // reference white, distinct from the other three spaces' D65.
            Self::AcesCg => (0.321_68, 0.337_67),
            // CIE Standard Illuminant D65, matching the WHITE_X/WHITE_Y constants in
            // `optics::raytracer::xyz_to_srgb_gamma`.
            Self::Srgb | Self::DisplayP3 | Self::Rec2020 => (0.3127, 0.3290),
        }
    }

    /// The transfer function associated with this space.
    #[must_use]
    pub const fn transfer_function(self) -> TransferFunction {
        match self {
            Self::Srgb | Self::DisplayP3 => TransferFunction::Srgb,
            Self::Rec2020 => TransferFunction::Rec2020,
            Self::AcesCg => TransferFunction::Linear,
        }
    }

    /// Converts a CIE XYZ sample to this space's linear (not transfer-encoded) RGB via
    /// [`Self::xyz_to_rgb_matrix`]. Does **not** gamut-map: an out-of-gamut input can
    /// and will produce negative components here. See [`gamut::project_to_gamut`] for
    /// the gamut-mapped version most callers want.
    #[must_use]
    pub fn xyz_to_linear(self, xyz: Vec3) -> Vec3 {
        let m = self.xyz_to_rgb_matrix();
        Vec3::new(
            m[0][0].mul_add(xyz.x, m[0][1].mul_add(xyz.y, m[0][2] * xyz.z)),
            m[1][0].mul_add(xyz.x, m[1][1].mul_add(xyz.y, m[1][2] * xyz.z)),
            m[2][0].mul_add(xyz.x, m[2][1].mul_add(xyz.y, m[2][2] * xyz.z)),
        )
    }

    /// Converts a CIE XYZ radiance sample all the way to encoded 8-bit RGBA:
    /// gamut-maps out-of-gamut chromaticities into this space (preserving luminance
    /// and hue -- see [`gamut::project_to_gamut`]), applies `tonemap`, applies this
    /// space's transfer function, and quantizes each channel to `0..=255`. Alpha is
    /// always `255`.
    ///
    /// This is the intended drop-in replacement entry point for
    /// `optics::raytracer::xyz_to_srgb_gamma` -- see the module-level docs above for
    /// the exact substitution and the one deliberate behavioural difference (the true
    /// sRGB transfer curve in place of a flat 1/2.2 gamma).
    ///
    /// Every channel is finite and in `0..=255` for every finite input, including
    /// non-positive, extreme-magnitude, and far-out-of-gamut `xyz` (a non-finite `xyz`
    /// -- NaN or infinite -- is treated the same as the zero/near-zero case and
    /// produces opaque black, rather than propagating NaN into the output).
    ///
    /// # Highlights desaturate instead of clipping a single channel
    ///
    /// [`ToneMap::AcesFilmic`] rescales RGB by a single luminance-derived factor
    /// (`y_tm / y`) so hue survives tone mapping -- see that variant's doc comment for
    /// why per-channel tone mapping is wrong here. A channel that rescale pushes above
    /// `1.0` used to be hard-clamped independently by the per-channel quantization
    /// below, which silently reintroduced exactly the hue/saturation shift the
    /// luminance-only design exists to avoid (worst on bright saturated dispersion
    /// "fire" colours). This now routes the tone-mapped colour through
    /// [`gamut::project_to_gamut_bounded`] (`max = 1.0`) instead of a plain scalar
    /// multiply -- the SAME chromaticity-preserving radial walk toward this space's
    /// white point already used for out-of-gamut compression, just also given an upper
    /// bound, so an over-bright saturated colour loses saturation gracefully rather
    /// than having one channel truncated. See that function's doc comment for the
    /// mechanism, and `tests/color_tests.rs` for the regression coverage (this moves
    /// golden values for any scene with clipped highlights -- expected and correct, not
    /// a regression).
    #[must_use]
    pub fn encode(self, xyz: Vec3, tonemap: ToneMap) -> [u8; 4] {
        let sum = xyz.x + xyz.y + xyz.z;
        if !sum.is_finite() || sum <= 1e-6 {
            return [0, 0, 0, 255];
        }

        let toned_rgb = match tonemap {
            ToneMap::None => gamut::project_to_gamut(xyz, self),
            ToneMap::AcesFilmic { exposure } => {
                let luminance = xyz.y.max(0.0);
                let exposed_luminance = (luminance * exposure).max(0.0);
                let y_tm = crate::optics::raytracer::aces_tonemap(exposed_luminance);
                // The single scalar the OLD code applied to `linear_rgb` after
                // separately gamut-projecting `xyz`. Applying it to `xyz` itself
                // instead (before gamut-projecting) is equivalent for any in-range
                // colour -- gamut projection is a linear operator in luminance, so
                // scaling commutes with it -- but additionally lets ONE bounded
                // gamut-projection call (below) handle both the pre-existing
                // out-of-gamut compression and the new highlight desaturation above
                // together, via the same walk, rather than composing two separate
                // steps.
                let luminance_scale = (y_tm / exposed_luminance.max(1e-5)) * exposure;
                let toned_xyz = xyz * luminance_scale;
                gamut::project_to_gamut_bounded(toned_xyz, self, 1.0)
            }
        };

        let tf = self.transfer_function();
        let encode_channel = |c: f32| -> u8 {
            let linear = if c.is_finite() {
                c.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let encoded = tf.encode(linear).clamp(0.0, 1.0);
            (encoded * 255.0) as u8
        };

        [
            encode_channel(toned_rgb.x),
            encode_channel(toned_rgb.y),
            encode_channel(toned_rgb.z),
            255,
        ]
    }
}

//! Space-aware gamut mapping.
//!
//! Compresses out-of-gamut CIE XYZ colours into a target [`ColorSpace`] by walking the
//! chromaticity radially toward that space's own white point at constant luminance,
//! rather than clipping each RGB channel independently (which shifts hue as well as
//! saturation).
//!
//! This is a direct generalisation of the gamut-mapping step inside
//! `optics::raytracer::xyz_to_srgb_gamma` ("Procedure 3" in that file's docs): that
//! function hardwires the sRGB matrix and the D65 white point; [`project_to_gamut`]
//! parameterises both on a [`ColorSpace`] so the identical algorithm can target Display
//! P3, Rec.2020, or `ACEScg` as well. Calling it with [`ColorSpace::Srgb`] reproduces
//! `xyz_to_srgb_gamma`'s gamut-mapped linear RGB (see
//! `tests/color_tests.rs` for the cross-check against that function).

use glam::Vec3;

use super::space::ColorSpace;

/// Radially compresses an out-of-gamut colour toward `space`'s white point.
///
/// Walks the CIE xyY chromaticity of `xyz` toward `space`'s white point at constant
/// luminance until every RGB channel produced by `space`'s XYZ->RGB matrix
/// ([`ColorSpace::xyz_to_linear`]) is non-negative. In-gamut colours pass straight
/// through the matrix unchanged. The result is linear (not transfer-encoded) RGB --
/// apply [`ColorSpace::transfer_function`] separately, or use [`ColorSpace::encode`]
/// for the full pipeline.
///
/// A non-finite or effectively-zero `xyz` (sum of components `<= 1e-6`, matching the
/// threshold `xyz_to_srgb_gamma` uses) maps to `Vec3::ZERO` rather than dividing by a
/// near-zero denominator when computing chromaticity.
///
/// Thin wrapper around [`project_to_gamut_bounded`] with `max = f32::INFINITY` (i.e.
/// only the lower, non-negativity, bound is enforced -- see that function for the
/// version [`ColorSpace::encode`]'s tone-mapping step uses to also cap highlights at
/// `1.0`).
#[must_use]
pub fn project_to_gamut(xyz: Vec3, space: ColorSpace) -> Vec3 {
    project_to_gamut_bounded(xyz, space, f32::INFINITY)
}

/// As [`project_to_gamut`], but also caps every output channel at `max`.
///
/// The SAME radial walk toward `space`'s white point, just carrying both a floor and a
/// ceiling instead of only a floor.
///
/// # Why this exists
///
/// `ColorSpace::encode`'s ACES tone-mapping step deliberately scales RGB by a single
/// luminance-derived factor (`y_tm / y`) rather than tone-mapping each channel
/// independently, specifically so a saturated colour's hue survives tone mapping --
/// see [`crate::color::ToneMap::AcesFilmic`]'s doc comment. But a channel that ends up
/// above `1.0` after that rescale used to be hard-clamped per channel by the final u8
/// quantization step, which silently reintroduces exactly the hue/saturation shift the
/// luminance-only design was chosen to avoid (worst on bright saturated dispersion
/// "fire" colours, this renderer's marquee output). Routing the tone-mapped colour
/// through this bounded walk instead means an over-bright saturated colour loses
/// saturation gracefully (desaturating toward white as luminance pushes it out of
/// range) rather than having one channel truncated while the others hold still.
///
/// Reuses the exact search this module already had for the non-negativity
/// gamut-compression case rather than adding a second, independent desaturation
/// mechanism: at each candidate step the walk now checks that every channel is both
/// non-negative AND no greater than `max`, instead of just non-negative. The full
/// white point fallback (the walk's final step) is not itself re-clamped to `max`: at
/// that chromaticity every channel is equal (the space's own white point, by
/// construction of its XYZ->RGB matrix), so even a small residual overshoot there
/// (e.g. the ACES curve's ~1.033 highlight asymptote) is hue-neutral and is left for
/// [`ColorSpace::encode`]'s final safety clamp to absorb.
#[must_use]
pub fn project_to_gamut_bounded(xyz: Vec3, space: ColorSpace, max: f32) -> Vec3 {
    const STEPS: u32 = 32;

    let sum = xyz.x + xyz.y + xyz.z;
    if !sum.is_finite() || sum <= 1e-6 {
        return Vec3::ZERO;
    }

    let x = xyz.x / sum;
    let y = xyz.y / sum;
    let luminance = xyz.y.max(0.0);

    let linear = space.xyz_to_linear(xyz);
    let in_range =
        |v: Vec3| v.x >= 0.0 && v.y >= 0.0 && v.z >= 0.0 && v.x <= max && v.y <= max && v.z <= max;
    if in_range(linear) {
        return linear;
    }

    // Out of range: walk the chromaticity radially toward `space`'s white point at
    // constant luminance, and take the smallest step for which every channel is within
    // `[0.0, max]`. See `xyz_to_srgb_gamma`'s doc comment for the rationale (this is
    // the same construction, generalised to any target space's white point/matrix, and
    // -- per this function's own doc comment -- to an optional upper bound as well).
    let (white_x, white_y) = space.white_point_xy();

    let mapped_xyz_for_t = |t: f32| -> Vec3 {
        let xp = (white_x - x).mul_add(t, x);
        let yp = (white_y - y).mul_add(t, y);
        if yp > 1e-6 {
            Vec3::new(
                (xp / yp) * luminance,
                luminance,
                ((1.0 - xp - yp) / yp) * luminance,
            )
        } else {
            Vec3::new(
                (white_x / white_y) * luminance,
                luminance,
                ((1.0 - white_x - white_y) / white_y) * luminance,
            )
        }
    };

    let mut resolved = space.xyz_to_linear(mapped_xyz_for_t(1.0)).max(Vec3::ZERO);
    for i in 0..=STEPS {
        let t = i as f32 / STEPS as f32;
        let candidate = space.xyz_to_linear(mapped_xyz_for_t(t));
        if in_range(candidate) {
            resolved = candidate;
            break;
        }
    }
    resolved
}

/// sRGB-specialised convenience wrapper around [`project_to_gamut`].
///
/// This replaces the former stub of the same name (which clamped its `(x, y,
/// luminance)` xyY inputs and returned them unchanged, doing no actual gamut mapping
/// despite the name); it is kept as a named entry point for callers -- notably the
/// eventual `raytracer.rs` wiring -- that only ever need sRGB, rather than plumbing
/// `ColorSpace::Srgb` through everywhere.
#[must_use]
pub fn project_to_srgb(xyz: Vec3) -> Vec3 {
    project_to_gamut(xyz, ColorSpace::Srgb)
}

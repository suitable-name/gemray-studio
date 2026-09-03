//! Angle <-> vector conversion for the settings dialog's crystal-axis orientation
//! override.
//!
//! `GemMaterial::c_axis` is a raw `Vec3`, but a *direction* only has two degrees of
//! freedom -- three free `f32` sliders would over-parameterize it and admit invalid
//! states (zero-length, non-unit) the physics would then have to defend against. So the
//! settings dialog drags two angles instead, exactly like the light-position controls
//! already do (`light_yaw_deg`/`light_pitch_deg`):
//!
//! - `tilt_deg` (theta): the angle from the table normal (`+Y`), 0-90 degrees. This is
//!   how lapidaries actually describe a cut-orientation choice -- see the long comment
//!   on Tourmaline's `c_axis: Vec3::X` in `crates/gemray/src/optics/materials.rs`,
//!   which frames it as "cutters orient the table perpendicular to the c-axis", i.e. in
//!   terms of tilt from the table plane, not raw XYZ.
//! - `azimuth_deg` (phi): rotation around `+Y`, 0-360 degrees.
//!
//! `c_axis = (sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi))`.
//!
//! No dependency on Slint or any other crate -- pure trigonometry, exercised directly
//! by the unit tests without spinning up a UI, matching this app's usual
//! `settings::model` convention of keeping plain-data/logic separate from the UI layer,
//! and this module's own sibling `gui::sample_scale`'s precedent for where such a
//! helper lives and how it's tested.

use glam::Vec3;

/// `(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi))` -- see the module doc
/// comment for the two angles' meaning. `tilt_deg`/`azimuth_deg` are not clamped here:
/// the settings-dialog sliders themselves already constrain them (0-90, 0-360), and
/// the trig functions below are well-defined for any real input regardless, so an
/// out-of-range value (e.g. a hand-edited settings file) just resolves to whatever
/// direction the formula gives rather than panicking.
///
/// `tilt_deg = 0.0` collapses to `Vec3::Y` for ANY `azimuth_deg` (`sin(0) == 0`
/// zeroes both the X and Z terms exactly in `f32`, regardless of `cos`/`sin` of
/// `azimuth_deg`) -- every built-in material's own default. `tilt_deg = 90.0`,
/// `azimuth_deg = 0.0` gives `Vec3::X` (Tourmaline's own cut-orientation override) to
/// within `f32` trig precision -- see
/// `tests::tilt_ninety_azimuth_zero_matches_tourmaline_x_axis` for exactly how close.
#[must_use]
pub fn angles_to_c_axis(tilt_deg: f32, azimuth_deg: f32) -> Vec3 {
    let theta = tilt_deg.to_radians();
    let phi = azimuth_deg.to_radians();
    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();
    Vec3::new(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi)
}

/// Inverse of [`angles_to_c_axis`]: recovers `(tilt_deg, azimuth_deg)` from a `c_axis`
/// direction. Used exactly once per toggle -- when the settings-dialog override switch
/// flips from off to on, `gui::mod` calls this on the currently selected material's OWN
/// `c_axis` to seed the two sliders, so enabling the override never makes the stone
/// visibly jump (see `AppSettings::c_axis_override_enabled`'s own doc comment).
///
/// `axis` need not already be unit-length (defensively normalized below) -- every
/// built-in material's `c_axis` already is, but this stays correct even if a future
/// caller hands it something else. A degenerate (zero-length) input falls back to
/// `Vec3::Y` (`tilt_deg = 0.0`) rather than dividing by zero.
///
/// The azimuth is mathematically undefined exactly at the poles (`tilt_deg == 0.0` or
/// `180.0`, where every azimuth maps to the same point) -- resolved to `0.0` there via
/// `atan2`'s own `atan2(0, 0) == 0` convention, matching `angles_to_c_axis(0.0, _)`'s
/// own azimuth-independence at that pole.
#[must_use]
pub fn c_axis_to_angles(axis: Vec3) -> (f32, f32) {
    let axis = if axis.length_squared() > 1e-12 {
        axis.normalize()
    } else {
        Vec3::Y
    };
    let tilt_deg = axis.y.clamp(-1.0, 1.0).acos().to_degrees();
    let azimuth_deg = axis.z.atan2(axis.x).to_degrees();
    let azimuth_deg = if azimuth_deg < 0.0 {
        azimuth_deg + 360.0
    } else {
        azimuth_deg
    };
    (tilt_deg, azimuth_deg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The strongest correctness anchor available (per this task's own brief): `tilt_deg
    /// = 0.0` must reproduce every built-in material's own default `c_axis` exactly, for
    /// ANY azimuth -- `sin(0.0.to_radians()) == 0.0` exactly in `f32` (no trig rounding
    /// noise at this particular endpoint), so both the X and Z terms are an exact `0.0 *
    /// anything == 0.0`, not merely close.
    #[test]
    fn tilt_zero_reproduces_default_y_axis_exactly_for_any_azimuth() {
        for azimuth_deg in [0.0, 45.0, 90.0, 123.4, 270.0, 359.9] {
            assert_eq!(
                angles_to_c_axis(0.0, azimuth_deg),
                Vec3::Y,
                "azimuth_deg={azimuth_deg}"
            );
        }
    }

    /// The other endpoint: `tilt_deg = 90.0, azimuth_deg = 0.0` must reproduce
    /// Tourmaline's own `c_axis: Vec3::X` cut-orientation override. Not bit-exact --
    /// `90.0f32.to_radians()` is itself an approximation of the true pi/2, so
    /// `cos(theta)` lands at roughly `-4.4e-7`, not exactly `0.0` -- but well within a
    /// tolerance many orders of magnitude tighter than anything visually or physically
    /// meaningful (`birefringence`/`absorption` effects this axis drives operate at the
    /// 1e-2..1e0 scale).
    #[test]
    fn tilt_ninety_azimuth_zero_matches_tourmaline_x_axis() {
        let axis = angles_to_c_axis(90.0, 0.0);
        assert!(
            (axis - Vec3::X).length() < 1e-6,
            "expected ~=Vec3::X, got {axis:?}"
        );
    }

    #[test]
    fn c_axis_to_angles_recovers_y_axis_as_zero_tilt() {
        let (tilt_deg, _azimuth_deg) = c_axis_to_angles(Vec3::Y);
        assert!(tilt_deg.abs() < 1e-4, "tilt_deg={tilt_deg}");
    }

    #[test]
    fn c_axis_to_angles_recovers_x_axis_as_ninety_tilt_zero_azimuth() {
        let (tilt_deg, azimuth_deg) = c_axis_to_angles(Vec3::X);
        assert!((tilt_deg - 90.0).abs() < 1e-3, "tilt_deg={tilt_deg}");
        assert!(azimuth_deg.abs() < 1e-3, "azimuth_deg={azimuth_deg}");
    }

    /// Degenerate (zero-length) input must not panic or divide by zero -- falls back to
    /// the every-material default instead.
    #[test]
    fn c_axis_to_angles_handles_zero_length_input_without_panicking() {
        let (tilt_deg, _azimuth_deg) = c_axis_to_angles(Vec3::ZERO);
        assert!(tilt_deg.abs() < 1e-4, "tilt_deg={tilt_deg}");
    }

    /// Round trip, angles -> vector -> angles, across the sliders' full legal range.
    /// Skips `tilt_deg == 0.0` (azimuth is mathematically undefined at that pole -- see
    /// `c_axis_to_angles`'s own doc comment) and the `azimuth_deg == 360.0` boundary
    /// (equivalent to `0.0`, which is what `c_axis_to_angles`'s `atan2` normalization
    /// actually returns there).
    #[test]
    fn round_trips_angles_through_vector_and_back() {
        for tilt_deg in [1.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0, 90.0] {
            for azimuth_deg in [0.0, 30.0, 90.0, 145.0, 180.0, 250.0, 300.0, 359.0] {
                let axis = angles_to_c_axis(tilt_deg, azimuth_deg);
                let (got_tilt, got_azimuth) = c_axis_to_angles(axis);
                assert!(
                    (got_tilt - tilt_deg).abs() < 1e-2,
                    "tilt_deg={tilt_deg} azimuth_deg={azimuth_deg} got_tilt={got_tilt}"
                );
                assert!(
                    (got_azimuth - azimuth_deg).abs() < 1e-2,
                    "tilt_deg={tilt_deg} azimuth_deg={azimuth_deg} got_azimuth={got_azimuth}"
                );
            }
        }
    }

    /// The other round-trip direction: vector -> angles -> vector, for a handful of
    /// arbitrary unit directions (not just the two axis-aligned anchors above).
    #[test]
    fn round_trips_vectors_through_angles_and_back() {
        let directions = [
            Vec3::Y,
            Vec3::X,
            Vec3::new(0.5, 0.5, 0.72).normalize(),
            Vec3::new(-0.3, 0.8, 0.5).normalize(),
            Vec3::new(0.6, -0.2, 0.776_25).normalize(),
        ];
        for axis in directions {
            let (tilt_deg, azimuth_deg) = c_axis_to_angles(axis);
            let round_tripped = angles_to_c_axis(tilt_deg, azimuth_deg);
            assert!(
                (round_tripped - axis).length() < 1e-4,
                "axis={axis:?} round_tripped={round_tripped:?}"
            );
        }
    }
}

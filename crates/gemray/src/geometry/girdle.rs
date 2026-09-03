//! Girdle-facet classification for the per-facet frosted/bruted-girdle finish.
//!
//! `optics::raytracer::FacetFinish::Frosted` and `renderer::buffers::encode_facet_finishes`
//! already know how to render a per-facet finish; what has been missing is any way to
//! say *which* facets are the girdle band for a design the app didn't hand-author --
//! see [`super::cuts::STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS`]'s own doc comment,
//! which is correct only for
//! [`super::cuts::StandardGemCuts::standard_round_brilliant`]'s one fixed
//! construction order.
//!
//! # Why plane geometry, not the cutting schedule
//!
//! [`super::meet_solver::classify_blocks`] already classifies
//! `Block::{Crown, Pavilion, Girdle}`, but only at the tier (cutting-schedule) level --
//! it needs a `&[MeetTierInput]`, which not every caller has. The renderer, however,
//! always works in [`GpuFacetPlane`]s, regardless of whether they came from an `.asc`
//! schedule, `from_database_angles`, or a hand-built cut. A girdle facet's normal is,
//! by construction, perpendicular to the stone's `+Y` symmetry axis (see e.g.
//! `standard_round_brilliant`'s own "16 Girdle Facets (90.0 deg vertical cylinder)"
//! comment, and `push_girdle_facets`'s literal `y == 0.0` normals in `emerald_cut`) --
//! directly readable from the plane alone, with no schedule needed at all. This module
//! classifies from the plane, and stays consistent with `classify_blocks`'s own girdle
//! rule (the same "normal is exactly horizontal" criterion it applies as
//! `y.abs() <= 1e-6` to an angle recovered from schedule text) rather than inventing a
//! second definition of "girdle" -- see [`GIRDLE_NORMAL_Y_EPSILON`] for why the
//! threshold value itself differs from that one.

use super::plane::GpuFacetPlane;
use crate::optics::raytracer::FacetFinish;

/// How close a (unit, `f32`) plane normal's `y`-component must be to zero to count as
/// girdle -- i.e. how close the facet is to perfectly horizontal / perpendicular to
/// the stone's `+Y` symmetry axis.
///
/// [`super::meet_solver::classify_blocks`] uses `y.abs() <= 1e-6`, but that `y` is
/// `cos(theta)` computed directly from an exact schedule-angle value in `f64`. Here
/// `y` is instead a [`GpuFacetPlane`] normal's own `f32` component, which has already
/// passed through a construction pipeline (angle -> `sin`/`cos` -> `Vec3::normalize`)
/// that leaves per-operation rounding noise even for an intended-exact 90 degree
/// facet -- verified at ~5e-8 for `StandardGemCuts::from_asc_schedule`'s 90 degree
/// girdle tiers, and exactly `0.0` for `standard_round_brilliant`'s and
/// `emerald_cut`'s girdle planes, which are constructed with a literal `0.0`
/// y-component rather than via `cos`. `1e-3` sits several orders of magnitude above
/// that noise floor, and several orders of magnitude below the shallowest real
/// non-girdle facet in either built-in cut (SRB's lower girdle break at
/// `|y| ~= 0.737`; `emerald_cut`'s pavilion Step 1 at `|y| ~= 0.602`), so it cannot
/// confuse a genuinely steep crown/pavilion facet for the girdle while still
/// tolerating realistic per-source float noise.
const GIRDLE_NORMAL_Y_EPSILON: f32 = 1e-3;

/// True iff `plane`'s normal is close enough to horizontal (perpendicular to the
/// stone's `+Y` symmetry axis) to count as a girdle facet. See
/// [`GIRDLE_NORMAL_Y_EPSILON`].
fn is_girdle_plane(plane: &GpuFacetPlane) -> bool {
    plane.normal[1].abs() <= GIRDLE_NORMAL_Y_EPSILON
}

/// Returns the plane indices (into `planes`, ascending) that classify as girdle
/// facets.
///
/// Deterministic and order-preserving: a plain forward scan, no hashing, so the same
/// `planes` slice always yields byte-identical output. An empty slice, or one with no
/// near-horizontal plane at all, returns an empty `Vec` rather than panicking.
///
/// Cross-checked against [`super::cuts::STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS`]: this
/// classifier applied to
/// [`super::cuts::StandardGemCuts::standard_round_brilliant`]'s own planes must
/// reproduce that constant's `33..49` exactly -- see this module's tests.
#[must_use]
pub fn classify_girdle_plane_indices(planes: &[GpuFacetPlane]) -> Vec<usize> {
    planes
        .iter()
        .enumerate()
        .filter_map(|(i, p)| is_girdle_plane(p).then_some(i))
        .collect()
}

/// Builds a ready `Vec<FacetFinish>` for a bruted-girdle variant of `planes`.
///
/// Sized to `planes.len()`, with every girdle facet (per
/// [`classify_girdle_plane_indices`]) set to [`FacetFinish::Frosted`] and every other
/// index left at [`FacetFinish::Polished`] (its `#[default]`).
///
/// This is the exact shape `optics::raytracer::trace_spectral_ray_with_finish`'s
/// `facet_finishes` argument and `renderer::buffers::encode_facet_finishes` already
/// consume, so a caller wanting a bruted-girdle variant of an arbitrary design needs
/// only this one call, not a loop of its own.
#[must_use]
pub fn girdle_facet_finishes(planes: &[GpuFacetPlane]) -> Vec<FacetFinish> {
    planes
        .iter()
        .map(|p| {
            if is_girdle_plane(p) {
                FacetFinish::Frosted
            } else {
                FacetFinish::Polished
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cuts::{STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS, StandardGemCuts};
    use glam::Vec3;

    /// The strongest cross-check available: the one design whose girdle band is
    /// already written down as a hand-verified constant.
    /// [`STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS`]'s doc comment lays out
    /// `standard_round_brilliant`'s own construction order (table, star, crown main,
    /// upper girdle break -- none of which are the girdle despite some names sounding
    /// like it -- then the 16 true girdle facets at `33..49`). If this classifier
    /// disagrees with that constant, the classifier is wrong, not the constant.
    #[test]
    fn matches_standard_round_brilliant_girdle_constant() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let got = classify_girdle_plane_indices(&planes);
        let expected: Vec<usize> = STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS.collect();
        assert_eq!(got, expected);
    }

    /// [`girdle_facet_finishes`] must mark exactly those same indices `Frosted`, and
    /// nothing else, on the same design.
    #[test]
    fn srb_finishes_mark_exactly_the_girdle_band() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let finishes = girdle_facet_finishes(&planes);
        assert_eq!(finishes.len(), planes.len());
        for (i, finish) in finishes.iter().enumerate() {
            let expected = if STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS.contains(&i) {
                FacetFinish::Frosted
            } else {
                FacetFinish::Polished
            };
            assert_eq!(*finish, expected, "index {i}");
        }
    }

    /// `emerald_cut`'s girdle band sits at a completely different position in its
    /// plane list (`13..21`, 8 facets from `push_girdle_facets`) than the SRB's
    /// `33..49` -- proving the classifier reads the plane geometry itself rather than
    /// accidentally keying off position. See `push_girdle_facets`'s call site in
    /// `emerald_cut` for the construction order this expects (table; 8 crown tier
    /// facets; 4 crown corner facets; 8 girdle facets; 8 pavilion tier facets; 4
    /// pavilion corner facets; keel).
    #[test]
    fn emerald_cut_girdle_is_a_different_range_than_srb() {
        let planes = StandardGemCuts::emerald_cut();
        let got = classify_girdle_plane_indices(&planes);
        let expected: Vec<usize> = (13..21).collect();
        assert_eq!(got, expected);
        assert_ne!(
            got,
            STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS.collect::<Vec<_>>()
        );
    }

    /// A design with no near-horizontal facet at all (e.g. a small tilted wedge with
    /// nothing resembling a girdle wall) must return an empty result, not panic.
    #[test]
    fn no_girdle_facets_returns_empty() {
        let planes = vec![
            GpuFacetPlane::new(Vec3::new(0.0, 1.0, 0.0), -1.0),
            GpuFacetPlane::new(Vec3::new(0.0, -1.0, 0.0), -1.0),
            GpuFacetPlane::new(Vec3::new(1.0, 1.0, 0.0), -1.0),
            GpuFacetPlane::new(Vec3::new(-1.0, 1.0, 0.0), -1.0),
        ];
        assert_eq!(classify_girdle_plane_indices(&planes), Vec::<usize>::new());
        assert_eq!(
            girdle_facet_finishes(&planes),
            vec![FacetFinish::Polished; planes.len()]
        );
    }

    /// An empty plane slice must not panic, for either entry point.
    #[test]
    fn empty_planes_does_not_panic() {
        assert_eq!(classify_girdle_plane_indices(&[]), Vec::<usize>::new());
        assert_eq!(girdle_facet_finishes(&[]), Vec::<FacetFinish>::new());
    }

    /// Same input, same output, bit-for-bit -- no iteration-order or float-ordering
    /// nondeterminism anywhere in this module (a plain forward scan, no hashing).
    #[test]
    fn deterministic_across_repeated_calls() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let a = classify_girdle_plane_indices(&planes);
        let b = classify_girdle_plane_indices(&planes);
        assert_eq!(a, b);

        let fa = girdle_facet_finishes(&planes);
        let fb = girdle_facet_finishes(&planes);
        assert_eq!(fa, fb);
    }
}

use super::{brep::GemPolyhedron, plane::GpuFacetPlane};
use glam::Vec3;
use lapidary::asc::AscSchedule;
use std::{collections::HashSet, f32::consts::PI};
use tracing::warn;

/// A single row of a GemCAD-style cutting schedule: one facet's angle, index
/// position(s), and any notes.
///
/// Exactly as scraped/stored (plain strings, no numeric parsing done yet -- see
/// `parse_angle_deg` / `parse_girdle_facet_count` below for the lenient parsing of
/// these fields).
///
/// This is a plain, UI-toolkit-agnostic type so that callers (e.g. a Slint-based
/// viewer) can convert their own generated row type into this one at the call
/// site, keeping this crate free of any UI dependency.
#[derive(Debug, Clone, Default)]
pub struct FacetSpec {
    pub facet: String,
    pub angle: String,
    pub index: String,
    pub notes: String,
}

/// Girdle finish: the facet-index range of the girdle band.
///
/// Within [`StandardGemCuts::standard_round_brilliant`]'s own construction order, the 16
/// vertical prism facets making up the physical girdle band -- see that function's own
/// "5. 16 Girdle Facets (90.0° vertical cylinder of radius 1.0)" comment for the layout
/// this indexes into (table: 1, star: 8, crown main: 8, upper girdle break: 16 -- NONE
/// of which are the girdle itself, despite the name -- THEN the 16 true girdle facets,
/// at indices 33..=48). Exposed so a caller building a bruted/frosted-girdle variant of
/// this cut (e.g. `optics::raytracer::trace_spectral_ray_with_finish`'s
/// `facet_finishes` argument) knows which facet indices to mark
/// `optics::raytracer::FacetFinish::Frosted` without hand-deriving them -- and so that
/// derivation stays correct automatically if this cut's facet-push order ever changes.
pub const STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS: std::ops::Range<usize> = 33..49;

pub struct StandardGemCuts;

impl StandardGemCuts {
    /// Generates exact 3D half-space planes for a 57-facet Standard Round Brilliant (SRB) Diamond cut
    /// with standard ideal proportions (Crown height ~15% diameter, Table ~56% width, Pavilion depth ~43%).
    #[must_use]
    pub fn standard_round_brilliant() -> Vec<GpuFacetPlane> {
        let mut planes = Vec::with_capacity(57);
        let gear_teeth = 96.0f32;

        // 1. Crown Table Facet (Top flat facet at Y = +0.32)
        planes.push(GpuFacetPlane::new(Vec3::new(0.0, 1.0, 0.0), -0.32));

        // 2. 8 Crown Star Facets (15.0°, index 6, 18, 30, 42, 54, 66, 78, 90)
        let star_angle = 15.0f32.to_radians();
        for &g in &[6.0, 18.0, 30.0, 42.0, 54.0, 66.0, 78.0, 90.0] {
            let phi = 2.0 * PI * g / gear_teeth;
            let n = Vec3::new(
                star_angle.sin() * phi.cos(),
                star_angle.cos(),
                star_angle.sin() * phi.sin(),
            );
            planes.push(GpuFacetPlane::new(n, -0.45));
        }

        // 3. 8 Crown Kite / Main Facets (34.5°, index 96, 12, 24, 36, 48, 60, 72, 84)
        let crown_main_angle = 34.5f32.to_radians();
        for &g in &[0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0] {
            let phi = 2.0 * PI * g / gear_teeth;
            let n = Vec3::new(
                crown_main_angle.sin() * phi.cos(),
                crown_main_angle.cos(),
                crown_main_angle.sin() * phi.sin(),
            );
            planes.push(GpuFacetPlane::new(n, -0.59));
        }

        // 4. 16 Upper Girdle Break Facets (41.0°, index 95, 1, 11, 13, 23, 25, 35, 37, 47, 49, 59, 61, 71, 73, 83, 85)
        let upper_girdle_angle = 41.0f32.to_radians();
        for &g in &[
            95.0, 1.0, 11.0, 13.0, 23.0, 25.0, 35.0, 37.0, 47.0, 49.0, 59.0, 61.0, 71.0, 73.0,
            83.0, 85.0,
        ] {
            let phi = 2.0 * PI * g / gear_teeth;
            let n = Vec3::new(
                upper_girdle_angle.sin() * phi.cos(),
                upper_girdle_angle.cos(),
                upper_girdle_angle.sin() * phi.sin(),
            );
            planes.push(GpuFacetPlane::new(n, -0.67));
        }

        // 5. 16 Girdle Facets (90.0° vertical cylinder of radius 1.0)
        for i in 0..16 {
            let phi = 2.0 * PI * (i as f32) / 16.0;
            let n = Vec3::new(phi.cos(), 0.0, phi.sin());
            planes.push(GpuFacetPlane::new(n, -1.0));
        }

        // 6. 8 Pavilion Main Facets (-41.0°, index 96, 12, 24, 36, 48, 60, 72, 84)
        let pav_main_angle = 41.0f32.to_radians();
        for &g in &[0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0] {
            let phi = 2.0 * PI * g / gear_teeth;
            let n = Vec3::new(
                pav_main_angle.sin() * phi.cos(),
                -pav_main_angle.cos(),
                pav_main_angle.sin() * phi.sin(),
            );
            planes.push(GpuFacetPlane::new(n, -0.67));
        }

        // 7. 16 Lower Girdle Break Facets (-42.5°, index 95, 1, 11, 13, 23, 25, 35, 37, 47, 49, 59, 61, 71, 73, 83, 85)
        let lower_girdle_angle = 42.5f32.to_radians();
        for &g in &[
            95.0, 1.0, 11.0, 13.0, 23.0, 25.0, 35.0, 37.0, 47.0, 49.0, 59.0, 61.0, 71.0, 73.0,
            83.0, 85.0,
        ] {
            let phi = 2.0 * PI * g / gear_teeth;
            let n = Vec3::new(
                lower_girdle_angle.sin() * phi.cos(),
                -lower_girdle_angle.cos(),
                lower_girdle_angle.sin() * phi.sin(),
            );
            planes.push(GpuFacetPlane::new(n, -0.68));
        }

        // 8. Culet (Bottom point at Y = -0.88)
        planes.push(GpuFacetPlane::new(Vec3::new(0.0, -1.0, 0.0), -0.88));

        planes
    }

    /// Generates exact 3D half-space planes for an Emerald Cut Gemstone.
    ///
    /// Every tier offset (`d`) below is *computed*, not pasted as a rounded literal, from
    /// one small shared profile: the girdle band half-height, the crown/pavilion crease
    /// ring heights, and the girdle radii. Two tiers that are meant to meet at an exact
    /// point (e.g. a girdle-adjacent tier and its neighbor on the crease ring, or all the
    /// facets converging on a girdle corner) are placed through that *same computed point*
    /// to full `f32` precision, rather than each being rounded to four decimals
    /// independently. That distinction matters here: `GemPolyhedron::from_planes`
    /// reconstructs vertices as 3-plane meets and welds ones closer than
    /// `VERTEX_WELD_EPS` (1e-4); a hand-rounded profile previously left several such
    /// intended-coincident points ~1.5e-4 apart -- just outside the weld radius -- which
    /// produced a dozen extra sliver vertices (60 instead of the true 48) even though every
    /// plane still contributed a facet and the volume was already correct. Deriving from
    /// the profile in code makes the coincidence exact by construction instead of by luck.
    #[must_use]
    pub fn emerald_cut() -> Vec<GpuFacetPlane> {
        // -- Shared profile -------------------------------------------------------------
        const TABLE_Y: f32 = 0.30; // crown table height
        const GIRDLE_HALF_BAND: f32 = 0.03; // girdle band spans y = -GIRDLE_HALF_BAND ..= +GIRDLE_HALF_BAND
        const CROWN_CREASE_Y: f32 = 0.15; // crown tiers meet each other on this ring
        const PAVILION_CREASE_MAG: f32 = 0.35; // pavilion tiers meet each other at y = -PAVILION_CREASE_MAG
        const GIRDLE_R_Z: f32 = 0.80;
        const GIRDLE_R_X: f32 = 1.10;
        const GIRDLE_R_DIAG: f32 = 1.00;
        // Pavilion keel truncation: just above the natural keel line where the two
        // keel-adjacent tiers (see `tier_d_at_crease` below, at PAVILION_CREASE_MAG) would
        // otherwise meet each other at x = z = 0 (that natural line sits at y ~= -0.8712),
        // so this plane slices through them just short of that point and leaves a real
        // keel flat instead of an untouched plane below the solid's deepest point.
        const PAVILION_KEEL_Y: f32 = -0.86;

        let diag = std::f32::consts::FRAC_1_SQRT_2;
        let mut planes = Vec::new();

        // Crown Table
        planes.push(GpuFacetPlane::new(Vec3::new(0.0, 1.0, 0.0), -TABLE_Y));

        // Crown tiers: the steeper (45 deg) tier is girdle-adjacent; the shallower
        // (35 deg) tier meets it exactly on the crown crease ring.
        let a1 = 35.0f32.to_radians(); // shallow, crease-ring-adjacent (Crown Step 1)
        let a2 = 45.0f32.to_radians(); // steep, girdle-adjacent (Crown Step 2)
        push_crown_tiers(
            &mut planes,
            a1,
            a2,
            GIRDLE_R_Z,
            GIRDLE_R_X,
            GIRDLE_HALF_BAND,
            CROWN_CREASE_Y,
        );

        // 4 Corner Crown Facets (40.0°), through the girdle's diagonal corner.
        let ac = 40.0f32.to_radians();
        push_corner_facets(&mut planes, ac, GIRDLE_R_DIAG, GIRDLE_HALF_BAND, diag, 1.0);

        // 8 Girdle Facets (90.0°)
        push_girdle_facets(&mut planes, GIRDLE_R_Z, GIRDLE_R_X, GIRDLE_R_DIAG, diag);

        // Pavilion tiers: the 53 deg tier is girdle-adjacent; the 43 deg tier meets it
        // exactly on the pavilion crease ring.
        //
        // NOTE: the tier angles are intentionally swapped from a naive reading of "step 1
        // then step 2" -- the shallower/more-horizontal 53 degree tier belongs immediately
        // below the girdle (its crease lands on the girdle edge), while the steeper/more-
        // vertical 43 degree tier belongs next to the keel. Assigning 43 degrees to the
        // girdle-adjacent tier (as this code previously did) leaves that tier's crease
        // strictly outside the girdle radius, which is why it never touched the hull.
        let p1 = 53.0f32.to_radians(); // girdle-adjacent (Pavilion Step 1)
        let p2 = 43.0f32.to_radians(); // keel-adjacent, crease-ring-adjacent (Pavilion Step 2)
        push_pavilion_tiers(
            &mut planes,
            p1,
            p2,
            GIRDLE_R_Z,
            GIRDLE_R_X,
            GIRDLE_HALF_BAND,
            PAVILION_CREASE_MAG,
        );

        // 4 Corner Pavilion Facets (-48.0°), through the girdle's diagonal corner.
        let pc = 48.0f32.to_radians();
        push_corner_facets(&mut planes, pc, GIRDLE_R_DIAG, GIRDLE_HALF_BAND, diag, -1.0);

        // Keel line base -- just above the natural keel line formed by the pavilion
        // tiers, so it truncates them into a real keel flat instead of sitting below the
        // solid's deepest point (which would leave it untouched).
        planes.push(GpuFacetPlane::new(
            Vec3::new(0.0, -1.0, 0.0),
            PAVILION_KEEL_Y,
        ));

        planes
    }

    /// Classifies whether a `FacetSpec` row belongs to the crown (true) or pavilion (false)
    /// side of the stone, using markers that are actually present in the scraped
    /// facetdiagrams.org data (verified against `facet_diagrams.sqlite`):
    fn classify_is_crown(item: &FacetSpec, angle_deg: f32, tier_idx: usize, total: usize) -> bool {
        let index_val = item.index.trim();
        if index_val.eq_ignore_ascii_case("table") {
            return true;
        }
        if index_val.eq_ignore_ascii_case("culet") {
            return false;
        }

        let facet = item.facet.trim();
        let mut chars = facet.chars();
        if let Some(first) = chars.next() {
            let rest = chars.as_str();
            let rest_is_all_digits = !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit());
            match first.to_ascii_lowercase() {
                'p' if rest_is_all_digits => return false,
                'c' if rest_is_all_digits || rest.is_empty() => return true,
                _ => {}
            }
        }
        if facet.eq_ignore_ascii_case("t") && angle_deg.abs() < 0.5 {
            return true;
        }

        if item.notes.to_lowercase().contains("crown") {
            return true;
        }

        // Last resort: no explicit marker available, guess from position in the list.
        tier_idx > total / 2
    }

    /// Parses the leading numeric angle prefix out of a scraped angle string.
    fn parse_angle_deg(s: &str) -> Option<f32> {
        let trimmed = s.trim();
        let bytes = trimmed.as_bytes();
        let mut end = 0;

        if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
            end += 1;
        }

        let mut saw_digit = false;
        let mut saw_dot = false;
        while end < bytes.len() {
            match bytes[end] {
                b'0'..=b'9' => {
                    saw_digit = true;
                    end += 1;
                }
                b'.' if !saw_dot => {
                    saw_dot = true;
                    end += 1;
                }
                _ => break,
            }
        }

        if !saw_digit {
            return None;
        }

        trimmed[..end].parse::<f32>().ok()
    }

    /// Recognizes the `"N girdle facets"` / `"N girdle facet"` form seen in
    /// `angle_settings.index_val` (e.g. "48 girdle facets", "96 girdle facets") and extracts `N`.
    fn parse_girdle_facet_count(index_val: &str) -> Option<u32> {
        let trimmed = index_val.trim();
        if !trimmed.to_lowercase().contains("girdle facet") {
            return None;
        }
        trimmed.split_whitespace().next()?.parse::<u32>().ok()
    }

    /// Reconstructs facet planes from parsed Angle items in database with realistic gemological proportions.
    pub fn from_database_angles(angles: &[FacetSpec], gear_teeth: u32) -> Vec<GpuFacetPlane> {
        const MAX_UNPARSEABLE_FRACTION: f32 = 0.1;

        if angles.is_empty() {
            return Self::standard_round_brilliant();
        }

        let gear_teeth_f = if gear_teeth > 0 {
            gear_teeth as f32
        } else {
            96.0
        };

        let unparseable = angles
            .iter()
            .filter(|item| Self::parse_angle_deg(&item.angle).is_none())
            .count();
        if unparseable as f32 > angles.len() as f32 * MAX_UNPARSEABLE_FRACTION {
            warn!(
                "from_database_angles: {}/{} angle values are unparseable; refusing to fabricate a solid, falling back to standard_round_brilliant()",
                unparseable,
                angles.len()
            );
            return Self::standard_round_brilliant();
        }

        let mut planes = Vec::new();

        for (tier_idx, item) in angles.iter().enumerate() {
            let Some(angle_deg) = Self::parse_angle_deg(&item.angle) else {
                warn!(
                    "from_database_angles: skipping facet '{}' with unparseable angle '{}'",
                    item.facet, item.angle
                );
                continue;
            };
            let theta = angle_deg.to_radians();
            let is_crown = Self::classify_is_crown(item, angle_deg, tier_idx, angles.len());

            // Exact proportional plane offset matching realistic cutting geometry
            let offset = if angle_deg >= 88.0 {
                -1.0
            } else if is_crown {
                if angle_deg < 5.0 {
                    -0.32 // Crown Table flat
                } else {
                    // Taper from girdle (r=1.0, y=0.03) to table
                    -0.04f32.mul_add(
                        -(1.0 - theta.sin()),
                        0.03f32.mul_add(theta.cos(), 1.0 * theta.sin()),
                    )
                }
            } else if angle_deg < 5.0 {
                -0.88 // Pavilion Culet
            } else {
                // Taper from girdle (r=1.0, y=-0.02) to culet
                -0.02f32.mul_add(theta.cos(), 1.0 * theta.sin())
            };

            // Parse index numbers from string e.g. "96, 12, 24, 36, 48, 60, 72, 84" or "96-12-24"
            let indices: Vec<f32> = Self::parse_girdle_facet_count(&item.index).map_or_else(
                || {
                    item.index
                        .split([',', '-', ' ', ';'])
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect()
                },
                |n| {
                    (0..n)
                        .map(|i| (i as f32) * gear_teeth_f / (n as f32))
                        .collect()
                },
            );

            if indices.is_empty() {
                // Single default orientation (e.g. Table or Culet)
                let n = if is_crown {
                    Vec3::new(0.0, theta.cos(), theta.sin())
                } else {
                    Vec3::new(0.0, -theta.cos(), theta.sin())
                };
                planes.push(GpuFacetPlane::new(n, offset));
            } else {
                for g in indices {
                    let phi = 2.0 * PI * g / gear_teeth_f;
                    let n = if is_crown {
                        Vec3::new(
                            theta.sin() * phi.cos(),
                            theta.cos(),
                            theta.sin() * phi.sin(),
                        )
                    } else {
                        Vec3::new(
                            theta.sin() * phi.cos(),
                            -theta.cos(),
                            theta.sin() * phi.sin(),
                        )
                    };
                    planes.push(GpuFacetPlane::new(n, offset));
                }
            }
        }

        if planes.len() < 4 {
            return Self::standard_round_brilliant();
        }

        planes
    }

    /// Reconstructs a validated boundary-representation solid from a cutting
    /// schedule's angle rows.
    ///
    /// This builds on [`Self::from_database_angles`] by using
    /// [`GemPolyhedron::from_planes`] as a plausibility gate on its output: a schedule
    /// whose planes don't actually bound a finite solid (or bound one only by leaving
    /// some of the schedule's own planes untouched -- an over-constrained, internally
    /// inconsistent schedule) renders silently wrong under the implicit half-space
    /// intersection alone, with no signal that anything is amiss. Here, either failure
    /// mode falls back to the same known-good `standard_round_brilliant()` cut that
    /// `from_database_angles` itself falls back to on badly unparseable input, and logs
    /// why.
    ///
    /// # Panics
    ///
    /// Panics only if `standard_round_brilliant()` itself ever failed to reconstruct
    /// into a valid polyhedron, which would indicate that reference cut regressed.
    #[must_use]
    pub fn reconstruct_validated_brep(angles: &[FacetSpec], gear_teeth: u32) -> GemPolyhedron {
        let planes = Self::from_database_angles(angles, gear_teeth);
        let plane_count = planes.len();

        match GemPolyhedron::from_planes(planes) {
            Ok(hull) => {
                let untouched = hull.untouched_planes();
                if untouched.is_empty() {
                    return hull;
                }
                warn!(
                    "reconstruct_validated_brep: {} of {plane_count} reconstructed planes contribute no facet \
                     (over-constrained/redundant schedule, plane indices {untouched:?}); falling back to \
                     standard_round_brilliant()",
                    untouched.len()
                );
            }
            Err(e) => {
                warn!(
                    "reconstruct_validated_brep: B-Rep reconstruction failed ({e}); falling back to \
                     standard_round_brilliant()"
                );
            }
        }

        GemPolyhedron::from_planes(Self::standard_round_brilliant()).expect(
            "standard_round_brilliant() must always reconstruct into a valid, finite polyhedron",
        )
    }

    /// Generates facet planes directly from a parsed `GemCAD` `.asc` cutting schedule.
    ///
    /// Uses [`lapidary::asc::parse_asc`]'s output and its real per-tier mast
    /// (depth) values as plane offsets, instead of the fabricated proportions
    /// [`Self::from_database_angles`] has to invent when only `angle_settings` (angle
    /// + index, no depth) is available.
    ///
    /// # Conventions (determined empirically against the real corpus; see the
    /// module-level report this function was built for)
    ///
    /// - **Angle sign** decides crown vs. pavilion: `.asc` angles are signed, and
    ///   negative means pavilion, matching `GemCAD`'s own convention (confirmed by 74
    ///   real files that carry both an explicit `-0.000000` pavilion culet and a
    ///   `0.000000` crown table in the same schedule -- the sign, not just the
    ///   magnitude, is meaningful). A tier whose angle is *unsigned* zero (the common
    ///   case -- about 98% of zero-angle tiers in the sampled corpus never bother
    ///   signing it, even for a culet) inherits the crown/pavilion side of the most
    ///   recent tier that did carry a nonzero (or explicitly signed) angle, since
    ///   `.asc` files consistently group a schedule's pavilion tiers before its crown
    ///   tiers; defaults to crown if it's the very first tier.
    /// - **Magnitude**: `theta` uses `angle_deg.abs()` -- the sign has already been
    ///   consumed above to pick crown vs. pavilion, so re-applying it to `sin`/`cos`
    ///   would rotate the facet to the wrong azimuthal quadrant.
    /// - **Plane offset**: `d = -mast.abs()`. `GemPolyhedron::from_planes` requires
    ///   every `d < 0` (the origin must lie inside every half-space); the file's mast
    ///   values are positive magnitudes in all but a handful of designs (a rare,
    ///   single-tier `"B"`-named exception with a negative mast at a near-zero angle,
    ///   seen in ~2.6% of sampled files), and even there the intent is a real
    ///   physical depth, not a sign-bearing offset, so the magnitude is what belongs
    ///   in the half-space equation.
    /// - **Azimuth**: `phi = 2*pi*index/gear_teeth_abs()`, matching
    ///   [`super::gemcad_parser::CuttingSchedule::into_planes`] and
    ///   [`Self::from_database_angles`]. A tier with no listed index (rare -- a bare
    ///   `angle mast` with only a name) produces one plane at `phi = 0`, the same
    ///   convention `from_database_angles` uses for an unlisted Table/Culet.
    #[must_use]
    pub fn from_asc_schedule(schedule: &AscSchedule) -> Vec<GpuFacetPlane> {
        let gear_teeth = (schedule.gear_teeth_abs().max(1)) as f32;
        let mut planes = Vec::with_capacity(schedule.facet_plane_count());

        // Most recently resolved crown/pavilion side, used to break the tie for a
        // tier whose angle is unsigned zero. See the doc comment above.
        let mut last_side_is_crown = true;

        for tier in &schedule.tiers {
            let is_crown = if tier.angle_deg == 0.0 {
                if tier.angle_deg.is_sign_negative() {
                    false
                } else {
                    last_side_is_crown
                }
            } else {
                tier.angle_deg > 0.0
            };
            last_side_is_crown = is_crown;

            let theta = (tier.angle_deg.abs() as f32).to_radians();
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let d = -(tier.mast.abs() as f32);

            if tier.indices.is_empty() {
                let n = if is_crown {
                    Vec3::new(0.0, cos_theta, sin_theta)
                } else {
                    Vec3::new(0.0, -cos_theta, sin_theta)
                };
                planes.push(GpuFacetPlane::new(n, d));
                continue;
            }

            for &idx in &tier.indices {
                let phi = 2.0 * PI * (idx as f32) / gear_teeth;
                let (sin_phi, cos_phi) = (phi.sin(), phi.cos());
                let n = if is_crown {
                    Vec3::new(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi)
                } else {
                    Vec3::new(sin_theta * cos_phi, -cos_theta, sin_theta * sin_phi)
                };
                planes.push(GpuFacetPlane::new(n, d));
            }
        }

        dedup_planes(planes)
    }

    /// Builds a validated B-Rep solid directly from a parsed `.asc` schedule.
    ///
    /// Uses [`Self::from_asc_schedule`] plus [`GemPolyhedron::from_planes`] and a
    /// check of [`GemPolyhedron::untouched_planes`] as the correctness oracle --
    /// exactly the same two-part gate [`Self::reconstruct_validated_brep`] uses for
    /// `angle_settings`-derived reconstructions.
    ///
    /// Unlike that path, there's no good "fabricated shape" fallback available here:
    /// the whole point of this function is that its offsets are the file's own real
    /// mast values, not invented proportions, so a validation failure means either the
    /// parse, the source file itself (e.g. a near-duplicate tier left in from a design
    /// revision), or the crown/pavilion sign convention above is off *for this
    /// particular file* -- silently substituting `standard_round_brilliant()` would
    /// hide that. Callers that need a guaranteed-renderable result on failure should
    /// fall back to [`Self::reconstruct_validated_brep`] (via `angle_settings`)
    /// themselves.
    ///
    /// # Errors
    ///
    /// Returns [`GemPolyhedron::from_planes`]'s error verbatim if the planes don't
    /// reconstruct into a valid, closed, finite solid at all, or a descriptive error
    /// if they do but leave one or more planes untouched (over-constrained --
    /// most often a near-duplicate tier revision in the source file, occasionally a
    /// zero-angle tier that resolved to the wrong side).
    pub fn reconstruct_validated_brep_from_asc(
        schedule: &AscSchedule,
    ) -> Result<GemPolyhedron, String> {
        let planes = Self::from_asc_schedule(schedule);
        let plane_count = planes.len();
        let hull = GemPolyhedron::from_planes(planes)?;
        let untouched = hull.untouched_planes();
        if untouched.is_empty() {
            Ok(hull)
        } else {
            Err(format!(
                "{} of {plane_count} planes from the .asc schedule contribute no facet (untouched indices: \
                 {untouched:?}); the schedule is over-constrained -- most often a near-duplicate tier revision \
                 left in the source file (e.g. the same facet listed twice at slightly different masts), \
                 occasionally a crown/pavilion sign misclassification on a zero-angle tier",
                untouched.len()
            ))
        }
    }
}

/// `d` for a tier plane with normal `(0, +-angle.cos(), +-angle.sin())` that passes
/// through the girdle edge at radius `r`, `half_band` above (crown) or below
/// (pavilion) the girdle band -- i.e. whichever tier is girdle-adjacent (crown: the
/// steeper 45 deg tier; pavilion: the 53 deg tier). Shared by [`push_crown_tiers`]
/// and [`push_pavilion_tiers`].
fn tier_d_at_girdle(angle: f32, r: f32, half_band: f32) -> f32 {
    -angle.sin().mul_add(r, angle.cos() * half_band)
}

/// `d` for the tier that is NOT girdle-adjacent: it instead meets the girdle-
/// adjacent tier exactly on the crease ring `y = +-crease_mag`. Solves for the point
/// where the `girdle_angle` tier's own plane crosses that ring, then places this
/// tier's plane through that same point, so the two tiers share an exact edge.
fn tier_d_at_crease(
    girdle_angle: f32,
    crease_angle: f32,
    r: f32,
    crease_mag: f32,
    half_band: f32,
) -> f32 {
    let z_c = r - girdle_angle.cos() * (crease_mag - half_band) / girdle_angle.sin();
    -crease_angle
        .sin()
        .mul_add(z_c, crease_angle.cos() * crease_mag)
}

/// `d` for a diagonal corner facet through the girdle's diagonal corner point
/// `(r_diag, +-half_band)`. Works for both crown (+y normal) and pavilion (-y
/// normal) corners: the sign flip in the normal's y-component and the sign flip in
/// the girdle band's y-coordinate cancel out.
fn corner_d(angle: f32, r_diag: f32, half_band: f32) -> f32 {
    -angle.cos().mul_add(half_band, angle.sin() * r_diag)
}

/// Pushes the 8 crown tier facets: Crown Step 1 (`a1`, shallow, meets the crease
/// ring) and Crown Step 2 (`a2`, steep, girdle-adjacent). See [`tier_d_at_crease`]
/// and [`tier_d_at_girdle`] for how the two tiers are made to share an exact edge.
fn push_crown_tiers(
    planes: &mut Vec<GpuFacetPlane>,
    a1: f32,
    a2: f32,
    r_z: f32,
    r_x: f32,
    half_band: f32,
    crease_y: f32,
) {
    let d_a1_z = tier_d_at_crease(a2, a1, r_z, crease_y, half_band);
    let d_a1_x = tier_d_at_crease(a2, a1, r_x, crease_y, half_band);
    let d_a2_z = tier_d_at_girdle(a2, r_z, half_band);
    let d_a2_x = tier_d_at_girdle(a2, r_x, half_band);

    // Crown Step 1
    planes.push(GpuFacetPlane::new(
        Vec3::new(0.0, a1.cos(), a1.sin()),
        d_a1_z,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(0.0, a1.cos(), -a1.sin()),
        d_a1_z,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(a1.sin(), a1.cos(), 0.0),
        d_a1_x,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(-a1.sin(), a1.cos(), 0.0),
        d_a1_x,
    ));

    // Crown Step 2
    planes.push(GpuFacetPlane::new(
        Vec3::new(0.0, a2.cos(), a2.sin()),
        d_a2_z,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(0.0, a2.cos(), -a2.sin()),
        d_a2_z,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(a2.sin(), a2.cos(), 0.0),
        d_a2_x,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(-a2.sin(), a2.cos(), 0.0),
        d_a2_x,
    ));
}

/// Pushes the 8 pavilion tier facets: Pavilion Step 1 (`p1`, girdle-adjacent) and
/// Pavilion Step 2 (`p2`, keel-adjacent, meets the crease ring). Mirror image of
/// [`push_crown_tiers`] on the -y side.
fn push_pavilion_tiers(
    planes: &mut Vec<GpuFacetPlane>,
    p1: f32,
    p2: f32,
    r_z: f32,
    r_x: f32,
    half_band: f32,
    crease_mag: f32,
) {
    let d_p1_z = tier_d_at_girdle(p1, r_z, half_band);
    let d_p1_x = tier_d_at_girdle(p1, r_x, half_band);
    let d_p2_z = tier_d_at_crease(p1, p2, r_z, crease_mag, half_band);
    let d_p2_x = tier_d_at_crease(p1, p2, r_x, crease_mag, half_band);

    // Pavilion Step 1 (girdle-adjacent)
    planes.push(GpuFacetPlane::new(
        Vec3::new(0.0, -p1.cos(), p1.sin()),
        d_p1_z,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(0.0, -p1.cos(), -p1.sin()),
        d_p1_z,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(p1.sin(), -p1.cos(), 0.0),
        d_p1_x,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(-p1.sin(), -p1.cos(), 0.0),
        d_p1_x,
    ));

    // Pavilion Step 2 (keel-adjacent)
    planes.push(GpuFacetPlane::new(
        Vec3::new(0.0, -p2.cos(), p2.sin()),
        d_p2_z,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(0.0, -p2.cos(), -p2.sin()),
        d_p2_z,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(p2.sin(), -p2.cos(), 0.0),
        d_p2_x,
    ));
    planes.push(GpuFacetPlane::new(
        Vec3::new(-p2.sin(), -p2.cos(), 0.0),
        d_p2_x,
    ));
}

/// Pushes the 4 diagonal corner facets at `angle`, through the girdle's diagonal
/// corner point. `y_sign` is `1.0` for crown corners (+y normal) or `-1.0` for
/// pavilion corners (-y normal); see [`corner_d`] for why the same formula works
/// for both.
fn push_corner_facets(
    planes: &mut Vec<GpuFacetPlane>,
    angle: f32,
    r_diag: f32,
    half_band: f32,
    diag: f32,
    y_sign: f32,
) {
    let d = corner_d(angle, r_diag, half_band);
    let y = y_sign * angle.cos();
    let s = angle.sin() * diag;
    planes.push(GpuFacetPlane::new(Vec3::new(s, y, s), d));
    planes.push(GpuFacetPlane::new(Vec3::new(-s, y, s), d));
    planes.push(GpuFacetPlane::new(Vec3::new(s, y, -s), d));
    planes.push(GpuFacetPlane::new(Vec3::new(-s, y, -s), d));
}

/// Pushes the 8 vertical girdle facets (4 axis-aligned + 4 diagonal).
fn push_girdle_facets(planes: &mut Vec<GpuFacetPlane>, r_z: f32, r_x: f32, r_diag: f32, diag: f32) {
    planes.push(GpuFacetPlane::new(Vec3::new(0.0, 0.0, 1.0), -r_z));
    planes.push(GpuFacetPlane::new(Vec3::new(0.0, 0.0, -1.0), -r_z));
    planes.push(GpuFacetPlane::new(Vec3::new(1.0, 0.0, 0.0), -r_x));
    planes.push(GpuFacetPlane::new(Vec3::new(-1.0, 0.0, 0.0), -r_x));
    planes.push(GpuFacetPlane::new(Vec3::new(diag, 0.0, diag), -r_diag));
    planes.push(GpuFacetPlane::new(Vec3::new(-diag, 0.0, diag), -r_diag));
    planes.push(GpuFacetPlane::new(Vec3::new(diag, 0.0, -diag), -r_diag));
    planes.push(GpuFacetPlane::new(Vec3::new(-diag, 0.0, -diag), -r_diag));
}

/// Drops planes that are near-duplicates of an earlier one in the list (same normal
/// and offset within a coarse tolerance), keeping the first occurrence.
///
/// Real `.asc` files occasionally list the same index position twice across two
/// separate tier rows that happen to share an identical angle and mast (a data-entry
/// artifact in the original hand-authored schedules, e.g. index `0` appearing in both
/// a `-90 ... 0 32` row and a later `-90 ... 0` row at the same mast). Two planes with
/// the same normal and offset are the same half-space -- geometrically redundant, not
/// a real second facet -- and `GemPolyhedron::from_planes` correctly rejects them
/// outright (two coincident half-spaces have coincident dual points, which the dual
/// convex hull cannot use). Removing the redundant copy here does not change the
/// resulting solid at all, only avoids handing `from_planes` a construction it cannot
/// use.
fn dedup_planes(planes: Vec<GpuFacetPlane>) -> Vec<GpuFacetPlane> {
    const QUANT: f32 = 1.0 / 2048.0; // ~5e-4, coarser than brep.rs's own coincidence epsilon
    let quantize = |v: f32| (v / QUANT).round() as i64;

    let mut seen: HashSet<(i64, i64, i64, i64)> = HashSet::with_capacity(planes.len());
    planes
        .into_iter()
        .filter(|p| {
            let key = (
                quantize(p.normal[0]),
                quantize(p.normal[1]),
                quantize(p.normal[2]),
                quantize(p.d),
            );
            seen.insert(key)
        })
        .collect()
}

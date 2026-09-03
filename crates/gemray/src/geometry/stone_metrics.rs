//! Deterministic physical measurement of a faceted stone from its plane
//! arrangement.
//!
//! [`super::meet_solver`]'s corpus work established that wrong mast
//! configurations are *self-consistent*: every tier of a wrong solve is still
//! vertex-incident, so nothing internal to the arrangement separates right from
//! wrong. The proportions printed on a real diagram (`Vol/W^3`, `L/W`, `C/W`,
//! `P/W`, `H/W`) are **external** constraints: a candidate configuration either
//! reproduces them or it does not. [`measure_solid`] computes those same figures
//! from a plane arrangement -- deterministically, in `f64`, with no convex-hull
//! library ([`super::brep`]'s `chull` hull is nondeterministic and must stay out
//! of every solver decision path).
//!
//! The mechanism: enumerate the solid's vertices as feasible triple
//! intersections of the planes (the same primitive `meet_solver` uses), then
//! reconstruct each facet's polygon by collecting the vertices on its plane and
//! ordering them by angle. Volume comes from the divergence theorem
//! (`V = (1/3) * sum over faces of offset * area`, exact for outward-oriented
//! planes), heights from vertex `y` extents against the girdle band, and
//! width/length from the girdle outline's `x`/`z` extents (both axis-aligned
//! and rotating-caliper are computed; corpus measurement settled that the
//! printed figures use the axis convention -- see [`ExternalProportions`]).

use glam::DVec3;

/// Half-extent of the bounding box standing in for the uncut rough. Matches
/// `meet_solver`'s own blank; a solid that reaches it is unbounded (a schedule
/// missing its closing planes), which [`measure_solid`] reports as `None`.
const BLANK_HALF_EXTENT: f64 = 64.0;

/// Feasibility slack: a vertex may poke this far (absolute; masts are ~1) beyond
/// a plane and still count as part of the solid. Matches `meet_solver::EPS_FEAS`.
const EPS_FEAS: f64 = 1e-5;

/// A vertex within this absolute distance of a plane counts as lying on its
/// face. Sized to cover [`EPS_FEAS`] feasibility drift and [`VERTEX_DEDUP`]
/// merging; the resulting polygon-corner error is quadratically small in it.
const EPS_FACE: f64 = 2e-5;

/// Minimum `|determinant|` for a triple of unit plane normals to define a
/// candidate vertex. Matches `meet_solver::MIN_TRIPLE_DET`.
const MIN_TRIPLE_DET: f64 = 1e-6;

/// Two candidate vertices within this distance (per axis) are one vertex.
const VERTEX_DEDUP: f64 = 1e-6;

/// `|normal.y|` at or below this means a vertical (girdle) plane -- the same
/// threshold `meet_solver::classify_blocks` uses.
const GIRDLE_NY: f64 = 1e-6;

/// Everything [`measure_solid`] reports about one solid.
///
/// All figures are in the arrangement's own mast units; callers compare
/// dimensionless ratios (`volume / width^3`, `length / width`, ...) so the
/// unit never matters.
#[derive(Debug, Clone, Copy)]
pub struct SolidMetrics {
    /// Volume of the solid.
    pub volume: f64,
    /// Width: the smaller of the two axis-aligned horizontal (`x`/`z`) extents.
    pub width_axis: f64,
    /// Length: the larger of the two axis-aligned horizontal extents.
    pub length_axis: f64,
    /// Width by rotating calipers over the horizontal outline: the smallest
    /// directional extent over all outline-edge directions.
    pub width_caliper: f64,
    /// Length measured along the direction perpendicular to the caliper width.
    pub length_caliper: f64,
    /// Total height: full vertical (`y`) extent, table to culet, girdle included.
    pub total_height: f64,
    /// Crown height: top of the solid above the girdle band's top edge. `None`
    /// when the arrangement has no vertical girdle plane with a live facet.
    pub crown_height: Option<f64>,
    /// Pavilion depth: bottom of the solid below the girdle band's bottom edge.
    /// `None` when there is no live girdle facet.
    pub pavilion_depth: Option<f64>,
    /// Vertical extent of the girdle band itself. `None` without a live girdle.
    pub girdle_thickness: Option<f64>,
    /// Distinct vertices of the solid (after dedup).
    pub vertex_count: usize,
}

/// A design's printed proportion figures, as scraped into `diagram_details`.
///
/// Columns `volume`, `lw_ratio`, `cw_ratio`, `pw_ratio`, `hw_ratio`: the
/// external targets a candidate mast configuration must reproduce. Any subset
/// may be present.
///
/// Corpus calibration (full 2,881-design `.asc` corpus, true masts, measured
/// by a temporary corpus probe, deterministic run): with the **axis** width
/// convention
/// (`W` = the smaller of the two axis-aligned horizontal extents, which is the
/// convention that matched -- rotating calipers measured strictly worse on
/// every figure), each printed figure reproduces the true solid's measurement
/// with a median deviation of ~0.1% (`Vol/W^3` 95.8% of designs within 1%,
/// `L/W` 98.7%, `C/W` 88.6%, `P/W` 91.2%, `H/W` 96.8%).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExternalProportions {
    /// Printed `Vol/W^3`.
    pub vol_w3: Option<f64>,
    /// Printed `L/W`.
    pub lw: Option<f64>,
    /// Printed `C/W` (crown height over width).
    pub cw: Option<f64>,
    /// Printed `P/W` (pavilion depth over width).
    pub pw: Option<f64>,
    /// Printed `H/W` (total height over width).
    pub hw: Option<f64>,
}

impl ExternalProportions {
    /// Mean relative deviation of `metrics` from the printed figures, over
    /// every figure both sides have (axis width convention -- see the type
    /// docs). `None` when nothing overlaps.
    #[must_use]
    pub fn combined_deviation(&self, metrics: &SolidMetrics) -> Option<f64> {
        let w = metrics.width_axis;
        if w < 1e-9 {
            return None;
        }
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        let mut add = |target: Option<f64>, measured: Option<f64>| {
            if let (Some(t), Some(v)) = (target, measured)
                && t > 1e-9
            {
                sum += (v - t).abs() / t;
                count += 1;
            }
        };
        add(self.vol_w3, Some(metrics.volume / (w * w * w)));
        add(self.lw, Some(metrics.length_axis / w));
        add(self.cw, metrics.crown_height.map(|c| c / w));
        add(self.pw, metrics.pavilion_depth.map(|p| p / w));
        add(self.hw, Some(metrics.total_height / w));
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }
}

/// One deduplicated vertex of the solid.
struct SolidVertex {
    v: DVec3,
}

/// Accepted vertices, in first-seen order, plus an index over them sorted by
/// `x` so a new candidate's duplicate check only has to scan the vertices
/// that could possibly be within [`VERTEX_DEDUP`] of it on every axis instead
/// of the full accepted set.
///
/// `verts`' order is exactly the push order of [`insert_if_new`](Self::insert_if_new)
/// calls (the divergence-theorem sum in [`measure_solid`] depends on it);
/// `by_x` is a lookup structure only, never observed by callers.
#[derive(Default)]
struct VertexAccumulator {
    verts: Vec<SolidVertex>,
    /// Indices into `verts`, kept sorted ascending by `verts[i].v.x`.
    by_x: Vec<usize>,
}

impl VertexAccumulator {
    /// Inserts `v` unless some already-accepted vertex is within
    /// [`VERTEX_DEDUP`] of it on every axis -- identical to a linear scan
    /// testing `(s.v - v).abs().max_element() < VERTEX_DEDUP` against every
    /// prior vertex, just restricted up front to the `x`-sorted window that
    /// could possibly match (any vertex outside `[v.x - VERTEX_DEDUP, v.x +
    /// VERTEX_DEDUP]` fails the `x`-axis check alone, so narrowing to that
    /// window changes no accept/reject decision).
    fn insert_if_new(&mut self, v: DVec3) {
        let verts = &self.verts;
        let lo = self
            .by_x
            .partition_point(|&i| verts[i].v.x < v.x - VERTEX_DEDUP);
        let hi = self
            .by_x
            .partition_point(|&i| verts[i].v.x <= v.x + VERTEX_DEDUP);
        for &i in &self.by_x[lo..hi] {
            if (self.verts[i].v - v).abs().max_element() < VERTEX_DEDUP {
                return;
            }
        }
        let idx = self.verts.len();
        self.verts.push(SolidVertex { v });
        let pos = self.by_x.partition_point(|&i| self.verts[i].v.x < v.x);
        self.by_x.insert(pos, idx);
    }
}

/// Measures the solid bounded by `planes` (`n . x <= m`, unit outward normals).
///
/// Returns `None` when the solid is degenerate or unbounded: fewer than four
/// distinct vertices, zero/negative volume, or any vertex escaping to the
/// bounding blank (a schedule missing its closing planes).
///
/// Deterministic by construction: plain nested loops over the given plane
/// order, total-order sorts, no hashing, no convex-hull library. Two calls with
/// identical inputs produce byte-identical results.
#[must_use]
pub fn measure_solid(planes: &[(DVec3, f64)]) -> Option<SolidMetrics> {
    let planes = dedup_planes(planes);
    let verts = feasible_vertices(&planes)?;
    if verts.len() < 4 {
        return None;
    }

    let volume: f64 = planes
        .iter()
        .map(|&(n, m)| m * face_area(n, m, &verts) / 3.0)
        .sum();
    if !(volume.is_finite() && volume > 0.0) {
        return None;
    }

    let y_max = verts
        .iter()
        .map(|s| s.v.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = verts.iter().map(|s| s.v.y).fold(f64::INFINITY, f64::min);

    // Girdle band: vertical extent of the vertices lying on any vertical plane.
    let mut girdle_top = f64::NEG_INFINITY;
    let mut girdle_bottom = f64::INFINITY;
    for &(n, m) in planes.iter().filter(|(n, _)| n.y.abs() <= GIRDLE_NY) {
        for s in &verts {
            if (n.dot(s.v) - m).abs() <= EPS_FACE {
                girdle_top = girdle_top.max(s.v.y);
                girdle_bottom = girdle_bottom.min(s.v.y);
            }
        }
    }
    let has_girdle = girdle_top >= girdle_bottom;

    let x_max = verts
        .iter()
        .map(|s| s.v.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let x_min = verts.iter().map(|s| s.v.x).fold(f64::INFINITY, f64::min);
    let z_max = verts
        .iter()
        .map(|s| s.v.z)
        .fold(f64::NEG_INFINITY, f64::max);
    let z_min = verts.iter().map(|s| s.v.z).fold(f64::INFINITY, f64::min);
    let (dx, dz) = (x_max - x_min, z_max - z_min);
    let (width_axis, length_axis) = if dx <= dz { (dx, dz) } else { (dz, dx) };

    let outline: Vec<(f64, f64)> = verts.iter().map(|s| (s.v.x, s.v.z)).collect();
    let (width_caliper, length_caliper) =
        caliper_extents(&outline).unwrap_or((width_axis, length_axis));

    Some(SolidMetrics {
        volume,
        width_axis,
        length_axis,
        width_caliper,
        length_caliper,
        total_height: y_max - y_min,
        crown_height: has_girdle.then_some(y_max - girdle_top),
        pavilion_depth: has_girdle.then_some(girdle_bottom - y_min),
        girdle_thickness: has_girdle.then_some(girdle_top - girdle_bottom),
        vertex_count: verts.len(),
    })
}

/// Drops duplicate planes (same normal and offset within tight tolerance) so a
/// tier that lists the same index twice can't double-count its face's area.
fn dedup_planes(planes: &[(DVec3, f64)]) -> Vec<(DVec3, f64)> {
    let mut out: Vec<(DVec3, f64)> = Vec::with_capacity(planes.len());
    for &(n, m) in planes {
        let dup = out
            .iter()
            .any(|&(n2, m2)| n.dot(n2) > 1.0 - 1e-12 && (m - m2).abs() < 1e-9);
        if !dup {
            out.push((n, m));
        }
    }
    out
}

/// Drains one full (or final partial) [`crate::simd::TripleBatch`] solve into
/// `verts`, in ascending lane order. Returns `None` the moment a vertex
/// escapes to the blank box, propagated by the caller via `?` -- matching the
/// original loop's immediate `return None`. Shared by
/// [`feasible_vertices`]'s batching loop.
fn flush_solid_batch(
    batch: &crate::simd::TripleBatch,
    soa: &crate::simd::PlanesSoA64,
    acc: &mut VertexAccumulator,
) -> Option<()> {
    let sol = crate::simd::solve_triple_batch(batch);
    for lane in 0..batch.len {
        if sol.det[lane].abs() < MIN_TRIPLE_DET {
            continue;
        }
        let v = DVec3::new(sol.vx[lane], sol.vy[lane], sol.vz[lane]);
        if v.abs().max_element() > BLANK_HALF_EXTENT + 1.0 {
            continue;
        }
        if crate::simd::any_violation(soa, v, EPS_FEAS) {
            continue;
        }
        // A feasible vertex at the blank box means the real planes never
        // closed the solid up -- there is no finite stone to measure.
        if v.abs().max_element() > BLANK_HALF_EXTENT - 1.0 {
            return None;
        }
        acc.insert_if_new(v);
    }
    Some(())
}

/// Enumerates the solid's distinct vertices: every well-conditioned plane triple
/// whose intersection satisfies all half-spaces (within [`EPS_FEAS`]), then
/// deduplicated by position. Returns `None` when any vertex reaches the bounding
/// blank (the real planes don't bound a finite solid).
///
/// Batched through `crate::simd`, matching
/// `meet_solver::enumerate_candidate_vertices`: one `PlanesSoA64` built up
/// front (owner is irrelevant to this owner-free scan, so every plane is
/// pushed with owner 0), triples solved via `solve_triple_batch` via
/// [`flush_solid_batch`], and the `any()` feasibility scan replaced by
/// `any_violation` -- bit-identical per lane to the `glam` `DMat3` sequence
/// and scalar scan they replace (see `src/simd.rs`'s determinism contract).
/// Lanes are drained in ascending order and triples are still generated by
/// the same nested loops in the same order, so vertex order and every
/// decision here (determinant check, bounds check, feasibility,
/// blank-escape) match the unbatched scalar version exactly.
fn feasible_vertices(planes: &[(DVec3, f64)]) -> Option<Vec<SolidVertex>> {
    let mut all: Vec<(DVec3, f64)> = planes.to_vec();
    for n in [
        DVec3::X,
        DVec3::NEG_X,
        DVec3::Y,
        DVec3::NEG_Y,
        DVec3::Z,
        DVec3::NEG_Z,
    ] {
        all.push((n, BLANK_HALF_EXTENT));
    }

    let mut soa = crate::simd::PlanesSoA64::with_capacity(all.len());
    for &(n, m) in &all {
        soa.push(n, m, 0);
    }

    let p = all.len();
    let mut acc = VertexAccumulator::default();
    let mut batch = crate::simd::TripleBatch::default();
    for a in 0..p {
        for b in (a + 1)..p {
            for c in (b + 1)..p {
                let (pa, pb, pc) = (all[a], all[b], all[c]);
                if batch.push((pa.0, pa.1), (pb.0, pb.1), (pc.0, pc.1)) {
                    flush_solid_batch(&batch, &soa, &mut acc)?;
                    batch = crate::simd::TripleBatch::default();
                }
            }
        }
    }
    if batch.len > 0 {
        flush_solid_batch(&batch, &soa, &mut acc)?;
    }
    Some(acc.verts)
}

/// Area of the face polygon that plane `(normal, offset)` contributes to the
/// solid: the vertices on the plane, ordered by angle about the normal,
/// shoelace-summed. Zero when fewer than three vertices lie on the plane (the
/// facet was cut away entirely).
fn face_area(normal: DVec3, offset: f64, verts: &[SolidVertex]) -> f64 {
    let on_face: Vec<DVec3> = verts
        .iter()
        .filter(|s| (normal.dot(s.v) - offset).abs() <= EPS_FACE)
        .map(|s| s.v)
        .collect();
    if on_face.len() < 3 {
        return 0.0;
    }

    // Deterministic in-plane basis: start from the world axis least aligned
    // with the normal.
    let seed = if normal.x.abs() <= normal.y.abs() && normal.x.abs() <= normal.z.abs() {
        DVec3::X
    } else if normal.y.abs() <= normal.z.abs() {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let basis_u = (seed - normal * normal.dot(seed)).normalize();
    let basis_v = normal.cross(basis_u);

    let centroid = on_face.iter().copied().sum::<DVec3>() / on_face.len() as f64;
    let mut angled: Vec<(f64, DVec3)> = on_face
        .into_iter()
        .map(|vert| {
            let d = vert - centroid;
            (basis_v.dot(d).atan2(basis_u.dot(d)), vert)
        })
        .collect();
    angled.sort_by(|x, y| x.0.total_cmp(&y.0));

    let mut cross_sum = DVec3::ZERO;
    for i in 0..angled.len() {
        let a = angled[i].1 - centroid;
        let b = angled[(i + 1) % angled.len()].1 - centroid;
        cross_sum += a.cross(b);
    }
    0.5 * normal.dot(cross_sum).abs()
}

/// Rotating-caliper width and length of a 2D point set: the smallest directional
/// extent over all convex-outline edge directions, and the extent along the
/// perpendicular direction. Returns `None` when the outline is degenerate
/// (fewer than three distinct hull points).
fn caliper_extents(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let hull = convex_hull_2d(points);
    if hull.len() < 3 {
        return None;
    }
    let mut best: Option<(f64, f64)> = None;
    for i in 0..hull.len() {
        let (px, pz) = hull[i];
        let (qx, qz) = hull[(i + 1) % hull.len()];
        let (ex, ez) = (qx - px, qz - pz);
        let len = ex.hypot(ez);
        if len < 1e-12 {
            continue;
        }
        let (dx, dz) = (ex / len, ez / len);
        let mut along = (f64::INFINITY, f64::NEG_INFINITY);
        let mut across = (f64::INFINITY, f64::NEG_INFINITY);
        for &(x, z) in &hull {
            let a = x.mul_add(dx, z * dz);
            let c = x.mul_add(-dz, z * dx);
            along = (along.0.min(a), along.1.max(a));
            across = (across.0.min(c), across.1.max(c));
        }
        let width = across.1 - across.0;
        let length = along.1 - along.0;
        if best.is_none_or(|(bw, _)| width < bw) {
            best = Some((width, length));
        }
    }
    best.map(|(w, l)| if w <= l { (w, l) } else { (l, w) })
}

/// Andrew's monotone-chain convex hull over 2D points, counterclockwise.
/// Deterministic: total-order lexicographic sort, no hashing.
fn convex_hull_2d(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - o.0).mul_add(b.1 - o.1, -((a.1 - o.1) * (b.0 - o.0)))
    };
    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box `[-1,1] x [-0.6,0.6] x [-1,1]`: volume 4.8, width 2,
    /// length 2, height 1.2. The four vertical walls are girdle planes cut by
    /// nothing, so the girdle band spans the full height and crown/pavilion are
    /// zero.
    #[test]
    fn measures_a_plain_box() {
        let planes = vec![
            (DVec3::X, 1.0),
            (DVec3::NEG_X, 1.0),
            (DVec3::Y, 0.6),
            (DVec3::NEG_Y, 0.6),
            (DVec3::Z, 1.0),
            (DVec3::NEG_Z, 1.0),
        ];
        let m = measure_solid(&planes).expect("box must measure");
        assert!((m.volume - 4.8).abs() < 1e-9, "volume {}", m.volume);
        assert!((m.width_axis - 2.0).abs() < 1e-9);
        assert!((m.length_axis - 2.0).abs() < 1e-9);
        assert!((m.width_caliper - 2.0).abs() < 1e-9);
        assert!((m.total_height - 1.2).abs() < 1e-9);
        assert_eq!(m.vertex_count, 8);
        assert!((m.crown_height.expect("girdle present")).abs() < 1e-9);
        assert!((m.pavilion_depth.expect("girdle present")).abs() < 1e-9);
        assert!((m.girdle_thickness.expect("girdle present") - 1.2).abs() < 1e-9);
    }

    /// A hip-roofed block: square girdle walls at `|x|,|z| <= 1`, flat floor at
    /// `y = -0.5`, and four 45-degree crown planes `y <= 1 - |x|`, `y <= 1 - |z|`.
    /// Hand-computed: volume `2 + 4/3`, ridge apex at `y = 1`, girdle band
    /// clipped at `y = 0`, so crown height 1, pavilion depth 0, girdle 0.5.
    #[test]
    fn measures_crown_height_against_the_girdle_band() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let planes = vec![
            (DVec3::X, 1.0),
            (DVec3::NEG_X, 1.0),
            (DVec3::Z, 1.0),
            (DVec3::NEG_Z, 1.0),
            (DVec3::NEG_Y, 0.5),
            // 45-degree crown planes: n = (+-s, s, 0) and (0, s, +-s), m = s,
            // i.e. x + y = 1 etc.
            (DVec3::new(s, s, 0.0), s),
            (DVec3::new(-s, s, 0.0), s),
            (DVec3::new(0.0, s, s), s),
            (DVec3::new(0.0, s, -s), s),
        ];
        let m = measure_solid(&planes).expect("roofed block must measure");
        assert!(
            (m.volume - (2.0 + 4.0 / 3.0)).abs() < 1e-9,
            "volume {}",
            m.volume
        );
        assert!((m.total_height - 1.5).abs() < 1e-9);
        assert!((m.crown_height.expect("girdle present") - 1.0).abs() < 1e-9);
        assert!((m.pavilion_depth.expect("girdle present")).abs() < 1e-9);
        assert!((m.girdle_thickness.expect("girdle present") - 0.5).abs() < 1e-9);
        assert!((m.width_axis - 2.0).abs() < 1e-9);
    }

    /// A solid the real planes never close (no floor): must report `None`, not a
    /// blank-box-clipped volume.
    #[test]
    fn unbounded_solid_reports_none() {
        let planes = vec![
            (DVec3::X, 1.0),
            (DVec3::NEG_X, 1.0),
            (DVec3::Y, 0.6),
            (DVec3::Z, 1.0),
            (DVec3::NEG_Z, 1.0),
        ];
        assert!(measure_solid(&planes).is_none());
    }

    /// A duplicated plane (same normal and offset listed twice) must not
    /// double-count its face's area.
    #[test]
    fn duplicate_planes_do_not_double_count() {
        let planes = vec![
            (DVec3::X, 1.0),
            (DVec3::X, 1.0),
            (DVec3::NEG_X, 1.0),
            (DVec3::Y, 0.6),
            (DVec3::NEG_Y, 0.6),
            (DVec3::Z, 1.0),
            (DVec3::NEG_Z, 1.0),
        ];
        let m = measure_solid(&planes).expect("box must measure");
        assert!((m.volume - 4.8).abs() < 1e-9, "volume {}", m.volume);
    }

    /// A 45-degree-rotated square girdle: axis extents see the diagonal
    /// (`2*sqrt(2)`), calipers must recover the true side length 2.
    #[test]
    fn caliper_width_beats_axis_width_on_a_rotated_outline() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let planes = vec![
            (DVec3::new(s, 0.0, s), 1.0),
            (DVec3::new(-s, 0.0, s), 1.0),
            (DVec3::new(s, 0.0, -s), 1.0),
            (DVec3::new(-s, 0.0, -s), 1.0),
            (DVec3::Y, 0.5),
            (DVec3::NEG_Y, 0.5),
        ];
        let m = measure_solid(&planes).expect("rotated box must measure");
        let diag = 2.0 * std::f64::consts::SQRT_2;
        assert!((m.width_axis - diag).abs() < 1e-9, "axis {}", m.width_axis);
        assert!(
            (m.width_caliper - 2.0).abs() < 1e-9,
            "caliper {}",
            m.width_caliper
        );
        // Side-2 square cross-section, height 1.
        assert!((m.volume - 4.0).abs() < 1e-9, "volume {}", m.volume);
    }

    /// Byte-identical determinism across repeated calls.
    #[test]
    fn measurement_is_deterministic() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let planes = vec![
            (DVec3::X, 1.0),
            (DVec3::NEG_X, 1.0),
            (DVec3::Z, 1.0),
            (DVec3::NEG_Z, 1.0),
            (DVec3::NEG_Y, 0.5),
            (DVec3::new(s, s, 0.0), s),
            (DVec3::new(-s, s, 0.0), s),
            (DVec3::new(0.0, s, s), s),
            (DVec3::new(0.0, s, -s), s),
        ];
        let a = measure_solid(&planes).expect("must measure");
        let b = measure_solid(&planes).expect("must measure");
        assert_eq!(a.volume.to_bits(), b.volume.to_bits());
        assert_eq!(a.width_caliper.to_bits(), b.width_caliper.to_bits());
        assert_eq!(a.total_height.to_bits(), b.total_height.to_bits());
    }

    // -----------------------------------------------------------------------
    // S3: the `VertexAccumulator` x-sorted index in `insert_if_new` must make
    // the exact same accept/reject decision, in the exact same insertion
    // order, as the O(V^2) linear scan it replaces. Proven here by running
    // both side by side over the module's own fixtures plus real cutting
    // schedules and comparing the resulting vertex lists bit-for-bit.
    // -----------------------------------------------------------------------

    /// Pre-optimization dedup, kept only as a reference: a duplicate is any
    /// already-accepted vertex within [`VERTEX_DEDUP`] on every axis, found
    /// by scanning the full accepted set (this is exactly the body
    /// `flush_solid_batch` had before `VertexAccumulator` existed).
    fn flush_solid_batch_linear_reference(
        batch: &crate::simd::TripleBatch,
        soa: &crate::simd::PlanesSoA64,
        verts: &mut Vec<SolidVertex>,
    ) -> Option<()> {
        let sol = crate::simd::solve_triple_batch(batch);
        for lane in 0..batch.len {
            if sol.det[lane].abs() < MIN_TRIPLE_DET {
                continue;
            }
            let v = DVec3::new(sol.vx[lane], sol.vy[lane], sol.vz[lane]);
            if v.abs().max_element() > BLANK_HALF_EXTENT + 1.0 {
                continue;
            }
            if crate::simd::any_violation(soa, v, EPS_FEAS) {
                continue;
            }
            if v.abs().max_element() > BLANK_HALF_EXTENT - 1.0 {
                return None;
            }
            let dup = verts
                .iter()
                .any(|s| (s.v - v).abs().max_element() < VERTEX_DEDUP);
            if !dup {
                verts.push(SolidVertex { v });
            }
        }
        Some(())
    }

    /// [`feasible_vertices`], but deduped by [`flush_solid_batch_linear_reference`]
    /// instead of [`VertexAccumulator`]. Otherwise byte-for-byte the same
    /// function (same plane augmentation, same batching loop).
    fn feasible_vertices_linear_reference(planes: &[(DVec3, f64)]) -> Option<Vec<DVec3>> {
        let mut all: Vec<(DVec3, f64)> = planes.to_vec();
        for n in [
            DVec3::X,
            DVec3::NEG_X,
            DVec3::Y,
            DVec3::NEG_Y,
            DVec3::Z,
            DVec3::NEG_Z,
        ] {
            all.push((n, BLANK_HALF_EXTENT));
        }

        let mut soa = crate::simd::PlanesSoA64::with_capacity(all.len());
        for &(n, m) in &all {
            soa.push(n, m, 0);
        }

        let p = all.len();
        let mut verts: Vec<SolidVertex> = Vec::new();
        let mut batch = crate::simd::TripleBatch::default();
        for a in 0..p {
            for b in (a + 1)..p {
                for c in (b + 1)..p {
                    let (pa, pb, pc) = (all[a], all[b], all[c]);
                    if batch.push((pa.0, pa.1), (pb.0, pb.1), (pc.0, pc.1)) {
                        flush_solid_batch_linear_reference(&batch, &soa, &mut verts)?;
                        batch = crate::simd::TripleBatch::default();
                    }
                }
            }
        }
        if batch.len > 0 {
            flush_solid_batch_linear_reference(&batch, &soa, &mut verts)?;
        }
        Some(verts.into_iter().map(|s| s.v).collect())
    }

    /// Runs both the production (`VertexAccumulator`-indexed) and reference
    /// (linear-scan) dedup over `planes` and asserts byte-identical vertex
    /// lists, in order.
    fn assert_dedup_matches_reference(planes: &[(DVec3, f64)], label: &str) {
        let deduped = dedup_planes(planes);
        let fast =
            feasible_vertices(&deduped).map(|v| v.into_iter().map(|s| s.v).collect::<Vec<_>>());
        let reference = feasible_vertices_linear_reference(&deduped);

        match (fast, reference) {
            (None, None) => {}
            (Some(f), Some(r)) => {
                assert_eq!(
                    f.len(),
                    r.len(),
                    "{label}: vertex count differs (indexed {} vs linear-scan reference {})",
                    f.len(),
                    r.len()
                );
                for (i, (fv, rv)) in f.iter().zip(r.iter()).enumerate() {
                    assert_eq!(
                        (fv.x.to_bits(), fv.y.to_bits(), fv.z.to_bits()),
                        (rv.x.to_bits(), rv.y.to_bits(), rv.z.to_bits()),
                        "{label}: vertex {i} differs (indexed {fv:?} vs reference {rv:?})"
                    );
                }
            }
            (f, r) => panic!(
                "{label}: indexed and reference dedup disagree on boundedness (indexed {:?}, reference {:?})",
                f.is_some(),
                r.is_some()
            ),
        }
    }

    #[test]
    fn dedup_matches_linear_scan_reference_on_module_fixtures() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_dedup_matches_reference(
            &[
                (DVec3::X, 1.0),
                (DVec3::NEG_X, 1.0),
                (DVec3::Y, 0.6),
                (DVec3::NEG_Y, 0.6),
                (DVec3::Z, 1.0),
                (DVec3::NEG_Z, 1.0),
            ],
            "plain box",
        );
        assert_dedup_matches_reference(
            &[
                (DVec3::X, 1.0),
                (DVec3::NEG_X, 1.0),
                (DVec3::Z, 1.0),
                (DVec3::NEG_Z, 1.0),
                (DVec3::NEG_Y, 0.5),
                (DVec3::new(s, s, 0.0), s),
                (DVec3::new(-s, s, 0.0), s),
                (DVec3::new(0.0, s, s), s),
                (DVec3::new(0.0, s, -s), s),
            ],
            "hip-roofed block",
        );
        assert_dedup_matches_reference(
            &[
                (DVec3::new(s, 0.0, s), 1.0),
                (DVec3::new(-s, 0.0, s), 1.0),
                (DVec3::new(s, 0.0, -s), 1.0),
                (DVec3::new(-s, 0.0, -s), 1.0),
                (DVec3::Y, 0.5),
                (DVec3::NEG_Y, 0.5),
            ],
            "rotated square girdle",
        );
    }

    /// Builds a real cutting schedule's plane arrangement (tier normals via
    /// `meet_solver::tier_instance_normals`, offsets from `solve_meet_points`'s
    /// solved masts) the same way `SolveContext::config_score` does when the
    /// solver scores a candidate configuration against a design's printed
    /// proportions -- this is the actual production caller of `measure_solid`
    /// that makes `feasible_vertices` dedup real numbers of colliding
    /// candidate vertices, not just the module's small hand-built fixtures.
    fn planes_from_asc_schedule(text: &str) -> Vec<(DVec3, f64)> {
        let schedule = lapidary::asc::parse_asc(text).expect("fixture schedule parses");
        let mut tiers = crate::geometry::meet_solver::meet_tier_inputs_from_asc(&schedule);
        for j in [0usize, 1, 2] {
            if let Some(t) = schedule.tiers.get(j) {
                tiers[j].constraint =
                    crate::geometry::meet_solver::MeetConstraint::ScaleReference(t.mast);
            }
        }
        let normals =
            crate::geometry::meet_solver::tier_instance_normals(schedule.gear_teeth_abs(), &tiers);
        let solved =
            crate::geometry::meet_solver::solve_meet_points(schedule.gear_teeth_abs(), &tiers);
        normals
            .iter()
            .zip(solved.iter().map(|s| s.mast))
            .flat_map(|(ns, m)| ns.iter().map(move |&n| (n, m)))
            .collect()
    }

    #[test]
    fn dedup_matches_linear_scan_reference_on_real_schedules() {
        // Same fixture text as `examples/simd_bench.rs`'s solver benchmark
        // ("Bench design" / pgo_train.rs's "Train A"): a 96-tooth round with
        // two crown tiers, table, and two pavilion tiers plus culet.
        assert_dedup_matches_reference(
            &planes_from_asc_schedule(
                "GemCad 5.0\n\
                 g 96 0.0\n\
                 y 6 y\n\
                 I 1.72\n\
                 H Bench design\n\
                 a -41.000000 0.64991234 92 n 1 84 76 68 60 52 44 36 28 20 12 4\n\
                 a -90.000000 1.07325092 92 n 2 84 76 68 60 52 44 36 28 20 12 4\n\
                 a 29.730000 0.65249790 4 n A 12 20 28 36 44 52 60 68 76 84 92\n\
                 a 25.000000 0.59508784 96 n B 16 32 48 64 80\n\
                 a 10.000000 0.48799664 96 n C 16 32 48 64 80\n\
                 a 0.000000 0.44000000 n T\n",
            ),
            "real schedule: Train A (96-tooth round)",
        );
        // pgo_train.rs's "Train B": mixed 96/6-tooth tiers, a heavier real
        // schedule with a different symmetry split.
        assert_dedup_matches_reference(
            &planes_from_asc_schedule(
                "GemCad 5.0\ng 96 0.0\ny 8 y\nI 1.54\nH Train B\n\
                 a -43.000000 0.70000000 96 n P1 12 24 36 48 60 72 84\n\
                 a -41.000000 0.68000000 6 n P2 18 30 42 54 66 78 90\n\
                 a -90.000000 1.00000000 96 n G 12 24 36 48 60 72 84\n\
                 a -90.000000 1.00000000 6 n G2 18 30 42 54 66 78 90\n\
                 a 42.000000 0.72000000 96 n C1 12 24 36 48 60 72 84\n\
                 a 27.000000 0.62000000 6 n C2 18 30 42 54 66 78 90\n\
                 a 0.000000 0.40000000 n T\n",
            ),
            "real schedule: Train B (mixed 96/6-tooth)",
        );
        // pgo_train.rs's "Train C": a simpler 4-fold real schedule.
        assert_dedup_matches_reference(
            &planes_from_asc_schedule(
                "GemCad 5.0\ng 96 0.0\ny 4 y\nI 1.62\nH Train C\n\
                 a -45.000000 0.75000000 96 n 1 24 48 72\n\
                 a -40.000000 0.70000000 12 n 2 36 60 84\n\
                 a -90.000000 1.05000000 96 n G 24 48 72\n\
                 a -90.000000 1.05000000 12 n G2 36 60 84\n\
                 a 35.000000 0.70000000 96 n 3 24 48 72\n\
                 a 20.000000 0.58000000 12 n 4 36 60 84\n\
                 a 0.000000 0.42000000 n T\n",
            ),
            "real schedule: Train C (4-fold)",
        );
    }
}

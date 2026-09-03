//! The deterministic candidate-vertex primitive: facet-plane normal
//! construction, the bounding blank, and enumerating every well-conditioned
//! plane triple into feasible candidate vertices, plus grouping their `n . v`
//! values into "levels". Everything here is plain nested loops over a fixed
//! plane order (no hashing, no convex-hull library) -- the property the whole
//! solver's determinism rests on -- and is batched through `crate::simd` for
//! speed without changing that order (see [`enumerate_candidate_vertices`]'s
//! doc comment).
//!
//! NOTE -- a per-design cache of every triple's mast-independent 3x3 inverse
//! (reused across phase-3 sweeps and repair-search pipeline runs) was built,
//! measured on the full corpus, and removed again (2026-09-02): phase-3 time is
//! the feasibility scan, not the triple solve, so the cache saved nothing
//! measurable (Reports A-D wall time 2489 s vs 2439 s without it) while writing
//! the ~13k cached inverses cost a single-sweep solve ~25% (2.44-2.9 ms vs
//! 2.12 ms on `examples/simd_bench.rs`).

use super::{
    BLANK_HALF_EXTENT, EPS_FEAS, LEVEL_TOL, MIN_TRIPLE_DET, MeetTierInput, blocks::tier_sides,
};
use glam::DVec3;

/// One plane of the working arrangement.
#[derive(Clone, Copy)]
pub(super) struct SolvePlane {
    pub(super) n: DVec3,
    pub(super) m: f64,
    /// Owning tier index, or `usize::MAX` for a bounding-blank plane.
    pub(super) owner: usize,
}

/// One candidate vertex of the working arrangement: position, the single tier whose
/// planes it violates (`None` = feasible against everything), and the owners of the
/// three planes that formed it.
pub(super) struct CandidateVertex {
    pub(super) v: DVec3,
    pub(super) violated: Option<usize>,
    pub(super) owners: [usize; 3],
}

/// Unit facet-plane normals for one tier's index instances, matching
/// [`super::super::cuts::StandardGemCuts::from_asc_schedule`]'s azimuth/crown
/// convention (`phi = 2*pi*index/gear_teeth_abs`, crown normals tilt toward `+y`,
/// pavilion toward `-y`), computed in `f64`.
pub(super) fn tier_normals(
    gear_teeth_abs: u32,
    angle_deg: f64,
    indices: &[f64],
    is_crown: bool,
) -> Vec<DVec3> {
    let theta = angle_deg.abs().to_radians();
    let (sin_t, cos_t) = (theta.sin(), theta.cos());
    let y = if is_crown { cos_t } else { -cos_t };
    let gear = f64::from(gear_teeth_abs.max(1));

    if indices.is_empty() {
        return vec![DVec3::new(0.0, y, sin_t)];
    }
    indices
        .iter()
        .map(|&idx| {
            let phi = 2.0 * std::f64::consts::PI * idx / gear;
            DVec3::new(sin_t * phi.cos(), y, sin_t * phi.sin())
        })
        .collect()
}

/// Every tier's per-instance unit facet-plane normals.
///
/// Exactly the construction [`solve_meet_points`](super::solve_meet_points) uses
/// internally (same gear/azimuth/crown conventions, same unsigned-zero side
/// inheritance), made public so external measurement code (e.g.
/// [`super::super::stone_metrics`]) can rebuild the same plane arrangement from
/// any mast vector, real or solved.
#[must_use]
pub fn tier_instance_normals(gear_teeth_abs: u32, tiers: &[MeetTierInput]) -> Vec<Vec<DVec3>> {
    let sides = tier_sides(tiers);
    tiers
        .iter()
        .zip(&sides)
        .map(|(t, &crown)| tier_normals(gear_teeth_abs, t.angle_deg, &t.indices, crown))
        .collect()
}

/// The six bounding-blank planes.
pub(super) fn blank_planes() -> Vec<SolvePlane> {
    [
        DVec3::X,
        DVec3::NEG_X,
        DVec3::Y,
        DVec3::NEG_Y,
        DVec3::Z,
        DVec3::NEG_Z,
    ]
    .into_iter()
    .map(|n| SolvePlane {
        n,
        m: BLANK_HALF_EXTENT,
        owner: usize::MAX,
    })
    .collect()
}

/// Drains one full (or final partial) [`crate::simd::TripleBatch`] solve into
/// `out`, in ascending lane order -- the same order the flushed triples were
/// pushed in. Shared by [`enumerate_candidate_vertices`]'s batching loop.
fn flush_candidate_batch(
    batch: &crate::simd::TripleBatch,
    soa: &crate::simd::PlanesSoA64,
    owner_meta: &[[usize; 3]; crate::simd::TRIPLE_LANES],
    out: &mut Vec<CandidateVertex>,
) {
    let sol = crate::simd::solve_triple_batch(batch);
    for (lane, owners) in owner_meta.iter().enumerate().take(batch.len) {
        if sol.det[lane].abs() < MIN_TRIPLE_DET {
            continue;
        }
        let v = DVec3::new(sol.vx[lane], sol.vy[lane], sol.vz[lane]);
        if v.x.abs() > BLANK_HALF_EXTENT + 1.0
            || v.y.abs() > BLANK_HALF_EXTENT + 1.0
            || v.z.abs() > BLANK_HALF_EXTENT + 1.0
        {
            continue;
        }
        match crate::simd::classify_feasibility(soa, v, EPS_FEAS) {
            crate::simd::Feasibility::Dead => {}
            crate::simd::Feasibility::Ok(violated) => {
                out.push(CandidateVertex {
                    v,
                    violated: violated.map(|o| o as usize),
                    owners: *owners,
                });
            }
        }
    }
}

/// Enumerates every candidate meet vertex of the arrangement: all triples of real
/// planes with `|det| >=` [`MIN_TRIPLE_DET`], solved in `f64`, kept when the point
/// violates the half-spaces of at most one tier (and never the bounding blank).
///
/// Deterministic by construction: plain nested loops over a fixed plane order, no
/// hashing, no convex-hull library.
///
/// Batched through `crate::simd`: the plane arena (`PlanesSoA64`) is built once
/// up front, and every triple's determinant/inverse solve runs through
/// `solve_triple_batch` via [`flush_candidate_batch`] (SIMD-dispatched,
/// bit-identical per lane to the `glam` `DMat3` sequence it replaces -- see the
/// determinism contract at the top of `src/simd/mod.rs`). Solved lanes are always
/// drained in ascending order, and triples are still generated by the same
/// nested loops in the same order as before, so the candidates produced here
/// are byte-for-byte identical to the unbatched scalar solve this replaces.
pub(super) fn enumerate_candidate_vertices(
    planes: &[SolvePlane],
    first_real: usize,
) -> Vec<CandidateVertex> {
    let mut soa = crate::simd::PlanesSoA64::with_capacity(planes.len());
    for q in planes {
        let owner = if q.owner == usize::MAX {
            crate::simd::BLANK_OWNER
        } else {
            q.owner as u32
        };
        soa.push(q.n, q.m, owner);
    }

    let p = planes.len();
    let mut out = Vec::new();
    let mut batch = crate::simd::TripleBatch::default();
    let mut owner_meta = [[0usize; 3]; crate::simd::TRIPLE_LANES];
    for a in first_real..p {
        for b in (a + 1)..p {
            for c in (b + 1)..p {
                let (pa, pb, pc) = (planes[a], planes[b], planes[c]);
                owner_meta[batch.len] = [pa.owner, pb.owner, pc.owner];
                if batch.push((pa.n, pa.m), (pb.n, pb.m), (pc.n, pc.m)) {
                    flush_candidate_batch(&batch, &soa, &owner_meta, &mut out);
                    batch = crate::simd::TripleBatch::default();
                }
            }
        }
    }
    if batch.len > 0 {
        flush_candidate_batch(&batch, &soa, &owner_meta, &mut out);
    }
    out
}

/// Groups candidate `n . v` values into distinct "levels" (descending; values within
/// [`LEVEL_TOL`] of a level's head belong to it).
pub(super) fn group_levels(mut vals: Vec<f64>) -> Vec<f64> {
    vals.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    let mut levels: Vec<f64> = Vec::new();
    for v in vals {
        match levels.last() {
            Some(&head) if (head - v).abs() <= LEVEL_TOL => {}
            _ => levels.push(v),
        }
    }
    levels
}

/// Keeps only the levels that are realizable for *every* index instance of the tier:
/// a level (derived from instance 0's values) survives if each other instance also
/// has a candidate value within tolerance of it. At the true configuration this
/// holds for 95% of tiers (measured), and it prunes levels that would place the
/// tier's symmetric copies off any meet vertex. Falls back to the unfiltered list
/// when the filter would empty it.
///
/// Per other-instance `j`, `usable`'s `n_j . v` values are sorted once (`total_cmp`,
/// so a value's presence within a level's `4*LEVEL_TOL` band is a single binary
/// search) instead of scanned once per level -- same result set and order (the
/// filter, like before, only ever drops elements of `levels`, never reorders it),
/// just O((instances-1)*usable*log(usable)) instead of
/// O(levels*instances*usable).
pub(super) fn filter_levels_by_instance_support(
    levels: Vec<f64>,
    normals_i: &[DVec3],
    usable: &[&CandidateVertex],
) -> Vec<f64> {
    if normals_i.len() < 2 || levels.is_empty() {
        return levels;
    }
    let sorted_per_instance: Vec<Vec<f64>> = normals_i[1..]
        .iter()
        .map(|&nj| {
            let mut vals: Vec<f64> = usable.iter().map(|c| nj.dot(c.v)).collect();
            vals.sort_by(f64::total_cmp);
            vals
        })
        .collect();
    let tol = 4.0 * LEVEL_TOL;
    let supported: Vec<f64> = levels
        .iter()
        .copied()
        .filter(|&head| {
            sorted_per_instance
                .iter()
                .all(|vals| any_within_tol(vals, head, tol))
        })
        .collect();
    if supported.is_empty() {
        levels
    } else {
        supported
    }
}

/// `true` iff `sorted` (ascending, `total_cmp` order) has a value within `tol` of
/// `target`, via binary search instead of a full scan. `total_cmp` (not `<`) drives
/// the search so it stays consistent with `sorted`'s own order on every input,
/// including the non-finite values a degenerate triple could in principle produce.
fn any_within_tol(sorted: &[f64], target: f64, tol: f64) -> bool {
    let lower = target - tol;
    let idx = sorted.partition_point(|v| v.total_cmp(&lower).is_lt());
    idx < sorted.len() && sorted[idx].total_cmp(&(target + tol)).is_le()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic LCG (no external RNG dependency), matching the
    /// convention `simd`'s own tests use.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (((*state >> 33) as f64) / f64::from(u32::MAX)).mul_add(2.0, -1.0)
    }

    /// A realistic-shaped multi-tier normal set (crown, pavilion and girdle
    /// tiers at varied angles/azimuth counts), matching the plane arrangement
    /// `SolveContext` builds from a real schedule.
    fn sample_normals() -> Vec<Vec<DVec3>> {
        vec![
            tier_normals(96, 90.0, &[0.0, 24.0, 48.0, 72.0], true), // girdle
            tier_normals(96, 41.0, &[12.0, 36.0, 60.0, 84.0], true),
            tier_normals(96, 25.0, &[16.0, 32.0, 48.0, 64.0, 80.0], true),
            tier_normals(96, 0.0, &[], true), // table
            tier_normals(96, -41.0, &[12.0, 36.0, 60.0, 84.0], false),
            tier_normals(96, -30.0, &[4.0, 20.0, 44.0, 68.0, 92.0], false),
            tier_normals(96, -90.0, &[], false), // culet
        ]
    }

    /// The full working arrangement (blanks first, then every tier's planes in
    /// tier/instance order) at the given masts -- exactly how phase 3 builds it.
    fn arrangement(normals: &[Vec<DVec3>], mast: &[f64]) -> Vec<SolvePlane> {
        let mut planes = blank_planes();
        for (i, ns) in normals.iter().enumerate() {
            for &nv in ns {
                planes.push(SolvePlane {
                    n: nv,
                    m: mast[i],
                    owner: i,
                });
            }
        }
        planes
    }

    /// Reference (pre-S2) linear-scan implementation, kept only to prove the
    /// binary-search version above returns an identical result set.
    fn filter_levels_by_instance_support_linear(
        levels: &[f64],
        normals_i: &[DVec3],
        usable: &[&CandidateVertex],
    ) -> Vec<f64> {
        if normals_i.len() < 2 || levels.is_empty() {
            return levels.to_vec();
        }
        let supported: Vec<f64> = levels
            .iter()
            .copied()
            .filter(|&head| {
                normals_i[1..].iter().all(|&nj| {
                    usable
                        .iter()
                        .any(|c| (nj.dot(c.v) - head).abs() <= 4.0 * LEVEL_TOL)
                })
            })
            .collect();
        if supported.is_empty() {
            levels.to_vec()
        } else {
            supported
        }
    }

    /// S2: the binary-search `filter_levels_by_instance_support` must return
    /// exactly the same levels, in the same order, as the linear-scan
    /// reference above, on random-but-seeded synthetic levels/candidates and
    /// on a real arrangement's own candidates.
    #[test]
    fn filter_levels_by_instance_support_matches_linear_reference() {
        let mut state = 99u64;
        for _case in 0..200 {
            let n_normals = 1 + (lcg(&mut state).abs() * 4.0) as usize; // 1..=4
            let normals_i: Vec<DVec3> = (0..n_normals)
                .map(|_| DVec3::new(lcg(&mut state), lcg(&mut state), lcg(&mut state)).normalize())
                .collect();
            let owned_verts: Vec<CandidateVertex> = (0..20)
                .map(|_| CandidateVertex {
                    v: DVec3::new(lcg(&mut state), lcg(&mut state), lcg(&mut state)) * 2.0,
                    violated: None,
                    owners: [0, 1, 2],
                })
                .collect();
            let usable: Vec<&CandidateVertex> = owned_verts.iter().collect();
            let levels: Vec<f64> = {
                let mut v: Vec<f64> = (0..8).map(|_| lcg(&mut state) * 2.0).collect();
                v.sort_by(|x, y| y.partial_cmp(x).unwrap());
                v
            };

            let want = filter_levels_by_instance_support_linear(&levels, &normals_i, &usable);
            let got = filter_levels_by_instance_support(levels, &normals_i, &usable);
            assert_eq!(want, got);
        }
    }

    /// Same equivalence check, on candidates actually produced by a real
    /// arrangement (rather than synthetic random vertices).
    #[test]
    fn filter_levels_by_instance_support_matches_linear_reference_on_real_arrangement() {
        let normals = sample_normals();
        let mast: Vec<f64> = normals.iter().map(|_| 1.0).collect();
        let cands = enumerate_candidate_vertices(&arrangement(&normals, &mast), 6);
        for (i, normals_i) in normals.iter().enumerate() {
            let usable: Vec<&CandidateVertex> = cands
                .iter()
                .filter(|c| {
                    !c.owners.contains(&i) && (c.violated.is_none() || c.violated == Some(i))
                })
                .collect();
            let n0 = normals_i[0];
            let levels = group_levels(
                usable
                    .iter()
                    .map(|c| n0.dot(c.v))
                    .filter(|v| *v > 1e-9)
                    .collect(),
            );
            let want = filter_levels_by_instance_support_linear(&levels, normals_i, &usable);
            let got = filter_levels_by_instance_support(levels, normals_i, &usable);
            assert_eq!(want, got, "tier {i}");
        }
    }

    /// The batched enumeration must produce exactly the candidates (same bits,
    /// same order, same owners/violations) an unbatched `glam` reference solve
    /// of every real-plane triple produces, across several mast vectors.
    #[test]
    fn batched_enumeration_matches_glam_reference_bitwise() {
        let normals = sample_normals();
        let mut state = 11u64;
        for round in 0..4 {
            let mast: Vec<f64> = (0..normals.len())
                .map(|_| 0.6f64.mul_add(lcg(&mut state).abs(), 0.4))
                .collect();
            let planes = arrangement(&normals, &mast);
            let mut soa = crate::simd::PlanesSoA64::with_capacity(planes.len());
            for q in &planes {
                let owner = if q.owner == usize::MAX {
                    crate::simd::BLANK_OWNER
                } else {
                    q.owner as u32
                };
                soa.push(q.n, q.m, owner);
            }

            let mut want: Vec<CandidateVertex> = Vec::new();
            let p = planes.len();
            for a in 6..p {
                for b in (a + 1)..p {
                    for c in (b + 1)..p {
                        let (pa, pb, pc) = (planes[a], planes[b], planes[c]);
                        let m = glam::DMat3::from_cols(pa.n, pb.n, pc.n).transpose();
                        if m.determinant().abs() < MIN_TRIPLE_DET {
                            continue;
                        }
                        let v = m.inverse() * DVec3::new(pa.m, pb.m, pc.m);
                        if v.x.abs() > BLANK_HALF_EXTENT + 1.0
                            || v.y.abs() > BLANK_HALF_EXTENT + 1.0
                            || v.z.abs() > BLANK_HALF_EXTENT + 1.0
                        {
                            continue;
                        }
                        match crate::simd::classify_feasibility(&soa, v, EPS_FEAS) {
                            crate::simd::Feasibility::Dead => {}
                            crate::simd::Feasibility::Ok(violated) => {
                                want.push(CandidateVertex {
                                    v,
                                    violated: violated.map(|o| o as usize),
                                    owners: [pa.owner, pb.owner, pc.owner],
                                });
                            }
                        }
                    }
                }
            }
            let got = enumerate_candidate_vertices(&planes, 6);
            assert_eq!(
                want.len(),
                got.len(),
                "candidate count mismatch, round {round}"
            );
            for (w, g) in want.iter().zip(&got) {
                assert_eq!(w.v.x.to_bits(), g.v.x.to_bits());
                assert_eq!(w.v.y.to_bits(), g.v.y.to_bits());
                assert_eq!(w.v.z.to_bits(), g.v.z.to_bits());
                assert_eq!(w.violated, g.violated);
                assert_eq!(w.owners, g.owners);
            }
        }
    }
}

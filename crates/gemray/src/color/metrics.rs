use crate::{
    geometry::plane::GpuFacetPlane,
    optics::{
        materials::GemMaterial,
        raytracer::{Ray, build_plane_soa, intersect_polyhedron_soa},
    },
};
use glam::Vec3;

/// Builds the observer-PoV view basis (forward, right, up) from camera yaw/pitch.
///
/// Uses the exact same convention as `Camera::new` in `optics::raytracer` (same
/// `world_up` fallback threshold and axis), so that gemological metrics are evaluated
/// against the same frame that is actually rendered.
#[must_use]
pub fn camera_view_basis(cam_yaw: f32, cam_pitch: f32) -> (Vec3, Vec3, Vec3) {
    let cos_cp = cam_pitch.cos();
    let sin_cp = cam_pitch.sin();
    let cos_cy = cam_yaw.cos();
    let sin_cy = cam_yaw.sin();
    let cam_forward = Vec3::new(-cos_cp * sin_cy, -sin_cp, -cos_cp * cos_cy).normalize();
    let world_up = if cos_cp.abs() < 1e-4 {
        Vec3::new(0.0, 0.0, -1.0)
    } else {
        Vec3::Y
    };
    let cam_right = cam_forward.cross(world_up).normalize();
    let cam_up = cam_right.cross(cam_forward).normalize();
    (cam_forward, cam_right, cam_up)
}

#[derive(Debug, Clone, Copy)]
pub struct GemOpticalMetrics {
    pub brilliance_pct: f32,
    pub fire_index: f32,
    pub scintillation_pct: f32,
    pub windowing_pct: f32,
    pub extinction_pct: f32,
}

/// 19 tilt elevation sample points in exact 5° steps across 0° to 90°
pub const PROFILE_ANGLES_DEG: [f32; 19] = [
    0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0,
    80.0, 85.0, 90.0,
];

/// Outcome of refracting an incident ray into the gemstone at a known entry point/normal
/// (for one specific refractive index) and following it through up to 10 internal bounces.
/// Factored out of the main grid loop so the exact same entry-refraction-then-bounce logic
/// can be replayed at the d-line index (for the windowing/extinction/brilliance
/// classification) and independently at the F-line and C-line indices (for the Fire
/// measurement below), from the identical physical entry point and incident direction.
#[derive(Debug, Clone, Copy)]
enum RayFate {
    /// Entry refraction has no real solution at this index (only possible for a
    /// pathological index < 1). Matches the pre-refactor code's silent `continue`: the
    /// sample is not classified into any bucket. Unreachable for the d-line index, which
    /// is clamped to >= 1.1 above, but kept for defensive symmetry with the F/C traces.
    EntryBlocked,
    /// Leaked out through the pavilion bottom (`n_out.y < -0.05`): windowing.
    Leaked,
    /// Exited back out through the upper hemisphere/crown with the given direction, the
    /// fraction of this ray's incident intensity that actually survives to be
    /// transmitted along this exact path (Fresnel transmittance at the entry interface
    /// times Fresnel transmittance at the exit interface -- see `fresnel_transmittance`),
    /// and the cosine of the exit angle measured from the exit facet's own normal (`cos
    /// theta_out` in the refraction construction below). TIR bounces in between are
    /// lossless in this model (no energy escapes a bounce that stays below the critical
    /// angle, matching the fact that windowing/extinction/brilliance above don't model
    /// absorption either), so entry x exit is the complete transmittance for this
    /// idealized dielectric. The exit cosine is carried separately from transmittance
    /// because it measures a distinct effect: Fresnel transmittance is "how much of this
    /// ray's energy crosses the interface at all", while the exit cosine is "how much
    /// solid angle that transmitted energy gets smeared across before it can reach an
    /// observer" (projected-area radiance falloff -- see the Fire weighting below).
    ///
    /// Also carries the index of the facet this ray exited through and the number of
    /// internal bounces it took to get there -- see `ExitPath`'s doc for why: the Fire
    /// measurement below uses these to detect when a ray's F-line and C-line companion
    /// traces took physically disjoint paths through the stone (see the F/C bifurcation
    /// gate in `evaluate_gem_optical_metrics`).
    ExitedUpward(ExitPath),
    /// Trapped internally: exhausted its bounce budget, exited sideways through the
    /// girdle, or hit no further facet.
    Absorbed,
}

/// The physical exit path of a ray that escaped upward through the crown: its exit
/// direction, the Fresnel entry*exit transmittance and exit-facet-normal cosine (see
/// `RayFate::ExitedUpward`'s doc), plus which facet it exited through and how many
/// internal TIR bounces it took to get there.
///
/// The facet index and bounce count exist purely so the Fire measurement can tell
/// whether a ray's F-line and C-line companion traces exited via the SAME physical
/// path. When a ray sits near a critical angle, F and C (whose refractive indices
/// differ) can straddle the TIR threshold at different bounces -- one refracts out
/// early, the other reflects on and exits several facets later through a completely
/// different part of the stone. `acos(dir_f . dir_c)` between two such unrelated exit
/// directions measures nothing physically meaningful (it is not "dispersion", it is two
/// different rays that happen to share an entry point), yet the raw angle can be tens of
/// degrees and dominate the weighted Fire sum. See the bifurcation gate in
/// `evaluate_gem_optical_metrics` for how this is used.
#[derive(Debug, Clone, Copy)]
struct ExitPath {
    dir: Vec3,
    transmittance: f32,
    exit_cos_theta: f32,
    facet_idx: usize,
    bounces: u32,
}

/// Unpolarized Fresnel transmittance at a dielectric interface, given the cosines of the
/// incident and transmitted angles on either side (`n1` -> `n2`). Averages the s- and
/// p-polarized reflectances (`Rs`, `Rp`) into a single scalar reflectance `R`, then
/// returns `T = 1 - R`, clamped to [0, 1]. Used to weight each ray's contribution to Fire
/// by the energy it actually delivers, rather than counting every surviving ray equally
/// regardless of how much of its light made it through the entry and exit interfaces.
fn fresnel_transmittance(n1: f32, n2: f32, cos_i: f32, cos_t: f32) -> f32 {
    let denom_s = n2.mul_add(cos_t, n1 * cos_i);
    let denom_p = n2.mul_add(cos_i, n1 * cos_t);
    // Denominators vanish only at grazing incidence (cos_i, cos_t -> 0 together), where
    // the physical reflectance already tends to 1 (T -> 0); reporting zero transmittance
    // there is both safe (avoids a 0/0 NaN) and physically correct.
    if denom_s.abs() < 1e-6 || denom_p.abs() < 1e-6 {
        return 0.0;
    }
    let rs = (n2.mul_add(-cos_t, n1 * cos_i) / denom_s).powi(2);
    let rp = (n2.mul_add(-cos_i, n1 * cos_t) / denom_p).powi(2);
    let r = f32::midpoint(rs, rp);
    (1.0 - r).clamp(0.0, 1.0)
}

/// Refracts `incoming_dir` into the gem at `entry_point`/`n_entry` using Snell's law for
/// `index`, then follows up to 10 internal bounces (TIR vs refract-out) exactly as the
/// original single-wavelength trace did. This is the physical core shared by the d-line,
/// F-line, and C-line traces -- identical control flow, parameterized only by which
/// refractive index the light is carrying.
fn trace_wavelength(
    entry_point: Vec3,
    incoming_dir: Vec3,
    n_entry: Vec3,
    cos_i: f32,
    plane_soa: &crate::simd::PlanesSoA32,
    index: f32,
) -> RayFate {
    let sin2_t = (1.0 / (index * index)) * cos_i.mul_add(-cos_i, 1.0);
    if sin2_t > 1.0 {
        return RayFate::EntryBlocked;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    let mut curr_dir =
        ((1.0 / index) * incoming_dir + (1.0 / index).mul_add(cos_i, -cos_t) * n_entry).normalize();
    let mut hit_point = entry_point + curr_dir * 1e-4;
    let sin_crit = 1.0 / index;
    // Entry transmittance (air -> gem): computed once here since cos_i/cos_t at entry
    // don't change across bounces; combined with the exit transmittance below to give
    // this path's total energy weight.
    let entry_transmittance = fresnel_transmittance(1.0, index, cos_i, cos_t);

    let mut leaked = false;
    let mut exited_upwards = false;
    let mut exit_dir = Vec3::ZERO;
    let mut exit_transmittance = 0.0f32;
    let mut exit_cos_theta = 0.0f32;
    let mut exit_facet_idx = usize::MAX;
    let mut exit_bounces = 0u32;

    for bounce in 0..10 {
        let inside_ray = Ray {
            origin: hit_point,
            dir: curr_dir,
        };
        let next_hit = intersect_polyhedron_soa(inside_ray, plane_soa);
        let Some(next_rec) = next_hit else { break };
        let next_point = inside_ray.origin + next_rec.t * inside_ray.dir;
        let n_out = next_rec.normal; // outward-pointing facet normal

        let cos_theta = curr_dir.dot(n_out).clamp(0.0, 1.0);
        let sin_theta = cos_theta.mul_add(-cos_theta, 1.0).max(0.0).sqrt();

        if sin_theta < sin_crit {
            // Refracts out of the stone (TIR failed)
            let sin2_out = (index * index) * cos_theta.mul_add(-cos_theta, 1.0);
            if sin2_out <= 1.0 {
                let cos_out = (1.0 - sin2_out).sqrt();
                let out_dir =
                    (index * curr_dir + index.mul_add(-cos_theta, cos_out) * n_out).normalize();
                if n_out.y < -0.05 {
                    // Leaks out through pavilion bottom -> WINDOWING
                    leaked = true;
                } else if out_dir.y > 0.05 {
                    // Exits back toward upper hemisphere / crown
                    exited_upwards = true;
                    exit_dir = out_dir;
                    // Exit transmittance (gem -> air) at this exact facet/angle.
                    exit_transmittance = fresnel_transmittance(index, 1.0, cos_theta, cos_out);
                    // Cosine of the exit angle from the exit facet's own normal --
                    // the projected-area radiometric factor used to weight Fire below.
                    exit_cos_theta = cos_out;
                    // Which facet this ray physically left through, and how many
                    // internal TIR bounces it took to get there -- see `ExitPath`'s doc.
                    exit_facet_idx = next_rec.facet_idx;
                    exit_bounces = bounce;
                }
                break;
            }
        }

        // Total Internal Reflection (TIR)
        curr_dir = (curr_dir - 2.0 * cos_theta * n_out).normalize();
        hit_point = next_point + curr_dir * 1e-4;
    }

    if leaked {
        RayFate::Leaked
    } else if exited_upwards {
        RayFate::ExitedUpward(ExitPath {
            dir: exit_dir,
            transmittance: entry_transmittance * exit_transmittance,
            exit_cos_theta,
            facet_idx: exit_facet_idx,
            bounces: exit_bounces,
        })
    } else {
        RayFate::Absorbed
    }
}

/// Whether a ray that exited the gem in direction `exit_dir` is visibly returned to an
/// observer at `cam_forward` under the given key/fill/overhead-ring illumination -- i.e.
/// not lost to the observer's own head-shadow, AND actually collected by at least one of
/// the three light sources. This is the exact same "is this light visible" test used for
/// `brilliance_pct` / `extinction_pct` classification in the main loop below (Directional
/// Extinction Analysis), factored out here so the Scintillation temporal sub-poses (see
/// `cell_returned_at_yaw_offset` below) apply an identical definition of "returned" at
/// each rotated pose rather than a hand-rolled approximation of it.
#[must_use]
fn ray_is_visibly_returned(
    exit_dir: Vec3,
    cam_forward: Vec3,
    key_dir: Vec3,
    fill_dir: Vec3,
    sin_lp: f32,
) -> bool {
    // 1. Head-shadow cone (angle < 16 deg from viewing vector)
    let is_head_shadow = (-exit_dir).dot(cam_forward) > 0.96;

    // 2. Light collection from Key, Fill, or Overhead ring illumination
    let key_dot = exit_dir.dot(key_dir).max(0.0);
    let fill_dot = exit_dir.dot(fill_dir).max(0.0);
    let ring_dot = sin_lp.mul_add(-0.8, exit_dir.y).abs() < 0.35;
    let is_illuminated = (key_dot > 0.70) || (fill_dot > 0.75) || (ring_dot && exit_dir.y > 0.2);

    !is_head_shadow && is_illuminated
}

/// Small camera-yaw offsets (degrees), sampled around the primary viewing azimuth, used
/// to measure Scintillation's TEMPORAL component: how much a given grid cell's light
/// return *changes* as the stone is gently rotated, as distinct from the static spatial
/// contrast measured across the grid at a single fixed pose. This is "a handful of
/// samples across a few degrees of rotation" -- five poses (odd count -- see
/// `cell_returned_at_yaw_offset`'s doc for why), spanning +/-3 deg, comparable in scale
/// to a hand gently tilting a stone for inspection, not a full turn.
const SCINT_TEMPORAL_YAW_OFFSETS_DEG: [f32; 5] = [-3.0, -1.5, 0.0, 1.5, 3.0];

/// Weight of the spatial (per-pose, per-grid-cell contrast) term in the combined
/// `scintillation_pct`. See `combine_scintillation_pct` for the full weighting rationale.
const SCINT_SPATIAL_WEIGHT: f32 = 0.6;
/// Weight of the temporal (per-cell, across-pose flicker) term in the combined
/// `scintillation_pct`. See `combine_scintillation_pct` for the full weighting rationale.
const SCINT_TEMPORAL_WEIGHT: f32 = 0.4;

/// Per-evaluation context threaded through the Scintillation temporal sub-poses
/// (`cell_returned_at_yaw_offset`): everything needed to fire and classify one extra
/// ray at a rotated camera azimuth, bundled into one struct purely to keep that
/// function's argument count within clippy's `too_many_arguments` lint -- these six
/// values are otherwise identical across all `grid_size * grid_size *
/// SCINT_TEMPORAL_YAW_OFFSETS_DEG.len()` calls per `evaluate_gem_optical_metrics`
/// invocation.
struct TemporalPoseContext<'a> {
    plane_soa: &'a crate::simd::PlanesSoA32,
    nd: f32,
    cam_yaw: f32,
    cam_pitch: f32,
    key_dir: Vec3,
    fill_dir: Vec3,
    sin_lp: f32,
}

/// Fires a single, non-jittered ray at grid cell `(u, v)` (same screen-space coordinates
/// as the main grid loop, in `[-1, 1]`) from the camera pose rotated by `yaw_offset_deg`
/// around `ctx.cam_yaw`, refracts it at the d-line only, and reports whether it is
/// visibly returned to an observer at that pose (`ray_is_visibly_returned` above).
/// Deliberately independent of the main grid loop's per-cell state: temporal modulation
/// is a distinct physical question (does the SAME screen-space cell's return status flip
/// as the viewpoint rotates) from the spatial sub-aperture contrast measured there, and
/// no fire (F-line/C-line) companion trace is needed since only whether light returns --
/// not its color separation -- is relevant to sparkle.
///
/// Odd offset counts (see [`SCINT_TEMPORAL_YAW_OFFSETS_DEG`]) avoid a subtle test hazard:
/// with an even sample count, a cell that flips exactly half the time lands at the
/// maximum possible Bernoulli variance (p=0.5, p*(1-p)=0.25) exactly, and if that
/// happened for literally every visited cell the temporal term would read exactly 100%
/// -- defeating the same "never let a percentage metric actually reach its ceiling"
/// property `scintillation_pct` already guards via the `cv / (1 + cv)` spatial squash
/// (see `no_built_in_material_saturates_scintillation_on_either_cut` in
/// `tests/metrics_tests.rs`). With 5 (odd) samples, the achievable discrete probabilities
/// are k/5 for k in 0..=5, whose variance k/5 * (1 - k/5) tops out at 0.24 (k=2 or 3),
/// strictly below the continuous maximum of 0.25 -- so the temporal term can approach but
/// never exactly reach 100% either, by the same construction as the spatial term, without
/// needing its own hyperbolic squash.
#[must_use]
fn cell_returned_at_yaw_offset(
    ctx: &TemporalPoseContext,
    yaw_offset_deg: f32,
    u: f32,
    v: f32,
) -> bool {
    let (forward, right, up) =
        camera_view_basis(ctx.cam_yaw + yaw_offset_deg.to_radians(), ctx.cam_pitch);
    let ray = Ray {
        origin: -forward * 2.5 + (u * 0.95) * right + (v * 0.95) * up,
        dir: forward,
    };
    let Some(hit_rec) = intersect_polyhedron_soa(ray, ctx.plane_soa) else {
        return false;
    };
    let hit_point = ray.origin + hit_rec.t * ray.dir;
    let n_entry = hit_rec.normal;
    let cos_i = (-ray.dir).dot(n_entry).clamp(0.0, 1.0);

    match trace_wavelength(hit_point, ray.dir, n_entry, cos_i, ctx.plane_soa, ctx.nd) {
        RayFate::ExitedUpward(exit) => {
            ray_is_visibly_returned(exit.dir, forward, ctx.key_dir, ctx.fill_dir, ctx.sin_lp)
        }
        RayFate::EntryBlocked | RayFate::Leaked | RayFate::Absorbed => false,
    }
}

/// Scintillation TEMPORAL contribution of one grid cell `(u, v)`: does its return status
/// flip as the camera nudges through the few small azimuth offsets in
/// [`SCINT_TEMPORAL_YAW_OFFSETS_DEG`]? Computed via independent single-ray samples (see
/// `cell_returned_at_yaw_offset`), reduced to the Bernoulli variance `p * (1 - p)` of the
/// fraction `p` of offsets at which the cell returned light: 0 when the cell's return
/// status never flips across the sampled poses (always on, or always off -- bright or
/// dark, but not sparkling), up to 0.24 (see `cell_returned_at_yaw_offset`'s doc for why
/// not the continuous maximum 0.25) when it flips close to half the time.
#[must_use]
fn cell_temporal_variance(ctx: &TemporalPoseContext, u: f32, v: f32) -> f32 {
    let mut temporal_returned = 0u32;
    for &yaw_offset_deg in &SCINT_TEMPORAL_YAW_OFFSETS_DEG {
        if cell_returned_at_yaw_offset(ctx, yaw_offset_deg, u, v) {
            temporal_returned += 1;
        }
    }
    let temporal_p = temporal_returned as f32 / SCINT_TEMPORAL_YAW_OFFSETS_DEG.len() as f32;
    temporal_p.mul_add(-temporal_p, temporal_p)
}

/// Scintillation SPATIAL term: coefficient of variation (std-dev / mean) of per-cell
/// light-return fraction across the 18x18 grid (`cell_fraction_sum`, `cell_fraction_sum_sq`,
/// `cell_count` accumulated in the main grid loop), mapped into a 0-100 percentage.
/// CV = 0 means every visited cell returns light at the same rate (a uniformly bright or
/// uniformly dark stone -- no contrast, no sparkle). CV climbs as bright and dark cells
/// increasingly diverge, and for these per-facet-cell light-return distributions (sparse
/// bright cells against a mostly-dark field is the common case, not the exception) CV
/// routinely lands well above 1.0 -- a straight `(cv * 100.0).clamp(0.0, 100.0)` was
/// measured to saturate at exactly 100% for 19 of 26 material/cut combinations swept
/// across the built-ins, which collapses the metric's ability to discriminate among most
/// stones (see `fire_and_scintillation_are_finite_and_in_range_for_every_material_on_
/// both_cuts` and `no_built_in_material_saturates_scintillation_on_either_cut` in
/// `metrics_tests.rs`).
///
/// Instead, squash CV through `cv / (1 + cv)`, a monotone bijection from [0, inf) to
/// [0, 1): CV=0 -> 0%, CV=1 -> 50%, CV=2 -> 66.7%, CV=9 -> 90%, asymptotically
/// approaching but mathematically never reaching 100% however large CV gets. This keeps
/// the metric strictly ordered (higher CV always reads higher) and strictly below the
/// ceiling, so two stones with different spatial contrast always produce different
/// displayed values instead of both clipping to the same 100. A stone with no visited
/// cells (e.g. camera facing entirely outside the gem) reports 0: there is no light
/// return to contrast.
#[must_use]
fn spatial_scintillation_pct(
    cell_fraction_sum: f32,
    cell_fraction_sum_sq: f32,
    cell_count: u32,
) -> f32 {
    if cell_count == 0 {
        return 0.0;
    }
    let mean_frac = cell_fraction_sum / cell_count as f32;
    if mean_frac <= 1e-4 {
        return 0.0;
    }
    let variance = mean_frac
        .mul_add(-mean_frac, cell_fraction_sum_sq / cell_count as f32)
        .max(0.0);
    let std_dev = variance.sqrt();
    let coefficient_of_variation = std_dev / mean_frac;
    (coefficient_of_variation / (1.0 + coefficient_of_variation) * 100.0).clamp(0.0, 100.0)
}

/// Scintillation TEMPORAL term: mean per-cell Bernoulli variance (`temporal_variance_sum`,
/// accumulated in the main grid loop via `cell_temporal_variance`) across the same
/// visited-cell set (`cell_count`) as the spatial term, normalized by its own achievable
/// maximum (0.24, not the continuous 0.25 -- see `cell_returned_at_yaw_offset`) into a
/// 0-100 percentage. This one needs no separate saturating squash: it is already bounded
/// by construction (mean of per-cell values that individually top out at 0.24 / 0.24 =
/// 100%), and averaging many cells' worth of samples makes the whole grid actually
/// landing on that per-cell ceiling simultaneously vanishingly unlikely in practice (see
/// `no_built_in_material_saturates_scintillation_on_either_cut`, which pins this for
/// every built-in material).
#[must_use]
fn temporal_scintillation_pct(temporal_variance_sum: f32, cell_count: u32) -> f32 {
    if cell_count == 0 {
        return 0.0;
    }
    ((temporal_variance_sum / cell_count as f32) / 0.24 * 100.0).clamp(0.0, 100.0)
}

/// Combines the Scintillation spatial and temporal terms (both already 0-100
/// percentages) into the final displayed `scintillation_pct`.
///
/// The reference gemological definition of scintillation is spatial AND temporal
/// modulation together -- a static bright/dark pattern that does not change as the stone
/// moves is not "sparkle", and a stone that flickers everywhere uniformly (no spatial
/// contrast to begin with) has nothing to flicker between either. Weighted 60/40 toward
/// the spatial term: the spatial CV is measured from a much larger effective sample (5
/// sub-aperture rays x 18x18 grid, the same population every other metric here is
/// measured against) and is the term validated against all thirteen built-in materials
/// as non-saturating and broadly discriminating (see the module-level doc and
/// `no_built_in_material_saturates_scintillation_on_either_cut`); the temporal term is
/// measured far more coarsely (5 single-ray azimuth samples per cell, no sub-aperture
/// averaging) specifically to keep the added cost modest (see the task report for the
/// before/after measurement), so it is kept a substantial but minority contribution
/// rather than an equal partner. Note this weighting is NOT strong enough to make SRB
/// out-scintillate the emerald cut unconditionally: the spatial term's own much larger
/// dynamic range across cuts still dominates when the two cuts return very different
/// amounts of light (e.g. a low-brilliance emerald-cut reading with a few very bright,
/// very contrasty cells can out-score a brighter, more evenly-lit SRB reading). It IS
/// decisive specifically where the reference definition asks it to be: at roughly
/// comparable brilliance between the two cuts, the temporal term reliably tips SRB above
/// the emerald cut -- see
/// `srb_can_scintillate_more_than_emerald_cut_at_matched_brilliance` in
/// `tests/metrics_tests.rs` and the task report for the measured sweep and before/after
/// cost numbers.
#[must_use]
fn combine_scintillation_pct(spatial_pct: f32, temporal_pct: f32) -> f32 {
    SCINT_TEMPORAL_WEIGHT
        .mul_add(temporal_pct, SCINT_SPATIAL_WEIGHT * spatial_pct)
        .clamp(0.0, 100.0)
}

/// Degrees-of-angular-separation -> display-scale multiplier for `fire_index`.
///
/// The measured quantity is the mean angle (in degrees) between a ray's F-line and
/// C-line exit directions, accumulated over the rays that are actually visibly
/// returned -- i.e. exit upward through the crown AND pass the same
/// `is_illuminated && !is_head_shadow` test used for `brilliance_pct` (see the main loop
/// below; rays that exit but are head-shadowed or unlit are physically invisible and are
/// intensity-weighted out of Fire, not just counted as "surviving"). Swept across every
/// built-in material, both `StandardGemCuts` cuts, and several camera pitches (see
/// `tests/metrics_tests.rs`), that mean separation typically lands somewhere in the
/// roughly 0.4-10 deg range depending on material dispersion, cut, and viewing/lighting
/// angle -- physically real (multiple TIR bounces each add their own chromatic
/// walk-off, so it is not surprising this is larger than a single-interface dispersion
/// angle), but not yet a legible UI number next to percentages like `brilliance_pct`.
///
/// This constant rescales that angle into a display range comparable to the old
/// closed-form `fire_index` values for ordinary gems (which topped out around 80 for
/// diamond), purely so the number stays legible in the existing UI widgets. It is a
/// *display* scale, not a fit: it was not tuned to reproduce old per-material numbers.
///
/// Re-derived twice so far. First (50 -> 175) after the exit-radiance-cosine weighting
/// was added to the Fire accumulation below (see the `radiance_weight` comment in the
/// main loop): that weighting multiplies in `exit_cos_f * exit_cos_c` on top of the
/// existing Fresnel transmittance product, suppressing every ray's contribution somewhat.
///
/// Second (175 -> 275) after the F/C bifurcation gate was added (see the gate's own
/// comment in the main loop below): requiring the F-line and C-line traces to exit
/// through the same facet after the same number of bounces discards the pairs whose raw
/// angle was dominated by disjoint, physically meaningless exit-direction divergence --
/// which, empirically, were also the pairs contributing the largest raw angles, so
/// discarding them shrinks the weighted sum more than it shrinks the ray count.
/// Recalibration method (same as the first pass): Diamond on the standard round
/// brilliant cut at the camera/light pose used throughout `tests/metrics_tests.rs`/
/// `tests/raytracer_tests.rs` (yaw 0.0, pitch 0.45, light yaw/pitch 0.85/0.95) measured a
/// raw weighted-mean separation of ~0.0716 deg post-gate (down from ~0.113 deg
/// pre-gate -- roughly a 37% drop, not quite the "halves" ballpark a cruder estimate
/// suggested, since the discarded pairs were disproportionately the largest-angle ones);
/// 275 was picked as the round multiplier landing that reference reading back at
/// `fire_index` ~= 19.7, matching the "tens" bucket the previous calibration pass
/// targeted. Sanity-checked afterward against the same pose's highest-dispersion
/// materials (Moissanite, Cubic Zirconia) and against
/// `fire_and_scintillation_are_finite_and_in_range_for_every_material_on_both_cuts`
/// (pitch 1.4): every built-in material on both cuts stays comfortably under 100 at this
/// scale, well inside that test's existing <1000 ceiling.
///
/// This scale, together with the bifurcation gate above it, DOES fix the emerald-cut
/// Fire ordering that the previous exit-radiance-cosine-only pass could not: see the
/// top-level comment on `evaluate_gem_optical_metrics` for the corrected measured tables
/// (Diamond now measures above Sapphire, Topaz, and Quartz on the emerald cut, as
/// dispersion predicts).
const FIRE_DEGREES_TO_DISPLAY_SCALE: f32 = 275.0;

/// Per-evaluation context threaded through [`classify_aperture_sample`]: everything
/// needed to fire and classify one grid-cell aperture-sample ray at the fixed camera
/// pose, bundled into one struct purely to keep that function's argument count within
/// clippy's `too_many_arguments` lint -- these ten values are otherwise identical
/// across all `grid_size * grid_size * aperture_samples.len()` calls per
/// `evaluate_gem_optical_metrics` invocation. Mirrors [`TemporalPoseContext`]'s reason
/// for existing, one level up (this context is built once per invocation at the fixed
/// pose; `TemporalPoseContext` is what the temporal sub-poses rebuild the view basis
/// from at each small yaw offset).
struct ApertureSampleContext<'a> {
    plane_soa: &'a crate::simd::PlanesSoA32,
    nd: f32,
    n_f: f32,
    n_c: f32,
    cam_forward: Vec3,
    cam_right: Vec3,
    cam_up: Vec3,
    key_dir: Vec3,
    fill_dir: Vec3,
    sin_lp: f32,
}

/// How one grid-cell aperture-sample ray was classified, mirroring the buckets the
/// pre-extraction main loop counted inline. `Returned` carries the qualifying Fire
/// sample's raw `(angle_deg, weight)` pair, if any -- deliberately NOT folded into the
/// Fire accumulator here, so the caller can apply the identical `f32::mul_add` chain
/// against its own running total, in the same grid-cell/aperture-sample iteration
/// order the pre-extraction code used. Folding it in here instead (e.g. via a `&mut
/// f32` accumulator parameter, or returning a partial sum to be added with `+`) would
/// change which intermediate values a fused multiply-add rounds against, silently
/// perturbing the bit-exact result -- see the module's task-level bit-exactness
/// requirement.
enum RayClassification {
    /// Matches the pre-refactor code's silent `continue`: not classified into any
    /// bucket. See `RayFate::EntryBlocked` doc.
    EntryBlocked,
    /// Leaked out through the pavilion bottom.
    Windowed,
    /// Trapped internally, absorbed, or exited but not visibly returned to the
    /// observer (head-shadowed or unlit).
    Extinct,
    /// Exited upward and was visibly returned to the observer. Carries the Fire
    /// `(angle_deg, weight)` pair if the F-line/C-line companion traces also both
    /// exited upward through the same facet after the same number of bounces (the F/C
    /// bifurcation gate -- see the comment on the gate itself, below).
    Returned(Option<(f32, f32)>),
}

/// Fires a single grid-cell aperture-sample ray at screen-space coordinates `(u, v)`
/// (in `[-1, 1]`) with sub-aperture jitter `(dx_sub, dz_sub)`, refracts and traces it
/// through the stone at the d-line (and, if it qualifies, replays the same entry
/// point/direction at the F-line and C-line indices for the Fire measurement), and
/// classifies the result. Returns `None` if the ray misses the stone geometry entirely
/// (not counted in any bucket, matching the pre-extraction code's `continue` on a
/// miss).
///
/// This is a direct extraction of the pre-extraction main loop's per-sample body: it
/// performs exactly the same sequence of floating-point operations in exactly the same
/// order, just packaged as a function so `evaluate_gem_optical_metrics` itself does not
/// have to spell out the F/C bifurcation gate and Fire weighting inline. See
/// [`RayClassification`]'s doc for why the Fire accumulator itself is threaded through
/// the caller rather than updated in here.
fn classify_aperture_sample(
    ctx: &ApertureSampleContext,
    u: f32,
    v: f32,
    dx_sub: f32,
    dz_sub: f32,
) -> Option<RayClassification> {
    let ray_dir = (ctx.cam_forward + dx_sub * ctx.cam_right + dz_sub * ctx.cam_up).normalize();
    let ray_origin = -ctx.cam_forward * 2.5 + (u * 0.95) * ctx.cam_right + (v * 0.95) * ctx.cam_up;
    let ray = Ray {
        origin: ray_origin,
        dir: ray_dir,
    };

    let hit = intersect_polyhedron_soa(ray, ctx.plane_soa);
    let hit_rec = hit?;

    let hit_point_entry = ray.origin + hit_rec.t * ray.dir;
    let n_entry = hit_rec.normal;

    // Refract from air (1.0) into gemstone (nd) using Snell's Law
    let cos_i = (-ray.dir).dot(n_entry).clamp(0.0, 1.0);

    match trace_wavelength(
        hit_point_entry,
        ray.dir,
        n_entry,
        cos_i,
        ctx.plane_soa,
        ctx.nd,
    ) {
        RayFate::EntryBlocked => Some(RayClassification::EntryBlocked),
        RayFate::Leaked => Some(RayClassification::Windowed),
        // The d-line's own exit cosine (`_exit_cos_d`) is intentionally not
        // used as a Fire weight here: it was tried (multiplied into
        // `radiance_weight` below alongside exit_cos_f/exit_cos_c) and made no
        // material difference to the emerald-cut ordering problem this fix
        // targets, while adding a factor not directly tied to the diagnosed
        // mechanism (the F/C exit-angle divergence, not the d-line ray's own
        // exit angle). See the Fire weighting comment below for what IS used.
        RayFate::ExitedUpward(exit_d) => {
            let exit_dir = exit_d.dir;
            let transmittance = exit_d.transmittance;
            // Directional Extinction Analysis: head-shadow cone vs. Key/Fill/
            // Overhead-ring illumination collection (see
            // `ray_is_visibly_returned` above for the shared definition, also
            // reused by the Scintillation temporal sub-poses below).
            if !ray_is_visibly_returned(
                exit_dir,
                ctx.cam_forward,
                ctx.key_dir,
                ctx.fill_dir,
                ctx.sin_lp,
            ) {
                return Some(RayClassification::Extinct);
            }

            // Fire: replay the SAME entry point/direction at the hydrogen
            // F-line and C-line indices, but ONLY for rays that pass this
            // exact same illumination test used for brilliance (not
            // head-shadowed, and actually collected by the key/fill/ring
            // lighting) -- gating alone, though, is not enough: an
            // unweighted MEAN over the surviving qualifying rays still lets
            // a badly-leaking cut win, because a small number of
            // near-critical-angle survivors (where a given index
            // difference produces an outsized angular deviation) shrinks
            // the denominator as fast as the numerator grows, inflating the
            // average regardless of how little light the stone actually
            // returns overall. Fire is what an observer *sees*, so each
            // ray's angular contribution is weighted here by `transmittance`
            // -- this exact ray's own Fresnel entry*exit transmittance
            // (energy actually delivered, not mere survival) -- and the
            // weighted sum is normalized by TOTAL incident rays rather than
            // by how many rays happened to qualify (see `fire_index` below).
            // A stone with only a handful of high-transmittance, wide-angle
            // survivors out of a large incident population still ends up
            // with a small contribution relative to that population, exactly
            // as it should.
            //
            // Only rays whose d-line trace exits upward AND passes
            // illumination are considered, and only if BOTH the F-line and
            // C-line companion traces also exit upward -- a ray whose F or C
            // image is lost to TIR or pavilion leakage (a real chromatic
            // effect: the critical angle itself depends on wavelength)
            // simply contributes no Fire sample, rather than a fabricated one.
            let fate_f = trace_wavelength(
                hit_point_entry,
                ray.dir,
                n_entry,
                cos_i,
                ctx.plane_soa,
                ctx.n_f,
            );
            let fate_c = trace_wavelength(
                hit_point_entry,
                ray.dir,
                n_entry,
                cos_i,
                ctx.plane_soa,
                ctx.n_c,
            );
            let Some(fire) = (match (fate_f, fate_c) {
                (RayFate::ExitedUpward(exit_f), RayFate::ExitedUpward(exit_c)) => {
                    // F/C bifurcation gate: when the F-line and C-line traces
                    // exit through a DIFFERENT facet, or after a different
                    // number of internal bounces, they took physically
                    // disjoint paths through the stone -- their critical
                    // angles straddled a TIR threshold at different points, so
                    // `acos(dir_f . dir_c)` below would be the angle between
                    // two physically unrelated exit directions, not a
                    // dispersion measurement. Measured: these bifurcated pairs
                    // carried 45-98% of the weighted Fire sum in every case
                    // where step cuts wrongly out-scored brilliants, with
                    // per-ray contributions up to ~27 fire-units from angles as
                    // large as 33 deg -- an artifact of sampling, not a real
                    // optical effect. A pair whose F or C trace took a
                    // different path contributes nothing, mirroring the
                    // existing rule just below that a pair contributes nothing
                    // if F or C fails to exit at all.
                    //
                    // A capped-credit variant was also measured (crediting a
                    // bifurcated pair at a small fixed separation instead of
                    // discarding it, on the theory that "different facets
                    // flash different colours" is itself a fire-positive
                    // event). It was rejected: on the corrected emerald-cut
                    // geometry at pitch 1.4, capped-credit put Quartz (raw
                    // dispersion n_F-n_C ~ 0.0078) ABOVE Sapphire (~0.0105) --
                    // a materially larger, unambiguous gap that strict-discard
                    // gets right (Sapphire 5.58 > Quartz 5.44). Strict-discard
                    // also reproduced the SRB reference readings almost
                    // exactly (Diamond 20.05 vs. an independently measured
                    // 20.1, Cubic Zirconia 30.08 vs. 30.1, Quartz 13.49 vs.
                    // 13.5), while capped-credit's numbers diverged further
                    // from that same reference. See the task report for the
                    // full measured tables.
                    if exit_f.facet_idx != exit_c.facet_idx || exit_f.bounces != exit_c.bounces {
                        // Contributes no Fire sample -- fall through without
                        // touching the accumulator, same as an F/C pair that
                        // failed to both exit upward.
                        None
                    } else {
                        let transmittance_f = exit_f.transmittance;
                        let transmittance_c = exit_c.transmittance;
                        let exit_cos_f = exit_f.exit_cos_theta;
                        let exit_cos_c = exit_c.exit_cos_theta;
                        let cos_sep = exit_f.dir.dot(exit_c.dir).clamp(-1.0, 1.0);
                        let angle_deg = cos_sep.acos().to_degrees();
                        // Weight by the PRODUCT of all three wavelengths' own
                        // transmittances, not just the d-line's. Right at a
                        // material's critical angle, the angular sensitivity to a
                        // small index change diverges (d(theta_exit)/dn ~
                        // 1/sqrt(epsilon), where epsilon is how far below
                        // critical the ray sits) at almost exactly the rate the
                        // d-line's own Fresnel exit transmittance vanishes (T ~
                        // sqrt(epsilon)) -- so angle * transmittance_d alone
                        // approaches a small but nonzero PLATEAU near critical
                        // rather than vanishing, which is exactly the residual
                        // artifact: a cut whose facet geometry puts many rays near
                        // ITS critical angle for a given (mismatched) material
                        // racks up many such near-constant contributions and still
                        // inflates Fire, even after entry/exit weighting on the
                        // d-line alone. Multiplying in transmittance_f and
                        // transmittance_c as well breaks that cancellation: near
                        // critical, T_f and T_c ALSO ~ sqrt(epsilon) each, so the
                        // product transmittance * transmittance_f * transmittance_c
                        // ~ epsilon^1.5, and angle * (that product) ~ Δn * epsilon
                        // -> 0. A ray only registers strongly if all three
                        // wavelengths are comfortably transmitted, not merely
                        // "not-yet-TIR" -- which matches what an observer actually
                        // perceives as a colorful flash (all three color channels
                        // visible with real intensity), not a single grazing sliver
                        // of light that happens to carry a wide chromatic spread.
                        //
                        // Transmittance alone is STILL not enough, though: acos(dir_f
                        // . dir_c) itself has a near-singularity at grazing exit --
                        // as an exit angle approaches 90 deg, d(theta_out)/dn
                        // diverges, so a tiny F/C index difference produces an
                        // enormous angular separation right as the ray is leaving
                        // almost tangent to the surface. Low-index stones (smaller
                        // critical angle margin against typical facet geometry, e.g.
                        // Quartz) push far more of their returning light out near
                        // that grazing regime than a high-index stone like Diamond,
                        // and Fresnel transmittance falls off too slowly to outrun
                        // the acos divergence -- so without a further correction a
                        // handful of near-grazing Quartz rays with huge (but not
                        // small-transmittance-suppressed) angular separations
                        // dominate the sum and make Quartz read as MORE fire than
                        // Diamond, backwards from reality.
                        //
                        // The missing physical factor is radiance, not energy: light
                        // leaving a surface at grazing incidence isn't just dimmer,
                        // it is smeared across a much larger solid angle before it
                        // can reach a fixed-aperture observer, so the radiance that
                        // actually arrives collapses (the standard projected-area /
                        // Lambert cosine factor for a radiating interface). Each
                        // wavelength's own exit ray undergoes this collapse
                        // independently at its own exit facet and its own exit
                        // angle, so -- mirroring exactly why transmittance_f and
                        // transmittance_c are multiplied in above rather than
                        // relying on the d-line's alone -- the F-line and C-line
                        // exit cosines (`exit_cos_f`, `exit_cos_c`; the cosine of
                        // each ray's exit angle measured from ITS OWN exit facet
                        // normal, returned by `trace_wavelength`) are multiplied in
                        // as well. Right at grazing, exit_cos -> 0 and directly
                        // cancels the acos divergence it is paired with, which
                        // transmittance alone could not do fast enough.
                        let energy_weight = transmittance * transmittance_f * transmittance_c;
                        let radiance_weight = exit_cos_f.max(0.0) * exit_cos_c.max(0.0);
                        let weight = energy_weight * radiance_weight;
                        Some((angle_deg, weight))
                    }
                }
                _ => None,
            }) else {
                return Some(RayClassification::Returned(None));
            };
            Some(RayClassification::Returned(Some(fire)))
        }
        RayFate::Absorbed => Some(RayClassification::Extinct),
    }
}

/// Aggregate accumulators threaded through `evaluate_gem_optical_metrics`'s main grid
/// loop. Bundled into one `#[derive(Default)]` struct (every field starts at zero)
/// purely to collapse what used to be a dozen separate `let mut ... = 0` declarations
/// -- each with its own explanatory comment -- into a single
/// `MetricsAccumulators::default()` call in the function itself, keeping it under
/// clippy's line-count budget without losing any of the original per-field rationale,
/// which now lives on the fields below instead.
#[derive(Default)]
struct MetricsAccumulators {
    total_rays: u32,
    windowed_rays: u32,
    extinct_rays: u32,
    returned_rays: u32,
    /// ENERGY-WEIGHTED sum of per-ray F-line/C-line exit angular separations (degrees),
    /// over rays that exit upward through the crown at the d-line, pass the same
    /// illumination test as brilliance, AND whose F-line and C-line companion traces
    /// also both exit upward (see the main loop). Each qualifying ray's
    /// angular-separation contribution is weighted by that ray's own Fresnel
    /// entry*exit transmittance (see `RayFate::ExitedUpward`), and the accumulated sum
    /// is normalized by TOTAL incident rays (`n_total`, computed after the loop)
    /// rather than by the count of qualifying rays -- so a stone that returns little
    /// light cannot score highly on the strength of a small, wide-angle survivor
    /// population. See the `fire_index` comment in `evaluate_gem_optical_metrics` for
    /// why this replaced a plain per-survivor mean.
    fire_energy_weighted_sum_deg: f32,
    /// Diagnostic-only accumulators for the `DIAG_FIRE_DEBUG` eprintln block at the end
    /// of `evaluate_gem_optical_metrics`, gated behind `diag_fire_debug` everywhere
    /// they're touched in the hot loop, so a normal run pays no cost beyond the extra
    /// additions.
    dbg_fire_qualifying: u32,
    dbg_fire_angle_sum_unweighted: f32,
    dbg_fire_transmittance_sum: f32,
    /// Scintillation accumulators: per-grid-cell fraction of aperture samples that
    /// returned illuminated brilliance, aggregated (via running sum / sum-of-squares)
    /// into a coefficient of variation across the 18x18 grid at the end.
    cell_fraction_sum: f32,
    cell_fraction_sum_sq: f32,
    cell_count: u32,
    /// Scintillation TEMPORAL accumulator: for each visited grid cell, the Bernoulli
    /// variance (p * (1 - p), max 0.24 at this sample count -- see
    /// `cell_returned_at_yaw_offset`) of that cell's return status across the small
    /// camera-yaw offsets in `SCINT_TEMPORAL_YAW_OFFSETS_DEG`, summed here and averaged
    /// (by `cell_count`, the exact same visited-cell count the spatial term above
    /// divides by, since both are only ever accumulated together) into
    /// `temporal_activity` after the grid loop. A cell that returns light identically
    /// at every offset (always on, or always off) contributes 0 -- it is bright or
    /// dark, but it does not sparkle. A cell that flips contributes up to 0.24.
    temporal_variance_sum: f32,
}

/// Prints the `DIAG_FIRE_DEBUG` Fire diagnostic line for one
/// `evaluate_gem_optical_metrics` invocation. Pure formatting over already-computed
/// values -- extracted purely to keep the `eprintln!`'s argument list out of the main
/// function's line count; touches no accumulator and performs no computation that
/// feeds back into any returned metric.
/// Bundles `MetricsAccumulators`' fields relevant to the Fire diagnostic line,
/// purely so [`log_fire_diagnostics`] takes `&MetricsAccumulators` directly instead of
/// unpacking each field into its own parameter (which pushed it over clippy's
/// `too_many_arguments` limit).
fn log_fire_diagnostics(
    material_name: &str,
    n_total: f32,
    fire_index: f32,
    acc: &MetricsAccumulators,
) {
    let mean_angle_unweighted = if acc.dbg_fire_qualifying > 0 {
        acc.dbg_fire_angle_sum_unweighted / acc.dbg_fire_qualifying as f32
    } else {
        0.0
    };
    let mean_transmittance = if acc.dbg_fire_qualifying > 0 {
        acc.dbg_fire_transmittance_sum / acc.dbg_fire_qualifying as f32
    } else {
        0.0
    };
    eprintln!(
        "DIAG material={material_name} n_total={n_total} returned_rays={} fire_qualifying={} mean_angle_unweighted={mean_angle_unweighted:.4} mean_transmittance={mean_transmittance:.4} weighted_sum={:.4} fire_index={fire_index:.4}",
        acc.returned_rays, acc.dbg_fire_qualifying, acc.fire_energy_weighted_sum_deg
    );
}

/// Prints the `DIAG_FIRE_DEBUG` Scintillation diagnostic line. Same rationale as
/// [`log_fire_diagnostics`].
fn log_scintillation_diagnostics(
    material_name: &str,
    spatial_scint_pct: f32,
    temporal_pct: f32,
    scintillation_pct: f32,
) {
    eprintln!(
        "DIAG-SCINT material={material_name} spatial={spatial_scint_pct:.4} temporal={temporal_pct:.4} combined={scintillation_pct:.4}"
    );
}

/// Everything the main grid loop in [`evaluate_gem_optical_metrics`] needs that is
/// fixed across the whole evaluation: the sampling grid resolution, the sub-aperture
/// jitter bundle, and the two shared per-ray contexts ([`TemporalPoseContext`],
/// [`ApertureSampleContext`]).
struct GridEvalSetup<'a> {
    grid_size: i32,
    aperture_samples: [(f32, f32); 5],
    temporal_ctx: TemporalPoseContext<'a>,
    aperture_ctx: ApertureSampleContext<'a>,
}

/// Builds [`GridEvalSetup`]. A pure setup extraction: every value here is computed
/// exactly once, unconditionally, with no accumulator or loop state involved, so
/// moving it into its own function changes nothing about the floating-point operations
/// `evaluate_gem_optical_metrics` performs -- only where they are textually written.
fn build_grid_eval_setup<'a>(
    plane_soa: &'a crate::simd::PlanesSoA32,
    material: &GemMaterial,
    cam_yaw: f32,
    cam_pitch: f32,
    light_yaw: f32,
    light_pitch: f32,
) -> GridEvalSetup<'a> {
    let nd = material.dispersion.evaluate(589.3).max(1.1);
    // Clamped the same way as `nd` above (defensively -- real materials never evaluate
    // below 1.0) so `trace_wavelength`'s entry refraction never hits the pathological
    // `EntryBlocked` case for these two indices either.
    let n_f = material.dispersion.evaluate(486.1).max(1.001);
    let n_c = material.dispersion.evaluate(656.3).max(1.001);

    let grid_size = 18;

    // Camera View Direction from spherical coordinates (Observer PoV), matching the
    // real render camera's frame exactly (see `camera_view_basis`).
    let (cam_forward, cam_right, cam_up) = camera_view_basis(cam_yaw, cam_pitch);

    // Key/Fill Light Direction, from the shared `StudioRig` (see its module doc under
    // `optics::studio_rig`) -- the SAME construction `sample_studio_environment` uses
    // to light the image these metrics are meant to describe, so the two can never
    // silently drift apart. This function does not consult `rig.ring_dirs`: its own
    // ring/annulus test below is a deliberately coarser approximation (see
    // `ray_is_visibly_returned`) that only needs `sin_light_pitch`.
    let rig = crate::optics::studio_rig::StudioRig::new(light_yaw, light_pitch);
    let key_dir = rig.key_dir;
    let fill_dir = rig.fill_dir;
    let sin_lp = rig.sin_light_pitch;

    // 5-point angular sub-aperture bundle (standard GIA 0° to 6° observer eye cone)
    let aperture_samples = [
        (0.0f32, 0.0f32),
        (0.08, 0.0),
        (-0.08, 0.0),
        (0.0, 0.08),
        (0.0, -0.08),
    ];

    // Shared context for the Scintillation temporal sub-poses (see
    // `TemporalPoseContext`'s doc for why this is bundled rather than passed field by
    // field): identical across every grid cell and every offset sample.
    let temporal_ctx = TemporalPoseContext {
        plane_soa,
        nd,
        cam_yaw,
        cam_pitch,
        key_dir,
        fill_dir,
        sin_lp,
    };

    // Shared context for the per-aperture-sample classification (see
    // `ApertureSampleContext`'s doc for why this is bundled rather than passed field by
    // field): identical across every grid cell and every aperture sample.
    let aperture_ctx = ApertureSampleContext {
        plane_soa,
        nd,
        n_f,
        n_c,
        cam_forward,
        cam_right,
        cam_up,
        key_dir,
        fill_dir,
        sin_lp,
    };

    GridEvalSetup {
        grid_size,
        aperture_samples,
        temporal_ctx,
        aperture_ctx,
    }
}

/// Evaluates true GIA / AGSL optical gemological metrics by firing an analytical grid
/// of rays with viewing aperture cone from the observer's **Point of View (`PoV`)**.
///
/// Rays are fired from (`cam_yaw`, `cam_pitch`) through the 3D cutting schedule facet
/// geometry, dynamically accounting for:
/// 1. Gemstone refractive index n(λ) from Sellmeier / Cauchy equations
/// 2. Snell's law refraction at inclined crown & girdle facet entry points
/// 3. Total Internal Reflection (TIR) vs bottom leakage (Windowing) on pavilion facets
/// 4. Light source illumination alignment (`light_yaw`, `light_pitch`) vs head-shadow extinction
/// 5. Fire: the angular separation between the F-line and C-line images of each ray that
///    is actually visibly returned (same illumination test as brilliance), so a
///    high-leakage cut with a few stray near-critical-angle rays cannot outscore a
///    well-performing one on angle alone
/// 6. Scintillation: the spatial contrast (coefficient of variation) of light return
///    across the 18x18 sampling grid over the stone's face
///
/// ## Resolved: Fire ordering on step (emerald) cuts, and the F/C bifurcation artifact
///
/// It was long believed this metric systematically scored step cuts above brilliants.
/// That premise turned out to be wrong: at the canonical test poses, the round brilliant
/// correctly out-fires the emerald cut for Diamond/Quartz/Sapphire/Topaz (e.g. Diamond at
/// pitch 1.4: SRB ~20.0 vs. emerald ~9.2). Two real, separable defects remained:
///
/// 1. Material ordering could invert on step cuts (a low-dispersion material reading
///    higher Fire than Diamond on the emerald cut specifically).
/// 2. Isolated pose flips where a step cut beat the brilliant.
///
/// Root cause of both, isolated by measurement: Fire is computed by tracing the
/// Fraunhofer F and C lines and measuring the angular separation between their exit
/// directions (see point 5 above, and the `radiance_weight` comment in the main loop
/// below). When the F and C traces exit via *different facets, or after a different
/// number of internal bounces* -- their critical angles straddled a TIR threshold at
/// different points inside the stone -- the two lines have taken physically disjoint
/// paths, so `acos(dir_f . dir_c)` measures the angle between two unrelated exit
/// directions rather than true chromatic dispersion. These "bifurcated" pairs measured
/// up to 33.5 deg mean separation and up to ~27 fire-units from a single ray, and carried
/// 45-98% of the weighted Fire sum in every case where a step cut wrongly out-scored a
/// brilliant. Two earlier fix attempts (radiance-cosine weighting alone, and a saturating
/// perceptual-resolvability map on the raw angle) narrowed the distortion but could not
/// eliminate it, because neither targeted this mechanism specifically: both continued to
/// treat bifurcated pairs as if they were legitimate same-path dispersion samples, just
/// suppressed by a shrinking multiplier or a saturating cap. See prior revisions of this
/// comment (in version control) for those attempts' measured tables.
///
/// The fix, implemented in the F/C qualification gate in the main loop below: a Fire
/// pair contributes only when both `RayFate::ExitedUpward` traces report the SAME exit
/// facet index AND the same bounce count (`ExitPath::facet_idx` / `ExitPath::bounces`).
/// This mirrors the existing rule that a pair contributes nothing if F or C fails to
/// exit upward at all -- a bifurcated pair is, physically, exactly that same situation:
/// no shared exit path exists to measure a dispersion angle across. A capped-credit
/// alternative (crediting a bifurcated pair at a small fixed angle instead of discarding
/// it, since "different facets flashing different colours" is itself a fire-positive
/// perceptual event) was measured and rejected: it put Quartz above Sapphire on the
/// emerald cut at pitch 1.4, when Sapphire's true dispersion is clearly higher and
/// strict discard gets that pair right. See the F/C gate's own comment for the full
/// measured comparison.
///
/// This fix was verified against corrected step-cut geometry: `emerald_cut()` was
/// separately found to be over-constrained (11 of its 34 declared planes contributed no
/// facet to the actual solid, leaving a 23-facet solid missing most of its step
/// structure) and has since been re-derived so all 34 planes contribute
/// (`hull.untouched_planes()` is empty; see `tests/optics_geometry_tests.rs`). All Fire
/// numbers in this comment and in `tests/metrics_tests.rs` were re-measured against that
/// corrected 34-facet geometry together with the F/C bifurcation gate above.
///
/// ## Investigated and closed: Quartz vs. Topaz ordering on the emerald cut is noise, not a defect
///
/// At the single canonical pose (emerald cut, yaw=0.0, pitch=1.4, light 0.85/0.95),
/// Quartz measures a higher `fire_index` than Topaz, even though Topaz has the larger
/// F-C dispersion (`n_F - n_C`: Topaz ~0.00816 vs. Quartz ~0.00781, a Topaz/Quartz ratio
/// of only ~1.045). This was investigated empirically (grid-size and pose sweeps via a
/// throwaway example, deleted afterward) rather than "fixed" on suspicion, per this
/// module's history of failed physically-motivated Fire tuning attempts. Findings:
///
/// - Sweeping camera pitch 0.15-1.50 at the canonical yaw/light (28 poses, default
///   `grid_size` = 18): Topaz reads higher at 12 poses, Quartz at 16, with the gap swinging
///   from -2.48 to +5.45 fire-units (mean +0.21, stddev 1.71) -- a swing an order of
///   magnitude larger than the ~4.5% dispersion difference could produce on its own, and
///   with no consistent sign.
/// - Re-run with `grid_size` quadrupled to 72 (5184 rays vs. 324 per pose, before the
///   5-point aperture bundle): the same sweep's mean gap collapsed to -0.0004 (stddev
///   0.735) and the canonical pose itself flipped to a ~0.05-unit Topaz lead -- consistent
///   with a real signal this small being dominated by discrete-grid quantization noise at
///   the production resolution, not a systematic material-ordering bug.
/// - `DIAG_FIRE_DEBUG` at the canonical pose shows *why* the two materials are close
///   even before quantization: Quartz's raw `mean_angle_unweighted` (0.291 deg) is
///   actually slightly ABOVE Topaz's (0.286 deg) despite Topaz's higher dispersion,
///   because per-ray angular spread from refraction is not purely proportional to
///   `n_F - n_C` -- it also depends on the base index `n_d` (Topaz `n_d` ~1.627 vs.
///   Quartz ~1.544). That same higher `n_d` gives Topaz systematically lower Fresnel
///   transmittance (`mean_transmittance` 0.451 vs. Quartz's 0.509 at this pose), and Fire
///   is transmittance-weighted by design (see point 5 above) -- so Topaz's dispersion
///   edge is legitimately offset by its own index raising Fresnel losses, not just
///   swamped by noise. Both effects are real physics; neither is a defect, and together
///   they explain why "higher dispersion always wins" is not a safe assumption for two
///   materials this close (~4.5%) in `n_F - n_C`.
///
/// Conclusion: not fixed, because there is nothing here to fix -- `fire_index` is not
/// simply a resampling of `n_F - n_C`, and a materially larger gap for two materials
/// this close in true dispersion would need many more rays per pose than this metric's
/// production resolution provides to resolve reliably. If this needs to be revisited,
/// raise `grid_size` in `build_grid_eval_setup` (verified above to shrink the noise) --
/// don't retune the F/C gate or `FIRE_DEGREES_TO_DISPLAY_SCALE` against this pair, since
/// neither one is the actual source of the imprecision.
#[must_use]
pub fn evaluate_gem_optical_metrics(
    planes: &[GpuFacetPlane],
    material: &GemMaterial,
    cam_yaw: f32,
    cam_pitch: f32,
    light_yaw: f32,
    light_pitch: f32,
) -> GemOpticalMetrics {
    if planes.is_empty() {
        // No facet geometry to trace at all: there is nothing to measure Fire or
        // Scintillation against, so fall back to neutral placeholder values rather than
        // a formula. These are display defaults for the "no geometry loaded" state, not
        // measurements.
        return GemOpticalMetrics {
            brilliance_pct: 85.0,
            fire_index: 25.0,
            scintillation_pct: 75.0,
            windowing_pct: 5.0,
            extinction_pct: 5.0,
        };
    }

    let mut acc = MetricsAccumulators::default();
    // Checked once here (not once per ray) and gated behind `diag_fire_debug` everywhere
    // it's touched in the hot loop below, so a normal run pays no cost for the env
    // lookup or the extra additions. See `MetricsAccumulators` for what each field
    // accumulates.
    let diag_fire_debug = std::env::var("DIAG_FIRE_DEBUG").is_ok();

    // See `build_grid_eval_setup`'s doc: a pure setup extraction, computed once,
    // unconditionally, with no accumulator involved.
    // SIMD slab arena, built once per evaluation: every ray this function
    // fires (grid, sub-aperture, temporal sub-poses, F/C lines) intersects the
    // same solid -- see `optics::raytracer::intersect_polyhedron_soa`, whose
    // results are bit-identical to the scalar intersection this replaced.
    let plane_soa = build_plane_soa(planes);
    let setup = build_grid_eval_setup(
        &plane_soa,
        material,
        cam_yaw,
        cam_pitch,
        light_yaw,
        light_pitch,
    );

    for ix in 0..setup.grid_size {
        for iz in 0..setup.grid_size {
            let u = ((ix as f32 + 0.5) / (setup.grid_size as f32)).mul_add(2.0, -1.0);
            let v = ((iz as f32 + 0.5) / (setup.grid_size as f32)).mul_add(2.0, -1.0);
            if v.mul_add(v, u * u) > 0.70 {
                continue; // Stay within gem perimeter
            }

            // Per-cell counters for the Scintillation spatial-contrast measurement below:
            // how many of this grid cell's aperture samples actually hit the stone, and
            // how many of those came back as illuminated brilliance.
            let mut cell_total = 0u32;
            let mut cell_returned = 0u32;

            for &(dx_sub, dz_sub) in &setup.aperture_samples {
                // See `classify_aperture_sample`'s doc: this performs exactly the same
                // sequence of floating-point operations, in the same order, as the
                // pre-extraction inline loop body did -- the Fire accumulator update
                // below is applied here (not inside the helper) specifically to
                // preserve the exact `f32::mul_add` chain across grid cells.
                let Some(classification) =
                    classify_aperture_sample(&setup.aperture_ctx, u, v, dx_sub, dz_sub)
                else {
                    continue;
                };

                acc.total_rays += 1;
                cell_total += 1;

                match classification {
                    RayClassification::EntryBlocked => {}
                    RayClassification::Windowed => acc.windowed_rays += 1,
                    RayClassification::Extinct => acc.extinct_rays += 1,
                    RayClassification::Returned(fire) => {
                        acc.returned_rays += 1;
                        cell_returned += 1;

                        if let Some((angle_deg, weight)) = fire {
                            acc.fire_energy_weighted_sum_deg =
                                f32::mul_add(angle_deg, weight, acc.fire_energy_weighted_sum_deg);
                            if diag_fire_debug {
                                acc.dbg_fire_qualifying += 1;
                                acc.dbg_fire_angle_sum_unweighted += angle_deg;
                                acc.dbg_fire_transmittance_sum += weight;
                            }
                        }
                    }
                }
            }

            if cell_total > 0 {
                let frac = cell_returned as f32 / cell_total as f32;
                acc.cell_fraction_sum += frac;
                acc.cell_fraction_sum_sq = frac.mul_add(frac, acc.cell_fraction_sum_sq);
                acc.cell_count += 1;

                // Scintillation TEMPORAL component for this cell -- see
                // `cell_temporal_variance`'s doc.
                acc.temporal_variance_sum += cell_temporal_variance(&setup.temporal_ctx, u, v);
            }
        }
    }

    let n_total = acc.total_rays.max(1) as f32;
    let windowing_pct = (acc.windowed_rays as f32 / n_total * 100.0).clamp(0.0, 100.0);
    let extinction_pct = (acc.extinct_rays as f32 / n_total * 100.0).clamp(0.0, 100.0);
    let brilliance_pct = (acc.returned_rays as f32 / n_total * 100.0).clamp(0.0, 100.0);

    // Fire: energy-weighted F-line/C-line angular separation, normalized by TOTAL
    // incident rays (n_total) -- NOT by the count of rays that happened to qualify. This
    // is the same normalization convention as brilliance_pct/windowing_pct/
    // extinction_pct above (all divided by n_total), and it is what closes the
    // structural loophole a plain per-survivor mean has: dividing by a population that
    // shrinks in lockstep with the numerator lets a handful of wide-angle survivors from
    // a badly-leaking cut inflate the average arbitrarily. See
    // `MetricsAccumulators::fire_energy_weighted_sum_deg`'s doc for the rest of the
    // rationale. Naturally floors at 0.1 (matching the old formula's floor) when the
    // weighted sum is zero -- no separate branch needed.
    let fire_index =
        (acc.fire_energy_weighted_sum_deg / n_total * FIRE_DEGREES_TO_DISPLAY_SCALE).max(0.1);
    if diag_fire_debug {
        log_fire_diagnostics(&material.name, n_total, fire_index, &acc);
    }

    // Scintillation: spatial term (`spatial_scintillation_pct`), temporal term
    // (`temporal_scintillation_pct`), combined per `combine_scintillation_pct` -- see
    // each helper's doc for the full rationale.
    let spatial_scint_pct = spatial_scintillation_pct(
        acc.cell_fraction_sum,
        acc.cell_fraction_sum_sq,
        acc.cell_count,
    );
    let temporal_pct = temporal_scintillation_pct(acc.temporal_variance_sum, acc.cell_count);
    let scintillation_pct = combine_scintillation_pct(spatial_scint_pct, temporal_pct);
    if diag_fire_debug {
        log_scintillation_diagnostics(
            &material.name,
            spatial_scint_pct,
            temporal_pct,
            scintillation_pct,
        );
    }

    GemOpticalMetrics {
        brilliance_pct,
        fire_index,
        scintillation_pct,
        windowing_pct,
        extinction_pct,
    }
}

/// Camera azimuths the Tilt Performance dialog sweeps a full tilt-elevation profile at,
/// in degrees -- see [`evaluate_angular_profile_at_azimuth`].
///
/// `0.0` looks straight down whatever direction `RenderContext::yaw == 0.0` frames (for
/// a round outline this is an arbitrary reference; for an elongated one -- marquise,
/// emerald cut, pear -- it is a real, physically distinct direction, conventionally the
/// table's long axis). Each further entry rotates the viewpoint another 45° around the
/// table-normal axis, so `90.0` looks down the perpendicular ("width") axis and `45.0`/
/// `135.0` bisect the two. Tilting toward a non-round stone's long axis and tilting
/// toward its short axis window very differently, which is exactly the information a
/// single azimuth-0 sweep hides.
pub const PROFILE_AZIMUTHS_DEG: [f32; 4] = [0.0, 45.0, 90.0, 135.0];

/// Calculates a 19-point angular profile of (Brilliance %, Extinction %, Windowing %)
/// at an explicit camera azimuth.
///
/// Sampled over `PoV` tilt elevation angles in exact 5° steps: [0°, 5°, 10°, 15°, 20°,
/// 25°, 30°, 35°, 40°, 45°, 50°, 55°, 60°, 65°, 70°, 75°, 80°, 85°, 90°]. `cam_yaw` is
/// in radians (matching `evaluate_gem_optical_metrics`'s own convention) -- see
/// [`PROFILE_AZIMUTHS_DEG`] for the four-azimuth convention the Tilt Performance
/// dialog's axis switcher uses. [`evaluate_angular_profile`] is just this function
/// called at `cam_yaw: 0.0`, so both share one code path.
#[must_use]
pub fn evaluate_angular_profile_at_azimuth(
    planes: &[GpuFacetPlane],
    material: &GemMaterial,
    cam_yaw: f32,
    light_yaw: f32,
    light_pitch: f32,
) -> ([f32; 19], [f32; 19], [f32; 19]) {
    let mut brilliance_curve = [0.0f32; 19];
    let mut extinction_curve = [0.0f32; 19];
    let mut windowing_curve = [0.0f32; 19];

    for (i, &deg) in PROFILE_ANGLES_DEG.iter().enumerate() {
        let cam_pitch_rad = deg.to_radians();
        let m = evaluate_gem_optical_metrics(
            planes,
            material,
            cam_yaw,
            cam_pitch_rad,
            light_yaw,
            light_pitch,
        );
        brilliance_curve[i] = m.brilliance_pct;
        extinction_curve[i] = m.extinction_pct;
        windowing_curve[i] = m.windowing_pct;
    }

    (brilliance_curve, extinction_curve, windowing_curve)
}

/// Calculates a 19-point angular profile of (Brilliance %, Extinction %, Windowing %)
/// at the canonical (0°) camera azimuth.
///
/// See [`evaluate_angular_profile_at_azimuth`] for the general form this delegates to,
/// and [`PROFILE_AZIMUTHS_DEG`] for what the other three azimuths mean. Sampled over
/// `PoV` tilt elevation angles in exact 5° steps: [0°, 5°, 10°, 15°, 20°, 25°, 30°,
/// 35°, 40°, 45°, 50°, 55°, 60°, 65°, 70°, 75°, 80°, 85°, 90°].
#[must_use]
pub fn evaluate_angular_profile(
    planes: &[GpuFacetPlane],
    material: &GemMaterial,
    light_yaw: f32,
    light_pitch: f32,
) -> ([f32; 19], [f32; 19], [f32; 19]) {
    evaluate_angular_profile_at_azimuth(planes, material, 0.0, light_yaw, light_pitch)
}

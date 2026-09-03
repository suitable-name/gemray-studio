//! Meet-point solver.
//!
//! Derives `GemCAD` "mast" (facet plane offset) distances from angles, index
//! positions, gear, and meet constraints alone -- the inverse of the forward
//! `from_asc_schedule` plane construction in [`super::cuts`].
//!
//! # The meet model: vertex incidence, not tangency
//!
//! Every facet plane is `n . x = mast` with `n` fixed by angle, index and gear, and
//! `mast` the unknown.
//!
//! An earlier revision of this module modeled "meet" as *tangency*: cut the plane
//! inward until it first touches the already-cut solid (`mast = max_v n . v`, the
//! support function). Corpus measurement showed that model to be wrong: over ~39,000
//! meet-derived tiers across 2,881 real designs, the recorded mast is essentially
//! *never* the support-function maximum (0.7% of tiers). The format's actual
//! semantics is **vertex incidence**: the facet is cut *past* first touch until its
//! plane passes exactly through a designated vertex of the arrangement formed by the
//! other facets' planes -- a real meet point of the design. Measured with every
//! other tier pinned at its true mast, the recorded mast is realized by such a
//! vertex (within 0.5% relative) for 96.2% of tiers; and among the arrangement's
//! candidate vertex "levels" ordered by `n . v` descending, the true one is level 1
//! (the first level past first touch) for 77.5% of tiers, and within the first
//! three levels for 94.4%.
//!
//! # What meets alone cannot determine: per-block anchors
//!
//! A design's plane arrangement has genuine continuous degrees of freedom that
//! preserve *every* vertex incidence. Girdle planes are vertical (`n.y = 0`), so
//! translating the whole crown block up or down (each crown mast shifted by
//! `cos(theta) * delta`) slides every crown-girdle meet along the girdle wall and
//! moves every crown-internal meet coherently -- nothing detects the shift. The
//! same holds for the pavilion block, and for the girdle's own radial scale.
//! (Verified empirically: a coherently shifted crown is a stable fixed point of
//! this solver's refinement stage, with zero restoring force.)
//!
//! Those free parameters are exactly the dimensions a designer chooses and a
//! printed diagram states outright (stone size, girdle thickness, crown/pavilion
//! height -- the `C/W`, `P/W`, `H/W` numbers on every `GemCAD` printout). So a
//! caller must supply one [`MeetConstraint::ScaleReference`] per block (crown,
//! pavilion, girdle) that the schedule itself doesn't anchor with a stated "Set
//! girdle thickness"-style instruction; masts within each block are then meet-
//! derived from that anchor. [`meet_tier_inputs_from_asc`] classifies stated
//! anchors; [`apply_ratio_anchors`] fills in whatever the schedule left unanchored
//! from a design's printed `C/W`/`P/W` proportions -- the only anchor source that
//! exists at all for a design with no `.asc` file. That estimate carries real
//! error, smaller than a real recorded mast's but still substantial (see
//! [`apply_ratio_anchors`]'s own doc comment for the measured end-to-end effect);
//! a caller with real recorded masts to bootstrap from should prefer those
//! instead, as the corpus validation probe's baseline report does.
//!
//! # Solving strategy
//!
//! Which vertex a tier's plane passes through depends on where every *other*
//! tier's plane sits, and real schedules are written with genuinely mutual
//! dependencies (crown tiers that only close against each other, say). The solve
//! therefore runs in three phases:
//!
//! 1. **Constructive pass in file order** ([`SolveStrategy::DependencyOrder`]):
//!    tiers settle one at a time in schedule order (which is overwhelmingly the
//!    real cutting order), each against the arrangement of everything settled so
//!    far -- the shallowest candidate vertex level incident to every resolved
//!    `"Meet <names>"` reference when the schedule states one, the rank-1 level
//!    (first vertex past first touch, the measured 77.5% prior) otherwise. A tier
//!    that cannot settle yet (a named tier with unsettled references, or no
//!    candidate levels) is retried on later passes.
//! 2. **Block estimate**: every still-unsettled tier gets a per-block least-squares
//!    estimate (`mast ~ a*cos(theta) + b*sin(theta)`, the planes-through-a-circle
//!    model of a block whose facets all close against one edge ring) as a starting
//!    point.
//! 3. **Nearest-level refinement** ([`SolveStrategy::JointGroup`] for tiers that
//!    entered it as estimates): Jacobi sweeps over the full arrangement in which
//!    every meet-derived tier snaps to the *shallowest* candidate vertex level
//!    incident to all of its resolved named references when one exists (the same
//!    rule phase 1 uses, and the oracle-measured best pick), else the level
//!    *nearest its current mast*. The true configuration is a stable fixed point
//!    of this update (measured); the sweeps both settle mutually-dependent groups
//!    and polish phase-1 values.
//!
//! `"Meet <names>"` references are resolved by [`MeetNameResolver`], which handles
//! the corpus's informal reference styles (unnamed girdle/culet/table references,
//! compound `"1-2-G1"` vertex specs, connective prose, case and side-prefix
//! mismatches) -- measured on the 2,881-design corpus this lifts fully-resolved
//! `MeetNamed` tiers from 30.6% to 82.5%, and an oracle test (every other tier
//! pinned at its true mast) confirms the newly resolvable references pin the true
//! mast at least as well as the long-resolvable ones (83.8% vs. 74.0% of picks
//! within 0.5% relative).
//!
//! All geometry runs in `f64` on a deterministic candidate-vertex primitive (every
//! well-conditioned plane triple, solved directly, filtered by feasibility) -- no
//! convex-hull library, no hashed iteration, byte-identical results run to run.
//!
//! # External verification: printed proportions
//!
//! A wrong solve is *self-consistent* -- every tier still lands on a real meet
//! vertex -- so nothing internal to the arrangement separates it from the
//! truth (four internal discriminators were built, measured and rejected; see
//! the NOTE comments in this module). What does separate them is **external**:
//! the proportion figures printed on every real diagram (`Vol/W^3`, `L/W`,
//! `C/W`, `P/W`, `H/W`, scraped into `diagram_details`). Measured on the
//! corpus, the true configuration reproduces them to ~0.1% median while a
//! wrong solve is off by ~30% ([`super::stone_metrics`] holds the calibration
//! and the deterministic measurement). [`solve_meet_points_verified`] exploits
//! this with a greedy repair search over phase-1 vertex-level picks, scored by
//! [`ExternalProportions::combined_deviation`]; it cut the corpus median
//! relative error 0.2110 -> 0.1278 and more than doubled fully-correct designs
//! (see its doc comment for the full measured figures).
//!
//! # Validating a schedule (forward direction)
//!
//! [`vertex_meet_groups`] is the reverse capability: given an already-solved
//! [`GemPolyhedron`] (e.g. from a hand-entered schedule with real masts), it
//! reports which input planes actually touch at each vertex -- geometric ground
//! truth independent of whatever `G`-field text a file happens to carry.
//!
//! # Module layout
//!
//! This module is split by seam, not by size: [`blocks`] classifies each tier
//! into crown/pavilion/girdle; [`anchors`] fills in per-block scale references
//! from printed proportions; [`names`] resolves stated `"Meet <names>"` text
//! against tier names; [`candidates`] is the deterministic candidate-vertex
//! primitive (plane triples -> feasible vertices -> `n . v` levels); `phase1_cache`
//! is [`solve`]'s incremental candidate-vertex cache for its constructive pass;
//! [`solve`] holds the three-phase pipeline itself
//! ([`SolveContext`](solve::SolveContext), [`solve_meet_points`]); `verify` layers
//! the externally-verified repair search ([`solve_meet_points_verified`]) on top;
//! and `validation` holds the reverse (forward-validation) and schedule-export
//! helpers. Every path that was reachable as `meet_solver::X` before this split
//! still is, via the re-exports below.

use lapidary::asc::AscSchedule;

mod anchors;
mod blocks;
mod candidates;
mod names;
mod phase1_cache;
mod solve;
mod validation;
mod verify;

pub use anchors::apply_ratio_anchors;
pub use blocks::{Block, classify_blocks};
pub use candidates::tier_instance_normals;
pub use names::{MeetNameResolver, ResolvedNames, TokenResolution};
pub use solve::solve_meet_points;
pub use validation::{build_reconstructed_schedule, vertex_meet_groups};
pub use verify::{VERIFY_ACCEPT_TOL, VerifiedSolveReport, solve_meet_points_verified};

/// Half-extent of the bounding box standing in for the uncut rough stone. Real
/// `.asc` masts sit close to 1.0, so this never masquerades as a real facet; it only
/// keeps the candidate-vertex feasibility test well-defined before the arrangement
/// closes up.
const BLANK_HALF_EXTENT: f64 = 64.0;

/// Feasibility slack: a candidate vertex may poke this far (absolute; masts are ~1)
/// beyond a plane before that plane's tier counts as violated.
const EPS_FEAS: f64 = 1e-5;

/// A plane within this absolute distance of a vertex counts as passing through it
/// (used to test incidence with named meet references).
const EPS_INCIDENT: f64 = 1e-4;

/// Two candidate `n . v` values within this absolute distance belong to one vertex
/// "level".
const LEVEL_TOL: f64 = 1e-5;

/// Minimum `|determinant|` for a triple of unit plane normals to define a candidate
/// vertex.
const MIN_TRIPLE_DET: f64 = 1e-6;

/// Designs with more facet planes than this are not solved (the candidate
/// enumeration is cubic in the plane count). No design in the 2,881-design corpus
/// comes anywhere near it.
const MAX_PLANES: usize = 400;

/// Maximum constructive sweeps (phase 1) and refinement sweeps (phase 3). Values
/// snap onto exact vertex levels, so convergence, when it happens, is exact; the
/// caps only guard against cycling.
const MAX_CONSTRUCTIVE_SWEEPS: usize = 64;
const MAX_REFINE_SWEEPS: usize = 16;

/// A picked level more than this many times the design's scale prior almost
/// certainly means the region isn't really bounded there yet (the pick hit
/// bounding-blank geometry); such picks are rejected.
const BLANK_DOMINATION_FACTOR: f64 = 4.0;

/// Default scale assumed when a solve has no scale-reference tier at all.
const DEFAULT_PLAUSIBLE_SCALE: f64 = 1.0;

/// One facet tier awaiting a solved mast distance.
///
/// Mirrors the fields of [`lapidary::asc::AscTier`] that matter for geometry
/// (`angle_deg`, `indices`), plus the constraint that determines its mast. Kept
/// independent of `AscTier` itself so this solver isn't coupled to one file format:
/// any producer of angle/index data can build this directly.
#[derive(Debug, Clone)]
pub struct MeetTierInput {
    /// Signed angle from the girdle plane, in degrees (`GemCAD` convention: negative
    /// is pavilion, non-negative is crown). An unsigned `0.0` inherits the previous
    /// tier's side, matching [`super::cuts::StandardGemCuts::from_asc_schedule`]'s
    /// documented convention exactly.
    pub angle_deg: f64,
    /// Index-wheel positions this tier's facet occurs at. Empty means a single
    /// facet at azimuth 0 (e.g. an unlisted table/culet).
    pub indices: Vec<f64>,
    pub constraint: MeetConstraint,
    /// Every name this tier is known by in the source schedule (e.g. `["P1"]`, or
    /// `["c", "d"]` for a tier folded from more than one named group -- see
    /// [`lapidary::asc::AscTier::names`]). Used only to resolve
    /// [`MeetConstraint::MeetNamed`]'s facet-name references back to tier indices.
    pub names: Vec<String>,
}

/// How a tier's mast distance is determined.
#[derive(Debug, Clone)]
pub enum MeetConstraint {
    /// Cut this facet past first touch until its plane passes through a meet vertex
    /// of the other facets' arrangement, with no further information about *which*
    /// vertex. Covers "Cut to centerpoint", "Table", and any meet-style instruction
    /// that doesn't name specific facets; solved with the measured rank-1 prior --
    /// see the module docs.
    MeetExisting,
    /// Like [`Self::MeetExisting`], but the source schedule states explicitly which
    /// facets this tier closes against (a real `.asc` `"G Meet P1, P2, G1"`
    /// instruction). Resolved against every tier's [`MeetTierInput::names`] in the
    /// same `solve_meet_points` call via [`MeetNameResolver`] (which also handles
    /// the corpus's informal reference styles -- unnamed girdle/culet/table
    /// references, compound `"1-2-G1"` vertex specs, prose words, case and
    /// side-prefix mismatches); a token that still doesn't resolve is dropped, and
    /// a tier none of whose tokens resolve degrades to [`Self::MeetExisting`]'s
    /// rank-1 handling.
    MeetNamed(Vec<String>),
    /// An externally supplied target mast: "Set girdle thickness" / "Set stone size"
    /// / "Level girdle", or a caller-supplied per-block dimension (see the module
    /// docs on anchors). Not derivable from geometry; a genuine design choice.
    ScaleReference(f64),
}

/// Builds [`MeetTierInput`]s directly from a parsed `.asc` schedule, classifying each
/// tier's [`MeetConstraint`] from its `G`-field text (see
/// [`lapidary::asc::AscTier::meet_instruction`]):
///
/// - [`lapidary::asc::MeetInstruction::Meet`] -> [`MeetConstraint::MeetNamed`];
/// - [`lapidary::asc::MeetInstruction::ScaleReference`] and
///   [`lapidary::asc::MeetInstruction::LevelGirdle`] -> [`MeetConstraint::ScaleReference`],
///   using the tier's own recorded `mast` as the externally-supplied value;
/// - everything else -> [`MeetConstraint::MeetExisting`].
///
/// This never fabricates a scale anchor: a block with no stated anchor still needs
/// one from the caller (see the module docs) -- a schedule alone does not always
/// carry that information.
#[must_use]
pub fn meet_tier_inputs_from_asc(schedule: &AscSchedule) -> Vec<MeetTierInput> {
    schedule
        .tiers
        .iter()
        .map(|tier| {
            let constraint = match tier.meet_instruction() {
                Some(lapidary::asc::MeetInstruction::Meet(names)) => {
                    MeetConstraint::MeetNamed(names)
                }
                Some(
                    lapidary::asc::MeetInstruction::ScaleReference
                    | lapidary::asc::MeetInstruction::LevelGirdle,
                ) => MeetConstraint::ScaleReference(tier.mast),
                _ => MeetConstraint::MeetExisting,
            };
            MeetTierInput {
                angle_deg: tier.angle_deg,
                indices: tier.indices.clone(),
                constraint,
                names: tier.names().into_iter().map(str::to_string).collect(),
            }
        })
        .collect()
}

/// Which technique actually produced a tier's solved mast, for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStrategy {
    /// The mast was supplied directly (a [`MeetConstraint::ScaleReference`]).
    ScaleReference,
    /// The mast was settled in the constructive fixed-point pass: a candidate meet
    /// vertex built from already-settled planes (named-reference incidence when the
    /// schedule stated one, the rank-1 level prior otherwise), possibly polished by
    /// the nearest-level refinement sweeps afterwards.
    DependencyOrder,
    /// The mast belonged to a mutually-dependent remainder the constructive pass
    /// could not order, and was settled by the nearest-level refinement sweeps over
    /// the full arrangement (from a per-block estimate starting point).
    JointGroup,
    /// The tier never had a usable candidate vertex; the returned mast is the
    /// per-block `a*cos(theta) + b*sin(theta)` estimate, not a vertex-derived
    /// value.
    LeastSquaresFallback,
    /// The solve could not produce even an estimate (e.g. the design exceeds
    /// [`MAX_PLANES`]). The returned mast is a placeholder and should not be
    /// trusted.
    Failed,
}

/// One tier's solved result.
#[derive(Debug, Clone)]
pub struct SolvedTier {
    pub mast: f64,
    pub strategy: SolveStrategy,
    /// Human-readable detail about how the value was obtained.
    pub detail: String,
}

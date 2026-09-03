//! [`SceneState`]: the fully-resolved description of one frame that a remote render
//! worker needs to reproduce it -- and nothing else.
//!
//! # Why "fully-resolved" matters
//!
//! The viewer stores a gem material as `material_name: String` plus a
//! `custom_materials: Vec<GemMaterial>` list loaded from its local SQLite database, and
//! stores a diagram as an id it looks up against `diagram-catalog`. A remote worker has
//! neither the database connection nor the catalog. Sending the *name* or the *id*
//! instead of the resolved value would compile, serialize, and deserialize just fine --
//! and then silently render the wrong stone (or the wrong facet geometry) the moment
//! someone selects a custom material or a diagram the worker's own copy of the data
//! doesn't happen to agree with, with no error at all. So [`SceneState`] carries the
//! resolved [`GemMaterial`] and the resolved facet-plane geometry directly, never a
//! name or an id.
//!
//! # Why NOT `dirty` / `running` / `paused` / `tab_visible` / `quality_preset`
//!
//! Those are local UI/session bookkeeping in the viewer -- whether its own render loop
//! is currently allowed to spend CPU, whether the window is visible, which
//! samples-per-frame preset the user picked. None of it changes what a worker computes:
//! the viewer already resolves `quality_preset` down to a concrete sample count before
//! asking for work (see the `samples` field on the `RENDER` message in
//! [`crate::messages`]), and the rest never crosses the wire at all.

use gemray::{
    geometry::GpuFacetPlane,
    optics::{materials::GemMaterial, raytracer::LightingPreset},
};

/// Everything a remote render worker needs to trace samples for one frame, fully
/// resolved.
///
/// See the module docs for why every field here is a value, never a name or an id that
/// would require the worker to consult data it doesn't have.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SceneState {
    /// Output image width, in pixels.
    pub width: u32,
    /// Output image height, in pixels.
    pub height: u32,
    /// Camera orbit yaw, radians.
    pub yaw: f32,
    /// Camera orbit pitch, radians.
    pub pitch: f32,
    /// Camera orbit distance from the origin.
    pub distance: f32,
    /// Key light yaw, radians -- see `gemray::optics::studio_rig::StudioRig`.
    pub light_yaw: f32,
    /// Key light pitch, radians.
    pub light_pitch: f32,
    /// Tone-mapping exposure multiplier.
    pub exposure: f32,
    /// Maximum ray bounce depth.
    pub max_bounces: u32,
    /// Which analytic studio lighting rig to sample when a ray misses the gem.
    pub lighting_preset: LightingPreset,
    /// The fully-resolved gem material -- never a `material_name`. See the module docs.
    pub material: GemMaterial,
    /// The fully-resolved facet-plane geometry -- never a diagram id. See the module
    /// docs.
    pub planes: Vec<GpuFacetPlane>,
    /// Whether the girdle band renders with a frosted (diffusely-scattering) finish
    /// rather than the default polished one -- the wire-format encoding of the viewer's
    /// frosted-girdle toggle.
    ///
    /// Deliberately a single `bool`, not a `Vec<optics::raytracer::FacetFinish>`
    /// parallel to `planes`: `gemray::geometry::girdle_facet_finishes` is a pure,
    /// deterministic function of `planes` alone (the girdle band is classified purely
    /// from each plane's normal -- see that function's own doc comment), so a worker
    /// re-derives the exact same per-facet finish list from `planes` and this one bit
    /// rather than needing it shipped explicitly -- keeping the wire format smaller and
    /// avoiding a second place the finish list could go stale relative to `planes`.
    /// `#[serde(default)]` so an on-disk `scene.json` written before this field existed
    /// (see `gemray-worker::render_cmd`) still deserializes, defaulting to `false`
    /// (all-polished) -- the exact behaviour every such file already rendered with.
    #[serde(default)]
    pub girdle_frosted: bool,
}

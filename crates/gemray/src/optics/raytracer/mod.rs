//! The spectral path tracer.
//!
//! Camera/ray generation, polyhedron intersection, Fresnel/TIR refraction,
//! Henyey-Greenstein scattering, pleochroic absorption, environment sampling,
//! low-discrepancy sampling, and spectral-to-tristimulus colour conversion.
//!
//! Split from a single `raytracer.rs` into this module tree by seam (see each
//! submodule's own doc comment for what it owns). This file is a pure re-export hub:
//! every path that was reachable as `gemray::optics::raytracer::X` before the split is
//! still reachable at exactly that path via the `pub use`/`pub(crate) use` lines below,
//! and every submodule can see every other submodule's `pub(super)` items directly.

// `pub`, not private: a `pub(crate)` item inside a private module is only reachable
// via this file's own re-export below, which clippy's `redundant_pub_crate` (nursery)
// flags as suspicious since it cannot see through the re-export to know that. Making
// the submodule `pub` resolves it; this does not widen any item's own effective
// visibility -- a `pub(crate)` (or private) item under a `pub` module is still capped
// at its own declared visibility, exactly as it was under a private module.
pub mod absorption;
pub mod camera;
pub mod color;
pub mod environment;
pub mod intersect;
pub mod refraction;
pub mod sampling;
pub mod scattering;
pub mod transport;

/// Number of spectral channels `trace_spectral_ray` traces per ray (8-channel
/// stratified hero-wavelength sampling). Module-level (rather than a `const` local to
/// `trace_spectral_ray`, as it used to be) so the per-bounce helper functions extracted
/// from that function can also reference it.
const NUM_CHANNELS: usize = 8;

// ---------------------------------------------------------------------------------
// Re-exports -- every one of these preserves a path that was reachable directly off
// `raytracer` before the module split. Visibility on each line matches the item's
// original visibility exactly (private items that needed to become `pub(super)` for
// cross-submodule access are NOT re-exported here, since they were never reachable at
// `raytracer::X` to begin with -- see the split's task report for that list).
//
// `pub(crate)` re-exports below carry `#[allow(unused_imports)]`: sibling submodules
// reach each other directly (`use super::sibling::*;`), never through this hub, so
// nothing in THIS crate necessarily names the re-exported path itself -- but the path
// must still resolve for consumers this exact build may not compile (in particular
// `renderer::gpu::transport_check`'s Tier 2 harness, gated on `feature = "gpu"`, and
// `crates/gemray/tests/`). Without the allow, a plain build sees a re-export nothing
// in it actually names and (correctly, from its own narrow view) flags it unused.
// ---------------------------------------------------------------------------------

// camera.rs
pub use camera::{Camera, FacetFinish, HitRecord, Ray};

// intersect.rs
pub use intersect::intersect_polyhedron;
#[allow(
    unused_imports,
    reason = "preserves the pre-split raytracer::X path for renderer::gpu::transport_check's \
              Tier 2 harness (feature = \"gpu\"), which this build may not compile"
)]
pub(crate) use intersect::{build_plane_soa, intersect_polyhedron_soa, shading_normal_near_edge};

// sampling.rs
#[allow(
    unused_imports,
    reason = "preserves the pre-split raytracer::X path for renderer::gpu::rng_check's Phase 0 \
              self-test (feature = \"gpu\"), which this build may not compile"
)]
pub(crate) use sampling::{
    BIREFRINGENT_SPLIT_STREAM, DISTANCE_SAMPLE_STREAM, FRESNEL_BRANCH_STREAM, FROSTED_DIR_U_STREAM,
    FROSTED_DIR_V_STREAM, MODE_COUPLING_STREAM, PHASE_DIR_U_STREAM, PHASE_DIR_V_STREAM,
    RUSSIAN_ROULETTE_STREAM,
};
pub use sampling::{
    HERO_WAVELENGTH_ROTATION_STREAM, PIXEL_JITTER_X_ROTATION_STREAM,
    PIXEL_JITTER_Y_ROTATION_STREAM, PixelRotations, SampleDraws, cranley_patterson_rotate,
    hash_u32, low_discrepancy_base2, pixel_rotations, radical_inverse_base, sample_draws,
};

// environment.rs
pub use environment::{
    EnvironmentSource, LightingPreset, LightingRigParams, blackbody_spectrum,
    sample_studio_environment,
};

// color.rs
pub use color::{aces_tonemap, cie_1931_cmf, xyz_to_rgb_in_space, xyz_to_srgb_gamma};
#[allow(
    unused_imports,
    reason = "preserves the pre-split raytracer::X path for crates/gemray/tests/ and same-crate \
              callers outside this module tree"
)]
pub(crate) use color::{
    apply_von_kries_white_balance, compute_illuminant_white_balance, illuminant_temperature_k,
    integrate_channels_to_xyz, spectral_mis_weight,
};

// absorption.rs
pub use absorption::spectral_absorption;
#[allow(
    unused_imports,
    reason = "preserves the pre-split raytracer::X path for same-crate callers outside this \
              module tree"
)]
pub(crate) use absorption::{channel_absorption_alphas, signed_frame_rotation_psi};

// refraction.rs
#[allow(
    unused_imports,
    reason = "preserves the pre-split raytracer::X path for same-crate callers outside this \
              module tree"
)]
pub(crate) use refraction::{
    BounceRefractionGeometry, RayMaterialContext, per_channel_uniaxial_indices, theta_c_for_bounce,
    tir_phase_delta,
};

// scattering.rs
#[cfg(any(test, feature = "gpu"))]
#[allow(
    unused_imports,
    reason = "preserves the pre-split raytracer::henyey_greenstein_phase path; only named by \
              this crate's own tests or the gpu-feature Tier 2 harness, not always both"
)]
pub(crate) use scattering::henyey_greenstein_phase;
#[allow(
    unused_imports,
    reason = "preserves the pre-split raytracer::X path for renderer::gpu::transport_check's \
              Tier 2 harness (feature = \"gpu\"), which this build may not compile"
)]
pub(crate) use scattering::{
    apply_frosted_bounce, cosine_weighted_hemisphere, maybe_scatter_or_extinguish,
    sample_henyey_greenstein_direction,
};

// transport.rs
pub use transport::{
    PathTermination, trace_spectral_ray, trace_spectral_ray_with_finish,
    trace_spectral_ray_with_finish_instrumented, wrapped_hero_wavelengths,
};

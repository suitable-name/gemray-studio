//! GPU compute infrastructure for `gemray` (behind the `gpu` feature) -- Phase 0.
//!
//! Adapter/device acquisition, a minimal compute-pipeline harness, buffer
//! upload/readback helpers, and the mandatory struct-layout/RNG/self-determinism
//! self-tests that any future physics port must pass before its output can be trusted.
//!
//! # What this is not
//!
//! Nothing here ports `optics::raytracer::trace_spectral_ray` -- see that module's own
//! doc comment and `renderer::pipeline`'s for why a real GPU raytracer does not exist
//! yet (the old shader was quarantined; `renderer::pipeline::GemRaytracerPipeline`
//! still panics unconditionally). This module exists so a *future* port has both the
//! plumbing and the verification harness to land on, not the physics itself.
//!
//! # Modules
//!
//! - [`context`]: adapter/device acquisition ([`GpuContext`]).
//! - [`compute`]: generic compute-pipeline/buffer helpers, not specific to any one
//!   kernel.
//! - [`layout_check`]: the mandatory struct-layout GPU echo test (Phase 0
//!   deliverable 2).
//! - [`rng_check`]: the mandatory RNG/integer bit-exactness test against the CPU
//!   (Phase 0 deliverable 3, "Tier 1").
//! - [`determinism_check`]: the mandatory GPU self-determinism test (Phase 0
//!   deliverable 3, "Tier 0").
//!
//! `examples/gpu_equivalence_harness.rs` (also gated on the `gpu` feature, and NOT a
//! `cargo test` target since it needs a real GPU adapter) wires all three self-tests
//! together into one pass/fail report.

pub mod compute;
pub mod context;
pub mod determinism_check;
pub mod layout_check;
pub mod rng_check;

// Phase 1: geometry/environment self-tests -- camera ray generation,
// `intersect_polyhedron`'s entry/exit slab test, analytic studio environment sampling,
// CIE 1931 CMF integration, von Kries white balance, and the furnace anchor tying all
// of those together and checking them against analytically computable truth. See each
// module's own doc comment. No transport physics (Fresnel, Stokes/Mueller, absorption,
// Russian roulette, birefringence, refraction) is ported here -- that is Phase 2+.
pub mod camera_check;
pub mod environment_check;
pub mod furnace_check;
pub mod polyhedron_check;
pub(crate) mod ulp;

// Phase 2: the full isotropic spectral estimator (Fresnel/TIR, Stokes-Mueller polarized
// transport, pleochroic Beer-Lambert absorption, Russian roulette, spectral MIS) --
// cubic (isotropic) materials only, birefringence is Phase 3. See each module's own doc
// comment.
pub mod estimator_check;
pub mod transport_check;

// A standalone Tier 2 kernel/self-test for
// `shading_normal_near_edge`, mirroring `polyhedron_check`'s own "dedicated file with
// its own `planes` binding" pattern rather than living inside `transport_check`/
// `transport_functions.wgsl` (which have no scene-geometry binding at all).
pub mod shading_normal_check;

// The one module here that is NOT a self-test: the general entry point that renders an
// arbitrary scene with the megakernel every module above verifies. Its dispatch routine
// is the one `estimator_check` uses, so those checks exercise the shipped path rather
// than a lookalike. See its own doc comment.
pub mod frame;

// CPU+GPU hybrid frame rendering: the GPU and every CPU core trace disjoint sample
// ranges of the same frame at the same time, merging into one image. Built entirely on
// top of `frame`'s existing `GpuFrameRenderer`/`sample_offset` plumbing -- see its own
// doc comment.
pub mod hybrid;

pub use context::{GpuAcquireError, GpuContext};
pub use frame::{GpuFrameError, GpuFrameRenderer, GpuFrameScene};

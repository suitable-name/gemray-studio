pub mod buffers;
pub mod denoise;
pub mod tonemap;

/// CPU-side HDR environment-map loading and importance sampling.
///
/// Always available (no `gpu` feature required) -- see the module docs for the `hdr`
/// feature that gates actual file decoding.
pub mod env_map;

#[cfg(feature = "gpu")]
pub mod env_map_gpu;

// GPU compute infrastructure and its mandatory self-tests -- see that module's own doc
// comment. Phase 0 infrastructure; no physics.
#[cfg(feature = "gpu")]
pub mod gpu;

// Deliberately NOT `#[cfg(feature = "gpu")]`, unlike everything else here that touches
// the GPU: this is the decline-and-fall-back wrapper both applications drive, and it
// exists in both feature configurations precisely so their render loops need no `#[cfg]`
// of their own. See its module doc comment.
pub mod gpu_backend;

#[cfg(feature = "gpu")]
pub mod pipeline;

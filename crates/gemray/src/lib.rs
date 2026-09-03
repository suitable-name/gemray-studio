//! `gemray` is a physically-based spectral gemstone renderer.
//!
//! It turns GemCAD-style cutting schedules -- a table of facet angles and
//! index positions, as used by faceting diagrams -- into rendered output:
//! full spectral analytical / Monte-Carlo raytracing through faceted
//! gemstone geometry, plus the gemological brilliance / fire / scintillation
//! metrics derived along the way.
//!
//! `gemray` has no dependency on any particular UI toolkit or data source:
//! callers supply plain [`FacetSpec`] rows (or hand-built [`geometry::GpuFacetPlane`]
//! sets) and a [`optics::materials::GemMaterial`], and get back rendered pixels
//! and/or optical metrics.

// `wgpu`'s own types (`renderer::gpu::frame::TransportOutputs` -> `wgpu::Buffer` ->
// several layers of internal `wgpu_core`/`wgpu_hal` Arc/Registry/Hub wrapping) nest
// deep enough that trait solving overflows the default recursion limit while proving
// `Send` for the closure `renderer::gpu::hybrid` spawns onto a scoped thread -- see
// that call site's own comment. This is a compiler resource limit on trait-solving
// depth, not a lint: raising it doesn't silence anything, it lets the solver finish a
// proof it was already correctly attempting. 256 is double rustc's default (128) and
// clears this crate's actual nesting with headroom; raise further only if a future
// `wgpu` upgrade nests deeper still.
#![recursion_limit = "256"]

pub mod color;
pub mod geometry;
pub mod optics;
pub mod renderer;
pub mod simd;

pub use geometry::cuts::FacetSpec;

/// A deterministic content hash of this crate's own source, computed at build time.
///
/// Covers `src/**/*.rs` plus `Cargo.toml` -- see `build.rs`'s doc comment for the full
/// rationale and the exact hashing procedure (sorted POSIX-normalized relative paths,
/// `\r\n` normalized to `\n`, FNV-1a).
///
/// 16 lowercase hex characters (a 64-bit FNV-1a digest). Two builds of `gemray` from
/// byte-identical (modulo line endings) source produce the same `BUILD_ID` regardless
/// of host OS, Rust toolchain version, or `.git` state; any change to any `.rs` file's
/// path or contents changes it.
///
/// This exists for `gemray-net`'s handshake: a viewer and a remote render worker must
/// refuse to combine samples unless their `gemray` builds match, since two different
/// physics implementations silently summed together produce a plausible-looking but
/// wrong image with no error and no crash.
pub const BUILD_ID: &str = env!("GEMRAY_BUILD_ID");

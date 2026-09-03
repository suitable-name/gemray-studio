//! `gemray-worker`: a library server for a `gemray`-backed gemstone design catalogue,
//! with optional render capacity on top.
//!
//! **Default role: library server.** Running `serve` with no extra features accepts
//! `gemray-net`'s read-only design-library protocol (see [`serve::library`]) over
//! mutual TLS -- listing/searching designs, fetching one in full, fetching an
//! attachment's bytes. This is deliberately the default, not an add-on: long term this
//! becomes full gem-CAD software with a possible Slint mobile client, and a mobile
//! client would talk to exactly this protocol and never have a renderer compiled in at
//! all. Serving the design library, not rendering, is this binary's primary role going
//! forward.
//!
//! **`worker` feature (off by default): render capacity.** `RenderRequest` handling,
//! `stream_emit`, `render_core`, `render_cmd`, the `render` subcommand, and `Backend`
//! advertisement in `WELCOME` are all gated behind it -- see this crate's own
//! `Cargo.toml` for exactly what that turns on/off, and [`serve`]'s module docs for how
//! one `serve` connection dispatches between the library and render protocols once both
//! are compiled in.
//!
//! - `render` (only meaningful with `worker` on): trace a scene straight to a PNG, no
//!   networking. See [`render_cmd::run`].
//! - `serve`: accept connections over TCP and serve the library protocol (always) and,
//!   under `worker`, `RenderRequest`s too. See [`serve::run`].
//!
//! # GPU
//!
//! An optional `gpu` feature (off by default, implies `worker` -- see this crate's
//! `Cargo.toml`) routes tracing through `gemray`'s GPU megakernel instead of
//! `optics::raytracer::trace_spectral_ray`. That port is complete and verified end to
//! end against a real adapter (Tier 2 per-function ULP budgets at max genuine ULP = 0,
//! energy-conservation furnace anchors, Tier 3 statistical image comparisons, uniaxial
//! birefringence through Phase 3 -- see `crates/gemray/examples/gpu_equivalence_harness.rs`),
//! which supersedes the rationale that used to live here (the WGSL kernels predating
//! several physics corrections the CPU path already had).
//!
//! [`gemray::renderer::gpu_backend::GpuBackend`] is the decline/fallback wrapper both
//! this crate and `apps/diagram-gui` drive. It lives in `gemray` rather than here
//! because two copies of a *policy* -- the biaxial-material and HDR-environment
//! declines are correctness rules, not conveniences -- is a copy that can drift into
//! silently rendering the wrong image.

// Trait-solver resource limit, not a lint: under `--features gpu`, proving the
// `thread::spawn` closure in `serve::run`'s accept loop is `Send` requires recursing
// through `wgpu`'s deeply nested internal types (see the `# GPU` section above), and
// the default recursion limit isn't enough to finish that proof -- rustc emits a
// future-incompatibility warning today and will hard-error once the default limit
// becomes non-negotiable upstream. Raising this doesn't silence anything; it just lets
// the solver complete the proof it was already attempting. Unconditional (not
// `cfg_attr`-gated to `gpu`) because a single crate-wide limit is simpler to reason
// about than tracking which feature combinations need which limit, and it costs
// nothing when the deeper recursion never happens (the `worker`/default builds don't
// pull in `wgpu` at all -- see `Cargo.toml`'s `worker`/`gpu` features -- so this is a
// no-op for them). Do not remove: the overflow reproduces on every `--features gpu`
// clippy/build.
#![recursion_limit = "256"]

pub mod cli;
pub mod enroll;
pub mod enroll_client;
pub mod pki;
#[cfg(feature = "worker")]
pub mod png_out;
#[cfg(feature = "worker")]
pub mod render_cmd;
#[cfg(feature = "worker")]
pub mod render_core;
pub mod serve;
#[cfg(feature = "worker")]
mod stream_emit;
#[cfg(feature = "worker")]
pub mod validate;

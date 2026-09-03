//! Wire protocol for a `gemray-worker`: the read-only design-library sync protocol
//! every build speaks, and (optionally) offloading `gemray` spectral ray-sample
//! computation to it as a remote render worker.
//!
//! **Types, codec, and framing only -- no networking, no sockets.** Every function here
//! operates on an in-memory buffer, a `[u8]` slice, or a generic `Read`/`Write`, so the
//! whole crate is testable with nothing more than a `std::io::Cursor`. `apps/gemray-worker`
//! wires these same functions to an actual `TcpStream`/TLS connection for the worker
//! binary and the viewer's client.
//!
//! # The `render` feature
//!
//! Off by default. [`scene`] (and, inside [`messages`], [`messages::RenderRequest`] and
//! the `ClientMessage::RenderRequest` variant) need `gemray`'s resolved scene/material
//! types and are gated behind it -- everything else (the [`library`] sync protocol,
//! [`tls`], [`enroll`], [`token`], [`framing`], [`handshake`]) is always available,
//! `gemray`-free. `apps/gemray-worker`'s own `worker` feature turns this on; see that
//! crate's docs for why serving the design library, not rendering, is the default,
//! primary role a build of that binary plays.
//!
//! # Why sample-index partitioning makes remote render offload correct
//!
//! `gemray::optics::raytracer::trace_spectral_ray` returns one sample's XYZ radiance,
//! and the viewer's accumulation buffer is just `Vec<Vec3>` of running per-pixel sums.
//! Samples are additive and order-independent, so a remote node's contribution is
//! nothing more than more terms in that same sum -- there is no partial-frame
//! reassembly or per-tile stitching to get right. Work is partitioned by SAMPLE INDEX
//! (every node traces disjoint ranges of samples across the WHOLE frame), not by
//! screen-space tile: a gem only occupies part of the frame and background pixels are
//! nearly free to trace, so tile partitioning would load-balance badly while sample
//! partitioning divides the work evenly by construction. This composes cleanly with the
//! renderer's existing RNG, whose seeds derive from `hash_u32(pixel_index,
//! sample_number)` with decorrelated per-bounce streams -- see
//! `apps/diagram-gui/src/bridge/render_thread.rs`'s per-sample seed derivation, which
//! `tests/partition_correctness.rs` reproduces directly to verify additivity end to end
//! against the real `trace_spectral_ray`.
//!
//! # Modules
//!
//! - [`library`]: the read-only design-library sync protocol -- always available,
//!   regardless of the `render` feature. See that module's docs.
//! - [`scene`] (`render` feature): [`scene::SceneState`], everything a worker needs to
//!   trace a frame's samples, fully resolved (never a material name or a diagram id).
//! - [`radiance`]: the per-pixel `Vec<Vec3>` radiance-buffer codec, raw POD bytes via
//!   `bytemuck` -- no serialization framework, since this is the hot-path payload.
//! - [`messages`]: `HELLO` / `WELCOME` / the tagged `ClientMessage`/`StreamEvent`
//!   families, and their `postcard` encode/decode.
//! - [`framing`]: length-prefixed message framing over any `Read`/`Write`.
//! - [`handshake`]: the build-compatibility check that refuses to pair a viewer and a
//!   worker running different `gemray` physics -- see that module's docs for why this
//!   is not optional.
//! - [`tls`]: mutual-TLS config building (against a private CA, TLS 1.3 only) and the
//!   client-certificate fingerprint allowlist that stands in for revocation. Answers
//!   "may this peer talk to me" -- a different question from [`handshake`]'s "are we
//!   running the same physics"; see that module's own doc comment on why both checks
//!   stay, neither substituting for the other. Backs BOTH the render protocol and the
//!   library protocol -- see `apps/gemray-worker`'s docs on why library requests sit
//!   behind exactly the same authentication as render requests, never a separate check.
//! - [`client`]: the viewer-side protocol driver -- `HELLO`/`WELCOME` (plus a
//!   handshake-only "test connection" operation), `RenderRequest`/`CANCEL` framing, and
//!   the epoch-gated accumulator that sums `FRAME` deltas while keeping `PREVIEW`
//!   display-only. Generic over `Read`/`Write`, mirroring `apps/gemray-worker`'s own
//!   `handle_connection` -- see that module's doc comment.
//! - [`token`]: the compact `GW1-...` codec for one-time worker-enrollment tokens.
//! - [`enroll`]: the enrollment wire messages and the claiming client -- verifying a
//!   worker's enrollment listener against the CA fingerprint a token commits to, and
//!   redeeming the token for a certificate bundle. Shared by `apps/gemray-worker`'s
//!   `cert claim` and `apps/diagram-gui`'s own token-redeem UI; see that module's doc
//!   comment for why both this and [`token`] moved here rather than staying
//!   worker-only.

pub mod client;
pub mod enroll;
pub mod framing;
pub mod handshake;
pub mod library;
pub mod messages;
pub mod radiance;
#[cfg(feature = "render")]
pub mod scene;
pub mod tls;
pub mod token;

#[cfg(feature = "render")]
pub use scene::SceneState;

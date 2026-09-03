//! Viewer-side (client) protocol driver for talking to a `gemray-worker`.
//!
//! Mirrors `apps/gemray-worker`'s own split: this module is generic over `Read` /
//! `Write` (never `TcpStream` or a TLS type by name), so the whole client protocol --
//! handshake, request/cancel framing, and accumulation -- is unit-testable against an
//! in-memory [`std::io::Cursor`] with no sockets and no GUI, exactly the way
//! `apps/gemray-worker/src/serve.rs::handle_connection` is on the server side. Wrapping
//! an actual `TcpStream` (optionally inside `rustls::StreamOwned`, via
//! [`crate::tls::client_config`]) happens at the call site -- a future
//! `apps/diagram-gui` bridge module -- which this crate deliberately knows nothing
//! about.
//!
//! # Modules
//!
//! - [`handshake`]: `HELLO` / `WELCOME`, including the client-side defense-in-depth
//!   [`crate::handshake::verify_compatible`] check (the worker already refuses an
//!   incompatible `HELLO` on its own side; this module refuses independently rather
//!   than trusting the worker's self-report alone) and the "test connection" operation
//!   (handshake only, no render).
//! - [`accumulate`]: [`accumulate::Accumulator`], the epoch-gated radiance sum --
//!   FRAME deltas are summed into it, PREVIEW snapshots replace a separate
//!   display-only slot (never summed), and anything whose `request_id` doesn't match
//!   the accumulator's current epoch is dropped. See that module's doc comment for the
//!   full argument -- this is the single mechanism that keeps a stale, in-flight
//!   partial from a just-cancelled request from ever corrupting the next one.
//! - [`session`]: wires a `RenderRequest` write and the reply stream together --
//!   [`session::send_render_request`] / [`session::send_cancel`] for the write side,
//!   [`session::run_client_session`] to drive an [`accumulate::Accumulator`] from
//!   whatever a connection sends back for as long as it stays open (including several
//!   pipelined requests on one connection -- see that function's doc comment).

pub mod accumulate;
pub mod handshake;
pub mod session;

pub use accumulate::{Accumulator, ApplyOutcome, PreviewSnapshot};
pub use handshake::{ConnectionInfo, test_connection};
#[cfg(feature = "render")]
pub use session::send_render_request;
pub use session::{SessionUpdate, run_client_session, send_cancel, send_library_request};

use crate::messages::{ErrorMsg, NetError};

/// Everything that can go wrong on the client side of a `gemray-net` connection, from
/// the initial `HELLO`/`WELCOME` handshake through the render stream itself.
#[derive(Debug)]
pub enum ClientError {
    /// A transport-level failure: a malformed frame, or an I/O error other than a
    /// clean EOF (which callers that expect the peer to eventually close the
    /// connection handle separately -- see [`session::run_client_session`]'s doc
    /// comment).
    Net(NetError),
    /// The worker itself refused to pair -- e.g. it detected a `HELLO` build/protocol
    /// mismatch on its own side and replied with `ERROR` instead of `WELCOME`. Carries
    /// the worker's own `ErrorMsg` verbatim.
    Refused(ErrorMsg),
    /// This client's own defense-in-depth check
    /// ([`crate::handshake::verify_compatible`]) refused pairing even though the
    /// worker replied with a `WELCOME` -- see [`handshake`]'s module doc comment for
    /// why this check exists independently of the worker's own.
    Incompatible(crate::handshake::Incompatible),
    /// The reply to `HELLO` decoded as neither `WELCOME` nor `ERROR` -- see
    /// [`handshake`]'s module doc comment for why that ambiguity has to be resolved by
    /// trying both, and why a reply that is neither is a distinct, reportable failure
    /// rather than a silently-swallowed one.
    MalformedHandshakeReply,
    /// A `FRAME`/`PREVIEW` payload failed [`crate::radiance::decode`] -- wrong length
    /// (a worker/viewer dimension mismatch) or misaligned bytes.
    Radiance(crate::radiance::RadianceError),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Net(e) => write!(f, "{e}"),
            Self::Refused(e) => write!(f, "worker refused to pair: {} ({})", e.message, e.code),
            Self::Incompatible(e) => write!(f, "refusing to pair: {e}"),
            Self::MalformedHandshakeReply => {
                write!(f, "handshake reply decoded as neither WELCOME nor ERROR")
            }
            Self::Radiance(e) => write!(f, "malformed radiance payload: {e:?}"),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<NetError> for ClientError {
    fn from(e: NetError) -> Self {
        Self::Net(e)
    }
}

impl From<crate::radiance::RadianceError> for ClientError {
    fn from(e: crate::radiance::RadianceError) -> Self {
        Self::Radiance(e)
    }
}

//! Client for the read-only design-library protocol (`gemray_net::library`): a one-shot
//! [`request`] that connects, sends exactly one [`LibraryRequest`], reads exactly one
//! [`LibraryResponse`], and disconnects; and [`LibrarySession`], which connects once and
//! holds that connection open across many requests.
//!
//! Unlike [`crate::bridge::remote_render`] (which owns a socket for the life of a whole
//! progressive render and interleaves reading `StreamEvent`s with a cancel channel), the
//! library protocol is plain request/response (see `gemray_net::library`'s module doc
//! comment: "deliberately request/response, not streamed, unlike RENDER") -- no
//! `StreamEvent` poll-loop/timeout-read machinery is needed either way. Both forms here
//! reuse [`crate::bridge::remote_render::connect_and_handshake`] for the actual
//! mutual-TLS connect+`HELLO`/`WELCOME` (identical for both protocols -- both ride the
//! same [`gemray_net::messages::ClientMessage`] envelope over the same authenticated
//! connection) rather than duplicating it.
//!
//! # One-shot vs. one held session
//!
//! [`request`] is for a genuinely single call -- the interactive remote-browse path
//! (`bridge::library_source::spawn_library_request`) uses it: a user opens a search box,
//! types a query, gets one page of results back. Reconnecting (a fresh TCP handshake plus
//! a full mutual-TLS handshake, certificate-chain verification, and a server-side
//! allowlist re-read) for each of those isolated calls is a non-issue -- there is no tight
//! loop of them.
//!
//! [`LibrarySession`] exists for the opposite shape: `bridge::library_mirror`'s sync
//! issues one `SearchPage` per catalogue page, then one `FetchDesign` per design, then one
//! `FetchAttachment` per attachment -- thousands of requests for a real catalogue. Paying
//! a full connect+handshake for every one of those turns a mirror sync into several
//! minutes of pure handshaking before any useful data moves, plus needless connection
//! churn on the server. A [`LibrarySession`] connects and handshakes once, then reuses
//! that one stream (and the `welcome.library` check that came with it) for every request
//! for the life of the sync -- see its own doc comment for what happens when that held
//! connection drops mid-sync.
//!
//! # Always off the UI thread
//!
//! Every function here (and every [`LibrarySession::request`] call) is a blocking network
//! call. Callers MUST run them on their own thread and marshal the result back via
//! `Weak::upgrade_in_event_loop` -- see `bridge::library_source::spawn_library_request`
//! and `bridge::library_mirror::spawn_mirror_sync` for the two places that do this, and
//! `bridge::export_thread`'s module doc comment for the general pattern this crate
//! already follows for other backgrounded work.

use crate::{
    bridge::remote_render::{RemoteError, RemoteStream, connect_and_handshake},
    settings::WorkerSettings,
};
use gemray_net::{
    client::{self, ClientError, ConnectionInfo},
    library::{LibraryRequest, LibraryResponse},
    messages,
};
use std::{
    io::{Read, Write},
    sync::{Mutex, PoisonError},
};

/// Everything that can go wrong making one library request against a remote worker.
#[derive(Debug)]
pub enum LibraryClientError {
    /// Connecting or handshaking failed -- see [`RemoteError`]'s own variants (TLS,
    /// I/O, handshake refusal/incompatibility, a malformed worker address).
    Connect(RemoteError),
    /// The worker connected and authenticated fine but advertises no library capacity
    /// (`Welcome::library` is `false`) -- distinct from every other variant here on
    /// purpose, the same posture [`RemoteError::NoRenderCapacity`] already takes for
    /// rendering: nothing is broken, the operator simply pointed the viewer at a server
    /// that doesn't serve a design library, and saying so plainly is more useful than a
    /// generic connection failure.
    NoLibraryCapacity,
    /// Sending the request or reading the reply frame failed at the transport level, or
    /// the worker's reply decoded as [`LibraryResponse::Error`] -- see
    /// [`request`]'s doc comment for why that variant is folded in here rather than
    /// returned inside `Ok`.
    Client(ClientError),
    /// The worker replied [`LibraryResponse::Error`] -- a request-level failure it could
    /// still form a normal reply for (see that variant's own doc comment). Carries the
    /// worker's message.
    WorkerError(String),
}

impl std::fmt::Display for LibraryClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect(e) => write!(f, "{e}"),
            Self::NoLibraryCapacity => write!(
                f,
                "this worker does not serve a design library -- it was started without a library database"
            ),
            Self::Client(e) => write!(f, "{e}"),
            Self::WorkerError(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LibraryClientError {}

impl From<ClientError> for LibraryClientError {
    fn from(e: ClientError) -> Self {
        Self::Client(e)
    }
}

impl From<messages::NetError> for LibraryClientError {
    fn from(e: messages::NetError) -> Self {
        Self::Client(e.into())
    }
}

/// Connects to `worker` and checks it actually advertises library capacity
/// (`Welcome::library`) without sending any [`LibraryRequest`] -- the "is this worker
/// reachable and does it serve a library at all" check a source-switch UI action runs
/// before committing to it, mirroring `bridge::remote_render::test_connection`'s own
/// handshake-only shape.
///
/// # Errors
///
/// See [`LibraryClientError`]'s variants ([`LibraryClientError::WorkerError`] is never
/// returned here -- no request is sent).
pub fn probe(worker: &WorkerSettings) -> Result<ConnectionInfo, LibraryClientError> {
    let (_stream, welcome) = connect_and_handshake(worker).map_err(LibraryClientError::Connect)?;
    if !welcome.library {
        return Err(LibraryClientError::NoLibraryCapacity);
    }
    Ok(welcome.into())
}

/// Connects to `worker`, handshakes, and checks `welcome.library` -- the shared guts of
/// both [`request`]'s one-shot connect and [`LibrarySession`]'s (re)connect step. Unlike
/// [`probe`], the `Welcome` itself is discarded once checked: neither caller needs
/// anything from it beyond the capability bit.
fn connect_checked(worker: &WorkerSettings) -> Result<RemoteStream, LibraryClientError> {
    let (stream, welcome) = connect_and_handshake(worker).map_err(LibraryClientError::Connect)?;
    if !welcome.library {
        return Err(LibraryClientError::NoLibraryCapacity);
    }
    Ok(stream)
}

/// Sends exactly one [`LibraryRequest`] to `worker` and returns the design-library data
/// it carries back, over a fresh connection made and torn down just for this call. See
/// the module doc comment's "One-shot vs. one held session" section for when to reach for
/// this instead of [`LibrarySession`].
///
/// [`LibraryResponse::Error`] (a request-level failure the worker could still form a
/// normal reply for -- e.g. a malformed filter) is unwrapped into
/// [`LibraryClientError::WorkerError`] rather than returned as `Ok`, so every caller
/// handles "the worker said no" through the same `Result::Err` path as a transport
/// failure, instead of every call site having to separately match on
/// `LibraryResponse::Error` itself.
///
/// # Errors
///
/// See [`LibraryClientError`]'s variants.
pub fn request(
    worker: &WorkerSettings,
    req: &LibraryRequest,
) -> Result<LibraryResponse, LibraryClientError> {
    let mut stream = connect_checked(worker)?;
    send_and_read(&mut stream, req)
}

/// Writes `req` and reads back exactly one reply, over any `Read + Write`.
///
/// Generic rather than tied to [`RemoteStream`] specifically so
/// [`request_with_reconnect`]'s retry decision is unit-testable against an in-memory
/// duplex buffer standing in for a dropped connection, with no live socket -- the same
/// reason `gemray_net::client::session` stays generic over `Read`/`Write` (see that
/// module's own doc comment).
fn send_and_read<S: Read + Write>(
    stream: &mut S,
    req: &LibraryRequest,
) -> Result<LibraryResponse, LibraryClientError> {
    client::send_library_request(stream, req)?;
    let response: LibraryResponse = messages::read_message(stream)?;
    match response {
        LibraryResponse::Error(e) => Err(LibraryClientError::WorkerError(e.message)),
        other => Ok(other),
    }
}

/// Whether `e` means the CONNECTION itself is broken -- worth reconnecting for -- as
/// opposed to a reply the worker formed just fine (`WorkerError`) or a failure
/// [`request_with_reconnect`]'s own `connect` step already reports on its own
/// (`Connect`/`NoLibraryCapacity`, neither of which comes from [`send_and_read`]).
/// [`ClientError::Net`] is the transport-level case -- a malformed frame or an I/O error
/// (including the `UnexpectedEof` a dropped socket or a server restart produces); every
/// other [`ClientError`] variant ([`ClientError::Refused`]/[`ClientError::Incompatible`]/
/// [`ClientError::MalformedHandshakeReply`]) only ever comes out of a fresh handshake
/// (`connect_checked`), never out of [`send_and_read`] on an already-paired connection, so
/// they never reach this function in practice -- excluded from "retry-worthy" regardless,
/// since none of them would be fixed by retrying the same request again.
const fn is_dead_connection(e: &LibraryClientError) -> bool {
    matches!(e, LibraryClientError::Client(ClientError::Net(_)))
}

/// The reconnect-on-dead-connection decision [`LibrarySession::request`] applies to its
/// real held [`RemoteStream`] -- factored out generic over any `Read + Write` stream type
/// and a `connect` closure purely so the DECISION (when to drop `held` and dial again) is
/// unit-testable against a scripted in-memory stream, with no live socket. Establishing a
/// real connection (whatever `connect` here actually does in production --
/// [`connect_checked`]) is, like the rest of `connect_and_handshake`, only exercisable
/// against a live `gemray-worker serve` process; this function's own tests script
/// `connect` as a plain closure instead.
///
/// `held` starts (or ends up, after a failed request) as `None` -- `connect()` is called
/// to fill it. [`send_and_read`] is tried against whatever's in `held`; on success that
/// connection is kept for next time. On a [`is_dead_connection`] failure, `held` is
/// dropped, `connect()` is called exactly once more, and the SAME `req` is retried on the
/// fresh connection -- this is the "reconnect and resume" policy `LibrarySession`'s own
/// doc comment commits to: one dropped connection costs one extra handshake and a retried
/// request, never an aborted sync. Any other error (from either `connect()` call, or a
/// non-dead-connection [`send_and_read`] failure) propagates immediately, with `held` left
/// however it was set along the way (always `None` after a failed reconnect attempt, so
/// the next call starts clean rather than retrying a connection already known bad).
fn request_with_reconnect<S: Read + Write>(
    held: &mut Option<S>,
    req: &LibraryRequest,
    mut connect: impl FnMut() -> Result<S, LibraryClientError>,
) -> Result<LibraryResponse, LibraryClientError> {
    if held.is_none() {
        *held = Some(connect()?);
    }
    let stream = held.as_mut().expect("just connected above if it was empty");
    match send_and_read(stream, req) {
        Ok(response) => Ok(response),
        Err(e) if is_dead_connection(&e) => {
            *held = None;
            let mut fresh = connect()?;
            let response = send_and_read(&mut fresh, req)?;
            *held = Some(fresh);
            Ok(response)
        }
        Err(e) => Err(e),
    }
}

/// A held mutual-TLS connection to `worker`'s design-library protocol, reused across many
/// [`LibraryRequest`]s for the life of one `bridge::library_mirror` sync. See the module
/// doc comment's "One-shot vs. one held session" section for why this exists alongside
/// [`request`], and the crate-level task report for the handshake-count difference this
/// makes against a real multi-thousand-design catalogue.
///
/// # Connecting, and the `welcome.library` check, happen once -- not per request
///
/// The very first [`Self::request`] call connects, handshakes, and checks
/// `welcome.library`, exactly [`connect_checked`]'s shape; every call after that reuses
/// the same stream, paying for none of that again -- unless the connection has to be
/// re-established after dropping (below), which re-runs the SAME check on the fresh
/// connection (a session that reconnects genuinely has a new peer/pairing to verify, so
/// this is still "per session", never "per request": one check per live connection this
/// session ever holds, not one per [`LibraryRequest`] sent over it).
///
/// # What happens when the held connection drops mid-sync: reconnect and retry once
///
/// A long-held connection breaking mid-sync (network blip, server restart, idle timeout)
/// is a certainty over a multi-thousand-design sync, not an edge case. [`Self::request`]
/// (via [`request_with_reconnect`]) handles it by dropping the dead stream, connecting
/// fresh exactly once, and retrying the SAME request on it -- transparently to the caller,
/// which is `bridge::library_mirror::run_mirror_sync`'s per-design loop; that loop already
/// only ever writes a design to the database after every network call for it has
/// succeeded (see that module's own "Cancellation leaves the database consistent"
/// section), so a reconnect landing in the middle of one design's fetches (say, between
/// `FetchDesign` and a `FetchAttachment`) just means the rest of that design's requests
/// happen on the new connection -- no half-written design results either way.
///
/// If the reconnect attempt ITSELF fails (the network is genuinely down, not just
/// blipped), [`Self::request`] returns that error rather than retrying further -- the sync
/// does not hang retrying forever. What that means for the caller: a failure during
/// catalogue enumeration (`SearchPage`) surfaces as `MirrorOutcome::Failed`, nothing
/// written, exactly as an enumeration failure already did before this session existed; a
/// failure fetching one design's data or an attachment is counted in
/// `MirrorCounts::failed` and that one design is left exactly as it was locally (never
/// marked synced), so the next sync retries it -- also exactly the existing per-design
/// failure handling, unchanged. Either way, the invariants `library_mirror`'s own tests
/// pin (no half-written design, cancellation only between designs, a local-only design
/// surviving, the content-hash skip logic) hold regardless of how many times this
/// session's underlying connection had to be re-established along the way -- a resumed
/// sync may re-fetch a handful of already-mirrored designs (whichever ones were in flight
/// when the drop happened), which the hash check on the NEXT sync would have skipped
/// anyway; cheap, never corrupting. See
/// `tests::a_mid_sync_connection_drop_reconnects_and_the_sync_still_completes` in
/// `bridge::library_mirror` for this proven at the `run_mirror_sync` level, and this
/// module's own `tests::request_with_reconnect_*` for the reconnect DECISION itself.
pub struct LibrarySession {
    worker: WorkerSettings,
    /// `None` before the first request, and again immediately after a dead connection is
    /// dropped (before its replacement is dialed) -- see [`request_with_reconnect`].
    /// Guarded by a [`Mutex`] purely so [`Self::request`] can take `&self` (required by
    /// `bridge::library_mirror::LibraryTransport`'s signature); nothing here is actually
    /// touched from more than one thread at a time -- the whole session lives on one
    /// mirror-sync worker thread, start to finish.
    stream: Mutex<Option<RemoteStream>>,
}

impl LibrarySession {
    #[must_use]
    pub const fn new(worker: WorkerSettings) -> Self {
        Self {
            worker,
            stream: Mutex::new(None),
        }
    }

    /// Sends `req` on the held connection, connecting first if this is the very first
    /// call on this session, and transparently reconnecting-and-retrying once if that
    /// connection turns out to be dead -- see this type's own doc comment for the full
    /// policy and why it preserves every safety guarantee a mirror sync depends on.
    ///
    /// # Errors
    ///
    /// See [`LibraryClientError`]'s variants; a [`LibraryClientError::Connect`] here may
    /// mean either the very first connect failed, or a reconnect attempt after a dropped
    /// connection did.
    pub fn request(&self, req: &LibraryRequest) -> Result<LibraryResponse, LibraryClientError> {
        let mut guard = self.stream.lock().unwrap_or_else(PoisonError::into_inner);
        request_with_reconnect(&mut guard, req, || connect_checked(&self.worker))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_client_error_display_is_human_readable() {
        let e = LibraryClientError::NoLibraryCapacity;
        assert!(e.to_string().contains("does not serve a design library"));

        let e = LibraryClientError::WorkerError("bad filter".to_string());
        assert_eq!(e.to_string(), "bad filter");
    }

    /// A `Read + Write` over two independent in-memory buffers, standing in for one held
    /// connection's stream -- the same shape as `gemray_net::client::handshake`'s own
    /// `DuplexHalf` test double, used here so [`request_with_reconnect`]'s retry decision
    /// is exercised with no live socket (see that function's own doc comment on why it's
    /// factored out generic for exactly this).
    struct FakeStream {
        input: std::io::Cursor<Vec<u8>>,
        output: Vec<u8>,
    }

    impl FakeStream {
        fn new(input: Vec<u8>) -> Self {
            Self {
                input: std::io::Cursor::new(input),
                output: Vec::new(),
            }
        }

        /// A stream with nothing left to read -- `send_and_read` writes the request fine
        /// (a `Vec<u8>` write never fails) but then hits an immediate EOF trying to read
        /// the reply frame, exactly how a connection dropped out from under the reader
        /// looks from this side (see `gemray_net::framing::read_frame`'s treatment of EOF
        /// as an I/O error, never a valid empty frame).
        fn dead() -> Self {
            Self::new(Vec::new())
        }
    }

    impl Read for FakeStream {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.input.read(buf)
        }
    }

    impl Write for FakeStream {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.output.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn encoded_response(resp: &LibraryResponse) -> Vec<u8> {
        let mut buf = Vec::new();
        messages::write_message(&mut buf, resp).unwrap();
        buf
    }

    #[test]
    fn request_with_reconnect_reconnects_once_after_a_dead_connection_and_succeeds() {
        let good_reply = encoded_response(&LibraryResponse::NotFound);
        let mut remaining = vec![FakeStream::new(good_reply)];
        let connect_calls = std::cell::Cell::new(0);

        // The FIRST connect hands back an already-dead stream (nothing to read), so the
        // request against it fails and a reconnect is expected; the SECOND connect hands
        // back a stream with a real, decodable reply queued up.
        let mut held: Option<FakeStream> = Some(FakeStream::dead());
        let response = request_with_reconnect(&mut held, &LibraryRequest::FilterOptions, || {
            connect_calls.set(connect_calls.get() + 1);
            Ok(remaining.pop().expect("only one reconnect expected"))
        })
        .unwrap();

        assert_eq!(response, LibraryResponse::NotFound);
        assert_eq!(
            connect_calls.get(),
            1,
            "exactly one reconnect after the held connection turned out dead"
        );
        assert!(
            held.is_some(),
            "the freshly reconnected stream is kept in `held` for the next request"
        );
    }

    #[test]
    fn request_with_reconnect_connects_lazily_on_the_first_call() {
        let good_reply = encoded_response(&LibraryResponse::NotFound);
        let mut held: Option<FakeStream> = None;
        let connect_calls = std::cell::Cell::new(0);

        let response = request_with_reconnect(&mut held, &LibraryRequest::FilterOptions, || {
            connect_calls.set(connect_calls.get() + 1);
            Ok(FakeStream::new(good_reply.clone()))
        })
        .unwrap();

        assert_eq!(response, LibraryResponse::NotFound);
        assert_eq!(connect_calls.get(), 1);
    }

    #[test]
    fn request_with_reconnect_does_not_reconnect_for_a_worker_error_reply() {
        // LibraryResponse::Error is a reply the worker formed just fine -- request-level,
        // not a broken connection -- so this must never trigger a reconnect attempt.
        let error_reply = encoded_response(&LibraryResponse::Error(messages::ErrorMsg {
            code: 1,
            message: "bad filter".to_string(),
        }));
        let mut held = Some(FakeStream::new(error_reply));
        let connect_calls = std::cell::Cell::new(0);

        let result = request_with_reconnect(&mut held, &LibraryRequest::FilterOptions, || {
            connect_calls.set(connect_calls.get() + 1);
            Ok::<FakeStream, LibraryClientError>(FakeStream::dead())
        });

        assert!(matches!(result, Err(LibraryClientError::WorkerError(_))));
        assert_eq!(
            connect_calls.get(),
            0,
            "a worker-level error reply must never trigger a reconnect"
        );
    }

    #[test]
    fn request_with_reconnect_propagates_the_error_when_reconnecting_also_fails() {
        // The held connection is dead AND the network is genuinely down (not just a
        // blip) -- the whole point of "reconnect once, don't retry forever".
        let mut held = Some(FakeStream::dead());
        let connect_calls = std::cell::Cell::new(0);

        let result = request_with_reconnect(&mut held, &LibraryRequest::FilterOptions, || {
            connect_calls.set(connect_calls.get() + 1);
            Err::<FakeStream, _>(LibraryClientError::Connect(RemoteError::Io(
                std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused"),
            )))
        });

        assert!(matches!(result, Err(LibraryClientError::Connect(_))));
        assert_eq!(
            connect_calls.get(),
            1,
            "exactly one reconnect attempt, not a retry loop"
        );
        assert!(
            held.is_none(),
            "a connection already known dead is never left in `held` after a failed reconnect"
        );
    }
}

//! Owns the actual mutual-TLS socket to one configured remote worker and drives one
//! `RenderRequest` against it.
//!
//! **This is the one piece of remote rendering that cannot be unit-tested without a
//! live `gemray-worker serve` process** -- everything it calls (`gemray_net::client`'s
//! handshake/accumulator/session logic, `gemray_net::tls`'s config building,
//! `crate::settings::WorkerSettings`) is already tested in isolation against
//! in-memory buffers; this module's only remaining job is wiring those tested pieces
//! to a real `TcpStream`. It compiles and its pure helpers (below) are unit-tested,
//! but the socket-owning driver itself is exercised only by the user actually running
//! a worker and a viewer against each other -- see the top-level task's own
//! instruction that the GUI must not be launched for verification here.
//!
//! # Why one thread owns both directions
//!
//! TLS record state (`rustls::ClientConnection`) is not safely readable and writable
//! from two threads at once the way a raw `TcpStream::try_clone` split would be for
//! plaintext -- both directions mutate the same connection state. So, mirroring
//! `apps/gemray-worker/src/stream_emit.rs`'s own emitter (one thread, a short socket
//! read timeout, polling an inbound channel between reads -- see that module's doc
//! comment on `TimeoutRead`), [`spawn_remote_render`]'s worker thread is the sole
//! owner of the stream for the whole request: it alternates between a short,
//! timeout-bounded attempt to read the next [`gemray_net::messages::StreamEvent`] and
//! a non-blocking check of its inbound [`RemoteCommand`] channel, so a `CANCEL` can be
//! written promptly without a second thread ever touching the same connection.

use crate::settings::WorkerSettings;
use gemray_net::{
    SceneState,
    client::{Accumulator, ApplyOutcome, ClientError, ConnectionInfo},
    framing::{self, LEN_PREFIX_BYTES, MAX_FRAME_LEN},
    messages::{NetError, RenderRequest, StreamEvent},
    tls::TlsError,
};
use std::{
    fmt,
    io::Read,
    net::TcpStream,
    sync::{
        Arc, Mutex, PoisonError,
        mpsc::{self, TryRecvError},
    },
    thread,
    time::Duration,
};

/// How often [`run`]'s loop checks for a pending [`RemoteCommand`] between read
/// attempts. Purely a responsiveness/CPU trade-off, not a protocol requirement.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

pub type RemoteStream = rustls::StreamOwned<rustls::ClientConnection, TcpStream>;

/// One decoded [`StreamEvent`] paired with its raw payload frame, where it has one
/// (`Frame`/`Preview`) -- the return shape of [`try_read_stream_event`], factored into
/// a named type purely to keep that function's signature simple.
type StreamEventFrame = (StreamEvent, Option<Vec<u8>>);

/// Everything that can go wrong establishing or running a remote render, folded into
/// one type so [`spawn_remote_render`]'s worker thread has a single error path to
/// report through [`RemoteUpdate::Failed`].
#[derive(Debug)]
pub enum RemoteError {
    Tls(TlsError),
    Io(std::io::Error),
    Client(ClientError),
    /// `WorkerSettings::address`'s host portion isn't a valid TLS server name (e.g.
    /// empty, or not parseable as a hostname/IP).
    InvalidServerName(String),
    /// The worker connected and authenticated fine but advertises no render capacity
    /// (`Welcome::render` is `None`) -- it is a library-only server, built without its
    /// `worker` feature. Distinct from every other variant here on purpose: nothing is
    /// broken, the operator simply pointed the viewer at a server that does not render,
    /// and telling them that is more useful than a generic connection failure.
    NoRenderCapacity,
}

impl fmt::Display for RemoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tls(e) => write!(f, "TLS error: {e}"),
            Self::Io(e) => write!(f, "connection error: {e}"),
            Self::Client(e) => write!(f, "{e}"),
            Self::InvalidServerName(host) => write!(f, "not a valid worker hostname: {host:?}"),
            Self::NoRenderCapacity => write!(
                f,
                "this worker serves the design library but cannot render -- it was built                  without its `worker` feature"
            ),
        }
    }
}

impl From<TlsError> for RemoteError {
    fn from(e: TlsError) -> Self {
        Self::Tls(e)
    }
}
impl From<std::io::Error> for RemoteError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<ClientError> for RemoteError {
    fn from(e: ClientError) -> Self {
        Self::Client(e)
    }
}
impl From<NetError> for RemoteError {
    fn from(e: NetError) -> Self {
        Self::Client(e.into())
    }
}

/// A command the UI/orchestrator side sends to the running worker thread -- see the
/// module doc comment on why this goes through a channel rather than a second thread
/// touching the socket directly.
pub enum RemoteCommand {
    Cancel,
}

/// One update surfaced to the caller's `on_update` callback as a remote render
/// progresses. Deliberately does not carry the accumulated buffer itself -- the caller
/// already shares the same `Arc<Mutex<Accumulator>>` passed to
/// [`spawn_remote_render`], and reads it directly (under its own lock) whenever it
/// wants to redraw; this just says WHEN something changed.
#[derive(Debug, Clone)]
pub enum RemoteUpdate {
    /// The handshake completed; streaming is about to begin. Corresponds to
    /// `bridge::handoff::HandoffEvent::RemoteStreamStarted`.
    Connected(ConnectionInfo),
    Frame {
        samples_done: u32,
    },
    Preview,
    Progress {
        samples_done: u32,
    },
    Done {
        cancelled: bool,
    },
    /// The attempt failed for any reason -- connection, handshake, or a transport
    /// error mid-stream. Corresponds to `bridge::handoff::HandoffEvent::RemoteFailed`.
    Failed(String),
}

/// Handle returned by [`spawn_remote_render`]. Cancelling is fire-and-forget: send
/// [`RemoteCommand::Cancel`] and let the worker thread's own `DONE { cancelled: true }`
/// (surfaced via `on_update`) confirm it, matching
/// `bridge::export_thread::ExportHandle`'s existing cooperative-cancellation shape in
/// this same crate.
pub struct RemoteRenderHandle {
    commands: mpsc::Sender<RemoteCommand>,
}

impl RemoteRenderHandle {
    pub fn cancel(&self) {
        let _ = self.commands.send(RemoteCommand::Cancel);
    }
}

/// Extracts the host portion of a `host:port` address (whatever follows the LAST `:`
/// is treated as the port -- tolerant of a bare IPv6 literal without brackets not being
/// supported, which matches this app's own `WorkerSettings::address` convention of a
/// plain `host:port` string, not a bracketed URI authority). Falls back to the whole
/// string if there's no `:` at all, so a malformed address still produces SOME
/// `ServerName` attempt (and a clear TLS-layer error) rather than silently doing
/// nothing.
#[must_use]
fn host_from_address(address: &str) -> &str {
    address.rsplit_once(':').map_or(address, |(host, _)| host)
}

/// Everything [`spawn_remote_render`] needs to know about the ONE `RenderRequest` it
/// will send, grouped into a struct purely to keep that function's (and [`run`]'s) own
/// parameter list short -- see [`WorkerSettings`]'s own doc comment for why `width`/
/// `height` travel alongside `worker` here rather than living on `WorkerSettings`
/// itself (render resolution is session-wide, not per-worker).
pub struct RemoteRenderRequest {
    pub worker: WorkerSettings,
    pub request_id: u32,
    pub scene: SceneState,
    pub first_sample: u32,
    pub samples: u32,
    pub width: u32,
    pub height: u32,
}

/// Connects to `request.worker`, performs the mutual-TLS handshake and
/// `HELLO`/`WELCOME`, then sends and streams one `RenderRequest` covering
/// `[request.first_sample, request.first_sample + request.samples)` of
/// `request.scene` at the session's `request.width x request.height` resolution.
///
/// `accumulator` is shared with the caller: [`Accumulator::begin_request`] is called
/// here (synchronously, right before the request is sent -- see
/// `gemray_net::client::session`'s module docs on why that ordering matters) and every
/// reply is applied into it as it arrives, under a short-held lock each time so the
/// caller can read a consistent snapshot from another thread at any point.
pub fn spawn_remote_render(
    request: RemoteRenderRequest,
    accumulator: Arc<Mutex<Accumulator>>,
    mut on_update: impl FnMut(RemoteUpdate) + Send + 'static,
) -> RemoteRenderHandle {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let result = run(&request, &accumulator, &rx, &mut on_update);
        if let Err(e) = result {
            on_update(RemoteUpdate::Failed(e.to_string()));
        }
    });

    RemoteRenderHandle { commands: tx }
}

/// # Known cost: no connection reuse across requests
///
/// [`spawn_remote_render`] calls this on every dispatch, and `gui::remote::orchestrator::
/// start_remote_render` calls `spawn_remote_render` on every camera settle -- so a live
/// session that drags-then-pauses repeatedly re-pays a full mutual-TLS handshake
/// (`load_ca`/`load_certs`/`load_private_key` re-reading three files from disk, a fresh
/// `rustls::ClientConfig`, a new `TcpStream::connect`, and this function's own
/// `complete_io` -- TCP's own handshake plus TLS 1.3's, roughly 2 network round trips)
/// EVERY settle, not just once per session. `gemray_net::messages::Cancel`'s own doc comment
/// already explains why a socket DROP mid-render is unacceptable for this reason; this
/// is the same cost, paid preemptively, on every settle rather than only a cancellation.
/// On localhost/LAN this is a few milliseconds, lost in the noise next to the render
/// itself; on a higher-latency link it is tens to low-hundreds of milliseconds -- a real,
/// user-visible delay before the FIRST `FRAME` of a settle's remote contribution can
/// possibly arrive.
///
/// Reusing one persistent connection across MULTIPLE settles would need
/// [`spawn_remote_render`]'s worker thread to outlive a single request -- accepting a
/// stream of `RenderRequest`s over an internal channel instead of being spawned fresh
/// per request and exiting after `DONE` -- which changes this module's whole threading
/// model (today: one thread, one connection, one request, exit) and needs new lifecycle
/// wiring in `gui::remote::orchestrator` (holding a handle across settles, tearing it
/// down on window close/settings change/worker-address edit) and in
/// `bridge::export_thread::remote` (whose own per-dispatch `spawn_remote_render` calls
/// would want the same treatment for a multi-batch export). That is a larger,
/// cross-cutting change than the combined-rendering feature this module currently
/// supports absorbed -- left as future work rather than attempted here; see
/// `gemray_net::messages::Cancel`'s doc comment for the wire-protocol piece (`CANCEL` without a
/// disconnect) that already exists specifically to make a persistent connection safe to
/// build on top of, once someone does.
///
/// Connects to `worker.address` over mutual TLS (certificates loaded from
/// `worker.cert_dir` -- see [`WorkerSettings::ca_path`]/[`WorkerSettings::client_cert_path`]/
/// [`WorkerSettings::client_key_path`]) and performs the `HELLO`/`WELCOME` handshake,
/// including [`gemray_net::handshake::verify_compatible`]'s client-side defense-in-depth
/// check (see `gemray_net::client::handshake`'s own doc comment). Shared by [`run`] (a
/// full render) and [`test_connection`] (handshake only, no render) -- and, `pub(crate)`,
/// by `bridge::library_client` (the same mutual-TLS connect+handshake serves the
/// read-only design-library protocol too, since both ride the same `ClientMessage`
/// envelope over the same authenticated connection -- see `gemray_net::library`'s module
/// doc comment) -- so every caller connects exactly the same way.
///
/// # Errors
///
/// See [`RemoteError`]'s variants.
pub fn connect_and_handshake(
    worker: &WorkerSettings,
) -> Result<(RemoteStream, gemray_net::messages::Welcome), RemoteError> {
    let ca = gemray_net::tls::load_ca(&worker.ca_path())?;
    let cert_chain = gemray_net::tls::load_certs(&worker.client_cert_path())?;
    let key = gemray_net::tls::load_private_key(&worker.client_key_path())?;
    let config = gemray_net::tls::client_config(ca, cert_chain, key)?;

    let tcp = TcpStream::connect(&worker.address)?;
    let host = host_from_address(&worker.address);
    let server_name = rustls::pki_types::ServerName::try_from(host.to_string())
        .map_err(|_| RemoteError::InvalidServerName(host.to_string()))?;
    let conn = rustls::ClientConnection::new(config, server_name).map_err(TlsError::Rustls)?;
    let mut stream = rustls::StreamOwned::new(conn, tcp);
    // Force the handshake to complete now, mirroring
    // `apps/gemray-worker/src/serve.rs::accept_tls` on the server side, so a TLS
    // failure (wrong CA, expired cert, clock skew, SAN mismatch) is diagnosed here
    // rather than surfacing later as an opaque I/O error out of the handshake below.
    stream
        .conn
        .complete_io(&mut stream.sock)
        .map_err(RemoteError::Io)?;

    let welcome = gemray_net::client::handshake::handshake(&mut stream)?;
    Ok((stream, welcome))
}

/// The settings UI's "Test connection" operation: connect, handshake, report worker
/// identity/backend/build compatibility, then disconnect (simply by dropping the
/// stream when this returns) -- no render. Blocking; callers run this on their own
/// worker thread (see `gui::remote::setup_worker_callbacks`) and report the result back
/// to the UI thread themselves.
///
/// # Errors
///
/// See [`RemoteError`]'s variants.
pub fn test_connection(worker: &WorkerSettings) -> Result<ConnectionInfo, RemoteError> {
    let (_stream, welcome) = connect_and_handshake(worker)?;
    Ok(welcome.into())
}

fn run(
    request: &RemoteRenderRequest,
    accumulator: &Arc<Mutex<Accumulator>>,
    commands: &mpsc::Receiver<RemoteCommand>,
    on_update: &mut dyn FnMut(RemoteUpdate),
) -> Result<(), RemoteError> {
    let request_id = request.request_id;
    let (mut stream, welcome) = connect_and_handshake(&request.worker)?;
    on_update(RemoteUpdate::Connected(welcome.clone().into()));

    // Ask before sending: the handshake advertises render capacity precisely so a client
    // never discovers its absence by having a `RenderRequest` rejected downstream.
    let Some(capability) = welcome.render.as_ref() else {
        return Err(RemoteError::NoRenderCapacity);
    };

    let render_request = RenderRequest {
        request_id,
        scene: request.scene.clone(),
        first_sample: request.first_sample,
        samples: request.samples,
        stream: request.worker.stream_config(
            capability.min_cadence_ms,
            request.width,
            request.height,
        ),
    };

    {
        let mut acc = accumulator.lock().unwrap_or_else(PoisonError::into_inner);
        acc.begin_request(request_id);
    }
    gemray_net::client::send_render_request(&mut stream, &render_request)?;

    loop {
        match commands.try_recv() {
            Ok(RemoteCommand::Cancel) => {
                gemray_net::client::send_cancel(&mut stream, request_id)?;
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                // `Disconnected` means the handle was dropped -- nobody can ever
                // cancel or observe this render again; nothing further to do but keep
                // draining until DONE so the connection ends cleanly rather than being
                // torn down mid-message.
            }
        }

        let Some((event, payload)) = try_read_stream_event(&mut stream, POLL_INTERVAL)? else {
            continue;
        };

        let outcome = {
            let mut acc = accumulator.lock().unwrap_or_else(PoisonError::into_inner);
            acc.apply(&event, payload.as_deref())
                .map_err(|e| RemoteError::Client(e.into()))?
        };

        let is_terminal = matches!(
            outcome,
            ApplyOutcome::Done { .. } | ApplyOutcome::WorkerError
        );
        if let Some(update) = to_remote_update(&event, outcome) {
            on_update(update);
        }
        if is_terminal {
            return Ok(());
        }
    }
}

fn to_remote_update(event: &StreamEvent, outcome: ApplyOutcome) -> Option<RemoteUpdate> {
    match outcome {
        ApplyOutcome::FrameSummed { samples_done } => Some(RemoteUpdate::Frame { samples_done }),
        ApplyOutcome::PreviewReplaced => Some(RemoteUpdate::Preview),
        ApplyOutcome::Progress { samples_done } => Some(RemoteUpdate::Progress { samples_done }),
        ApplyOutcome::Done { cancelled } => Some(RemoteUpdate::Done { cancelled }),
        ApplyOutcome::WorkerError => {
            let StreamEvent::Error(e) = event else {
                unreachable!("WorkerError only ever comes from applying an Error event")
            };
            Some(RemoteUpdate::Failed(e.message.clone()))
        }
        ApplyOutcome::StaleDropped => None,
    }
}

/// Attempts to read one length-prefixed frame from `stream`, tolerating a
/// `WouldBlock`/`TimedOut` I/O error (from the short read timeout this sets) as
/// "nothing pending yet" rather than a fatal error -- mirrors
/// `apps/gemray-worker/src/stream_emit.rs::poll_for_client_message`'s own approach on
/// the server side (see that function's doc comment for why the timeout can only ever
/// land BEFORE any byte of a new frame arrives, never mid-frame: once even one byte has
/// been observed, this commits to blocking for the rest of it).
fn try_read_one_frame(
    stream: &mut RemoteStream,
    poll_timeout: Duration,
) -> Result<Option<Vec<u8>>, NetError> {
    stream
        .sock
        .set_read_timeout(Some(poll_timeout))
        .map_err(|e| NetError::Framing(framing::FramingError::Io(e)))?;

    let mut len_bytes = [0u8; LEN_PREFIX_BYTES];
    let n = match stream.read(&mut len_bytes) {
        Ok(0) => {
            return Err(NetError::Framing(framing::FramingError::Io(
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "worker closed the connection",
                ),
            )));
        }
        Ok(n) => n,
        Err(e)
            if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut =>
        {
            return Ok(None);
        }
        Err(e) => return Err(NetError::Framing(framing::FramingError::Io(e))),
    };

    stream
        .sock
        .set_read_timeout(None)
        .map_err(|e| NetError::Framing(framing::FramingError::Io(e)))?;
    if n < len_bytes.len() {
        stream
            .read_exact(&mut len_bytes[n..])
            .map_err(|e| NetError::Framing(framing::FramingError::Io(e)))?;
    }
    let len = u32::from_le_bytes(len_bytes);
    if len > MAX_FRAME_LEN {
        return Err(NetError::Framing(framing::FramingError::FrameTooLarge {
            len,
            max: MAX_FRAME_LEN,
        }));
    }
    let mut payload = vec![0u8; len as usize];
    stream
        .read_exact(&mut payload)
        .map_err(|e| NetError::Framing(framing::FramingError::Io(e)))?;
    Ok(Some(payload))
}

/// The timeout-tolerant equivalent of [`gemray_net::messages::read_stream_event`]:
/// `Ok(None)` means "nothing arrived within `poll_timeout`", not an error -- lets
/// [`run`]'s loop interleave checking for a [`RemoteCommand`] between read attempts
/// without a second thread ever touching `stream`. See the module doc comment.
fn try_read_stream_event(
    stream: &mut RemoteStream,
    poll_timeout: Duration,
) -> Result<Option<StreamEventFrame>, NetError> {
    let Some(header_bytes) = try_read_one_frame(stream, poll_timeout)? else {
        return Ok(None);
    };
    let event: StreamEvent = postcard::from_bytes(&header_bytes)?;
    let expected_len = match &event {
        StreamEvent::Frame(h) => Some(h.payload_len),
        StreamEvent::Preview(h) => Some(h.payload_len),
        StreamEvent::Progress(_) | StreamEvent::Done(_) | StreamEvent::Error(_) => None,
    };
    let payload = match expected_len {
        Some(expected) => {
            let bytes = framing::read_frame(stream).map_err(NetError::Framing)?;
            if bytes.len() as u32 != expected {
                return Err(NetError::FramePayloadLenMismatch {
                    declared: expected,
                    actual: bytes.len(),
                });
            }
            Some(bytes)
        }
        None => None,
    };
    Ok(Some((event, payload)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_from_address_strips_the_trailing_port() {
        assert_eq!(host_from_address("worker.local:9443"), "worker.local");
        assert_eq!(host_from_address("192.168.1.50:9443"), "192.168.1.50");
    }

    #[test]
    fn host_from_address_falls_back_to_the_whole_string_without_a_colon() {
        assert_eq!(host_from_address("worker.local"), "worker.local");
        assert_eq!(host_from_address(""), "");
    }

    #[test]
    fn remote_error_display_is_human_readable() {
        let e = RemoteError::InvalidServerName("bad host".to_string());
        assert!(e.to_string().contains("bad host"));
    }
}

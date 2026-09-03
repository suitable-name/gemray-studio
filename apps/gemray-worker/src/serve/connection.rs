//! Per-connection request handling: the `HELLO`/`WELCOME` handshake and the loop
//! dispatching whatever the peer sends -- `ClientMessage::Library` always, and (only on
//! a `worker` build) `ClientMessage::RenderRequest` too -- until the peer closes the
//! connection. See `crate::serve`'s module docs for the full architecture this
//! participates in.

use diagram_catalog::db::sqlite::Database;
use gemray_net::messages::{ClientMessage, NetError};
#[cfg(not(feature = "worker"))]
use gemray_net::messages::{ErrorMsg, Hello, PROTOCOL_VERSION, Welcome};
#[cfg(feature = "worker")]
use std::io::Write;
#[cfg(not(feature = "worker"))]
use std::io::{Read, Write};
use std::net::SocketAddr;

pub(super) fn report_connection_result(
    peer: Option<SocketAddr>,
    result: std::thread::Result<Result<(), NetError>>,
) {
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!("connection {peer:?} ended with an error: {e}"),
        Err(_) => tracing::warn!("connection {peer:?} panicked and was dropped"),
    }
}

/// `<- ERROR` code for a `HELLO` this worker refuses to pair with -- a protocol-version
/// mismatch (library-only builds) or (also) a `gemray` build-hash mismatch (`worker`
/// builds -- see [`handle_connection_with_gpu`]).
pub(super) const BUILD_MISMATCH_CODE: u32 = 1;

/// `<- ERROR` code for a `RenderRequest` arriving at a worker with no render capacity.
/// Distinct from [`BUILD_MISMATCH_CODE`] on purpose: that one means "we cannot pair at
/// all", this one means "we paired fine, but I only serve the library" -- a client can
/// act on the difference (fall back to local rendering rather than dropping the worker).
pub(super) const NO_RENDER_CAPACITY_CODE: u32 = 2;

/// Reads one [`ClientMessage`] request, dispatching `ClientMessage::Library` to
/// [`super::library::handle_request`] and writing its reply directly (a plain
/// request/response, never a `StreamEvent` stream -- see `gemray_net::library`'s module
/// docs). Returns `Ok(false)` for a `Cancel` (nothing to cancel outside a render
/// stream -- logged and otherwise ignored, matching the render path's own
/// "nothing currently streaming" handling) so the caller's loop can just keep reading,
/// and `Ok(true)` once a `Library` reply has been written. Shared by both build modes.
///
/// # Errors
///
/// Returns [`NetError`] for a transport-level failure.
#[allow(
    unreachable_patterns,
    clippy::match_wildcard_for_single_variants,
    reason = "the final arm must stay a wildcard: it matches ClientMessage::RenderRequest only \
              when gemray-net's `render` feature is on, and this crate can neither name that \
              variant otherwise nor detect another crate's feature. Spelling the variant out, \
              as clippy suggests, fails to compile in a library-only build; the wildcard is \
              correct in both. Kept as `allow` (not `expect`) because firing is conditional on \
              that feature -- unconditionally expecting it would warn on every build where the \
              variant IS nameable."
)]
fn handle_non_render_message<S: Write>(
    stream: &mut S,
    msg: &ClientMessage,
    db: &Database,
) -> Result<(), NetError> {
    match msg {
        ClientMessage::Cancel(c) => {
            tracing::debug!(
                "received CANCEL for request_id={} with no request currently streaming on this connection -- ignoring",
                c.request_id
            );
            Ok(())
        }
        ClientMessage::Library(req) => {
            let response = super::library::handle_request(req, db);
            gemray_net::messages::write_message(stream, &response)
        }
        // NOT gated on this crate's `worker` feature, though it once was. The variant is
        // gated on `gemray-net`'s `render` feature, and those are different flags on
        // different crates: `worker` turns `render` on, but any OTHER crate can turn it
        // on too (`apps/diagram-gui` does, being a render client), and cargo unifies
        // features across a workspace build. So a library-only worker can genuinely be
        // compiled with this variant in scope, and the match must cover it.
        //
        // It is also reachable at RUNTIME, which is the more important reason. `WELCOME`
        // advertises `render: None` on a library-only server, but nothing forces a peer
        // to respect that -- a buggy or hostile client can send a `RenderRequest`
        // anyway. Answering with a protocol error is correct; the `unreachable!()` this
        // replaces would have panicked a connection thread on peer-controlled input.
        _ => {
            tracing::warn!(
                "received a RenderRequest, but this worker advertises no render capacity                  (built without its `worker` feature) -- replying with a protocol error"
            );
            gemray_net::messages::write_message(
                stream,
                &gemray_net::messages::StreamEvent::Error(gemray_net::messages::ErrorMsg {
                    code: NO_RENDER_CAPACITY_CODE,
                    message: "this worker serves the design library only and cannot render;                               its WELCOME advertises render capacity as absent"
                        .to_string(),
                }),
            )
        }
    }
}

/// Reads and dispatches the peer's next post-handshake message directly off `stream`
/// (a blocking read). Every `Cancel`/`Library` message is handled here (via
/// [`handle_non_render_message`]) and the loop continues; only a `RenderRequest`
/// (`worker` builds only) is handed back to the caller to drive, and a clean EOF ends
/// the connection.
///
/// # Errors
///
/// Returns [`NetError`] for a transport-level failure. `Ok(None)` (not an error) for a
/// clean EOF -- the peer closing the connection, the normal end of the caller's loop.
#[cfg(not(feature = "worker"))]
fn serve_until_render_request_or_eof<S: Read + Write>(
    stream: &mut S,
    db: &Database,
) -> Result<Option<std::convert::Infallible>, NetError> {
    loop {
        let msg: ClientMessage = match gemray_net::messages::read_message(stream) {
            Ok(m) => m,
            Err(NetError::Framing(gemray_net::framing::FramingError::Io(e)))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        handle_non_render_message(stream, &msg, db)?;
    }
}

/// Handles one connection end to end on a library-only build (no `worker` feature).
///
/// `HELLO`/`WELCOME` (`WELCOME::render` always `None` -- this build has no render
/// capacity to advertise, so no gemray build-compatibility check runs either; see
/// `gemray_net::handshake`'s module docs on why that check is skipped entirely rather
/// than run against a placeholder), then a loop dispatching `Cancel`/`Library` messages
/// until the peer closes the connection.
///
/// Only a `HELLO` [`Welcome::protocol_version`] mismatch is refused -- the one
/// wire-format compatibility question that's still meaningful without any `gemray`
/// build to compare (see the crate docs on why the full build-hash gate is a `worker`-
/// only concern).
///
/// # Errors
///
/// See [`handle_connection_with_gpu`] (the `worker`-build counterpart) for the shape of
/// errors this can return; the failure modes are the same, minus anything render-
/// specific.
#[cfg(not(feature = "worker"))]
pub fn handle_connection<S: Read + Write>(mut stream: S, db: &Database) -> Result<(), NetError> {
    let remote_hello: Hello = gemray_net::messages::read_message(&mut stream)?;

    if remote_hello.protocol_version != PROTOCOL_VERSION {
        let message = format!(
            "refusing to pair: this worker speaks protocol_version={PROTOCOL_VERSION}, peer speaks \
             protocol_version={}",
            remote_hello.protocol_version
        );
        tracing::warn!("{message}");
        let _ = gemray_net::messages::write_message(
            &mut stream,
            &ErrorMsg {
                code: BUILD_MISMATCH_CODE,
                message,
            },
        );
        return Ok(());
    }

    let welcome = Welcome {
        protocol_version: PROTOCOL_VERSION,
        build_hash: gemray_net::handshake::UNKNOWN_BUILD_HASH,
        render: None,
        library: true,
    };
    gemray_net::messages::write_message(&mut stream, &welcome)?;

    serve_until_render_request_or_eof(&mut stream, db)?.map_or(Ok(()), |never| match never {})
}

#[cfg(feature = "worker")]
mod worker {
    use super::{BUILD_MISMATCH_CODE, Database, handle_non_render_message};
    use crate::{
        render_core,
        stream_emit::{self, StreamOutcome, TimeoutRead},
        validate,
    };
    use gemray::renderer::gpu_backend::GpuBackend;
    use gemray_net::{
        framing::FramingError,
        handshake,
        messages::{
            Backend, ClientMessage, ErrorMsg, Hello, NetError, PROTOCOL_VERSION, RenderCapability,
            RenderRequest, StreamEvent, Welcome,
        },
    };
    use std::{
        io::{Read, Write},
        sync::Arc,
    };

    /// Handles one connection end to end, always on the CPU.
    ///
    /// A thin wrapper around [`handle_connection_with_gpu`] that passes
    /// [`GpuBackend::disabled`], so this crate's own tests (which call this, not the
    /// `_with_gpu` form) stay exactly as deterministic under `cargo test --features gpu`
    /// as under plain `cargo test`, tracing on the CPU reference path every test already
    /// assumes rather than whatever real adapter happens to be on the test machine.
    /// `crate::serve::run`'s accept loop calls [`handle_connection_with_gpu`] directly,
    /// with the real, shared [`GpuBackend`] it acquired at startup.
    ///
    /// # Errors
    ///
    /// See [`handle_connection_with_gpu`].
    pub fn handle_connection<S: Read + Write + TimeoutRead>(
        stream: S,
        threads: usize,
        db: &Database,
    ) -> Result<(), NetError> {
        handle_connection_with_gpu(stream, threads, &Arc::new(GpuBackend::disabled()), db)
    }

    /// Handles one connection end to end.
    ///
    /// First the `HELLO`/`WELCOME` handshake (refusing on a build-hash or protocol-
    /// version mismatch, per [`handshake::verify_compatible`]) -- this is the ONE code
    /// path in this crate that still runs the full gemray build-compatibility gate; a
    /// library-only build's [`super::handle_connection`] never does, since it has no
    /// gemray build to compare (see that function's own doc comment). Then a loop
    /// dispatching `Cancel`/`Library` (via [`handle_non_render_message`]) and
    /// `RenderRequest` messages until the peer closes the connection. Each
    /// `RenderRequest` normally comes from a fresh (blocking) [`read_next_message`]
    /// call, but one may instead arrive already in hand -- pipelined by the client
    /// ahead of the PREVIOUS request's `DONE` and handed back by
    /// [`stream_emit::run_stream`] -- in which case this loop starts it immediately
    /// rather than reading again; see that function's doc comment.
    ///
    /// Generic over `Read + Write` rather than `TcpStream` specifically -- see
    /// `crate::serve`'s module docs on why, and where a TLS stream substitutes later.
    ///
    /// `gpu` decides both what `WELCOME::render.backend` reports and what actually
    /// traces `RenderRequest`s: [`Backend::Gpu`] is reported iff `gpu.adapter_label()`
    /// is `Some` (an adapter was genuinely acquired at `crate::serve::run` startup, not
    /// merely that the `gpu` feature is compiled in) -- otherwise [`Backend::Cpu`],
    /// matching what this connection will actually trace with. A worker that claims GPU
    /// while silently tracing on CPU is worse than one that never claimed it; this is
    /// decided once, at handshake time, from the SAME acquisition state every request on
    /// this connection will subsequently consult (`gpu` is `Arc`-shared, not
    /// re-acquired per request) -- a later per-request decline (an unsupported
    /// material) still falls back silently, exactly as documented in `gpu_backend`,
    /// since the protocol has no per-request backend field to report it on.
    ///
    /// # Errors
    ///
    /// Returns [`NetError`] for a transport-level failure (a malformed frame, an I/O
    /// error other than the peer cleanly closing the connection). A validation failure
    /// or a caught tracing panic is NOT an error return -- both are reported to the peer
    /// as an `ErrorMsg`/`StreamEvent::Error` and the loop continues, since neither
    /// indicates this connection itself is broken.
    pub fn handle_connection_with_gpu<S: Read + Write + TimeoutRead>(
        mut stream: S,
        threads: usize,
        gpu: &Arc<GpuBackend>,
        db: &Database,
    ) -> Result<(), NetError> {
        let remote_hello: Hello = gemray_net::messages::read_message(&mut stream)?;
        let local_hello = handshake::local_hello();

        if let Err(incompatible) = handshake::verify_compatible(&local_hello, &remote_hello) {
            let message = format!(
                "refusing to pair: worker build_hash={:02x?} protocol_version={}, viewer build_hash={:02x?} \
                 protocol_version={} ({incompatible})",
                local_hello.build_hash,
                local_hello.protocol_version,
                remote_hello.build_hash,
                remote_hello.protocol_version
            );
            tracing::warn!("{message}");
            // Best-effort: the peer may already have hung up, in which case this write
            // failing is not itself worth reporting as this connection's error -- the
            // handshake refusal (already logged above) is the meaningful outcome here.
            let _ = gemray_net::messages::write_message(
                &mut stream,
                &ErrorMsg {
                    code: BUILD_MISMATCH_CODE,
                    message,
                },
            );
            return Ok(());
        }

        let backend = gpu.adapter_label().map_or_else(
            || Backend::Cpu {
                threads: render_core::effective_thread_count(threads) as u32,
            },
            |adapter| Backend::Gpu { adapter },
        );
        let welcome = Welcome {
            protocol_version: PROTOCOL_VERSION,
            build_hash: local_hello.build_hash,
            render: Some(RenderCapability {
                backend,
                max_pixels: validate::MAX_PIXELS,
                min_cadence_ms: stream_emit::MIN_CADENCE_FLOOR_MS,
            }),
            library: true,
        };
        gemray_net::messages::write_message(&mut stream, &welcome)?;

        // A `RenderRequest` [`stream_emit::run_stream`] already pulled off the wire
        // while streaming the PREVIOUS request (pipelined ahead of that request's
        // `DONE` -- see this module's doc comment and `stream_emit::run_stream`'s), to
        // be processed on the next iteration exactly as if it had just been read fresh
        // here. `None` means the next iteration should read one instead.
        let mut pending_request: Option<RenderRequest> = None;

        loop {
            let request: RenderRequest = match pending_request.take() {
                Some(r) => r,
                None => match read_next_message(&mut stream, db)? {
                    Some(r) => r,
                    // The peer closing the connection is the normal end of this loop,
                    // not a failure.
                    None => return Ok(()),
                },
            };

            if let Err(msg) =
                validate::validate_request(&request.scene, request.first_sample, request.samples)
                    .and_then(|()| {
                        validate::validate_stream_config(&request.stream, &request.scene)
                    })
            {
                gemray_net::messages::write_stream_event(
                    &mut stream,
                    &StreamEvent::Error(ErrorMsg {
                        code: VALIDATION_FAILED_CODE,
                        message: msg,
                    }),
                    None,
                )?;
                continue;
            }

            // Runs the tracer on its own thread (never touching `stream` itself) and
            // this thread as the emitter -- see `stream_emit`'s and this module's own
            // docs for the full architecture (why the split exists, delta vs. preview
            // arithmetic, cooperative cancellation via `CANCEL`). Defense in depth:
            // validation above should already reject anything that would make
            // `trace_spectral_ray` misbehave, but a network service accepting
            // caller-supplied geometry can't rely on validation alone catching every
            // pathological case gemray's own physics might not yet defend against --
            // `stream_emit::run_stream`'s tracer thread runs inside `catch_unwind` for
            // exactly that reason, surfaced here as [`StreamOutcome::TracePanicked`].
            let (outcome, next) = stream_emit::run_stream(&mut stream, &request, threads, gpu)?;
            // `next` is the client's already-pipelined `RenderRequest` (if any), which
            // `run_stream` pulled off the wire while `request` was still streaming --
            // see its own doc comment. Queuing it here means the loop's next iteration
            // processes it directly, without blocking on another read.
            pending_request = next;
            match outcome {
                StreamOutcome::Completed => {}
                StreamOutcome::TracePanicked => {
                    tracing::warn!(
                        "tracing panicked for a request that passed validation (request_id={}, first_sample={}, samples={})",
                        request.request_id,
                        request.first_sample,
                        request.samples
                    );
                    gemray_net::messages::write_stream_event(
                        &mut stream,
                        &StreamEvent::Error(ErrorMsg {
                            code: TRACE_PANIC_CODE,
                            message: "internal error while tracing this request".to_string(),
                        }),
                        None,
                    )?;
                }
            }
        }
    }

    /// Reads the next message directly off `stream` -- a blocking read, since no
    /// request is currently streaming when this is called. Dispatches `Cancel`/
    /// `Library` inline (via [`handle_non_render_message`]) and keeps reading; only a
    /// `RenderRequest` ends the wait.
    ///
    /// # Errors
    ///
    /// Returns [`NetError`] for a transport-level failure, same as
    /// [`gemray_net::messages::read_message`]. `Ok(None)` (not an error) for a clean
    /// EOF -- the peer closing the connection, the normal end of
    /// [`handle_connection_with_gpu`]'s loop.
    fn read_next_message<S: Read + Write>(
        stream: &mut S,
        db: &Database,
    ) -> Result<Option<RenderRequest>, NetError> {
        loop {
            let msg: ClientMessage = match gemray_net::messages::read_message(stream) {
                Ok(m) => m,
                Err(NetError::Framing(FramingError::Io(e)))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None);
                }
                Err(e) => return Err(e),
            };
            match msg {
                ClientMessage::RenderRequest(r) => return Ok(Some(*r)),
                other @ (ClientMessage::Cancel(_) | ClientMessage::Library(_)) => {
                    handle_non_render_message(stream, &other, db)?;
                }
            }
        }
    }

    pub(in crate::serve) const VALIDATION_FAILED_CODE: u32 = 2;
    pub(in crate::serve) const TRACE_PANIC_CODE: u32 = 3;
}

#[cfg(all(feature = "worker", test))]
pub(super) use worker::VALIDATION_FAILED_CODE;
#[cfg(feature = "worker")]
pub use worker::{handle_connection, handle_connection_with_gpu};

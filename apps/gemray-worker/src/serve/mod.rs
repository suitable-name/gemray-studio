//! The `serve` subcommand: accept `gemray-net` connections over TCP and reply.
//!
//! Performs mutual TLS (unless `--insecure-no-tls`) and a `HELLO`/`WELCOME` handshake,
//! then loops handling whatever the peer sends. **Every build serves the read-only
//! design-library protocol** ([`library`], backed by [`diagram_catalog`]) -- listing/
//! searching designs, fetching one in full, fetching an attachment's bytes. A build
//! with the `worker` feature additionally serves `RenderRequest`s.
//!
//! # One connection, two possible protocol families
//!
//! After the handshake, every message a client sends is a tagged
//! `gemray_net::messages::ClientMessage` -- `Cancel`, `Library(LibraryRequest)`, and
//! (only on a `worker` build) `RenderRequest`. [`connection::handle_connection`] reads
//! and dispatches these in one loop; `ClientMessage::Library` is handled the same way
//! regardless of `worker`, by the shared [`library::handle_request`]. See that module's
//! doc comment for why this is a plain request/response call (unlike `RenderRequest`,
//! never streamed) and how it reuses the diagram-catalog `Database` this module opens
//! once at startup.
//!
//! **The library protocol sits behind the exact same authentication as the render
//! protocol -- there is no separate check.** Both are just message variants read off
//! the SAME already-authenticated stream, after the SAME mutual-TLS handshake and the
//! SAME `Auth::Allowlist`/`Auth::AnyCaSignedClient` decision in [`accept_tls`]. This is
//! deliberate, not an oversight: inventing a second access-control path for the library
//! protocol would be exactly the kind of duplicated security policy this workspace has
//! already burned itself on once (see `gemray_net::enroll`'s module doc comment on
//! `PinnedCaVerifier` not being copied into `apps/diagram-gui`) -- reusing the existing,
//! tested, threat-modelled mechanism (`apps/gemray-worker/docs/security.md`) is what
//! keeps the catalogue access-controlled by construction rather than by discipline.
//!
//! # Honest capability advertisement
//!
//! `WELCOME` (`gemray_net::messages::Welcome`) carries `render: Option<RenderCapability>`
//! (`Some` only on a `worker` build that actually acquired -- or was told to use --
//! compute capacity) and `library: bool` (always `true` in this phase). A client checks
//! `render.is_some()` BEFORE ever sending a `RenderRequest` -- on a library-only build,
//! that message doesn't even exist on the wire (see `ClientMessage`'s own doc comment),
//! so sending one anyway fails to decode rather than being gracefully refused.
//!
//! # Render-specific behavior (only under `worker`)
//!
//! Still accurate for everything RENDER-shaped, now conditionally compiled:
//!
//! - **Progressive streaming, delta/preview arithmetic, cancellation, pipelining**: see
//!   [`crate::stream_emit`]'s module docs.
//! - **The tracer never touches the socket**; a separate emitter thread owns the stream.
//! - **The worker never denoises.**
//!
//! # TLS
//!
//! [`connection::handle_connection`] (library-only builds) / [`connection::handle_connection_with_gpu`]
//! (`worker` builds) are generic over `Read + Write` and never name `TcpStream` in their
//! own signature -- only this module's accept loop (and the TLS-specific helpers below
//! it) know about sockets at all. `rustls::StreamOwned<ServerConnection, TcpStream>`
//! also implements `Read + Write`, so wrapping an accepted `TcpStream` in one at
//! [`accept_tls`] (called from the accept loop) is all that's needed -- nothing in the
//! connection handler changes.
//!
//! Mutual TLS answers "may this peer talk to me at all" -- a different question from
//! `gemray_net::handshake::verify_compatible`'s "are we running the same `gemray`
//! physics" (itself only run on a `worker` build against a peer that also claims render
//! capacity -- see `connection`'s module doc comment). Neither check substitutes for
//! the other.
//!
//! TLS validates the certificate CHAIN (was this peer's certificate signed by the CA
//! this worker trusts) but says nothing about which specific signed peer to trust --
//! anyone who gets a certificate signed by the CA can complete the handshake. The
//! [`Auth::Allowlist`] check in [`accept_tls`], run against
//! `ServerConnection::peer_certificates()` immediately after a successful handshake, is
//! the actual authorization decision (see `gemray_net::tls`'s doc comment: there is no
//! CRL or OCSP here, just this fingerprint list). `--trust-any-client-cert` skips that
//! check -- see [`Auth::AnyCaSignedClient`] and [`run`]'s doc comment on why it has to
//! be an explicit flag.
//!
//! `--insecure-no-tls` skips TLS entirely: refused outright on a non-loopback `--bind`
//! (see [`run`]), and every connection accepted this way logs a warning (see the accept
//! loop below) so a plaintext listener can never go unnoticed in the logs.
//!
//! # Robustness
//!
//! - Every `RenderRequest` is validated (`worker` builds only) before a single sample is
//!   traced -- rejected requests get a `StreamEvent::Error` reply, not a dropped
//!   connection. Every `LibraryRequest` is handled defensively too -- an unknown id gets
//!   `LibraryResponse::NotFound`, a database error gets `LibraryResponse::Error`,
//!   neither drops the connection.
//! - Tracing itself runs inside `std::panic::catch_unwind` (`worker` builds; inside
//!   `stream_emit`'s tracer thread), so a `gemray` panic on some unanticipated
//!   pathological (but validation-passing) input can't take the whole process down.
//!   [`run`]'s accept loop wraps the entire per-connection handler in a second
//!   `catch_unwind` as a further backstop (a panic anywhere in handshake parsing or
//!   library-request handling, not just tracing, still can't take down the listener
//!   thread).
//! - The listener binds to loopback only unless the caller passes `--allow-remote` --
//!   see [`run`]. This is independent of TLS: a non-loopback bind is opt-in either way.
//! - TLS and certificate authentication answer "may this peer talk to me", not "is
//!   whatever this peer asks for reasonable" -- an authenticated client can still ask
//!   for an absurd render or an absurd search. `crate::validate`'s caps bound the
//!   former (`worker` builds); `diagram_catalog::db::sqlite::Database::search_diagrams`'s
//!   own 1000-row cap bounds the latter, unconditionally.

use std::{
    net::{SocketAddr, TcpListener, TcpStream},
    sync::Arc,
    thread,
};

use diagram_catalog::db::sqlite::Database;

use crate::{cli::ServeArgs, enroll};

mod connection;
pub mod library;
mod tls;
// Every test in this module drives a real `RenderRequest`/`WELCOME::render` round
// trip -- meaningless (and, since `RenderRequest`/`Backend`/etc. don't even exist in
// `gemray_net::messages` without its own `render` feature, uncompilable) on a
// library-only build. `crate::serve::library`'s own tests cover the library protocol,
// unconditionally.
#[cfg(all(test, feature = "worker"))]
mod tests;

use connection::report_connection_result;
use tls::{Auth, Transport, accept_tls, build_transport};

#[cfg(not(feature = "worker"))]
pub use connection::handle_connection;
#[cfg(feature = "worker")]
pub use connection::{handle_connection, handle_connection_with_gpu};

/// Opens the design-library database this `serve` instance serves: `args.db` if given,
/// else the default `facet_diagrams.sqlite` resolved relative to the process's working
/// directory -- see [`crate::cli::ServeArgs::db`]'s own doc comment for why that default
/// path is unchanged from before `--db` existed.
///
/// Always opened READ-ONLY (`Database::open_read_only`) -- this phase never writes to
/// the catalogue (see `library`'s module docs), so there is no reason for `serve` to
/// hold a writable handle to it at all. Unlike the historical `Database::new` default
/// (still used elsewhere in this workspace, e.g. by the local-library GUI, which
/// legitimately writes), this means `serve` no longer silently creates an empty
/// database at the default path if one doesn't already exist there -- it fails fast
/// with a clear error instead, which is the right behavior for a server: provision the
/// catalogue first, don't let it start pointed at nothing.
///
/// A FRESH connection every call, deliberately -- see [`resolve_library_db_path`]'s doc
/// comment on why this is called once per connection (from each connection's own
/// thread) rather than opened once in [`run`] and shared: `rusqlite::Connection` is
/// `Send` but not `Sync`, and SQLite's own concurrency model is many independent reader
/// connections, not one connection multiplexed across threads.
///
/// # Errors
///
/// A human-readable message if the database can't be opened read-only (missing file,
/// permissions, not a valid SQLite database).
fn open_library_database(path: &std::path::Path) -> Result<Database, String> {
    let path_str = path
        .to_str()
        .ok_or_else(|| format!("--db {}: path is not valid Unicode", path.display()))?;
    Database::open_read_only(path_str).map_err(|e| {
        format!(
            "failed to open the design-library database read-only at {}: {e:#}",
            path.display()
        )
    })
}

/// Resolves `--db` (or the default `facet_diagrams.sqlite`, relative to the process's
/// working directory) to the path every connection thread opens its own
/// [`open_library_database`] handle from.
///
/// A path, not an already-open [`Database`], is what [`run`] hands to each connection
/// thread: SQLite's concurrency model is many independent reader connections against
/// the same file (which is what read-only, `WAL`-or-rollback-journal-mode SQLite is
/// designed for), not one connection shared across threads -- and
/// `rusqlite::Connection` isn't `Sync`, so sharing one via `Arc` the way
/// `gemray::renderer::gpu_backend::GpuBackend` is shared below wouldn't even compile.
/// [`run`] still opens (and immediately drops) one [`Database`] at startup via this
/// path purely to fail fast on a bad `--db` before ever binding a socket, exactly the
/// same "validate once at startup, re-check per use" split `crate::serve::tls`'s own
/// allowlist preflight already uses.
#[must_use]
fn resolve_library_db_path(args: &ServeArgs) -> std::path::PathBuf {
    args.db
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from(diagram_catalog::db::sqlite::DEFAULT_DB_FILE))
}

/// Runs the `serve` subcommand: binds `args.bind` and accepts connections forever.
///
/// Refuses a non-loopback bind address unless `args.allow_remote` is set, independent
/// of TLS. Builds a mutual-TLS server config from `args.ca`/`args.cert`/`args.key`
/// (required unless `args.insecure_no_tls`) via `gemray_net::tls`, and pre-loads the
/// fingerprint allowlist once at startup purely to fail fast on a bad path -- the real,
/// per-connection check in [`accept_tls`] re-reads it every time (see [`Auth::Allowlist`]).
/// Each accepted connection is handled on its own thread.
///
/// Resolves and preflights the design-library database path once here (see
/// [`resolve_library_db_path`]/[`open_library_database`]) -- every connection thread
/// then opens its OWN read-only handle from that same path, never a shared one (see
/// [`resolve_library_db_path`]'s doc comment for why). `worker` builds also acquire
/// this process's [`gemray::renderer::gpu_backend::GpuBackend`] exactly once here
/// (shared via `Arc`, unlike `Database` -- `GpuBackend` IS `Sync`) -- `args.no_gpu`
/// forces `GpuBackend::disabled` instead of a real acquisition attempt.
///
/// # Errors
///
/// Returns a human-readable message if `args.bind` doesn't parse as a socket address,
/// if it's non-loopback without `--allow-remote`, if `--insecure-no-tls` is combined
/// with a non-loopback bind, if TLS is enabled but `--ca`/`--cert`/`--key` are missing
/// or unloadable, if the allowlist can't be loaded (and `--trust-any-client-cert` isn't
/// set), if the design-library database can't be opened, or if the bind itself fails
/// (e.g. the port is already in use).
pub fn run(args: &ServeArgs) -> Result<(), String> {
    let bind_addr: SocketAddr = args
        .bind
        .parse()
        .map_err(|e| format!("invalid --bind address {:?}: {e}", args.bind))?;

    if !bind_addr.ip().is_loopback() && !args.allow_remote {
        return Err(format!(
            "refusing to bind non-loopback address {bind_addr} without --allow-remote -- exposing this worker \
             beyond localhost must be explicit (see --help)"
        ));
    }

    let transport = build_transport(args, bind_addr)?;
    let db_path = resolve_library_db_path(args);
    // Fail fast on a bad `--db` before ever binding a socket -- see
    // `resolve_library_db_path`'s doc comment. The handle itself is dropped
    // immediately; every connection opens its own.
    drop(open_library_database(&db_path)?);

    #[cfg(feature = "worker")]
    let gpu = Arc::new(if args.no_gpu {
        gemray::renderer::gpu_backend::GpuBackend::disabled()
    } else {
        gemray::renderer::gpu_backend::GpuBackend::acquire()
    });

    let listener =
        TcpListener::bind(bind_addr).map_err(|e| format!("failed to bind {bind_addr}: {e}"))?;
    tracing::info!(
        "gemray-worker serve: listening on {bind_addr} ({}, {})",
        match &transport {
            Transport::Tls {
                auth: Auth::Allowlist(_),
                ..
            } => "mutual TLS, allowlist enforced",
            Transport::Tls {
                auth: Auth::AnyCaSignedClient,
                ..
            } => "mutual TLS, ANY CA-signed client trusted",
            Transport::Insecure => "PLAINTEXT, no TLS",
        },
        capability_summary(args),
    );

    // Token-based enrollment (see `crate::enroll`'s module doc comment): a separate
    // listener, never a mode of the render listener above, so it can't accidentally
    // relax mutual TLS for everyone.
    enroll::maybe_start_from_serve_args(args, bind_addr)?;

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let peer = stream.peer_addr().ok();
                #[cfg(feature = "worker")]
                spawn_connection_handler(stream, peer, &transport, &db_path, args.threads, &gpu);
                #[cfg(not(feature = "worker"))]
                spawn_connection_handler(stream, peer, &transport, &db_path);
            }
            Err(e) => tracing::warn!("accept error: {e}"),
        }
    }

    Ok(())
}

/// Dispatches one just-accepted `stream` on `transport` (mutual-TLS handshake vs
/// plaintext) and spawns its own thread running [`connection::handle_connection_with_gpu`]
/// (`worker` builds) or [`connection::handle_connection`] otherwise. Split out of
/// [`run`]'s accept loop purely to keep that function under `clippy::too_many_lines` --
/// still called exactly once per accepted connection, in the same place the inlined
/// version used to run.
#[cfg(feature = "worker")]
fn spawn_connection_handler(
    stream: TcpStream,
    peer: Option<SocketAddr>,
    transport: &Transport,
    db_path: &std::path::Path,
    threads: usize,
    gpu: &Arc<gemray::renderer::gpu_backend::GpuBackend>,
) {
    let db_path = db_path.to_path_buf();
    let gpu = Arc::clone(gpu);
    match transport {
        Transport::Insecure => {
            tracing::warn!(
                "--insecure-no-tls: accepted PLAINTEXT connection from {peer:?} -- no TLS, no \
                 authentication"
            );
            thread::spawn(move || {
                let Some(db) = open_connection_database(&db_path, peer) else {
                    return;
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    connection::handle_connection_with_gpu(stream, threads, &gpu, &db)
                }));
                report_connection_result(peer, result);
            });
        }
        Transport::Tls { config, auth } => {
            let config = Arc::clone(config);
            let auth = auth.clone();
            thread::spawn(move || {
                let Some(tls_stream) = accept_tls(stream, &config, &auth, peer) else {
                    return; // accept_tls already logged why
                };
                let Some(db) = open_connection_database(&db_path, peer) else {
                    return;
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    connection::handle_connection_with_gpu(tls_stream, threads, &gpu, &db)
                }));
                report_connection_result(peer, result);
            });
        }
    }
}

/// See the `#[cfg(feature = "worker")]` overload's doc comment -- identical dispatch,
/// minus the GPU/thread-count plumbing a library-only build has nothing to pass.
#[cfg(not(feature = "worker"))]
fn spawn_connection_handler(
    stream: TcpStream,
    peer: Option<SocketAddr>,
    transport: &Transport,
    db_path: &std::path::Path,
) {
    let db_path = db_path.to_path_buf();
    match transport {
        Transport::Insecure => {
            tracing::warn!(
                "--insecure-no-tls: accepted PLAINTEXT connection from {peer:?} -- no TLS, no \
                 authentication"
            );
            thread::spawn(move || {
                let Some(db) = open_connection_database(&db_path, peer) else {
                    return;
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    connection::handle_connection(stream, &db)
                }));
                report_connection_result(peer, result);
            });
        }
        Transport::Tls { config, auth } => {
            let config = Arc::clone(config);
            let auth = auth.clone();
            thread::spawn(move || {
                let Some(tls_stream) = accept_tls(stream, &config, &auth, peer) else {
                    return; // accept_tls already logged why
                };
                let Some(db) = open_connection_database(&db_path, peer) else {
                    return;
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    connection::handle_connection(tls_stream, &db)
                }));
                report_connection_result(peer, result);
            });
        }
    }
}

/// Opens this connection's own read-only [`Database`] handle from `path` -- see
/// [`resolve_library_db_path`]'s doc comment for why each connection gets a fresh one
/// rather than sharing. `None` (after logging why) if it can't be opened, so a
/// transient failure (or the file having been removed after `run` already validated it
/// once) drops just this one connection rather than panicking the accept loop.
fn open_connection_database(path: &std::path::Path, peer: Option<SocketAddr>) -> Option<Database> {
    match open_library_database(path) {
        Ok(db) => Some(db),
        Err(e) => {
            tracing::warn!("connection {peer:?}: {e}");
            None
        }
    }
}

/// One-line capability summary for [`run`]'s startup log line -- `worker`-conditional,
/// so the message honestly matches what `WELCOME` will actually advertise (see the
/// module docs).
#[cfg(feature = "worker")]
fn capability_summary(args: &ServeArgs) -> String {
    if args.no_gpu {
        "library + render (CPU only, --no-gpu)".to_string()
    } else {
        "library + render".to_string()
    }
}

#[cfg(not(feature = "worker"))]
fn capability_summary(_args: &ServeArgs) -> String {
    "library only (no render capacity -- build with --features worker to add it)".to_string()
}

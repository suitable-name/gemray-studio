//! Token-based enrollment.
//!
//! Replaces "copy the certificate bundle to the client machine by hand" with a one-time,
//! time-limited token the operator reads out (or pastes) to the person enrolling a new
//! viewer.
//!
//! # Why this needs its own listener, not the render port
//!
//! The render listener (`crate::serve`) requires mutual TLS -- every connection must
//! already present a certificate signed by this worker's CA. An enrolling client has no
//! such certificate yet (that's the bootstrap problem this module exists to solve), so it
//! structurally cannot use that listener. Rather than relaxing the render listener's
//! requirement for one special case (the surest way to eventually relax it by accident
//! for everyone), enrollment gets its own listener, on its own port, with its own TLS
//! config that never requires a client certificate and -- critically -- whose connection
//! handler ([`handle_enroll_connection`]) never calls into [`crate::stream_emit`] or
//! [`crate::serve::handle_connection_with_gpu`] at all. A claim connection cannot reach
//! `RenderRequest` handling not because it's rejected at runtime, but because the code
//! path to get there doesn't exist on this listener.
//!
//! # The token
//!
//! See `gemray_net::token`'s doc comment for the wire encoding. Three properties matter for
//! security, all enforced here:
//!
//! 1. **The secret is a fresh 256-bit CSPRNG value, never the certificate's fingerprint.**
//!    A certificate's fingerprint is a public value (it crosses the wire in the clear on
//!    every TLS handshake, and is exactly what `allowlist.txt` stores as a permanent
//!    identifier) -- a bearer secret and a long-term public identifier must never be the
//!    same value, or possessing the public one would be enough to claim.
//! 2. **The server stores only [`sha2::Sha256`] of the secret**, in
//!    [`PendingEnrollment::secret_hash`], and [`EnrollRegistry::claim`] compares it using
//!    [`subtle::ConstantTimeEq`] rather than `==` -- see that function's doc comment.
//! 3. **The token commits to the server's identity.** It carries the worker's CA
//!    certificate's SHA-256 fingerprint alongside the secret, and
//!    `gemray_net::enroll::claim` (called by this crate's own `enroll_client::claim`, and
//!    by `apps/diagram-gui`'s token-redeem UI) verifies the enrollment listener's
//!    presented certificate chains to a CA matching that exact fingerprint *before* it
//!    ever sends the secret -- see `gemray_net::enroll`'s `PinnedCaVerifier`. Without
//!    this, an active attacker sitting between the enrolling client and the real worker
//!    could impersonate the worker, relay the handshake, and walk away with a validly
//!    issued client certificate.
//!
//! # Lifecycle
//!
//! - Single use: [`EnrollRegistry::claim`] removes the matching entry from the registry
//!   before returning it, so a second attempt with the same secret finds nothing.
//! - [`TOKEN_TTL_SECS`] (180s) TTL, checked server-side against [`Instant::now`] on every
//!   claim attempt -- never a client-side courtesy, since the client making the claim is
//!   exactly the party a clock-skew or malicious client-clock argument shouldn't be
//!   trusted from.
//! - The bundle a token was issued with lives only in [`EnrollRegistry`]'s in-memory
//!   `Vec` -- see [`crate::pki::issue_client_in_memory`] -- and is never written to disk
//!   by this process. It only reaches disk on the CLAIMING machine, via
//!   `crate::enroll_client::claim`, exactly like an `issue-client` bundle does today.
//! - The claimed certificate's fingerprint is appended to `allowlist.txt` only inside
//!   [`handle_enroll_connection`]'s `Claim` arm, strictly after [`EnrollRegistry::claim`]
//!   has already returned a match -- never at issue time. See
//!   [`allowlist_gains_the_fingerprint_only_after_a_successful_claim`] for the test
//!   proving this ordering.
//! - On claim, expiry, or the registry itself being dropped (including at process exit,
//!   for whatever's still in scope when unwinding runs), the pending record's key
//!   material is explicitly zeroized: [`PendingEnrollment`] and [`EnrollBundle`] hold
//!   their sensitive fields as [`zeroize::Zeroizing`] wrappers rather than plain
//!   `String`/`[u8; 32]`, so the ordinary `Drop` glue -- already triggered by removal from
//!   the registry's `Vec` on claim or expiry-sweep, or by the whole `Vec` dropping at
//!   process exit -- overwrites the bytes before the allocator ever gets to free them
//!   unscrubbed. This does not (nothing in userspace can) protect against the process
//!   being killed abruptly rather than exiting -- `Drop` never runs in that case.
//! - A `serve` restart starts a brand new, empty [`EnrollRegistry`]: nothing persists any
//!   pending enrollment across a restart. This is fail-closed and intentional, not a gap
//!   -- an operator who restarted `serve` mid-enrollment just re-issues a token.
//!
//! # Where issuing happens
//!
//! Because the bundle has to live in memory from the moment it's minted, issuing a token
//! and claiming it must happen inside the *same* running `serve` process -- unlike `cert
//! issue-client`, which is a one-shot process with nothing to hand off to anything else.
//! So `cert issue-token` (`crate::enroll_client::run_issue_token`) is a small TLS client
//! that connects to this listener and asks the running `serve` process to mint one, using
//! the operator's own already-possessed `ca.pem` for normal (non-pinned) server
//! verification -- the operator issuing a token is, by definition, someone who already has
//! the real CA file sitting next to their `serve` invocation, so there's no bootstrap
//! problem on that side the way there is for the claiming client.
//!
//! Issuing is further restricted to loopback peers only, checked in
//! [`handle_enroll_connection`] against the actual accepted `TcpStream::peer_addr()` --
//! not merely "this listener happens to be bound to loopback", since the SAME listener
//! also accepts (non-loopback, if `serve --allow-remote` is set) claim connections from
//! wherever the enrolling viewer actually is. This mirrors the trust `serve`'s own
//! loopback-bind gating already relies on: local access already means the operator can
//! read `ca.key` and mint whatever certificates they like directly, so no further
//! in-band credential is layered on top of "the OS says this connection genuinely
//! originated from `127.0.0.1`/`::1`" (which a remote attacker cannot forge -- packets
//! claiming a loopback source arriving on a real network interface are dropped by the OS
//! network stack, not merely by this process).

use crate::pki;
use std::{
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};

mod connection;
mod registry;
#[cfg(test)]
mod tests;

pub use registry::EnrollRegistry;

/// How long an issued token remains claimable.
///
/// Fixed, not operator-configurable -- see this module's doc comment on why the TTL is
/// enforced server-side rather than being a courtesy either side can override.
pub const TOKEN_TTL_SECS: u64 = 180;

/// Caps how many enrollments can be pending at once, purely to bound this process's
/// memory footprint against a runaway or misbehaving admin caller -- the `Issue` path is
/// already loopback-only (see the module doc comment), so this is a sanity bound, not a
/// defense against a remote attacker.
const MAX_PENDING: usize = 64;

/// Builds the enrollment listener's TLS server config: TLS 1.3 only (matching
/// `gemray_net::tls`), presenting `[server_cert, ca_cert]` as the chain -- deliberately
/// including the CA certificate itself, not just the server leaf -- so a claiming client
/// (which has no CA file of its own to load) receives the actual CA certificate bytes it
/// needs to hash and compare against the fingerprint carried in its token. See
/// `crate::enroll_client::PinnedCaVerifier`.
///
/// Requires no client certificate (`with_no_client_auth`) -- an enrolling client has none
/// yet, and an operator's `cert issue-token` call presents none either (it authenticates
/// itself by connecting from loopback, not by certificate; see the module doc comment).
/// This is a genuinely different, weaker-on-the-client-auth-axis TLS posture than the
/// render listener's -- which is exactly why it's a separate `rustls::ServerConfig` on a
/// separate port, never a mode of the render listener's own config.
///
/// # Errors
///
/// A human-readable message if the server certificate/key/CA can't be loaded from
/// `cert_path`/`key_path`/`ca_path`, or don't form a valid TLS server configuration.
fn build_enroll_server_config(
    ca_path: &Path,
    cert_path: &Path,
    key_path: &Path,
) -> Result<Arc<rustls::ServerConfig>, String> {
    let mut chain = gemray_net::tls::load_certs(cert_path)
        .map_err(|e| format!("--cert {}: {e}", cert_path.display()))?;
    let ca_certs = gemray_net::tls::load_certs(ca_path)
        .map_err(|e| format!("--ca {}: {e}", ca_path.display()))?;
    chain.extend(ca_certs);
    let key = gemray_net::tls::load_private_key(key_path)
        .map_err(|e| format!("--key {}: {e}", key_path.display()))?;

    let config = rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(|e| format!("failed to build the enrollment TLS config: {e}"))?;
    Ok(Arc::new(config))
}

/// Configuration [`spawn_enroll_listener`] needs, gathered once at `serve` startup from
/// the same `--ca`/`--cert`/`--key`/`--allowlist` args the render listener already
/// validated -- see `crate::serve::run`.
pub struct EnrollConfig {
    pub bind_addr: SocketAddr,
    pub pki_dir: PathBuf,
    /// `None` when `--trust-any-client-cert` is set: there's no allowlist to append to
    /// in that mode (any CA-signed client is already trusted), so a successful claim
    /// simply skips that step -- see [`connection::handle_enroll_connection`].
    pub allowlist_path: Option<PathBuf>,
    pub tls_config: Arc<rustls::ServerConfig>,
}

impl EnrollConfig {
    /// Builds an [`EnrollConfig`] from the same paths `serve --ca/--cert/--key` were
    /// given, plus the resolved allowlist path `crate::serve::build_transport` would
    /// have used. `enroll_bind` is `--enroll-bind` if given, else the same host as
    /// `bind_addr` on the next port up -- see `crate::cli::USAGE`'s `--enroll-bind`
    /// entry for why that's a reasonable default rather than a second required flag.
    ///
    /// # Errors
    ///
    /// A human-readable message if `enroll_bind` doesn't parse as a socket address, if
    /// it's non-loopback without `allow_remote`, or if the TLS config can't be built
    /// (see [`build_enroll_server_config`]).
    pub fn build(
        bind_addr: SocketAddr,
        enroll_bind: Option<&str>,
        allow_remote: bool,
        ca_path: &Path,
        cert_path: &Path,
        key_path: &Path,
        allowlist_path: Option<PathBuf>,
    ) -> Result<Self, String> {
        let enroll_addr_str = enroll_bind.map_or_else(
            || format!("{}:{}", bind_addr.ip(), bind_addr.port().saturating_add(1)),
            str::to_string,
        );
        let enroll_bind_addr: SocketAddr = enroll_addr_str
            .parse()
            .map_err(|e| format!("invalid --enroll-bind address {enroll_addr_str:?}: {e}"))?;
        if !enroll_bind_addr.ip().is_loopback() && !allow_remote {
            return Err(format!(
                "refusing to bind non-loopback enrollment address {enroll_bind_addr} without --allow-remote -- \
                 exposing the enrollment listener beyond localhost must be explicit, same as --bind (see --help)"
            ));
        }

        let tls_config = build_enroll_server_config(ca_path, cert_path, key_path)?;
        let pki_dir = ca_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        Ok(Self {
            bind_addr: enroll_bind_addr,
            pki_dir,
            allowlist_path,
            tls_config,
        })
    }
}

/// Starts the enrollment listener for a `serve` invocation, straight from its
/// [`crate::cli::ServeArgs`].
///
/// This is the one call `crate::serve::run` makes, so it doesn't need to know any of
/// `EnrollConfig::build`'s parameters itself.
///
/// A no-op (not an error) when `args.insecure_no_tls` (no CA to enroll against) or
/// `args.no_enroll` is set. Otherwise resolves the same allowlist path
/// `crate::serve::build_transport` would have (`None` under `--trust-any-client-cert`,
/// matching [`EnrollConfig::allowlist_path`]'s own doc comment on why) and hands
/// everything to [`EnrollConfig::build`] then [`spawn_enroll_listener`].
///
/// # Errors
///
/// Whatever [`EnrollConfig::build`] or [`spawn_enroll_listener`] returns.
pub fn maybe_start_from_serve_args(
    args: &crate::cli::ServeArgs,
    bind_addr: SocketAddr,
) -> Result<(), String> {
    if args.insecure_no_tls || args.no_enroll {
        return Ok(());
    }
    let (Some(ca_path), Some(cert_path), Some(key_path)) = (&args.ca, &args.cert, &args.key) else {
        return Ok(()); // `crate::serve::build_transport` already requires these; nothing to do here.
    };

    let allowlist_path = if args.trust_any_client_cert {
        None
    } else {
        Some(
            args.allowlist
                .clone()
                .unwrap_or_else(|| pki::default_allowlist_path(ca_path)),
        )
    };
    let config = EnrollConfig::build(
        bind_addr,
        args.enroll_bind.as_deref(),
        args.allow_remote,
        ca_path,
        cert_path,
        key_path,
        allowlist_path,
    )?;
    spawn_enroll_listener(config).map(|_bound_addr| ())
}

/// Binds `config.bind_addr` and spawns a background thread accepting enrollment
/// connections forever.
///
/// Mirrors `crate::serve::run`'s own accept-loop shape (one `thread::spawn` per
/// connection, `catch_unwind`-wrapped, every panic logged rather than taking the
/// listener down). Returns as soon as the bind succeeds (so a bad `--enroll-bind` is
/// reported at `serve` startup, synchronously, like every other startup
/// misconfiguration), with the accept loop itself running in the background for the rest
/// of the process's life.
///
/// Returns the listener's actual bound address (useful when `config.bind_addr`'s port was
/// `0`, letting the OS choose -- this crate's own tests use that to find an ephemeral
/// port for a real loopback round trip; `crate::serve::run` ignores it, since operators
/// specify a concrete port).
///
/// # Errors
///
/// A human-readable message if the bind itself fails (e.g. the port is already in use).
pub fn spawn_enroll_listener(config: EnrollConfig) -> Result<SocketAddr, String> {
    let listener = TcpListener::bind(config.bind_addr).map_err(|e| {
        format!(
            "failed to bind enrollment listener {}: {e}",
            config.bind_addr
        )
    })?;
    let bound_addr = listener
        .local_addr()
        .map_err(|e| format!("enrollment listener bound but local_addr() failed: {e}"))?;
    tracing::info!(
        "gemray-worker serve: enrollment listener on {bound_addr} (token TTL {TOKEN_TTL_SECS}s; `gemray-worker cert \
         issue-token` to mint one)"
    );

    let registry = Arc::new(EnrollRegistry::new());
    let tls_config = config.tls_config;
    let pki_dir = config.pki_dir;
    let allowlist_path = config.allowlist_path;

    thread::spawn(move || {
        for incoming in listener.incoming() {
            match incoming {
                Ok(stream) => {
                    let peer = stream.peer_addr().ok();
                    let tls_config = Arc::clone(&tls_config);
                    let registry = Arc::clone(&registry);
                    let pki_dir = pki_dir.clone();
                    let allowlist_path = allowlist_path.clone();
                    thread::spawn(move || {
                        let Some(tls_stream) =
                            connection::accept_enroll_tls(stream, &tls_config, peer)
                        else {
                            return;
                        };
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            connection::handle_enroll_connection(
                                tls_stream,
                                &registry,
                                &pki_dir,
                                allowlist_path.as_deref(),
                                peer,
                            )
                        }));
                        match result {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => {
                                tracing::warn!(
                                    "enrollment connection {peer:?} ended with an error: {e}"
                                );
                            }
                            Err(_) => {
                                tracing::warn!(
                                    "enrollment connection {peer:?} panicked and was dropped"
                                );
                            }
                        }
                    });
                }
                Err(e) => tracing::warn!("enrollment accept error: {e}"),
            }
        }
    });

    Ok(bound_addr)
}

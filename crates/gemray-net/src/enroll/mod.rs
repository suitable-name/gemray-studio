//! The enrollment wire protocol, and the claiming half of token-based enrollment.
//!
//! # Why this lives in `gemray-net`, not `apps/gemray-worker`
//!
//! `apps/gemray-worker` originally owned all of token-based enrollment: the message
//! types ([`EnrollRequest`]/[`EnrollResponse`]), the server-side registry that mints and
//! tracks pending tokens, AND the claiming client that redeems one. When
//! `apps/diagram-gui` needed to redeem a token directly (so a viewer user never has to
//! install the worker binary or run `cert claim` in a terminal), only PART of that had
//! to move: `apps/diagram-gui` must not depend on `apps/gemray-worker` -- a viewer
//! depending on the server application is the wrong direction, and there is already one
//! shared crate both apps depend on for exactly this kind of thing.
//!
//! The split follows what's actually shared:
//!
//! - **The wire messages and the claiming client move here.** Both `apps/gemray-worker`
//!   (`cert claim`) and `apps/diagram-gui` need to encode an [`EnrollRequest::Claim`],
//!   decode an [`EnrollResponse`], and -- the security-critical part -- verify the
//!   enrollment listener's presented certificate chain against the CA fingerprint the
//!   token commits to, via [`PinnedCaVerifier`], *before* ever sending the secret. That
//!   verification is a POLICY, not a convenience function: this workspace already
//!   consolidated one accidentally-duplicated policy
//!   (`gemray::renderer::gpu_backend::GpuBackend`, which used to exist twice) because two
//!   copies of a correctness/security rule is a copy that can drift. Copying
//!   `PinnedCaVerifier` into `apps/diagram-gui` instead of moving it here would be
//!   exactly that mistake, just for authentication instead of rendering correctness.
//! - **The server-side registry stays in `apps/gemray-worker`.** `EnrollRegistry`
//!   (minting bundles, hashing secrets, the constant-time compare, the TTL sweep) has no
//!   viewer-side counterpart -- nothing in `apps/diagram-gui` ever issues a token or
//!   holds one pending -- so there is nothing shared to pull out. It stays in
//!   `apps/gemray-worker/src/enroll.rs`, which now imports [`EnrollRequest`]/
//!   [`EnrollResponse`] from here instead of defining them.
//! - [`crate::token`] (the `GW1-...` codec) moved here for the same reason as this
//!   module: [`crate::token::decode`] is exactly what [`claim`] needs, and there is no
//!   sense in a wire format only one side of a two-party protocol is allowed to decode.
//!
//! See `apps/gemray-worker/src/enroll.rs`'s own module doc comment for the full
//! enrollment design (the token's three security properties, the lifecycle, why issuing
//! is loopback-only) -- none of that changed by this move, only where the code that
//! implements it lives.

use crate::token;
use rustls::{
    DigitallySignedStruct, Error as RustlsError, RootCertStore, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{CertificateDer, ServerName, UnixTime},
};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    net::TcpStream,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zeroize::{Zeroize, Zeroizing};

/// One request on the enrollment wire protocol.
///
/// Deliberately a *different* message schema than [`crate::messages::ClientMessage`]
/// (the render protocol), not merely a different enum variant of it, so a
/// `RenderRequest`'s bytes can never accidentally decode as one of these (or vice versa)
/// even if a wire got crossed. See `apps/gemray-worker/src/enroll.rs`'s module doc
/// comment on why a claim connection structurally cannot reach rendering.
#[derive(Debug, Serialize, Deserialize)]
pub enum EnrollRequest {
    /// Mint a new token for a viewer that will be labeled `name` once claimed (matching
    /// `cert issue-client --name`'s role: a human label for the allowlist entry and log
    /// messages). Only honored from a loopback peer -- see
    /// `apps/gemray-worker/src/enroll.rs`'s `handle_enroll_connection`.
    Issue { name: String },
    /// Attempt to claim a pending enrollment with this secret. See
    /// `apps/gemray-worker/src/enroll.rs`'s `EnrollRegistry::claim`.
    Claim { secret: [u8; token::SECRET_LEN] },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum EnrollResponse {
    Issued {
        /// The full `GW1-...` token string -- see [`crate::token`].
        token: String,
        expires_in_secs: u64,
    },
    /// Deliberately carries a reason (this is the loopback-only / registry-full path,
    /// not attacker-facing) -- contrast with [`EnrollResponse::ClaimFailed`], whose
    /// reason is always the same generic string.
    IssueRefused { reason: String },
    Claimed {
        ca_pem: String,
        client_cert_pem: String,
        client_key_pem: String,
    },
    /// Deliberately the same message for "wrong secret", "expired", "already claimed",
    /// and "internal error appending to the allowlist" -- distinguishing them to whoever
    /// is on the other end of a claim connection would hand a remote prober a way to
    /// enumerate which failure mode it hit. The real reason is still logged server-side
    /// (`RUST_LOG=debug` or above) for the operator's own troubleshooting. [`claim`]'s
    /// own caller-facing [`ClaimError::Refused`] preserves this same deliberate
    /// ambiguity -- see that variant's doc comment.
    ClaimFailed,
}

/// Verifies the enrollment listener's presented certificate chain against a CA whose
/// SHA-256 fingerprint matches [`Self::expected_ca_fingerprint`] -- never against a CA
/// file on disk, since a claiming client (this is only ever used by [`claim`]) has none
/// yet. See `apps/gemray-worker/src/enroll.rs`'s module doc comment, point 3, and
/// `crate::token`'s.
///
/// This is certificate PINNING, a legitimate pattern `rustls` itself exposes
/// `ClientConfig::dangerous()` specifically to support -- distinct from disabling
/// verification. Every actual cryptographic operation (X.509 chain construction,
/// signature verification) is still performed by `rustls`/`rustls-webpki`/`ring` through
/// their own public APIs, delegated to below -- this type supplies only the trust-anchor
/// *selection* (which of the presented certificates counts as "the CA"), which is exactly
/// what the token's fingerprint is for.
#[derive(Debug)]
struct PinnedCaVerifier {
    expected_ca_fingerprint: crate::tls::Fingerprint,
    provider: Arc<rustls::crypto::CryptoProvider>,
    /// Set to `true` the one time [`Self::verify_server_cert`] fails specifically
    /// because none of the presented certificates matched
    /// [`Self::expected_ca_fingerprint`] -- the security-relevant failure mode. [`claim`]
    /// reads this after a failed handshake to distinguish "this server is not who the
    /// token says it should be" (an attacker-in-the-middle-shaped failure, worth its own
    /// distinct, unmissable wording) from an ordinary handshake failure on the pinned CA
    /// itself (e.g. an expired presented certificate) or a plain connection failure. A
    /// shared flag rather than a richer `rustls::Error` variant because
    /// `ServerCertVerifier::verify_server_cert` must return a real `rustls::Error` --
    /// there is no `Other(Box<dyn Error>)` constructor generic enough to carry a typed
    /// payload back out through `rustls`'s own `complete_io` -- and `Arc<AtomicBool>`
    /// rather than `Cell<bool>` because [`ServerCertVerifier`] requires `Send + Sync`.
    ca_mismatch: Arc<AtomicBool>,
}

impl PinnedCaVerifier {
    fn new(expected_ca_fingerprint: crate::tls::Fingerprint, ca_mismatch: Arc<AtomicBool>) -> Self {
        Self {
            expected_ca_fingerprint,
            provider: Arc::new(rustls::crypto::ring::default_provider()),
            ca_mismatch,
        }
    }
}

impl ServerCertVerifier for PinnedCaVerifier {
    /// Finds, among the certificates the server presented alongside its leaf
    /// (`intermediates` -- see `apps/gemray-worker/src/enroll.rs`'s
    /// `build_enroll_server_config`, which deliberately sends `[server_cert, ca_cert]`),
    /// the one whose SHA-256 fingerprint matches the token's. If none matches, this
    /// server is not the one the token commits to and verification fails outright -- no
    /// fallback to "trust it anyway" -- and [`Self::ca_mismatch`] is set so [`claim`] can
    /// report this distinctly from any other handshake failure. Once found, a one-off
    /// `RootCertStore` containing just that certificate is built and the ENTIRE rest of
    /// chain/signature verification (expiry, SAN-vs-hostname, signature validity) is
    /// delegated to `rustls`'s own `WebPkiServerVerifier` against it -- see this type's
    /// own doc comment on why that split is what keeps this "pinning", not "skipping
    /// verification".
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, RustlsError> {
        let Some(pinned_ca) = intermediates
            .iter()
            .find(|c| crate::tls::fingerprint(c) == self.expected_ca_fingerprint)
        else {
            self.ca_mismatch.store(true, Ordering::SeqCst);
            return Err(RustlsError::General(
                "the server did not present a certificate chain containing the CA this \
                 enrollment token commits to -- refusing to trust it (this is either the \
                 wrong worker, or an attacker in the middle)"
                    .to_string(),
            ));
        };

        let mut roots = RootCertStore::empty();
        roots.add(pinned_ca.clone()).map_err(|e| {
            RustlsError::General(format!(
                "the pinned CA certificate is not usable as a trust anchor: {e}"
            ))
        })?;
        let verifier = rustls::client::WebPkiServerVerifier::builder(Arc::new(roots))
            .build()
            .map_err(|e| {
                RustlsError::General(format!("failed to build a verifier for the pinned CA: {e}"))
            })?;

        verifier.verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        Err(RustlsError::General(
            "TLS 1.2 is not supported by the enrollment listener".to_string(),
        ))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Splits `addr` (`host:port`) into its host part, for building a TLS [`ServerName`].
/// Using the last `:` (via `rsplit_once`) rather than the first tolerates a bracketed
/// IPv6 literal's own internal colons only if further validated by
/// `ServerName::try_from` -- this function just isolates the substring before the final
/// port separator. `None` if `addr` has no `:` at all.
fn host_of(addr: &str) -> Option<&str> {
    addr.rsplit_once(':').map(|(host, _)| host)
}

/// A freshly claimed client-certificate bundle, still in memory.
///
/// The same three PEM blocks `gemray-worker cert issue-client`/`cert claim` would write
/// to `ca.pem`/`client.pem`/`client.key`. Writing them to disk (with the private key's
/// permissions restricted -- see [`crate::tls::write_private_key_pem`]) is the caller's
/// job, since where the bundle belongs is caller-specific: `apps/gemray-worker`'s `cert
/// claim` writes to `--out`, `apps/diagram-gui` writes to a per-worker directory under
/// its own settings folder.
#[derive(Debug)]
pub struct ClaimedBundle {
    pub ca_pem: String,
    pub client_cert_pem: String,
    /// Wrapped in [`Zeroizing`] since this is the client's private key, plaintext,
    /// having just crossed the wire -- see `apps/gemray-worker/docs/security.md`'s
    /// "Known limitation: the private key transits" section. Overwritten the moment the
    /// caller is done with it (typically: written to disk, then dropped), rather than
    /// lingering in a freed heap allocation.
    pub client_key_pem: Zeroizing<String>,
}

/// Everything that can go wrong redeeming an enrollment token.
///
/// One type so a caller (a CLI's error formatting, or a GUI's toast) has a single match
/// to render instead of parsing a generic string. Every variant here maps to genuinely
/// different advice for whoever is staring at the failure -- see [`fmt::Display`]'s impl
/// below, and each variant's own doc comment for why it's distinguished from its
/// neighbors.
#[derive(Debug)]
pub enum ClaimError {
    /// `token` isn't a well-formed `GW1-...` string -- see [`token::TokenError`].
    InvalidToken(token::TokenError),
    /// `addr` isn't `host:port`, or its host half isn't valid for TLS server-name
    /// verification.
    InvalidAddr(String),
    /// The TCP connection itself could not be established -- most likely a wrong or
    /// unreachable address (a typo, the worker not running, a firewall), not a security
    /// concern. Distinguished from [`Self::CaFingerprintMismatch`] precisely so a wrong
    /// address never gets alarming security wording, and a genuine impersonation never
    /// reads like an everyday "check your network" message.
    Connect {
        addr: String,
        source: std::io::Error,
    },
    /// The TLS handshake failed for a reason OTHER than the pinned CA fingerprint not
    /// matching -- e.g. the pinned certificate itself failed ordinary `webpki`
    /// validation (expired, not yet valid). Rare in practice (the token and the CA it
    /// pins are both minted together, moments apart), but not conflated with
    /// [`Self::CaFingerprintMismatch`]: this is not evidence of impersonation.
    Handshake {
        addr: String,
        source: std::io::Error,
    },
    /// The security-relevant handshake failure: the server presented a certificate
    /// chain that does not contain a certificate matching the CA fingerprint this token
    /// commits to. This is either the wrong worker or an attacker in the middle -- see
    /// [`PinnedCaVerifier`]. Deliberately its own variant (not folded into
    /// [`Self::Handshake`]) so callers can -- and must -- word this differently from an
    /// ordinary connection problem.
    CaFingerprintMismatch { addr: String },
    /// A transport-level failure sending the claim request or reading the response,
    /// after a successful, correctly-pinned handshake.
    Protocol(String),
    /// The server explicitly reported [`EnrollResponse::ClaimFailed`]: the token was
    /// wrong, already claimed, or expired (tokens live 180 seconds). The server
    /// deliberately does not say which -- see [`EnrollResponse::ClaimFailed`]'s own doc
    /// comment on why distinguishing them would hand a remote prober an enumeration
    /// oracle -- so this crate cannot honestly report more than the server chose to
    /// reveal. A caller wanting to name expiry specifically as ONE of the possibilities
    /// (it is the single most likely one in practice, given the 180s window) may say so
    /// in its own wording, but must not claim to have distinguished it from "already
    /// claimed".
    Refused,
    /// The server responded with something other than `Claimed`/`ClaimFailed` to a
    /// `Claim` request -- a protocol/version mismatch rather than any of the above.
    UnexpectedResponse,
}

impl fmt::Display for ClaimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken(e) => write!(f, "not a valid enrollment token: {e}"),
            // `InvalidAddr` and `Protocol` already carry a complete, self-describing
            // message from where they were constructed -- there's nothing this impl
            // could usefully add to either, so both just pass it through verbatim.
            Self::InvalidAddr(msg) | Self::Protocol(msg) => write!(f, "{msg}"),
            Self::Connect { addr, source } => write!(f, "could not connect to {addr}: {source}"),
            Self::Handshake { addr, source } => {
                write!(f, "TLS handshake with {addr} failed: {source}")
            }
            Self::CaFingerprintMismatch { addr } => write!(
                f,
                "TLS handshake with {addr} failed: the server did not present a certificate chain \
                 containing the CA this enrollment token commits to -- refusing to trust it (this is \
                 either the wrong worker, or an attacker in the middle)"
            ),
            Self::Refused => write!(
                f,
                "claim failed: the enrollment token was invalid, expired, or already used"
            ),
            Self::UnexpectedResponse => {
                write!(f, "unexpected response from the enrollment listener")
            }
        }
    }
}

impl std::error::Error for ClaimError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidToken(e) => Some(e),
            Self::Connect { source, .. } | Self::Handshake { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Redeems `token` against the enrollment listener at `addr` (`host:port`), the way
/// `gemray-worker cert claim` and (via this same function) `apps/diagram-gui`'s
/// "Redeem token" dialog action both do.
///
/// Decodes `token` (see [`crate::token`]), connects to `addr`, and verifies the
/// enrollment listener there against the CA fingerprint the token carries -- via
/// [`PinnedCaVerifier`] -- *before* ever sending the token's secret. Zeroizes its own
/// copy of the secret immediately after sending it, and again when the decoded token is
/// dropped. On success, returns the bundle in memory; writing it to disk (with the
/// private key's permissions restricted) is the caller's job -- see [`ClaimedBundle`]'s
/// own doc comment for why that split is caller-specific.
///
/// This function itself never logs `token`, the decoded secret, or the claimed private
/// key at any level, on success or failure -- there is no `tracing`/`println!` call
/// anywhere in this function or in [`PinnedCaVerifier`]. Every [`ClaimError`] this can
/// return is built from `addr` and the underlying I/O/TLS error only.
///
/// # Errors
///
/// See [`ClaimError`]'s variants -- one for each distinguishable way this can fail, on
/// purpose: a caller rendering this to a human (a CLI's `--token`/`--addr`-flavored
/// message, or a GUI toast) needs to say something different for "you typo'd the
/// address" than for "something is impersonating the worker".
pub fn claim(token: &str, addr: &str) -> Result<ClaimedBundle, ClaimError> {
    let decoded = token::decode(token).map_err(ClaimError::InvalidToken)?;

    let host = host_of(addr)
        .ok_or_else(|| ClaimError::InvalidAddr(format!("{addr:?} must be host:port")))?;
    let server_name = ServerName::try_from(host.to_string()).map_err(|e| {
        ClaimError::InvalidAddr(format!("{addr:?}: invalid host for TLS verification: {e}"))
    })?;

    let ca_mismatch = Arc::new(AtomicBool::new(false));
    let verifier = Arc::new(PinnedCaVerifier::new(
        decoded.ca_fingerprint,
        Arc::clone(&ca_mismatch),
    ));
    let client_config =
        rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();

    let tcp = TcpStream::connect(addr).map_err(|e| ClaimError::Connect {
        addr: addr.to_string(),
        source: e,
    })?;
    let conn =
        rustls::ClientConnection::new(Arc::new(client_config), server_name).map_err(|e| {
            ClaimError::Handshake {
                addr: addr.to_string(),
                source: std::io::Error::other(e),
            }
        })?;
    let mut stream = rustls::StreamOwned::new(conn, tcp);
    if let Err(e) = stream.conn.complete_io(&mut stream.sock) {
        return Err(if ca_mismatch.load(Ordering::SeqCst) {
            ClaimError::CaFingerprintMismatch {
                addr: addr.to_string(),
            }
        } else {
            ClaimError::Handshake {
                addr: addr.to_string(),
                source: e,
            }
        });
    }

    // Send the secret, then immediately zeroize both the request's own copy and the
    // token's -- see `apps/gemray-worker/src/enroll.rs`'s module doc comment on why the
    // secret shouldn't linger in memory a moment longer than it has to, on either side
    // of the wire.
    let mut request = EnrollRequest::Claim {
        secret: *decoded.secret,
    };
    let write_result = crate::messages::write_message(&mut stream, &request);
    if let EnrollRequest::Claim { secret } = &mut request {
        secret.zeroize();
    }
    drop(decoded); // zeroizes its own `Zeroizing<[u8; 32]>` secret on drop
    write_result.map_err(|e| ClaimError::Protocol(e.to_string()))?;

    let response: EnrollResponse = crate::messages::read_message(&mut stream)
        .map_err(|e| ClaimError::Protocol(e.to_string()))?;

    match response {
        EnrollResponse::Claimed {
            ca_pem,
            client_cert_pem,
            client_key_pem,
        } => Ok(ClaimedBundle {
            ca_pem,
            client_cert_pem,
            client_key_pem: Zeroizing::new(client_key_pem),
        }),
        EnrollResponse::ClaimFailed => Err(ClaimError::Refused),
        EnrollResponse::Issued { .. } | EnrollResponse::IssueRefused { .. } => {
            Err(ClaimError::UnexpectedResponse)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_of_splits_on_the_last_colon() {
        assert_eq!(host_of("127.0.0.1:7879"), Some("127.0.0.1"));
        assert_eq!(host_of("worker.lan:7879"), Some("worker.lan"));
    }

    #[test]
    fn host_of_returns_none_for_a_missing_port() {
        assert_eq!(host_of("worker.lan"), None);
    }

    #[test]
    fn claim_rejects_an_invalid_token_before_ever_touching_the_network() {
        // No listener is running at this address at all -- if `claim` tried to connect
        // before validating the token, this would fail with a connection error instead
        // of `InvalidToken`, which is exactly the ordering this test pins down.
        let err = claim("not-a-token", "127.0.0.1:1").unwrap_err();
        assert!(matches!(err, ClaimError::InvalidToken(_)), "{err}");
    }

    #[test]
    fn claim_rejects_an_address_with_no_port() {
        let secret = [0u8; token::SECRET_LEN];
        let fp: crate::tls::Fingerprint = std::array::from_fn(|i| i as u8);
        let token = token::encode(&secret, &fp);
        let err = claim(&token, "worker-with-no-port").unwrap_err();
        assert!(matches!(err, ClaimError::InvalidAddr(_)), "{err}");
    }

    #[test]
    fn claim_error_display_distinguishes_ca_mismatch_from_an_ordinary_handshake_failure() {
        let mismatch = ClaimError::CaFingerprintMismatch {
            addr: "worker.lan:7879".to_string(),
        }
        .to_string();
        let handshake = ClaimError::Handshake {
            addr: "worker.lan:7879".to_string(),
            source: std::io::Error::other("boom"),
        }
        .to_string();
        assert_ne!(mismatch, handshake);
        assert!(mismatch.contains("attacker in the middle"), "{mismatch}");
        assert!(!handshake.contains("attacker in the middle"), "{handshake}");
    }
}

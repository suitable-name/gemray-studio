//! The enrollment listener's per-connection handling: a bare (no client-certificate
//! requirement) TLS accept, then exactly one [`gemray_net::enroll::EnrollRequest`]/
//! [`gemray_net::enroll::EnrollResponse`] exchange -- see `crate::enroll`'s module doc
//! comment for why this listener structurally cannot reach `RenderRequest` handling.

use super::registry::{EnrollRegistry, ZeroizeLocal};
use gemray_net::enroll::{EnrollRequest, EnrollResponse};
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::Path,
    sync::Arc,
};

pub(super) type EnrollTlsStream = rustls::StreamOwned<rustls::ServerConnection, TcpStream>;

/// Completes the TLS handshake for one just-accepted enrollment connection. No
/// client-certificate check follows (unlike `crate::serve::accept_tls`) -- this listener
/// never requires one; see the module doc comment.
pub(super) fn accept_enroll_tls(
    stream: TcpStream,
    config: &Arc<rustls::ServerConfig>,
    peer: Option<SocketAddr>,
) -> Option<EnrollTlsStream> {
    let conn = match rustls::ServerConnection::new(Arc::clone(config)) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("enrollment connection {peer:?}: could not start TLS: {e}");
            return None;
        }
    };
    let mut tls_stream = rustls::StreamOwned::new(conn, stream);
    if let Err(e) = tls_stream.conn.complete_io(&mut tls_stream.sock) {
        tracing::warn!("enrollment connection {peer:?}: TLS handshake failed: {e}");
        return None;
    }
    Some(tls_stream)
}

/// Handles exactly one [`EnrollRequest`]/[`EnrollResponse`] exchange on `stream`, then
/// returns -- unlike the render listener's `handle_connection`, this is not a loop, since
/// both `Issue` and `Claim` are one-shot operations with nothing to pipeline.
///
/// Generic over `Read + Write` (not `TcpStream`/`EnrollTlsStream` specifically) so this
/// crate's own tests can drive it over an in-memory duplex, exactly like
/// `crate::serve::handle_connection` already does -- see this module's tests.
///
/// # Errors
///
/// A human-readable message for a transport-level failure decoding the request or
/// encoding the response. An `Issue` refused for being non-loopback, or a `Claim` that
/// doesn't match a pending enrollment, are NOT error returns -- both are reported to the
/// peer as a normal [`EnrollResponse`] and this function returns `Ok(())`, since neither
/// indicates the connection itself is broken.
pub(super) fn handle_enroll_connection<S: Read + Write>(
    mut stream: S,
    registry: &EnrollRegistry,
    pki_dir: &Path,
    allowlist_path: Option<&Path>,
    peer: Option<SocketAddr>,
) -> Result<(), String> {
    let request: EnrollRequest =
        gemray_net::messages::read_message(&mut stream).map_err(|e| e.to_string())?;

    match request {
        EnrollRequest::Issue { name } => {
            let is_loopback = peer.is_some_and(|p| p.ip().is_loopback());
            let response = if is_loopback {
                match registry.issue(pki_dir, &name) {
                    Ok((token, expires_in_secs)) => EnrollResponse::Issued {
                        token,
                        expires_in_secs,
                    },
                    Err(reason) => EnrollResponse::IssueRefused { reason },
                }
            } else {
                tracing::warn!(
                    "enrollment connection {peer:?}: refused Issue from a non-loopback peer"
                );
                EnrollResponse::IssueRefused {
                    reason: "issuing enrollment tokens is only permitted from loopback".to_string(),
                }
            };
            gemray_net::messages::write_message(&mut stream, &response).map_err(|e| e.to_string())
        }
        EnrollRequest::Claim { mut secret } => {
            let claimed = registry.claim(&secret);
            secret.zeroize_local();

            let response = if let Some(claimed) = claimed {
                let allowlisted = allowlist_path.map_or(Ok(()), |path| {
                    gemray_net::tls::append_to_allowlist(
                        path,
                        &claimed.bundle.client_fingerprint,
                        &claimed.name,
                    )
                    .map_err(|e| e.to_string())
                });
                match allowlisted {
                    Ok(()) => EnrollResponse::Claimed {
                        ca_pem: claimed.bundle.ca_pem,
                        client_cert_pem: claimed.bundle.client_cert_pem,
                        client_key_pem: claimed.bundle.client_key_pem.to_string(),
                    },
                    Err(e) => {
                        // The certificate was already minted and removed from the
                        // registry (single-use, so it's gone regardless), but it must
                        // NOT be handed to the client without also being allowlisted --
                        // an unlisted certificate would just fail the render listener's
                        // own check on first connect, confusingly far from this actual
                        // cause. Fail the claim instead, with the real reason logged
                        // here for the operator.
                        tracing::error!(
                            "enrollment connection {peer:?}: claim succeeded but appending to the allowlist \
                             failed, so it is being reported as a failed claim instead: {e}"
                        );
                        EnrollResponse::ClaimFailed
                    }
                }
            } else {
                tracing::debug!(
                    "enrollment connection {peer:?}: claim did not match any pending enrollment (wrong \
                     secret, already claimed, or expired)"
                );
                EnrollResponse::ClaimFailed
            };
            gemray_net::messages::write_message(&mut stream, &response).map_err(|e| e.to_string())
        }
    }
}

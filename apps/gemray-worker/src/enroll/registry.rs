//! The pending-entry registry: mint, hash, constant-time compare, TTL sweep, cap -- see
//! `crate::enroll`'s module doc comment for the full security rationale.

use crate::pki;
use gemray_net::{tls::Fingerprint, token};
use std::{
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use super::{MAX_PENDING, TOKEN_TTL_SECS};

// [`gemray_net::enroll::EnrollRequest`]/[`gemray_net::enroll::EnrollResponse`] (the
// enrollment wire protocol) used to be defined here, but now live in `gemray_net::enroll`
// -- see that module's doc comment for why: `apps/diagram-gui` needs the same message
// types and the same claiming client `apps/gemray-worker`'s own `cert claim` uses, and
// `gemray-net` is the crate both apps already depend on. `crate::enroll` (the server-side
// registry) imports them from there, unchanged in shape.

/// A freshly minted client-certificate bundle held in memory for one pending enrollment.
/// See the module doc comment on why `client_key_pem` is wrapped in [`Zeroizing`].
pub(super) struct EnrollBundle {
    pub(super) ca_pem: String,
    pub(super) client_cert_pem: String,
    pub(super) client_key_pem: Zeroizing<String>,
    pub(super) client_fingerprint: Fingerprint,
}

impl From<pki::InMemoryClientBundle> for EnrollBundle {
    fn from(b: pki::InMemoryClientBundle) -> Self {
        Self {
            ca_pem: b.ca_pem,
            client_cert_pem: b.client_cert_pem,
            client_key_pem: Zeroizing::new(b.client_key_pem),
            client_fingerprint: b.client_fingerprint,
        }
    }
}

pub(super) struct PendingEnrollment {
    /// `SHA-256(secret)` -- never the secret itself. See the module doc comment's
    /// point 2.
    secret_hash: Zeroizing<[u8; 32]>,
    expires_at: Instant,
    name: String,
    bundle: EnrollBundle,
}

/// The result of a successful [`EnrollRegistry::claim`]: everything
/// [`handle_enroll_connection`] needs to both reply to the claiming client and append to
/// `allowlist.txt`.
pub(super) struct ClaimedEnrollment {
    pub(super) name: String,
    pub(super) bundle: EnrollBundle,
}

/// All enrollments issued by this `serve` process that haven't yet been claimed or
/// expired.
///
/// Lives for exactly the lifetime of one `serve` invocation -- see the module doc
/// comment on why a restart dropping everything here is intended, not a bug.
#[derive(Default)]
pub struct EnrollRegistry {
    pub(super) pending: Mutex<Vec<PendingEnrollment>>,
}

impl EnrollRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mints a new client-certificate bundle for `name` (via
    /// [`pki::issue_client_in_memory`]) and registers it as pending, with a fresh
    /// 256-bit CSPRNG secret (via [`ring::rand::SystemRandom`], the same CSPRNG this
    /// workspace's own TLS/PKI stack -- `rustls` and `rcgen` -- already uses internally,
    /// rather than a separately hand-rolled one). Returns the encoded token (see
    /// `gemray_net::token`) and the TTL it's valid for.
    ///
    /// # Errors
    ///
    /// A human-readable message if too many enrollments are already pending (see
    /// [`MAX_PENDING`]), certificate minting fails (see
    /// [`pki::issue_client_in_memory`]'s own errors), or the CSPRNG fails (extremely
    /// rare -- see [`ring::rand::SecureRandom::fill`]'s own docs).
    pub fn issue(&self, pki_dir: &Path, name: &str) -> Result<(String, u64), String> {
        self.issue_with_ttl(pki_dir, name, Duration::from_secs(TOKEN_TTL_SECS))
    }

    /// The TTL-parameterized implementation behind [`Self::issue`] -- a real (not fake)
    /// short TTL, exercised directly by this module's own expiry tests, since actually
    /// waiting out [`TOKEN_TTL_SECS`] (180s) in a unit test would make the suite
    /// unusably slow. Not part of the public API: the TTL is fixed at [`TOKEN_TTL_SECS`]
    /// for every real caller -- see the module doc comment on why it isn't
    /// operator-configurable.
    pub(super) fn issue_with_ttl(
        &self,
        pki_dir: &Path,
        name: &str,
        ttl: Duration,
    ) -> Result<(String, u64), String> {
        let bundle = pki::issue_client_in_memory(pki_dir, name)?;

        let mut secret = [0u8; token::SECRET_LEN];
        random_bytes(&mut secret)?;
        let secret_hash = sha256(&secret);
        let encoded = token::encode(&secret, &bundle.ca_fingerprint);
        secret.zeroize_local();

        let mut pending = self.pending.lock().unwrap();
        if pending.len() >= MAX_PENDING {
            return Err(format!(
                "{MAX_PENDING} enrollments are already pending -- claim or wait for one to expire before issuing another"
            ));
        }
        pending.push(PendingEnrollment {
            secret_hash: Zeroizing::new(secret_hash),
            expires_at: Instant::now() + ttl,
            name: name.to_string(),
            bundle: EnrollBundle::from(bundle),
        });
        drop(pending);

        Ok((encoded, ttl.as_secs()))
    }

    /// Attempts to claim the pending enrollment whose secret hashes to `secret`'s hash.
    ///
    /// First sweeps every currently-pending entry, dropping (and so zeroizing -- see the
    /// module doc comment) any already past its `expires_at`, regardless of whether this
    /// particular claim would have matched it. Then checks the candidate's SHA-256
    /// against every *remaining* entry's stored hash using
    /// [`subtle::ConstantTimeEq::ct_eq`] rather than `==` -- comparing every entry (never
    /// short-circuiting on the first match) so the total comparison work doesn't itself
    /// leak which entry (if any) matched through timing. On a match, that one entry is
    /// removed from the registry before returning it -- single use, per the module doc
    /// comment.
    ///
    /// Returns `None` for "expired", "wrong secret", and "no such token" alike -- the
    /// caller ([`handle_enroll_connection`]) collapses all of these to the same
    /// [`EnrollResponse::ClaimFailed`] wire message; see that type's doc comment.
    pub(super) fn claim(&self, secret: &[u8; token::SECRET_LEN]) -> Option<ClaimedEnrollment> {
        let candidate_hash = sha256(secret);
        let now = Instant::now();

        let mut pending = self.pending.lock().unwrap();
        pending.retain(|p| p.expires_at > now); // expired entries dropped (and zeroized) here

        let mut matched_index = None;
        for (i, p) in pending.iter().enumerate() {
            let is_match: bool = candidate_hash.ct_eq(&*p.secret_hash).into();
            if is_match {
                matched_index = Some(i);
                // Deliberately no `break`: every remaining entry still gets compared, so
                // the loop's total work doesn't depend on *where* (or whether) a match
                // was found.
            }
        }

        let entry = pending.remove(matched_index?);
        drop(pending);
        Some(ClaimedEnrollment {
            name: entry.name,
            bundle: entry.bundle,
        })
    }
}

/// Fills `dest` with cryptographically secure random bytes via
/// [`ring::rand::SystemRandom`] -- `ring` is already this workspace's chosen `rustls`/
/// `rcgen` crypto provider (see the workspace root `Cargo.toml`'s `rustls` and `rcgen`
/// feature lists), so this reuses the CSPRNG already trusted to generate every key pair
/// and TLS nonce in this crate, rather than hand-rolling one or pulling in a new one.
///
/// # Errors
///
/// A human-readable message in the (extremely rare, effectively "the OS RNG is
/// unavailable") case `ring` itself reports failure.
fn random_bytes(dest: &mut [u8]) -> Result<(), String> {
    use ring::rand::SecureRandom;
    ring::rand::SystemRandom::new()
        .fill(dest)
        .map_err(|_| "system CSPRNG failed to produce randomness".to_string())
}

fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(data).into()
}

/// Small local extension so a stack-allocated `[u8; N]` secret can be explicitly
/// zeroized in place without promoting it to a heap-allocated `Zeroizing<Vec<u8>>` just
/// for that one call.
pub(super) trait ZeroizeLocal {
    fn zeroize_local(&mut self);
}

impl<const N: usize> ZeroizeLocal for [u8; N] {
    fn zeroize_local(&mut self) {
        use zeroize::Zeroize;
        self.zeroize();
    }
}

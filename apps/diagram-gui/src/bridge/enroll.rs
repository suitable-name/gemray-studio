//! Redeeming a worker's one-time enrollment token directly from the viewer, instead of
//! requiring the user to install `gemray-worker` and run `cert claim` in a terminal.
//!
//! The actual claim -- the pinned-CA TLS handshake and the wire exchange -- is
//! [`gemray_net::enroll::claim`], shared with `apps/gemray-worker`'s own `cert claim`
//! (see that module's doc comment for why the policy lives there and not duplicated in
//! either app). This module is everything specific to being the VIEWER side of that:
//! where a freshly claimed bundle gets written on disk, and how a [`ClaimError`] reads
//! as a message a person looking at a dialog can act on.
//!
//! # Where the bundle goes, and why the user never chooses
//!
//! [`bundle_dir_for`] writes into a per-worker subdirectory of the settings directory
//! (`settings::store::default_settings_path`'s parent) -- the same directory this crate
//! already writes `settings.toml` into without asking, so there is precedent for "this
//! app owns a folder here" and no new permission/location surprise. Keying the
//! subdirectory off the worker's own name (not, say, a random id) means re-claiming for
//! a worker you've already named simply overwrites that worker's own bundle in place --
//! which is the right behavior for "I re-ran `cert issue-token` because the old
//! certificate was revoked", not an edge case to guard against.
//!
//! # Why claiming runs on a background thread
//!
//! [`claim_and_write_bundle`] performs a real TCP connect and TLS handshake against a
//! host the operator typed in, which may be slow or simply unreachable -- exactly the
//! kind of blocking call `bridge::remote_render::test_connection` already keeps off the
//! Slint UI thread (see that function's own doc comment and `gui::remote::setup_worker_callbacks`'s
//! `on_test_worker_connection`, which wraps it in `std::thread::spawn` +
//! `Weak::upgrade_in_event_loop`). `gui::remote`'s claim-token callback follows the exact
//! same pattern for this function -- there is no cancellation or progress reporting
//! needed here any more than `test_connection` has any, since a claim is a single
//! request/response, not a multi-second stream.

use gemray_net::enroll::ClaimError;
use std::path::{Path, PathBuf};

/// Directory name (under the settings directory) holding every worker's claimed
/// certificate bundle, one subdirectory per worker -- kept separate from
/// `settings.toml` itself so "delete my settings" and "delete my saved certificates"
/// stay two different, individually reasonable things to do by hand if it ever comes to
/// that.
const BUNDLES_DIR_NAME: &str = "worker-certs";

/// Computes the directory a freshly claimed bundle for `worker_name` should be written
/// to, under `settings_dir` (the parent of `settings::store::default_settings_path()`).
///
/// Pure and total: every `worker_name`, including an empty one, maps to SOME valid
/// directory name -- see [`slugify`] -- so this never needs to ask the caller what to do
/// about an awkward name. Two different worker names that slugify to the same string
/// (e.g. "My Worker" and "my-worker") share a bundle directory; this is the same
/// "index/name is identity, not enforced-unique" convention
/// `settings::model::WorkerSettings`'s own doc comment already accepts for the worker
/// list itself (see that struct's `name` field), not a new relaxation introduced here.
#[must_use]
pub fn bundle_dir_for(settings_dir: &Path, worker_name: &str) -> PathBuf {
    settings_dir
        .join(BUNDLES_DIR_NAME)
        .join(slugify(worker_name))
}

/// Turns an arbitrary worker name into a filesystem-safe directory-name component:
/// lowercased, runs of anything other than an ASCII letter/digit collapsed to a single
/// `-`, with leading/trailing `-` trimmed. Falls back to `"worker"` for a name that
/// slugifies to nothing at all (empty, or entirely punctuation/whitespace/non-ASCII) --
/// this function is total, never `None`/error, since [`bundle_dir_for`] must always
/// produce somewhere to write.
fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_was_dash = true; // suppresses a leading '-'
    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            out.push('-');
            last_was_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "worker".to_string()
    } else {
        out
    }
}

/// Maps a [`ClaimError`] to the text a person redeeming a token should see.
///
/// Every variant reads as genuinely different advice, on purpose -- see [`ClaimError`]'s
/// own doc comment for why each is kept distinct at the type level. The one place this
/// function does NOT try to say more than [`ClaimError::Refused`] itself carries: the
/// server deliberately collapses "wrong token", "already claimed", and "expired" into
/// one indistinguishable outcome (see `EnrollResponse::ClaimFailed`'s doc comment in
/// `gemray-net`) so a failed claim can never be used to enumerate which of those three
/// happened -- this function names all three as possibilities rather than claiming to
/// have told them apart, because doing otherwise would either lie or quietly defeat that
/// protection.
#[must_use]
pub fn claim_error_message(err: &ClaimError) -> String {
    match err {
        ClaimError::InvalidToken(e) => format!("That doesn't look like a valid token: {e}"),
        ClaimError::InvalidAddr(msg) => format!("Enrollment address is invalid: {msg}"),
        ClaimError::Connect { addr, source } => format!(
            "Could not reach {addr}: {source}. Check the address and that the worker's \
             `serve` process is running."
        ),
        ClaimError::Handshake { addr, source } => {
            format!("Could not establish a secure connection to {addr}: {source}.")
        }
        ClaimError::CaFingerprintMismatch { addr } => format!(
            "Security warning: {addr} did not present the certificate authority this token was \
             issued for. This is not an ordinary connection problem -- it means either the \
             address is wrong, or something between you and the worker is impersonating it. \
             Do not retry against a different address without confirming it with whoever gave \
             you this token."
        ),
        ClaimError::Protocol(msg) => {
            format!("The connection was interrupted while redeeming the token: {msg}")
        }
        ClaimError::Refused => "This token was not accepted -- it may be mistyped, already \
             used, or expired (tokens are single-use and expire 180 seconds after being \
             issued). Ask for a fresh one and try again."
            .to_string(),
        ClaimError::UnexpectedResponse => {
            "The worker responded unexpectedly -- it may be running an incompatible version \
             of gemray-worker."
                .to_string()
        }
    }
}

/// Redeems `token` against the enrollment listener at `addr`, and writes the resulting
/// bundle to a fresh `bundle_dir` (created if needed) using the exact same three
/// filenames -- `ca.pem`, `client.pem`, `client.key` -- `settings::WorkerSettings::ca_path`/
/// `client_cert_path`/`client_key_path` already expect, so the caller only has to set
/// `WorkerSettings::cert_dir` to `bundle_dir` afterward.
///
/// Synchronous and blocking (a real TCP connect and TLS handshake) -- see this module's
/// doc comment on why the CALLER (`gui::remote`) is responsible for running this off the
/// Slint UI thread, matching `bridge::remote_render::test_connection`'s own contract.
///
/// The token itself is never logged: this function neither logs on entry, nor lets
/// [`gemray_net::enroll::claim`]'s own error path do so (it doesn't -- see that
/// function's doc comment), nor persists it anywhere; the caller is likewise expected to
/// discard it (clear the UI field) once this returns, on success or failure alike.
///
/// # Errors
///
/// [`claim_error_message`] applied to whatever [`gemray_net::enroll::claim`] returned,
/// or a message naming `bundle_dir` if creating it or writing one of the three files
/// fails.
pub fn claim_and_write_bundle(
    token: &str,
    addr: &str,
    bundle_dir: &Path,
) -> Result<PathBuf, String> {
    let bundle = gemray_net::enroll::claim(token, addr).map_err(|e| claim_error_message(&e))?;

    std::fs::create_dir_all(bundle_dir)
        .map_err(|e| format!("could not create {}: {e}", bundle_dir.display()))?;
    let ca_path = bundle_dir.join("ca.pem");
    let cert_path = bundle_dir.join("client.pem");
    let key_path = bundle_dir.join("client.key");
    std::fs::write(&ca_path, &bundle.ca_pem)
        .map_err(|e| format!("could not write {}: {e}", ca_path.display()))?;
    std::fs::write(&cert_path, &bundle.client_cert_pem)
        .map_err(|e| format!("could not write {}: {e}", cert_path.display()))?;
    gemray_net::tls::write_private_key_pem(&key_path, &bundle.client_key_pem)?;
    // `bundle.client_key_pem` (a `Zeroizing<String>`) is overwritten in place when
    // `bundle` drops at the end of this function -- see `ClaimedBundle`'s own doc
    // comment.

    Ok(bundle_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_dir_for_slugifies_the_worker_name_under_a_fixed_subdirectory() {
        let settings_dir = Path::new("/home/user/.config/diagram-gui");
        assert_eq!(
            bundle_dir_for(settings_dir, "Office Workstation"),
            settings_dir.join("worker-certs").join("office-workstation")
        );
    }

    #[test]
    fn bundle_dir_for_falls_back_to_worker_for_an_empty_or_punctuation_only_name() {
        let settings_dir = Path::new("C:/Users/me/AppData/Roaming/diagram-gui");
        assert_eq!(
            bundle_dir_for(settings_dir, ""),
            settings_dir.join("worker-certs").join("worker")
        );
        assert_eq!(
            bundle_dir_for(settings_dir, "   ---   "),
            settings_dir.join("worker-certs").join("worker")
        );
    }

    #[test]
    fn slugify_collapses_runs_of_punctuation_and_trims_edges() {
        assert_eq!(slugify("My Laptop!!"), "my-laptop");
        assert_eq!(slugify("--leading and trailing--"), "leading-and-trailing");
        assert_eq!(slugify("C++ Rig #2"), "c-rig-2");
    }

    #[test]
    fn slugify_is_stable_for_an_already_clean_name() {
        assert_eq!(slugify("laptop-2"), "laptop-2");
    }

    #[test]
    fn claim_error_message_gives_distinguishable_wording_for_each_failure_mode() {
        let addr = "worker.example:7879".to_string();
        let messages = [
            claim_error_message(&ClaimError::InvalidToken(
                gemray_net::token::decode("nope").unwrap_err(),
            )),
            claim_error_message(&ClaimError::InvalidAddr("bad address".to_string())),
            claim_error_message(&ClaimError::Connect {
                addr: addr.clone(),
                source: std::io::Error::other("connection refused"),
            }),
            claim_error_message(&ClaimError::Handshake {
                addr: addr.clone(),
                source: std::io::Error::other("boom"),
            }),
            claim_error_message(&ClaimError::CaFingerprintMismatch { addr }),
            claim_error_message(&ClaimError::Protocol("short read".to_string())),
            claim_error_message(&ClaimError::Refused),
            claim_error_message(&ClaimError::UnexpectedResponse),
        ];
        // Every one of the eight is worded differently -- no two failure modes collapse
        // to the same user-facing text.
        for i in 0..messages.len() {
            for j in (i + 1)..messages.len() {
                assert_ne!(
                    messages[i], messages[j],
                    "messages {i} and {j} are identical"
                );
            }
        }
    }

    #[test]
    fn claim_error_message_flags_the_ca_mismatch_as_security_relevant_and_distinct_from_network_errors()
     {
        let mismatch = claim_error_message(&ClaimError::CaFingerprintMismatch {
            addr: "worker.example:7879".to_string(),
        });
        let unreachable = claim_error_message(&ClaimError::Connect {
            addr: "worker.example:7879".to_string(),
            source: std::io::Error::other("connection refused"),
        });
        assert!(
            mismatch.to_lowercase().contains("security"),
            "the CA-fingerprint mismatch must not read like an ordinary network error: {mismatch}"
        );
        assert!(
            !unreachable.to_lowercase().contains("security"),
            "{unreachable}"
        );
    }

    #[test]
    fn claim_error_message_for_refused_names_expiry_and_reuse_without_claiming_to_distinguish_them()
    {
        // The server deliberately can't tell the caller which of these happened (see
        // `ClaimError::Refused`'s own doc comment) -- the message must mention the real
        // possibilities without implying it knows which one occurred.
        let msg = claim_error_message(&ClaimError::Refused).to_lowercase();
        assert!(msg.contains("expired"), "{msg}");
        assert!(msg.contains("used"), "{msg}");
    }
}

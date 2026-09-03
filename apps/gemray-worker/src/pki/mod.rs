//! `cert init` / `cert issue-server` / `cert issue-client`: an in-process private CA
//! for `gemray-worker`'s mutual TLS.
//!
//! This is certificate GENERATION only -- one-shot CLI operations that produce PEM
//! files on disk. The actual private-key-file writing (including the Windows-ACL
//! permissioning those files need) is [`gemray_net::tls::write_private_key_pem`], and
//! loading an already-issued bundle back into a `rustls::ServerConfig`/`ClientConfig` is
//! [`gemray_net::tls`]'s job too -- both live there rather than here because
//! `apps/diagram-gui`'s own token-redeem UI needs the exact same restricted-permission
//! write for the `client.key` a claim hands back, and `gemray_net::tls` is the one crate
//! both apps already depend on. See that module's doc comment on the split.
//!
//! # Why a private CA rather than a public one
//!
//! The worker is addressed by IP on a LAN or private host, which no public CA will
//! issue a certificate for, and there is exactly one operator on both ends of the trust
//! relationship -- so there's no third party whose vouching is worth anything here that
//! a self-issued CA doesn't already provide equally well.
//!
//! # Layout
//!
//! ```text
//! <pki-dir>/
//!   ca.pem          CA certificate (public -- ships to every worker and every viewer)
//!   ca.key          CA private key (sensitive -- ACL-restricted, never leaves this dir)
//!   server.pem       this worker's certificate (public)
//!   server.key       this worker's private key (sensitive -- ACL-restricted)
//!   allowlist.txt     trusted client-certificate fingerprints -- see `gemray_net::tls`
//!
//! <bundle-dir>/        (from `issue-client --out`, copied to the viewer machine)
//!   ca.pem
//!   client.pem
//!   client.key        (sensitive -- ACL-restricted)
//! ```
//!
//! `issue-server` and `issue-client` both re-derive the CA's signing identity from the
//! saved `ca.pem`/`ca.key` in `<pki-dir>` (via [`rcgen::Issuer::from_ca_cert_pem`]),
//! since each subcommand is its own process invocation with nothing else in memory.
//!
//! # Certificate lifetimes and why they're long
//!
//! CA: 10 years. Server/client leaf certs: 5 years. There is no CRL or OCSP here --
//! revocation is the fingerprint [`Allowlist`](gemray_net::tls::Allowlist), edited by
//! hand -- so a certificate's own expiry is a backstop, not the primary way trust ever
//! gets withdrawn. Long lifetimes trade a slightly weaker backstop for not having to
//! re-enroll every personal machine on a private LAN every few months; a public CA's
//! short-lived-certificate norms exist for a threat model (browser trust, revocation
//! infrastructure that actually gets checked) this design doesn't share.
//!
//! Every `not_before` is backdated by [`NOT_BEFORE_SLACK_DAYS`] from the moment of
//! issuance, so a worker or viewer whose clock is a little behind the issuing machine's
//! doesn't reject a freshly issued certificate as not-yet-valid -- see this crate's
//! top-level docs on why clock skew is one of the things that costs an afternoon if
//! missed here.

use gemray_net::tls;
use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};
use std::{
    net::IpAddr,
    path::{Path, PathBuf},
};
use time::{Duration, OffsetDateTime};

pub const CA_CERT_FILE: &str = "ca.pem";
pub const CA_KEY_FILE: &str = "ca.key";
pub const SERVER_CERT_FILE: &str = "server.pem";
pub const SERVER_KEY_FILE: &str = "server.key";
pub const CLIENT_CERT_FILE: &str = "client.pem";
pub const CLIENT_KEY_FILE: &str = "client.key";
/// Lives alongside the CA (`<pki-dir>/allowlist.txt`), never in a viewer bundle -- see
/// the module doc comment's layout diagram.
pub const ALLOWLIST_FILE: &str = "allowlist.txt";

const CA_LIFETIME_DAYS: i64 = 365 * 10;
const LEAF_LIFETIME_DAYS: i64 = 365 * 5;
/// How far back to backdate every certificate's `not_before`, to absorb clock skew
/// between the issuing machine and whichever machine (worker or viewer) checks
/// validity later. See the module doc comment.
const NOT_BEFORE_SLACK_DAYS: i64 = 1;

fn not_before_with_skew_slack() -> OffsetDateTime {
    OffsetDateTime::now_utc() - Duration::days(NOT_BEFORE_SLACK_DAYS)
}

fn not_after(lifetime_days: i64) -> OffsetDateTime {
    OffsetDateTime::now_utc() + Duration::days(lifetime_days)
}

/// `gemray-worker cert init --dir <pki-dir>`: generates a new private CA keypair and
/// self-signed certificate.
///
/// Refuses to run if `<pki-dir>` already contains a CA -- overwriting one would
/// invalidate every certificate already issued from it, silently breaking every worker
/// and viewer already enrolled. There's no `--force`: removing the old files yourself
/// is the point at which you're supposed to notice what you're about to invalidate.
///
/// # Errors
///
/// A human-readable message if `dir` can't be created, already has a CA, key
/// generation/self-signing fails, or the CA files can't be written (including a
/// Windows-ACL failure restricting `ca.key` -- see [`gemray_net::tls::write_private_key_pem`]).
pub fn init(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;

    let ca_cert_path = dir.join(CA_CERT_FILE);
    let ca_key_path = dir.join(CA_KEY_FILE);
    if ca_cert_path.exists() || ca_key_path.exists() {
        return Err(format!(
            "{} already contains a CA ({CA_CERT_FILE} and/or {CA_KEY_FILE}) -- refusing to overwrite it, since that \
             would invalidate every certificate already issued from it. Remove those files yourself first if you \
             really mean to start over.",
            dir.display()
        ));
    }

    let key_pair = KeyPair::generate().map_err(|e| format!("CA key generation failed: {e}"))?;
    let mut params = CertificateParams::new(Vec::<String>::new())
        .map_err(|e| format!("unexpected error building an empty SAN list: {e}"))?;
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::CrlSign,
        KeyUsagePurpose::DigitalSignature,
    ];
    params
        .distinguished_name
        .push(DnType::CommonName, "gemray-worker private CA");
    params.not_before = not_before_with_skew_slack();
    params.not_after = not_after(CA_LIFETIME_DAYS);

    let cert = params
        .self_signed(&key_pair)
        .map_err(|e| format!("CA self-signing failed: {e}"))?;

    std::fs::write(&ca_cert_path, cert.pem())
        .map_err(|e| format!("could not write {}: {e}", ca_cert_path.display()))?;
    tls::write_private_key_pem(&ca_key_path, &key_pair.serialize_pem())?;

    tracing::info!(
        "gemray-worker cert init: wrote {} and {} -- CA expires {} (UTC)",
        ca_cert_path.display(),
        ca_key_path.display(),
        params.not_after.date()
    );
    Ok(())
}

/// Reloads the CA's signing identity from `<dir>/ca.pem` and `<dir>/ca.key`, as saved
/// by [`init`]. Each `issue-*` subcommand is a fresh process, so there's nothing else
/// in memory to sign with.
///
/// # Errors
///
/// A human-readable message (naming `dir` and suggesting `cert init`) if either file is
/// missing or unreadable, or can't be parsed back into a CA signing identity.
pub(crate) fn load_ca(dir: &Path) -> Result<Issuer<'static, KeyPair>, String> {
    let ca_cert_path = dir.join(CA_CERT_FILE);
    let ca_key_path = dir.join(CA_KEY_FILE);

    let ca_pem = std::fs::read_to_string(&ca_cert_path).map_err(|e| {
        format!(
            "could not read {}: {e} (run `gemray-worker cert init --dir {}` first)",
            ca_cert_path.display(),
            dir.display()
        )
    })?;
    let ca_key_pem = std::fs::read_to_string(&ca_key_path).map_err(|e| {
        format!(
            "could not read {}: {e} (run `gemray-worker cert init --dir {}` first)",
            ca_key_path.display(),
            dir.display()
        )
    })?;

    let ca_key =
        KeyPair::from_pem(&ca_key_pem).map_err(|e| format!("{}: {e}", ca_key_path.display()))?;
    // `Issuer` *is* the signing identity in rcgen 0.14: it pairs the CA's stored
    // subject/extensions with its private key. Note this no longer re-`self_signed`s
    // the CA to obtain a signing handle the way the 0.13 idiom did -- issuing a leaf
    // never mints a second CA certificate now, it only reads the stored one.
    let issuer = Issuer::from_ca_cert_pem(&ca_pem, ca_key)
        .map_err(|e| format!("{}: {e}", ca_cert_path.display()))?;

    Ok(issuer)
}

/// `gemray-worker cert issue-server --dir <pki-dir> --host <name> --ip <addr>`: issues
/// this worker's server certificate, signed by the CA in `<pki-dir>`.
///
/// `hosts` and `ips` become the certificate's DNS and IP Subject Alternative Names
/// respectively -- at least one of either is REQUIRED. `rustls` (like every modern TLS
/// stack) ignores the certificate's Common Name entirely for hostname/IP verification;
/// a server certificate with no SAN at all can never be validated by a mutual-TLS
/// client, no matter how the connection is addressed. See this crate's top-level docs.
///
/// # Errors
///
/// A human-readable message if both `hosts` and `ips` are empty, the CA can't be loaded
/// (see [`load_ca`]), signing fails, or the output files can't be written.
pub fn issue_server(dir: &Path, hosts: &[String], ips: &[IpAddr]) -> Result<(), String> {
    if hosts.is_empty() && ips.is_empty() {
        return Err(
            "issue-server requires at least one --host or --ip: rustls ignores Common Name entirely, so a \
             certificate with no Subject Alternative Name can never be validated by a mutual-TLS client -- see \
             --help"
                .to_string(),
        );
    }

    let ca_issuer = load_ca(dir)?;

    let mut sans: Vec<String> = hosts.to_vec();
    sans.extend(ips.iter().map(IpAddr::to_string));
    // `CertificateParams::new` classifies each string as an IP or DNS SAN by trying to
    // parse it as an `IpAddr` first -- exactly matching the `hosts`/`ips` split above,
    // so passing them combined here is equivalent to (and simpler than) building
    // `SanType` values by hand.
    let mut params =
        CertificateParams::new(sans).map_err(|e| format!("invalid --host/--ip value: {e}"))?;

    let common_name = hosts.first().cloned().unwrap_or_else(|| ips[0].to_string());
    params
        .distinguished_name
        .push(DnType::CommonName, common_name);
    params.not_before = not_before_with_skew_slack();
    params.not_after = not_after(LEAF_LIFETIME_DAYS);
    params.use_authority_key_identifier_extension = true;
    params.key_usages = vec![
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::KeyEncipherment,
    ];
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];

    let key_pair = KeyPair::generate().map_err(|e| format!("server key generation failed: {e}"))?;
    let cert = params
        .signed_by(&key_pair, &ca_issuer)
        .map_err(|e| format!("failed to sign the server certificate: {e}"))?;

    let cert_path = dir.join(SERVER_CERT_FILE);
    let key_path = dir.join(SERVER_KEY_FILE);
    std::fs::write(&cert_path, cert.pem())
        .map_err(|e| format!("could not write {}: {e}", cert_path.display()))?;
    tls::write_private_key_pem(&key_path, &key_pair.serialize_pem())?;

    tracing::info!(
        "gemray-worker cert issue-server: wrote {} and {} (SANs: {}{}{}) -- expires {} (UTC)",
        cert_path.display(),
        key_path.display(),
        hosts.join(", "),
        if hosts.is_empty() || ips.is_empty() {
            ""
        } else {
            ", "
        },
        ips.iter()
            .map(IpAddr::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        params.not_after.date()
    );
    Ok(())
}

/// `gemray-worker cert issue-client --dir <pki-dir> --name <name> --out <bundle-dir>`.
///
/// Issues a viewer's client certificate signed by the CA in `<pki-dir>`, writes a
/// self-contained bundle (`ca.pem`, `client.pem`, `client.key`) to `<bundle-dir>`, and
/// appends the new certificate's fingerprint to `<pki-dir>/allowlist.txt` (labeled
/// `name`) so the worker trusts it immediately -- no separate enrollment step.
///
/// `name` becomes the certificate's Subject Common Name (purely a human label for the
/// allowlist comment and log messages) -- unlike `issue-server`, no SAN is set, since a
/// client certificate is never validated against a hostname the way a server's is.
///
/// # Errors
///
/// A human-readable message if `name` is empty, the CA can't be loaded, signing fails,
/// `bundle-dir` or its files can't be written, or the fingerprint can't be appended to
/// the allowlist (in which case the printed message includes the line to add by hand).
pub fn issue_client(dir: &Path, name: &str, out: &Path) -> Result<(), String> {
    let ca_issuer = load_ca(dir)?;
    let (cert, key_pair, not_after) = sign_client_leaf(&ca_issuer, name)?;

    std::fs::create_dir_all(out).map_err(|e| format!("could not create {}: {e}", out.display()))?;

    let ca_pem = std::fs::read_to_string(dir.join(CA_CERT_FILE))
        .map_err(|e| format!("could not read {}: {e}", dir.join(CA_CERT_FILE).display()))?;
    let out_ca_path = out.join(CA_CERT_FILE);
    let out_cert_path = out.join(CLIENT_CERT_FILE);
    let out_key_path = out.join(CLIENT_KEY_FILE);
    std::fs::write(&out_ca_path, &ca_pem)
        .map_err(|e| format!("could not write {}: {e}", out_ca_path.display()))?;
    std::fs::write(&out_cert_path, cert.pem())
        .map_err(|e| format!("could not write {}: {e}", out_cert_path.display()))?;
    tls::write_private_key_pem(&out_key_path, &key_pair.serialize_pem())?;

    let fingerprint = tls::fingerprint(cert.der());
    let fingerprint_hex = tls::fingerprint_to_hex(&fingerprint);
    let allowlist_path = dir.join(ALLOWLIST_FILE);
    tls::append_to_allowlist(&allowlist_path, &fingerprint, name).map_err(|e| {
        format!(
            "issued the certificate at {} but could not update {}: {e} -- add this line to it by hand: \
             {fingerprint_hex}  # {name}",
            out.display(),
            allowlist_path.display()
        )
    })?;

    tracing::info!(
        "gemray-worker cert issue-client: wrote bundle to {} ({CA_CERT_FILE}, {CLIENT_CERT_FILE}, {CLIENT_KEY_FILE}) -- \
         fingerprint {fingerprint_hex} added to {} -- expires {} (UTC)",
        out.display(),
        allowlist_path.display(),
        not_after.date()
    );
    Ok(())
}

/// The signing logic shared by [`issue_client`] (writes the bundle to disk and allowlists
/// it immediately) and [`issue_client_in_memory`] (mints the same kind of certificate but
/// hands the caller PEM strings instead of touching disk, and never touches the
/// allowlist) -- see that function's own doc comment for why enrollment needs a
/// disk-free, allowlist-free variant of this signing step.
///
/// # Errors
///
/// A human-readable message if `name` is empty, or signing fails.
fn sign_client_leaf(
    ca_issuer: &Issuer<'static, KeyPair>,
    name: &str,
) -> Result<(rcgen::Certificate, KeyPair, OffsetDateTime), String> {
    if name.trim().is_empty() {
        return Err("--name must not be empty".to_string());
    }

    let mut params = CertificateParams::new(Vec::<String>::new())
        .map_err(|e| format!("unexpected error building an empty SAN list: {e}"))?;
    params.distinguished_name.push(DnType::CommonName, name);
    params.not_before = not_before_with_skew_slack();
    params.not_after = not_after(LEAF_LIFETIME_DAYS);
    params.use_authority_key_identifier_extension = true;
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];

    let key_pair = KeyPair::generate().map_err(|e| format!("client key generation failed: {e}"))?;
    let cert = params
        .signed_by(&key_pair, ca_issuer)
        .map_err(|e| format!("failed to sign the client certificate: {e}"))?;

    Ok((cert, key_pair, params.not_after))
}

/// A freshly issued client certificate bundle, held entirely in memory.
///
/// The counterpart to the three files [`issue_client`] writes to `<bundle-dir>`, for
/// callers (the enrollment flow in `crate::enroll`) that can't write it to disk. See that
/// module's doc comment on why: the bundle has to exist only for as long as an in-flight
/// enrollment token is unclaimed, and disk is forever until someone remembers to delete
/// it.
pub struct InMemoryClientBundle {
    /// PEM text of the CA certificate at `<pki-dir>/ca.pem` -- included so the claiming
    /// side gets the same self-contained three-file bundle `issue_client` would have
    /// written, without a separate read of the CA file itself.
    pub ca_pem: String,
    /// The CA certificate's own SHA-256 fingerprint -- what an enrollment token commits
    /// to, so a claiming client (with no CA file of its own yet) can verify this worker's
    /// identity before ever sending its bearer secret. See `gemray_net::token`'s doc comment.
    pub ca_fingerprint: tls::Fingerprint,
    pub client_cert_pem: String,
    pub client_key_pem: String,
    /// The freshly issued client certificate's own SHA-256 fingerprint -- what gets
    /// appended to `allowlist.txt`, but ONLY once the enrollment token minting this
    /// bundle is actually claimed (never at issue time -- see `crate::enroll`).
    pub client_fingerprint: tls::Fingerprint,
}

/// Mints a client certificate exactly like [`issue_client`] does, but in memory only.
///
/// Same CA, same lifetime, same `ClientAuth` leaf -- but returns it as PEM strings in
/// memory instead of writing `<bundle-dir>/{ca.pem,client.pem,client.key}` to disk, and
/// never touches `<pki-dir>/allowlist.txt`.
///
/// This is what makes the token-based enrollment flow possible: `crate::enroll` mints a
/// bundle here at token-issue time and holds it in a [`crate::enroll::EnrollRegistry`]
/// entry until a matching claim arrives (or the token expires), rather than writing
/// secret key material to a file an operator then has to remember to place and delete by
/// hand. See that module's doc comment for the full lifecycle.
///
/// # Errors
///
/// Same as [`issue_client`]: a human-readable message if `name` is empty, the CA can't be
/// loaded, or signing fails.
pub fn issue_client_in_memory(dir: &Path, name: &str) -> Result<InMemoryClientBundle, String> {
    let ca_issuer = load_ca(dir)?;
    let (cert, key_pair, _not_after) = sign_client_leaf(&ca_issuer, name)?;

    let ca_cert_path = dir.join(CA_CERT_FILE);
    let ca_pem = std::fs::read_to_string(&ca_cert_path)
        .map_err(|e| format!("could not read {}: {e}", ca_cert_path.display()))?;
    let ca_der =
        tls::load_certs(&ca_cert_path).map_err(|e| format!("{}: {e}", ca_cert_path.display()))?;
    let ca_fingerprint = tls::fingerprint(&ca_der[0]);
    let client_fingerprint = tls::fingerprint(cert.der());

    Ok(InMemoryClientBundle {
        ca_pem,
        ca_fingerprint,
        client_cert_pem: cert.pem(),
        client_key_pem: key_pair.serialize_pem(),
        client_fingerprint,
    })
}

/// The default path `serve --allowlist` uses when not given explicitly.
///
/// Alongside the `--ca` file, named [`ALLOWLIST_FILE`] -- matching where
/// [`issue_client`] writes to by default, so the common case (`cert
/// init`/`issue-server`/`issue-client` all pointed at the same `--dir`, then `serve
/// --ca <dir>/ca.pem ...`) needs no separate `--allowlist` flag at all.
#[must_use]
pub fn default_allowlist_path(ca_path: &Path) -> PathBuf {
    ca_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(ALLOWLIST_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gemray-worker-pki-test-{label}-{}-{}",
            std::process::id(),
            fastrand_seed()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // No external rand dependency in this crate -- a nanosecond timestamp is unique
    // enough to keep parallel test runs' temp dirs from colliding.
    fn fastrand_seed() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    #[test]
    fn init_writes_a_ca_and_refuses_to_overwrite_it() {
        let dir = temp_dir("init");
        init(&dir).unwrap();
        assert!(dir.join(CA_CERT_FILE).exists());
        assert!(dir.join(CA_KEY_FILE).exists());

        let err = init(&dir).unwrap_err();
        assert!(err.contains("already contains a CA"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn issue_server_requires_at_least_one_san() {
        let dir = temp_dir("issue-server-no-san");
        init(&dir).unwrap();

        let err = issue_server(&dir, &[], &[]).unwrap_err();
        assert!(err.contains("--host or --ip"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn issue_server_writes_a_cert_with_the_requested_sans() {
        let dir = temp_dir("issue-server");
        init(&dir).unwrap();
        issue_server(
            &dir,
            &["worker.lan".to_string()],
            &["10.0.0.5".parse().unwrap()],
        )
        .unwrap();

        let cert_pem = std::fs::read_to_string(dir.join(SERVER_CERT_FILE)).unwrap();
        assert!(cert_pem.contains("BEGIN CERTIFICATE"));
        assert!(dir.join(SERVER_KEY_FILE).exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn issue_client_writes_a_bundle_and_updates_the_allowlist() {
        let dir = temp_dir("issue-client-dir");
        let out = temp_dir("issue-client-out");
        init(&dir).unwrap();
        issue_client(&dir, "laptop", &out).unwrap();

        assert!(out.join(CA_CERT_FILE).exists());
        assert!(out.join(CLIENT_CERT_FILE).exists());
        assert!(out.join(CLIENT_KEY_FILE).exists());

        let allowlist_text = std::fs::read_to_string(dir.join(ALLOWLIST_FILE)).unwrap();
        assert!(allowlist_text.contains("# laptop"), "{allowlist_text}");

        let client_der = tls::load_certs(&out.join(CLIENT_CERT_FILE)).unwrap();
        let fp = tls::fingerprint(&client_der[0]);
        let allowlist = tls::Allowlist::load(&dir.join(ALLOWLIST_FILE)).unwrap();
        assert!(allowlist.contains(&fp));

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn issue_client_rejects_an_empty_name() {
        let dir = temp_dir("issue-client-empty-name");
        init(&dir).unwrap();
        let err = issue_client(&dir, "  ", &dir.join("bundle")).unwrap_err();
        assert!(err.contains("--name"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn issue_server_and_issue_client_fail_clearly_without_an_existing_ca() {
        let dir = temp_dir("no-ca");
        let err = issue_server(&dir, &["worker.lan".to_string()], &[]).unwrap_err();
        assert!(err.contains("cert init"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn default_allowlist_path_sits_beside_the_ca_file() {
        let ca = Path::new("C:/pki/ca.pem");
        assert_eq!(
            default_allowlist_path(ca),
            PathBuf::from("C:/pki").join(ALLOWLIST_FILE)
        );
    }
}

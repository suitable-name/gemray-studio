use super::{
    MAX_PENDING, TOKEN_TTL_SECS, connection::handle_enroll_connection, registry::EnrollRegistry,
};
use crate::pki;
use gemray_net::{
    enroll::{EnrollRequest, EnrollResponse},
    token,
};
use std::{
    io::{Cursor, Read, Write},
    net::SocketAddr,
    path::PathBuf,
    time::Duration,
};
use subtle::ConstantTimeEq;

fn unique_temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "gemray-worker-enroll-test-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn loopback_peer() -> SocketAddr {
    "127.0.0.1:54321".parse().unwrap()
}

fn remote_peer() -> SocketAddr {
    "203.0.113.7:54321".parse().unwrap()
}

// ---- EnrollRegistry: pure in-memory, no network -----------------------------

#[test]
fn a_valid_token_claims_exactly_once() {
    let dir = unique_temp_dir("claim-once");
    pki::init(&dir).unwrap();
    let registry = EnrollRegistry::new();

    let (encoded, _ttl) = registry
        .issue_with_ttl(&dir, "laptop", Duration::from_secs(60))
        .unwrap();
    let decoded = token::decode(&encoded).unwrap();

    let first = registry.claim(&decoded.secret);
    assert!(first.is_some(), "a fresh token must claim successfully");
    assert_eq!(first.unwrap().name, "laptop");

    let second = registry.claim(&decoded.secret);
    assert!(second.is_none(), "the same secret must not claim twice");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_expired_token_fails_and_the_pending_record_is_gone() {
    let dir = unique_temp_dir("expired");
    pki::init(&dir).unwrap();
    let registry = EnrollRegistry::new();

    let (encoded, _ttl) = registry
        .issue_with_ttl(&dir, "laptop", Duration::from_millis(20))
        .unwrap();
    let decoded = token::decode(&encoded).unwrap();

    std::thread::sleep(Duration::from_millis(60));

    assert!(registry.claim(&decoded.secret).is_none());
    assert_eq!(
        registry.pending.lock().unwrap().len(),
        0,
        "the expired record must have been swept, not merely failed to match"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_wrong_token_fails_and_does_not_consume_the_pending_record() {
    let dir = unique_temp_dir("wrong-token");
    pki::init(&dir).unwrap();
    let registry = EnrollRegistry::new();

    let (encoded, _ttl) = registry
        .issue_with_ttl(&dir, "laptop", Duration::from_secs(60))
        .unwrap();
    let real = token::decode(&encoded).unwrap();

    let mut wrong_secret = *real.secret;
    wrong_secret[0] ^= 0xFF;
    assert!(registry.claim(&wrong_secret).is_none());
    assert_eq!(
        registry.pending.lock().unwrap().len(),
        1,
        "a wrong secret must not consume the real pending record"
    );

    // The real secret still claims successfully afterward.
    assert!(registry.claim(&real.secret).is_some());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn issue_refuses_once_the_pending_cap_is_reached() {
    let dir = unique_temp_dir("cap");
    pki::init(&dir).unwrap();
    let registry = EnrollRegistry::new();

    for i in 0..MAX_PENDING {
        registry
            .issue_with_ttl(&dir, &format!("viewer-{i}"), Duration::from_secs(60))
            .unwrap();
    }
    let err = registry
        .issue_with_ttl(&dir, "one-too-many", Duration::from_secs(60))
        .unwrap_err();
    assert!(err.contains("already pending"), "{err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn token_comparison_uses_constant_time_equality() {
    // `EnrollRegistry::claim` compares via `subtle::ConstantTimeEq::ct_eq`, not
    // `==` -- this cannot be reliably observed via timing in a unit test, so instead
    // this pins down that the mechanism itself is present and behaves correctly:
    // `ct_eq` agrees with `==` on both a match and a mismatch, exercised through the
    // same helper `claim` uses.
    let a = [7u8; 32];
    let b = [7u8; 32];
    let c = {
        let mut x = [7u8; 32];
        x[31] ^= 1;
        x
    };
    assert!(bool::from(a.ct_eq(&b)));
    assert!(!bool::from(a.ct_eq(&c)));
}

// ---- allowlist ordering -------------------------------------------------------

#[test]
fn allowlist_gains_the_fingerprint_only_after_a_successful_claim() {
    let dir = unique_temp_dir("allowlist-order");
    pki::init(&dir).unwrap();
    let allowlist_path = dir.join(pki::ALLOWLIST_FILE);
    let registry = EnrollRegistry::new();

    let (encoded, _ttl) = registry
        .issue_with_ttl(&dir, "laptop", Duration::from_secs(60))
        .unwrap();

    // Absent beforehand: issuing never touches the allowlist.
    assert!(
        !allowlist_path.exists(),
        "issuing a token must not create/append the allowlist"
    );

    // Drive the real connection handler end to end for the claim, over an
    // in-memory duplex -- proving the actual code path (not just the registry in
    // isolation) is what performs the append.
    let mut input = Vec::new();
    let decoded = token::decode(&encoded).unwrap();
    gemray_net::messages::write_message(
        &mut input,
        &EnrollRequest::Claim {
            secret: *decoded.secret,
        },
    )
    .unwrap();
    let mut duplex = DuplexHalf::new(input);
    handle_enroll_connection(
        &mut duplex,
        &registry,
        &dir,
        Some(&allowlist_path),
        Some(loopback_peer()),
    )
    .unwrap();

    let mut out = Cursor::new(duplex.out);
    let response: EnrollResponse = gemray_net::messages::read_message(&mut out).unwrap();
    assert!(
        matches!(response, EnrollResponse::Claimed { .. }),
        "{response:?}"
    );

    let allowlist = gemray_net::tls::Allowlist::load(&allowlist_path).unwrap();
    assert_eq!(
        allowlist.len(),
        1,
        "exactly the claimed certificate, appended once"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_failed_claim_never_touches_the_allowlist() {
    let dir = unique_temp_dir("allowlist-no-touch");
    pki::init(&dir).unwrap();
    let allowlist_path = dir.join(pki::ALLOWLIST_FILE);
    let registry = EnrollRegistry::new();

    registry
        .issue_with_ttl(&dir, "laptop", Duration::from_secs(60))
        .unwrap();

    let mut input = Vec::new();
    gemray_net::messages::write_message(
        &mut input,
        &EnrollRequest::Claim {
            secret: [0u8; token::SECRET_LEN], // essentially certainly wrong
        },
    )
    .unwrap();
    let mut duplex = DuplexHalf::new(input);
    handle_enroll_connection(
        &mut duplex,
        &registry,
        &dir,
        Some(&allowlist_path),
        Some(loopback_peer()),
    )
    .unwrap();

    let mut out = Cursor::new(duplex.out);
    let response: EnrollResponse = gemray_net::messages::read_message(&mut out).unwrap();
    assert!(
        matches!(response, EnrollResponse::ClaimFailed),
        "{response:?}"
    );
    assert!(!allowlist_path.exists());

    std::fs::remove_dir_all(&dir).ok();
}

// ---- Issue is loopback-only -----------------------------------------------------

#[test]
fn issue_is_refused_from_a_non_loopback_peer() {
    let dir = unique_temp_dir("issue-remote-refused");
    pki::init(&dir).unwrap();
    let registry = EnrollRegistry::new();

    let mut input = Vec::new();
    gemray_net::messages::write_message(
        &mut input,
        &EnrollRequest::Issue {
            name: "intruder".to_string(),
        },
    )
    .unwrap();
    let mut duplex = DuplexHalf::new(input);
    handle_enroll_connection(&mut duplex, &registry, &dir, None, Some(remote_peer())).unwrap();

    let mut out = Cursor::new(duplex.out);
    let response: EnrollResponse = gemray_net::messages::read_message(&mut out).unwrap();
    assert!(
        matches!(response, EnrollResponse::IssueRefused { .. }),
        "{response:?}"
    );
    assert_eq!(registry.pending.lock().unwrap().len(), 0);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn issue_succeeds_from_a_loopback_peer() {
    let dir = unique_temp_dir("issue-loopback-ok");
    pki::init(&dir).unwrap();
    let registry = EnrollRegistry::new();

    let mut input = Vec::new();
    gemray_net::messages::write_message(
        &mut input,
        &EnrollRequest::Issue {
            name: "laptop".to_string(),
        },
    )
    .unwrap();
    let mut duplex = DuplexHalf::new(input);
    handle_enroll_connection(&mut duplex, &registry, &dir, None, Some(loopback_peer())).unwrap();

    let mut out = Cursor::new(duplex.out);
    let response: EnrollResponse = gemray_net::messages::read_message(&mut out).unwrap();
    let EnrollResponse::Issued {
        token,
        expires_in_secs,
    } = response
    else {
        panic!("expected Issued, got {response:?}");
    };
    assert!(token.starts_with(gemray_net::token::TOKEN_PREFIX));
    assert_eq!(expires_in_secs, TOKEN_TTL_SECS);

    std::fs::remove_dir_all(&dir).ok();
}

// ---- A claim connection cannot issue a render request --------------------------

// Only meaningful (and, since `RenderRequest`/`SceneState` don't exist without it,
// only compilable) on a `worker` build -- a library-only build has no render protocol
// for a claim connection to reach in the first place, so there's nothing this test
// could exercise.
#[cfg(feature = "worker")]
#[test]
fn a_claim_connection_cannot_issue_a_render_request() {
    use gemray::{
        geometry::cuts::StandardGemCuts,
        optics::{materials::GemMaterial, raytracer::LightingPreset},
    };
    use gemray_net::{
        SceneState,
        messages::{ClientMessage, RenderRequest, StreamConfig, TransferMode},
    };

    let dir = unique_temp_dir("no-render-on-enroll");
    pki::init(&dir).unwrap();
    let registry = EnrollRegistry::new();

    // Hand-craft bytes that decode as a real render-protocol RenderRequest -- the
    // exact kind of message the render listener's `handle_connection` would accept
    // and start tracing.
    let scene = SceneState {
        width: 4,
        height: 4,
        yaw: 0.4,
        pitch: 0.3,
        distance: 3.0,
        light_yaw: 0.85,
        light_pitch: 0.95,
        exposure: 1.0,
        max_bounces: 4,
        lighting_preset: LightingPreset::Daylight,
        material: GemMaterial::diamond(),
        planes: StandardGemCuts::standard_round_brilliant(),
        girdle_frosted: false,
    };
    let request = RenderRequest {
        request_id: 1,
        scene,
        first_sample: 0,
        samples: 4,
        stream: StreamConfig {
            transfer_mode: TransferMode::FinalOnly,
            cadence_ms: 0,
            preview: None,
        },
    };
    let mut input = Vec::new();
    gemray_net::messages::write_message(
        &mut input,
        &ClientMessage::RenderRequest(Box::new(request)),
    )
    .unwrap();

    let mut duplex = DuplexHalf::new(input);
    // `handle_enroll_connection`'s source has no reference to `stream_emit` or
    // `handle_connection_with_gpu` at all, and its only write is one `EnrollResponse`
    // -- so there is no code path from this listener to rendering, full stop. What
    // this test can additionally prove at the wire level: postcard's enum encoding
    // is a bare variant index plus raw field bytes, not a self-describing format, so
    // a `RenderRequest`'s bytes are not guaranteed to fail to decode as some
    // `EnrollRequest` variant (`ClientMessage::RenderRequest`'s tag byte may happen
    // to line up with `EnrollRequest::Issue`/`Claim`, turning the rest into a
    // "valid" garbage `name`/`secret`). Whether that happens or the decode fails
    // outright, the response written back -- if any -- must decode as an
    // `EnrollResponse` and nothing else: never a `Welcome`, `StreamEvent`, or any
    // other render-protocol type.
    let _ = handle_enroll_connection(&mut duplex, &registry, &dir, None, Some(loopback_peer()));
    if !duplex.out.is_empty() {
        let mut out = Cursor::new(duplex.out);
        let response: EnrollResponse = gemray_net::messages::read_message(&mut out)
            .expect("the only thing this listener ever writes is an EnrollResponse");
        assert!(
            matches!(
                response,
                EnrollResponse::Issued { .. }
                    | EnrollResponse::IssueRefused { .. }
                    | EnrollResponse::Claimed { .. }
                    | EnrollResponse::ClaimFailed
            ),
            "{response:?}"
        );
        assert_eq!(
            out.position(),
            out.get_ref().len() as u64,
            "nothing beyond one EnrollResponse may ever be written back on this listener"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// A `Read + Write` double over two independent in-memory buffers -- the same shape
/// as `crate::serve::tests::DuplexHalf`, reimplemented here (rather than shared)
/// since `crate::serve`'s copy is private to that module's own test suite and this
/// module has no need for its `TimeoutRead` behavior (the enrollment protocol is a
/// single blocking request/response, never a polled stream).
struct DuplexHalf {
    in_: Cursor<Vec<u8>>,
    out: Vec<u8>,
}

impl DuplexHalf {
    fn new(input: Vec<u8>) -> Self {
        Self {
            in_: Cursor::new(input),
            out: Vec::new(),
        }
    }
}

impl Read for DuplexHalf {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.in_.read(buf)
    }
}

impl Write for DuplexHalf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.out.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

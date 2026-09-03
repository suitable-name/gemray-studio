use super::{
    CertClaimArgs, CertInitArgs, CertIssueClientArgs, CertIssueServerArgs, CertIssueTokenArgs,
    Command, DEFAULT_BIND, RenderArgs, ServeArgs, parse,
};
use std::path::PathBuf;

#[test]
fn no_args_is_help() {
    assert_eq!(parse(&[]).unwrap(), Command::Help);
}

#[test]
fn help_flag_short_circuits_even_with_other_stuff_present() {
    let argv = vec!["render".to_string(), "--help".to_string()];
    assert_eq!(parse(&argv).unwrap(), Command::Help);

    let argv = vec!["--help".to_string(), "render".to_string()];
    assert_eq!(parse(&argv).unwrap(), Command::Help);

    let argv = vec!["-h".to_string()];
    assert_eq!(parse(&argv).unwrap(), Command::Help);
}

#[test]
fn unknown_subcommand_is_rejected() {
    assert!(parse(&["frobnicate".to_string()]).is_err());
}

#[test]
fn parses_a_well_formed_render_command() {
    let argv = [
        "render",
        "--scene",
        "scene.json",
        "--out",
        "out.png",
        "--width",
        "3840",
        "--height",
        "2160",
        "--samples",
        "4096",
        "--threads",
        "8",
    ]
    .map(String::from);
    let cmd = parse(&argv).unwrap();
    assert_eq!(
        cmd,
        Command::Render(RenderArgs {
            scene: PathBuf::from("scene.json"),
            out: PathBuf::from("out.png"),
            width: 3840,
            height: 2160,
            samples: 4096,
            threads: 8,
            no_gpu: false,
        })
    );
}

#[test]
fn render_threads_defaults_to_zero_meaning_auto() {
    let argv = [
        "render",
        "--scene",
        "s.json",
        "--out",
        "o.png",
        "--width",
        "8",
        "--height",
        "8",
        "--samples",
        "4",
    ]
    .map(String::from);
    let Command::Render(args) = parse(&argv).unwrap() else {
        panic!("expected Render")
    };
    assert_eq!(args.threads, 0);
}

#[test]
fn render_no_gpu_flag_defaults_to_false_and_parses_when_present() {
    let argv = [
        "render",
        "--scene",
        "s.json",
        "--out",
        "o.png",
        "--width",
        "8",
        "--height",
        "8",
        "--samples",
        "4",
    ]
    .map(String::from);
    let Command::Render(args) = parse(&argv).unwrap() else {
        panic!("expected Render")
    };
    assert!(!args.no_gpu);

    let argv = [
        "render",
        "--scene",
        "s.json",
        "--out",
        "o.png",
        "--width",
        "8",
        "--height",
        "8",
        "--samples",
        "4",
        "--no-gpu",
    ]
    .map(String::from);
    let Command::Render(args) = parse(&argv).unwrap() else {
        panic!("expected Render")
    };
    assert!(args.no_gpu);
}

#[test]
fn render_rejects_missing_required_flags() {
    let argv = ["render", "--scene", "s.json"].map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn render_rejects_a_negative_width_at_the_parse_layer() {
    let argv = [
        "render",
        "--scene",
        "s.json",
        "--out",
        "o.png",
        "--width",
        "-100",
        "--height",
        "8",
        "--samples",
        "4",
    ]
    .map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn render_rejects_a_non_numeric_samples_value() {
    let argv = [
        "render",
        "--scene",
        "s.json",
        "--out",
        "o.png",
        "--width",
        "8",
        "--height",
        "8",
        "--samples",
        "lots",
    ]
    .map(String::from);
    assert!(parse(&argv).is_err());
}

fn default_serve_args() -> ServeArgs {
    ServeArgs {
        bind: DEFAULT_BIND.to_string(),
        threads: 0,
        allow_remote: false,
        ca: None,
        cert: None,
        key: None,
        allowlist: None,
        trust_any_client_cert: false,
        insecure_no_tls: false,
        no_gpu: false,
        enroll_bind: None,
        no_enroll: false,
        db: None,
    }
}

#[test]
fn parses_a_well_formed_serve_command_with_defaults() {
    let argv = ["serve".to_string()];
    assert_eq!(parse(&argv).unwrap(), Command::Serve(default_serve_args()));
}

#[test]
fn parses_serve_flags() {
    let argv = [
        "serve",
        "--bind",
        "0.0.0.0:9000",
        "--threads",
        "4",
        "--allow-remote",
    ]
    .map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::Serve(ServeArgs {
            bind: "0.0.0.0:9000".to_string(),
            threads: 4,
            allow_remote: true,
            ..default_serve_args()
        })
    );
}

#[test]
fn parses_serve_tls_flags() {
    let argv = [
        "serve",
        "--ca",
        "ca.pem",
        "--cert",
        "server.pem",
        "--key",
        "server.key",
        "--allowlist",
        "trusted.txt",
        "--trust-any-client-cert",
    ]
    .map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::Serve(ServeArgs {
            ca: Some(PathBuf::from("ca.pem")),
            cert: Some(PathBuf::from("server.pem")),
            key: Some(PathBuf::from("server.key")),
            allowlist: Some(PathBuf::from("trusted.txt")),
            trust_any_client_cert: true,
            ..default_serve_args()
        })
    );
}

#[test]
fn parses_serve_insecure_no_tls_flag() {
    let argv = ["serve", "--insecure-no-tls"].map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::Serve(ServeArgs {
            insecure_no_tls: true,
            ..default_serve_args()
        })
    );
}

#[test]
fn parses_serve_no_gpu_flag() {
    let argv = ["serve", "--no-gpu"].map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::Serve(ServeArgs {
            no_gpu: true,
            ..default_serve_args()
        })
    );
}

#[test]
fn parses_serve_db_flag() {
    let argv = ["serve", "--db", "custom.sqlite"].map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::Serve(ServeArgs {
            db: Some(PathBuf::from("custom.sqlite")),
            ..default_serve_args()
        })
    );
}

#[test]
fn serve_rejects_an_unknown_flag() {
    let argv = ["serve", "--bogus"].map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn parses_cert_init() {
    let argv = ["cert", "init", "--dir", "pki"].map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::CertInit(CertInitArgs {
            dir: PathBuf::from("pki")
        })
    );
}

#[test]
fn cert_init_requires_dir() {
    let argv = ["cert", "init"].map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn parses_cert_issue_server_with_repeated_host_and_ip() {
    let argv = [
        "cert",
        "issue-server",
        "--dir",
        "pki",
        "--host",
        "worker.lan",
        "--host",
        "worker2.lan",
        "--ip",
        "10.0.0.5",
    ]
    .map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::CertIssueServer(CertIssueServerArgs {
            dir: PathBuf::from("pki"),
            hosts: vec!["worker.lan".to_string(), "worker2.lan".to_string()],
            ips: vec!["10.0.0.5".parse().unwrap()],
        })
    );
}

#[test]
fn cert_issue_server_rejects_an_unparseable_ip() {
    let argv = ["cert", "issue-server", "--dir", "pki", "--ip", "not-an-ip"].map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn parses_cert_issue_client() {
    let argv = [
        "cert",
        "issue-client",
        "--dir",
        "pki",
        "--name",
        "laptop",
        "--out",
        "bundle",
    ]
    .map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::CertIssueClient(CertIssueClientArgs {
            dir: PathBuf::from("pki"),
            name: "laptop".to_string(),
            out: PathBuf::from("bundle"),
        })
    );
}

#[test]
fn cert_issue_client_requires_name_and_out() {
    let argv = ["cert", "issue-client", "--dir", "pki"].map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn parses_serve_enroll_flags() {
    let argv = ["serve", "--enroll-bind", "127.0.0.1:7879", "--no-enroll"].map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::Serve(ServeArgs {
            enroll_bind: Some("127.0.0.1:7879".to_string()),
            no_enroll: true,
            ..default_serve_args()
        })
    );
}

#[test]
fn parses_cert_issue_token() {
    let argv = [
        "cert",
        "issue-token",
        "--ca",
        "pki/ca.pem",
        "--admin-addr",
        "127.0.0.1:7879",
        "--name",
        "laptop",
    ]
    .map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::CertIssueToken(CertIssueTokenArgs {
            ca: PathBuf::from("pki/ca.pem"),
            admin_addr: "127.0.0.1:7879".to_string(),
            name: "laptop".to_string(),
        })
    );
}

#[test]
fn cert_issue_token_requires_ca_admin_addr_and_name() {
    let argv = ["cert", "issue-token", "--ca", "pki/ca.pem"].map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn parses_cert_claim() {
    let argv = [
        "cert",
        "claim",
        "--token",
        "GW1-XXXXX",
        "--addr",
        "127.0.0.1:7879",
        "--out",
        "bundle",
    ]
    .map(String::from);
    assert_eq!(
        parse(&argv).unwrap(),
        Command::CertClaim(CertClaimArgs {
            token: "GW1-XXXXX".to_string(),
            addr: "127.0.0.1:7879".to_string(),
            out: PathBuf::from("bundle"),
        })
    );
}

#[test]
fn cert_claim_requires_token_addr_and_out() {
    let argv = ["cert", "claim", "--token", "GW1-XXXXX"].map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn unknown_cert_sub_subcommand_is_rejected() {
    let argv = ["cert", "frobnicate"].map(String::from);
    assert!(parse(&argv).is_err());
}

#[test]
fn bare_cert_with_no_sub_subcommand_is_rejected() {
    let argv = ["cert".to_string()];
    assert!(parse(&argv).is_err());
}

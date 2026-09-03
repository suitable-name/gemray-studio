//! The actual argument-parsing logic: [`parse`] and one `parse_*` function per
//! subcommand (and cert sub-subcommand).

use super::args::{
    CertClaimArgs, CertInitArgs, CertIssueClientArgs, CertIssueServerArgs, CertIssueTokenArgs,
    Command, RenderArgs, ServeArgs,
};
use std::{net::IpAddr, path::PathBuf};

/// Parses `argv` (without the program name).
///
/// `-h`/`--help` anywhere in the arguments short-circuits to [`Command::Help`], even if
/// other required flags are missing -- asking for help should never itself produce a
/// "missing --scene" error.
///
/// # Errors
///
/// Returns a human-readable message (never [`USAGE`] itself -- callers print that
/// separately alongside the error) describing the first thing wrong with `argv`.
pub fn parse(argv: &[String]) -> Result<Command, String> {
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(Command::Help);
    }
    match argv[0].as_str() {
        "render" => parse_render(&argv[1..]).map(Command::Render),
        "serve" => parse_serve(&argv[1..]).map(Command::Serve),
        "cert" => parse_cert(&argv[1..]),
        other => Err(format!(
            "unknown subcommand {other:?} (expected \"render\", \"serve\", or \"cert\"; see --help)"
        )),
    }
}

fn parse_cert(args: &[String]) -> Result<Command, String> {
    if args.is_empty() {
        return Err("\"cert\" requires a sub-subcommand: \"init\", \"issue-server\", or \"issue-client\" (see --help)".to_string());
    }
    match args[0].as_str() {
        "init" => parse_cert_init(&args[1..]).map(Command::CertInit),
        "issue-server" => parse_cert_issue_server(&args[1..]).map(Command::CertIssueServer),
        "issue-client" => parse_cert_issue_client(&args[1..]).map(Command::CertIssueClient),
        "issue-token" => parse_cert_issue_token(&args[1..]).map(Command::CertIssueToken),
        "claim" => parse_cert_claim(&args[1..]).map(Command::CertClaim),
        other => Err(format!(
            "unknown \"cert\" sub-subcommand {other:?} (expected \"init\", \"issue-server\", \"issue-client\", \
             \"issue-token\", or \"claim\"; see --help)"
        )),
    }
}

fn parse_cert_init(args: &[String]) -> Result<CertInitArgs, String> {
    let mut dir = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                i += 1;
                dir = Some(PathBuf::from(arg_at(args, i, "--dir")?));
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?} for \"cert init\" (see --help)"
                ));
            }
        }
        i += 1;
    }
    Ok(CertInitArgs {
        dir: dir.ok_or_else(|| "\"cert init\" requires --dir <path>".to_string())?,
    })
}

fn parse_cert_issue_server(args: &[String]) -> Result<CertIssueServerArgs, String> {
    let mut dir = None;
    let mut hosts = Vec::new();
    let mut ips = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                i += 1;
                dir = Some(PathBuf::from(arg_at(args, i, "--dir")?));
            }
            "--host" => {
                i += 1;
                hosts.push(arg_at(args, i, "--host")?.to_string());
            }
            "--ip" => {
                i += 1;
                let raw = arg_at(args, i, "--ip")?;
                ips.push(
                    raw.parse::<IpAddr>()
                        .map_err(|_| format!("--ip expects an IP address, got {raw:?}"))?,
                );
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?} for \"cert issue-server\" (see --help)"
                ));
            }
        }
        i += 1;
    }
    Ok(CertIssueServerArgs {
        dir: dir.ok_or_else(|| "\"cert issue-server\" requires --dir <path>".to_string())?,
        hosts,
        ips,
    })
}

fn parse_cert_issue_client(args: &[String]) -> Result<CertIssueClientArgs, String> {
    let mut dir = None;
    let mut name = None;
    let mut out = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                i += 1;
                dir = Some(PathBuf::from(arg_at(args, i, "--dir")?));
            }
            "--name" => {
                i += 1;
                name = Some(arg_at(args, i, "--name")?.to_string());
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(arg_at(args, i, "--out")?));
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?} for \"cert issue-client\" (see --help)"
                ));
            }
        }
        i += 1;
    }
    Ok(CertIssueClientArgs {
        dir: dir.ok_or_else(|| "\"cert issue-client\" requires --dir <path>".to_string())?,
        name: name.ok_or_else(|| "\"cert issue-client\" requires --name <label>".to_string())?,
        out: out.ok_or_else(|| "\"cert issue-client\" requires --out <path>".to_string())?,
    })
}

fn parse_cert_issue_token(args: &[String]) -> Result<CertIssueTokenArgs, String> {
    let mut ca = None;
    let mut admin_addr = None;
    let mut name = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ca" => {
                i += 1;
                ca = Some(PathBuf::from(arg_at(args, i, "--ca")?));
            }
            "--admin-addr" => {
                i += 1;
                admin_addr = Some(arg_at(args, i, "--admin-addr")?.to_string());
            }
            "--name" => {
                i += 1;
                name = Some(arg_at(args, i, "--name")?.to_string());
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?} for \"cert issue-token\" (see --help)"
                ));
            }
        }
        i += 1;
    }
    Ok(CertIssueTokenArgs {
        ca: ca.ok_or_else(|| "\"cert issue-token\" requires --ca <path>".to_string())?,
        admin_addr: admin_addr
            .ok_or_else(|| "\"cert issue-token\" requires --admin-addr <host:port>".to_string())?,
        name: name.ok_or_else(|| "\"cert issue-token\" requires --name <label>".to_string())?,
    })
}

fn parse_cert_claim(args: &[String]) -> Result<CertClaimArgs, String> {
    let mut token = None;
    let mut addr = None;
    let mut out = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--token" => {
                i += 1;
                token = Some(arg_at(args, i, "--token")?.to_string());
            }
            "--addr" => {
                i += 1;
                addr = Some(arg_at(args, i, "--addr")?.to_string());
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(arg_at(args, i, "--out")?));
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?} for \"cert claim\" (see --help)"
                ));
            }
        }
        i += 1;
    }
    Ok(CertClaimArgs {
        token: token.ok_or_else(|| "\"cert claim\" requires --token <token>".to_string())?,
        addr: addr.ok_or_else(|| "\"cert claim\" requires --addr <host:port>".to_string())?,
        out: out.ok_or_else(|| "\"cert claim\" requires --out <path>".to_string())?,
    })
}

fn arg_at<'a>(args: &'a [String], i: usize, flag: &str) -> Result<&'a str, String> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_u32(flag: &str, raw: &str) -> Result<u32, String> {
    raw.parse::<u32>()
        .map_err(|_| format!("{flag} expects a non-negative integer, got {raw:?}"))
}

fn parse_render(args: &[String]) -> Result<RenderArgs, String> {
    let mut scene = None;
    let mut out = None;
    let mut width = None;
    let mut height = None;
    let mut samples = None;
    let mut threads = 0usize;
    let mut no_gpu = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--scene" => {
                i += 1;
                scene = Some(PathBuf::from(arg_at(args, i, "--scene")?));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(arg_at(args, i, "--out")?));
            }
            "--width" => {
                i += 1;
                width = Some(parse_u32("--width", arg_at(args, i, "--width")?)?);
            }
            "--height" => {
                i += 1;
                height = Some(parse_u32("--height", arg_at(args, i, "--height")?)?);
            }
            "--samples" => {
                i += 1;
                samples = Some(parse_u32("--samples", arg_at(args, i, "--samples")?)?);
            }
            "--threads" => {
                i += 1;
                threads = parse_u32("--threads", arg_at(args, i, "--threads")?)? as usize;
            }
            "--no-gpu" => no_gpu = true,
            other => {
                return Err(format!(
                    "unknown flag {other:?} for \"render\" (see --help)"
                ));
            }
        }
        i += 1;
    }

    Ok(RenderArgs {
        scene: scene.ok_or_else(|| "\"render\" requires --scene <path>".to_string())?,
        out: out.ok_or_else(|| "\"render\" requires --out <path>".to_string())?,
        width: width.ok_or_else(|| "\"render\" requires --width <pixels>".to_string())?,
        height: height.ok_or_else(|| "\"render\" requires --height <pixels>".to_string())?,
        samples: samples.ok_or_else(|| "\"render\" requires --samples <count>".to_string())?,
        threads,
        no_gpu,
    })
}

fn parse_serve(args: &[String]) -> Result<ServeArgs, String> {
    let mut bind = super::DEFAULT_BIND.to_string();
    let mut threads = 0usize;
    let mut allow_remote = false;
    let mut ca = None;
    let mut cert = None;
    let mut key = None;
    let mut allowlist = None;
    let mut trust_any_client_cert = false;
    let mut insecure_no_tls = false;
    let mut no_gpu = false;
    let mut enroll_bind = None;
    let mut no_enroll = false;
    let mut db = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                bind = arg_at(args, i, "--bind")?.to_string();
            }
            "--threads" => {
                i += 1;
                threads = parse_u32("--threads", arg_at(args, i, "--threads")?)? as usize;
            }
            "--allow-remote" => allow_remote = true,
            "--ca" => {
                i += 1;
                ca = Some(PathBuf::from(arg_at(args, i, "--ca")?));
            }
            "--cert" => {
                i += 1;
                cert = Some(PathBuf::from(arg_at(args, i, "--cert")?));
            }
            "--key" => {
                i += 1;
                key = Some(PathBuf::from(arg_at(args, i, "--key")?));
            }
            "--allowlist" => {
                i += 1;
                allowlist = Some(PathBuf::from(arg_at(args, i, "--allowlist")?));
            }
            "--trust-any-client-cert" => trust_any_client_cert = true,
            "--insecure-no-tls" => insecure_no_tls = true,
            "--no-gpu" => no_gpu = true,
            "--enroll-bind" => {
                i += 1;
                enroll_bind = Some(arg_at(args, i, "--enroll-bind")?.to_string());
            }
            "--no-enroll" => no_enroll = true,
            "--db" => {
                i += 1;
                db = Some(PathBuf::from(arg_at(args, i, "--db")?));
            }
            other => return Err(format!("unknown flag {other:?} for \"serve\" (see --help)")),
        }
        i += 1;
    }

    Ok(ServeArgs {
        bind,
        threads,
        allow_remote,
        ca,
        cert,
        key,
        allowlist,
        trust_any_client_cert,
        insecure_no_tls,
        no_gpu,
        enroll_bind,
        no_enroll,
        db,
    })
}

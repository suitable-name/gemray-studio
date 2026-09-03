//! The parsed-argument types [`super::parse::parse`] produces -- one struct per
//! subcommand, plus the [`Command`] enum that wraps them.

use std::{net::IpAddr, path::PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderArgs {
    pub scene: PathBuf,
    pub out: PathBuf,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    /// `0` means "let the OS decide" -- see `render_core::effective_thread_count`.
    /// Governs only the CPU tracer -- see `USAGE`'s RENDER `--threads` entry.
    pub threads: usize,
    /// `--no-gpu`: force the CPU tracer even on a `gpu`-feature build with a usable
    /// adapter. See `USAGE`'s RENDER `--no-gpu` entry and `gemray::renderer::gpu_backend::GpuBackend::disabled`.
    pub no_gpu: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeArgs {
    pub bind: String,
    /// `0` means "let the OS decide" -- see `render_core::effective_thread_count`.
    /// Governs only the CPU tracer -- see `USAGE`'s SERVE `--threads` entry.
    pub threads: usize,
    pub allow_remote: bool,
    /// CA certificate path. Required by `serve::run` unless `insecure_no_tls` is set --
    /// left optional here since flag PARSING doesn't know that dependency, only
    /// `serve::run`'s own validation does (matching how `bind`'s loopback/allow-remote
    /// check already works: parsed unconditionally here, validated in `serve::run`).
    pub ca: Option<PathBuf>,
    pub cert: Option<PathBuf>,
    pub key: Option<PathBuf>,
    /// Defaults to `pki::default_allowlist_path(ca)` (allowlist.txt next to `--ca`)
    /// when `None` -- see `serve::run`.
    pub allowlist: Option<PathBuf>,
    pub trust_any_client_cert: bool,
    pub insecure_no_tls: bool,
    /// `--no-gpu`: force `Backend::Cpu` and the CPU tracer for every request, even on a
    /// `gpu`-feature build with a usable adapter. See `USAGE`'s SERVE `--no-gpu` entry
    /// and `gemray::renderer::gpu_backend::GpuBackend::disabled`.
    pub no_gpu: bool,
    /// `--enroll-bind`: address for the token-based enrollment listener. Defaults (when
    /// `None`) to the same host as `bind`, one port up -- see `crate::enroll::EnrollConfig::build`.
    /// Meaningless (and unused) when `insecure_no_tls` is set.
    pub enroll_bind: Option<String>,
    /// `--no-enroll`: don't start the enrollment listener at all.
    pub no_enroll: bool,
    /// `--db <path>`: the design-library database this `serve` instance serves (see
    /// `crate::serve::library`). `None` keeps the current default behavior -- a
    /// `facet_diagrams.sqlite` resolved relative to the process's working directory
    /// (see `diagram_catalog::db::sqlite::Database::new`'s own default) -- so an
    /// existing invocation with no `--db` keeps working unchanged; this flag is an
    /// addition for long-running servers that want to point at a specific file, not a
    /// replacement for that default.
    pub db: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertInitArgs {
    pub dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertIssueServerArgs {
    pub dir: PathBuf,
    pub hosts: Vec<String>,
    pub ips: Vec<IpAddr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertIssueClientArgs {
    pub dir: PathBuf,
    pub name: String,
    pub out: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertIssueTokenArgs {
    pub ca: PathBuf,
    pub admin_addr: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertClaimArgs {
    pub token: String,
    pub addr: String,
    pub out: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Render(RenderArgs),
    Serve(ServeArgs),
    CertInit(CertInitArgs),
    CertIssueServer(CertIssueServerArgs),
    CertIssueClient(CertIssueClientArgs),
    CertIssueToken(CertIssueTokenArgs),
    CertClaim(CertClaimArgs),
    Help,
}

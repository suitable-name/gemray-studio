//! Hand-rolled argument parsing -- no external CLI-parsing crate.
//!
//! The surface here is small (two subcommands, five flags each at most) and every
//! value needs its own validation message anyway, so a dependency buys little; the
//! workspace's other binaries (`diagram-loader`, `diagram-gui`) also don't pull one in.

mod args;
mod parse;
#[cfg(test)]
mod tests;

pub use args::{
    CertClaimArgs, CertInitArgs, CertIssueClientArgs, CertIssueServerArgs, CertIssueTokenArgs,
    Command, RenderArgs, ServeArgs,
};
pub use parse::parse;

pub const DEFAULT_BIND: &str = "127.0.0.1:7878";

pub const USAGE: &str = "gemray-worker -- headless gemray render worker

USAGE:
    gemray-worker render --scene <scene.json> --out <render.png> --width <px> --height <px> --samples <n> [--threads <n>] [--no-gpu]
    gemray-worker serve  [--bind <host:port>] [--threads <n>] [--allow-remote] [--no-gpu] [--db <path>]
                         --ca <ca.pem> --cert <server.pem> --key <server.key> [--allowlist <path>] [--trust-any-client-cert]
                         [--enroll-bind <host:port>] [--no-enroll]
    gemray-worker serve  [--bind <host:port>] [--threads <n>] [--allow-remote] [--no-gpu] [--db <path>] --insecure-no-tls

    gemray-worker cert init         --dir <pki-dir>
    gemray-worker cert issue-server --dir <pki-dir> --host <name> [--host <name> ...] --ip <addr> [--ip <addr> ...]
    gemray-worker cert issue-client --dir <pki-dir> --name <label> --out <bundle-dir>
    gemray-worker cert issue-token  --ca <ca.pem> --admin-addr <host:port> --name <label>
    gemray-worker cert claim        --token <token> --addr <host:port> --out <bundle-dir>

RENDER
    Traces a scene straight to a PNG. No networking.

    --scene <path>    Path to a JSON-encoded scene (a gemray-net SceneState). Its own
                       width/height fields, if present, are ignored -- --width/--height
                       below are authoritative for the output image, so the same
                       scene.json can be re-rendered at different resolutions.
    --out <path>      Output PNG path. Parent directories are created if missing.
    --width <px>      Output image width, in pixels.
    --height <px>     Output image height, in pixels.
    --samples <n>     Total samples per pixel to trace.
    --threads <n>     CPU threads to use (default: all available cores). Only governs
                       the CPU tracer -- ignored by the GPU path itself (a single
                       compute-pipeline dispatch, not thread-parallel), so it still
                       applies whenever the GPU declines a request (no adapter,
                       --no-gpu, built without the gpu feature, or an unsupported
                       material) and this falls back to the CPU tracer.
    --no-gpu          Force the CPU tracer even if this binary was built with the gpu
                       feature and a usable adapter is present. For A/B comparison
                       against the GPU path, or a machine whose adapter misbehaves.
                       Meaningless (and harmless) on a binary built without the gpu
                       feature, which never uses the GPU regardless.

SERVE
    Serves gemray-net's read-only design-library protocol over TCP (listing/searching
    designs, fetching one in full, fetching an attachment). A build with the `worker`
    feature also accepts RenderRequests and replies with traced radiance, so a viewer
    can offload samples to this process. Requires mutual TLS by default -- both sides
    must present a certificate signed by the same private CA (see the CERT subcommands
    below for how to create one). WELCOME advertises exactly which of the two this
    instance actually supports.

    --bind <addr>            Address to listen on (default: 127.0.0.1:7878, loopback only).
    --db <path>              The design-library database to serve (see diagram-catalog).
                              Opened read-only when possible. Defaults to
                              facet_diagrams.sqlite resolved relative to the process's
                              working directory -- unchanged from before this flag
                              existed; --db is an addition for long-running servers, not
                              a replacement for that default.
    --threads <n>             CPU threads used per render request (default: all cores).
                               Only governs the CPU tracer -- see RENDER's --threads
                               entry above for why it's ignored by the GPU path itself,
                               and still applies to that request's CPU fallback.
    --no-gpu                  Force the CPU tracer for every request, even if this
                               binary was built with the gpu feature and a usable
                               adapter is present -- WELCOME then reports Backend::Cpu.
                               For A/B comparison, or a machine whose adapter misbehaves.
    --allow-remote            Required to bind any non-loopback address, TLS or not.
    --ca <path>               CA certificate that issued both --cert and every trusted
                               client certificate. Required unless --insecure-no-tls.
    --cert <path>             This worker's own certificate (see `cert issue-server`).
    --key <path>              This worker's own private key.
    --allowlist <path>        SHA-256 fingerprints of trusted client certificates, one
                               per line (see `cert issue-client`). Defaults to
                               allowlist.txt next to --ca.
    --trust-any-client-cert   Skip the fingerprint allowlist -- trust any client whose
                               certificate chains to --ca. Off by default: the allowlist
                               is what actually decides which signed clients may connect,
                               not just which CA signed them, so skipping it is an
                               explicit, visible choice, never a silent default.
    --insecure-no-tls         Serve plaintext, no TLS, no authentication at all. Refused
                               on a non-loopback --bind. A warning is logged for every
                               connection accepted this way. For local debugging only.
    --enroll-bind <host:port> Address for the token-based enrollment listener (see CERT
                               ISSUE-TOKEN/CLAIM below). Defaults to the same host as
                               --bind, one port up. Ignored with --insecure-no-tls (there
                               is no CA to enroll against). Follows the same loopback /
                               --allow-remote gate as --bind.
    --no-enroll               Don't start the enrollment listener at all -- the manual
                               `cert issue-client` + copy-the-bundle path (see CERT below)
                               still works, this just opens no extra port for it.

CERT
    Manages an in-process private CA for SERVE's mutual TLS. There is no CRL or OCSP --
    revoking a client is deleting its line from the allowlist file `issue-client` (or a
    successful `cert claim`) wrote to (see --allowlist above).

    cert init          Generates a new CA keypair and self-signed certificate in --dir.
                        Refuses to run if --dir already has one (regenerating it would
                        invalidate every certificate already issued from it).

    cert issue-server   Issues this worker's certificate, signed by the CA in --dir.
                        --host/--ip become the certificate's Subject Alternative Names
                        and may each be repeated; at least one of either is required --
                        TLS ignores Common Name entirely, so a viewer connecting by IP
                        needs an IP SAN specifically, not just a DNS name.

    cert issue-client   Issues one viewer's certificate, signed by the CA in --dir, and
                        writes a self-contained bundle (ca.pem, client.pem, client.key)
                        to --out for copying to that viewer's machine BY HAND. --name
                        labels the certificate's subject and the allowlist entry this
                        also adds immediately. Still here, unchanged, for air-gapped or
                        offline setups where nothing can dial out to a running `serve`.

    cert issue-token    Asks a RUNNING `serve` process's enrollment listener to mint a
                        one-time, 180-second enrollment token for a viewer named --name,
                        and prints it for the operator to read or send to whoever is
                        enrolling. Connects using --ca (the operator's own already-
                        possessed CA file -- ordinary TLS verification, no bootstrap
                        problem on this side) to --admin-addr, which the running `serve`
                        process logged at startup (see --enroll-bind above) and only
                        honors from a loopback connection, regardless of --admin-addr.
                        Mints the certificate bundle immediately but holds it in that
                        `serve` process's memory only -- nothing is written to disk, and
                        nothing is added to the allowlist, until the token is claimed.

    cert claim          Redeems a token from `cert issue-token`, on the machine being
                        enrolled: connects to --addr (the enrollment listener), verifies
                        it presents a certificate chain rooted at the CA fingerprint the
                        token itself carries -- BEFORE sending the token's secret, so an
                        attacker who isn't the real worker can't collect a valid client
                        certificate by impersonating it -- and on success writes the same
                        three-file bundle (ca.pem, client.pem, client.key) to --out that
                        `cert issue-client` would have, so diagram-gui's
                        WorkerSettings.cert_dir keeps working unchanged either way. Only
                        works once per token: a second `cert claim` with the same token
                        fails, whether or not the first one succeeded.

    --dir <path>       The private CA's directory (ca.pem, ca.key, allowlist.txt).
    --host <name>       (issue-server, repeatable) a DNS Subject Alternative Name.
    --ip <addr>         (issue-server, repeatable) an IP Subject Alternative Name.
    --name <label>      (issue-client, issue-token) a human label for the certificate
                        and allowlist entry.
    --out <path>        (issue-client, claim) where to write the viewer's certificate
                        bundle.
    --ca <path>         (issue-token) the CA certificate to verify the enrollment
                        listener against -- ordinary verification, not pinned, since the
                        operator already has this file.
    --admin-addr <addr> (issue-token) the running worker's enrollment listener address.
    --token <token>     (claim) the token text printed by `cert issue-token`.
    --addr <addr>       (claim) the worker's enrollment listener address.
";

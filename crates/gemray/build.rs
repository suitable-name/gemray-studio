//! Computes a stable content hash of this crate's own `src/**/*.rs` and `src/**/*.wgsl`
//! tree (plus `Cargo.toml`) and exposes it to `lib.rs` as the `GEMRAY_BUILD_ID` env var,
//! which `lib.rs` re-exports as `gemray::BUILD_ID`.
//!
//! # Why this exists
//!
//! `gemray-net`'s wire protocol handshake refuses to pair a viewer and a remote worker
//! whose `gemray` builds disagree -- see `gemray-net`'s `handshake` module. Mixing
//! samples traced by two different physics implementations (say, one with a fix to the
//! spectral MIS weighting and one without) produces a silently, plausibly wrong image:
//! no crash, no error, just numbers that look like a render and aren't. This crate's
//! physics code has changed many times in rapid succession during development, which is
//! exactly the situation a version number can't catch (there isn't one yet) and a naive
//! "protocol version" field can't catch either (the wire format didn't change, the
//! *physics* did).
//!
//! # Why a content hash, not `git describe`
//!
//! This repository has a `.git` directory but no commits yet, so `git describe` /
//! `git rev-parse HEAD` both fail unconditionally -- not a fallback-worthy edge case,
//! the only case. A content hash of the source itself works regardless of VCS state,
//! and (unlike a commit hash) also flags the case a real deployment most needs to
//! catch: uncommitted local edits that a `git`-based check would silently treat as
//! whatever the last commit was.
//!
//! # Why not `DefaultHasher`
//!
//! `std::collections::hash_map::DefaultHasher`'s output is explicitly NOT guaranteed
//! stable across Rust versions -- two machines building the same source on different
//! toolchains could disagree and refuse a perfectly compatible pairing, which is worse
//! than not checking at all (a false "incompatible" that never clears). FNV-1a below is
//! a fixed, fully-specified algorithm: same input bytes always produce the same output,
//! on any platform, on any Rust version, forever. It costs about ten lines and zero
//! dependencies -- no build-dependency shows up in `cargo tree -p gemray --depth 1`,
//! so the published crate's dependency count is unaffected.
//!
//! # Determinism
//!
//! - Files are visited in sorted, POSIX-normalized relative-path order (`/`, not `\`),
//!   so the hash does not depend on the host OS's directory-listing order or path
//!   separator.
//! - The relative path itself is hashed alongside each file's contents, so renaming a
//!   file changes the id even if no byte of any file's content changed.
//! - Line endings are normalized (`\r\n` -> `\n`) before hashing, so an identical
//!   checkout differing only by CRLF/LF (e.g. Windows author, Linux worker) hashes
//!   identically.

use std::{
    fs,
    path::{Path, PathBuf},
};

/// The 64-bit FNV-1a offset basis and prime -- see
/// <https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function>. Fully
/// specified, versioned nowhere, and will never change out from under this build.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Recursively collects every file under `dir`, returned as paths relative to `root`.
fn collect_files(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, root, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_path_buf());
        }
    }
}

/// POSIX-normalizes a relative path (`/` separators) so the hash is identical whether
/// computed on Windows or Linux.
fn normalize_path(rel: &Path) -> String {
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Concatenates `shaders/transport_physics.wgsl` (the single shared source of the
/// Phase 2 transport physics functions -- see that file's own header comment) ahead of
/// `spectral_transport.wgsl` and `transport_functions.wgsl` and writes each result into
/// `$OUT_DIR`, since WGSL has no `#include` and both files assume the prelude's symbols
/// are already in scope.
///
/// Deliberately writes into `$OUT_DIR` (under `target/`), never under `src/`: the
/// content hash below walks `src/**/*.wgsl` on disk, and a generated file that leaked
/// into `src/` would either get silently picked up as a second, redundant hash input
/// (double-counting text that's already hashed via its two source pieces) or -- worse,
/// if `.gitignore`d and read before this function ran on a clean checkout -- be
/// missing, making the hash nondeterministic across a `cargo clean`. Neither can
/// happen when the generated files live outside `src_dir` entirely.
///
/// `include_str!(concat!(env!("OUT_DIR"), "/..."))` in `renderer::gpu::estimator_check`
/// and `renderer::gpu::transport_check` reads these generated files back at compile
/// time (`OUT_DIR` is a cargo-provided compile-time env var, resolved by `env!`, not a
/// runtime path lookup).
fn generate_transport_shaders(manifest_dir: &Path, out_dir: &Path) {
    let shaders_dir = manifest_dir.join("src").join("renderer").join("shaders");
    let prelude = fs::read_to_string(shaders_dir.join("transport_physics.wgsl"))
        .expect("shaders/transport_physics.wgsl must exist -- see build.rs");

    for name in ["spectral_transport.wgsl", "transport_functions.wgsl"] {
        let body = fs::read_to_string(shaders_dir.join(name))
            .unwrap_or_else(|e| panic!("failed to read shaders/{name}: {e}"));
        let generated = format!(
            "// GENERATED by build.rs: shaders/transport_physics.wgsl + shaders/{name}.\n\
             // Do not edit this file -- edit the two sources above instead.\n\n{prelude}\n{body}"
        );
        let out_name = name.replace(".wgsl", ".generated.wgsl");
        fs::write(out_dir.join(&out_name), generated)
            .unwrap_or_else(|e| panic!("failed to write {out_name} to OUT_DIR: {e}"));
    }
}

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"),
    );
    let src_dir = manifest_dir.join("src");
    let cargo_toml = manifest_dir.join("Cargo.toml");
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo for build scripts"));

    generate_transport_shaders(&manifest_dir, &out_dir);

    let mut rel_paths = Vec::new();
    collect_files(&src_dir, &manifest_dir, &mut rel_paths);
    // `.rs` and `.wgsl` both carry physics this hash must fingerprint. `.rs` was always
    // included; `.wgsl` joined it once a GPU path became something a worker could
    // actually run (see `renderer::gpu`). The premise for excluding `.wgsl` -- "shader
    // source is GPU-only presentation, not the CPU `trace_spectral_ray` path a remote
    // worker actually runs" -- died the moment that stopped being true: a worker
    // running a GPU backend executes WGSL as its physics, not as decoration on top of
    // it. Two workers with byte-identical Rust but divergent WGSL would otherwise
    // handshake as identical (see `gemray-net::handshake`), which is exactly the
    // silent-divergence failure this whole hash exists to catch -- summed samples from
    // a GPU-diverged worker are numerically indistinguishable from a converged render.
    // Deliberately still ONE hash, not two: `gemray-net::handshake::verify_compatible`
    // has no "close enough" tier and this file doesn't add one either -- any change to
    // either language is a build-identity change, full stop.
    rel_paths.retain(|p| {
        p.extension()
            .is_some_and(|ext| ext == "rs" || ext == "wgsl")
    });

    // Pathological case: no `.rs` files found under `src/` at all (e.g. a stripped
    // source tarball missing it). Rather than silently emit a hash computed from
    // `Cargo.toml` alone -- which would look valid but identify nothing -- fall back to
    // an explicit sentinel. `gemray-net::handshake::parse_build_id` maps any string
    // that isn't exactly 16 hex characters (this included) to
    // `UNKNOWN_BUILD_HASH`, which its `verify_compatible` refuses unconditionally, even
    // against another `"unknown"` build.
    if rel_paths.is_empty() {
        println!("cargo:rerun-if-changed={}", src_dir.display());
        println!("cargo:rustc-env=GEMRAY_BUILD_ID=unknown");
        return;
    }

    rel_paths.push(PathBuf::from("Cargo.toml"));
    rel_paths.sort();

    let mut hash = FNV_OFFSET_BASIS;
    for rel in &rel_paths {
        let abs = manifest_dir.join(rel);
        let Ok(contents) = fs::read(&abs) else {
            continue;
        };
        // Normalize CRLF -> LF before hashing so a Windows checkout and a Linux
        // checkout of byte-identical source hash identically.
        let normalized: Vec<u8> = {
            let mut out = Vec::with_capacity(contents.len());
            let mut i = 0;
            while i < contents.len() {
                if contents[i] == b'\r' && contents.get(i + 1) == Some(&b'\n') {
                    // Drop the \r; the loop picks up the \n on the next iteration.
                } else {
                    out.push(contents[i]);
                }
                i += 1;
            }
            out
        };

        let rel_str = normalize_path(rel);
        hash = fnv1a_update(hash, rel_str.as_bytes());
        hash = fnv1a_update(hash, &[0u8]); // separator between path and contents
        hash = fnv1a_update(hash, &normalized);

        println!("cargo:rerun-if-changed={}", abs.display());
    }
    // Also react to files being added/removed (rerun-if-changed on the directory
    // itself catches that on most platforms/cargo versions).
    println!("cargo:rerun-if-changed={}", src_dir.display());
    println!("cargo:rerun-if-changed={}", cargo_toml.display());

    let build_id = format!("{hash:016x}");
    println!("cargo:rustc-env=GEMRAY_BUILD_ID={build_id}");
}

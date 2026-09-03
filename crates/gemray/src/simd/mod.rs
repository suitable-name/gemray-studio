//! Runtime-dispatched SIMD kernels (AVX2 / AVX-512, with a scalar fallback).
//!
//! Covers the two hot loops that dominate CPU time: the meet-solver's
//! candidate-vertex enumeration (`f64`) and the spectral raytracer's
//! plane-intersection and per-channel absorption math (`f32`).
//!
//! # Determinism contract
//!
//! Every kernel here is **bit-identical across dispatch levels** (scalar,
//! AVX2, AVX-512) and bit-identical to the scalar code it replaces, with one
//! documented exception:
//!
//! - The `f64` and `f32` geometry kernels replicate the exact operation order
//!   of the `glam` scalar expressions they replace -- left-associated dot
//!   products (`(x*x' + y*y') + z*z'`), `glam`'s cross/determinant/inverse
//!   sequence, separate multiply and add (no FMA, matching `glam`'s scalar
//!   arithmetic), and per-lane decisions replayed in ascending plane/triple
//!   order. ([`exp_f32x8`] is the one FMA user: its scalar lane uses
//!   `mul_add` and its vector body the fused intrinsics, so the two stay
//!   bit-identical -- which is why the dispatch levels also require the `fma`
//!   feature.) IEEE arithmetic is exactly rounded per operation, so identical
//!   operation sequences give identical bits at any vector width.
//! - [`exp_f32x8`] is a polynomial exponential that is bit-identical across
//!   its own dispatch levels but deliberately **not** identical to
//!   `f32::exp` (libm): a vectorizable exponential cannot reproduce libm
//!   bit-for-bit. Callers switching to it change results by a couple of ULP,
//!   once, uniformly across machines -- a loud, re-baselined change, per this
//!   crate's golden-test convention.
//!
//! No cross-lane floating-point reductions feed any decision: horizontal
//! steps only *select* existing lane values, with ties broken toward the
//! lowest plane index, matching the sequential scans they replace.

mod exp_poly;
mod feasibility;
mod slab;
mod triple_solve;

pub use exp_poly::exp_f32x8;
pub use feasibility::{BLANK_OWNER, Feasibility, PlanesSoA64, any_violation, classify_feasibility};
pub use slab::{PlanesSoA32, SlabScan, slab_scan};
pub use triple_solve::{TRIPLE_LANES, TripleBatch, TripleSolution, solve_triple_batch};

use std::sync::OnceLock;

/// Which instruction set the kernels dispatch to on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    Scalar,
    Avx2,
    Avx512,
}

static LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Detected dispatch level, cached for the process lifetime.
///
/// The environment variable `GEMRAY_SIMD` (read once, at first call) can cap the
/// level below what the CPU supports: `scalar`, `avx2` or `avx512`. A cap never
/// raises the level above what was detected. Because every kernel is bit-identical
/// across levels (see the module docs), capping changes only speed, never output --
/// it exists so a profile-guided build (`scripts/pgo-build.ps1`) can train the scalar
/// kernels on an AVX-capable machine, and so a level can be A/B-timed in place.
pub fn simd_level() -> SimdLevel {
    *LEVEL.get_or_init(|| {
        let detected = detect_level();
        let cap = match std::env::var("GEMRAY_SIMD").as_deref() {
            Ok("scalar") => SimdLevel::Scalar,
            Ok("avx2") => SimdLevel::Avx2,
            _ => SimdLevel::Avx512,
        };
        if rank(cap) < rank(detected) {
            cap
        } else {
            detected
        }
    })
}

const fn rank(level: SimdLevel) -> u8 {
    match level {
        SimdLevel::Scalar => 0,
        SimdLevel::Avx2 => 1,
        SimdLevel::Avx512 => 2,
    }
}

fn detect_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return SimdLevel::Avx512;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

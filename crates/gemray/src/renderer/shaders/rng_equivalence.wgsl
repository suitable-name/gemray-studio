// RNG/integer bit-exactness self-test kernel (Phase 0 deliverable 3) -- driven by
// `renderer::gpu::rng_check`, NOT a physics kernel.
//
// Ports exactly five pieces of `gemray`'s deterministic-sampling machinery to WGSL, each
// bit-for-bit against its Rust source (WGSL `u32` arithmetic is exact and wraps modulo
// 2^32 the same way Rust's `wrapping_*` methods do, so "bit-exact" is the actual bar,
// not "close"):
//   1. `optics::raytracer::hash_u32` (the integer hash everything else below is built
//      from).
//   2. `apps/gemray-worker/src/render_core.rs::trace_samples`'s per-sample seed formula.
//   3. Fix 4's stratified pixel-jitter and hero-wavelength draw
//      (`optics::raytracer::{low_discrepancy_base2, cranley_patterson_rotate}`, and the
//      `PIXEL_JITTER_X_ROTATION_STREAM`/`PIXEL_JITTER_Y_ROTATION_STREAM`/
//      `HERO_WAVELENGTH_ROTATION_STREAM` salts), replacing the old unstratified
//      `hash_u32(seed) % 10000` jitter and `hash_u32(seed)` hero draw.
//   4. `optics::raytracer::trace_spectral_ray`'s four salted per-bounce draws
//      (`FRESNEL_BRANCH_STREAM`/`RUSSIAN_ROULETTE_STREAM`/`BIREFRINGENT_SPLIT_STREAM`/
//      `MODE_COUPLING_STREAM`, see that file's own doc comment on the construction) --
//      still seeded from `seed` exactly as before; Fix 4 does not touch these. Task 1
//      added `MODE_COUPLING_STREAM`/`mode_coupling_draws`.
//   5. `optics::raytracer::wrapped_hero_wavelengths`'s hero-wavelength comb.
//
// # `%` vs `rem_euclid` (see `wrapped_hero_wavelengths`'s own doc comment for the CPU
// side of this same note)
//
// WGSL's `%` operator is NOT `rem_euclid` for a negative left operand -- it is a
// truncating remainder (result takes the sign of the dividend), the same as Rust's bare
// `%` on floats. `wrapped_hero_wavelengths` uses `rem_euclid` specifically because its
// CPU call site cannot otherwise prove `offset` stays non-negative for every `k`. This
// kernel's `offset` is provably non-negative by construction: `lambda_hero >= 380.0`
// (so `lambda_hero - SPECTRUM_MIN >= 0.0`) and `k * channel_width >= 0.0` for every
// `k >= 0`, so their sum is always >= 0.0. For a non-negative dividend and a positive
// divisor, truncating `%` and `rem_euclid` agree exactly -- so plain `%` is safe to use
// here, but ONLY because of that invariant; it would silently diverge from the CPU's
// `rem_euclid` the moment this were reused somewhere `offset` could go negative.
//
// # `fma`, not `a * b + c` -- and a measured caveat
//
// The CPU formula uses `f32::mul_add` (a true fused multiply-add: one rounding, not
// two) in two places. WGSL's `fma()` builtin carries the same single-rounding guarantee
// on paper (https://www.w3.org/TR/WGSL/#fma-builtin: "computed to infinite precision and
// then rounded once"), so this kernel uses `fma()` at both call sites rather than a bare
// `*`/`+` pair.
//
// MEASURED on this workspace's dev hardware (AMD Radeon 680M-class RDNA2 iGPU, Vulkan
// backend), prior to Fix 4: the integer-only fields (`seed`, `jx_raw`, `jy_raw`,
// `hero_hash`, all three per-bounce draws) came back byte-for-byte identical to the CPU
// across a 1,048,576-tuple run -- exactly zero mismatches. `lambdas` (the only field
// built from `fma`) did not: roughly 0.02% of records differed from the CPU by 1 ULP in
// one or more channels. This is consistent with the AMD/Vulkan shader compiler lowering
// `fma()` to a non-fused multiply-add (a real, hardware/driver-level choice `wgpu`
// exposes no portable knob to override) rather than an actual disagreement in this
// port's algebra -- `renderer::gpu::rng_check::cpu_record` calls the exact same Rust
// functions this file was translated from, so there is no separate "reference
// implementation" to have gotten wrong. Fix 4 extends the same float/integer split to
// the new `jx`/`jy`/`hero_rand` fields (Tier 2, alongside `lambdas`) while keeping their
// integer precursors (`rot_jx_hash`/`rot_jy_hash`/`rot_hero_hash`/`sample_reversed`) at
// Tier 1's zero tolerance -- see `renderer::gpu::rng_check`'s module doc comment.

struct RngRecord {
    seed: u32,
    rot_jx_hash: u32,
    rot_jy_hash: u32,
    rot_hero_hash: u32,
    sample_reversed: u32,
    jx: f32,
    jy: f32,
    hero_rand: f32,
    lambdas: array<f32, 8>,
    fresnel_draws: array<u32, 4>,
    rr_draws: array<u32, 4>,
    biref_draws: array<u32, 4>,
    mode_coupling_draws: array<u32, 4>,
    // Task 2 (girdle finish): the Tier 1 integer draws for apply_frosted_bounce's 2D
    // cosine-weighted-hemisphere direction sample -- see optics::raytracer::
    // {FROSTED_DIR_U_STREAM, FROSTED_DIR_V_STREAM}.
    frosted_dir_u_draws: array<u32, 4>,
    frosted_dir_v_draws: array<u32, 4>,
}

struct Params {
    num_samples: u32,
    /// Must be <= 4 -- `RngRecord`'s three per-bounce arrays are fixed at capacity 4.
    /// The Rust-side harness (`renderer::gpu::rng_check`) enforces this before dispatch;
    /// this kernel does not re-check it (an out-of-bounds `b` here would be a
    /// caller bug, not a runtime input to validate).
    num_bounces: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_records: array<RngRecord>;

const SPECTRUM_MIN: f32 = 380.0;
const SPECTRUM_SPAN: f32 = 400.0; // 780.0 - 380.0
const NUM_CHANNELS: u32 = 8u;

// Same four stream salts as `optics::raytracer::{FRESNEL_BRANCH_STREAM,
// RUSSIAN_ROULETTE_STREAM, BIREFRINGENT_SPLIT_STREAM, MODE_COUPLING_STREAM}`.
const FRESNEL_BRANCH_STREAM: u32 = 0x9E3779B1u;
const RUSSIAN_ROULETTE_STREAM: u32 = 0x517CC1B7u;
const BIREFRINGENT_SPLIT_STREAM: u32 = 0x2545F491u;
const MODE_COUPLING_STREAM: u32 = 0xCC9E2D51u;
// Task 2 (girdle finish): optics::raytracer::{FROSTED_DIR_U_STREAM, FROSTED_DIR_V_STREAM}.
const FROSTED_DIR_U_STREAM: u32 = 0x27D4EB2Fu;
const FROSTED_DIR_V_STREAM: u32 = 0x165667B1u;

// Same three stream salts as `optics::raytracer::{PIXEL_JITTER_X_ROTATION_STREAM,
// PIXEL_JITTER_Y_ROTATION_STREAM, HERO_WAVELENGTH_ROTATION_STREAM}` (Fix 4).
const PIXEL_JITTER_X_ROTATION_STREAM: u32 = 0xA511E9B3u;
const PIXEL_JITTER_Y_ROTATION_STREAM: u32 = 0x63D81B23u;
const HERO_WAVELENGTH_ROTATION_STREAM: u32 = 0x1B873593u;

fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x * 0x85ebca6bu;
    x = x ^ (x >> 13u);
    x = x * 0xc2b2ae35u;
    x = x ^ (x >> 16u);
    return x;
}

// optics::raytracer::{low_discrepancy_base2, radical_inverse_base,
// cranley_patterson_rotate} (Fix 4). jx/jy/hero_rand use bases 2/3/5 respectively --
// see `optics::raytracer`'s Fix 4 section doc comment for why (measured: same base for
// all three made variance WORSE, not better, for exactly the highest-variance pixels).
fn low_discrepancy_base2(n: u32) -> f32 {
    return f32(reverseBits(n)) / 4294967296.0;
}

fn radical_inverse_base(n_in: u32, base: u32) -> f32 {
    var n = n_in;
    var val: f32 = 0.0;
    var inv_base: f32 = 1.0 / f32(base);
    loop {
        if (n == 0u) {
            break;
        }
        let digit = n % base;
        val = fma(f32(digit), inv_base, val);
        inv_base = inv_base / f32(base);
        n = n / base;
    }
    return val;
}

fn cranley_patterson_rotate(x: f32, offset: f32) -> f32 {
    let sum = x + offset;
    return sum - floor(sum);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&out_records)) {
        return;
    }
    let pixel = idx / params.num_samples;
    let sample = idx % params.num_samples;

    // apps/gemray-worker/src/render_core.rs::trace_samples's seed formula -- still
    // seeds every per-bounce draw below unchanged; Fix 4 does not touch it.
    let seed = hash_u32((pixel * 0x9e3779b9u) ^ (sample * 0x85ebca6bu));

    // Fix 4: stratified pixel jitter and hero wavelength.
    let rot_jx_hash = hash_u32(pixel ^ PIXEL_JITTER_X_ROTATION_STREAM);
    let rot_jy_hash = hash_u32(pixel ^ PIXEL_JITTER_Y_ROTATION_STREAM);
    let rot_hero_hash = hash_u32(pixel ^ HERO_WAVELENGTH_ROTATION_STREAM);
    let sample_reversed = reverseBits(sample);

    var rec: RngRecord;
    rec.seed = seed;
    rec.rot_jx_hash = rot_jx_hash;
    rec.rot_jy_hash = rot_jy_hash;
    rec.rot_hero_hash = rot_hero_hash;
    rec.sample_reversed = sample_reversed;

    let rot_jx = low_discrepancy_base2(rot_jx_hash);
    let rot_jy = low_discrepancy_base2(rot_jy_hash);
    let rot_hero = low_discrepancy_base2(rot_hero_hash);
    let jx = cranley_patterson_rotate(low_discrepancy_base2(sample), rot_jx) - 0.5;
    let jy = cranley_patterson_rotate(radical_inverse_base(sample, 3u), rot_jy) - 0.5;
    let hero_rand = cranley_patterson_rotate(radical_inverse_base(sample, 5u), rot_hero);
    rec.jx = jx;
    rec.jy = jy;
    rec.hero_rand = hero_rand;

    let channel_width = SPECTRUM_SPAN / f32(NUM_CHANNELS);
    let lambda_hero = fma(hero_rand, SPECTRUM_SPAN, SPECTRUM_MIN);
    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
        let offset = fma(f32(k), channel_width, lambda_hero - SPECTRUM_MIN);
        // Invariant documented above: `offset` is always >= 0.0 here, so plain `%`
        // agrees with the CPU's `rem_euclid`.
        let wrapped = offset % SPECTRUM_SPAN;
        rec.lambdas[k] = SPECTRUM_MIN + wrapped;
    }

    for (var b: u32 = 0u; b < params.num_bounces; b = b + 1u) {
        rec.fresnel_draws[b] = hash_u32(seed ^ hash_u32(b ^ FRESNEL_BRANCH_STREAM));
        rec.rr_draws[b] = hash_u32(seed ^ hash_u32(b ^ RUSSIAN_ROULETTE_STREAM));
        rec.biref_draws[b] = hash_u32(seed ^ hash_u32(b ^ BIREFRINGENT_SPLIT_STREAM));
        rec.mode_coupling_draws[b] = hash_u32(seed ^ hash_u32(b ^ MODE_COUPLING_STREAM));
        rec.frosted_dir_u_draws[b] = hash_u32(seed ^ hash_u32(b ^ FROSTED_DIR_U_STREAM));
        rec.frosted_dir_v_draws[b] = hash_u32(seed ^ hash_u32(b ^ FROSTED_DIR_V_STREAM));
    }

    out_records[idx] = rec;
}

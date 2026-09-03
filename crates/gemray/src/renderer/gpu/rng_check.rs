//! RNG/integer bit-exactness self-test (Phase 0 deliverable 3, Tier 1), plus a Tier 2
//! ULP-budget check for the fields in the same dispatch that aren't integer-derived.
//!
//! Dispatches `shaders/rng_equivalence.wgsl` over ~10^6 `(pixel, sample, bounce)` tuples
//! and compares every value against the identical computation performed on the CPU with
//! this crate's real `hash_u32`/seed-formula/`low_discrepancy_base2`/
//! `cranley_patterson_rotate`/`wrapped_hero_wavelengths` code (not a reimplementation of
//! them -- see [`cpu_record`] below, which calls straight into `optics::raytracer` and
//! `apps/gemray-worker`'s own seed formula, copied verbatim per that formula's own "keep
//! in sync" doc comment).
//!
//! # Two tiers, two tolerances, in one dispatch
//!
//! [`compare_record`] (Tier 1, [`RngCheckResult`]) covers `seed`, `rot_jx_hash`,
//! `rot_jy_hash`, `rot_hero_hash`, `sample_reversed`, and the three per-bounce stream
//! draws -- all pure `u32` arithmetic (`rot_*_hash`/`sample_reversed` are the
//! integer building blocks of the stratified pixel-jitter/hero-wavelength draw --
//! `hash_u32(pixel ^ ROTATION_STREAM)` and `sample.reverse_bits()` respectively -- the
//! same role the old `jx_raw`/`jy_raw`/`hero_hash` fields played for the unstratified
//! formula they replace). WGSL `u32` arithmetic is exact, so this tier's tolerance is
//! zero: any disagreement at all is a bug, not noise to average away.
//!
//! [`check_float_ulp`] (Tier 2, [`FloatUlpResult`]) covers `jx`, `jy`, `hero_rand`, and
//! `lambdas` -- every `RngRecord` field built from float arithmetic (a division to turn
//! an integer into `[0, 1)`, the Cranley-Patterson rotation's add/floor/subtract, or
//! `fma`) rather than pure integer ops. These do NOT belong in Tier 1: measured on this
//! workspace's dev hardware (AMD Radeon 680M-class RDNA2 iGPU, Vulkan backend), the
//! AMD/Vulkan shader compiler does not always fuse WGSL's `fma()` into a true
//! single-rounding hardware FMA the way `f32::mul_add` is guaranteed to on the CPU side
//! -- `wgpu` exposes no portable knob to force this. That is a real, measured
//! small-ULP-scale discrepancy, not a porting bug, so holding it to Tier 1's zero
//! tolerance would make this check permanently red for a reason unrelated to
//! correctness. See [`FLOAT_ULP_BUDGET`]'s doc comment for the budget and its
//! justification.

use crate::{
    optics::raytracer::{
        BIREFRINGENT_SPLIT_STREAM, FRESNEL_BRANCH_STREAM, FROSTED_DIR_U_STREAM,
        FROSTED_DIR_V_STREAM, HERO_WAVELENGTH_ROTATION_STREAM, MODE_COUPLING_STREAM,
        PIXEL_JITTER_X_ROTATION_STREAM, PIXEL_JITTER_Y_ROTATION_STREAM, RUSSIAN_ROULETTE_STREAM,
        cranley_patterson_rotate, hash_u32, low_discrepancy_base2, radical_inverse_base,
        wrapped_hero_wavelengths,
    },
    renderer::gpu::compute,
};

const SHADER_SRC: &str = include_str!("../shaders/rng_equivalence.wgsl");

/// Must match `shaders/rng_equivalence.wgsl`'s `RngRecord` field-for-field.
///
/// See that file's own layout note: every field here is a plain scalar/array of
/// scalars, so there is no vec3/vec4 alignment pitfall to worry about -- Rust's natural
/// `#[repr(C)]` packing already agrees with WGSL's rules for this specific field set.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RngRecord {
    pub seed: u32,
    /// `hash_u32(pixel ^ PIXEL_JITTER_X_ROTATION_STREAM)`: the integer building
    /// block of the pixel-jitter-X Cranley-Patterson rotation, Tier 1 (pure integer).
    pub rot_jx_hash: u32,
    /// As [`Self::rot_jx_hash`], for `PIXEL_JITTER_Y_ROTATION_STREAM`.
    pub rot_jy_hash: u32,
    /// As [`Self::rot_jx_hash`], for `HERO_WAVELENGTH_ROTATION_STREAM`.
    pub rot_hero_hash: u32,
    /// `sample.reverse_bits()`: the base-2 van der Corput term index, Tier 1
    /// (pure integer -- the bit reversal itself, before the `/ 2^32` that turns it into
    /// a float).
    pub sample_reversed: u32,
    /// The stratified pixel-jitter-X draw actually fed to `Camera::generate_ray`
    ///. Tier 2 (float: division + a Cranley-Patterson add/floor/subtract).
    pub jx: f32,
    /// As [`Self::jx`], for pixel-jitter-Y.
    pub jy: f32,
    /// The stratified hero-wavelength draw actually fed to `wrapped_hero_wavelengths`
    ///. Tier 2, same reasoning as [`Self::jx`].
    pub hero_rand: f32,
    pub lambdas: [f32; 8],
    pub fresnel_draws: [u32; 4],
    pub rr_draws: [u32; 4],
    pub biref_draws: [u32; 4],
    /// `hash_u32(seed ^ hash_u32(bounce ^ MODE_COUPLING_STREAM))`: the Tier 1
    /// integer draw for the o<->e (uniaxial) / mode-A<->mode-B (biaxial) re-coupling
    /// decision at an internal reflection -- see
    /// `optics::raytracer::apply_internal_mode_coupling`'s doc comment.
    pub mode_coupling_draws: [u32; 4],
    /// Girdle finish: the Tier 1 integer draws for
    /// `apply_frosted_bounce`'s 2D cosine-weighted-hemisphere direction sample -- see
    /// `optics::raytracer::FROSTED_DIR_U_STREAM`.
    pub frosted_dir_u_draws: [u32; 4],
    /// As [`Self::frosted_dir_u_draws`], for `optics::raytracer::FROSTED_DIR_V_STREAM`.
    pub frosted_dir_v_draws: [u32; 4],
}

const _: () = assert!(size_of::<RngRecord>() == 160);

/// Must match `shaders/rng_equivalence.wgsl`'s `Params` uniform.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    num_samples: u32,
    num_bounces: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Number of bounces exercised per `(pixel, sample)` pair. Fixed at 4 to match
/// `RngRecord`'s fixed-capacity per-bounce arrays (see `shaders/rng_equivalence.wgsl`'s
/// `Params::num_bounces` doc comment).
pub const NUM_BOUNCES: u32 = 4;

/// Computes the exact CPU-side equivalent of one GPU `RngRecord`.
///
/// Calls straight into `optics::raytracer`'s real hashing/wavelength functions -- this
/// is deliberately NOT a parallel reimplementation, since a bug shared between "the real
/// code" and "the thing that checks the real code" would never be caught by comparing
/// them.
#[must_use]
pub fn cpu_record(pixel: u32, sample: u32) -> RngRecord {
    let seed = hash_u32(pixel.wrapping_mul(0x9e37_79b9) ^ sample.wrapping_mul(0x85eb_ca6b));

    // Stratified pixel jitter and hero wavelength, each on a DIFFERENT prime
    // base (2, 3, 5) -- see `optics::raytracer::sampling`'s doc comments for why
    // (measured: using the same base for all three made variance WORSE for the
    // highest-variance pixels). See `apps/gemray-worker/src/render_core.rs::trace_samples`
    // for the production formula this mirrors.
    let rot_jx_hash = hash_u32(pixel ^ PIXEL_JITTER_X_ROTATION_STREAM);
    let rot_jy_hash = hash_u32(pixel ^ PIXEL_JITTER_Y_ROTATION_STREAM);
    let rot_hero_hash = hash_u32(pixel ^ HERO_WAVELENGTH_ROTATION_STREAM);
    // Tier 1 integer precursor for `jx` (base 2) only -- `jy`/`hero_rand` (bases 3 and
    // 5) have no comparably cheap pure-integer precursor to expose separately, since
    // `radical_inverse_base`'s digit extraction is interleaved with float accumulation
    // rather than resolving to one final integer the way bit-reversal does.
    let sample_reversed = sample.reverse_bits();

    let rot_jx = low_discrepancy_base2(rot_jx_hash);
    let rot_jy = low_discrepancy_base2(rot_jy_hash);
    let rot_hero = low_discrepancy_base2(rot_hero_hash);
    let jx = cranley_patterson_rotate(low_discrepancy_base2(sample), rot_jx) - 0.5;
    let jy = cranley_patterson_rotate(radical_inverse_base(sample, 3), rot_jy) - 0.5;
    let hero_rand = cranley_patterson_rotate(radical_inverse_base(sample, 5), rot_hero);

    let lambdas: [f32; 8] = wrapped_hero_wavelengths(hero_rand);

    let mut fresnel_draws = [0u32; 4];
    let mut rr_draws = [0u32; 4];
    let mut biref_draws = [0u32; 4];
    let mut mode_coupling_draws = [0u32; 4];
    let mut frosted_dir_u_draws = [0u32; 4];
    let mut frosted_dir_v_draws = [0u32; 4];
    for bounce in 0..NUM_BOUNCES {
        let b = bounce;
        fresnel_draws[bounce as usize] = hash_u32(seed ^ hash_u32(b ^ FRESNEL_BRANCH_STREAM));
        rr_draws[bounce as usize] = hash_u32(seed ^ hash_u32(b ^ RUSSIAN_ROULETTE_STREAM));
        biref_draws[bounce as usize] = hash_u32(seed ^ hash_u32(b ^ BIREFRINGENT_SPLIT_STREAM));
        mode_coupling_draws[bounce as usize] = hash_u32(seed ^ hash_u32(b ^ MODE_COUPLING_STREAM));
        frosted_dir_u_draws[bounce as usize] = hash_u32(seed ^ hash_u32(b ^ FROSTED_DIR_U_STREAM));
        frosted_dir_v_draws[bounce as usize] = hash_u32(seed ^ hash_u32(b ^ FROSTED_DIR_V_STREAM));
    }

    RngRecord {
        seed,
        rot_jx_hash,
        rot_jy_hash,
        rot_hero_hash,
        sample_reversed,
        jx,
        jy,
        hero_rand,
        lambdas,
        fresnel_draws,
        rr_draws,
        biref_draws,
        mode_coupling_draws,
        frosted_dir_u_draws,
        frosted_dir_v_draws,
    }
}

/// One (pixel, sample) record's disagreement between GPU and CPU, with enough detail to
/// diagnose without re-running anything.
#[derive(Debug, Clone)]
pub struct RngMismatch {
    pub pixel: u32,
    pub sample: u32,
    pub field: &'static str,
    pub cpu: String,
    pub gpu: String,
}

/// Tier 1's zero-tolerance integer result, plus Tier 2's float-field ULP-budget result.
///
/// Both come out of the same GPU dispatch and CPU comparison loop in [`run`], so they
/// travel together here rather than requiring a caller to run the check twice.
#[derive(Debug, Clone)]
pub struct RngCheckResult {
    pub total_records: usize,
    /// Tier 1: pure-integer fields, zero tolerance.
    pub mismatches: Vec<RngMismatch>,
    /// Tier 2: `jx`/`jy`/`hero_rand`/`lambdas` against [`FLOAT_ULP_BUDGET`].
    pub float_ulp: FloatUlpResult,
}

impl RngCheckResult {
    /// Tier 1 alone: true iff every pure-integer field matched exactly.
    #[must_use]
    pub const fn tier1_passed(&self) -> bool {
        self.mismatches.is_empty()
    }

    /// Both tiers: Tier 1 exact AND Tier 2 within [`FLOAT_ULP_BUDGET`].
    #[must_use]
    pub const fn passed(&self) -> bool {
        self.tier1_passed() && self.float_ulp.passed()
    }
}

/// Tier 1 ONLY: pure-integer fields, zero tolerance. `jx`/`jy`/`hero_rand`/`lambdas` are
/// deliberately excluded here -- see [`FloatUlpAccumulator`] and this module's doc
/// comment for why they belong in Tier 2 instead.
fn compare_record(
    pixel: u32,
    sample: u32,
    cpu: &RngRecord,
    gpu: &RngRecord,
    out: &mut Vec<RngMismatch>,
) {
    macro_rules! check {
        ($field:ident, $name:literal) => {
            if cpu.$field != gpu.$field {
                out.push(RngMismatch {
                    pixel,
                    sample,
                    field: $name,
                    cpu: format!("{:?}", cpu.$field),
                    gpu: format!("{:?}", gpu.$field),
                });
            }
        };
    }
    check!(seed, "seed");
    check!(rot_jx_hash, "rot_jx_hash");
    check!(rot_jy_hash, "rot_jy_hash");
    check!(rot_hero_hash, "rot_hero_hash");
    check!(sample_reversed, "sample_reversed");
    check!(fresnel_draws, "fresnel_draws");
    check!(rr_draws, "rr_draws");
    check!(biref_draws, "biref_draws");
    check!(mode_coupling_draws, "mode_coupling_draws");
    check!(frosted_dir_u_draws, "frosted_dir_u_draws");
    check!(frosted_dir_v_draws, "frosted_dir_v_draws");
}

/// Tier 2 ULP budget for `jx`/`jy`/`hero_rand`/`lambdas` (every `RngRecord` field built
/// from float arithmetic -- a division, a Cranley-Patterson add/floor/subtract, or
/// `fma` -- rather than pure integer ops).
///
/// # Where this number comes from
///
/// Measured on this workspace's dev hardware (AMD Radeon 680M-class RDNA2 iGPU, Vulkan
/// backend) over a 1,048,576-tuple run of the ORIGINAL (pre-Fix-4, `lambdas`-only)
/// version of this check: the observed max was **1 ULP**, on 64 of 262,144 `(pixel,
/// sample)` records (~0.024%), consistent with the GPU shader compiler not always
/// fusing `fma()` into a true single-rounding hardware FMA (see this module's doc
/// comment). A budget of 4 ULP is used here: generous enough to absorb that measured
/// noise plus a comfortable margin for a different GPU/driver on someone else's machine
/// exhibiting slightly more of the same effect -- including `jx`/`jy`/`hero_rand`'s
/// extra division and add/floor/subtract, each its own single rounding-noise
/// opportunity -- while nowhere near loose enough to hide a real porting bug -- an
/// actual algebra error (a wrong constant, a dropped term, a sign flip, `%` used where
/// `rem_euclid` was needed on a negative operand) produces errors of a completely
/// different character: many ULP at minimum, more typically whole-unit-scale or larger,
/// i.e. many THOUSANDS of ULP, not single digits. This is an integer ULP bound, not a
/// relative epsilon, because ULP is the natural unit for "how many representable `f32`
/// values apart are these two results" -- a relative tolerance would either be too
/// loose near zero or too tight near the top of each field's range for the same
/// underlying bit-level distance.
///
/// This 1-ULP-scale, unavoidable-at-the-driver-level divergence is also the concrete
/// reason a later Tier 3 (statistical image comparison against a full CPU render) must
/// be variance-scaled statistical rather than exact: if even this single, simple,
/// already-bit-exact-ported function cannot reproduce the CPU bit-for-bit, a full render
/// accumulating many such operations across many bounces certainly cannot either. Tier 3
/// inherits this budget's lesson, not its number.
pub const FLOAT_ULP_BUDGET: u32 = 4;

/// Absolute-difference floor for `jx`/`jy`/`hero_rand`/`lambdas` comparisons.
///
/// See [`crate::renderer::gpu::ulp::within_tolerance`]'s doc comment for the general
/// rationale (ULP distance is a poor metric exactly where a value legitimately crosses,
/// or nearly crosses, zero). `jx`/`jy` range over `[-0.5, 0.5)` and DO legitimately land
/// very close to zero (measured: a genuine, non-buggy CPU/GPU pair at `jy ~= -6e-6`
/// registered as 262,144 ULP apart purely from proximity to the sign boundary, before
/// this floor was added) -- `lambdas` never approaches zero (`[380.0, 780.0]`) so the
/// floor is inert there, exactly as intended.
pub const FLOAT_ABS_FLOOR: f32 = 1e-4;

/// One float field's ULP distance from the CPU reference, kept only when it's the
/// running argmax -- see [`FloatUlpResult`].
///
/// `field` names which `RngRecord` field this came from (`"jx"`, `"jy"`, `"hero_rand"`,
/// or `"lambdas"`); `channel` is only meaningful for `"lambdas"` (the index into its
/// 8-element array) and is `0` otherwise.
#[derive(Debug, Clone, Copy)]
pub struct FloatUlpArgmax {
    pub pixel: u32,
    pub sample: u32,
    pub field: &'static str,
    pub channel: usize,
    pub cpu: f32,
    pub gpu: f32,
    pub ulp: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct FloatUlpResult {
    pub budget: u32,
    /// Max ULP among comparisons NOT exempted by [`FLOAT_ABS_FLOOR`] -- what
    /// [`Self::passed`] checks against `budget`. See
    /// [`crate::renderer::gpu::ulp::within_tolerance`]'s doc comment for why a bare ULP
    /// distance alone is the wrong metric near zero.
    pub max_ulp: u32,
    /// Max ULP among ALL comparisons, including exempted ones -- purely informational
    /// (matches the "max raw ULP" every other Phase-1 ULP check reports).
    pub max_raw_ulp: u32,
    pub argmax: Option<FloatUlpArgmax>,
    pub over_budget_count: usize,
    pub exempted_count: usize,
    pub total_values_compared: usize,
}

impl FloatUlpResult {
    #[must_use]
    pub const fn passed(&self) -> bool {
        self.over_budget_count == 0
    }
}

/// Streaming accumulator for the Tier 2 float-field ULP check -- fed one `(pixel,
/// sample)` record pair at a time from the same loop [`compare_record`] runs in, so
/// `run` never needs to hold all `total` CPU/GPU record pairs in memory at once just to
/// compare these fields.
#[derive(Debug, Default)]
struct FloatUlpAccumulator {
    max_ulp: u32,
    max_raw_ulp: u32,
    argmax: Option<FloatUlpArgmax>,
    over_budget_count: usize,
    exempted_count: usize,
    total_values_compared: usize,
}

impl FloatUlpAccumulator {
    fn record_one(
        &mut self,
        pixel: u32,
        sample: u32,
        field: &'static str,
        channel: usize,
        c: f32,
        g: f32,
    ) {
        use crate::renderer::gpu::ulp::{ulp_distance, within_tolerance};

        self.total_values_compared += 1;
        let ulp = ulp_distance(c, g);
        self.max_raw_ulp = self.max_raw_ulp.max(ulp);

        if within_tolerance(c, g, FLOAT_ULP_BUDGET, FLOAT_ABS_FLOOR) {
            if ulp > FLOAT_ULP_BUDGET {
                // Within tolerance only via the abs-floor clause, not the ULP budget
                // itself -- a genuine near-zero exemption, not a pass.
                self.exempted_count += 1;
            }
            return;
        }

        self.over_budget_count += 1;
        if ulp > self.max_ulp {
            self.max_ulp = ulp;
            self.argmax = Some(FloatUlpArgmax {
                pixel,
                sample,
                field,
                channel,
                cpu: c,
                gpu: g,
                ulp,
            });
        }
    }

    fn record(&mut self, pixel: u32, sample: u32, cpu: &RngRecord, gpu: &RngRecord) {
        self.record_one(pixel, sample, "jx", 0, cpu.jx, gpu.jx);
        self.record_one(pixel, sample, "jy", 0, cpu.jy, gpu.jy);
        self.record_one(pixel, sample, "hero_rand", 0, cpu.hero_rand, gpu.hero_rand);
        for (channel, (&c, &g)) in cpu.lambdas.iter().zip(gpu.lambdas.iter()).enumerate() {
            self.record_one(pixel, sample, "lambdas", channel, c, g);
        }
    }

    const fn finish(self) -> FloatUlpResult {
        FloatUlpResult {
            budget: FLOAT_ULP_BUDGET,
            max_ulp: self.max_ulp,
            max_raw_ulp: self.max_raw_ulp,
            argmax: self.argmax,
            over_budget_count: self.over_budget_count,
            exempted_count: self.exempted_count,
            total_values_compared: self.total_values_compared,
        }
    }
}

/// Runs the RNG bit-exactness self-test against a live GPU.
///
/// Exercises `num_pixels * num_samples` `(pixel, sample)` pairs, each carrying
/// [`NUM_BOUNCES`] bounces -- i.e. `num_pixels * num_samples * NUM_BOUNCES` total
/// `(pixel, sample, bounce)` tuples.
///
/// # Panics
///
/// Panics on `wgpu` API misuse (see [`crate::renderer::gpu::layout_check::run`]'s doc
/// comment for the same rationale).
#[must_use]
pub fn run(
    ctx: &crate::renderer::gpu::GpuContext,
    num_pixels: u32,
    num_samples: u32,
) -> RngCheckResult {
    let total = (num_pixels as usize) * (num_samples as usize);

    let params = Params {
        num_samples,
        num_bounces: NUM_BOUNCES,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = compute::upload(
        &ctx.device,
        "rng_equivalence params",
        std::slice::from_ref(&params),
        wgpu::BufferUsages::UNIFORM,
    );
    let out_buf = compute::zeroed_buffer::<RngRecord>(
        &ctx.device,
        "rng_equivalence output",
        total,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let pipeline =
        compute::create_compute_pipeline(&ctx.device, "rng_equivalence", SHADER_SRC, "main");
    let bind_group = compute::bind_buffers(
        &ctx.device,
        "rng_equivalence bind group",
        &pipeline,
        &[(0, &params_buf), (1, &out_buf)],
    );

    let workgroups = (total as u32).div_ceil(64);
    compute::dispatch_and_wait(
        &ctx.device,
        &ctx.queue,
        &pipeline,
        &bind_group,
        (workgroups, 1, 1),
    );

    let gpu_records: Vec<RngRecord> = compute::readback(&ctx.device, &ctx.queue, &out_buf, total);

    // Both tiers walk every record in one pass. Tier 1's diagnostic list is capped at 64
    // entries (a mismatch means something is badly broken, and 64 examples is already
    // plenty to diagnose it -- no need to keep formatting more), but that cap must NOT
    // cut the Tier 2 float-field scan short: capping the loop itself would understate
    // `max_ulp`/`over_budget_count` the moment Tier 1 ever failed, which is exactly the
    // scenario where an accurate Tier 2 reading matters most (a real bug can corrupt
    // both tiers at once).
    let mut mismatches = Vec::new();
    let mut float_acc = FloatUlpAccumulator::default();
    for (idx, gpu_record) in gpu_records.iter().enumerate() {
        let pixel = (idx as u32) / num_samples;
        let sample = (idx as u32) % num_samples;
        let cpu = cpu_record(pixel, sample);
        if mismatches.len() < 64 {
            compare_record(pixel, sample, &cpu, gpu_record, &mut mismatches);
        }
        float_acc.record(pixel, sample, &cpu, gpu_record);
    }

    RngCheckResult {
        total_records: total,
        mismatches,
        float_ulp: float_acc.finish(),
    }
}

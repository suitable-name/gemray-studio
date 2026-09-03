// Phase 1: environment-sampling / CMF-integration / white-balance kernels -- driven by
// `renderer::gpu::environment_check`.
//
// Fresh translations of, in order: `color::cie1931::cie_1931_cmf` (the single source of
// truth for the CIE 1931 CMF fit -- note the DELIBERATELY different sigma below/above
// each lobe peak; do not "simplify" to one sigma, see that module's own doc comment),
// `optics::raytracer::blackbody_spectrum`, `optics::raytracer::sample_studio_environment`
// (plus the `optics::studio_rig::StudioRig` key/fill/ring directions it depends on),
// and `optics::raytracer::compute_illuminant_white_balance` (the 401-point 380..=780nm
// von Kries integration).
//
// Every output is a flat `array<f32>` (never `array<vec3<f32>>`/`array<vec2<f32>>`)
// specifically to sidestep WGSL's storage-array element-stride rounding for vec2/vec3
// (`roundUp(align, size)`, which does NOT match a tightly-packed Rust `[f32; N]`) --
// see `renderer::buffers`' module doc comment for the bug class that kind of mismatch
// causes. Multi-component results are written at `idx * N + component`.

const PI: f32 = 3.14159265358979323846;
const RING_LIGHT_COUNT: u32 = 16u;

// Rust's `f32::powi(n)` lowers to exponentiation-by-squaring (LLVM's `llvm.powi`
// intrinsic), not the general `exp(log(x) * n)` path WGSL's `pow()` builtin normally
// takes for a non-integer exponent. Every `powi` call site in the ported CPU code
// (`blackbody_spectrum`'s `.powi(5)`, `sample_studio_environment`'s `.powi(28)`
// /`.powi(18)`/`.powi(6)`) uses this instead of `pow()`, to keep the GPU port on the
// same, tighter-rounding algorithm rather than introducing an avoidable extra source of
// ULP divergence.
fn powi_u(base: f32, exp: u32) -> f32 {
    var result: f32 = 1.0;
    var b: f32 = base;
    var e: u32 = exp;
    loop {
        if (e == 0u) {
            break;
        }
        if ((e & 1u) == 1u) {
            result = result * b;
        }
        b = b * b;
        e = e >> 1u;
    }
    return result;
}

// ---------------------------------------------------------------------------------
// cie_1931_cmf
// ---------------------------------------------------------------------------------

fn cmf_lobe(x: f32, mu: f32, sigma_lo: f32, sigma_hi: f32) -> f32 {
    var sigma: f32;
    if (x < mu) {
        sigma = sigma_lo;
    } else {
        sigma = sigma_hi;
    }
    let t = (x - mu) / sigma;
    return exp(-0.5 * t * t);
}

fn cie_1931_cmf(l: f32) -> vec3<f32> {
    let x_mid = fma(1.056, cmf_lobe(l, 599.8, 37.9, 31.0), 0.362 * cmf_lobe(l, 442.0, 16.0, 26.7));
    let x = fma(0.065, -cmf_lobe(l, 501.1, 20.4, 26.2), x_mid);
    let y = fma(0.286, cmf_lobe(l, 530.9, 16.3, 31.1), 0.821 * cmf_lobe(l, 568.8, 46.9, 40.5));
    let z = fma(0.681, cmf_lobe(l, 459.0, 26.0, 13.8), 1.217 * cmf_lobe(l, 437.0, 11.8, 36.0));
    return vec3<f32>(max(x, 0.0), max(y, 0.0), max(z, 0.0));
}

@group(0) @binding(0) var<storage, read> cmf_lambdas: array<f32>;
@group(0) @binding(1) var<storage, read_write> cmf_out: array<f32>;

@compute @workgroup_size(64)
fn cmf_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&cmf_lambdas)) {
        return;
    }
    let xyz = cie_1931_cmf(cmf_lambdas[idx]);
    cmf_out[idx * 3u + 0u] = xyz.x;
    cmf_out[idx * 3u + 1u] = xyz.y;
    cmf_out[idx * 3u + 2u] = xyz.z;
}

// ---------------------------------------------------------------------------------
// blackbody_spectrum
// ---------------------------------------------------------------------------------

fn blackbody_spectrum(lambda_nm: f32, temp_k: f32) -> f32 {
    let t_k = max(temp_k, 1000.0);
    let h_c_k: f32 = 14388000.0;
    let exp_val = exp(min(h_c_k / (lambda_nm * t_k), 80.0));
    let exp_560 = exp(min(h_c_k / (560.0 * t_k), 80.0));
    let denom = max(exp_val - 1.0, 1e-6);
    let denom_560 = max(exp_560 - 1.0, 1e-6);
    let ratio = denom_560 / denom;
    return clamp(powi_u(560.0 / lambda_nm, 5u) * ratio, 0.01, 20.0);
}

struct BlackbodyCase {
    lambda_nm: f32,
    temp_k: f32,
}

@group(0) @binding(2) var<storage, read> blackbody_cases: array<BlackbodyCase>;
@group(0) @binding(3) var<storage, read_write> blackbody_out: array<f32>;

@compute @workgroup_size(64)
fn blackbody_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&blackbody_cases)) {
        return;
    }
    let c = blackbody_cases[idx];
    blackbody_out[idx] = blackbody_spectrum(c.lambda_nm, c.temp_k);
}

// ---------------------------------------------------------------------------------
// sample_studio_environment (+ StudioRig)
// ---------------------------------------------------------------------------------

struct StudioEnvCase {
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    lambda_nm: f32,
    temp_k: f32,
    spot_mult: f32,
    exposure: f32,
    light_yaw: f32,
    light_pitch: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

fn studio_rig_key_dir(light_yaw: f32, light_pitch: f32) -> vec3<f32> {
    let cos_lp = cos(light_pitch);
    let sin_lp = sin(light_pitch);
    let cos_ly = cos(light_yaw);
    let sin_ly = sin(light_yaw);
    return normalize(vec3<f32>(cos_lp * sin_ly, sin_lp, cos_lp * cos_ly));
}

fn studio_rig_fill_dir(light_yaw: f32, light_pitch: f32) -> vec3<f32> {
    let fill_yaw = fma(PI, 0.78, light_yaw);
    let fill_pitch = clamp(light_pitch * 0.65, 0.15, 1.2);
    return normalize(vec3<f32>(cos(fill_pitch) * sin(fill_yaw), sin(fill_pitch), cos(fill_pitch) * cos(fill_yaw)));
}

fn studio_rig_ring_dir(i: u32, light_yaw: f32, sin_lp: f32) -> vec3<f32> {
    let angle = fma(f32(i), PI * 2.0 / f32(RING_LIGHT_COUNT), light_yaw);
    return normalize(vec3<f32>(cos(angle) * 0.75, sin_lp * 0.8, sin(angle) * 0.75));
}

fn sample_studio_environment(
    dir_in: vec3<f32>,
    lambda_nm: f32,
    temp_k: f32,
    spot_mult: f32,
    exposure: f32,
    light_yaw: f32,
    light_pitch: f32,
) -> f32 {
    let d = normalize(dir_in);
    let spec_power = blackbody_spectrum(lambda_nm, temp_k);

    let bg_val = max(fma(0.012, fma(d.y, 0.5, 0.5), 0.015), 0.005) * exposure;
    var radiance = bg_val * spec_power;

    let key_dir = studio_rig_key_dir(light_yaw, light_pitch);
    let key_dot = max(dot(d, key_dir), 0.0);
    if (key_dot > 0.0) {
        let softbox = powi_u(key_dot, 28u) * 12.0 * spot_mult * exposure;
        radiance = fma(softbox, spec_power, radiance);
    }

    let fill_dir = studio_rig_fill_dir(light_yaw, light_pitch);
    let fill_dot = max(dot(d, fill_dir), 0.0);
    if (fill_dot > 0.0) {
        let fill = powi_u(fill_dot, 18u) * 4.5 * exposure;
        radiance = fma(fill, spec_power, radiance);
    }

    let sin_lp = sin(light_pitch);
    for (var i: u32 = 0u; i < RING_LIGHT_COUNT; i = i + 1u) {
        let ring_dir = studio_rig_ring_dir(i, light_yaw, sin_lp);
        let ring_dot = max(dot(d, ring_dir), 0.0);
        if (ring_dot > 0.96) {
            let spark = (ring_dot - 0.96) / 0.04;
            let intensity = powi_u(spark, 6u) * 22.0 * spot_mult * exposure;
            radiance = fma(intensity, spec_power, radiance);
        }
    }

    return radiance;
}

@group(0) @binding(4) var<storage, read> studio_cases: array<StudioEnvCase>;
@group(0) @binding(5) var<storage, read_write> studio_out: array<f32>;

@compute @workgroup_size(64)
fn studio_env_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&studio_cases)) {
        return;
    }
    let c = studio_cases[idx];
    studio_out[idx] = sample_studio_environment(
        vec3<f32>(c.dir_x, c.dir_y, c.dir_z),
        c.lambda_nm,
        c.temp_k,
        c.spot_mult,
        c.exposure,
        c.light_yaw,
        c.light_pitch,
    );
}

// ---------------------------------------------------------------------------------
// compute_illuminant_white_balance: 401-point (380..=780nm, 1nm step) quadrature.
//
// Fix 3: diagonalised in Bradford LMS space, not raw XYZ -- see
// `optics::raytracer::compute_illuminant_white_balance`'s doc comment for the full
// rationale. `BRADFORD_XYZ_TO_LMS`/`BRADFORD_LMS_TO_XYZ` and `D65_WHITE_X`/
// `D65_WHITE_Y` below are the exact same constants as that Rust function's, so this
// kernel's output stays within `environment_check::WHITE_BALANCE_ULP_BUDGET` of it.
// ---------------------------------------------------------------------------------

const D65_WHITE_X: f32 = 0.3127;
const D65_WHITE_Y: f32 = 0.3290;

const BRADFORD_XYZ_TO_LMS = mat3x3<f32>(
    vec3<f32>(0.8951, -0.7502, 0.0389),
    vec3<f32>(0.2664, 1.7135, -0.0685),
    vec3<f32>(-0.1614, 0.0367, 1.0296),
);

const BRADFORD_LMS_TO_XYZ = mat3x3<f32>(
    vec3<f32>(0.986993, 0.432305, -0.008529),
    vec3<f32>(-0.147054, 0.518360, 0.040043),
    vec3<f32>(0.159963, 0.049291, 0.968487),
);

struct WhiteBalanceCase {
    temp_k: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(6) var<storage, read> wb_cases: array<WhiteBalanceCase>;
@group(0) @binding(7) var<storage, read_write> wb_out: array<f32>;

@compute @workgroup_size(64)
fn white_balance_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&wb_cases)) {
        return;
    }
    let temp_k = wb_cases[idx].temp_k;

    var xyz_w = vec3<f32>(0.0, 0.0, 0.0);
    for (var step: i32 = 0; step <= (780 - 380); step = step + 1) {
        let lambda = 380.0 + f32(step);
        xyz_w = xyz_w + cie_1931_cmf(lambda) * blackbody_spectrum(lambda, temp_k);
    }

    let target_y = max(xyz_w.y, 1e-6);
    let xyz_target = vec3<f32>(
        (D65_WHITE_X / D65_WHITE_Y) * target_y,
        target_y,
        ((1.0 - D65_WHITE_X - D65_WHITE_Y) / D65_WHITE_Y) * target_y,
    );

    let lms_source = BRADFORD_XYZ_TO_LMS * xyz_w;
    let lms_target = BRADFORD_XYZ_TO_LMS * xyz_target;

    var scale: vec3<f32>;
    if (lms_source.x > 1e-6) {
        scale.x = lms_target.x / lms_source.x;
    } else {
        scale.x = 1.0;
    }
    if (lms_source.y > 1e-6) {
        scale.y = lms_target.y / lms_source.y;
    } else {
        scale.y = 1.0;
    }
    if (lms_source.z > 1e-6) {
        scale.z = lms_target.z / lms_source.z;
    } else {
        scale.z = 1.0;
    }

    wb_out[idx * 3u + 0u] = scale.x;
    wb_out[idx * 3u + 1u] = scale.y;
    wb_out[idx * 3u + 2u] = scale.z;
}

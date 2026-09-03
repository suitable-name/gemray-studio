// Phase 1: furnace-anchor kernel -- driven by `renderer::gpu::furnace_check`.
//
// Glues together every Phase-1 ported function into one end-to-end pipeline (camera ray
// generation with RNG-derived jitter, the hero-wavelength comb, CIE 1931 CMF
// integration) with a deliberately UNIFORM constant-radiance "environment" (independent
// of both direction and wavelength) and, implicitly, zero gemstone facets -- so the
// resulting XYZ is analytically computable from the CMF integral alone, checking this
// pipeline against TRUE values rather than merely against the CPU (a shared porting
// mistake could otherwise self-certify a CPU-vs-GPU-only comparison).
//
// # Why this kernel never calls `intersect_polyhedron`
//
// `optics::raytracer::intersect_polyhedron(ray, &[])` (zero facet planes) does NOT
// return `None` -- its `t_near`/`t_far` sentinels (`-1e30`/`+1e30`) fall through to the
// "origin is inside the solid" EXIT branch (vacuously true: with zero half-space
// constraints, every point satisfies all of them), producing
// `Some(HitRecord { t: 1e30, normal: Vec3::ZERO, facet_idx: 0 })`. See
// `renderer::gpu::furnace_check`'s own doc comment and its
// `empty_planes_intersect_returns_the_sentinel_hit_not_a_miss` test, which pins this
// exact CPU behavior down. Functionally this IS "the ray reaches the environment
// unobstructed" (there is no real geometry at `t = 1e30` to interact with), which is
// the property this kernel's "uniform environment, unconditionally sampled" design
// relies on -- it does not need to branch on the `Option` wrapper at all, because with
// zero planes by construction there is nothing that could ever block a ray.

struct GpuCameraParams {
    origin: vec3<f32>,
    fov_tan: f32,
    forward: vec3<f32>,
    width: f32,
    right: vec3<f32>,
    height: f32,
    up: vec3<f32>,
    num_samples: u32,
}

struct FurnaceExtra {
    l0: f32,
    num_pixels: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> camera: GpuCameraParams;
@group(0) @binding(1) var<uniform> extra: FurnaceExtra;
@group(0) @binding(2) var<storage, read_write> out_persample: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_pixel_sum: array<f32>;

const SPECTRUM_MIN: f32 = 380.0;
const SPECTRUM_SPAN: f32 = 400.0;
const NUM_CHANNELS: u32 = 8u;
// (400.0 / 8) / 106.856 -- must match `optics::raytracer::integrate_channels_to_xyz`'s
// `norm_factor` exactly. Computed via the SAME division WGSL will perform at
// const-evaluation time, deliberately not a hand-rounded decimal literal: an earlier
// version of this constant was hand-computed as 0.4680704632214494 (a transcription
// error -- the correct value is ~0.4679194), which produced a ~0.03% systematic bias
// caught by this module's own per-tuple ULP check (see `furnace_check`'s negative
// control / this bug's own discovery for how that showed up in practice).
const NORM_FACTOR: f32 = (400.0 / 8.0) / 106.856;

fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x * 0x85ebca6bu;
    x = x ^ (x >> 13u);
    x = x * 0xc2b2ae35u;
    x = x ^ (x >> 16u);
    return x;
}

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

/// One (pixel, sample) tuple's furnace XYZ estimate: camera ray generation (with
/// RNG-derived jitter, bit-exact per Phase 0) + the hero-wavelength comb + CMF
/// integration against the uniform constant-radiance environment. Shared by both entry
/// points below.
fn furnace_sample_xyz(pixel: u32, sample: u32) -> vec3<f32> {
    let x = f32(pixel % u32(camera.width));
    let y = f32(pixel / u32(camera.width));

    let seed = hash_u32((pixel * 0x9e3779b9u) ^ (sample * 0x85ebca6bu));
    let jx = f32(hash_u32(seed) % 10000u) / 10000.0 - 0.5;
    let jy = f32(hash_u32(seed + 0x7feb352du) % 10000u) / 10000.0 - 0.5;

    // Camera::generate_ray -- exercised for pipeline fidelity even though this
    // furnace's environment is direction-independent (see the file header): `dir` is
    // computed exactly as the real ray-generation path would, it just happens not to
    // affect this particular environment's radiance.
    let aspect = camera.width / camera.height;
    let u = ((x + jx) / camera.width - 0.5) * 2.0 * aspect * camera.fov_tan;
    let v = (0.5 - (y + jy) / camera.height) * 2.0 * camera.fov_tan;
    let dir = normalize(camera.forward + camera.right * u + camera.up * v);
    // `dir` deliberately unused beyond this point -- see above.
    _ = dir;

    let hero_hash = hash_u32(seed);
    let hero_rand = f32(hero_hash) / 4294967295.0;
    let channel_width = SPECTRUM_SPAN / f32(NUM_CHANNELS);
    let lambda_hero = fma(hero_rand, SPECTRUM_SPAN, SPECTRUM_MIN);

    var xyz = vec3<f32>(0.0, 0.0, 0.0);
    for (var k: u32 = 0u; k < NUM_CHANNELS; k = k + 1u) {
        let offset = fma(f32(k), channel_width, lambda_hero - SPECTRUM_MIN);
        let wrapped = offset % SPECTRUM_SPAN;
        let lambda = SPECTRUM_MIN + wrapped;
        // Uniform environment: `extra.l0` regardless of `lambda`/`dir`. `mis_weight` is
        // exactly 1.0 here (every channel shares the same, uniform path_pdf), matching
        // `integrate_channels_to_xyz`'s `spectral_mis_weight` degenerating to 1.0 when
        // every channel's technique agrees.
        xyz = xyz + cie_1931_cmf(lambda) * (extra.l0 * NORM_FACTOR);
    }
    return xyz;
}

@compute @workgroup_size(64)
fn furnace_samples_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = extra.num_pixels * camera.num_samples;
    if (idx >= total) {
        return;
    }
    let pixel = idx / camera.num_samples;
    let sample = idx % camera.num_samples;
    let xyz = furnace_sample_xyz(pixel, sample);
    out_persample[idx * 3u + 0u] = xyz.x;
    out_persample[idx * 3u + 1u] = xyz.y;
    out_persample[idx * 3u + 2u] = xyz.z;
}

// Strictly sequential, this-thread-only accumulation across `camera.num_samples` --
// exactly `self_determinism.wgsl`'s proven-safe pattern (see that file's own doc
// comment): no atomics, no cross-thread reduction, so the result is bit-for-bit
// reproducible run to run regardless of GPU scheduling.
@compute @workgroup_size(64)
fn furnace_accumulate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.x;
    if (pixel >= extra.num_pixels) {
        return;
    }
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var s: u32 = 0u; s < camera.num_samples; s = s + 1u) {
        sum = sum + furnace_sample_xyz(pixel, s);
    }
    out_pixel_sum[pixel * 3u + 0u] = sum.x;
    out_pixel_sum[pixel * 3u + 1u] = sum.y;
    out_pixel_sum[pixel * 3u + 2u] = sum.z;
}

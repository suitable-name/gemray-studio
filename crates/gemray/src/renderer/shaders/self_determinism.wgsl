// GPU self-determinism self-test kernel (Phase 0 deliverable 3, Tier 0) -- driven by
// `renderer::gpu::determinism_check`, NOT a physics kernel.
//
// Demonstrates the accumulation pattern any future real GPU raytracer kernel MUST use:
// each thread owns exactly one output slot (`out_sums[pixel]`) and accumulates into it
// with a strictly sequential, in-thread loop -- no `atomicAdd`, no cross-thread
// reduction, no dependency on warp/subgroup scheduling order. Float addition is not
// associative, so an accumulation whose term order depends on GPU scheduling (as
// `atomicAdd`-based accumulation's does) is not guaranteed to reproduce the same bit
// pattern from run to run, even on identical input on identical hardware. An
// accumulation where each thread alone determines its own term order is: the same
// thread executes the same loop over the same data every time, so its final float sum
// is bit-for-bit reproducible. `renderer::gpu::determinism_check` proves this by running
// this exact kernel twice against the same input and comparing every byte of the output
// buffer.
//
// The per-sample term itself reuses the RNG this crate already needs to be bit-exact
// (see `rng_equivalence.wgsl`) purely so this kernel does something nontrivial and
// float-shaped, not because self-determinism depends on the RNG in any way -- it would
// hold for any strictly-sequential per-thread accumulation.

struct Params {
    num_pixels: u32,
    num_samples: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_sums: array<f32>;

fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x * 0x85ebca6bu;
    x = x ^ (x >> 13u);
    x = x * 0xc2b2ae35u;
    x = x ^ (x >> 16u);
    return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.x;
    if (pixel >= params.num_pixels) {
        return;
    }

    var sum: f32 = 0.0;
    for (var s: u32 = 0u; s < params.num_samples; s = s + 1u) {
        let seed = hash_u32((pixel * 0x9e3779b9u) ^ (s * 0x85ebca6bu));
        let v = f32(hash_u32(seed)) / 4294967295.0;
        // Strictly sequential, this-thread-only accumulation -- see the file header.
        sum = sum + v;
    }
    out_sums[pixel] = sum;
}

// Phase 2 GPU struct-layout self-test kernel -- driven by
// `renderer::gpu::layout_check::run_transport_params`, NOT a physics kernel.
//
// Same purpose and mechanism as `layout_echo.wgsl`/`phase1_layout_echo.wgsl` (see
// `layout_echo.wgsl`'s own doc comment for the bug class this is built to catch): echo
// every named field of the new Phase-2 `GpuTransportParams` uniform struct straight
// through to an independent output buffer, so `layout_check` can compare the two
// buffers' raw bytes (padding included) and prove the hand-written `#[repr(C)]` offsets
// in `renderer::buffers` actually agree with what WGSL computes.

struct GpuTransportParams {
    num_pixels: u32,
    max_bounces: u32,
    sample_offset: u32,
    env_mode: u32,
    l0: f32,
    studio_temp_k: f32,
    studio_spot_mult: f32,
    studio_exposure: f32,
    studio_light_yaw: f32,
    studio_light_pitch: f32,
    // R4: previously omitted (this struct relied on `vec3<f32>`'s own 16-byte alignment
    // rounding 40 up to 48 regardless, so leaving these two out never changed
    // `white_balance`'s offset) -- now echoed explicitly like every other field, closing
    // that gap: `renderer::buffers::GpuTransportParams::new`'s default changed
    // `write_debug_buffers` from an always-zero pad float to a nonzero flag, so this
    // struct's own byte-echo comparison needs to actually copy it (see
    // `layout_check::run_transport_params`'s doc comment).
    pixel_offset: u32,
    write_debug_buffers: u32,
    white_balance: vec3<f32>,
}

@group(0) @binding(0) var<storage, read> in_params: GpuTransportParams;
@group(0) @binding(1) var<storage, read_write> out_params: GpuTransportParams;

@compute @workgroup_size(1)
fn echo_transport_params() {
    out_params.num_pixels = in_params.num_pixels;
    out_params.max_bounces = in_params.max_bounces;
    out_params.sample_offset = in_params.sample_offset;
    out_params.env_mode = in_params.env_mode;
    out_params.l0 = in_params.l0;
    out_params.studio_temp_k = in_params.studio_temp_k;
    out_params.studio_spot_mult = in_params.studio_spot_mult;
    out_params.studio_exposure = in_params.studio_exposure;
    out_params.studio_light_yaw = in_params.studio_light_yaw;
    out_params.studio_light_pitch = in_params.studio_light_pitch;
    out_params.pixel_offset = in_params.pixel_offset;
    out_params.write_debug_buffers = in_params.write_debug_buffers;
    out_params.white_balance = in_params.white_balance;
}

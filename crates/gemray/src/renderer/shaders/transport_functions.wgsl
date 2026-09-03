// Phase 2, Tier 2: standalone per-function ULP checks for the small pieces
// `shaders/spectral_transport.wgsl`'s megakernel calls -- driven by
// `renderer::gpu::transport_check`. Every kernel below calls the SAME shared function
// the megakernel calls: both files consume `shaders/transport_physics.wgsl`,
// concatenated ahead of each of them by `build.rs` (see that file's header comment for
// the mechanism and why it exists). There is no manually-kept-in-sync copy here any
// more -- a dense-grid sweep mismatch found by a kernel below now localizes to exactly
// one named function in the ONE place it's defined, and -- because that place is also
// what the megakernel calls -- it is necessarily testing the shipped code path, not a
// duplicate of it.
//
// Every case bank is dispatched against the REAL CPU function it was translated from
// (`optics::polarization::MuellerMatrix::*`, `optics::raytracer::{tir_phase_delta,
// signed_frame_rotation_psi}`, `optics::dispersion::DispersionModel::evaluate`,
// `optics::raytracer::spectral_absorption`, `optics::birefringence::pleochroic_channel_alpha`)
// -- never a hand-written parallel reimplementation of the physics -- by
// `renderer::gpu::transport_check`.

// ---------------------------------------------------------------------------------
// MuellerMatrix::frame_rotation + StokesVector::apply_matrix
// ---------------------------------------------------------------------------------

struct FrameRotationCase {
    psi: f32,
    si: f32,
    sq: f32,
    su: f32,
    sv: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> frame_rotation_cases: array<FrameRotationCase>;
@group(0) @binding(1) var<storage, read_write> frame_rotation_out: array<f32>;

@compute @workgroup_size(64)
fn frame_rotation_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&frame_rotation_cases)) {
        return;
    }
    let c = frame_rotation_cases[idx];
    let m = mueller_frame_rotation(c.psi);
    let out = m * vec4<f32>(c.si, c.sq, c.su, c.sv);
    frame_rotation_out[idx * 4u + 0u] = out.x;
    frame_rotation_out[idx * 4u + 1u] = out.y;
    frame_rotation_out[idx * 4u + 2u] = out.z;
    frame_rotation_out[idx * 4u + 3u] = out.w;
}

// ---------------------------------------------------------------------------------
// MuellerMatrix::fresnel_reflection + StokesVector::apply_matrix
// ---------------------------------------------------------------------------------

struct FresnelReflectionCase {
    r_s: f32,
    r_p: f32,
    si: f32,
    sq: f32,
    su: f32,
    sv: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(2) var<storage, read> fresnel_reflection_cases: array<FresnelReflectionCase>;
@group(0) @binding(3) var<storage, read_write> fresnel_reflection_out: array<f32>;

@compute @workgroup_size(64)
fn fresnel_reflection_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&fresnel_reflection_cases)) {
        return;
    }
    let c = fresnel_reflection_cases[idx];
    let m = mueller_fresnel_reflection(c.r_s, c.r_p);
    let out = m * vec4<f32>(c.si, c.sq, c.su, c.sv);
    fresnel_reflection_out[idx * 4u + 0u] = out.x;
    fresnel_reflection_out[idx * 4u + 1u] = out.y;
    fresnel_reflection_out[idx * 4u + 2u] = out.z;
    fresnel_reflection_out[idx * 4u + 3u] = out.w;
}

// ---------------------------------------------------------------------------------
// MuellerMatrix::fresnel_transmission + StokesVector::apply_matrix
// ---------------------------------------------------------------------------------

struct FresnelTransmissionCase {
    n1: f32,
    n2: f32,
    cos_i: f32,
    cos_t: f32,
    t_s: f32,
    t_p: f32,
    si: f32,
    sq: f32,
    su: f32,
    sv: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(4) var<storage, read> fresnel_transmission_cases: array<FresnelTransmissionCase>;
@group(0) @binding(5) var<storage, read_write> fresnel_transmission_out: array<f32>;

@compute @workgroup_size(64)
fn fresnel_transmission_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&fresnel_transmission_cases)) {
        return;
    }
    let c = fresnel_transmission_cases[idx];
    let m = mueller_fresnel_transmission(c.n1, c.n2, c.cos_i, c.cos_t, c.t_s, c.t_p);
    let out = m * vec4<f32>(c.si, c.sq, c.su, c.sv);
    fresnel_transmission_out[idx * 4u + 0u] = out.x;
    fresnel_transmission_out[idx * 4u + 1u] = out.y;
    fresnel_transmission_out[idx * 4u + 2u] = out.z;
    fresnel_transmission_out[idx * 4u + 3u] = out.w;
}

// ---------------------------------------------------------------------------------
// MuellerMatrix::tir_retardation + StokesVector::apply_matrix
// ---------------------------------------------------------------------------------

struct TirRetardationCase {
    delta: f32,
    si: f32,
    sq: f32,
    su: f32,
    sv: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(6) var<storage, read> tir_retardation_cases: array<TirRetardationCase>;
@group(0) @binding(7) var<storage, read_write> tir_retardation_out: array<f32>;

@compute @workgroup_size(64)
fn tir_retardation_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&tir_retardation_cases)) {
        return;
    }
    let c = tir_retardation_cases[idx];
    let m = mueller_tir_retardation(c.delta);
    let out = m * vec4<f32>(c.si, c.sq, c.su, c.sv);
    tir_retardation_out[idx * 4u + 0u] = out.x;
    tir_retardation_out[idx * 4u + 1u] = out.y;
    tir_retardation_out[idx * 4u + 2u] = out.z;
    tir_retardation_out[idx * 4u + 3u] = out.w;
}

// ---------------------------------------------------------------------------------
// optics::raytracer::signed_frame_rotation_psi
// ---------------------------------------------------------------------------------

struct SignedPsiCase {
    prev: vec3<f32>,
    _pad0: f32,
    curr: vec3<f32>,
    _pad1: f32,
    axis: vec3<f32>,
    _pad2: f32,
}

@group(0) @binding(8) var<storage, read> signed_psi_cases: array<SignedPsiCase>;
@group(0) @binding(9) var<storage, read_write> signed_psi_out: array<f32>;

@compute @workgroup_size(64)
fn signed_psi_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&signed_psi_cases)) {
        return;
    }
    let c = signed_psi_cases[idx];
    signed_psi_out[idx] = signed_frame_rotation_psi(c.prev, c.curr, c.axis);
}

// ---------------------------------------------------------------------------------
// optics::raytracer::tir_phase_delta
// ---------------------------------------------------------------------------------

struct TirPhaseDeltaCase {
    n1k: f32,
    cos_i: f32,
    sin_i: f32,
    _pad0: f32,
}

@group(0) @binding(10) var<storage, read> tir_phase_delta_cases: array<TirPhaseDeltaCase>;
@group(0) @binding(11) var<storage, read_write> tir_phase_delta_out: array<f32>;

@compute @workgroup_size(64)
fn tir_phase_delta_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&tir_phase_delta_cases)) {
        return;
    }
    let c = tir_phase_delta_cases[idx];
    tir_phase_delta_out[idx] = tir_phase_delta(c.n1k, c.cos_i, c.sin_i);
}

// ---------------------------------------------------------------------------------
// optics::dispersion::DispersionModel::evaluate
// ---------------------------------------------------------------------------------

struct DispersionCase {
    model_type: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    param_a: vec4<f32>,
    param_b: vec4<f32>,
    lambda_nm: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

@group(0) @binding(12) var<storage, read> dispersion_cases: array<DispersionCase>;
@group(0) @binding(13) var<storage, read_write> dispersion_out: array<f32>;

@compute @workgroup_size(64)
fn dispersion_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&dispersion_cases)) {
        return;
    }
    let c = dispersion_cases[idx];
    dispersion_out[idx] = dispersion_evaluate(c.model_type, c.param_a, c.param_b, c.lambda_nm);
}

// ---------------------------------------------------------------------------------
// optics::raytracer::spectral_absorption
// ---------------------------------------------------------------------------------

struct AbsorptionCase {
    band_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    bands: array<AbsorptionBand, 8>,
    lambda_nm: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

@group(0) @binding(14) var<storage, read> absorption_cases: array<AbsorptionCase>;
@group(0) @binding(15) var<storage, read_write> absorption_out: array<f32>;

@compute @workgroup_size(64)
fn absorption_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&absorption_cases)) {
        return;
    }
    let c = absorption_cases[idx];
    absorption_out[idx] = spectral_absorption(c.bands, c.band_count, c.lambda_nm);
}

// ---------------------------------------------------------------------------------
// optics::birefringence::pleochroic_channel_alpha (end to end: electric_field_direction,
// ordinary/extraordinary eigen polarization inputs supplied directly, AbsorptionTensor3
// quadratic form, effective_pleochroic_alpha combination).
// ---------------------------------------------------------------------------------

struct PleochroicCase {
    alpha_o: f32,
    alpha_e: f32,
    _pad0: f32,
    _pad1: f32,
    c_axis: vec3<f32>,
    _pad2: f32,
    s_axis: vec3<f32>,
    _pad3: f32,
    propagation_dir: vec3<f32>,
    _pad4: f32,
    eigen_a: vec3<f32>,
    _pad5: f32,
    eigen_b: vec3<f32>,
    _pad6: f32,
    stokes: vec4<f32>,
}

@group(0) @binding(16) var<storage, read> pleochroic_cases: array<PleochroicCase>;
@group(0) @binding(17) var<storage, read_write> pleochroic_out: array<f32>;

@compute @workgroup_size(64)
fn pleochroic_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&pleochroic_cases)) {
        return;
    }
    let cs = pleochroic_cases[idx];
    pleochroic_out[idx] = pleochroic_channel_alpha(
        cs.alpha_o, cs.alpha_e, cs.c_axis, cs.s_axis, cs.propagation_dir, cs.eigen_a, cs.eigen_b, cs.stokes,
    );
}

// ---------------------------------------------------------------------------------
// optics::birefringence::{BirefringenceParams::ordinary_eigen_polarization,
// BirefringenceParams::extraordinary_eigen_polarization}
// ---------------------------------------------------------------------------------

struct EigenPolarizationCase {
    wave_normal: vec3<f32>,
    _pad0: f32,
    c_axis: vec3<f32>,
    _pad1: f32,
}

@group(0) @binding(18) var<storage, read> eigen_polarization_cases: array<EigenPolarizationCase>;
@group(0) @binding(19) var<storage, read_write> eigen_polarization_out: array<f32>;

@compute @workgroup_size(64)
fn eigen_polarization_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&eigen_polarization_cases)) {
        return;
    }
    let c = eigen_polarization_cases[idx];
    let o_hat = ordinary_eigen_polarization(c.wave_normal, c.c_axis);
    let e_hat = extraordinary_eigen_polarization(c.wave_normal, c.c_axis);
    eigen_polarization_out[idx * 6u + 0u] = o_hat.x;
    eigen_polarization_out[idx * 6u + 1u] = o_hat.y;
    eigen_polarization_out[idx * 6u + 2u] = o_hat.z;
    eigen_polarization_out[idx * 6u + 3u] = e_hat.x;
    eigen_polarization_out[idx * 6u + 4u] = e_hat.y;
    eigen_polarization_out[idx * 6u + 5u] = e_hat.z;
}

// ---------------------------------------------------------------------------------
// Phase 3: optics::raytracer::theta_c_for_bounce (the theta_c fixed-point iteration --
// see `transport_physics.wgsl`'s Phase 3 section for why `is_biaxial` is omitted).
// ---------------------------------------------------------------------------------

struct ThetaCCase {
    normal: vec3<f32>,
    _pad0: f32,
    ray_dir: vec3<f32>,
    _pad1: f32,
    c_axis: vec3<f32>,
    _pad2: f32,
    cos_i: f32,
    inside_gem: u32,
    is_anisotropic: u32,
    n_o_hero_seed: f32,
    birefringence_delta: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

@group(0) @binding(20) var<storage, read> theta_c_cases: array<ThetaCCase>;
@group(0) @binding(21) var<storage, read_write> theta_c_out: array<f32>;

@compute @workgroup_size(64)
fn theta_c_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&theta_c_cases)) {
        return;
    }
    let c = theta_c_cases[idx];
    theta_c_out[idx] = theta_c_for_bounce(
        c.normal, c.ray_dir, c.cos_i, c.inside_gem != 0u, c.is_anisotropic != 0u, c.c_axis,
        c.n_o_hero_seed, c.birefringence_delta,
    );
}

// ---------------------------------------------------------------------------------
// Phase 3: optics::birefringence::BirefringenceParams::extraordinary_poynting_dir (the
// extraordinary ray's walk-off direction).
// ---------------------------------------------------------------------------------

struct WalkOffCase {
    wave_normal: vec3<f32>,
    _pad0: f32,
    c_axis: vec3<f32>,
    _pad1: f32,
    n_o: f32,
    n_e: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(22) var<storage, read> walk_off_cases: array<WalkOffCase>;
@group(0) @binding(23) var<storage, read_write> walk_off_out: array<f32>;

@compute @workgroup_size(64)
fn walk_off_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&walk_off_cases)) {
        return;
    }
    let c = walk_off_cases[idx];
    let dir = extraordinary_poynting_dir(c.wave_normal, c.c_axis, c.n_o, c.n_e);
    walk_off_out[idx * 3u + 0u] = dir.x;
    walk_off_out[idx * 3u + 1u] = dir.y;
    walk_off_out[idx * 3u + 2u] = dir.z;
}

// ---------------------------------------------------------------------------------
// Phase 3: optics::raytracer::per_channel_uniaxial_indices (one channel's per-mode
// (n_o, n_eff) index pair, via `per_channel_uniaxial_index` -- see
// `transport_physics.wgsl`'s Phase 3 section for why the CPU's NUM_CHANNELS loop is the
// caller's responsibility here).
// ---------------------------------------------------------------------------------

struct PerModeIndexCase {
    model_type: u32,
    is_anisotropic: u32,
    _pad0: u32,
    _pad1: u32,
    param_a: vec4<f32>,
    param_b: vec4<f32>,
    lambda_nm: f32,
    birefringence_delta: f32,
    theta_c: f32,
    _pad2: f32,
}

@group(0) @binding(24) var<storage, read> per_mode_index_cases: array<PerModeIndexCase>;
@group(0) @binding(25) var<storage, read_write> per_mode_index_out: array<f32>;

@compute @workgroup_size(64)
fn per_mode_index_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&per_mode_index_cases)) {
        return;
    }
    let c = per_mode_index_cases[idx];
    let pair = per_channel_uniaxial_index(
        c.model_type, c.param_a, c.param_b, c.lambda_nm, c.birefringence_delta, c.is_anisotropic != 0u, c.theta_c,
    );
    per_mode_index_out[idx * 2u + 0u] = pair.x;
    per_mode_index_out[idx * 2u + 1u] = pair.y;
}

// ---------------------------------------------------------------------------------
// Task 2 GPU port: optics::raytracer::cosine_weighted_hemisphere -- the frosted-bounce
// direction sampler (Malley's method).
// ---------------------------------------------------------------------------------

// Field order matters here: `n` (a vec3, needing 16-byte WGSL alignment) is placed
// FIRST so it lands at offset 0 with no implicit leading padding, then `u1`/`u2` pack
// into its trailing 4 bytes plus one more (the same "vec3 + scalar(s)" pattern this
// crate's struct-layout doc comments describe elsewhere) -- `renderer::gpu::
// transport_check::CosineHemisphereCase` mirrors this EXACT field order for that
// reason; reordering either side without the other would silently misalign every case.
struct CosineHemisphereCase {
    n: vec3<f32>,
    u1: f32,
    u2: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(26) var<storage, read> cosine_hemisphere_cases: array<CosineHemisphereCase>;
@group(0) @binding(27) var<storage, read_write> cosine_hemisphere_out: array<f32>;

@compute @workgroup_size(64)
fn cosine_hemisphere_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&cosine_hemisphere_cases)) {
        return;
    }
    let c = cosine_hemisphere_cases[idx];
    let dir = cosine_weighted_hemisphere(c.u1, c.u2, c.n);
    cosine_hemisphere_out[idx * 3u + 0u] = dir.x;
    cosine_hemisphere_out[idx * 3u + 1u] = dir.y;
    cosine_hemisphere_out[idx * 3u + 2u] = dir.z;
}

// ---------------------------------------------------------------------------------
// Task 2 GPU port: optics::raytracer::apply_frosted_bounce -- the full frosted-facet
// bounce dispatch (TIR-forced / reflect / transmit branch selection, the broadband
// hero-only r_unpol split, Stokes depolarization, path_pdf scaling). Calls the SAME
// `transport_physics.wgsl` function `spectral_transport.wgsl`'s megakernel calls for a
// `FacetFinish::Frosted` facet -- see that shared function's own doc comment.
// ---------------------------------------------------------------------------------

struct FrostedBounceCase {
    is_anisotropic: u32,
    sin2_t: f32,
    n1: f32,
    n2: f32,
    cos_i: f32,
    inside_gem: u32,
    is_extraordinary: u32,
    rng_seed: u32,
    bounce: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    normal: vec3<f32>,
    _pad3: f32,
    stokes_in: array<vec4<f32>, 8>,
    path_pdf_in: array<f32, 8>,
}

@group(0) @binding(28) var<storage, read> frosted_bounce_cases: array<FrostedBounceCase>;
// Layout per case, 46 floats: [0..3) new_dir, [3] new_inside_gem (0.0/1.0),
// [4] has_extraordinary_update (0.0/1.0), [5] extraordinary_update (0.0/1.0),
// [6..38) stokes_out (8 vec4s), [38..46) path_pdf_out.
@group(0) @binding(29) var<storage, read_write> frosted_bounce_out: array<f32>;

@compute @workgroup_size(64)
fn frosted_bounce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&frosted_bounce_cases)) {
        return;
    }
    let c = frosted_bounce_cases[idx];
    var stokes: array<vec4<f32>, 8> = c.stokes_in;
    var path_pdf: array<f32, 8> = c.path_pdf_in;
    let result = apply_frosted_bounce(
        c.is_anisotropic != 0u, c.sin2_t, c.n1, c.n2, c.cos_i, c.normal,
        c.inside_gem != 0u, c.is_extraordinary != 0u, c.rng_seed, c.bounce,
        &stokes, &path_pdf,
    );
    let base = idx * 46u;
    frosted_bounce_out[base + 0u] = result.new_dir.x;
    frosted_bounce_out[base + 1u] = result.new_dir.y;
    frosted_bounce_out[base + 2u] = result.new_dir.z;
    frosted_bounce_out[base + 3u] = f32(result.new_inside_gem);
    frosted_bounce_out[base + 4u] = f32(result.has_extraordinary_update);
    frosted_bounce_out[base + 5u] = f32(result.extraordinary_update);
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        frosted_bounce_out[base + 6u + k * 4u + 0u] = stokes[k].x;
        frosted_bounce_out[base + 6u + k * 4u + 1u] = stokes[k].y;
        frosted_bounce_out[base + 6u + k * 4u + 2u] = stokes[k].z;
        frosted_bounce_out[base + 6u + k * 4u + 3u] = stokes[k].w;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        frosted_bounce_out[base + 38u + k] = path_pdf[k];
    }
}

// ---------------------------------------------------------------------------------
// Physics review, Task 1 GPU port: optics::raytracer::{henyey_greenstein_phase,
// sample_henyey_greenstein_direction, maybe_scatter_or_extinguish}. Calls the SAME
// `transport_physics.wgsl` functions `spectral_transport.wgsl`'s megakernel calls for a
// scattering-active material -- see that shared file's own doc comment.
// ---------------------------------------------------------------------------------

struct HgPhaseCase {
    cos_theta: f32,
    g: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(30) var<storage, read> hg_phase_cases: array<HgPhaseCase>;
@group(0) @binding(31) var<storage, read_write> hg_phase_out: array<f32>;

@compute @workgroup_size(64)
fn hg_phase_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&hg_phase_cases)) {
        return;
    }
    let c = hg_phase_cases[idx];
    hg_phase_out[idx] = henyey_greenstein_phase(c.cos_theta, c.g);
}

struct HgSampleCase {
    u1: f32,
    u2: f32,
    g: f32,
    _pad0: f32,
    forward: vec3<f32>,
    _pad1: f32,
}

@group(0) @binding(32) var<storage, read> hg_sample_cases: array<HgSampleCase>;
@group(0) @binding(33) var<storage, read_write> hg_sample_out: array<f32>;

@compute @workgroup_size(64)
fn hg_sample_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&hg_sample_cases)) {
        return;
    }
    let c = hg_sample_cases[idx];
    let dir = sample_henyey_greenstein_direction(c.u1, c.u2, c.g, c.forward);
    hg_sample_out[idx * 3u + 0u] = dir.x;
    hg_sample_out[idx * 3u + 1u] = dir.y;
    hg_sample_out[idx * 3u + 2u] = dir.z;
}

// ---------------------------------------------------------------------------------
// Phase 4 GPU port: optics::birefringence::BiaxialIndicatrix -- standalone per-function
// checks for the genuinely biaxial machinery, mirroring the uniaxial Phase 3 checks
// above. Every case carries the indicatrix's three principal indices plus its
// `gamma_axis` (not the derived `axes` frame directly -- `biaxial_axes_from_gamma` is
// itself part of what's being checked, exactly as the CPU side's
// `BiaxialIndicatrix::from_gamma_axis` derives `axes` from `gamma_axis` fresh).
// ---------------------------------------------------------------------------------

struct BiaxialWaveIndicesCase {
    n_alpha: f32,
    n_beta: f32,
    n_gamma: f32,
    _pad0: f32,
    gamma_axis: vec3<f32>,
    _pad1: f32,
    wave_normal: vec3<f32>,
    _pad2: f32,
}

@group(0) @binding(36) var<storage, read> biaxial_wave_indices_cases: array<BiaxialWaveIndicesCase>;
@group(0) @binding(37) var<storage, read_write> biaxial_wave_indices_out: array<f32>;

@compute @workgroup_size(64)
fn biaxial_wave_indices_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&biaxial_wave_indices_cases)) {
        return;
    }
    let c = biaxial_wave_indices_cases[idx];
    let ax = biaxial_axes_from_gamma(c.gamma_axis);
    let ni = biaxial_wave_indices(c.n_alpha, c.n_beta, c.n_gamma, ax.ax0, ax.ax1, ax.ax2, c.wave_normal);
    biaxial_wave_indices_out[idx * 2u + 0u] = ni.x;
    biaxial_wave_indices_out[idx * 2u + 1u] = ni.y;
}

struct BiaxialEigenPolarizationCase {
    n_alpha: f32,
    n_beta: f32,
    n_gamma: f32,
    _pad0: f32,
    gamma_axis: vec3<f32>,
    _pad1: f32,
    wave_normal: vec3<f32>,
    _pad2: f32,
}

@group(0) @binding(38) var<storage, read> biaxial_eigen_polarization_cases: array<BiaxialEigenPolarizationCase>;
@group(0) @binding(39) var<storage, read_write> biaxial_eigen_polarization_out: array<f32>;

@compute @workgroup_size(64)
fn biaxial_eigen_polarization_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&biaxial_eigen_polarization_cases)) {
        return;
    }
    let c = biaxial_eigen_polarization_cases[idx];
    let ax = biaxial_axes_from_gamma(c.gamma_axis);
    let eig = biaxial_eigen_polarizations(c.n_alpha, c.n_beta, c.n_gamma, ax.ax0, ax.ax1, ax.ax2, c.wave_normal);
    biaxial_eigen_polarization_out[idx * 6u + 0u] = eig.d_slow.x;
    biaxial_eigen_polarization_out[idx * 6u + 1u] = eig.d_slow.y;
    biaxial_eigen_polarization_out[idx * 6u + 2u] = eig.d_slow.z;
    biaxial_eigen_polarization_out[idx * 6u + 3u] = eig.d_fast.x;
    biaxial_eigen_polarization_out[idx * 6u + 4u] = eig.d_fast.y;
    biaxial_eigen_polarization_out[idx * 6u + 5u] = eig.d_fast.z;
}

struct BiaxialModePoyntingCase {
    n_alpha: f32,
    n_beta: f32,
    n_gamma: f32,
    _pad0: f32,
    gamma_axis: vec3<f32>,
    _pad1: f32,
    wave_normal: vec3<f32>,
    want_slow: u32,
}

@group(0) @binding(40) var<storage, read> biaxial_mode_poynting_cases: array<BiaxialModePoyntingCase>;
@group(0) @binding(41) var<storage, read_write> biaxial_mode_poynting_out: array<f32>;

@compute @workgroup_size(64)
fn biaxial_mode_poynting_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&biaxial_mode_poynting_cases)) {
        return;
    }
    let c = biaxial_mode_poynting_cases[idx];
    let ax = biaxial_axes_from_gamma(c.gamma_axis);
    let dir = biaxial_mode_poynting_dir(c.n_alpha, c.n_beta, c.n_gamma, ax.ax0, ax.ax1, ax.ax2, c.wave_normal, c.want_slow != 0u);
    biaxial_mode_poynting_out[idx * 3u + 0u] = dir.x;
    biaxial_mode_poynting_out[idx * 3u + 1u] = dir.y;
    biaxial_mode_poynting_out[idx * 3u + 2u] = dir.z;
}

struct BiaxialResolveEntryModeCase {
    n_alpha: f32,
    n_beta: f32,
    n_gamma: f32,
    _pad0: f32,
    gamma_axis: vec3<f32>,
    _pad1: f32,
    incident_dir: vec3<f32>,
    _pad2: f32,
    normal: vec3<f32>,
    _pad3: f32,
    cos_i: f32,
    n_seed: f32,
    want_slow: u32,
    _pad4: f32,
}

@group(0) @binding(42) var<storage, read> biaxial_resolve_entry_mode_cases: array<BiaxialResolveEntryModeCase>;
@group(0) @binding(43) var<storage, read_write> biaxial_resolve_entry_mode_out: array<f32>;

@compute @workgroup_size(64)
fn biaxial_resolve_entry_mode_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&biaxial_resolve_entry_mode_cases)) {
        return;
    }
    let c = biaxial_resolve_entry_mode_cases[idx];
    let ax = biaxial_axes_from_gamma(c.gamma_axis);
    let result = biaxial_resolve_entry_mode(
        c.n_alpha, c.n_beta, c.n_gamma, ax.ax0, ax.ax1, ax.ax2,
        c.incident_dir, c.normal, c.cos_i, c.n_seed, c.want_slow != 0u,
    );
    biaxial_resolve_entry_mode_out[idx * 4u + 0u] = result.n;
    biaxial_resolve_entry_mode_out[idx * 4u + 1u] = result.wave_dir.x;
    biaxial_resolve_entry_mode_out[idx * 4u + 2u] = result.wave_dir.y;
    biaxial_resolve_entry_mode_out[idx * 4u + 3u] = result.wave_dir.z;
}

// optics::birefringence::pleochroic_channel_alpha with `alpha_beta = Some(alpha_beta)`
// -- the genuinely biaxial (trichroic) three-coefficient absorption path.
struct BiaxialPleochroicCase {
    alpha_o: f32,
    alpha_beta: f32,
    alpha_e: f32,
    _pad0: f32,
    c_axis: vec3<f32>,
    _pad1: f32,
    s_axis: vec3<f32>,
    _pad2: f32,
    propagation_dir: vec3<f32>,
    _pad3: f32,
    eigen_a: vec3<f32>,
    _pad4: f32,
    eigen_b: vec3<f32>,
    _pad5: f32,
    stokes: vec4<f32>,
}

@group(0) @binding(44) var<storage, read> biaxial_pleochroic_cases: array<BiaxialPleochroicCase>;
@group(0) @binding(45) var<storage, read_write> biaxial_pleochroic_out: array<f32>;

@compute @workgroup_size(64)
fn biaxial_pleochroic_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&biaxial_pleochroic_cases)) {
        return;
    }
    let cs = biaxial_pleochroic_cases[idx];
    biaxial_pleochroic_out[idx] = pleochroic_channel_alpha_biaxial(
        cs.alpha_o, cs.alpha_beta, cs.alpha_e, cs.c_axis, cs.s_axis, cs.propagation_dir, cs.eigen_a, cs.eigen_b, cs.stokes,
    );
}

struct ScatterOrExtinguishCase {
    sigma_s: f32,
    g: f32,
    hit_t: f32,
    rng_seed: u32,
    bounce: u32,
    // P1 (absorption path scale): reuses what was `_pad0` -- see the Rust-side
    // `ScatterOrExtinguishCase`'s own doc comment.
    path_scale: f32,
    _pad1: u32,
    _pad2: u32,
    ray_dir: vec3<f32>,
    _pad3: f32,
    alphas: array<f32, 8>,
    stokes_in: array<vec4<f32>, 8>,
    path_pdf_in: array<f32, 8>,
}

@group(0) @binding(34) var<storage, read> scatter_cases: array<ScatterOrExtinguishCase>;
// Layout per case, 45 floats: [0] scattered (0.0/1.0), [1] t_free, [2..5) new_dir,
// [5..37) stokes_out (8 vec4s), [37..45) path_pdf_out.
@group(0) @binding(35) var<storage, read_write> scatter_out: array<f32>;

@compute @workgroup_size(64)
fn scatter_or_extinguish_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&scatter_cases)) {
        return;
    }
    let c = scatter_cases[idx];
    var stokes: array<vec4<f32>, 8> = c.stokes_in;
    var path_pdf: array<f32, 8> = c.path_pdf_in;
    let result = maybe_scatter_or_extinguish(
        c.alphas, c.sigma_s, c.g, c.ray_dir, c.hit_t, c.path_scale, c.rng_seed, c.bounce, &stokes, &path_pdf,
    );
    let base = idx * 45u;
    scatter_out[base + 0u] = f32(result.scattered);
    scatter_out[base + 1u] = result.t_free;
    scatter_out[base + 2u] = result.new_dir.x;
    scatter_out[base + 3u] = result.new_dir.y;
    scatter_out[base + 4u] = result.new_dir.z;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        scatter_out[base + 5u + k * 4u + 0u] = stokes[k].x;
        scatter_out[base + 5u + k * 4u + 1u] = stokes[k].y;
        scatter_out[base + 5u + k * 4u + 2u] = stokes[k].z;
        scatter_out[base + 5u + k * 4u + 3u] = stokes[k].w;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        scatter_out[base + 37u + k] = path_pdf[k];
    }
}

//! `spectral_absorption` and `pleochroic_channel_alpha` (uniaxial: `alpha_beta = None`).

use crate::{
    optics::{
        birefringence::{BirefringenceParams, pleochroic_channel_alpha},
        materials::GemMaterial,
        polarization::StokesVector,
        raytracer::spectral_absorption,
    },
    renderer::gpu::compute,
};
use glam::Vec3;

use super::{SHADER_SRC, STOKES_SAMPLES, UlpAccumulator, UlpCheckResult};

// ---------------------------------------------------------------------------------
// spectral_absorption
// ---------------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AbsorptionBandGpu {
    center_nm: f32,
    width_nm: f32,
    peak: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AbsorptionCase {
    band_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    bands: [AbsorptionBandGpu; 8],
    lambda_nm: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

const ABSORPTION_ULP_BUDGET: u32 = 24;
const ABSORPTION_ABS_FLOOR: f32 = 1e-6;

fn build_absorption_cases() -> Vec<(
    AbsorptionCase,
    Vec<crate::optics::absorption::AbsorptionBand>,
)> {
    let materials = GemMaterial::all_materials();
    let mut band_sets: Vec<Vec<crate::optics::absorption::AbsorptionBand>> = Vec::new();
    for name in ["Ruby", "Sapphire", "Emerald", "Spinel"] {
        if let Some(m) = materials.iter().find(|m| m.name == name) {
            band_sets.push(m.absorption.o_ray.clone());
            band_sets.push(m.absorption.e_ray.clone());
        }
    }
    let mut lambdas: Vec<f32> = Vec::new();
    let steps = 80;
    for i in 0..=steps {
        lambdas.push((i as f32 / steps as f32).mul_add(400.0, 380.0));
    }

    let mut out = Vec::new();
    for bands in band_sets {
        let mut gpu_bands = [AbsorptionBandGpu {
            center_nm: 0.0,
            width_nm: 1.0,
            peak: 0.0,
        }; 8];
        for (slot, band) in gpu_bands.iter_mut().zip(bands.iter()) {
            *slot = AbsorptionBandGpu {
                center_nm: band.center_nm,
                width_nm: band.width_nm,
                peak: band.peak,
            };
        }
        // Adversarial: exactly each band's centre wavelength (peak evaluation) as well
        // as the dense sweep.
        let mut lambdas_with_centers = lambdas.clone();
        for band in &bands {
            lambdas_with_centers.push(band.center_nm);
        }
        for &lambda_nm in &lambdas_with_centers {
            out.push((
                AbsorptionCase {
                    band_count: bands.len() as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    bands: gpu_bands,
                    lambda_nm,
                    _pad3: 0.0,
                    _pad4: 0.0,
                    _pad5: 0.0,
                },
                bands.clone(),
            ));
        }
    }
    out
}

#[must_use]
pub fn run_absorption(ctx: &crate::renderer::gpu::GpuContext) -> UlpCheckResult<AbsorptionCase> {
    let with_bands = build_absorption_cases();
    let cases: Vec<AbsorptionCase> = with_bands.iter().map(|(c, _)| *c).collect();
    let total = cases.len();
    let in_buf = compute::upload(
        &ctx.device,
        "absorption in",
        &cases,
        wgpu::BufferUsages::STORAGE,
    );
    let out_buf = compute::zeroed_buffer::<f32>(
        &ctx.device,
        "absorption out",
        total,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let pipeline = compute::create_compute_pipeline(
        &ctx.device,
        "absorption_main",
        SHADER_SRC,
        "absorption_main",
    );
    let bind_group = compute::bind_buffers(
        &ctx.device,
        "absorption bind group",
        &pipeline,
        &[(14, &in_buf), (15, &out_buf)],
    );
    let workgroups = (total as u32).div_ceil(64);
    compute::dispatch_and_wait(
        &ctx.device,
        &ctx.queue,
        &pipeline,
        &bind_group,
        (workgroups, 1, 1),
    );
    let gpu_out: Vec<f32> = compute::readback(&ctx.device, &ctx.queue, &out_buf, total);

    let mut acc = UlpAccumulator::new(
        "spectral_absorption",
        ABSORPTION_ULP_BUDGET,
        ABSORPTION_ABS_FLOOR,
    );
    for (idx, (case, bands)) in with_bands.iter().enumerate() {
        let cpu = spectral_absorption(bands, case.lambda_nm);
        acc.record(case, "alpha", cpu, gpu_out[idx]);
    }
    acc.finish()
}

// ---------------------------------------------------------------------------------
// pleochroic_channel_alpha
// ---------------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PleochroicCase {
    alpha_o: f32,
    alpha_e: f32,
    _pad0: f32,
    _pad1: f32,
    c_axis: [f32; 3],
    _pad2: f32,
    s_axis: [f32; 3],
    _pad3: f32,
    propagation_dir: [f32; 3],
    _pad4: f32,
    eigen_a: [f32; 3],
    _pad5: f32,
    eigen_b: [f32; 3],
    _pad6: f32,
    stokes: [f32; 4],
}

const PLEOCHROIC_ULP_BUDGET: u32 = 48;
const PLEOCHROIC_ABS_FLOOR: f32 = 1e-5;

fn build_pleochroic_cases() -> Vec<PleochroicCase> {
    let mut cases = Vec::new();
    let dirs = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.4, 0.6, 0.693).normalize(),
        Vec3::new(-0.2, 0.5, -0.843).normalize(),
    ];
    let alphas = [(0.0f32, 0.0f32), (1.0, 1.0), (0.5, 2.0), (3.0, 0.2)];
    for &c_axis in &dirs {
        for &s_axis in &dirs {
            for &prop in &dirs {
                for &(alpha_o, alpha_e) in &alphas {
                    let eigen_a = BirefringenceParams::ordinary_eigen_polarization(prop, c_axis);
                    let eigen_b =
                        BirefringenceParams::extraordinary_eigen_polarization(prop, c_axis);
                    for s in [STOKES_SAMPLES[0], STOKES_SAMPLES[4]] {
                        cases.push(PleochroicCase {
                            alpha_o,
                            alpha_e,
                            _pad0: 0.0,
                            _pad1: 0.0,
                            c_axis: c_axis.to_array(),
                            _pad2: 0.0,
                            s_axis: s_axis.to_array(),
                            _pad3: 0.0,
                            propagation_dir: prop.to_array(),
                            _pad4: 0.0,
                            eigen_a: eigen_a.to_array(),
                            _pad5: 0.0,
                            eigen_b: eigen_b.to_array(),
                            _pad6: 0.0,
                            stokes: s,
                        });
                    }
                }
            }
        }
    }
    cases
}

#[must_use]
pub fn run_pleochroic(ctx: &crate::renderer::gpu::GpuContext) -> UlpCheckResult<PleochroicCase> {
    let cases = build_pleochroic_cases();
    let total = cases.len();
    let in_buf = compute::upload(
        &ctx.device,
        "pleochroic in",
        &cases,
        wgpu::BufferUsages::STORAGE,
    );
    let out_buf = compute::zeroed_buffer::<f32>(
        &ctx.device,
        "pleochroic out",
        total,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let pipeline = compute::create_compute_pipeline(
        &ctx.device,
        "pleochroic_main",
        SHADER_SRC,
        "pleochroic_main",
    );
    let bind_group = compute::bind_buffers(
        &ctx.device,
        "pleochroic bind group",
        &pipeline,
        &[(16, &in_buf), (17, &out_buf)],
    );
    let workgroups = (total as u32).div_ceil(64);
    compute::dispatch_and_wait(
        &ctx.device,
        &ctx.queue,
        &pipeline,
        &bind_group,
        (workgroups, 1, 1),
    );
    let gpu_out: Vec<f32> = compute::readback(&ctx.device, &ctx.queue, &out_buf, total);

    let mut acc = UlpAccumulator::new(
        "pleochroic_channel_alpha",
        PLEOCHROIC_ULP_BUDGET,
        PLEOCHROIC_ABS_FLOOR,
    );
    for (idx, case) in cases.iter().enumerate() {
        let s = StokesVector::new(
            case.stokes[0],
            case.stokes[1],
            case.stokes[2],
            case.stokes[3],
        );
        let cpu = pleochroic_channel_alpha(
            case.alpha_o,
            case.alpha_e,
            None, // GPU never sees a biaxial material -- see spectral_transport.wgsl's doc comment
            Vec3::from_array(case.c_axis),
            Vec3::from_array(case.s_axis),
            Vec3::from_array(case.propagation_dir),
            Vec3::from_array(case.eigen_a),
            Vec3::from_array(case.eigen_b),
            &s,
        );
        acc.record(case, "alpha", cpu, gpu_out[idx]);
    }
    acc.finish()
}

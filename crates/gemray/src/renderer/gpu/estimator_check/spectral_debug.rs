//! Spectral-space debug comparison: "before XYZ integration... so a CMF bug cannot
//! masquerade as a transport bug". See [`run_spectral_debug`]'s own doc comment for the
//! honest scope limit on what this can and cannot verify given the CPU
//! visibility-only constraint.

use crate::{
    optics::raytracer::{
        LightingPreset, apply_von_kries_white_balance, compute_illuminant_white_balance,
        illuminant_temperature_k, integrate_channels_to_xyz,
    },
    renderer::buffers::{GpuGemMaterial, GpuTransportParams, transport_env_mode},
};
use glam::Vec3;

use super::{
    all_polished_finishes, camera_params_for, dispatch_transport, round_brilliant_planes,
    test_camera, tier3_material,
};

#[derive(Debug, Clone)]
pub struct SpectralDebugResult {
    pub total_cases: usize,
    /// Max ULP distance between the GPU's own `out_xyz` and re-integrating the GPU's
    /// own per-channel `(radiance, lambdas, path_pdf)` debug output through the REAL
    /// CPU `optics::raytracer::integrate_channels_to_xyz` -- see this function's own
    /// doc comment for exactly what this does and does not prove.
    pub max_self_consistency_ulp: u32,
    pub over_budget_count: usize,
}

const SPECTRAL_DEBUG_ULP_BUDGET: u32 = 64;
const SPECTRAL_DEBUG_ABS_FLOOR: f32 = 1e-5;

/// Spectral-space debug comparison: "before XYZ integration... so a CMF bug cannot
/// masquerade as a transport bug".
///
/// # What this actually checks, and the honest gap
///
/// `optics::raytracer::trace_spectral_ray` has no CPU debug hook that returns its
/// internal per-channel `radiance`/`lambdas`/`path_pdf` arrays -- only the final
/// integrated `Vec3`. Adding one would mean either changing `trace_spectral_ray`'s own
/// signature (out of scope: this phase's CPU changes are limited to `pub(crate)`
/// visibility, never new behavior or new surface area) or hand-duplicating the ENTIRE
/// bounce loop a second time in this test file to capture its intermediates -- exactly
/// the "parallel reimplementation of the highest-risk code path" this harness otherwise
/// avoids everywhere else. Neither is acceptable, so there is no independent CPU
/// per-channel reference to compare the GPU's per-channel output against here.
///
/// What this DOES check, using zero reimplementation: the GPU already writes its own
/// pre-integration `(radiance, lambdas, path_pdf)` to debug buffers (`shaders/
/// spectral_transport.wgsl`'s `out_radiance`/`out_lambdas`/`out_path_pdf`). Feeding
/// those straight into the REAL CPU `integrate_channels_to_xyz` (`pub(crate)` since
/// Phase 1's furnace anchor) and comparing the result against the GPU's OWN `out_xyz`
/// isolates the CMF-fit-and-MIS-weight integration step from everything upstream of it:
/// if they agree closely, the integration step is correct and any observed CPU/GPU
/// divergence found elsewhere (furnace anchor, Tier 3) must live in the transport
/// physics upstream of this point, not the integration; if they disagree, the
/// integration step itself has a porting bug. It does NOT independently verify the
/// per-channel radiance values themselves against a CPU transport reference -- that
/// remains covered only indirectly, via the furnace anchor and Tier 3's final-XYZ
/// comparison against the real `trace_spectral_ray`.
#[must_use]
pub fn run_spectral_debug(ctx: &crate::renderer::gpu::GpuContext) -> SpectralDebugResult {
    let camera = test_camera();
    let (width, height, samples) = (16u32, 16u32, 8u32);
    let planes = round_brilliant_planes();
    let material = tier3_material();
    let gpu_material = GpuGemMaterial::encode(&material);
    let temp_k = illuminant_temperature_k(LightingPreset::Daylight);
    let wb = compute_illuminant_white_balance(temp_k);
    let camera_params = camera_params_for(&camera, width, height, samples);
    let params = GpuTransportParams::new(
        width * height,
        10,
        0,
        transport_env_mode::STUDIO_RIG,
        0.0,
        temp_k,
        1.0,
        1.0,
        0.0,
        0.0,
        wb.to_array(),
    );
    let total = (width * height * samples) as usize;
    let finishes = all_polished_finishes(planes.len());
    let dispatch = dispatch_transport(
        ctx,
        &camera_params,
        &params,
        &gpu_material,
        &planes,
        &finishes,
        total,
    );

    let mut max_ulp = 0u32;
    let mut over_budget = 0usize;
    for idx in 0..total {
        let mut radiance = [0.0f32; 8];
        let mut lambdas = [0.0f32; 8];
        let mut path_pdf = [0.0f32; 8];
        radiance.copy_from_slice(&dispatch.radiance[idx * 8..idx * 8 + 8]);
        lambdas.copy_from_slice(&dispatch.lambdas[idx * 8..idx * 8 + 8]);
        path_pdf.copy_from_slice(&dispatch.path_pdf[idx * 8..idx * 8 + 8]);

        // `dispatch.xyz` already has the WGSL kernel's own `params.white_balance`
        // applied via `apply_von_kries_white_balance` (env_mode == STUDIO_RIG here,
        // Bradford LMS space, not a raw XYZ multiply) -- apply the SAME
        // transform to the recombined value so this compares like for like, not the
        // pre-white-balance integration result against the post-white-balance kernel
        // output.
        let recombined = apply_von_kries_white_balance(
            integrate_channels_to_xyz(&radiance, &lambdas, &path_pdf, 0),
            wb,
        );
        let gpu_xyz = Vec3::new(
            dispatch.xyz[idx * 3],
            dispatch.xyz[idx * 3 + 1],
            dispatch.xyz[idx * 3 + 2],
        );

        for (cpu, gpu) in [
            (recombined.x, gpu_xyz.x),
            (recombined.y, gpu_xyz.y),
            (recombined.z, gpu_xyz.z),
        ] {
            let ulp = crate::renderer::gpu::ulp::ulp_distance(cpu, gpu);
            if !crate::renderer::gpu::ulp::within_tolerance(
                cpu,
                gpu,
                SPECTRAL_DEBUG_ULP_BUDGET,
                SPECTRAL_DEBUG_ABS_FLOOR,
            ) {
                over_budget += 1;
                max_ulp = max_ulp.max(ulp);
            }
        }
    }

    SpectralDebugResult {
        total_cases: total,
        max_self_consistency_ulp: max_ulp,
        over_budget_count: over_budget,
    }
}

impl SpectralDebugResult {
    #[must_use]
    pub const fn passed(&self) -> bool {
        self.over_budget_count == 0
    }
}

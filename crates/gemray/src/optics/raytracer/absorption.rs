//! Pleochroic Beer-Lambert absorption and Stokes/Mueller frame rotation.
//!
//! [`apply_absorption`]'s deterministic per-bounce attenuation, the shared
//! [`channel_absorption_alphas`] it and [`super::scattering::maybe_scatter_or_extinguish`]
//! both build on, and [`signed_frame_rotation_psi`]'s signed rotation angle recovery.

use super::{
    NUM_CHANNELS,
    refraction::{RayMaterialContext, RayWavelengthCache},
};
use crate::optics::{
    absorption::AbsorptionBand,
    birefringence::{BirefringenceParams, effective_pleochroic_alpha},
    polarization::{MuellerMatrix, StokesVector, electric_field_direction},
};
use glam::Vec3;

/// Evaluates a material's absorption coefficient at `lambda_nm` as the sum of its
/// individual chromophore [`AbsorptionBand`](crate::optics::absorption::AbsorptionBand)s.
///
/// Replaces the previous three-fixed-lobe RGB-triple model (see
/// `AbsorptionTensor`'s doc comment for why) with a direct sum of real Gaussian
/// absorption bands -- the function's SHAPE (wavelength in, coefficient out) is
/// unchanged, so the Beer-Lambert application in `trace_spectral_ray` needed no
/// change, only its first argument's type. An empty band slice sums to `0.0`
/// (colourless material), matching the old `[0.0, 0.0, 0.0]` triples' behaviour.
#[must_use]
pub fn spectral_absorption(bands: &[AbsorptionBand], lambda_nm: f32) -> f32 {
    bands.iter().map(|band| band.evaluate(lambda_nm)).sum()
}

/// The SIGNED azimuth needed to rotate the Stokes reference frame from
/// `prev_normal` (previous bounce's plane-of-incidence normal) to `current_normal`
/// (this bounce's), about propagation `axis`. Plain `acos` alone always returns a
/// value in [0, pi] and so discards which way the frame actually turned; recovering
/// the sign via `atan2(sin_psi, cos_psi)`, with `sin_psi` read off the component of
/// `prev_normal x current_normal` along `axis`, is what makes the resulting rotation
/// match the true rotation between the two frames instead of mixing Q and U with an
/// arbitrary sign on roughly half of all bounces. Pulled out to a named function so it
/// can be unit tested directly; the call site in `trace_spectral_ray` guards
/// near-zero-length normals (an undefined plane of incidence, e.g. at normal
/// incidence) before calling this.
#[inline]
pub(crate) fn signed_frame_rotation_psi(
    prev_normal: Vec3,
    current_normal: Vec3,
    axis: Vec3,
) -> f32 {
    let cos_psi = prev_normal.dot(current_normal).clamp(-1.0, 1.0);
    let sin_psi = prev_normal
        .cross(current_normal)
        .dot(axis.normalize_or_zero());
    sin_psi.atan2(cos_psi)
}

/// Plane of incidence normal. The rotation angle psi must be SIGNED -- `acos`
/// alone always returns a value in `[0, pi]`, which discards the direction of rotation
/// and mixes Q/U with the wrong sign on roughly half of all bounces. Recover the sign
/// via atan2 (see [`signed_frame_rotation_psi`]), using the propagation axis to resolve
/// which way "positive" rotation goes between the previous and current planes of
/// incidence. A no-op on `stokes` when there is no previous well-defined plane, or
/// either plane's cross product degenerates (near-normal incidence) -- exactly the
/// pre-extraction guard. Returns this bounce's plane-of-incidence normal, for the
/// caller to store as `prev_plane_normal` for the NEXT bounce.
///
/// P2: `k_hat` is the WAVE NORMAL `k`, not the Poynting direction `S` -- the plane of
/// incidence (and the Fresnel/TIR physics that plane feeds) is a property of `k`, not
/// `S`. `k_hat == S` trivially outside the crystal and for the uniaxial ordinary
/// eigenmode, so this is bit-identical to the pre-P2 `ray_dir`-fed call in every case
/// this task's bit-identity requirement covers. See `refraction`'s own design note.
pub(super) fn rotate_stokes_to_plane_of_incidence(
    k_hat: Vec3,
    normal: Vec3,
    prev_plane_normal: Option<Vec3>,
    stokes: &mut [StokesVector; NUM_CHANNELS],
) -> Vec3 {
    let current_plane_normal = k_hat.cross(normal).normalize_or_zero();
    if let Some(prev_normal) = prev_plane_normal
        && current_plane_normal.length_squared() > 1e-6
        && prev_normal.length_squared() > 1e-6
    {
        let psi = signed_frame_rotation_psi(prev_normal, current_plane_normal, k_hat);
        let rot_matrix = MuellerMatrix::frame_rotation(psi);
        for s in stokes.iter_mut() {
            *s = s.apply_matrix(&rot_matrix);
        }
    }
    current_plane_normal
}

/// Directional Pleochroic Beer-Lambert absorption via the polarization quadratic form
/// `alpha = e_hat . A . e_hat` -- applied to every channel's Stokes vector
/// for one internal bounce's path length. `e_hat` is derived per-channel from that
/// channel's own Stokes vector; the two eigenmode directions are shared across channels
/// (they depend only on geometry, not wavelength) so are computed once per call. See
/// `birefringence::pleochroic_channel_alpha` for the full combination (quadratic form
/// for the polarized fraction, eigenmode average for the unpolarized fraction, weighted
/// by degree of polarization). For a biaxial material the two eigenmodes
/// come from the biaxial Fresnel index equation's own eigenvectors (`eigen_polarizations`)
/// rather than the uniaxial ordinary/extraordinary approximation -- see
/// `ordinary_eigen_polarization`'s doc comment. A direct extraction of the
/// pre-extraction inline block, called only when `inside_gem` -- see the call site.
///
/// `ctx.material.absorption.beta_ray`, when `Some`, supplies the third
/// principal band set for a genuinely trichroic material. Only actually consulted when
/// `is_biaxial` (this bounce's eigenmodes came from
/// `BiaxialIndicatrix::eigen_polarizations`, not the uniaxial approximation) -- a
/// biaxial material's own two-set entry (Topaz) leaves `beta_ray` at
/// `None` regardless, so it is unaffected, and a (hypothetical) uniaxial material's
/// `beta_ray` would be ignored even if somehow populated, matching
/// `pleochroic_channel_alpha`'s own contract. Read directly off `ctx.material` (rather
/// than threaded through as separate parameters, the way `abs_o`/`abs_e` used to be)
/// specifically to keep this function's argument count within clippy's
/// `too_many_arguments` limit without a new `#[allow]`.
pub(super) fn apply_absorption(
    ctx: &RayMaterialContext,
    cache: &RayWavelengthCache,
    current_plane_normal: Vec3,
    k_hat: Vec3,
    path_len: f32,
    stokes: &mut [StokesVector; NUM_CHANNELS],
) {
    let alphas = channel_absorption_alphas(ctx, cache, current_plane_normal, k_hat, stokes);
    // P1 (absorption path scale): model units -> absorption-length units. See
    // `GemMaterial::absorption_path_scale`'s doc comment. `absorption_path_scale ==
    // 1.0` (every built-in) makes this multiply an exact IEEE 754 no-op, so `args`
    // below is bit-identical to the pre-P1 `-alpha * path_len` for every existing
    // material/scene.
    let scaled_path_len = path_len * ctx.material.absorption_path_scale;
    // Vectorized Beer-Lambert; exp_f32x8 is a few-ULP polynomial exponential, NOT
    // bit-identical to f32::exp -- golden values were re-baselined loudly for this
    // change (see src/simd.rs module docs).
    let mut args = [0f32; NUM_CHANNELS];
    for (a, alpha) in args.iter_mut().zip(&alphas) {
        *a = -alpha * scaled_path_len;
    }
    let trans = crate::simd::exp_f32x8(args);
    for (k, s) in stokes.iter_mut().enumerate() {
        *s = s.scale(trans[k]);
    }
}

/// The per-channel pleochroic absorption coefficient (`alpha_eff` in
/// [`apply_absorption`]'s original derivation) for every spectral channel, evaluated
/// once and shared by two callers: [`apply_absorption`] itself (the unmodified
/// deterministic Beer-Lambert path -- factored out here as a pure "direct extraction,
/// same floating-point operations in the same order" per this file's established
/// precedent, so `apply_absorption`'s own numeric behaviour is completely unchanged) and
/// [`maybe_scatter_or_extinguish`], which needs the SAME per-channel `sigma_a`
/// to build `sigma_t = sigma_a + sigma_s` (see that function's doc comment for hazard 1:
/// "Beer-Lambert becomes extinction, not absorption").
///
/// Splitting the original single-pass loop (compute `alpha_eff` for channel k, then
/// immediately scale `stokes[k]` by it) into two passes (this function, then
/// `apply_absorption`'s own loop) is safe because channel k's `alpha_eff` here only
/// reads `stokes[k]` (never another channel's Stokes vector) and does so BEFORE that
/// channel's own vector is scaled in the original code, so the value read is identical
/// either way.
///
/// `pub(crate)`, not private: `renderer::gpu::transport_check`'s Tier 2 GPU self-test for
/// [`maybe_scatter_or_extinguish`] needs to compute the SAME real per-channel alphas this
/// function produces to feed the standalone WGSL kernel's explicit `alphas` input (that
/// kernel takes alphas as a parameter rather than re-deriving them from band data, the
/// same "one shared body, two different binding shapes" convention
/// `shaders/transport_physics.wgsl`'s `dispersion_evaluate` already establishes) --
/// calling this REAL function, never a reimplementation. Visibility only, no numerical
/// change.
///
/// P2: `k_hat` is the WAVE NORMAL `k`, not the Poynting direction `S` -- the eigen-
/// polarizations (transverse to the wave that's actually propagating) and the electric
/// field direction extracted from each channel's Stokes vector are both properties of
/// `k`. `k_hat == S` trivially outside the crystal and for the uniaxial ordinary
/// eigenmode, so every call site feeding this from those cases is bit-identical to
/// before. See `refraction`'s own design note, rule 6.
pub(crate) fn channel_absorption_alphas(
    ctx: &RayMaterialContext,
    cache: &RayWavelengthCache,
    current_plane_normal: Vec3,
    k_hat: Vec3,
    stokes: &[StokesVector; NUM_CHANNELS],
) -> [f32; NUM_CHANNELS] {
    let c_axis = ctx.c_axis;
    // Bit-identical to a fresh `material.biaxial_indicatrix(ctx.lambdas[ctx.hero_idx])`
    // call -- see `RayWavelengthCache::hero_indicatrix`'s doc comment.
    let indicatrix = cache.hero_indicatrix;
    let (eigen_a, eigen_b) = indicatrix.map_or_else(
        || {
            (
                BirefringenceParams::ordinary_eigen_polarization(k_hat, c_axis),
                BirefringenceParams::extraordinary_eigen_polarization(k_hat, c_axis),
            )
        },
        |ind| ind.eigen_polarizations(k_hat),
    );

    let mut alphas = [0.0f32; NUM_CHANNELS];
    for (k, alpha_slot) in alphas.iter_mut().enumerate() {
        // R3: `pleochroic_channel_alpha` used to rebuild this channel's
        // `AbsorptionTensor3` (a fresh `stable_orthonormal_basis` + `Mat3`) from scratch
        // every bounce, even though it depends only on wavelength-fixed inputs -- see
        // `RayWavelengthCache::tensor_ch`'s doc comment. Calling
        // `effective_pleochroic_alpha` directly against the cached tensor is exactly
        // `pleochroic_channel_alpha`'s own body with that one rebuild replaced by a cache
        // read; `electric_field_direction`'s call and `effective_pleochroic_alpha`'s own
        // arguments are unchanged.
        let e_hat = electric_field_direction(&stokes[k], current_plane_normal, k_hat);
        *alpha_slot = effective_pleochroic_alpha(
            &cache.tensor_ch[k],
            e_hat,
            eigen_a,
            eigen_b,
            stokes[k].degree_of_polarization(),
        );
    }
    alphas
}

/// P1 (absorption path scale): a Ruby-like slab at `absorption_path_scale = 2.0` must
/// attenuate EXACTLY like a slab of twice the model-unit thickness at
/// `absorption_path_scale = 1.0` -- the whole physical point of the field
/// (`GemMaterial::absorption_path_scale`'s doc comment). Proven algebraically, not just
/// numerically: `apply_absorption` multiplies `path_len * ctx.material.absorption_path_scale`
/// before it ever reaches `alpha`, so `(2.0 * d) * 1.0` (scale=1, thickness=2d) and
/// `d * 2.0` (scale=2, thickness=d) are the SAME IEEE 754 value (multiplication by an
/// exactly-representable `2.0` is exact, and IEEE 754 multiplication is commutative), and
/// every other input (`alphas`, `stokes`) is identical between the two calls -- so the two
/// results must come out bit-for-bit identical, not merely numerically close.
#[cfg(test)]
mod absorption_path_scale_tests {
    use super::*;
    use crate::optics::{
        materials::GemMaterial, polarization::StokesVector,
        raytracer::refraction::build_ray_wavelength_cache,
    };

    #[test]
    fn scaled_slab_attenuates_bit_identically_to_doubled_thickness_slab() {
        let ruby_like =
            GemMaterial::new_custom("ruby-like slab probe", 1.76, 0.0, 0.0, [0.5, 1.0, 1.5]);
        assert_eq!(
            ruby_like.crystal_system,
            crate::optics::materials::CrystalSystem::Cubic,
            "test premise: birefringence_delta=0.0 must yield an isotropic material \
             (no birefringence machinery to complicate the comparison)"
        );

        let d = 0.37f32;
        let material_scale1_thickness2d = ruby_like.clone(); // absorption_path_scale == 1.0
        let material_scale2_thicknessd = ruby_like.with_absorption_path_scale(2.0);

        let lambdas: [f32; NUM_CHANNELS] = [420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0, 700.0];
        let ray_dir = Vec3::new(0.0, -1.0, 0.0);
        let current_plane_normal = Vec3::new(1.0, 0.0, 0.0);
        let stokes_in = [StokesVector::unpolarized(1.0); NUM_CHANNELS];

        let ctx_a = RayMaterialContext {
            material: &material_scale1_thickness2d,
            lambdas,
            hero_idx: 0,
            c_axis: Vec3::Y,
            is_anisotropic: false,
            enable_internal_mode_coupling: true,
        };
        let ctx_b = RayMaterialContext {
            material: &material_scale2_thicknessd,
            lambdas,
            hero_idx: 0,
            c_axis: Vec3::Y,
            is_anisotropic: false,
            enable_internal_mode_coupling: true,
        };
        let cache_a = build_ray_wavelength_cache(&ctx_a);
        let cache_b = build_ray_wavelength_cache(&ctx_b);

        let mut stokes_a = stokes_in;
        let mut stokes_b = stokes_in;
        apply_absorption(
            &ctx_a,
            &cache_a,
            current_plane_normal,
            ray_dir,
            2.0 * d, // scale=1.0: full doubled thickness in model units
            &mut stokes_a,
        );
        apply_absorption(
            &ctx_b,
            &cache_b,
            current_plane_normal,
            ray_dir,
            d, // scale=2.0: half the model-unit thickness
            &mut stokes_b,
        );

        for k in 0..NUM_CHANNELS {
            let a = (stokes_a[k].i, stokes_a[k].q, stokes_a[k].u, stokes_a[k].v);
            let b = (stokes_b[k].i, stokes_b[k].q, stokes_b[k].u, stokes_b[k].v);
            assert_eq!(
                a, b,
                "channel {k}: scale=1.0/thickness=2d must attenuate bit-identically to \
                 scale=2.0/thickness=d (got {a:?} vs {b:?})"
            );
        }
    }
}

#[cfg(test)]
mod frame_rotation_sign_tests {
    use super::*;

    /// Two successive bounces that share the SAME plane of incidence (parallel
    /// plane normals) must produce a rotation angle psi ~= 0 -- no Q/U mixing should be
    /// introduced when the frame hasn't actually turned.
    #[test]
    fn same_plane_of_incidence_gives_zero_psi() {
        let prev_normal = Vec3::new(0.0, 0.0, 1.0);
        let current_normal = Vec3::new(0.0, 0.0, 1.0);
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let psi = signed_frame_rotation_psi(prev_normal, current_normal, axis);
        assert!(
            psi.abs() < 1e-5,
            "psi should be ~0 for identical plane-of-incidence normals (got {psi})"
        );
    }

    /// Swapping which plane normal is "previous" and which is "current" must
    /// flip the sign of psi -- this is exactly the defect the old unsigned `acos`-only
    /// computation could not represent (it always returned the same non-negative angle
    /// regardless of rotation direction).
    #[test]
    fn reversing_plane_normal_order_flips_psi_sign() {
        // Two plane-of-incidence normals at a genuine angle to each other, and a
        // propagation axis with a nonzero component along their cross product so the
        // sign is well-defined (not a degenerate coplanar case).
        let normal_a = Vec3::new(1.0, 0.0, 0.0);
        let normal_b = Vec3::new(0.0, 1.0, 0.0).normalize();
        let axis = Vec3::new(0.0, 0.0, 1.0);

        let psi_forward = signed_frame_rotation_psi(normal_a, normal_b, axis);
        let psi_reversed = signed_frame_rotation_psi(normal_b, normal_a, axis);

        assert!(
            psi_forward.abs() > 0.1,
            "psi_forward should be a genuine nonzero rotation (got {psi_forward})"
        );
        assert!(
            (psi_forward + psi_reversed).abs() < 1e-4,
            "reversing the normal order should flip the sign of psi (forward={psi_forward}, reversed={psi_reversed})"
        );
    }

    /// Sanity check that the recovered angle actually matches the geometric angle
    /// between the two normals (90 degrees here), not just that it's nonzero.
    #[test]
    fn psi_magnitude_matches_geometric_angle_between_normals() {
        let normal_a = Vec3::new(1.0, 0.0, 0.0);
        let normal_b = Vec3::new(0.0, 1.0, 0.0);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let psi = signed_frame_rotation_psi(normal_a, normal_b, axis);
        assert!(
            (psi.abs() - std::f32::consts::FRAC_PI_2).abs() < 1e-4,
            "psi magnitude should match the 90 degree angle between the normals (got {})",
            psi.to_degrees()
        );
    }
}

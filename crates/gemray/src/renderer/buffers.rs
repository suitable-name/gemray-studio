//! GPU-side buffer layouts for the `gpu`-feature compute infrastructure (see
//! `renderer::gpu`) and its mandatory struct-layout self-test
//! (`renderer::gpu::layout_check`).
//!
//! # Why every struct here is designed around WGSL's alignment rules first
//!
//! WGSL's host-shareable layout rules
//! (<https://www.w3.org/TR/WGSL/#alignment-and-size>) are NOT the same as Rust's
//! `#[repr(C)]` rules: a `vec3<f32>`/`vec4<f32>` struct member must start at a
//! 16-byte-aligned offset in WGSL, while the equivalent Rust `[f32; 3]`/`[f32; 4]`
//! field is only 4-byte aligned. A struct that "looks like" a direct translation --
//! same field order, same types -- can silently diverge in per-field byte offset (and
//! total size) the instant a smaller scalar sits in front of a vec3/vec4 field.
//!
//! That is exactly what happened to the previous version of [`DispersionParams`] in
//! this file: `model_type: u32` (4 bytes) was followed immediately by
//! `param_a: [f32; 4]` at Rust offset 4, while WGSL puts `param_a` at offset 16 (its
//! own 16-byte alignment requirement) -- every field after that was shifted out of
//! alignment the same way, and the two layouts disagreed on total size too (84 bytes in
//! Rust vs 96 in WGSL). Nothing caught this: the struct was unused scaffolding with no
//! GPU code path exercising it.
//!
//! Every struct below is laid out so each field lands on a WGSL-legal offset, either
//! because its Rust-natural offset already happens to be a multiple of its WGSL
//! alignment requirement (documented per-struct), or, where that isn't naturally true,
//! via explicit `_pad*` fields that reproduce WGSL's implicit padding byte-for-byte.
//! Hand-derived offset comments are not trusted on their own, though -- see
//! `renderer::gpu::layout_check` for the mandatory runtime GPU struct-echo test that is
//! this file's actual authority: it uploads a populated instance, has a compute shader
//! echo every field straight through to an independent output buffer, and compares the
//! raw bytes. A hand-derived comment can be wrong (or become wrong after an edit); the
//! echo test cannot lie about what the GPU actually did with the bytes it was given.

use crate::optics::{absorption::AbsorptionBand, raytracer::FacetFinish};
use core::mem::offset_of;

/// Per-frame camera/render-target uniform.
///
/// # Layout
///
/// Every field's Rust-natural offset already happens to land on a WGSL-legal boundary
/// for this exact field order and type set (`mat4x4<f32>`/`vec3<f32>` both need 16-byte
/// alignment in WGSL; every other field here is a plain scalar needing only 4-byte
/// alignment). Concretely: `camera_pos: vec3<f32>` and `c_axis: vec3<f32>` both happen
/// to start at offsets (64 and 96) that are already multiples of 16, so WGSL inserts no
/// padding either side of them that Rust wouldn't also insert -- this is a property of
/// this specific field order, not a general guarantee, which is why the `offset_of!`
/// assertions below pin it down explicitly rather than leaving it to happenstance:
/// reordering these fields could silently break the coincidence.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj_inv: [f32; 16],
    pub camera_pos: [f32; 3],
    pub frame_index: u32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub max_bounces: u32,
    pub gem_material_id: u32,
    pub c_axis: [f32; 3],
    pub env_intensity: f32,
}

const _: () = {
    assert!(offset_of!(CameraUniform, view_proj_inv) == 0);
    assert!(offset_of!(CameraUniform, camera_pos) == 64);
    assert!(offset_of!(CameraUniform, frame_index) == 76);
    assert!(offset_of!(CameraUniform, screen_width) == 80);
    assert!(offset_of!(CameraUniform, screen_height) == 84);
    assert!(offset_of!(CameraUniform, max_bounces) == 88);
    assert!(offset_of!(CameraUniform, gem_material_id) == 92);
    assert!(offset_of!(CameraUniform, c_axis) == 96);
    assert!(offset_of!(CameraUniform, env_intensity) == 108);
    assert!(size_of::<CameraUniform>() == 112);
};

/// One dispersion curve (`optics::dispersion::DispersionModel`), GPU-encoded.
///
/// `model_type` selects the interpretation of `param_a`/`param_b`:
/// - `0` (Sellmeier1 `{b1, c1}`): `param_a[0] = b1`, `param_b[0] = c1`.
/// - `1` (Sellmeier3 `{b: [f32;3], c: [f32;3]}`): `param_a[0..3] = b`, `param_b[0..3] = c`.
/// - `2` (Cauchy `{a, b, c}`): `param_a[0..3] = [a, b, c]`.
///
/// `c_axis_and_birefringence.xyz` is `GemMaterial::c_axis`; `.w` is
/// `GemMaterial::birefringence_delta`. `biaxial_delta_beta_alpha` /
/// `has_biaxial_delta` mirror `GemMaterial::biaxial_delta_beta_alpha: Option<f32>`
/// (`has_biaxial_delta != 0` <=> `Some`).
///
/// # Layout -- the known bug this struct previously had
///
/// See the module doc comment: the field immediately after `model_type` needs 16-byte
/// alignment in WGSL (every `param_*`/`c_axis_and_birefringence` field is a
/// `vec4<f32>`), so `_pad_after_model_type` reproduces WGSL's implicit 12-byte padding
/// explicitly. Every field from `param_a` through `c_axis_and_birefringence` is exactly
/// 16 bytes, so they pack back-to-back with no further padding; the trailing three
/// scalars (`is_anisotropic`, `biaxial_delta_beta_alpha`, `has_biaxial_delta`) only need
/// 4-byte alignment and pack tightly at the end -- WGSL rounds the struct's own size up
/// to its 16-byte struct alignment (inherited from the vec4 fields), so
/// `_pad_tail` reproduces that final 4-byte rounding explicitly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispersionParams {
    pub model_type: u32,
    _pad_after_model_type: [u32; 3],
    pub param_a: [f32; 4],
    pub param_b: [f32; 4],
    pub param_c: [f32; 4],
    pub c_axis_and_birefringence: [f32; 4],
    pub is_anisotropic: u32,
    pub biaxial_delta_beta_alpha: f32,
    pub has_biaxial_delta: u32,
    _pad_tail: f32,
}

const _: () = {
    assert!(offset_of!(DispersionParams, model_type) == 0);
    assert!(offset_of!(DispersionParams, param_a) == 16);
    assert!(offset_of!(DispersionParams, param_b) == 32);
    assert!(offset_of!(DispersionParams, param_c) == 48);
    assert!(offset_of!(DispersionParams, c_axis_and_birefringence) == 64);
    assert!(offset_of!(DispersionParams, is_anisotropic) == 80);
    assert!(offset_of!(DispersionParams, biaxial_delta_beta_alpha) == 84);
    assert!(offset_of!(DispersionParams, has_biaxial_delta) == 88);
    assert!(offset_of!(DispersionParams, _pad_tail) == 92);
    assert!(size_of::<DispersionParams>() == 96);
};

/// `model_type` discriminants for [`DispersionParams`] -- must match
/// `renderer/shaders/layout_echo.wgsl` and (eventually) any real dispersion-evaluating
/// kernel.
pub mod dispersion_model_type {
    pub const SELLMEIER1: u32 = 0;
    pub const SELLMEIER3: u32 = 1;
    pub const CAUCHY: u32 = 2;
}

/// Hard cap on how many [`GpuAbsorptionBand`]s either eigenmode of a [`GpuGemMaterial`]
/// can carry.
///
/// `GemMaterial::absorption`'s `Vec<AbsorptionBand>` is unbounded on the CPU side, but a
/// GPU encoding needs a fixed-capacity array. Enforced on scene ingest by
/// `apps/gemray-worker/src/validate.rs`'s `validate_scene`, so a scene that would
/// silently truncate on the GPU is rejected before it ever gets there.
///
/// 8 is comfortably above every built-in material's real band count (the widest is 3,
/// `legacy_rgb_bands`), while staying small enough that the fixed array costs nothing
/// worth measuring for materials that use far fewer.
pub const MAX_ABSORPTION_BANDS: usize = 8;

/// One Gaussian absorption band (`optics::absorption::AbsorptionBand`), GPU-encoded.
///
/// # Layout
///
/// All three fields are 4-byte-aligned `f32` scalars, so this struct's WGSL alignment
/// is 4 (not 16 -- there is no vec3/vec4 field here to trigger the usual pitfall) and
/// its size is exactly 12 bytes with no padding, matching Rust's natural `#[repr(C)]`
/// layout exactly. This is also why [`GpuGemMaterial`]'s band arrays are safe to declare
/// as plain `[GpuAbsorptionBand; MAX_ABSORPTION_BANDS]` with no per-element padding:
/// WGSL's storage-buffer array-stride rule only requires a stride that is a multiple of
/// the element's own alignment (4 here), unlike a *uniform* buffer's array, which would
/// additionally require the stride to be a multiple of 16. [`GpuGemMaterial`] is meant
/// for a storage buffer binding specifically because of this.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAbsorptionBand {
    pub center_nm: f32,
    pub width_nm: f32,
    pub peak: f32,
}

const _: () = {
    assert!(offset_of!(GpuAbsorptionBand, center_nm) == 0);
    assert!(offset_of!(GpuAbsorptionBand, width_nm) == 4);
    assert!(offset_of!(GpuAbsorptionBand, peak) == 8);
    assert!(size_of::<GpuAbsorptionBand>() == 12);
};

/// `crystal_system` discriminants for [`GpuGemMaterial`].
///
/// Must match `renderer/shaders/layout_echo.wgsl`'s and (eventually) any real
/// material-evaluating kernel's own numbering. Order matches
/// `optics::materials::CrystalSystem`'s own declaration order.
pub mod crystal_system {
    pub const CUBIC: u32 = 0;
    pub const TETRAGONAL: u32 = 1;
    pub const HEXAGONAL: u32 = 2;
    pub const TRIGONAL: u32 = 3;
    pub const ORTHORHOMBIC: u32 = 4;
    pub const MONOCLINIC: u32 = 5;
    pub const TRICLINIC: u32 = 6;
}
/// `optical_character` discriminants for [`GpuGemMaterial`].
///
/// Must match `renderer/shaders/layout_echo.wgsl`'s and (eventually) any real
/// material-evaluating kernel's own numbering. Order matches
/// `optics::materials::OpticalCharacter`'s own declaration order.
pub mod optical_character {
    pub const ISOTROPIC: u32 = 0;
    pub const UNIAXIAL_POSITIVE: u32 = 1;
    pub const UNIAXIAL_NEGATIVE: u32 = 2;
    pub const BIAXIAL_POSITIVE: u32 = 3;
    pub const BIAXIAL_NEGATIVE: u32 = 4;
}

/// A full `optics::materials::GemMaterial`, GPU-encoded.
///
/// [`DispersionParams`] plus crystal/optical-character discriminants and both
/// eigenmodes' absorption band sets (flattened to [`MAX_ABSORPTION_BANDS`]-capacity
/// arrays with an explicit count, per this crate's Phase-0 plan).
///
/// # Layout
///
/// `dispersion` is 96 bytes (a multiple of its own 16-byte WGSL struct alignment), so it
/// occupies offset 0..96 with no leading padding needed regardless of what follows.
/// `crystal_system` through `e_ray_band_count` are five 4-byte-aligned `u32` scalars
/// packing tightly at 96..116. `o_ray_bands`/`e_ray_bands` are arrays of
/// [`GpuAbsorptionBand`] (WGSL/Rust alignment 4, see that type's doc comment), so they
/// need only a 4-byte-aligned offset -- 116 already qualifies, no padding needed there
/// either. `scattering_sigma_s` through `edge_rounding_radius` pack tightly at
/// 308..320 (a 16-byte multiple, so no padding was needed there either, pre-Phase-4).
///
/// Phase 4 (biaxial GPU port): `beta_ray_band_count`/`has_beta_ray`/`beta_ray_bands`
/// are APPENDED after `edge_rounding_radius` rather than inserted alongside
/// `o_ray_band_count`/`e_ray_band_count` -- purely so every earlier field keeps its
/// existing offset unchanged (no shift for any already-shipped field, no
/// already-passing check anywhere in this crate needs updating for an offset it
/// doesn't actually reference). `beta_ray_band_count`/`has_beta_ray` are two more
/// 4-byte-aligned `u32` scalars (320..328), and `beta_ray_bands` is one more
/// [`GpuAbsorptionBand`] array needing only 4-byte alignment (328..424, same
/// reasoning as `o_ray_bands`/`e_ray_bands` above). P1 (absorption path scale):
/// `absorption_path_scale` is appended once more after `beta_ray_bands` (424..428, one
/// more 4-byte-aligned `f32` scalar), for the same "no shift for any earlier field"
/// reason. The struct's own WGSL alignment is still 16 (inherited from `dispersion`'s
/// vec4 members), so WGSL rounds the new total size up to the next multiple of 16:
/// 428 -> 432. `_pad_trailing` (now `[u32; 1]`, shrunk from `[u32; 2]`) reproduces that
/// implicit trailing padding explicitly so Rust's `size_of` agrees.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGemMaterial {
    pub dispersion: DispersionParams,
    pub crystal_system: u32,
    pub optical_character: u32,
    pub is_pleochroic: u32,
    pub o_ray_band_count: u32,
    pub e_ray_band_count: u32,
    pub o_ray_bands: [GpuAbsorptionBand; MAX_ABSORPTION_BANDS],
    pub e_ray_bands: [GpuAbsorptionBand; MAX_ABSORPTION_BANDS],
    /// Inclusion/subsurface scattering: mirrors
    /// `optics::materials::GemMaterial::scattering_sigma_s`/`scattering_g` exactly --
    /// see those fields' doc comments. Slotted into two of the three trailing `u32`
    /// padding slots this struct already carried (`_pad_trailing` shrinks from `[u32;
    /// 3]` to a single `u32`), so the struct's total size stays 320 bytes -- no
    /// offset shift for any earlier field, and no WGSL-side re-derivation of the
    /// trailing padding needed.
    pub scattering_sigma_s: f32,
    pub scattering_g: f32,
    /// Facet edge rounding: mirrors
    /// `optics::materials::GemMaterial::edge_rounding_radius` exactly. Takes the LAST
    /// remaining trailing padding slot (`_pad_trailing` shrinks from `u32` to nothing),
    /// so the struct's total size stays 320 bytes.
    pub edge_rounding_radius: f32,
    /// Phase 4 (biaxial GPU port): mirrors `optics::optics::absorption::AbsorptionTensor::beta_ray`'s
    /// presence -- `optics::materials::GemMaterial::absorption.beta_ray.is_some()`.
    /// `beta_ray_band_count`/`beta_ray_bands` are meaningless when this is 0 (mirroring
    /// `o_ray_band_count`/`e_ray_band_count`'s own "count governs which array entries
    /// are real" convention, `has_beta_ray` additionally distinguishes "genuinely no
    /// third band set" from "a third band set with zero bands", the same distinction
    /// `Option<Vec<AbsorptionBand>>` makes on the CPU side that a bare count cannot).
    pub has_beta_ray: u32,
    /// The third, `n_beta`, principal direction's absorption bands' count -- see
    /// `has_beta_ray`.
    pub beta_ray_band_count: u32,
    /// The third, `n_beta`, principal direction's absorption bands -- see
    /// `has_beta_ray`. Mirrors `o_ray_bands`/`e_ray_bands`'s own fixed-capacity
    /// encoding exactly (same [`GpuAbsorptionBand`] element type, same
    /// [`MAX_ABSORPTION_BANDS`] capacity).
    pub beta_ray_bands: [GpuAbsorptionBand; MAX_ABSORPTION_BANDS],
    /// P1 (absorption path scale): mirrors
    /// `optics::materials::GemMaterial::absorption_path_scale` exactly -- every model-unit
    /// length that enters Beer-Lambert absorption or the scattering estimator is
    /// multiplied by this before use (both in the CPU tracer and in
    /// `spectral_transport.wgsl`'s mirrored absorption/scatter blocks). Takes one of the
    /// two remaining trailing padding slots (`_pad_trailing` shrinks from `[u32; 2]` to
    /// `[u32; 1]`), so the struct's total size stays 432 bytes -- no offset shift for any
    /// earlier field.
    pub absorption_path_scale: f32,
    /// Explicit trailing padding: unlike WGSL (which rounds a storage-buffer struct's
    /// size up to its own 16-byte struct alignment, inherited here from `dispersion`'s
    /// `vec4<f32>` members), Rust's `#[repr(C)]` never infers alignment-16 padding
    /// from a struct that has no field whose OWN Rust type demands it -- every field
    /// in this struct is `u32`/`f32`/a fixed-size array of one of those (all
    /// naturally 4-byte aligned in Rust, unlike WGSL's `vec4<f32>`), so without this
    /// field Rust would size the struct at exactly 428 bytes (328 + 96 + 4), 4 bytes
    /// short of WGSL's 432. See the module doc comment's general warning about this
    /// exact divergence class.
    _pad_trailing: [u32; 1],
}

const _: () = {
    assert!(offset_of!(GpuGemMaterial, dispersion) == 0);
    assert!(offset_of!(GpuGemMaterial, crystal_system) == 96);
    assert!(offset_of!(GpuGemMaterial, optical_character) == 100);
    assert!(offset_of!(GpuGemMaterial, is_pleochroic) == 104);
    assert!(offset_of!(GpuGemMaterial, o_ray_band_count) == 108);
    assert!(offset_of!(GpuGemMaterial, e_ray_band_count) == 112);
    assert!(offset_of!(GpuGemMaterial, o_ray_bands) == 116);
    assert!(offset_of!(GpuGemMaterial, e_ray_bands) == 212);
    assert!(offset_of!(GpuGemMaterial, scattering_sigma_s) == 308);
    assert!(offset_of!(GpuGemMaterial, scattering_g) == 312);
    assert!(offset_of!(GpuGemMaterial, edge_rounding_radius) == 316);
    assert!(offset_of!(GpuGemMaterial, has_beta_ray) == 320);
    assert!(offset_of!(GpuGemMaterial, beta_ray_band_count) == 324);
    assert!(offset_of!(GpuGemMaterial, beta_ray_bands) == 328);
    assert!(offset_of!(GpuGemMaterial, absorption_path_scale) == 424);
    assert!(size_of::<GpuGemMaterial>() == 432);
};

// ---------------------------------------------------------------------------------
// Phase 1: geometry/environment GPU-check structs.
//
// Every struct below carries state across the CPU/GPU boundary for a Phase-1 self-test
// (`renderer::gpu::{camera_check, polyhedron_check, environment_check, furnace_check}`),
// so each one gets its own `renderer::gpu::layout_check` echo test too -- see that
// module's doc comment for why a hand-derived offset comment is never trusted on its
// own. Every vec3 field here is deliberately followed immediately by a plain `f32`
// scalar (never left to trail into unused padding) so the WGSL-mandated 16-byte
// alignment of the *next* vec3 is met with no separate `_pad*` field needed -- the same
// "vec3 + scalar packs to 16 bytes" pattern documented on `CameraUniform` above.
// ---------------------------------------------------------------------------------

/// A camera pose's screen-space ray-generation basis (`optics::raytracer::Camera`),
/// GPU-encoded.
///
/// Produced by porting `Camera::new` (from `(yaw, pitch, distance, fov_deg)`) and
/// consumed by porting `Camera::generate_ray`.
///
/// # Layout
///
/// `origin`/`forward`/`right` each pack with the scalar immediately following them into
/// a 16-byte block (see the section doc comment); `up` is the last vec3 and needs no
/// trailing scalar to pad it out since `num_samples` -- itself only 4-byte-aligned --
/// already lands on a legal offset (60) right after it, with the whole 64-byte struct
/// already a multiple of its own 16-byte alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCameraParams {
    pub origin: [f32; 3],
    pub fov_tan: f32,
    pub forward: [f32; 3],
    pub width: f32,
    pub right: [f32; 3],
    pub height: f32,
    pub up: [f32; 3],
    pub num_samples: u32,
}

const _: () = {
    assert!(offset_of!(GpuCameraParams, origin) == 0);
    assert!(offset_of!(GpuCameraParams, fov_tan) == 12);
    assert!(offset_of!(GpuCameraParams, forward) == 16);
    assert!(offset_of!(GpuCameraParams, width) == 28);
    assert!(offset_of!(GpuCameraParams, right) == 32);
    assert!(offset_of!(GpuCameraParams, height) == 44);
    assert!(offset_of!(GpuCameraParams, up) == 48);
    assert!(offset_of!(GpuCameraParams, num_samples) == 60);
    assert!(size_of::<GpuCameraParams>() == 64);
};

/// A traced ray (`optics::raytracer::Ray`), GPU-encoded. Both an intersection kernel's
/// input and a camera-ray-generation kernel's output.
///
/// # Layout
///
/// Neither vec3 here has a natural scalar to pack with (a `Ray` is just two vec3s), so
/// each needs an explicit `_pad*` field reproducing WGSL's implicit trailing padding.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRay {
    pub origin: [f32; 3],
    _pad0: f32,
    pub dir: [f32; 3],
    _pad1: f32,
}

impl GpuRay {
    #[must_use]
    pub const fn new(origin: [f32; 3], dir: [f32; 3]) -> Self {
        Self {
            origin,
            _pad0: 0.0,
            dir,
            _pad1: 0.0,
        }
    }
}

const _: () = {
    assert!(offset_of!(GpuRay, origin) == 0);
    assert!(offset_of!(GpuRay, dir) == 16);
    assert!(size_of::<GpuRay>() == 32);
};

/// `intersect_polyhedron`'s `Option<HitRecord>` result, GPU-encoded.
///
/// `hit == 0` encodes `None`; `hit != 0` encodes `Some(HitRecord { t, normal,
/// facet_idx })` (`facet_idx` stored as `i32` with `-1` reserved as an additional "no
/// hit" sentinel for
/// diagnostics, mirroring `intersect_polyhedron`'s own `.unwrap_or(0)` fallback never
/// actually being reachable on the Rust side either -- `near_facet`/`far_facet` are
/// always `Some` whenever `t_near`/`t_far` were updated from their `+-1e30` sentinels,
/// which is exactly when a hit is reported).
///
/// # Layout
///
/// `t`/`facet_idx`/`hit`/`_pad0` are four 4-byte-aligned scalars packing tightly into a
/// 16-byte block, so `normal` (a vec3, needing 16-byte alignment) lands at offset 16
/// with no additional padding; its own trailing 4 bytes are reproduced by `_pad1`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuHitRecord {
    pub t: f32,
    pub facet_idx: i32,
    pub hit: u32,
    _pad0: u32,
    pub normal: [f32; 3],
    _pad1: f32,
}

impl GpuHitRecord {
    #[must_use]
    pub const fn miss() -> Self {
        Self {
            t: 0.0,
            facet_idx: -1,
            hit: 0,
            _pad0: 0,
            normal: [0.0, 0.0, 0.0],
            _pad1: 0.0,
        }
    }

    #[must_use]
    pub const fn hit(t: f32, facet_idx: i32, normal: [f32; 3]) -> Self {
        Self {
            t,
            facet_idx,
            hit: 1,
            _pad0: 0,
            normal,
            _pad1: 0.0,
        }
    }
}

const _: () = {
    assert!(offset_of!(GpuHitRecord, t) == 0);
    assert!(offset_of!(GpuHitRecord, facet_idx) == 4);
    assert!(offset_of!(GpuHitRecord, hit) == 8);
    assert!(offset_of!(GpuHitRecord, normal) == 16);
    assert!(size_of::<GpuHitRecord>() == 32);
};

// ---------------------------------------------------------------------------------
// Phase 2: the isotropic spectral estimator (`shaders/spectral_transport.wgsl`, driven
// by `renderer::gpu::{transport_check, estimator_check}`). See those modules' own doc
// comments for what each self-test exercises.
// ---------------------------------------------------------------------------------

/// Per-dispatch kernel parameters for `shaders/spectral_transport.wgsl`'s
/// `transport_main` entry point.
///
/// `env_mode` selects which environment model to sample (`0` = the direction-independent
/// "uniform furnace" grey environment at `l0`, `1` = the analytic studio rig at the
/// given colour temperature/exposure/rig-pose). `sample_offset` + `camera.num_samples`
/// select the sample range this dispatch should trace, mirroring
/// `apps/gemray-worker/src/render_core.rs`'s own `first_sample`/`samples` convention --
/// this is what lets `estimator_check`'s Tier 3 statistical comparison give the CPU and
/// GPU DISJOINT sample ranges, as production would. `white_balance` is the precomputed
/// von-Kries white balance (`compute_illuminant_white_balance`, already ULP-verified by
/// Phase 1's `environment_check`; recomputing its 401-point quadrature per-thread here
/// would re-test Phase 1 machinery instead of Phase 2's own transport physics). This is
/// a **Bradford LMS-space** diagonal scale, not an XYZ-space one -- the
/// megakernel (`shaders/spectral_transport.wgsl`'s `apply_von_kries_white_balance`)
/// transforms to Bradford LMS, applies this scale, and transforms back, mirroring
/// `optics::raytracer::apply_von_kries_white_balance` on the CPU side exactly.
///
/// # Layout
///
/// The ten leading `u32`/`f32` scalars pack tightly into 40 bytes;
/// `pixel_offset`/`write_debug_buffers` (2 more scalars, the latter R4's repurposed
/// `_pad1`) bring that to 48 -- a multiple of 16 -- so `white_balance`
/// (`vec3<f32>`, 16-byte WGSL alignment) needs no further padding before it. The
/// struct's own WGSL alignment is 16 (inherited from `white_balance`), and `12 + 48 =
/// 60` is not itself a multiple of 16, so WGSL rounds the struct size up to 64 -- there
/// is no explicit `_pad2` field for this because `white_balance: [f32; 3]` occupying
/// bytes 48..60 already leaves Rust's own natural `#[repr(C)]` size at 60, and adding a
/// trailing `f32` reproduces WGSL's implicit 4-byte tail padding explicitly (see the
/// module doc comment's general rationale for why this crate never leaves such padding
/// to happenstance).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTransportParams {
    pub num_pixels: u32,
    pub max_bounces: u32,
    pub sample_offset: u32,
    pub env_mode: u32,
    pub l0: f32,
    pub studio_temp_k: f32,
    pub studio_spot_mult: f32,
    pub studio_exposure: f32,
    pub studio_light_yaw: f32,
    pub studio_light_pitch: f32,
    /// Index of the first pixel this dispatch covers, added to the shader's own
    /// `idx / num_samples` to recover a GLOBAL pixel index for camera-ray generation
    /// while output slots stay dispatch-local. Zero for a dispatch covering a whole
    /// frame, which is every call in `renderer::gpu`'s self-tests -- it exists for
    /// `renderer::gpu::frame`, which splits a frame too large to fit a memory budget
    /// into chunks (see that module's doc comment).
    ///
    /// Deliberately reuses what was `_pad0`: a `u32` in a slot that was already four
    /// bytes of explicit padding, so every following field's offset is unchanged and
    /// `layout_check`'s echo test needs no new expected offsets.
    pub pixel_offset: u32,
    /// R4: whether `transport_main` should write its three per-channel debug output
    /// buffers (`out_radiance`/`out_lambdas`/`out_path_pdf`) this dispatch. Nonzero
    /// (the default -- see [`Self::new`]) reproduces every existing dispatch's exact
    /// behaviour, debug writes included, so every self-test in `renderer::gpu` that
    /// reads those buffers keeps working unchanged. Zero (see
    /// [`Self::with_debug_buffers_disabled`]) skips those three writes -- only
    /// `out_xyz` (the only buffer [`super::gpu::frame::GpuFrameRenderer::accumulate`]
    /// ever reads back) is written -- so a production dispatch does 9x less write
    /// traffic and its chunk budget holds 9x more samples per dispatch. Reuses what
    /// was `_pad1` for the same "no offset shifts, no new `layout_check` expectations"
    /// reason `pixel_offset` reused `_pad0`.
    pub write_debug_buffers: u32,
    pub white_balance: [f32; 3],
    _pad2: f32,
}

/// `env_mode` discriminants for [`GpuTransportParams`]. Must match
/// `shaders/spectral_transport.wgsl`'s own `params.env_mode` branch.
pub mod transport_env_mode {
    pub const UNIFORM_FURNACE: u32 = 0;
    pub const STUDIO_RIG: u32 = 1;
}

impl GpuTransportParams {
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "one parameter per GpuTransportParams field, in field order, because \
                  that field order is itself the #[repr(C)] layout spectral_transport.wgsl's \
                  uniform buffer binds against; bundling them into a context struct here \
                  would just re-introduce this exact struct one level removed"
    )]
    pub const fn new(
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
        white_balance: [f32; 3],
    ) -> Self {
        Self {
            num_pixels,
            max_bounces,
            sample_offset,
            env_mode,
            l0,
            studio_temp_k,
            studio_spot_mult,
            studio_exposure,
            studio_light_yaw,
            studio_light_pitch,
            pixel_offset: 0,
            write_debug_buffers: 1,
            white_balance,
            _pad2: 0.0,
        }
    }

    /// Returns a copy covering pixels `[pixel_offset, pixel_offset + num_pixels)` of a
    /// larger frame -- see [`Self::pixel_offset`].
    ///
    /// A separate builder rather than an eleventh parameter to [`Self::new`], so every
    /// existing caller (all of `renderer::gpu`'s self-tests, whose baselines are
    /// byte-reproduced across refactors) keeps the exact constructor call it had.
    #[must_use]
    pub const fn with_pixel_offset(mut self, pixel_offset: u32) -> Self {
        self.pixel_offset = pixel_offset;
        self
    }

    /// Returns a copy that skips `transport_main`'s three per-channel debug output
    /// writes -- see [`Self::write_debug_buffers`]'s doc comment. Only
    /// `GpuFrameRenderer::accumulate`'s production dispatch calls this; every
    /// self-test keeps [`Self::new`]'s default (debug buffers written, exactly the
    /// pre-R4 behaviour).
    #[must_use]
    pub const fn with_debug_buffers_disabled(mut self) -> Self {
        self.write_debug_buffers = 0;
        self
    }
}

const _: () = {
    assert!(offset_of!(GpuTransportParams, num_pixels) == 0);
    assert!(offset_of!(GpuTransportParams, max_bounces) == 4);
    assert!(offset_of!(GpuTransportParams, sample_offset) == 8);
    assert!(offset_of!(GpuTransportParams, env_mode) == 12);
    assert!(offset_of!(GpuTransportParams, l0) == 16);
    assert!(offset_of!(GpuTransportParams, studio_temp_k) == 20);
    assert!(offset_of!(GpuTransportParams, studio_spot_mult) == 24);
    assert!(offset_of!(GpuTransportParams, studio_exposure) == 28);
    assert!(offset_of!(GpuTransportParams, studio_light_yaw) == 32);
    assert!(offset_of!(GpuTransportParams, studio_light_pitch) == 36);
    assert!(offset_of!(GpuTransportParams, white_balance) == 48);
    assert!(size_of::<GpuTransportParams>() == 64);
};

/// Encodes a CPU `optics::materials::GemMaterial` into a [`GpuGemMaterial`] for upload.
///
/// Never a hand-copied duplicate of the material data: every field read here is the
/// SAME field `optics::raytracer::trace_spectral_ray` itself reads (`material.dispersion`,
/// `material.crystal_system`, `material.absorption.{o_ray,e_ray}`, `material.c_axis`,
/// `material.birefringence_delta`) -- see `renderer::gpu::estimator_check` and
/// `renderer::gpu::transport_check`'s doc comments for why this matters: the whole
/// point of this equivalence harness is that the GPU sees the exact scene the CPU
/// reference traced, not a re-derived approximation of it.
///
/// Phase 4 (biaxial GPU port): `material.absorption.beta_ray` (the optional third,
/// trichroic band set) IS now read and encoded, into `has_beta_ray`/
/// `beta_ray_band_count`/`beta_ray_bands` -- unlike before this port, when it was
/// deliberately left unencoded because no shader-side consumer existed for it (every
/// biaxial material was CPU-only). `shaders/spectral_transport.wgsl`'s `transport_main`
/// now has a genuinely biaxial (`AbsorptionTensor3`-equivalent, three-independent-
/// principal-coefficient) absorption path that consumes this data whenever
/// `dispersion.has_biaxial_delta != 0` -- see that shader's own doc comment.
impl GpuGemMaterial {
    /// # Panics
    ///
    /// Panics if `material` has more than [`MAX_ABSORPTION_BANDS`] bands in either
    /// eigenmode's `Vec<AbsorptionBand>` -- every built-in material has at most 3 (see
    /// [`MAX_ABSORPTION_BANDS`]'s doc comment), so this is only reachable for a
    /// hand-constructed test material that exceeds the GPU's fixed-capacity encoding;
    /// acceptable to panic in this self-test-only encoder rather than silently
    /// truncate a band set the equivalence check would then be comparing against a
    /// materially different material.
    #[must_use]
    pub fn encode(material: &crate::optics::materials::GemMaterial) -> Self {
        use crate::optics::materials::{CrystalSystem, OpticalCharacter};

        let crystal_system_val = match material.crystal_system {
            CrystalSystem::Cubic => crystal_system::CUBIC,
            CrystalSystem::Tetragonal => crystal_system::TETRAGONAL,
            CrystalSystem::Hexagonal => crystal_system::HEXAGONAL,
            CrystalSystem::Trigonal => crystal_system::TRIGONAL,
            CrystalSystem::Orthorhombic => crystal_system::ORTHORHOMBIC,
            CrystalSystem::Monoclinic => crystal_system::MONOCLINIC,
            CrystalSystem::Triclinic => crystal_system::TRICLINIC,
        };
        let optical_character_val = match material.optical_character {
            OpticalCharacter::Isotropic => optical_character::ISOTROPIC,
            OpticalCharacter::UniaxialPositive => optical_character::UNIAXIAL_POSITIVE,
            OpticalCharacter::UniaxialNegative => optical_character::UNIAXIAL_NEGATIVE,
            OpticalCharacter::BiaxialPositive => optical_character::BIAXIAL_POSITIVE,
            OpticalCharacter::BiaxialNegative => optical_character::BIAXIAL_NEGATIVE,
        };

        let (o_ray_bands, o_ray_band_count) = encode_bands(&material.absorption.o_ray);
        let (e_ray_bands, e_ray_band_count) = encode_bands(&material.absorption.e_ray);
        let (beta_ray_bands, beta_ray_band_count) = material
            .absorption
            .beta_ray
            .as_deref()
            .map_or_else(empty_bands, encode_bands);

        Self {
            dispersion: DispersionParams::encode(material),
            crystal_system: crystal_system_val,
            optical_character: optical_character_val,
            is_pleochroic: u32::from(material.absorption.is_pleochroic),
            o_ray_band_count,
            e_ray_band_count,
            o_ray_bands,
            e_ray_bands,
            scattering_sigma_s: material.scattering_sigma_s,
            scattering_g: material.scattering_g,
            edge_rounding_radius: material.edge_rounding_radius,
            has_beta_ray: u32::from(material.absorption.beta_ray.is_some()),
            beta_ray_band_count,
            beta_ray_bands,
            absorption_path_scale: material.absorption_path_scale,
            _pad_trailing: [0; 1],
        }
    }
}

impl DispersionParams {
    #[must_use]
    fn encode(material: &crate::optics::materials::GemMaterial) -> Self {
        use crate::optics::dispersion::DispersionModel;

        let (model_type, param_a, param_b, param_c) = match material.dispersion {
            DispersionModel::Sellmeier1 { b1, c1 } => (
                dispersion_model_type::SELLMEIER1,
                [b1, 0.0, 0.0, 0.0],
                [c1, 0.0, 0.0, 0.0],
                [0.0; 4],
            ),
            DispersionModel::Sellmeier3 { b, c } => (
                dispersion_model_type::SELLMEIER3,
                [b[0], b[1], b[2], 0.0],
                [c[0], c[1], c[2], 0.0],
                [0.0; 4],
            ),
            DispersionModel::Cauchy { a, b, c } => (
                dispersion_model_type::CAUCHY,
                [a, b, c, 0.0],
                [0.0; 4],
                [0.0; 4],
            ),
        };

        let c_axis = material.c_axis;
        Self {
            model_type,
            _pad_after_model_type: [0; 3],
            param_a,
            param_b,
            param_c,
            c_axis_and_birefringence: [c_axis.x, c_axis.y, c_axis.z, material.birefringence_delta],
            is_anisotropic: u32::from(
                material.crystal_system != crate::optics::materials::CrystalSystem::Cubic
                    && material.birefringence_delta.abs() > 1e-4,
            ),
            biaxial_delta_beta_alpha: material.biaxial_delta_beta_alpha.unwrap_or(0.0),
            has_biaxial_delta: u32::from(material.biaxial_delta_beta_alpha.is_some()),
            _pad_tail: 0.0,
        }
    }
}

/// The all-zero-bands, zero-count encoding used for `beta_ray_bands` when
/// `material.absorption.beta_ray` is `None` -- mirrors [`encode_bands`]'s own default
/// slot values (`width_nm: 1.0`, everything else `0.0`) so an unused slot is never a
/// degenerate (zero-width) Gaussian even though its `peak` is always `0.0` regardless.
const fn empty_bands() -> ([GpuAbsorptionBand; MAX_ABSORPTION_BANDS], u32) {
    (
        [GpuAbsorptionBand {
            center_nm: 0.0,
            width_nm: 1.0,
            peak: 0.0,
        }; MAX_ABSORPTION_BANDS],
        0,
    )
}

/// Encodes a `Vec<AbsorptionBand>` into a fixed [`MAX_ABSORPTION_BANDS`]-capacity array
/// plus its real length, panicking (see [`GpuGemMaterial::encode`]'s doc comment) if the
/// source has more bands than the GPU encoding can hold.
fn encode_bands(bands: &[AbsorptionBand]) -> ([GpuAbsorptionBand; MAX_ABSORPTION_BANDS], u32) {
    assert!(
        bands.len() <= MAX_ABSORPTION_BANDS,
        "material has {} absorption bands, exceeding MAX_ABSORPTION_BANDS ({MAX_ABSORPTION_BANDS})",
        bands.len()
    );
    let mut out = [GpuAbsorptionBand {
        center_nm: 0.0,
        width_nm: 1.0,
        peak: 0.0,
    }; MAX_ABSORPTION_BANDS];
    for (slot, band) in out.iter_mut().zip(bands.iter()) {
        *slot = GpuAbsorptionBand {
            center_nm: band.center_nm,
            width_nm: band.width_nm,
            peak: band.peak,
        };
    }
    (out, bands.len() as u32)
}

// ---------------------------------------------------------------------------------
// Girdle finish (bruted/frosted facets).
//
// `optics::raytracer::FacetFinish` is looked up per-hit via `HitRecord::facet_idx` into
// a `&[FacetFinish]` slice PARALLEL to `&[GpuFacetPlane]` on the CPU (see that enum's
// own doc comment for why: extending `GpuFacetPlane` itself would disturb that struct's
// byte layout, its own Phase-1 echo test, and the Tier 2 `intersect_polyhedron` kernel,
// none of which need to know about finish at all). The GPU encoding mirrors that
// decision with a SEPARATE storage buffer (`array<u32>`, one entry per facet, bound
// alongside -- not merged into -- the existing `planes` buffer) rather than widening
// `GpuFacetPlane`'s layout, for the exact same reason: it costs one extra bind-group
// slot instead of perturbing an already-shipped, already-echo-tested struct and its
// Tier 2 kernel. A bare `u32` per facet needs no `#[repr(C)]`/`offset_of!` layout
// reasoning of its own (no vec3/vec4 alignment pitfall is possible for a flat scalar
// array -- see the module doc comment's general warning, which this type is simple
// enough to sidestep entirely) but still gets a Tier 1 struct-echo test
// (`renderer::gpu::layout_check::run_facet_finish`) for the same reason every other
// host-uploaded type here does: an echo test proves what the GPU actually did with the
// uploaded bytes, a doc comment does not.
pub mod facet_finish {
    pub const POLISHED: u32 = 0;
    pub const FROSTED: u32 = 1;
}

/// Encodes a `&[FacetFinish]` slice into a `facet_finish::{POLISHED,FROSTED}`-valued
/// `Vec<u32>` of exactly `num_planes` entries.
///
/// Mirrors `optics::raytracer::trace_spectral_ray_with_finish`'s own per-facet lookup
/// semantics (`facet_finishes.get(i).copied().unwrap_or_default()`, i.e. a facet index
/// past the end of a shorter slice -- or the whole slice being empty -- defaults to
/// `FacetFinish::Polished`) so that passing `&[]` here uploads an all-`POLISHED` buffer
/// of the same length a full `vec![FacetFinish::Polished; num_planes]` would, matching
/// the CPU's own "no explicit finish means polished" default exactly.
#[must_use]
pub fn encode_facet_finishes(finishes: &[FacetFinish], num_planes: usize) -> Vec<u32> {
    (0..num_planes)
        .map(|i| match finishes.get(i).copied().unwrap_or_default() {
            FacetFinish::Polished => facet_finish::POLISHED,
            FacetFinish::Frosted => facet_finish::FROSTED,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `bytemuck::Pod` already guards against internal padding *bytes* being
    /// uninitialized (it would refuse to derive if the compiler couldn't prove the
    /// type has none), but it says nothing about whether the offsets line up with
    /// WGSL's rules -- that's what the `offset_of!` assertions above are for. This
    /// test just pins the sizes these structs are documented to have, so a future
    /// field addition/removal that silently changes a size gets caught here too, in
    /// addition to the `const _` assertions (which already run at compile time; this
    /// duplicates them as ordinary `#[test]`s purely so `cargo test -p gemray` prints
    /// a named failure instead of a compile error buried in this module).
    #[test]
    fn struct_sizes_match_documented_wgsl_layout() {
        assert_eq!(size_of::<CameraUniform>(), 112);
        assert_eq!(size_of::<DispersionParams>(), 96);
        assert_eq!(size_of::<GpuAbsorptionBand>(), 12);
        assert_eq!(size_of::<GpuGemMaterial>(), 432);
        assert_eq!(size_of::<GpuTransportParams>(), 64);
    }

    /// [`encode_facet_finishes`]'s default-fallback semantics must match
    /// `trace_spectral_ray_with_finish`'s own `.get(i).copied().unwrap_or_default()`
    /// lookup exactly: an empty slice, or one shorter than `num_planes`, defaults every
    /// uncovered index to `facet_finish::POLISHED`.
    #[test]
    fn encode_facet_finishes_defaults_to_polished() {
        assert_eq!(encode_facet_finishes(&[], 4), vec![0, 0, 0, 0]);

        let finishes = vec![FacetFinish::Frosted, FacetFinish::Polished];
        assert_eq!(
            encode_facet_finishes(&finishes, 4),
            vec![
                facet_finish::FROSTED,
                facet_finish::POLISHED,
                facet_finish::POLISHED,
                facet_finish::POLISHED,
            ]
        );
    }
}

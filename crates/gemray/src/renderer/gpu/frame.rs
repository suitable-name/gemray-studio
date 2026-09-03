//! Production GPU frame rendering: the general entry point to the `transport_main`
//! megakernel that `renderer::gpu`'s self-tests verify.
//!
//! # Why this module exists
//!
//! Every other public entry point in `renderer::gpu` is a *check*: it renders one
//! hardcoded scene (`estimator_check::furnace_material`, `test_camera`, a fixed 48x48)
//! and reports a verdict, because that is all Phases 0-3 needed. The physics was
//! therefore implemented and verified without ever being *callable* on an arbitrary
//! scene -- there was no way to hand it a user's gem, camera and resolution and get
//! pixels back. This module is that missing function, and nothing more: it introduces
//! no physics of its own.
//!
//! # Sharing the verified path, not copying it
//!
//! [`encode_and_dispatch`] is the single buffer-binding and dispatch routine, used both
//! here and by `estimator_check`'s own `dispatch_transport`. That direction matters: the
//! Tier 2/Tier 3 equivalence checks exercise the code this module ships, in the same way
//! `shaders/transport_physics.wgsl` is concatenated into both the megakernel and the
//! Tier 2 kernels rather than duplicated between them. A parallel "production" dispatch
//! path would be unverified by construction, however similar it looked.
//!
//! # Chunking
//!
//! `transport_main` writes up to four output buffers per (pixel, sample) thread: final
//! XYZ plus three 8-channel debug arrays (radiance, lambdas, `path_pdf`), 27 floats (108
//! bytes) per tuple in total. Every self-test in `renderer::gpu` reads the debug arrays
//! too (Tier 2 ULP checks, the spectral-debug self-consistency check), so the shader
//! still writes all four whenever asked -- but this module's production dispatch never
//! reads anything but XYZ (see the module doc comment's "No guide buffers" bullet: no
//! consumer here wants per-channel radiance/lambdas/`path_pdf`). R4:
//! `GpuTransportParams::write_debug_buffers` (a uniform flag, defaulting to "on" so every
//! self-test's behaviour is unchanged) lets THIS module's dispatches skip those three
//! writes entirely -- [`GpuFrameScene`]'s params are built via
//! `GpuTransportParams::with_debug_buffers_disabled`, and [`TransportOutputs::new_production`]
//! backs the skipped writes with tiny fixed-size buffers instead of ones sized like a
//! chunk. A chunk's byte budget is therefore spent on 3 floats (12 bytes) per tuple, not
//! 27 (108 bytes) -- 9x more samples per [`CHUNK_BUDGET_BYTES`] dispatch, and what a
//! 1920x1080 frame at 4 spp needed in one allocation drops from ~855 MiB to ~95 MiB.
//!
//! So a frame is split into pixel chunks that fit [`CHUNK_BUDGET_BYTES`], dispatched in
//! sequence. `GpuTransportParams::pixel_offset` (which reuses a padding slot, so no
//! struct layout changed) tells the shader where a chunk starts, keeping camera-ray
//! generation and the per-pixel Cranley-Patterson rotations a function of the pixel's
//! true place in the frame while output slots stay chunk-local.
//!
//! Chunking is over pixels, never over samples: a pixel's samples all land in one
//! dispatch. Splitting them would be harmless for correctness (each thread's output
//! depends only on its own `(pixel, sample_num)`) but would re-upload the whole scene
//! per sample for nothing.
//!
//! # Overlapped chunk pipeline
//!
//! Before R4, one chunk's dispatch was `submit -> poll(wait) -> submit copy -> poll(wait)
//! -> map -> read -> sum`, entirely serial: the GPU sat idle during the CPU-side
//! poll/map/read/sum, and the CPU sat idle during the GPU's dispatch. `accumulate` now
//! alternates between TWO [`TransportOutputs`] (`self.outputs[chunk_index % 2]`), each
//! with its own staging buffer: chunk i's dispatch and readback-copy are both submitted
//! (via [`compute::dispatch`]/[`compute::copy_to_staging`], neither blocking) BEFORE
//! chunk i-1's result is mapped/read/summed, so chunk i's GPU work is already queued
//! while the CPU blocks on chunk i-1's map. Reusing `outputs[chunk_index % 2]` two
//! chunks later (chunk i+2 reuses chunk i's slot) is safe because chunk i's own result is
//! always drained (mapped, read, unmapped) before chunk i+2's dispatch is even built --
//! the pending queue is exactly one chunk deep. Every thread's output still depends only
//! on its own `(pixel, sample_num)`, so this reordering of WHEN a chunk's result is read
//! back changes nothing about WHAT is computed -- results stay bit-identical, which is
//! what [`run_chunk_equivalence`] checks.
//!
//! # Material-class kernel specialisation
//!
//! `transport_main` handles three material classes with runtime branches inside one
//! megakernel (isotropic/cubic, uniaxial ordinary/extraordinary, biaxial mode-A/mode-B),
//! and until this task every dispatch carried every class's per-ray register state --
//! the biaxial indicatrix axes/wave directions/per-channel index arrays, the uniaxial
//! `theta_c`/eigenmode-split machinery -- even for a plain isotropic diamond, because
//! `is_anisotropic`/`is_biaxial` were ordinary runtime booleans no compiler could prove
//! false ahead of time. [`compute::create_compute_pipeline_with_constants`] now lets a
//! `wgpu::ComputePipeline` fix `spectral_transport.wgsl`'s `MATERIAL_CLASS`
//! pipeline-overridable constant at PIPELINE creation time (see that override's own
//! doc comment for the mechanism); [`GpuFrameRenderer`] builds one specialised pipeline
//! per class LAZILY, the first time [`GpuFrameRenderer::accumulate`] dispatches that
//! class (`pipeline_isotropic`/`pipeline_uniaxial`/`pipeline_biaxial`, alongside the
//! GENERIC pipeline every self-test in this crate keeps compiling and dispatching
//! unmodified) -- lazy rather than eager in [`GpuFrameRenderer::new`] because a caller
//! (a still image, an interactive viewport) may only ever render one or two of the
//! three classes in a session, and eager compilation of all four pipelines up front
//! would pay every class's shader-compile cost on startup regardless.
//!
//! [`classify_material`] is the single place that decision is made, and it MIRRORS
//! (never duplicates the meaning of) `renderer::buffers::GpuGemMaterial::encode`'s own
//! `is_anisotropic`/`has_biaxial_delta` derivation exactly: biaxial takes priority
//! (the material's `biaxial_delta_beta_alpha` field is set), then uniaxial (a
//! non-cubic crystal system with a birefringence magnitude above the same 1e-4
//! threshold `encode` uses), else isotropic. [`GpuFrameRenderer::accumulate`] calls it
//! once per scene and dispatches every chunk of that frame through the resulting
//! pipeline -- see `accumulate_via_pipeline`'s doc comment for why
//! [`run_chunk_equivalence`] and this module's own [`run_specialisation_equivalence`]
//! can force a specific pipeline (GENERIC vs. the class [`classify_material`] would
//! pick) for the SAME material.
//!
//! **The GENERIC and specialised pipelines are NOT guaranteed byte-identical on the
//! same input, and this module does not require it.** Measured on the real AMD Radeon
//! (Vulkan) adapter this crate targets: removing the anisotropic/biaxial branches
//! changes the compiled kernel's register pressure and instruction scheduling enough
//! that a handful of stochastic-branch threshold comparisons (Fresnel reflect-vs-refract,
//! Russian roulette) round 1 ULP differently between the two pipelines, flipping which
//! discrete branch a small number of (pixel, sample) tuples take -- Diamond measured
//! 9/1024 pixels differing, isolated and large-delta per pixel, the signature of a
//! flipped branch rather than a bug. This is the SAME class of divergence this crate
//! already tolerates statistically between the CPU and GPU estimators (see this
//! module's header comment on the PDF-division structural rule that makes float
//! divergence harmless), so it is verified the same way:
//! [`run_specialisation_equivalence`] gates on GPU dispatch DETERMINISM (the same
//! specialised pipeline, dispatched twice, must be byte-identical) plus a diagnostic
//! diff count, while `estimator_check::run_specialisation_image_comparison` is the
//! rigorous GENERIC-vs-specialised correctness gate (Tier 3 statistical image
//! comparison, the SAME z-score/clustering criteria already used for CPU-vs-GPU).
//!
//! # What this does NOT do
//!
//! - **No guide buffers.** The megakernel returns radiance only, with no first-hit
//!   depth/normal/facet-id, so the A-Trous denoiser has nothing to key on. This is the
//!   same gap a remote worker's `FRAME` payload has, and it has the same answer:
//!   `apps/diagram-gui`'s `bridge::guide_pass` regenerates them locally from one
//!   un-jittered primary ray per pixel. That module was written for the remote path and
//!   is reused unchanged here.
//! - **Material routing is a contract, not a restriction today.** `GemMaterial::gpu_supported`
//!   is this crate's routing predicate and [`GpuFrameRenderer::accumulate`] enforces it
//!   ([`GpuFrameError::UnsupportedMaterial`]). Since 2026-09-02 it is unconditionally
//!   `true`: the genuinely biaxial stones (Alexandrite, Topaz, Tanzanite) that used to
//!   be CPU-only are ported (`hero_biaxial_wave_dirs`, the eigen-polarization and
//!   mode-Poynting machinery, verified at 0 ULP by the Tier 2/3 checks), so the decline
//!   path never fires for any built-in material. It stays because a future material
//!   kind the megakernel cannot handle would flip the predicate, and this module must
//!   keep agreeing with it (see `every_builtin_routes_the_way_gpu_supported_says`).
//! - **No HDR environment maps.** The megakernel's `env_mode` covers the uniform furnace
//!   and the analytic studio rig; `EnvironmentSource::HdrMap` has no GPU counterpart, so
//!   it is reported as [`GpuFrameError::UnsupportedEnvironment`] for the caller to fall
//!   back on rather than silently rendered under the wrong lighting.

use glam::Vec3;
use wgpu::BufferUsages;

use crate::{
    geometry::GpuFacetPlane,
    optics::{
        materials::{CrystalSystem, GemMaterial},
        raytracer::{
            Camera, EnvironmentSource, FacetFinish, compute_illuminant_white_balance,
            illuminant_temperature_k,
        },
    },
    renderer::{
        buffers::{
            GpuCameraParams, GpuGemMaterial, GpuTransportParams, encode_facet_finishes,
            transport_env_mode,
        },
        gpu::{GpuAcquireError, GpuContext, compute},
    },
};

/// `spectral_transport.wgsl` alone is not valid WGSL -- it assumes
/// `shaders/transport_physics.wgsl`'s functions are already in scope, and `build.rs`
/// concatenates the two. Shared with `estimator_check` so both compile the identical
/// source text.
pub(crate) const SHADER_SRC: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/spectral_transport.generated.wgsl"
));

/// Floats this module's production dispatch actually writes and reads back per (pixel,
/// sample) thread: XYZ only. `GpuFrameRenderer::accumulate`'s chunk budget is sized
/// against this, not the shader's full 27-floats-per-tuple write capacity -- see the
/// module doc comment's "Chunking" section for why the other 24 (the three debug arrays)
/// don't count against a production chunk's budget any more.
const FLOATS_PER_TUPLE: usize = 3;

/// R4: fixed, deliberately tiny float count backing each of a production
/// [`TransportOutputs`]'s three debug buffers (`radiance`/`lambdas`/`path_pdf`) --
/// `write_debug_buffers = 0` (see [`GpuTransportParams::write_debug_buffers`]) means the
/// shader's debug writes never execute for these dispatches, so these buffers only need
/// to exist to satisfy the megakernel's static bind-group layout, never to hold real
/// per-chunk data. One tuple's worth (8 floats per channel array) is an arbitrary safe
/// size that never needs to grow with chunk size.
const DEBUG_BUFFER_FLOATS: usize = 8;

/// Output-buffer budget for a single dispatch, in bytes.
///
/// 64 MiB is well inside what an integrated GPU will allocate without complaint, and
/// keeps a chunk short enough that a progressive frame stays responsive. It bounds only
/// how many chunks a frame takes, never what a frame can contain.
pub const CHUNK_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// Why a GPU frame could not be produced. Every variant is a condition the caller is
/// expected to handle by falling back to the CPU tracer -- none is a bug.
#[derive(Debug)]
pub enum GpuFrameError {
    /// No usable adapter or device on this machine. Expected on plenty of systems; see
    /// [`GpuContext::acquire`].
    Acquire(GpuAcquireError),
    /// The scene uses an environment the megakernel has no `env_mode` for -- today, an
    /// HDR environment map. See the module doc comment.
    UnsupportedEnvironment,
    /// The scene's material is one [`GemMaterial::gpu_supported`] rejects. Currently
    /// unreachable: that predicate is unconditionally `true` since the biaxial port
    /// (2026-09-02). Kept as defensive future-proofing -- `gpu_supported` is the crate's
    /// routing contract, and a future material kind the megakernel cannot handle would
    /// flip it. Carries the material's name, for a log line that says which stone.
    UnsupportedMaterial(String),
}

impl std::fmt::Display for GpuFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Acquire(e) => write!(f, "no usable GPU adapter or device: {e:?}"),
            Self::UnsupportedEnvironment => write!(
                f,
                "the GPU megakernel has no env_mode for this environment (HDR maps are CPU-only)"
            ),
            Self::UnsupportedMaterial(name) => write!(
                f,
                "{name} is not GPU-supported (GemMaterial::gpu_supported), so it renders on the CPU"
            ),
        }
    }
}

impl std::error::Error for GpuFrameError {}

/// Everything the megakernel needs to render a frame, bundled so
/// [`GpuFrameRenderer::accumulate`] stays within clippy's argument-count limit and so
/// the caller assembles the scene once rather than per chunk.
pub struct GpuFrameScene<'a> {
    pub camera: &'a Camera,
    pub width: u32,
    pub height: u32,
    pub planes: &'a [GpuFacetPlane],
    /// Per-plane finish, indexed in step with `planes`. A shorter slice is padded with
    /// [`FacetFinish::default`], matching [`encode_facet_finishes`].
    pub facet_finishes: &'a [FacetFinish],
    pub material: &'a GemMaterial,
    pub max_bounces: u32,
    pub environment: EnvironmentSource<'a>,
}

/// The four `transport_main` output buffers, sized for `capacity` (pixel, sample)
/// tuples.
///
/// Held across dispatches by [`GpuFrameRenderer`] and reallocated only when a frame
/// needs more capacity than the last one did -- the buffers are large (see the module
/// doc comment) and a progressive render re-dispatches the same geometry many times per
/// second, so recreating them every frame would dominate the frame's own cost.
pub(crate) struct TransportOutputs {
    xyz: wgpu::Buffer,
    radiance: wgpu::Buffer,
    lambdas: wgpu::Buffer,
    path_pdf: wgpu::Buffer,
    capacity: usize,
}

impl TransportOutputs {
    pub(crate) fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        Self {
            xyz: compute::zeroed_buffer::<f32>(device, "transport out xyz", capacity * 3, usage),
            radiance: compute::zeroed_buffer::<f32>(
                device,
                "transport out radiance",
                capacity * 8,
                usage,
            ),
            lambdas: compute::zeroed_buffer::<f32>(
                device,
                "transport out lambdas",
                capacity * 8,
                usage,
            ),
            path_pdf: compute::zeroed_buffer::<f32>(
                device,
                "transport out path_pdf",
                capacity * 8,
                usage,
            ),
            capacity,
        }
    }

    /// R4: like [`Self::new`], but for `renderer::gpu::frame`'s production dispatch
    /// only -- `xyz` is sized for `capacity` tuples (what
    /// [`GpuFrameRenderer::accumulate`] actually reads back), while the three debug
    /// buffers are fixed at [`DEBUG_BUFFER_FLOATS`] regardless of `capacity`, since a
    /// production dispatch's `GpuTransportParams::write_debug_buffers` is always off (see
    /// that field's doc comment) -- the shader never writes them, so they only need to
    /// satisfy the megakernel's bind-group layout, not hold real chunk data.
    pub(crate) fn new_production(device: &wgpu::Device, capacity: usize) -> Self {
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        Self {
            xyz: compute::zeroed_buffer::<f32>(
                device,
                "transport out xyz (production)",
                capacity * 3,
                usage,
            ),
            radiance: compute::zeroed_buffer::<f32>(
                device,
                "transport out radiance (production, write_debug_buffers=0, unused)",
                DEBUG_BUFFER_FLOATS,
                usage,
            ),
            lambdas: compute::zeroed_buffer::<f32>(
                device,
                "transport out lambdas (production, write_debug_buffers=0, unused)",
                DEBUG_BUFFER_FLOATS,
                usage,
            ),
            path_pdf: compute::zeroed_buffer::<f32>(
                device,
                "transport out path_pdf (production, write_debug_buffers=0, unused)",
                DEBUG_BUFFER_FLOATS,
                usage,
            ),
            capacity,
        }
    }

    pub(crate) const fn xyz(&self) -> &wgpu::Buffer {
        &self.xyz
    }

    pub(crate) const fn radiance(&self) -> &wgpu::Buffer {
        &self.radiance
    }

    pub(crate) const fn lambdas(&self) -> &wgpu::Buffer {
        &self.lambdas
    }

    pub(crate) const fn path_pdf(&self) -> &wgpu::Buffer {
        &self.path_pdf
    }
}

/// Bundles the material/scene/output-buffer inputs shared by [`build_bind_group`],
/// [`encode_and_dispatch`] and [`encode_dispatch_no_wait`] -- purely to keep each of
/// their argument counts within clippy's `too_many_arguments` limit, the same reason
/// `optics::raytracer::refraction::RayMaterialContext` bundles the per-trace fields
/// threaded through the CPU bounce loop. `total_tuples` deliberately stays its own
/// parameter on the two dispatch functions rather than joining this struct: it drives
/// the workgroup count and the output-capacity assert, not what gets bound, and differs
/// from `outputs.capacity` (this struct's `outputs` field) whenever a chunk is smaller
/// than its buffers' full capacity.
pub(crate) struct TransportDispatchArgs<'a> {
    pub(crate) ctx: &'a GpuContext,
    pub(crate) pipeline: &'a wgpu::ComputePipeline,
    pub(crate) camera_params: &'a GpuCameraParams,
    pub(crate) params: &'a GpuTransportParams,
    pub(crate) material: &'a GpuGemMaterial,
    pub(crate) planes: &'a [GpuFacetPlane],
    pub(crate) facet_finishes: &'a [u32],
    pub(crate) outputs: &'a TransportOutputs,
}

/// Uploads the scene inputs and binds all eight buffers, WITHOUT dispatching.
///
/// R4: factored out of what used to be `encode_and_dispatch`'s own body so
/// [`encode_and_dispatch`] (blocking, every self-test's path) and
/// [`encode_dispatch_no_wait`] (non-blocking, `GpuFrameRenderer::accumulate`'s pipelined
/// production path) share the exact same upload/bind code and differ only in which
/// `compute::dispatch*` primitive they call -- see [`compute::dispatch`]'s doc comment
/// for why the production path needs the non-blocking one. The upload buffers
/// (`camera_buf`/`params_buf`/... ) are local and dropped when this function returns,
/// which is safe even before the GPU has consumed them: `wgpu` keeps a resource alive
/// internally for as long as any submitted (but not yet completed) command buffer
/// references it, regardless of whether the Rust-side handle was dropped.
fn build_bind_group(args: &TransportDispatchArgs<'_>) -> wgpu::BindGroup {
    let camera_buf = compute::upload(
        &args.ctx.device,
        "transport camera",
        std::slice::from_ref(args.camera_params),
        BufferUsages::UNIFORM,
    );
    let params_buf = compute::upload(
        &args.ctx.device,
        "transport params",
        std::slice::from_ref(args.params),
        BufferUsages::UNIFORM,
    );
    let material_buf = compute::upload(
        &args.ctx.device,
        "transport material",
        std::slice::from_ref(args.material),
        BufferUsages::STORAGE,
    );
    let planes_buf = compute::upload(
        &args.ctx.device,
        "transport planes",
        args.planes,
        BufferUsages::STORAGE,
    );
    // A SEPARATE storage buffer, parallel to `planes_buf`, never merged into
    // `GpuFacetPlane` itself -- see `renderer::buffers::facet_finish`'s module doc
    // comment for why.
    let facet_finishes_buf = compute::upload(
        &args.ctx.device,
        "transport facet finishes",
        args.facet_finishes,
        BufferUsages::STORAGE,
    );

    compute::bind_buffers(
        &args.ctx.device,
        "transport bind group",
        args.pipeline,
        &[
            (0, &camera_buf),
            (1, &params_buf),
            (2, &material_buf),
            (3, &planes_buf),
            (4, &args.outputs.xyz),
            (5, &args.outputs.radiance),
            (6, &args.outputs.lambdas),
            (7, &args.outputs.path_pdf),
            (8, &facet_finishes_buf),
        ],
    )
}

/// Uploads the scene inputs, binds all eight buffers, and dispatches `transport_main`
/// over `total_tuples` threads -- then BLOCKS until it finishes.
///
/// The single blocking dispatch routine for the megakernel:
/// `estimator_check::dispatch_transport` and every other self-test in `renderer::gpu` go
/// through here, so the Tier 2/Tier 3 equivalence checks verify the exact binding code
/// the renderer ships (see [`build_bind_group`]). `GpuFrameRenderer::accumulate`'s
/// production dispatch uses [`encode_dispatch_no_wait`] instead -- see that function's
/// doc comment for why.
pub(crate) fn encode_and_dispatch(args: &TransportDispatchArgs<'_>, total_tuples: usize) {
    assert!(
        total_tuples <= args.outputs.capacity,
        "dispatch of {total_tuples} tuples exceeds output capacity {}",
        args.outputs.capacity
    );
    let bind_group = build_bind_group(args);
    let workgroups = (total_tuples as u32).div_ceil(64);
    compute::dispatch_and_wait(
        &args.ctx.device,
        &args.ctx.queue,
        args.pipeline,
        &bind_group,
        (workgroups, 1, 1),
    );
}

/// R4: like [`encode_and_dispatch`], but submits the dispatch WITHOUT blocking on it --
/// `GpuFrameRenderer::accumulate`'s pipelined production path uses this so a following
/// chunk's dispatch (and its own readback copy) can be queued before the CPU blocks on
/// any one chunk's result. See the module doc comment's "Overlapped chunk pipeline"
/// section.
fn encode_dispatch_no_wait(args: &TransportDispatchArgs<'_>, total_tuples: usize) {
    assert!(
        total_tuples <= args.outputs.capacity,
        "dispatch of {total_tuples} tuples exceeds output capacity {}",
        args.outputs.capacity
    );
    let bind_group = build_bind_group(args);
    let workgroups = (total_tuples as u32).div_ceil(64);
    // The compute dispatch's own submission index is unused: waiting for the readback
    // copy's index (submitted after this one, on the same queue) is sufficient -- see
    // `compute::finish_map_read`'s doc comment.
    let _ = compute::dispatch(
        &args.ctx.device,
        &args.ctx.queue,
        args.pipeline,
        &bind_group,
        (workgroups, 1, 1),
    );
}

/// The values `spectral_transport.wgsl`'s `MATERIAL_CLASS` pipeline-overridable
/// constant accepts -- see that override's own doc comment. `pub(crate)` rather than
/// private: `estimator_check::dispatch_transport_for_class` (Tier 3 statistical image
/// comparisons) also needs these to dispatch through the same specialised pipelines
/// [`GpuFrameRenderer::accumulate`] does -- see the module doc comment's "Material-class
/// kernel specialisation" section.
pub(crate) mod material_class {
    /// Every class, runtime-dispatched inside the kernel exactly as before this task --
    /// the override's own declared default, so a dispatch that never sets it (every
    /// self-test) is unaffected.
    pub const GENERIC: u32 = 0;
    pub const ISOTROPIC: u32 = 1;
    pub const UNIAXIAL: u32 = 2;
    pub const BIAXIAL: u32 = 3;
}

/// Which of [`material_class`]'s values a real render of `material` should dispatch
/// through.
///
/// MIRRORS `renderer::buffers::GpuGemMaterial::encode`'s own `is_anisotropic`/
/// `has_biaxial_delta` derivation exactly -- never a second, independently-maintained
/// definition of "is this material anisotropic/biaxial": biaxial takes priority
/// (`biaxial_delta_beta_alpha.is_some()`), then uniaxial (`crystal_system != Cubic &&
/// |birefringence_delta| > 1e-4`, matching the kernel's own `is_anisotropic` exactly),
/// else isotropic. Kept in lock-step deliberately: if `encode`'s formula ever changes,
/// this must change with it, or [`GpuFrameRenderer::accumulate`] would pick a
/// specialised pipeline whose `is_anisotropic`/`is_biaxial` derivation (see
/// `spectral_transport.wgsl`) forces off state the material's OWN encoded flags say it
/// needs -- silently wrong output, not a crash.
/// `estimator_check::run_specialisation_image_comparison` (a statistical, not bit-exact,
/// comparison -- see this module's "Material-class kernel specialisation" doc section
/// for why) is the check that would catch such a drift: a wrong class forces off state
/// the material genuinely needs, which biases the whole image, not just a handful of
/// isolated pixels.
#[must_use]
pub(crate) fn classify_material(material: &GemMaterial) -> u32 {
    if material.biaxial_delta_beta_alpha.is_some() {
        material_class::BIAXIAL
    } else if material.crystal_system != CrystalSystem::Cubic
        && material.birefringence_delta.abs() > 1e-4
    {
        material_class::UNIAXIAL
    } else {
        material_class::ISOTROPIC
    }
}

/// A GPU-backed renderer for arbitrary scenes, owning its device and pipeline across
/// frames.
///
/// Construct once and keep it: [`GpuFrameRenderer::new`] acquires an adapter and
/// compiles the megakernel, both of which take long enough to be worth doing off the
/// frame loop.
pub struct GpuFrameRenderer {
    ctx: GpuContext,
    /// The GENERIC (`MATERIAL_CLASS = 0`) pipeline -- built eagerly here since every
    /// self-test in `renderer::gpu` dispatches it, and `run_chunk_equivalence`/
    /// `run_specialisation_equivalence` need it available without any prior
    /// `accumulate` call. See the module doc comment's "Material-class kernel
    /// specialisation" section.
    pipeline: wgpu::ComputePipeline,
    /// The three per-class specialised pipelines, built LAZILY (see
    /// [`Self::ensure_specialized_pipeline`]) on first use of that class -- `None` until
    /// a scene of that class is actually dispatched, so a session that only ever
    /// renders (say) uniaxial stones never pays isotropic/biaxial shader-compile cost.
    pipeline_isotropic: Option<wgpu::ComputePipeline>,
    pipeline_uniaxial: Option<wgpu::ComputePipeline>,
    pipeline_biaxial: Option<wgpu::ComputePipeline>,
    adapter_label: String,
    /// R4: TWO chunk-output buffer sets, alternated by chunk index so one chunk's
    /// dispatch can be queued into the other slot while the previous chunk's readback is
    /// still in flight -- see the module doc comment's "Overlapped chunk pipeline"
    /// section.
    outputs: [Option<TransportOutputs>; 2],
    chunk_budget_bytes: usize,
}

/// One in-flight chunk's readback state, between "dispatch and copy submitted" and
/// "mapped, read, and summed into `accum`" -- see [`GpuFrameRenderer::accumulate`]'s
/// pipeline.
struct PendingChunk {
    staging: wgpu::Buffer,
    copy_index: wgpu::SubmissionIndex,
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    first_pixel: usize,
    pixels_this_chunk: usize,
}

impl GpuFrameRenderer {
    /// Acquires a GPU device and compiles `transport_main`.
    ///
    /// # Errors
    ///
    /// [`GpuFrameError::Acquire`] if this machine has no usable adapter or device --
    /// an expected outcome, not a bug; the caller should fall back to the CPU tracer.
    pub fn new() -> Result<Self, GpuFrameError> {
        let ctx = GpuContext::acquire().map_err(GpuFrameError::Acquire)?;
        let info = ctx.adapter.get_info();
        let adapter_label = format!("{} ({:?})", info.name, info.backend);
        let pipeline = compute::create_compute_pipeline(
            &ctx.device,
            "transport_main",
            SHADER_SRC,
            "transport_main",
        );
        Ok(Self {
            ctx,
            pipeline,
            pipeline_isotropic: None,
            pipeline_uniaxial: None,
            pipeline_biaxial: None,
            adapter_label,
            outputs: [None, None],
            chunk_budget_bytes: CHUNK_BUDGET_BYTES,
        })
    }

    /// Overrides the per-dispatch output-buffer budget (default [`CHUNK_BUDGET_BYTES`]).
    ///
    /// Exists so a check can force a frame through many small chunks and confirm the
    /// result is identical to the single-chunk one -- see [`run_chunk_equivalence`].
    /// Lowering it trades more dispatches for less peak VRAM; it never changes what a
    /// frame renders.
    pub const fn set_chunk_budget_bytes(&mut self, bytes: usize) {
        self.chunk_budget_bytes = bytes;
    }

    /// Human-readable adapter name and backend, for logging and for telling the user
    /// which device is actually rendering.
    #[must_use]
    pub fn adapter_label(&self) -> &str {
        &self.adapter_label
    }

    /// Traces `spp` samples per pixel, starting at sample index `sample_offset`, and
    /// ADDS each pixel's summed XYZ into `accum`.
    ///
    /// Accumulating rather than overwriting mirrors the CPU path's
    /// `acc_chunk[i] += sample_sum`, so a caller's progressive-accumulation buffer and
    /// its sample counter mean exactly what they meant before, whichever backend
    /// produced the samples. `sample_offset` must be the number of samples already in
    /// `accum` for these pixels: the shader derives each thread's seed and its
    /// stratified jitter from the absolute sample index, so reusing an offset would
    /// re-draw identical samples and bias the average.
    ///
    /// # Errors
    ///
    /// [`GpuFrameError::UnsupportedEnvironment`] if the scene uses an HDR environment
    /// map, which the megakernel has no `env_mode` for.
    ///
    /// # Panics
    ///
    /// Panics if `accum.len()` is not `width * height`.
    pub fn accumulate(
        &mut self,
        scene: &GpuFrameScene<'_>,
        sample_offset: u32,
        spp: u32,
        accum: &mut [Vec3],
    ) -> Result<(), GpuFrameError> {
        // Kernel specialisation: [`classify_material`] is the ONE place this decision is
        // made -- see the module doc comment's "Material-class kernel specialisation"
        // section. `accumulate_via_pipeline` does everything this function used to do,
        // parameterised on which pipeline to dispatch through, so
        // [`run_specialisation_equivalence`] can call it directly with a FORCED class
        // (bypassing this classification) and compare against what this wrapper picks.
        let pipeline_class = classify_material(scene.material);
        self.accumulate_via_pipeline(scene, pipeline_class, sample_offset, spp, accum)
    }

    /// The body [`Self::accumulate`] delegates to, parameterised on `pipeline_class`
    /// (one of [`material_class`]'s values) rather than deriving it from
    /// `scene.material` itself -- see [`Self::accumulate`]'s doc comment for why a
    /// caller-supplied class matters (it lets [`run_specialisation_equivalence`] force
    /// the GENERIC pipeline for a material [`classify_material`] would otherwise route
    /// to a specialised one, and vice versa, which is exactly what proves the two
    /// pipelines agree).
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::accumulate`]'s.
    ///
    /// # Panics
    ///
    /// Same conditions as [`Self::accumulate`]'s.
    fn accumulate_via_pipeline(
        &mut self,
        scene: &GpuFrameScene<'_>,
        pipeline_class: u32,
        sample_offset: u32,
        spp: u32,
        accum: &mut [Vec3],
    ) -> Result<(), GpuFrameError> {
        let num_pixels = scene.width as usize * scene.height as usize;
        assert_eq!(
            accum.len(),
            num_pixels,
            "accumulation buffer must have one entry per pixel"
        );
        if spp == 0 || num_pixels == 0 {
            return Ok(());
        }

        // `GemMaterial::gpu_supported` is the crate's routing predicate, and its doc
        // comment requires a caller assembling a full scene to consult it before routing
        // that scene to the GPU. Currently this branch is unreachable -- the predicate is
        // unconditionally `true` since the biaxial port (2026-09-02) -- but it stays as
        // the enforcement of that contract: a future material kind the megakernel cannot
        // handle would flip the predicate, and routing it anyway would produce a
        // plausible-looking but wrong image rather than an error.
        if !scene.material.gpu_supported() {
            return Err(GpuFrameError::UnsupportedMaterial(
                scene.material.name.clone(),
            ));
        }

        let (env_mode, temp_k, spot_mult, exposure, light_yaw, light_pitch) =
            environment_params(scene.environment)?;
        let white_balance = compute_illuminant_white_balance(temp_k);

        let gpu_material = GpuGemMaterial::encode(scene.material);
        let gpu_finishes = encode_facet_finishes(scene.facet_finishes, scene.planes.len());

        self.ensure_specialized_pipeline(pipeline_class);

        let chunk_pixels = chunk_pixels_for(self.chunk_budget_bytes, spp, num_pixels);
        self.ensure_capacity(chunk_pixels * spp as usize);

        // R4: one chunk's dispatch+copy is submitted, then (if a PREVIOUS chunk is still
        // pending) that older chunk is mapped/read/summed -- so by the time the CPU
        // blocks on chunk i's result, chunk i+1's GPU work is already queued behind it.
        // See the module doc comment's "Overlapped chunk pipeline" section for why
        // reusing `outputs[chunk_index % 2]` two chunks later is safe.
        let mut pending: Option<PendingChunk> = None;
        let mut first_pixel = 0usize;
        let mut chunk_index = 0usize;
        while first_pixel < num_pixels {
            let pixels_this_chunk = chunk_pixels.min(num_pixels - first_pixel);
            let tuples = pixels_this_chunk * spp as usize;

            // `num_samples` is per-pixel within this dispatch; `width`/`height` stay the
            // FULL frame's, since the shader turns a global pixel index into (x, y) with
            // them -- a chunk is a range of pixels, not a smaller image.
            let camera_params = GpuCameraParams {
                origin: scene.camera.origin.to_array(),
                fov_tan: scene.camera.fov_tan,
                forward: scene.camera.forward.to_array(),
                width: scene.width as f32,
                right: scene.camera.right.to_array(),
                height: scene.height as f32,
                up: scene.camera.up.to_array(),
                num_samples: spp,
            };
            let params = GpuTransportParams::new(
                pixels_this_chunk as u32,
                scene.max_bounces,
                sample_offset,
                env_mode,
                0.0,
                temp_k,
                spot_mult,
                exposure,
                light_yaw,
                light_pitch,
                white_balance.to_array(),
            )
            .with_pixel_offset(first_pixel as u32)
            .with_debug_buffers_disabled();

            let outputs = self.outputs[chunk_index % 2]
                .as_ref()
                .expect("ensure_capacity just populated both slots");

            encode_dispatch_no_wait(
                &TransportDispatchArgs {
                    ctx: &self.ctx,
                    pipeline: self.pipeline_for_class(pipeline_class),
                    camera_params: &camera_params,
                    params: &params,
                    material: &gpu_material,
                    planes: scene.planes,
                    facet_finishes: &gpu_finishes,
                    outputs,
                },
                tuples,
            );
            let (staging, copy_index) = compute::copy_to_staging::<f32>(
                &self.ctx.device,
                &self.ctx.queue,
                outputs.xyz(),
                tuples * 3,
                "transport out xyz staging (pipelined production)",
            );
            let rx = compute::begin_map_read(&staging);

            if let Some(prev) = pending.take() {
                self.drain_pending_chunk(prev, spp, accum);
            }
            pending = Some(PendingChunk {
                staging,
                copy_index,
                rx,
                first_pixel,
                pixels_this_chunk,
            });

            first_pixel += pixels_this_chunk;
            chunk_index += 1;
        }
        if let Some(last) = pending.take() {
            self.drain_pending_chunk(last, spp, accum);
        }

        Ok(())
    }

    /// Blocks until `chunk`'s readback copy has completed, reads its XYZ, and sums each
    /// pixel's `spp` samples into `accum` -- the second half of the pipeline
    /// [`Self::accumulate`] builds; split out purely so that function's dispatch loop
    /// stays readable.
    fn drain_pending_chunk(&self, chunk: PendingChunk, spp: u32, accum: &mut [Vec3]) {
        let xyz: Vec<f32> = compute::finish_map_read(
            &self.ctx.device,
            &chunk.staging,
            chunk.copy_index,
            &chunk.rx,
        );
        for local_pixel in 0..chunk.pixels_this_chunk {
            let mut sum = Vec3::ZERO;
            for s in 0..spp as usize {
                let base = (local_pixel * spp as usize + s) * 3;
                sum += Vec3::new(xyz[base], xyz[base + 1], xyz[base + 2]);
            }
            accum[chunk.first_pixel + local_pixel] += sum;
        }
    }

    /// Builds and caches the specialised pipeline for `class`, if it isn't already --
    /// see the module doc comment's "Material-class kernel specialisation" section for
    /// why this is lazy rather than done once in [`Self::new`]. A no-op for
    /// [`material_class::GENERIC`], which [`Self::new`] already built eagerly.
    fn ensure_specialized_pipeline(&mut self, class: u32) {
        let (slot, label) = match class {
            material_class::ISOTROPIC => (
                &mut self.pipeline_isotropic,
                "transport_main (MATERIAL_CLASS=isotropic)",
            ),
            material_class::UNIAXIAL => (
                &mut self.pipeline_uniaxial,
                "transport_main (MATERIAL_CLASS=uniaxial)",
            ),
            material_class::BIAXIAL => (
                &mut self.pipeline_biaxial,
                "transport_main (MATERIAL_CLASS=biaxial)",
            ),
            // GENERIC (and any other value -- there is no other caller-reachable one)
            // has nothing to build: `Self::new` already compiled `self.pipeline`.
            _ => return,
        };
        if slot.is_none() {
            *slot = Some(compute::create_compute_pipeline_with_constants(
                &self.ctx.device,
                label,
                SHADER_SRC,
                "transport_main",
                &[("MATERIAL_CLASS", f64::from(class))],
            ));
        }
    }

    /// Returns the already-built pipeline for `class` -- [`Self::ensure_specialized_pipeline`]
    /// must have been called for this exact `class` first (every caller in this module
    /// does so immediately before dispatching).
    ///
    /// # Panics
    ///
    /// Panics if `class` is a specialised value whose pipeline was never built via
    /// [`Self::ensure_specialized_pipeline`] -- a bug in this module's own call
    /// ordering, never a condition a caller outside it can trigger.
    const fn pipeline_for_class(&self, class: u32) -> &wgpu::ComputePipeline {
        match class {
            material_class::ISOTROPIC => self.pipeline_isotropic.as_ref(),
            material_class::UNIAXIAL => self.pipeline_uniaxial.as_ref(),
            material_class::BIAXIAL => self.pipeline_biaxial.as_ref(),
            _ => Some(&self.pipeline),
        }
        .expect("ensure_specialized_pipeline must be called for this class before dispatching")
    }

    /// Grows the cached output buffers to hold at least `tuples`, reusing them when they
    /// are already large enough (the common case: a progressive render re-dispatches the
    /// same resolution until the camera moves). Grows BOTH double-buffered slots (see
    /// [`Self::outputs`]'s doc comment) identically, since a chunk can land in either.
    fn ensure_capacity(&mut self, tuples: usize) {
        for slot in &mut self.outputs {
            let big_enough = slot.as_ref().is_some_and(|o| o.capacity >= tuples);
            if !big_enough {
                *slot = Some(TransportOutputs::new_production(&self.ctx.device, tuples));
            }
        }
    }
}

/// `transport_main`'s declared `@workgroup_size(64)` -- a dispatch of `n` workgroups
/// covers `n * WORKGROUP_SIZE` (pixel, sample) tuples.
const WORKGROUP_SIZE: usize = 64;

/// The lowest cross-backend guarantee for `maxComputeWorkgroupsPerDimension`
/// (WebGPU/Vulkan/D3D12's shared floor) -- `dispatch_workgroups(x, 1, 1)` is only valid
/// for `x <=` this. `transport_main` indexes purely off `global_invocation_id.x` (see its
/// own doc comment), so every dispatch in this module is one-dimensional and this bounds
/// how many tuples ONE dispatch may cover, independent of [`CHUNK_BUDGET_BYTES`].
///
/// R4 found this the hard way: raising a chunk's byte budget 9x (see the module doc
/// comment) meant an 800x600 frame at 32 spp needed only one ~5.6M-tuple chunk, whose
/// `87381` implied workgroups tripped `wgpu`'s dispatch validation
/// (`Each current dispatch group size dimension (...) must be less or equal to 65535`)
/// -- a real error the R4 benchmark hit, not a theoretical concern. [`chunk_pixels_for`]
/// caps against this in addition to the byte budget so a chunk is never too large to
/// dispatch, regardless of how generous [`CHUNK_BUDGET_BYTES`] or a caller's override is.
const MAX_WORKGROUPS_PER_DIMENSION: usize = 65_535;

/// How many pixels one dispatch covers, given a byte budget for its output buffers.
///
/// A pixel's `spp` samples always stay in one dispatch (see the module doc comment), so
/// the budget is divided by `spp` first. Always at least 1 -- a single pixel over budget
/// is still dispatched rather than looping forever on a zero-width chunk -- and never
/// more than the frame has. Also never large enough to need more than
/// [`MAX_WORKGROUPS_PER_DIMENSION`] workgroups -- see that constant's doc comment.
fn chunk_pixels_for(budget_bytes: usize, spp: u32, num_pixels: usize) -> usize {
    let budget_tuples = budget_bytes / (FLOATS_PER_TUPLE * size_of::<f32>());
    let dispatch_limited_tuples = MAX_WORKGROUPS_PER_DIMENSION * WORKGROUP_SIZE;
    let tuples_per_chunk = budget_tuples.min(dispatch_limited_tuples);
    (tuples_per_chunk / spp as usize).max(1).min(num_pixels)
}

/// Maps an [`EnvironmentSource`] onto the megakernel's `env_mode` and its studio-rig
/// parameters.
///
/// Returns `(env_mode, temp_k, spot_mult, exposure, light_yaw, light_pitch)`.
const fn environment_params(
    environment: EnvironmentSource<'_>,
) -> Result<(u32, f32, f32, f32, f32, f32), GpuFrameError> {
    match environment {
        EnvironmentSource::Studio {
            preset,
            exposure,
            light_yaw,
            light_pitch,
        } => Ok((
            transport_env_mode::STUDIO_RIG,
            illuminant_temperature_k(preset),
            preset.params().spot_mult,
            exposure,
            light_yaw,
            light_pitch,
        )),
        EnvironmentSource::HdrMap(_) => Err(GpuFrameError::UnsupportedEnvironment),
    }
}

/// Result of [`run_chunk_equivalence`].
pub struct ChunkEquivalenceResult {
    /// How many chunks the deliberately-small budget forced. A run where this is 1
    /// proves nothing, so [`Self::passed`] requires more.
    pub chunks_forced: usize,
    /// Pixels whose accumulated XYZ differed between the two runs, in raw bits.
    pub differing_pixels: usize,
    pub total_pixels: usize,
    /// Largest absolute component difference seen, for a failure message that says how
    /// far off it was rather than only that it differed.
    pub max_abs_diff: f32,
}

impl ChunkEquivalenceResult {
    /// Bit-exact equality, over a run that actually chunked.
    ///
    /// Exact rather than tolerant on purpose: chunking must be a pure partition of the
    /// same threads, since each thread's output depends only on its own
    /// `(pixel, sample_num)` and nothing else. Any difference at all means
    /// `pixel_offset` is not reconstructing the global pixel index correctly, and a
    /// tolerance would hide exactly the off-by-one that would produce.
    #[must_use]
    pub const fn passed(&self) -> bool {
        self.chunks_forced > 1 && self.differing_pixels == 0
    }
}

/// Renders one frame twice -- once in a single dispatch, once forced into many small
/// chunks -- and requires the two to be bit-identical.
///
/// This is the check for `GpuTransportParams::pixel_offset`, the one piece of shader
/// logic this module added: every other GPU self-test dispatches a whole frame at once
/// and so runs with `pixel_offset == 0`, leaving the chunked path unexercised. An
/// off-by-one there would misplace camera rays and per-pixel jitter rotations by a chunk
/// boundary -- visible as a seam, but only at resolutions large enough to chunk, which is
/// exactly where nothing else was looking.
///
/// # Panics
///
/// Panics if the scene's material is not GPU-supported or its environment is
/// unsupported -- both are fixed here (a cubic stone under the studio rig), so either
/// would be a bug in this function, not a runtime condition.
#[must_use]
pub fn run_chunk_equivalence(renderer: &mut GpuFrameRenderer) -> ChunkEquivalenceResult {
    use crate::{geometry::cuts::StandardGemCuts, optics::raytracer::LightingPreset};

    let camera = Camera::new(0.35, 0.28, 5.0, 18.0);
    let planes = StandardGemCuts::standard_round_brilliant();
    let material = GemMaterial::by_name("Spinel").expect("Spinel is a built-in cubic material");
    let (width, height) = (64u32, 64u32);
    let spp = 2u32;
    let num_pixels = (width * height) as usize;

    let scene = GpuFrameScene {
        camera: &camera,
        width,
        height,
        planes: &planes,
        facet_finishes: &[],
        material: &material,
        max_bounces: 8,
        environment: LightingPreset::Daylight.studio(1.0, 0.4, 0.35),
    };

    let mut whole = vec![Vec3::ZERO; num_pixels];
    renderer.set_chunk_budget_bytes(CHUNK_BUDGET_BYTES);
    renderer
        .accumulate(&scene, 0, spp, &mut whole)
        .expect("a cubic material under the studio rig is GPU-supported");

    // Small enough to force several chunks, and deliberately NOT a divisor of the pixel
    // count, so the last chunk is short and a boundary lands mid-row.
    let budget = 700 * FLOATS_PER_TUPLE * size_of::<f32>();
    let chunk_pixels = chunk_pixels_for(budget, spp, num_pixels);
    let chunks_forced = num_pixels.div_ceil(chunk_pixels);

    let mut chunked = vec![Vec3::ZERO; num_pixels];
    renderer.set_chunk_budget_bytes(budget);
    renderer
        .accumulate(&scene, 0, spp, &mut chunked)
        .expect("a cubic material under the studio rig is GPU-supported");
    renderer.set_chunk_budget_bytes(CHUNK_BUDGET_BYTES);

    let mut differing_pixels = 0usize;
    let mut max_abs_diff = 0.0f32;
    for (a, b) in whole.iter().zip(&chunked) {
        if a.to_array()
            .iter()
            .zip(b.to_array().iter())
            .any(|(x, y)| x.to_bits() != y.to_bits())
        {
            differing_pixels += 1;
            max_abs_diff = max_abs_diff.max((*a - *b).abs().max_element());
        }
    }

    ChunkEquivalenceResult {
        chunks_forced,
        differing_pixels,
        total_pixels: num_pixels,
        max_abs_diff,
    }
}

/// One material's result within [`SpecialisationEquivalenceResult`].
///
/// # Why this is self-determinism, not GENERIC-vs-specialised bit-identity
///
/// An earlier version of this check required the GENERIC and specialised pipelines to
/// produce byte-identical XYZ for the same input, on the theory that forcing
/// `is_anisotropic`/`is_biaxial` to a value that already matches the material's own
/// runtime flags (see `spectral_transport.wgsl`'s `MATERIAL_CLASS` doc comment) could
/// not change what is computed. Measured on the real AMD Radeon (Vulkan) adapter this
/// crate targets, that is FALSE for the isotropic pipeline: dead-code elimination
/// removing the anisotropic/biaxial branches changes the compiled kernel's register
/// pressure and instruction scheduling enough that a handful of stochastic-branch
/// threshold comparisons (Fresnel reflect-vs-refract, Russian roulette) round 1 ULP
/// differently -- Diamond measured 9/1024 pixels differing, isolated and large-delta
/// per pixel, the exact signature of a flipped discrete branch rather than diffuse
/// drift (confirmed against the per-channel debug buffers: the differing tuples show a
/// different reflect/refract/TIR history, not a uniformly nudged radiance). Zircon and
/// Alexandrite measured bit-identical, but nothing guarantees a future driver keeps
/// that -- so this check does not rely on it either.
///
/// This is EXACTLY the same class of divergence this crate already tolerates
/// statistically between the CPU and GPU estimators (see this module's own header
/// comment on "the structural rule that makes float divergence harmless": every
/// stochastic decision divides by its own locally-recomputed probability, so whichever
/// branch a 1-ULP-different comparison takes, the estimator stays unbiased). The
/// rigorous GENERIC-vs-specialised correctness gate is therefore
/// `estimator_check::run_specialisation_image_comparison` (Tier 3 statistical image
/// comparison, the SAME z-score/clustering criteria already used for CPU-vs-GPU), not
/// this function.
///
/// What GPU dispatch determinism DOES guarantee -- and what this struct's
/// [`Self::passed`] actually gates on -- is that the SAME compiled pipeline, dispatched
/// twice against identical input, produces byte-identical output (see
/// `determinism_check`'s own doc comment): no thread ever reads another thread's
/// output, so scheduling order can never matter WITHIN one pipeline. A specialised
/// pipeline failing that would mean the dead-code elimination this task relies on left
/// behind something genuinely broken (e.g. a read of a variable a naga/driver bug
/// failed to keep initialized) rather than merely a differently-scheduled but still
/// internally-consistent kernel.
#[derive(Debug, Clone)]
pub struct SpecialisationCaseResult {
    pub material_name: String,
    /// Which [`material_class`] value [`classify_material`] picked for this material --
    /// the specialised pipeline actually under test.
    pub material_class: u32,
    /// Pixels that differed between two dispatches of the SAME specialised pipeline
    /// against identical input -- see this struct's own doc comment. Must be 0 for
    /// [`Self::passed`].
    pub self_determinism_differing_pixels: usize,
    /// Diagnostic only, NOT part of [`Self::passed`]: how many pixels differed between
    /// the GENERIC pipeline and the specialised one, and the largest per-component
    /// difference seen -- reported so a reader can see the magnitude this module's own
    /// doc comment describes (a small minority of pixels, each differing by a
    /// branch-flip-sized amount) rather than taking "expected to sometimes differ" on
    /// faith. The rigorous pass/fail gate for this comparison is
    /// `estimator_check::run_specialisation_image_comparison`.
    pub generic_vs_specialised_differing_pixels: usize,
    pub total_pixels: usize,
    pub generic_vs_specialised_max_abs_diff: f32,
}

impl SpecialisationCaseResult {
    #[must_use]
    pub const fn passed(&self) -> bool {
        self.self_determinism_differing_pixels == 0
    }
}

/// Result of [`run_specialisation_equivalence`].
pub struct SpecialisationEquivalenceResult {
    pub cases: Vec<SpecialisationCaseResult>,
}

impl SpecialisationEquivalenceResult {
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.cases.is_empty() && self.cases.iter().all(SpecialisationCaseResult::passed)
    }
}

/// For each of one representative isotropic, uniaxial, and biaxial built-in material,
/// checks GPU dispatch determinism and records a diagnostic diff count.
///
/// Dispatches that material's specialised pipeline TWICE against identical input (must
/// be byte-identical -- GPU dispatch determinism, see [`SpecialisationCaseResult`]'s own
/// doc comment for why that is the right invariant here, not GENERIC-vs-specialised
/// bit-identity), and additionally records how many pixels differ against a GENERIC
/// dispatch of the same input, purely as a diagnostic.
///
/// Diamond stands in for isotropic (cubic, no birefringence), Zircon for uniaxial (the
/// largest built-in birefringence), and Alexandrite for biaxial (a populated `beta_ray`
/// band set, so the biaxial-only pleochroic absorption path is exercised too, not just
/// the index/direction machinery).
///
/// Uses [`GpuFrameRenderer::accumulate_via_pipeline`] (through the `pipeline_class`
/// argument) rather than a separate, ad hoc dispatch routine, so this check exercises
/// the SAME chunking/bind/dispatch code [`GpuFrameRenderer::accumulate`] ships,
/// differing only in which pipeline is forced -- see the module doc comment's
/// "Material-class kernel specialisation" section.
///
/// # Panics
///
/// Panics if `"Diamond"`, `"Zircon"`, or `"Alexandrite"` is ever removed from
/// [`GemMaterial::all_materials`] -- self-test scaffolding, not a code path a real
/// caller can reach with a name that might legitimately be missing.
#[must_use]
pub fn run_specialisation_equivalence(
    renderer: &mut GpuFrameRenderer,
) -> SpecialisationEquivalenceResult {
    use crate::{geometry::cuts::StandardGemCuts, optics::raytracer::LightingPreset};

    let camera = Camera::new(0.35, 0.28, 5.0, 18.0);
    let planes = StandardGemCuts::standard_round_brilliant();
    let (width, height) = (32u32, 32u32);
    let spp = 3u32;
    let num_pixels = (width * height) as usize;
    let environment = LightingPreset::Daylight.studio(1.0, 0.4, 0.35);

    let representative_materials = ["Diamond", "Zircon", "Alexandrite"];

    let mut cases = Vec::with_capacity(representative_materials.len());
    for name in representative_materials {
        let material = GemMaterial::by_name(name).unwrap_or_else(|| {
            panic!("{name:?} is a built-in material in GemMaterial::all_materials()")
        });
        let scene = GpuFrameScene {
            camera: &camera,
            width,
            height,
            planes: &planes,
            facet_finishes: &[],
            material: &material,
            max_bounces: 8,
            environment,
        };
        let class = classify_material(&material);

        // Self-determinism: the SAME specialised pipeline, dispatched twice against
        // identical input (sample_offset 0 both times) -- must be byte-identical.
        let mut run1 = vec![Vec3::ZERO; num_pixels];
        renderer
            .accumulate_via_pipeline(&scene, class, 0, spp, &mut run1)
            .expect("every representative material here is GPU-supported under the studio rig");
        let mut run2 = vec![Vec3::ZERO; num_pixels];
        renderer
            .accumulate_via_pipeline(&scene, class, 0, spp, &mut run2)
            .expect("every representative material here is GPU-supported under the studio rig");
        let self_determinism_differing_pixels = run1
            .iter()
            .zip(&run2)
            .filter(|(a, b)| {
                a.to_array()
                    .iter()
                    .zip(b.to_array().iter())
                    .any(|(x, y)| x.to_bits() != y.to_bits())
            })
            .count();

        // Diagnostic only: GENERIC vs specialised, same input -- see
        // `SpecialisationCaseResult`'s doc comment for why this is NOT required to be
        // zero.
        let mut generic = vec![Vec3::ZERO; num_pixels];
        renderer
            .accumulate_via_pipeline(&scene, material_class::GENERIC, 0, spp, &mut generic)
            .expect("every representative material here is GPU-supported under the studio rig");
        let mut generic_vs_specialised_differing_pixels = 0usize;
        let mut generic_vs_specialised_max_abs_diff = 0.0f32;
        for (a, b) in generic.iter().zip(&run1) {
            if a.to_array()
                .iter()
                .zip(b.to_array().iter())
                .any(|(x, y)| x.to_bits() != y.to_bits())
            {
                generic_vs_specialised_differing_pixels += 1;
                generic_vs_specialised_max_abs_diff =
                    generic_vs_specialised_max_abs_diff.max((*a - *b).abs().max_element());
            }
        }

        cases.push(SpecialisationCaseResult {
            material_name: name.to_string(),
            material_class: class,
            self_determinism_differing_pixels,
            generic_vs_specialised_differing_pixels,
            total_pixels: num_pixels,
            generic_vs_specialised_max_abs_diff,
        });
    }

    SpecialisationEquivalenceResult { cases }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics::raytracer::LightingPreset;

    #[test]
    fn chunking_divides_the_budget_by_samples_per_pixel() {
        // R4: 12 bytes per tuple (XYZ only -- see `FLOATS_PER_TUPLE`'s doc comment), so a
        // 120-byte budget is exactly 10 tuples: 10 pixels at 1 spp, 5 at 2 spp, 3 at 3 spp
        // (integer division, never rounding up past the budget).
        let budget = 10 * FLOATS_PER_TUPLE * size_of::<f32>();
        assert_eq!(chunk_pixels_for(budget, 1, 10_000), 10);
        assert_eq!(chunk_pixels_for(budget, 2, 10_000), 5);
        assert_eq!(chunk_pixels_for(budget, 3, 10_000), 3);
    }

    #[test]
    fn a_frame_smaller_than_the_budget_is_one_chunk() {
        let pixels = 800 * 600;
        assert_eq!(chunk_pixels_for(CHUNK_BUDGET_BYTES, 1, pixels), pixels);
    }

    /// A budget too small for even one tuple must still dispatch one pixel rather than
    /// returning zero, which would loop forever on a zero-width chunk.
    #[test]
    fn a_budget_below_one_tuple_still_yields_a_chunk() {
        assert_eq!(chunk_pixels_for(0, 4, 10_000), 1);
        assert_eq!(chunk_pixels_for(8, 1, 10_000), 1);
    }

    #[test]
    fn studio_environments_map_onto_the_studio_rig_mode() {
        let (mode, temp_k, spot_mult, exposure, yaw, pitch) =
            environment_params(LightingPreset::Daylight.studio(1.5, 0.4, 0.35))
                .expect("the studio rig is a supported environment");
        assert_eq!(mode, transport_env_mode::STUDIO_RIG);
        assert_eq!(temp_k, illuminant_temperature_k(LightingPreset::Daylight));
        assert_eq!(spot_mult, LightingPreset::Daylight.params().spot_mult);
        assert_eq!((exposure, yaw, pitch), (1.5, 0.4, 0.35));
    }

    /// An HDR map has no `env_mode`, so it must be declined rather than rendered under
    /// whatever the studio-rig parameters happened to default to.
    #[test]
    fn hdr_environment_maps_are_declined() {
        let map = crate::renderer::env_map::EnvironmentMap::uniform(4, 2, [1.0, 1.0, 1.0]);
        assert!(matches!(
            environment_params(EnvironmentSource::HdrMap(&map)),
            Err(GpuFrameError::UnsupportedEnvironment)
        ));
    }

    /// This module must ENFORCE `GemMaterial::gpu_supported`, not merely document it --
    /// routing a material the megakernel cannot handle produces a plausible-looking but
    /// wrong image rather than a failure, which is the worst kind of bug to ship.
    ///
    /// The assertion used to be the inverse: biaxial stones (Alexandrite, Topaz,
    /// Tanzanite) were permanently CPU-only, because no WGSL indicatrix existed. The
    /// Phase 4 port changed that -- `hero_biaxial_wave_dirs`/`per_channel_biaxial_indices`
    /// and the eigen-polarization/mode-Poynting machinery are now in the megakernel,
    /// verified at genuine 0 ULP with a Tier 3 image comparison on all three of those
    /// stones -- so `gpu_supported` is now unconditionally `true`.
    ///
    /// The test is kept, inverted rather than deleted, because its real job never was
    /// "biaxial is special": it is that **this module agrees with the predicate**,
    /// whatever the predicate currently says. If a future material type is added that
    /// the megakernel cannot handle, `gpu_supported` gains a condition and this test
    /// starts failing again -- which is exactly what caught the Phase 4 change here.
    #[test]
    fn every_builtin_routes_the_way_gpu_supported_says() {
        for material in GemMaterial::all_materials() {
            assert!(
                material.gpu_supported(),
                "{} is not GPU-supported, but the megakernel claims to cover every                  built-in since the Phase 4 biaxial port -- if this is a deliberate new                  exclusion, `accumulate`'s decline path must cover it too",
                material.name
            );
        }
    }
}

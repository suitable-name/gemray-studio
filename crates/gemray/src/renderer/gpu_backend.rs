//! The decline-and-fall-back GPU wrapper, shared by every application in this
//! workspace.
//!
//! [`GpuBackend`] wraps [`GpuFrameRenderer`](super::gpu::GpuFrameRenderer) with the one
//! policy every caller needs and none of them should re-derive: try the GPU, fall back
//! to the CPU tracer whenever it declines, and never pretend the GPU ran when it didn't.
//!
//! # Why this lives in `gemray` rather than in each app
//!
//! It previously existed twice -- once in `apps/diagram-gui` and once in
//! `apps/gemray-worker` -- as near-verbatim copies, because the worker cannot depend on
//! the GUI. Two copies of a *policy* is worse than two copies of a helper: the three
//! decline reasons below are correctness rules, not conveniences, and a copy that
//! drifted (forgetting the biaxial check, say) would render a plausible-looking wrong
//! image rather than fail. Both apps already depend on this crate, so one copy lives
//! here.
//!
//! # This module is NOT `#[cfg(feature = "gpu")]`
//!
//! Deliberately, and it is the whole point. Everything under [`super::gpu`] is gated on
//! that feature, so a caller written against it directly needs `#[cfg]` at every call
//! site. [`GpuBackend`] instead exists in *both* configurations -- as the real thing
//! with the feature on, and as a stand-in whose [`GpuBackend::try_accumulate`] always
//! declines with it off -- so an application has exactly one call site and no `#[cfg]`
//! anywhere in its own render loop.
//!
//! # Declining is normal, and decided per call
//!
//! A decline is not an error. It happens when:
//!
//! - this build has no `gpu` feature;
//! - [`GpuBackend::disabled`] was chosen explicitly (a `--no-gpu` flag, or a test that
//!   wants determinism rather than whatever adapter the machine happens to have);
//! - the machine has no usable adapter;
//! - the environment is an **HDR map**, which the megakernel has no `env_mode` for.
//!
//! Biaxial materials (Alexandrite, Topaz, Tanzanite) do NOT decline any more: the
//! `BiaxialIndicatrix` machinery is ported to WGSL and verified at the same Tier
//! 2/Tier 3 bar as every other material, so `GemMaterial::gpu_supported()` is now
//! unconditionally `true` -- see that method's own doc comment.
//!
//! Because the decision is re-made on every call rather than once per session, a scene
//! that declines moves only its own samples to the CPU.
//!
//! # Sample-range additivity
//!
//! [`GpuBackend::try_accumulate`] ADDS into the caller's buffer, exactly as the CPU
//! tracers do. `sample_offset` must be the number of samples already folded into those
//! pixels: both the CPU formula and the GPU shader derive each sample's jitter and RNG
//! from the *absolute* sample index, so reusing an offset redraws identical samples and
//! biases the average instead of extending it. Honouring that is what keeps a GPU
//! worker's samples mergeable with a CPU viewer's -- see this crate's Tier 3 check,
//! which validates CPU and GPU tracing *disjoint* ranges of one image and merging them.
//!
//! # Concurrency
//!
//! Acquire once and share. The renderer lives behind a `Mutex` and every method takes
//! `&self`, so a `GpuBackend` can be wrapped in an `Arc` and handed to many threads (as
//! `gemray-worker`'s `serve` does, one clone per connection): concurrent dispatches
//! serialize on the single GPU queue rather than racing `GpuFrameRenderer`'s own `&mut
//! self` state (its lazily-grown chunk-output buffers). A single-threaded caller pays
//! only an uncontended lock. Re-acquiring per frame or per connection would repeat
//! adapter acquisition and megakernel compilation, both far too slow for that.

use glam::Vec3;

use crate::{
    geometry::GpuFacetPlane,
    optics::{
        materials::GemMaterial,
        raytracer::{Camera, EnvironmentSource, FacetFinish},
    },
};

/// The scene inputs one GPU accumulation call needs.
///
/// Every type here is available whether or not the `gpu` feature is on, which is what
/// lets the stand-in below share this exact signature.
//
// Read only by the `gpu`-gated `try_accumulate`; the stand-in ignores the whole bundle.
// The struct stays defined in both configurations on purpose -- that is what keeps
// callers free of `#[cfg]` -- so its fields being unread in a CPU-only build is the
// expected shape of that build, not an oversight.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub struct GpuSceneRef<'a> {
    pub camera: &'a Camera,
    pub width: u32,
    pub height: u32,
    pub planes: &'a [GpuFacetPlane],
    /// Per-plane surface finish, indexed in step with `planes`. `&[]` means every facet
    /// is polished -- see `GpuFrameScene::facet_finishes` on how a shorter slice is
    /// padded. A caller with no finish concept at all (`gemray-net`'s `SceneState`
    /// carries none) passes `&[]` and gets exactly the behaviour it had before finishes
    /// existed.
    pub facet_finishes: &'a [FacetFinish],
    pub material: &'a GemMaterial,
    pub max_bounces: u32,
    pub environment: EnvironmentSource<'a>,
}

#[cfg(feature = "gpu")]
pub struct GpuBackend {
    renderer: Option<std::sync::Mutex<super::gpu::GpuFrameRenderer>>,
}

#[cfg(feature = "gpu")]
impl GpuBackend {
    /// Acquires an adapter and compiles the megakernel.
    ///
    /// Do this once, off the frame or request path: both steps are slow enough to
    /// matter. A machine with no usable GPU is an expected outcome, logged at `info`,
    /// after which every call declines.
    #[must_use]
    pub fn acquire() -> Self {
        let renderer = match super::gpu::GpuFrameRenderer::new() {
            Ok(r) => {
                tracing::info!(adapter = r.adapter_label(), "GPU render backend active");
                Some(std::sync::Mutex::new(r))
            }
            Err(e) => {
                tracing::info!("GPU render backend unavailable, using CPU tracer: {e}");
                None
            }
        };
        Self { renderer }
    }

    /// Never acquires an adapter -- every call declines, regardless of what hardware is
    /// present.
    ///
    /// The runtime opt-out (a `--no-gpu` flag), and what a test constructs instead of
    /// [`Self::acquire`] so that running with `--features gpu` stays as deterministic as
    /// running without it, rather than tracing on whatever adapter the machine has.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { renderer: None }
    }

    /// This backend's adapter and backend label, if one was genuinely acquired.
    ///
    /// `None` whenever [`Self::disabled`] was chosen or acquisition failed. A caller
    /// that advertises its backend to a peer (`gemray-worker`'s `WELCOME`) must key on
    /// this rather than on whether the feature is compiled in -- claiming a GPU while
    /// tracing on the CPU is worse than never claiming one.
    #[must_use]
    pub fn adapter_label(&self) -> Option<String> {
        self.renderer.as_ref().map(|m| {
            m.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .adapter_label()
                .to_string()
        })
    }

    /// Traces `spp` samples per pixel starting at `sample_offset` and ADDS each pixel's
    /// summed XYZ into `accum`.
    ///
    /// Returns `false` without touching `accum` if the GPU declines -- see the module
    /// doc comment for the five reasons and why none of them is an error.
    pub fn try_accumulate(
        &self,
        scene: &GpuSceneRef<'_>,
        sample_offset: u32,
        spp: u32,
        accum: &mut [Vec3],
    ) -> bool {
        let Some(mutex) = &self.renderer else {
            return false;
        };
        let mut renderer = mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let gpu_scene = super::gpu::GpuFrameScene {
            camera: scene.camera,
            width: scene.width,
            height: scene.height,
            planes: scene.planes,
            facet_finishes: scene.facet_finishes,
            material: scene.material,
            max_bounces: scene.max_bounces,
            environment: scene.environment,
        };
        match renderer.accumulate(&gpu_scene, sample_offset, spp, accum) {
            Ok(()) => true,
            Err(e) => {
                // Not worth `warn`: an unsupported material or environment is a
                // documented limit of the megakernel, not a malfunction, and the CPU
                // path produces the image either way.
                tracing::debug!("GPU declined this dispatch, using CPU tracer: {e}");
                false
            }
        }
    }
}

/// Stand-in for a build without the `gpu` feature -- see the `gpu`-gated [`GpuBackend`]
/// above for what it stands in for and why it exists at all.
#[cfg(not(feature = "gpu"))]
pub struct GpuBackend;

#[cfg(not(feature = "gpu"))]
impl GpuBackend {
    #[must_use]
    pub const fn acquire() -> Self {
        Self
    }

    #[must_use]
    pub const fn disabled() -> Self {
        Self
    }

    /// Always `None`: with no `gpu` feature there is no adapter to name, so a caller
    /// advertising its backend correctly reports CPU.
    #[must_use]
    pub const fn adapter_label(&self) -> Option<String> {
        None
    }

    /// Always declines, leaving `accum` untouched.
    ///
    /// Ignores every argument only to stay signature-compatible with the real
    /// `try_accumulate` above -- that identical signature is the entire point of this
    /// stand-in, and is what keeps callers free of `#[cfg]`.
    #[allow(
        clippy::unused_self,
        reason = "signature must match the `gpu`-gated GpuBackend::try_accumulate"
    )]
    pub const fn try_accumulate(
        &self,
        _scene: &GpuSceneRef<'_>,
        _sample_offset: u32,
        _spp: u32,
        _accum: &mut [Vec3],
    ) -> bool {
        false
    }
}

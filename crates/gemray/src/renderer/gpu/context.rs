//! Adapter/device acquisition for `gemray`'s `gpu`-feature compute infrastructure.

use std::fmt;

/// Why [`GpuContext::acquire`] could not obtain a usable GPU.
///
/// Both variants carry a human-readable message (via each underlying `wgpu` error's
/// `Debug` output, since neither `wgpu::RequestAdapterError` nor
/// `wgpu::RequestDeviceError` is guaranteed `'static`-friendly to store directly here
/// without pulling their exact type into this crate's public API) rather than the raw
/// `wgpu` error type, so callers -- notably the equivalence harness -- can report a
/// clear diagnostic and exit nonzero without panicking, per this crate's Phase-0
/// constraint that a machine with no usable GPU must fail gracefully.
#[derive(Debug, Clone)]
pub enum GpuAcquireError {
    /// No backend/driver on this system produced an adapter at all.
    NoAdapter(String),
    /// An adapter was found but creating a logical device from it failed.
    RequestDevice(String),
}

impl fmt::Display for GpuAcquireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(msg) => write!(
                f,
                "no wgpu adapter is available on this system (no compatible GPU/driver \
                 found via any backend): {msg}"
            ),
            Self::RequestDevice(msg) => write!(f, "failed to acquire a wgpu device: {msg}"),
        }
    }
}

impl std::error::Error for GpuAcquireError {}

/// A live `wgpu` adapter/device/queue, acquired once and reused by every Phase-0
/// self-test (and, eventually, a real compute pipeline).
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Synchronously acquires a `wgpu` adapter/device/queue via [`pollster::block_on`].
    ///
    /// Prefers a high-performance adapter but accepts whatever the system offers (this
    /// workspace's own dev/CI machines include an integrated AMD RDNA2-class GPU via
    /// Vulkan, which is what Phase 0's correctness-only work needs -- no discrete-GPU
    /// requirement).
    ///
    /// # Errors
    ///
    /// [`GpuAcquireError::NoAdapter`] if no backend/driver on this system can produce an
    /// adapter at all, or [`GpuAcquireError::RequestDevice`] if an adapter was found but
    /// device creation failed. Callers MUST treat either as a clean, diagnosable
    /// condition to report and exit on -- never a reason to panic or `unwrap`, since
    /// "no GPU here" is an expected outcome on some machines, not a bug.
    pub fn acquire() -> Result<Self, GpuAcquireError> {
        pollster::block_on(Self::acquire_async())
    }

    async fn acquire_async() -> Result<Self, GpuAcquireError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| GpuAcquireError::NoAdapter(format!("{e:?}")))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gemray gpu-feature compute device"),
                ..Default::default()
            })
            .await
            .map_err(|e| GpuAcquireError::RequestDevice(format!("{e:?}")))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

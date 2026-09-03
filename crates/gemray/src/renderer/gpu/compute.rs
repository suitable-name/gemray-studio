//! Minimal compute-pipeline harness shared by every `gpu`-feature self-test.
//!
//! Shader-module/pipeline creation, POD buffer upload, and blocking storage-buffer
//! readback -- shared by every self-test in `renderer::gpu` (and, eventually, a real GPU
//! raytracer).
//!
//! Nothing here is specific to `gemray`'s physics -- it is generic `wgpu` plumbing, kept
//! deliberately small (no render passes, no textures, no async scheduling beyond a
//! single blocking [`wgpu::Device::poll`] per readback) because Phase 0 only needs
//! compute dispatch and buffer round-trips for its self-tests.

use wgpu::util::DeviceExt;

/// Creates a compute pipeline from inline WGSL source with a single entry point.
///
/// Lets `wgpu` infer the bind group layout from the shader itself (`layout: None`) --
/// every self-test kernel in this module binds a small, fixed set of buffers, so there
/// is no benefit to hand-declaring a [`wgpu::BindGroupLayout`] for each one.
///
/// Every existing caller of this function keeps getting `PipelineCompilationOptions::
/// default()` -- no pipeline-overridable constants set, so a shader's `override`
/// declarations (see [`create_compute_pipeline_with_constants`]) keep their declared
/// defaults. `spectral_transport.wgsl`'s `MATERIAL_CLASS` override defaults to `0u`
/// (`MATERIAL_CLASS_GENERIC`), so this function alone is exactly the GENERIC,
/// runtime-dispatch-over-every-material-class pipeline every self-test in
/// `renderer::gpu` compiles and keeps dispatching -- this task's kernel specialisation
/// (`renderer::gpu::frame`'s per-class pipelines) is additive, not a replacement.
///
/// # Panics
///
/// Panics (via `wgpu`'s own validation/shader-compile error surfacing) if `wgsl_source`
/// fails to parse or validate, or if `entry_point` does not name a `@compute` entry
/// point in it -- a self-test kernel with a bad shader is a bug in this crate, not a
/// runtime input to recover from.
#[must_use]
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    wgsl_source: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    create_compute_pipeline_with_constants(device, label, wgsl_source, entry_point, &[])
}

/// Like [`create_compute_pipeline`], but also sets `constants` on
/// [`wgpu::PipelineCompilationOptions`].
///
/// Each pair is an override's name (or `@id`) and value -- WGSL pipeline-overridable
/// `override` declarations, resolved at PIPELINE creation time rather than at
/// shader-module parse time. [`create_compute_pipeline`] delegates here with
/// `constants: &[]`, which leaves every `override` at its declared default (see that
/// function's doc comment).
///
/// `renderer::gpu::frame::GpuFrameRenderer`'s material-class-specialised pipelines are
/// this function's only caller with a non-empty `constants` today: a fixed
/// `MATERIAL_CLASS` value per specialised pipeline lets naga/the driver dead-code
/// eliminate the other material classes' per-ray state entirely for that pipeline,
/// since the override becomes a compile-time-known value for that specific
/// `wgpu::ComputePipeline` object once these options are applied -- see
/// `spectral_transport.wgsl`'s `MATERIAL_CLASS` declaration for the full mechanism and
/// [`frame`](super::frame)'s module doc comment for why specialisation is done this
/// way (one shared shader source, guards on the override, never a duplicated formula)
/// rather than as separate WGSL files.
///
/// # Panics
///
/// Same conditions as [`create_compute_pipeline`]'s, plus: panics (via `wgpu`'s own
/// validation) if `constants` names an identifier `wgsl_source` has no matching
/// `override` for, or supplies a value that doesn't fit that override's declared type.
#[must_use]
pub fn create_compute_pipeline_with_constants(
    device: &wgpu::Device,
    label: &str,
    wgsl_source: &str,
    entry_point: &str,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    })
}

/// Uploads `data` into a new buffer with `usage` via
/// [`wgpu::util::DeviceExt::create_buffer_init`].
#[must_use]
pub fn upload<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

/// Creates a new, zero-initialized buffer sized for `count` `T`s with `usage`.
///
/// Used for a self-test's output buffer: starting from a known-zero state means any
/// byte the kernel does NOT write (e.g. a struct-layout mismatch landing a field's
/// bytes on top of what should have stayed padding) is distinguishable from a byte the
/// kernel deliberately wrote, rather than being masked by leftover memory-allocator
/// garbage.
#[must_use]
pub fn zeroed_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    count: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let zeros = vec![T::zeroed(); count];
    upload(device, label, &zeros, usage)
}

/// Blocking readback of `count` `T`s from a `COPY_SRC` buffer.
///
/// Copies into a fresh `MAP_READ` staging buffer, submits, blocks on
/// [`wgpu::PollType::wait_indefinitely`], and returns the mapped bytes reinterpreted as
/// `T`.
///
/// # Panics
///
/// Panics if the device poll reports an error (a driver-level failure, not something a
/// caller can meaningfully recover from in a Phase-0 self-test) or if the buffer never
/// finishes mapping.
#[must_use]
pub fn readback<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let byte_len = (count * size_of::<T>()) as wgpu::BufferAddress;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemray gpu self-test readback staging buffer"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemray gpu self-test readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed during readback");
    rx.recv()
        .expect("map_async callback never fired")
        .expect("failed to map readback staging buffer");

    let out = {
        let data = slice
            .get_mapped_range()
            .expect("staging buffer should be mapped at this point");
        bytemuck::cast_slice::<u8, T>(&data).to_vec()
    };
    staging.unmap();
    out
}

/// Dispatches `pipeline` once with a single bind group (group 0), then blocks until the
/// GPU finishes via [`wgpu::PollType::wait_indefinitely`].
///
/// # Panics
///
/// Panics if the device poll reports an error -- see [`readback`]'s doc comment for why
/// that's the right behavior for a Phase-0 self-test.
pub fn dispatch_and_wait(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: (u32, u32, u32),
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemray gpu self-test dispatch encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemray gpu self-test compute pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    queue.submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed during dispatch");
}

/// The non-blocking counterpart to [`dispatch_and_wait`]: submits `pipeline` once with a
/// single bind group (group 0) but does not poll or wait for it.
///
/// R4: `renderer::gpu::frame::GpuFrameRenderer::accumulate`'s chunked production dispatch
/// uses this (paired with [`copy_to_staging`]/[`begin_map_read`]/[`finish_map_read`]
/// below) to queue a chunk's compute work and its readback copy WITHOUT stalling the CPU
/// between them, so a following chunk's dispatch can be queued before the CPU blocks on
/// any one chunk's result -- see that module's doc comment for the double-buffered
/// pipeline this enables. Every self-test keeps using [`dispatch_and_wait`] unchanged.
#[must_use]
pub fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: (u32, u32, u32),
) -> wgpu::SubmissionIndex {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemray gpu pipelined dispatch encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemray gpu pipelined compute pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    queue.submit(Some(encoder.finish()))
}

/// Submits a copy of `count` `T`s from `source` into a fresh `MAP_READ` staging buffer,
/// without blocking or mapping.
///
/// `source` must carry `COPY_SRC`. Pairs with [`begin_map_read`]/[`finish_map_read`] to
/// read the result later, once enough subsequent GPU work has been queued that the
/// eventual blocking wait overlaps that later work rather than idling the GPU -- see
/// [`dispatch`]'s doc comment.
#[must_use]
pub fn copy_to_staging<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
    label: &str,
) -> (wgpu::Buffer, wgpu::SubmissionIndex) {
    let byte_len = (count * size_of::<T>()) as wgpu::BufferAddress;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemray gpu pipelined readback copy encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
    let index = queue.submit(Some(encoder.finish()));
    (staging, index)
}

/// Begins the async map for a staging buffer created by [`copy_to_staging`].
///
/// Returns a receiver that resolves once [`finish_map_read`] drives the device's
/// callbacks -- splitting "begin the map" from "block until it's done" is what lets a
/// caller queue several chunks' worth of GPU work before blocking on any single one of
/// them.
#[must_use]
pub fn begin_map_read(
    staging: &wgpu::Buffer,
) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    rx
}

/// Blocks until `staging`'s producing submission (`copy_index`, from
/// [`copy_to_staging`]) completes and `rx` (from [`begin_map_read`]) resolves, then reads
/// it back and unmaps it.
///
/// The read-back length is whatever `count` [`copy_to_staging`] sized `staging` for.
/// Waiting on `copy_index` specifically (rather than [`wgpu::PollType::wait_indefinitely`]'s
/// "most recent submission") is what avoids this call ALSO waiting for a later chunk's
/// dispatch that may already be queued ahead of it -- see [`dispatch`]'s doc comment.
///
/// # Panics
///
/// Same conditions as [`readback`]'s.
#[must_use]
pub fn finish_map_read<T: bytemuck::Pod>(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    copy_index: wgpu::SubmissionIndex,
    rx: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
) -> Vec<T> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(copy_index),
            timeout: None,
        })
        .expect("device poll failed during pipelined readback");
    rx.recv()
        .expect("map_async callback never fired")
        .expect("failed to map readback staging buffer");

    let out = {
        let data = staging
            .slice(..)
            .get_mapped_range()
            .expect("staging buffer should be mapped at this point");
        bytemuck::cast_slice::<u8, T>(&data).to_vec()
    };
    staging.unmap();
    out
}

/// Builds a single-group bind group from `(binding, buffer)` pairs, using `pipeline`'s
/// own auto-inferred layout for bind group 0 (see [`create_compute_pipeline`]'s
/// `layout: None`).
#[must_use]
pub fn bind_buffers(
    device: &wgpu::Device,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries: &entries,
    })
}

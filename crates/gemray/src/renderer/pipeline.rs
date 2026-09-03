//! GPU compute pipeline for `shaders/gem_raytracer.wgsl` -- **unused scaffolding,
//! pending a real GPU port**.
//!
//! `GemRaytracerPipeline` is never constructed anywhere in this workspace, and calling
//! [`GemRaytracerPipeline::new`] today would panic in `create_shader_module`: the
//! shader it loads is quarantined (see that file's header comment) and no longer
//! contains a valid compute entry point. This module is kept only as a hook for a
//! future GPU port; any such port must translate the CURRENT CPU renderer
//! (`optics::raytracer`) from scratch and validate it with a CPU/GPU equivalence
//! harness, not resurrect the old shader.
use std::borrow::Cow;
use wgpu::PipelineCompilationOptions;

pub struct GemRaytracerPipeline {
    pub compute_pipeline: wgpu::ComputePipeline,
}

impl GemRaytracerPipeline {
    /// # Panics
    ///
    /// Always, currently -- see the module doc comment. The shader this loads is
    /// quarantined dead scaffolding with no `main` entry point, so
    /// `create_compute_pipeline` will fail. Do not call this until a real GPU port
    /// lands.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gem Raytracer Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/gem_raytracer.wgsl"
            ))),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gem Raytracer Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { compute_pipeline }
    }
}

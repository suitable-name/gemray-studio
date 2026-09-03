//! GPU texture upload for an environment map (behind the `gpu` feature).
//!
//! **Unused scaffolding, pending a GPU port** -- see `renderer::pipeline`'s module doc
//! comment for the broader context: the GPU raytracer path this would feed has never
//! run in this workspace.
//!
//! This is the `wgpu` counterpart to the CPU-side [`super::env_map::EnvironmentMap`]:
//! once a caller has decoded/importance-sampled an HDR environment on the CPU (or built
//! one procedurally), `GpuEnvironmentMap::new` uploads its `RGBA32Float` texels to a GPU
//! texture for use as a background/skybox in a rasterized or hybrid preview path.
//! Nothing in `gemray` constructs one yet; it is kept alongside `env_map` for whichever
//! renderer path needs GPU display of an environment.
pub struct GpuEnvironmentMap {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl GpuEnvironmentMap {
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Environment Map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 16),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}

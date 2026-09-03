use glam::Vec3;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuFacetPlane {
    pub normal: [f32; 3],
    pub d: f32,
}

impl GpuFacetPlane {
    #[must_use]
    pub fn new(normal: Vec3, d: f32) -> Self {
        let n = normal.normalize();
        Self {
            normal: [n.x, n.y, n.z],
            d,
        }
    }
}

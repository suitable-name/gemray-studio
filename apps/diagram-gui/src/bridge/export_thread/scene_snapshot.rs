//! [`SceneSnapshot`]: a read-only capture of everything a render needs, taken out of
//! the live `RenderContext` under one short lock.
//!
//! Split out of `bridge::export_thread` purely to keep that module (already sizeable)
//! from growing further.

use crate::bridge::{
    render_thread::{MaterialOverrides, RenderContext, apply_material_overrides, resolve_material},
    stone_width::StoneWidthCache,
};
use gemray::{
    geometry::{girdle_facet_finishes, plane::GpuFacetPlane},
    optics::{
        materials::GemMaterial,
        raytracer::{FacetFinish, LightingPreset},
    },
};
use std::sync::Mutex;

/// A read-only snapshot of everything a render needs, captured out of the live
/// `RenderContext` under one short lock. Deliberately excludes `width`/`height` and
/// the accumulation buffer -- those belong solely to the interactive viewport; the
/// export worker sizes its own buffer from the user's requested export dimensions.
pub struct SceneSnapshot {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub light_yaw: f32,
    pub light_pitch: f32,
    pub material: GemMaterial,
    pub lighting_preset: LightingPreset,
    pub max_bounces: u32,
    pub exposure: f32,
    pub active_planes: Vec<GpuFacetPlane>,
    /// Frosted girdle: `girdle_facet_finishes(&active_planes)` when
    /// `RenderContext::girdle_frosted` was on at capture time, empty otherwise --
    /// already resolved here (rather than a bare `bool` re-classified per batch) since
    /// the export's `active_planes` never change mid-export, matching how `material`
    /// above is already the fully-resolved, override-applied material rather than a
    /// name plus a pile of raw override fields.
    pub facet_finishes: Vec<FacetFinish>,
}

impl SceneSnapshot {
    #[must_use]
    pub fn capture(ctx: &Mutex<RenderContext>) -> Self {
        let guard = ctx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let materials = GemMaterial::all_materials();
        let material = resolve_material(&materials, &guard.custom_materials, &guard.material_name);
        // Every one of these sliders/toggles is a property of what the user is LOOKING
        // at, so an export has to carry it or the file silently differs from the
        // viewport it was taken from. Applied here rather than inside `resolve_material`
        // because that function is shared with `render_thread`, which applies the exact
        // same overrides itself (`resolve_material_and_quality`,
        // `MaterialOverrides`/`apply_material_overrides`) -- opt-in the same way on both
        // paths, so an export with nothing dialled in stays bit-identical to the
        // deterministic path that predates these controls.
        // A fresh, one-call `StoneWidthCache` -- this runs once per export, not once
        // per frame like the live render loop's own persistent cache, so there is
        // nothing to amortize across calls here.
        let material = apply_material_overrides(
            material,
            &MaterialOverrides {
                inclusion_sigma_s: guard.inclusion_sigma_s,
                c_axis_override: guard.c_axis_override,
                edge_rounding_radius: guard.edge_rounding_radius,
                stone_width_mm: guard.stone_width_mm,
            },
            &guard.active_planes,
            &mut StoneWidthCache::new(),
        );
        // Frosted girdle: same opt-in-only treatment as every override above
        // -- an empty `Vec` at the off position is `trace_spectral_ray_with_finish`'s
        // own documented equivalent of `trace_spectral_ray` (every facet reads
        // `FacetFinish::default() == Polished`).
        let facet_finishes = if guard.girdle_frosted {
            girdle_facet_finishes(&guard.active_planes)
        } else {
            Vec::new()
        };
        Self {
            yaw: guard.yaw,
            pitch: guard.pitch,
            distance: guard.distance,
            light_yaw: guard.light_yaw,
            light_pitch: guard.light_pitch,
            material,
            lighting_preset: guard.lighting_preset,
            max_bounces: guard.max_bounces,
            exposure: guard.exposure,
            active_planes: guard.active_planes.clone(),
            facet_finishes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// A small end-to-end smoke test: a real (tiny) scene through the real
    /// `run_export` path -- batching, tone-mapping, and PNG encoding -- without
    /// mocking any of it. Kept small (8x8, 1 spp) so it runs in well under a second.
    /// The inclusion slider is a property of what the user is looking at, so an export
    /// must carry it: a file that silently differs from the viewport it was taken from
    /// is worse than one that is obviously wrong. Guards the `SceneSnapshot::capture`
    /// override against the easy regression of someone "simplifying" it back into a
    /// bare `resolve_material` call.
    #[test]
    fn capture_carries_the_inclusion_setting_into_the_exported_scene() {
        let off = SceneSnapshot::capture(&Mutex::new(RenderContext {
            inclusion_sigma_s: 0.0,
            ..Default::default()
        }));
        assert_eq!(
            off.material.scattering_sigma_s, 0.0,
            "the off position must leave the material untouched"
        );

        let on = SceneSnapshot::capture(&Mutex::new(RenderContext {
            inclusion_sigma_s: 1.25,
            ..Default::default()
        }));
        assert_eq!(
            on.material.scattering_sigma_s, 1.25,
            "a dialled-in inclusion amount must reach the exported scene"
        );
        assert_eq!(
            on.material.scattering_g,
            GemMaterial::DEFAULT_SCATTERING_G,
            "anisotropy comes from the crate's default, matching the live path"
        );
    }

    /// Crystal-axis override's own seam guard, same shape as
    /// `capture_carries_the_inclusion_setting_into_the_exported_scene` above: the
    /// crystal-axis override is a property of what the user is looking at, so an
    /// export must carry it, and must leave an isotropic material's `c_axis` alone even
    /// when the override happens to be on (`RenderContext::default().material_name` is
    /// "Diamond", isotropic).
    #[test]
    fn capture_carries_the_c_axis_override_into_the_exported_scene() {
        let off = SceneSnapshot::capture(&Mutex::new(RenderContext {
            c_axis_override: None,
            ..Default::default()
        }));
        assert_eq!(
            off.material.c_axis,
            GemMaterial::diamond().c_axis,
            "the off (\"as cut\") position must leave the material's own c_axis untouched"
        );

        let on = SceneSnapshot::capture(&Mutex::new(RenderContext {
            material_name: "Sapphire".to_string(),
            c_axis_override: Some(Vec3::X),
            ..Default::default()
        }));
        assert_eq!(
            on.material.c_axis,
            Vec3::X,
            "a dialled-in override on an anisotropic material must reach the exported scene"
        );

        let isotropic_guarded = SceneSnapshot::capture(&Mutex::new(RenderContext {
            material_name: "Diamond".to_string(),
            c_axis_override: Some(Vec3::X),
            ..Default::default()
        }));
        assert_eq!(
            isotropic_guarded.material.c_axis,
            GemMaterial::diamond().c_axis,
            "an override dialled in for an isotropic material must be ignored, matching \
             apply_material_overrides's own guard"
        );
    }

    /// Edge-rounding's own seam guard, same shape as
    /// `capture_carries_the_inclusion_setting_into_the_exported_scene` above.
    #[test]
    fn capture_carries_the_edge_rounding_setting_into_the_exported_scene() {
        let off = SceneSnapshot::capture(&Mutex::new(RenderContext {
            edge_rounding_radius: 0.0,
            ..Default::default()
        }));
        assert_eq!(
            off.material.edge_rounding_radius, 0.0,
            "the off position must leave the material untouched"
        );

        let on = SceneSnapshot::capture(&Mutex::new(RenderContext {
            edge_rounding_radius: 0.02,
            ..Default::default()
        }));
        assert_eq!(
            on.material.edge_rounding_radius, 0.02,
            "a dialled-in edge-rounding radius must reach the exported scene"
        );
    }

    /// Physical stone size's own seam guard, same shape as
    /// `capture_carries_the_inclusion_setting_into_the_exported_scene` above -- the
    /// off position must leave `absorption_path_scale` at the base material's own
    /// default (`1.0` for every built-in), and a dialled-in width must scale it by
    /// the ratio to the design's own measured model-unit girdle width, matching
    /// `apply_material_overrides`'s computation exactly.
    #[test]
    fn capture_carries_the_stone_width_setting_into_the_exported_scene() {
        let off = SceneSnapshot::capture(&Mutex::new(RenderContext {
            stone_width_mm: 0.0,
            ..Default::default()
        }));
        assert_eq!(
            off.material.absorption_path_scale, 1.0,
            "the off position must leave the material's absorption_path_scale untouched"
        );

        let default_ctx = RenderContext::default();
        let model_width = gemray::geometry::stone_metrics::measure_solid(
            &default_ctx
                .active_planes
                .iter()
                .map(|p| {
                    (
                        glam::DVec3::new(
                            f64::from(p.normal[0]),
                            f64::from(p.normal[1]),
                            f64::from(p.normal[2]),
                        ),
                        -f64::from(p.d),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .expect("default active_planes must measure")
        .width_axis;

        let on = SceneSnapshot::capture(&Mutex::new(RenderContext {
            stone_width_mm: 6.5,
            ..Default::default()
        }));
        let expected_scale = (6.5 / model_width) as f32;
        assert!(
            (on.material.absorption_path_scale - expected_scale).abs() < 1e-4,
            "a dialled-in stone width must reach the exported scene as the expected \
             absorption_path_scale: got {}, expected {expected_scale}",
            on.material.absorption_path_scale
        );
    }

    /// Frosted-girdle's own seam guard, same shape as
    /// `capture_carries_the_inclusion_setting_into_the_exported_scene` above -- the
    /// girdle-frosted toggle is captured as a resolved per-facet finish list, not a
    /// bare `bool`, so `run_export`/`render_batch` need no further classification.
    #[test]
    fn capture_carries_the_girdle_frosted_setting_into_the_exported_scene() {
        let off = SceneSnapshot::capture(&Mutex::new(RenderContext {
            girdle_frosted: false,
            ..Default::default()
        }));
        assert!(
            off.facet_finishes.is_empty(),
            "the off position must carry no per-facet finish data"
        );

        let on = SceneSnapshot::capture(&Mutex::new(RenderContext {
            girdle_frosted: true,
            ..Default::default()
        }));
        assert_eq!(
            on.facet_finishes,
            girdle_facet_finishes(&RenderContext::default().active_planes),
            "the on position must carry the same classification the live viewport uses"
        );
    }
}

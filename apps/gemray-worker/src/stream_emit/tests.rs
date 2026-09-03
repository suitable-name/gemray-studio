use super::{
    downsample::downsample_preview,
    emitter::{PendingDelta, effective_cadence_ms},
    sizing::next_batch_size,
};
use crate::render_core;
use gemray::{
    geometry::cuts::StandardGemCuts,
    optics::{materials::GemMaterial, raytracer::LightingPreset},
};
use gemray_net::SceneState;
use glam::Vec3;
use std::time::Duration;

fn tiny_scene() -> SceneState {
    SceneState {
        width: 4,
        height: 4,
        yaw: 0.4,
        pitch: 0.3,
        distance: 3.0,
        light_yaw: 0.85,
        light_pitch: 0.95,
        exposure: 1.0,
        max_bounces: 4,
        lighting_preset: LightingPreset::Daylight,
        material: GemMaterial::diamond(),
        planes: StandardGemCuts::standard_round_brilliant(),
        girdle_frosted: false,
    }
}

#[test]
fn coalesced_deltas_sum_identically_to_un_coalesced_ones() {
    let scene = tiny_scene();
    let pixel_count = scene.width as usize * scene.height as usize;

    let a = render_core::trace_samples(&scene, 0, 3, 1);
    let b = render_core::trace_samples(&scene, 3, 5, 1);

    let mut pending = PendingDelta::new(pixel_count);
    pending.add(0, 3, &a);
    pending.add(3, 5, &b);
    let (first_sample, samples, coalesced) = pending.take().unwrap();
    assert_eq!(first_sample, 0);
    assert_eq!(samples, 8);

    let direct = render_core::trace_samples(&scene, 0, 8, 1);
    for (c, d) in coalesced.iter().zip(&direct) {
        let diff = (*c - *d).abs();
        let scale = c.abs().max(d.abs()).max(Vec3::splat(1e-6));
        assert!((diff / scale).max_element() < 1e-3, "c={c:?} d={d:?}");
    }
}

#[test]
fn pending_delta_take_returns_none_when_empty() {
    let mut pending = PendingDelta::new(16);
    assert!(pending.take().is_none());
}

#[test]
fn pending_delta_is_empty_again_immediately_after_take() {
    let scene = tiny_scene();
    let pixel_count = scene.width as usize * scene.height as usize;
    let a = render_core::trace_samples(&scene, 0, 2, 1);

    let mut pending = PendingDelta::new(pixel_count);
    pending.add(0, 2, &a);
    assert!(pending.take().is_some());
    assert!(pending.take().is_none());
}

#[test]
fn next_batch_size_grows_when_well_under_budget() {
    let next = next_batch_size(4, Duration::from_millis(10));
    assert!(next > 4, "expected growth from 4, got {next}");
}

#[test]
fn next_batch_size_shrinks_when_over_budget() {
    let next = next_batch_size(100, Duration::from_millis(400));
    assert!(next < 100, "expected shrink from 100, got {next}");
    assert!(next >= 1);
}

#[test]
fn next_batch_size_never_returns_zero() {
    assert!(next_batch_size(0, Duration::from_millis(1)) >= 1);
    assert!(next_batch_size(1, Duration::from_secs(10)) >= 1);
}

#[test]
fn downsample_preview_produces_the_requested_dimensions() {
    let buf = vec![Vec3::ONE; 8 * 8];
    let out = downsample_preview(&buf, 8, 8, 2, 2);
    assert_eq!(out.len(), 4);
}

#[test]
fn downsample_preview_of_a_uniform_buffer_preserves_the_value() {
    let buf = vec![Vec3::new(2.0, 4.0, 6.0); 8 * 8];
    let out = downsample_preview(&buf, 8, 8, 2, 2);
    for v in out {
        assert!((v - Vec3::new(2.0, 4.0, 6.0)).length() < 1e-5);
    }
}

#[test]
fn effective_cadence_ms_is_zero_for_fewer_than_two_emissions() {
    assert_eq!(effective_cadence_ms(Duration::from_secs(1), 0), 0);
    assert_eq!(effective_cadence_ms(Duration::from_secs(1), 1), 0);
}

#[test]
fn effective_cadence_ms_averages_the_interval() {
    // 3 emissions over 2 seconds -> 2 intervals -> 1000ms average.
    assert_eq!(effective_cadence_ms(Duration::from_secs(2), 3), 1000);
}

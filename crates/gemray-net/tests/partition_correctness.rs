//! The property the whole remote-offload design rests on: sample partitioning is
//! additive. Tracing samples `[0, 64)` for a pixel in one batch must sum to bit-for-bit
//! the same radiance as tracing `[0, 32)` and `[32, 64)` in two separate batches and
//! adding the results -- exactly what letting two different nodes each trace a disjoint
//! sample range and summing their contributions requires to be correct.
//!
//! This reproduces the EXACT per-sample seed derivation
//! `apps/diagram-gui/src/bridge/render_thread.rs` uses against the real
//! `gemray::optics::raytracer::trace_spectral_ray` (not a stand-in): the seed is a
//! function of `(pixel_index, sample_number)` alone, via `hash_u32`, so it depends only
//! on the pixel and the ABSOLUTE sample number -- never on how many samples happen to
//! be in the batch a given call computes, or which batch a sample happens to land in.
//! That's precisely what makes the partition arbitrary: any way of slicing
//! `[0, total_samples)` into disjoint ranges produces the same sum.

use gemray::{
    geometry::cuts::StandardGemCuts,
    optics::{
        materials::GemMaterial,
        raytracer::{
            Camera, HERO_WAVELENGTH_ROTATION_STREAM, LightingPreset,
            PIXEL_JITTER_X_ROTATION_STREAM, PIXEL_JITTER_Y_ROTATION_STREAM,
            cranley_patterson_rotate, hash_u32, low_discrepancy_base2, radical_inverse_base,
            trace_spectral_ray,
        },
    },
};
use glam::Vec3;

/// The fixed per-comparison scene `trace_one_sample`/`sum_samples` trace against --
/// bundled per `clippy::too_many_arguments` (see
/// `crates/gemray/src/optics/raytracer/refraction.rs`'s `RayMaterialContext` for the
/// established pattern this follows). Every field here is held constant across one
/// whole partition-correctness comparison; only `(x, y, sample_num)` -- the thing
/// under test -- varies per call, so those stay separate arguments rather than
/// joining this struct.
struct SceneUnderTest<'a> {
    camera: &'a Camera,
    planes: &'a [gemray::geometry::GpuFacetPlane],
    material: &'a GemMaterial,
    max_bounces: u32,
    environment: gemray::optics::raytracer::EnvironmentSource<'a>,
    width: u32,
    height: u32,
}

/// Traces one sample for pixel `(x, y)` of `scene`, sample number `sample_num`
/// (absolute, not batch-relative) -- byte-for-byte the same seed, jitter, and
/// stratified hero-wavelength derivation as `render_frame_scanlines` in
/// `apps/diagram-gui/src/bridge/render_thread.rs`. The property this test exists to
/// prove -- sample generation is a PURE function of `(pixel_index,
/// absolute_sample_index)` alone -- holds just as much for the stratified construction
/// as it did for the old unstratified one: `low_discrepancy_base2` reads only
/// `sample_num`, and the Cranley-Patterson rotation offsets read only
/// `global_pixel_idx`, so nothing here depends on batch boundaries or call order.
fn trace_one_sample(scene: &SceneUnderTest<'_>, x: u32, y: u32, sample_num: u32) -> Vec3 {
    let global_pixel_idx = y * scene.width + x;
    let seed =
        hash_u32(global_pixel_idx.wrapping_mul(0x9e37_79b9) ^ sample_num.wrapping_mul(0x85eb_ca6b));

    let rot_jx = low_discrepancy_base2(hash_u32(global_pixel_idx ^ PIXEL_JITTER_X_ROTATION_STREAM));
    let rot_jy = low_discrepancy_base2(hash_u32(global_pixel_idx ^ PIXEL_JITTER_Y_ROTATION_STREAM));
    let rot_hero =
        low_discrepancy_base2(hash_u32(global_pixel_idx ^ HERO_WAVELENGTH_ROTATION_STREAM));
    let jx = cranley_patterson_rotate(low_discrepancy_base2(sample_num), rot_jx) - 0.5;
    let jy = cranley_patterson_rotate(radical_inverse_base(sample_num, 3), rot_jy) - 0.5;
    let hero_rand = cranley_patterson_rotate(radical_inverse_base(sample_num, 5), rot_hero);

    let ray = scene.camera.generate_ray(
        x as f32,
        y as f32,
        scene.width as f32,
        scene.height as f32,
        jx,
        jy,
    );
    trace_spectral_ray(
        ray,
        scene.planes,
        scene.material,
        scene.max_bounces,
        scene.environment,
        seed,
        hero_rand,
        None,
    )
}

/// Asserts `a` and `b` agree to within a tight relative tolerance.
///
/// Not `assert_eq!`: floating-point addition is not associative, so summing the same
/// 64 `f32` terms in a different grouping (one running total vs. two partial totals
/// added together at the end) can differ in the last one or two bits of precision even
/// though the mathematical sums are identical -- that is a property of `f32`
/// arithmetic, not a bug in the partitioning. What this test exists to catch is a REAL
/// discrepancy (a seed depending on batch-relative state, a dropped sample, a
/// systematic bias), which would show up as an error many orders of magnitude larger
/// than float-rounding noise -- so `1e-4` relative is tight enough to catch a real bug
/// while tolerant of legitimate summation-order rounding.
fn assert_vec3_approx_eq(a: Vec3, b: Vec3, msg: &str) {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(Vec3::splat(1e-6));
    let rel = diff / scale;
    assert!(
        rel.max_element() < 1e-4,
        "{msg}: left={a:?} right={b:?} rel_diff={rel:?}"
    );
}

/// Sums `trace_one_sample` over `sample_range`, exactly what one worker node
/// accumulating its assigned batch of sample indices for one pixel would do.
fn sum_samples(
    scene: &SceneUnderTest<'_>,
    x: u32,
    y: u32,
    sample_range: std::ops::Range<u32>,
) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for sample_num in sample_range {
        sum += trace_one_sample(scene, x, y, sample_num);
    }
    sum
}

#[test]
fn batch_0_64_equals_batch_0_32_plus_batch_32_64() {
    let planes = StandardGemCuts::standard_round_brilliant();
    let material = GemMaterial::by_name("Ruby").expect("Ruby is a built-in material");
    let camera = Camera::new(0.4, 0.3, 3.0, 45.0);
    let environment = LightingPreset::Daylight.studio(1.0, 0.85, 0.95);
    let width = 64;
    let height = 64;
    let max_bounces = 6;
    let scene = SceneUnderTest {
        camera: &camera,
        planes: &planes,
        material: &material,
        max_bounces,
        environment,
        width,
        height,
    };

    // A handful of pixels: dead center (through the table, definitely hits the gem),
    // an off-center pixel still inside the gem's silhouette, and a corner pixel that
    // misses the gem entirely and only samples the studio background -- the property
    // must hold for the miss path too, not just the hit path.
    for (x, y) in [(32, 32), (20, 40), (2, 2)] {
        let whole: Vec3 = sum_samples(&scene, x, y, 0..64);
        let first_half: Vec3 = sum_samples(&scene, x, y, 0..32);
        let second_half: Vec3 = sum_samples(&scene, x, y, 32..64);
        let split_sum = first_half + second_half;

        assert_vec3_approx_eq(
            whole,
            split_sum,
            &format!("pixel ({x}, {y}): sum over [0,64) must equal sum over [0,32) + [32,64)"),
        );
    }
}

#[test]
fn an_uneven_three_way_split_also_sums_exactly() {
    // Not just a symmetric halves split: [0,7) + [7,19) + [19,64) is a deliberately
    // uneven three-way partition, closer to how real worker nodes with different
    // throughput would actually divide up a sample budget.
    let planes = StandardGemCuts::standard_round_brilliant();
    let material = GemMaterial::diamond();
    let camera = Camera::new(-0.6, 0.5, 3.2, 40.0);
    let environment = LightingPreset::RingLights.studio(1.2, 0.5, 1.1);
    let width = 48;
    let height = 48;
    let max_bounces = 8;
    let x = 24;
    let y = 24;
    let scene = SceneUnderTest {
        camera: &camera,
        planes: &planes,
        material: &material,
        max_bounces,
        environment,
        width,
        height,
    };

    let whole = sum_samples(&scene, x, y, 0..64);
    let part_a = sum_samples(&scene, x, y, 0..7);
    let part_b = sum_samples(&scene, x, y, 7..19);
    let part_c = sum_samples(&scene, x, y, 19..64);

    assert_vec3_approx_eq(whole, part_a + part_b + part_c, "uneven three-way split");
}

#[test]
fn single_sample_batches_summed_one_at_a_time_still_match_the_whole_batch() {
    // The extreme case: every sample computed in its own separate call (as if every
    // sample went to a different worker node), summed one at a time. If ANY hidden
    // batch-relative state had leaked into the seed derivation, this is the test most
    // likely to catch it.
    let planes = StandardGemCuts::standard_round_brilliant();
    let material = GemMaterial::by_name("Sapphire").expect("Sapphire is a built-in material");
    let camera = Camera::new(0.0, 0.0, 3.0, 45.0);
    let environment = LightingPreset::DarkSpotlight.studio(1.5, 0.2, 0.6);
    let width = 32;
    let height = 32;
    let max_bounces = 4;
    let x = 16;
    let y = 16;
    let scene = SceneUnderTest {
        camera: &camera,
        planes: &planes,
        material: &material,
        max_bounces,
        environment,
        width,
        height,
    };

    let whole = sum_samples(&scene, x, y, 0..16);

    let mut one_at_a_time = Vec3::ZERO;
    for sample_num in 0..16 {
        one_at_a_time += sum_samples(&scene, x, y, sample_num..sample_num + 1);
    }

    assert_vec3_approx_eq(
        whole,
        one_at_a_time,
        "16 single-sample batches summed one at a time",
    );
}

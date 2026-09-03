//! `SceneState` must round-trip through `postcard` unchanged -- including a material
//! with absorption bands (and biaxial data) and a full facet-plane set, since those are
//! exactly the fields a name/id-based scheme would have gotten wrong (see
//! `gemray_net::scene`'s module docs).

use gemray::{
    geometry::cuts::StandardGemCuts,
    optics::{materials::GemMaterial, raytracer::LightingPreset},
};
use gemray_net::SceneState;

fn sample_scene() -> SceneState {
    // Alexandrite: has both absorption bands (Cr3+ chromophore, via legacy_rgb_bands)
    // AND biaxial_delta_beta_alpha = Some(..) -- the two nested-data cases most likely
    // to be dropped by an incomplete Serialize/Deserialize derive.
    let material = GemMaterial::by_name("Alexandrite").expect("Alexandrite is a built-in material");
    assert!(
        !material.absorption.o_ray.is_empty(),
        "test material should actually exercise absorption bands"
    );
    assert!(
        material.biaxial_delta_beta_alpha.is_some(),
        "test material should actually exercise biaxial data"
    );

    let planes = StandardGemCuts::standard_round_brilliant();
    assert!(
        planes.len() >= 57,
        "test should exercise a full facet-plane set"
    );

    SceneState {
        width: 640,
        height: 480,
        yaw: 0.37,
        pitch: -0.12,
        distance: 3.5,
        light_yaw: 0.85,
        light_pitch: 0.95,
        exposure: 1.25,
        max_bounces: 12,
        lighting_preset: LightingPreset::RingLights,
        material,
        planes,
        girdle_frosted: false,
    }
}

#[test]
fn scene_state_round_trips_unchanged_through_postcard() {
    let scene = sample_scene();

    let bytes = postcard::to_allocvec(&scene).expect("SceneState must serialize");
    let decoded: SceneState = postcard::from_bytes(&bytes).expect("SceneState must deserialize");

    assert_eq!(
        scene, decoded,
        "SceneState must round-trip bit-for-bit through postcard"
    );
}

#[test]
fn scene_state_round_trip_preserves_every_lighting_preset() {
    for preset in LightingPreset::ALL {
        let mut scene = sample_scene();
        scene.lighting_preset = preset;

        let bytes = postcard::to_allocvec(&scene).unwrap();
        let decoded: SceneState = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.lighting_preset, preset);
    }
}

#[test]
fn scene_state_round_trip_preserves_a_diamond_with_empty_absorption() {
    // The opposite corner case from `sample_scene`'s Alexandrite: a colourless
    // built-in whose absorption band vectors are empty, and whose
    // biaxial_delta_beta_alpha is None -- make sure the empty/None cases round-trip
    // too, not just the populated ones.
    let mut scene = sample_scene();
    scene.material = GemMaterial::diamond();
    assert_eq!(
        scene.material.absorption.o_ray,
        Vec::new(),
        "diamond should have no absorption bands"
    );
    assert!(scene.material.biaxial_delta_beta_alpha.is_none());

    let bytes = postcard::to_allocvec(&scene).unwrap();
    let decoded: SceneState = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(scene, decoded);
}

#[test]
fn scene_state_round_trip_preserves_an_empty_plane_set() {
    let mut scene = sample_scene();
    scene.planes = Vec::new();

    let bytes = postcard::to_allocvec(&scene).unwrap();
    let decoded: SceneState = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(scene, decoded);
    assert_eq!(decoded.planes, Vec::new());
}

/// `girdle_frosted` is the wire-format encoding of the viewer's frosted-girdle toggle
/// (see [`SceneState::girdle_frosted`]'s doc comment for why a bool rather than a
/// shipped `Vec<FacetFinish>`) -- both its values must round-trip, not just the default.
#[test]
fn scene_state_round_trip_preserves_girdle_frosted_both_ways() {
    for girdle_frosted in [false, true] {
        let mut scene = sample_scene();
        scene.girdle_frosted = girdle_frosted;

        let bytes = postcard::to_allocvec(&scene).unwrap();
        let decoded: SceneState = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.girdle_frosted, girdle_frosted);
        assert_eq!(scene, decoded);
    }
}

/// `GemMaterial::absorption_path_scale` is embedded inside `SceneState::material`, not
/// a top-level `SceneState` field like `girdle_frosted` -- but the same round-trip risk
/// applies (postcard's fixed field-order layout, see `messages::PROTOCOL_VERSION`'s doc
/// comment), so both the default (`1.0`, every built-in material) and a dialled-in
/// physical-stone-size scale must survive the wire unchanged.
#[test]
fn scene_state_round_trip_preserves_absorption_path_scale_both_default_and_scaled() {
    for scale in [1.0f32, 0.42, 2.75] {
        let mut scene = sample_scene();
        scene.material = scene.material.with_absorption_path_scale(scale);

        let bytes = postcard::to_allocvec(&scene).unwrap();
        let decoded: SceneState = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.material.absorption_path_scale, scale);
        assert_eq!(scene, decoded);
    }
}

// NOTE on `#[serde(default)]` and postcard: unlike a self-describing format (JSON), a
// postcard-encoded struct is a plain sequence of field encodings with no names and no
// "field present/absent" signal on the wire -- so a byte stream literally one field
// shorter than the current shape does NOT gracefully default the missing field the way
// it would deserializing a JSON object missing a key; postcard reports
// `DeserializeUnexpectedEnd` instead (confirmed empirically -- an earlier version of
// this test asserted the opposite and failed). `#[serde(default)]` on
// `SceneState::girdle_frosted` is for `gemray-worker::render_cmd`'s on-disk
// `scene.json` files (loaded via `serde_json`, which IS self-describing), not for this
// crate's postcard wire format -- the network path's cross-version protection is
// `PROTOCOL_VERSION` (see `messages::tests::protocol_version_is_1` and that const's own
// doc comment), not the `serde` attribute. Nothing here exercises the JSON path since
// this crate has no `serde_json` dependency of its own.

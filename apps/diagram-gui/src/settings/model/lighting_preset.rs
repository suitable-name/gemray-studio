//! [`LightingPreset`]: a named, user-saveable snapshot of the lighting rig,
//! plus the built-in presets shipped with the app.
//!
//! Split out of `settings::model` purely to keep that module (already sizeable) from
//! growing further.

use serde::{Deserialize, Serialize};

/// A named, user-saveable snapshot of the lighting rig.
///
/// Deliberately excludes camera yaw/pitch: yaw/pitch is "what part of the gem am I
/// looking at", which a user switching lighting moods generally wants left alone,
/// whereas camera *distance* (included) interacts with exposure/framing enough to be
/// part of a lighting "look". Applying a preset must never spin the gem out from under
/// the user.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightingPreset {
    pub name: String,
    /// Built-in presets ship with the app and cannot be renamed or deleted -- see
    /// `SettingsFile::rename_preset` / `delete_preset`, which both refuse to act on one.
    #[serde(default)]
    pub built_in: bool,
    pub light_yaw_deg: f32,
    pub light_pitch_deg: f32,
    pub exposure: f32,
    pub lighting_rig: String,
    pub camera_distance: f32,
}

/// The 2-3 built-in presets shipped with the app, deliberately kept to a small,
/// sensible set that cannot be deleted. Names double as their stable identity --
/// `SettingsFile::ensure_built_in_presets` matches on `name` to decide whether a
/// loaded file already has one.
#[must_use]
pub fn built_in_presets() -> Vec<LightingPreset> {
    vec![
        LightingPreset {
            name: "Studio Softbox".to_string(),
            built_in: true,
            light_yaw_deg: 48.0,
            light_pitch_deg: 54.0,
            exposure: 1.0,
            lighting_rig: "Gem Studio Ring Lights".to_string(),
            camera_distance: 2.4,
        },
        LightingPreset {
            name: "Daylight Bright".to_string(),
            built_in: true,
            light_yaw_deg: 30.0,
            light_pitch_deg: 65.0,
            exposure: 1.3,
            // Corrected label -- D65 is 6500K, not 5500K. Graceful migration
            // for any settings file already persisted with the old, mislabelled string
            // lives in `gemray::optics::LightingPreset::from_label`, which both strings
            // parse identically (both resolve to the same D65/6500K preset).
            lighting_rig: "D65 Daylight (6500K)".to_string(),
            camera_distance: 2.2,
        },
        LightingPreset {
            name: "Dramatic Spotlight".to_string(),
            built_in: true,
            light_yaw_deg: 300.0,
            light_pitch_deg: 25.0,
            exposure: 0.7,
            lighting_rig: "Dramatic Dark Spotlight".to_string(),
            camera_distance: 2.6,
        },
    ]
}

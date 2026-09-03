//! Plain-data settings model: settings persistence + lighting presets.
//!
//! Deliberately has no dependency on Slint or any threading primitive -- every type
//! and method here is pure data plus pure logic, so it can be exercised directly by
//! unit tests without spinning up a UI or a background thread. `gui::mod` is the only
//! place that bridges these types to Slint properties / render-thread state.
//!
//! Split into submodules purely to keep this file from growing further:
//! [`app_settings`] (`AppSettings`, the persisted render/UI configuration),
//! [`worker`] (`PreviewScale`/`LocalPreviewScale`/`WorkerSettings`, remote-rendering
//! settings), [`lighting_preset`] (`LightingPreset`, saveable lighting-rig
//! snapshots), and [`settings_file`] (`SettingsFile`, the full on-disk document).

mod app_settings;
mod lighting_preset;
mod settings_file;
mod worker;

pub use lighting_preset::LightingPreset;
pub use settings_file::SettingsFile;
pub use worker::{LiveComputeTarget, LocalPreviewScale, PreviewScale, WorkerSettings};
// Re-exported for path compatibility only -- each of these is named only from
// `#[cfg(test)]` code (this module's own tests, or `settings::store`'s) via this
// `model::` path, which isn't part of a plain (non-test) build. Trimmed to exactly the
// names actually reached that way; verified by temporarily dropping the `#[allow]` and
// reading which names clippy still flags as unused with `--tests` enabled -- 16 sibling
// `DEFAULT_*` consts in `app_settings` did NOT survive that check (nothing reaches them
// via `model::`, only via `app_settings::` directly) and were dropped from here rather
// than carried along under a blanket allow.
#[allow(
    unused_imports,
    reason = "AppSettings/DEFAULT_REMOTE_RENDER_SAMPLES/DEFAULT_RENDER_HEIGHT/\
              DEFAULT_RENDER_WIDTH/DEFAULT_TARGET_SAMPLES are named via this exact \
              `model::` path only from #[cfg(test)] code, so a plain (non-test) build \
              sees them as unused"
)]
pub use app_settings::{
    AppSettings, DEFAULT_REMOTE_RENDER_SAMPLES, DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH,
    DEFAULT_TARGET_SAMPLES,
};
#[allow(
    unused_imports,
    reason = "named via this `model::` path only from settings::store's #[cfg(test)] \
              code, so a plain (non-test) build sees it as unused"
)]
pub use lighting_preset::built_in_presets;
#[allow(
    unused_imports,
    reason = "named via this `model::` path only from #[cfg(test)] code, so a plain \
              (non-test) build sees it as unused"
)]
pub use worker::DEFAULT_WORKER_CADENCE_MS;

#[cfg(test)]
mod tests {
    use super::*;
    use gemray_net::messages::TransferMode;

    #[test]
    fn default_settings_file_carries_all_built_ins() {
        let file = SettingsFile::default();
        assert_eq!(file.presets.len(), built_in_presets().len());
        assert!(file.presets.iter().all(|p| p.built_in));
    }

    #[test]
    fn ensure_built_in_presets_is_idempotent_and_additive() {
        let mut file = SettingsFile {
            settings: AppSettings::default(),
            presets: vec![],
        };
        file.ensure_built_in_presets();
        let after_first = file.presets.clone();
        file.ensure_built_in_presets();
        assert_eq!(
            file.presets, after_first,
            "calling twice must not duplicate entries"
        );
        assert_eq!(file.presets.len(), built_in_presets().len());
    }

    #[test]
    fn ensure_built_in_presets_preserves_user_presets() {
        let mut file = SettingsFile {
            settings: AppSettings::default(),
            presets: vec![LightingPreset {
                name: "My Custom Look".to_string(),
                built_in: false,
                light_yaw_deg: 10.0,
                light_pitch_deg: 20.0,
                exposure: 0.9,
                lighting_rig: "Incandescent (3200K)".to_string(),
                camera_distance: 3.0,
            }],
        };
        file.ensure_built_in_presets();
        assert!(file.presets.iter().any(|p| p.name == "My Custom Look"));
        assert_eq!(file.presets.len(), built_in_presets().len() + 1);
    }

    #[test]
    fn upsert_user_preset_creates_and_overwrites() {
        let mut file = SettingsFile::default();
        let preset = LightingPreset {
            name: "Evening Glow".to_string(),
            built_in: false,
            light_yaw_deg: 200.0,
            light_pitch_deg: 40.0,
            exposure: 0.85,
            lighting_rig: "Incandescent (3200K)".to_string(),
            camera_distance: 2.8,
        };
        file.upsert_user_preset(preset.clone()).unwrap();
        assert_eq!(file.find_preset("Evening Glow"), Some(&preset));

        let mut updated = preset;
        updated.exposure = 1.1;
        file.upsert_user_preset(updated.clone()).unwrap();
        assert_eq!(file.find_preset("Evening Glow"), Some(&updated));
        // still only one entry with that name
        assert_eq!(
            file.presets
                .iter()
                .filter(|p| p.name == "Evening Glow")
                .count(),
            1
        );
    }

    #[test]
    fn upsert_user_preset_refuses_to_shadow_a_built_in() {
        let mut file = SettingsFile::default();
        let clash = LightingPreset {
            name: "Studio Softbox".to_string(),
            built_in: false,
            light_yaw_deg: 0.0,
            light_pitch_deg: 0.0,
            exposure: 1.0,
            lighting_rig: "Gem Studio Ring Lights".to_string(),
            camera_distance: 2.0,
        };
        let result = file.upsert_user_preset(clash);
        assert!(result.is_err());
        // the built-in must be untouched
        assert_eq!(
            file.find_preset("Studio Softbox").unwrap().light_pitch_deg,
            54.0
        );
    }

    #[test]
    fn upsert_user_preset_rejects_empty_name() {
        let mut file = SettingsFile::default();
        let preset = LightingPreset {
            name: "   ".to_string(),
            built_in: false,
            light_yaw_deg: 0.0,
            light_pitch_deg: 0.0,
            exposure: 1.0,
            lighting_rig: "Gem Studio Ring Lights".to_string(),
            camera_distance: 2.0,
        };
        assert!(file.upsert_user_preset(preset).is_err());
    }

    #[test]
    fn rename_preset_renames_user_preset() {
        let mut file = SettingsFile::default();
        file.upsert_user_preset(LightingPreset {
            name: "Draft".to_string(),
            built_in: false,
            light_yaw_deg: 0.0,
            light_pitch_deg: 0.0,
            exposure: 1.0,
            lighting_rig: "Gem Studio Ring Lights".to_string(),
            camera_distance: 2.0,
        })
        .unwrap();
        file.rename_preset("Draft", "Final Cut").unwrap();
        assert!(file.find_preset("Draft").is_none());
        assert!(file.find_preset("Final Cut").is_some());
    }

    #[test]
    fn rename_preset_refuses_built_ins_and_name_collisions() {
        let mut file = SettingsFile::default();
        assert!(file.rename_preset("Studio Softbox", "New Name").is_err());

        file.upsert_user_preset(LightingPreset {
            name: "Second".to_string(),
            built_in: false,
            light_yaw_deg: 0.0,
            light_pitch_deg: 0.0,
            exposure: 1.0,
            lighting_rig: "Gem Studio Ring Lights".to_string(),
            camera_distance: 2.0,
        })
        .unwrap();
        // Renaming "Second" to a built-in's name must fail (collision).
        assert!(file.rename_preset("Second", "Studio Softbox").is_err());
    }

    #[test]
    fn delete_preset_removes_user_preset_but_refuses_built_ins() {
        let mut file = SettingsFile::default();
        file.upsert_user_preset(LightingPreset {
            name: "Temp".to_string(),
            built_in: false,
            light_yaw_deg: 0.0,
            light_pitch_deg: 0.0,
            exposure: 1.0,
            lighting_rig: "Gem Studio Ring Lights".to_string(),
            camera_distance: 2.0,
        })
        .unwrap();
        file.delete_preset("Temp").unwrap();
        assert!(file.find_preset("Temp").is_none());

        assert!(file.delete_preset("Studio Softbox").is_err());
        assert!(file.find_preset("Studio Softbox").is_some());
    }

    #[test]
    fn toml_round_trip_preserves_settings_and_presets() {
        let mut file = SettingsFile::default();
        file.settings.exposure = 1.42;
        file.settings.selected_material = "Sapphire".to_string();
        file.upsert_user_preset(LightingPreset {
            name: "Round Trip Test".to_string(),
            built_in: false,
            light_yaw_deg: 123.0,
            light_pitch_deg: 45.0,
            exposure: 1.2,
            lighting_rig: "Incandescent (3200K)".to_string(),
            camera_distance: 3.3,
        })
        .unwrap();

        let toml_str = toml::to_string_pretty(&file).expect("serialize");
        let parsed: SettingsFile = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed, file);
    }

    // ---- Remote rendering: PreviewScale / WorkerSettings / worker CRUD -----------

    #[test]
    fn preview_scale_percentages_match_their_names() {
        assert_eq!(PreviewScale::Full.percent(), 100);
        assert_eq!(PreviewScale::Half.percent(), 50);
        assert_eq!(PreviewScale::Quarter.percent(), 25);
        assert_eq!(PreviewScale::Custom(10).percent(), 10);
    }

    #[test]
    fn preview_scale_custom_clamps_to_one_through_one_hundred() {
        assert_eq!(PreviewScale::Custom(0).percent(), 1);
        assert_eq!(PreviewScale::Custom(500).percent(), 100);
        assert_eq!(PreviewScale::Custom(1).percent(), 1);
        assert_eq!(PreviewScale::Custom(100).percent(), 100);
    }

    #[test]
    fn preview_scale_resolve_scales_dimensions_and_floors_at_one_by_one() {
        assert_eq!(PreviewScale::Full.resolve(1920, 1080), (1920, 1080));
        assert_eq!(PreviewScale::Half.resolve(200, 100), (100, 50));
        assert_eq!(PreviewScale::Quarter.resolve(200, 100), (50, 25));
        // A tiny session resolution must never resolve to a zero-area preview.
        assert_eq!(PreviewScale::Quarter.resolve(2, 2), (1, 1));
        assert_eq!(PreviewScale::Custom(1).resolve(10, 10), (1, 1));
    }

    #[test]
    fn worker_settings_default_is_live_progressive_500ms_full_preview() {
        let w = WorkerSettings::default();
        assert_eq!(w.transfer_mode, TransferMode::LiveProgressive);
        assert_eq!(w.cadence_ms, DEFAULT_WORKER_CADENCE_MS);
        assert_eq!(w.preview_scale, PreviewScale::Full);
    }

    #[test]
    fn worker_settings_cert_paths_join_the_bundle_directory() {
        let w = WorkerSettings {
            cert_dir: "C:/certs/laptop".to_string(),
            ..WorkerSettings::default()
        };
        assert_eq!(w.ca_path(), std::path::Path::new("C:/certs/laptop/ca.pem"));
        assert_eq!(
            w.client_cert_path(),
            std::path::Path::new("C:/certs/laptop/client.pem")
        );
        assert_eq!(
            w.client_key_path(),
            std::path::Path::new("C:/certs/laptop/client.key")
        );
    }

    #[test]
    fn effective_cadence_ms_never_goes_below_the_workers_advertised_floor() {
        let fast_request = WorkerSettings {
            cadence_ms: 50,
            ..WorkerSettings::default()
        };
        // The worker can only usefully do 100ms -- the UI-requested 50ms is clamped up.
        assert_eq!(fast_request.effective_cadence_ms(100), 100);

        let slow_request = WorkerSettings {
            cadence_ms: 2000,
            ..WorkerSettings::default()
        };
        // Already slower than the floor -- left untouched.
        assert_eq!(slow_request.effective_cadence_ms(100), 2000);
    }

    #[test]
    fn stream_config_reflects_transfer_mode_clamped_cadence_and_scaled_preview() {
        let worker = WorkerSettings {
            transfer_mode: TransferMode::FinalOnly,
            cadence_ms: 10,
            preview_scale: PreviewScale::Half,
            ..WorkerSettings::default()
        };
        let cfg = worker.stream_config(100, 800, 600);
        assert_eq!(cfg.transfer_mode, TransferMode::FinalOnly);
        assert_eq!(cfg.cadence_ms, 100); // clamped up to the worker's floor
        assert_eq!(
            cfg.preview,
            Some(gemray_net::messages::PreviewConfig {
                width: 400,
                height: 300
            })
        );
    }

    #[test]
    fn app_settings_default_has_denoise_enabled_and_no_workers() {
        let settings = AppSettings::default();
        assert!(settings.denoise_enabled);
        assert_eq!(settings.remote_workers.len(), 0);
    }

    #[test]
    fn add_update_remove_worker_round_trip() {
        let mut settings = AppSettings::default();
        settings.add_worker(WorkerSettings {
            name: "Laptop".to_string(),
            address: "192.168.1.50:9443".to_string(),
            ..WorkerSettings::default()
        });
        assert_eq!(settings.remote_workers.len(), 1);
        assert_eq!(settings.remote_workers[0].name, "Laptop");

        settings
            .update_worker(
                0,
                WorkerSettings {
                    name: "Laptop (renamed)".to_string(),
                    address: "192.168.1.50:9443".to_string(),
                    ..WorkerSettings::default()
                },
            )
            .unwrap();
        assert_eq!(settings.remote_workers[0].name, "Laptop (renamed)");

        settings.remove_worker(0).unwrap();
        assert_eq!(settings.remote_workers.len(), 0);
    }

    #[test]
    fn update_and_remove_worker_report_an_error_for_an_out_of_range_index() {
        let mut settings = AppSettings::default();
        assert!(
            settings
                .update_worker(0, WorkerSettings::default())
                .is_err()
        );
        assert!(settings.remove_worker(0).is_err());
    }

    #[test]
    fn toml_round_trip_preserves_remote_worker_settings_and_denoise_toggle() {
        let mut file = SettingsFile::default();
        file.settings.denoise_enabled = false;
        file.settings.add_worker(WorkerSettings {
            name: "Workstation".to_string(),
            address: "10.0.0.5:9443".to_string(),
            cert_dir: "C:/Users/me/gemray-certs".to_string(),
            transfer_mode: TransferMode::FinalOnly,
            cadence_ms: 250,
            preview_scale: PreviewScale::Custom(33),
        });

        let toml_str = toml::to_string_pretty(&file).expect("serialize");
        let parsed: SettingsFile = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed, file);
        assert!(!parsed.settings.denoise_enabled);
        assert_eq!(parsed.settings.remote_workers.len(), 1);
        assert_eq!(parsed.settings.remote_workers[0].cadence_ms, 250);
    }

    /// A settings file saved before remote rendering existed (no `remote_workers` key,
    /// no `denoise_enabled` key at all) must still load -- see `store::load_or_default`'s
    /// doc comment on never letting a settings file block startup.
    #[test]
    fn a_settings_file_predating_remote_rendering_still_parses_with_sensible_defaults() {
        let toml_str = "[settings]\nexposure = 1.1\n";
        let parsed: SettingsFile = toml::from_str(toml_str).expect("deserialize");
        assert!(parsed.settings.denoise_enabled);
        assert_eq!(parsed.settings.remote_workers.len(), 0);
    }

    /// A settings file saved before the sample-count slider existed -- still
    /// carrying the OLD `quality_preset` key, and no `target_samples` key at all --
    /// must still load with every other field intact, and `target_samples` defaulted.
    /// TOML deserialization silently drops unrecognized keys (there is no
    /// `#[serde(deny_unknown_fields)]` anywhere in this file), and `#[serde(default)]`
    /// on `AppSettings` defaults any field the document doesn't provide -- this test
    /// pins that those two facts actually compose the way `AppSettings::target_samples`'s
    /// doc comment claims, rather than assuming it.
    #[test]
    fn a_settings_file_with_the_old_quality_preset_key_still_loads_with_target_samples_defaulted() {
        let toml_str =
            "[settings]\nquality_preset = \"High / Quality\"\nexposure = 1.75\nmax_bounces = 20\n";
        let parsed: SettingsFile = toml::from_str(toml_str).expect("deserialize");
        assert_eq!(parsed.settings.target_samples, DEFAULT_TARGET_SAMPLES);
        // Fields the old file DID specify must still come through untouched -- the
        // obsolete key must not have knocked the whole document back to defaults.
        assert_eq!(parsed.settings.exposure, 1.75);
        assert_eq!(parsed.settings.max_bounces, 20);
    }

    /// Task: a settings file saved before the render-resolution setting existed (no
    /// `render_width`/`render_height` keys at all) must still load, defaulting both to
    /// `DEFAULT_RENDER_WIDTH`/`DEFAULT_RENDER_HEIGHT` -- same `#[serde(default)]`-on-
    /// the-struct mechanism `a_settings_file_with_the_old_quality_preset_key_still_loads_with_target_samples_defaulted`
    /// pins for `target_samples`, exercised here for the newer field.
    #[test]
    fn a_settings_file_predating_render_resolution_still_loads_with_800x600_defaulted() {
        let toml_str = "[settings]\nexposure = 1.2\n";
        let parsed: SettingsFile = toml::from_str(toml_str).expect("deserialize");
        assert_eq!(parsed.settings.render_width, DEFAULT_RENDER_WIDTH);
        assert_eq!(parsed.settings.render_height, DEFAULT_RENDER_HEIGHT);
        assert_eq!(parsed.settings.exposure, 1.2);
    }

    #[test]
    fn toml_round_trip_preserves_render_resolution() {
        let mut file = SettingsFile::default();
        file.settings.render_width = 1920;
        file.settings.render_height = 1080;

        let toml_str = toml::to_string_pretty(&file).expect("serialize");
        let parsed: SettingsFile = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.settings.render_width, 1920);
        assert_eq!(parsed.settings.render_height, 1080);
    }

    /// A corrupt `remote_workers` entry (e.g. a hand-edited file with an unknown
    /// `transfer_mode` variant) must not crash `toml::from_str` in a way that isn't
    /// already covered by `store::load_or_default`'s own corrupt-file fallback --
    /// exercised at the `store` layer, not duplicated here. This test instead pins
    /// that a WELL-FORMED but partially-specified worker entry (missing optional-ish
    /// fields) still deserializes, defaulting the rest via `WorkerSettings`'s own
    /// `#[serde(default)]`.
    #[test]
    fn a_partially_specified_worker_entry_defaults_its_missing_fields() {
        let toml_str = r#"
[settings]
exposure = 1.0

[[settings.remote_workers]]
name = "Partial"
address = "10.0.0.9:9443"
"#;
        let parsed: SettingsFile = toml::from_str(toml_str).expect("deserialize");
        assert_eq!(parsed.settings.remote_workers.len(), 1);
        let worker = &parsed.settings.remote_workers[0];
        assert_eq!(worker.name, "Partial");
        assert_eq!(worker.transfer_mode, TransferMode::LiveProgressive);
        assert_eq!(worker.cadence_ms, DEFAULT_WORKER_CADENCE_MS);
        assert_eq!(worker.preview_scale, PreviewScale::Full);
    }

    // ---- Local preview-then-settle rendering: LocalPreviewScale -----------------

    #[test]
    fn local_preview_scale_divisors_match_their_names() {
        assert_eq!(LocalPreviewScale::Off.divisor(), 1);
        assert_eq!(LocalPreviewScale::Half.divisor(), 2);
        assert_eq!(LocalPreviewScale::Quarter.divisor(), 4);
    }

    #[test]
    fn app_settings_default_has_local_preview_off() {
        assert_eq!(
            AppSettings::default().local_preview_scale,
            LocalPreviewScale::Off
        );
    }

    /// A settings file saved before this control existed (no `local_preview_scale`
    /// key at all) must still load, defaulting to `Off` -- same `#[serde(default)]`-
    /// on-the-struct mechanism `a_settings_file_predating_render_resolution_still_loads_with_800x600_defaulted`
    /// pins for `render_width`/`render_height`.
    #[test]
    fn a_settings_file_predating_local_preview_still_loads_with_it_off() {
        let toml_str = "[settings]\nexposure = 1.2\n";
        let parsed: SettingsFile = toml::from_str(toml_str).expect("deserialize");
        assert_eq!(parsed.settings.local_preview_scale, LocalPreviewScale::Off);
    }

    #[test]
    fn toml_round_trip_preserves_local_preview_scale() {
        let mut file = SettingsFile::default();
        file.settings.local_preview_scale = LocalPreviewScale::Quarter;

        let toml_str = toml::to_string_pretty(&file).expect("serialize");
        let parsed: SettingsFile = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(
            parsed.settings.local_preview_scale,
            LocalPreviewScale::Quarter
        );
    }

    // ---- Remote render sample budget: AppSettings::remote_render_samples --------

    #[test]
    fn app_settings_default_has_512_remote_render_samples() {
        assert_eq!(AppSettings::default().remote_render_samples, 512);
    }

    /// A settings file saved before this control existed (no `remote_render_samples`
    /// key at all) must still load, defaulting to the same `512` the constant used to
    /// be hardcoded to -- so an existing user's remote renders are unaffected by
    /// upgrading to a version of this app with this setting.
    #[test]
    fn a_settings_file_predating_remote_render_samples_still_loads_with_512_defaulted() {
        let toml_str = "[settings]\nexposure = 1.2\n";
        let parsed: SettingsFile = toml::from_str(toml_str).expect("deserialize");
        assert_eq!(
            parsed.settings.remote_render_samples,
            DEFAULT_REMOTE_RENDER_SAMPLES
        );
    }

    #[test]
    fn toml_round_trip_preserves_remote_render_samples() {
        let mut file = SettingsFile::default();
        file.settings.remote_render_samples = 4096;

        let toml_str = toml::to_string_pretty(&file).expect("serialize");
        let parsed: SettingsFile = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.settings.remote_render_samples, 4096);
    }

    // ---- HDR environment maps: AppSettings::env_map_path -------------------------

    #[test]
    fn app_settings_default_has_no_env_map_loaded() {
        assert_eq!(AppSettings::default().env_map_path, "");
    }

    /// A settings file saved before this control existed (no `env_map_path` key at
    /// all) must still load, defaulting to the empty ("no map, use the studio rig")
    /// string -- same `#[serde(default)]`-on-the-struct mechanism
    /// `a_settings_file_predating_local_preview_still_loads_with_it_off` pins for
    /// `local_preview_scale`.
    #[test]
    fn a_settings_file_predating_env_map_still_loads_with_none_loaded() {
        let toml_str = "[settings]\nexposure = 1.2\n";
        let parsed: SettingsFile = toml::from_str(toml_str).expect("deserialize");
        assert_eq!(parsed.settings.env_map_path, "");
    }

    #[test]
    fn toml_round_trip_preserves_env_map_path() {
        let mut file = SettingsFile::default();
        file.settings.env_map_path = "C:/hdri/studio.hdr".to_string();

        let toml_str = toml::to_string_pretty(&file).expect("serialize");
        let parsed: SettingsFile = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.settings.env_map_path, "C:/hdri/studio.hdr");
    }
}

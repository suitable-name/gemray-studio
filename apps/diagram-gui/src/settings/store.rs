//! Load/save the settings TOML file (Task: settings persistence).
//!
//! A settings file must never prevent the app from starting: every failure mode here
//! (missing file, unreadable file, corrupt/unparseable TOML) is caught and logged,
//! falling back to `SettingsFile::default()` rather than propagating an error.

use super::model::SettingsFile;
use std::{
    io,
    path::{Path, PathBuf},
};
use tracing::{info, warn};

const APP_DIR_NAME: &str = "diagram-gui";
const SETTINGS_FILE_NAME: &str = "settings.toml";

/// Resolves the settings file path in the platform config directory (not next to the
/// executable -- see the report for the reasoning: avoids requiring write access to
/// the install directory, e.g. under `Program Files` on Windows).
///
/// - Windows: `%APPDATA%\diagram-gui\settings.toml`
/// - macOS: `~/Library/Application Support/diagram-gui/settings.toml`
/// - Linux/other Unix: `$XDG_CONFIG_HOME/diagram-gui/settings.toml`, falling back to
///   `~/.config/diagram-gui/settings.toml`
///
/// If none of the expected environment variables are set (unusual, but not
/// impossible), falls back to a `diagram-gui/settings.toml` path relative to the
/// current working directory rather than failing outright.
#[must_use]
pub fn default_settings_path() -> PathBuf {
    platform_config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_DIR_NAME)
        .join(SETTINGS_FILE_NAME)
}

#[cfg(target_os = "windows")]
fn platform_config_dir() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(PathBuf::from)
}

#[cfg(target_os = "macos")]
fn platform_config_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join("Library")
            .join("Application Support")
    })
}

#[cfg(all(unix, not(target_os = "macos")))]
fn platform_config_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
}

/// Loads settings from `path`, falling back to defaults (with built-in presets) on
/// any failure -- missing file, unreadable file, or corrupt/unparseable TOML. Never
/// panics and never propagates an error: this is deliberately infallible so a broken
/// settings file can never block startup.
#[must_use]
pub fn load_or_default(path: &Path) -> SettingsFile {
    let mut file = match std::fs::read_to_string(path) {
        Ok(contents) => match toml::from_str::<SettingsFile>(&contents) {
            Ok(file) => file,
            Err(e) => {
                warn!(
                    "Settings file at {} is corrupt ({e}); falling back to defaults.",
                    path.display()
                );
                SettingsFile::default()
            }
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            info!(
                "No settings file at {} yet; using defaults.",
                path.display()
            );
            SettingsFile::default()
        }
        Err(e) => {
            warn!(
                "Could not read settings file at {} ({e}); falling back to defaults.",
                path.display()
            );
            SettingsFile::default()
        }
    };
    file.ensure_built_in_presets();
    file
}

/// Writes `settings` to `path` as pretty-printed TOML, creating the parent directory
/// if needed. Writes to a temporary sibling file and renames it into place so a crash
/// or power loss mid-write can never leave a half-written, corrupt settings file
/// behind -- the rename is the only step that can make the new content visible, and
/// `std::fs::rename` replaces an existing destination atomically on both Windows
/// (`MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`) and Unix.
pub fn save(path: &Path, settings: &SettingsFile) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let toml_str = toml::to_string_pretty(settings)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let tmp_path = path.with_extension("toml.tmp");
    std::fs::write(&tmp_path, toml_str)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::model::{AppSettings, LightingPreset};
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// A fresh, unique scratch directory under the OS temp dir, cleaned up when the
    /// returned guard drops. Avoids adding a `tempfile` dependency for what is, here,
    /// a handful of small round-trip tests.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "diagram-gui-settings-test-{tag}-{}-{n}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn missing_file_falls_back_to_defaults_without_panicking() {
        let dir = TempDir::new("missing");
        let path = dir.path().join("does-not-exist.toml");
        let loaded = load_or_default(&path);
        assert_eq!(loaded, SettingsFile::default());
    }

    #[test]
    fn corrupt_file_falls_back_to_defaults_without_panicking() {
        let dir = TempDir::new("corrupt");
        let path = dir.path().join("settings.toml");
        std::fs::write(&path, "this is { not [ valid toml at all").unwrap();
        let loaded = load_or_default(&path);
        assert_eq!(loaded, SettingsFile::default());
    }

    #[test]
    fn unreadable_directory_in_place_of_file_falls_back_to_defaults() {
        // A path that points at a directory, not a file, makes `read_to_string` fail
        // with something other than `NotFound` (it's `IsADirectory`/similar) -- this
        // exercises the third fallback arm distinctly from "missing" and "corrupt".
        let dir = TempDir::new("isdir");
        let path = dir.path().join("settings.toml");
        std::fs::create_dir_all(&path).unwrap();
        let loaded = load_or_default(&path);
        assert_eq!(loaded, SettingsFile::default());
    }

    #[test]
    fn partially_corrupt_file_keeps_valid_fields_and_defaults_missing_ones() {
        let dir = TempDir::new("partial");
        let path = dir.path().join("settings.toml");
        // Valid TOML, but missing most fields and with an extra unknown key -- must
        // still parse, defaulting everything not present.
        std::fs::write(
            &path,
            "[settings]\nexposure = 1.75\nsomething_unknown = true\n",
        )
        .unwrap();
        let loaded = load_or_default(&path);
        assert_eq!(loaded.settings.exposure, 1.75);
        assert_eq!(
            loaded.settings.target_samples,
            AppSettings::default().target_samples
        );
        // Built-ins still get filled in even though the file had no [[presets]] at all.
        assert_eq!(
            loaded.presets.len(),
            super::super::model::built_in_presets().len()
        );
    }

    #[test]
    fn save_then_load_round_trips_settings_and_presets() {
        let dir = TempDir::new("roundtrip");
        let path = dir.path().join("nested").join("settings.toml");

        let mut original = SettingsFile::default();
        original.settings.exposure = 1.65;
        original.settings.max_bounces = 20;
        original.settings.selected_material = "Ruby".to_string();
        original
            .upsert_user_preset(LightingPreset {
                name: "Golden Hour".to_string(),
                built_in: false,
                light_yaw_deg: 88.0,
                light_pitch_deg: 33.0,
                exposure: 1.15,
                lighting_rig: "D65 Daylight (5500K)".to_string(),
                camera_distance: 2.9,
            })
            .unwrap();

        save(&path, &original).expect("save should succeed, creating parent dirs");
        assert!(path.exists());

        let loaded = load_or_default(&path);
        assert_eq!(loaded, original);
    }

    #[test]
    fn save_overwrites_existing_file_atomically() {
        let dir = TempDir::new("overwrite");
        let path = dir.path().join("settings.toml");

        let first = SettingsFile::default();
        save(&path, &first).unwrap();

        let mut second = SettingsFile::default();
        second.settings.exposure = 2.0;
        save(&path, &second).unwrap();

        let loaded = load_or_default(&path);
        assert_eq!(loaded.settings.exposure, 2.0);
        // no leftover temp file
        assert!(!path.with_extension("toml.tmp").exists());
    }

    #[test]
    fn default_settings_path_is_non_empty_and_ends_with_settings_file_name() {
        let path = default_settings_path();
        assert!(path.to_string_lossy().ends_with("settings.toml"));
        assert!(path.components().any(|c| c.as_os_str() == APP_DIR_NAME));
    }
}

//! Remote-worker settings: [`PreviewScale`] (a worker's live preview-stream
//! resolution), [`LocalPreviewScale`] (the local preview-then-settle resolution
//! reduction), and [`WorkerSettings`] (one configured remote render worker).
//!
//! Split out of `settings::model` purely to keep that module (already sizeable) from
//! growing further.

use gemray_net::messages::{PreviewConfig, StreamConfig, TransferMode};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The reduced resolution a remote worker's `PREVIEW` stream is rendered at, relative
/// to the session's full render resolution -- a per-worker setting (the worker's own
/// link/hardware determines what's worth spending on a live low-res look) distinct from
/// [`WorkerSettings::transfer_mode`], which governs the FULL-resolution `FRAME` payload
/// instead. See `gemray_net::messages`'s module docs for why `PREVIEW` and `FRAME` are
/// not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PreviewScale {
    #[default]
    Full,
    Half,
    Quarter,
    /// A custom percentage of the session's full resolution, `1..=100`. Values outside
    /// that range are clamped by [`PreviewScale::percent`] rather than rejected --
    /// keeps a hand-edited or migrated settings file loadable (see
    /// `store::load_or_default`'s doc comment on never letting a settings file block
    /// startup) instead of falling back to a whole-file default over one bad number.
    Custom(u32),
}

impl PreviewScale {
    /// This scale as an integer percentage of the full render resolution, clamped to
    /// `1..=100` (a `0%` preview would be a zero-area request the worker's own
    /// validation -- `apps/gemray-worker/src/validate.rs` -- would have to reject).
    #[must_use]
    pub const fn percent(self) -> u32 {
        match self {
            Self::Full => 100,
            Self::Half => 50,
            Self::Quarter => 25,
            Self::Custom(p) => {
                if p == 0 {
                    1
                } else if p > 100 {
                    100
                } else {
                    p
                }
            }
        }
    }

    /// Resolves this scale against a session-wide `width x height` render resolution,
    /// floored at `1x1` so a worker is never asked for a zero-area preview regardless
    /// of how small `width`/`height` are.
    #[must_use]
    pub fn resolve(self, width: u32, height: u32) -> (u32, u32) {
        let pct = self.percent();
        let w = (width * pct / 100).max(1);
        let h = (height * pct / 100).max(1);
        (w, h)
    }
}

/// The resolution reduction applied while the camera is moving, for Task: local
/// preview-then-settle rendering -- the SAME reduced-resolution idea
/// [`PreviewScale`] above already offers a remote worker's live `PREVIEW` stream, but
/// applied to the LOCAL render loop instead (see
/// `bridge::local_preview::effective_dimensions`), and deliberately without a
/// [`PreviewScale::Custom`] equivalent: the settings-dialog control is a discrete pill
/// choice, not a slider -- there's no meaningful continuum between "half" and "0.47".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LocalPreviewScale {
    /// No reduction: the render loop always traces at the full configured
    /// `render_width`/`render_height`, regardless of whether the camera is moving --
    /// bit-identical to this crate's behaviour before this control existed. The
    /// default -- see [`super::app_settings::DEFAULT_LOCAL_PREVIEW_SCALE`]'s own doc
    /// comment for why.
    #[default]
    Off,
    Half,
    Quarter,
}

impl LocalPreviewScale {
    /// The integer divisor applied to each configured dimension while the camera is
    /// moving (see `bridge::local_preview::effective_dimensions`). `Off` divides by
    /// `1`, i.e. no change at all -- so a caller that forgets to special-case `Off`
    /// still gets the correct (unreduced) answer for free.
    #[must_use]
    pub const fn divisor(self) -> u32 {
        match self {
            Self::Off => 1,
            Self::Half => 2,
            Self::Quarter => 4,
        }
    }
}

/// Which engine(s) should contribute to the LIVE viewport once the camera settles --
/// the live-rendering analogue of `bridge::export_thread::ComputeTarget`, offered as the
/// same Local/Remote/Local+Remote choice (see `settings_dialog.slint`'s "Live Compute"
/// section) but kept as a separate type rather than shared with the export's: the two
/// features' callers already live in disjoint module trees (`bridge::render_thread`/
/// `gui::remote` here vs. `bridge::export_thread` there), matching this codebase's own
/// precedent of duplicating small pose/scene-shaping logic across that boundary rather
/// than coupling the two (see `gui::remote::orchestrator::scene_state_from_snapshot`'s
/// doc comment for the same reasoning applied to a different function).
///
/// Unlike the export dialog's `compute_target` (re-derived fresh every dialog open, never
/// persisted), this IS a persisted `AppSettings` field: the live viewport's choice is a
/// standing preference, not a one-shot dialog default. `Both` is deliberately the
/// `Default` -- see [`super::app_settings::DEFAULT_LIVE_COMPUTE_TARGET`]'s own doc
/// comment for why that default is safe even before any worker is ever configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LiveComputeTarget {
    LocalOnly,
    RemoteOnly,
    #[default]
    Both,
}

/// The default per-request cadence a newly added [`WorkerSettings`] starts with, before
/// any clamping to a specific worker's advertised `Welcome::min_cadence_ms` floor --
/// see [`WorkerSettings::effective_cadence_ms`].
pub const DEFAULT_WORKER_CADENCE_MS: u32 = 500;

/// One configured remote render worker: the connection details and per-request
/// preferences for a specific machine and link.
///
/// Deliberately does NOT include render resolution (`width`/`height`) -- that stays
/// session-wide (set once, shared by every worker AND the local CPU path), since every
/// worker in a session must trace identical dimensions or the summed samples don't
/// compose. See `crates/gemray-net`'s crate docs on additive sample partitioning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerSettings {
    /// User-facing label for this worker in the list UI. Not required to be unique --
    /// the worker-list panel addresses entries by their position in
    /// `AppSettings::remote_workers`, same as the lighting-preset and sync-source
    /// lists elsewhere in this module.
    pub name: String,
    /// `host:port` this worker listens on (`gemray-worker serve --bind`/`--allow-remote`).
    pub address: String,
    /// Path to the directory holding this worker's mutual-TLS certificate bundle --
    /// `ca.pem`, `client.pem`, and `client.key`, exactly as `gemray-worker cert
    /// issue-client` produces them. See [`WorkerSettings::ca_path`] /
    /// [`WorkerSettings::client_cert_path`] / [`WorkerSettings::client_key_path`].
    ///
    /// A plain `String`, not `PathBuf` -- matches every other user-facing text field in
    /// this file (`selected_material`, `lighting_rig`, ...) and keeps this struct
    /// trivially `PartialEq`/TOML-serializable without a path-specific serde adapter.
    pub cert_dir: String,
    pub transfer_mode: TransferMode,
    /// This worker's requested cadence, in milliseconds, before clamping to its own
    /// advertised floor -- see [`WorkerSettings::effective_cadence_ms`].
    pub cadence_ms: u32,
    pub preview_scale: PreviewScale,
}

impl Default for WorkerSettings {
    fn default() -> Self {
        Self {
            name: String::new(),
            address: String::new(),
            cert_dir: String::new(),
            transfer_mode: TransferMode::LiveProgressive,
            cadence_ms: DEFAULT_WORKER_CADENCE_MS,
            preview_scale: PreviewScale::Full,
        }
    }
}

impl WorkerSettings {
    #[must_use]
    pub fn ca_path(&self) -> PathBuf {
        Path::new(&self.cert_dir).join("ca.pem")
    }

    #[must_use]
    pub fn client_cert_path(&self) -> PathBuf {
        Path::new(&self.cert_dir).join("client.pem")
    }

    #[must_use]
    pub fn client_key_path(&self) -> PathBuf {
        Path::new(&self.cert_dir).join("client.key")
    }

    /// Clamps [`WorkerSettings::cadence_ms`] to `min_cadence_ms` -- a specific worker's
    /// own advertised floor, from its `Welcome::min_cadence_ms` (see that field's doc
    /// comment: requesting faster than a worker can usefully deliver just means it
    /// does its best via delta coalescing anyway, but a viewer UI clamping up front
    /// avoids the false impression of a faster cadence than will ever actually be
    /// observed).
    #[must_use]
    pub const fn effective_cadence_ms(&self, min_cadence_ms: u32) -> u32 {
        if self.cadence_ms < min_cadence_ms {
            min_cadence_ms
        } else {
            self.cadence_ms
        }
    }

    /// Builds the [`StreamConfig`] this worker's settings imply for a render at the
    /// session's `width x height` resolution, against a specific worker's advertised
    /// `min_cadence_ms` (from its `Welcome`).
    #[must_use]
    pub fn stream_config(&self, min_cadence_ms: u32, width: u32, height: u32) -> StreamConfig {
        let (preview_width, preview_height) = self.preview_scale.resolve(width, height);
        StreamConfig {
            transfer_mode: self.transfer_mode,
            cadence_ms: self.effective_cadence_ms(min_cadence_ms),
            preview: Some(PreviewConfig {
                width: preview_width,
                height: preview_height,
            }),
        }
    }
}

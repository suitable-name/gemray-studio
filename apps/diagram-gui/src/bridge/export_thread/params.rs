//! Export request validation and the default output path.
//!
//! Split out of `bridge::export_thread` purely to keep that module (already sizeable)
//! from growing further.

use std::path::PathBuf;

/// Sane bounds for user-supplied export dimensions. `MAX_EXPORT_DIM` exists
/// specifically so nobody can point the exporter at, say, 100000x100000 and lock up
/// (or OOM) the machine -- see `validate_export_params`'s tests for exactly that case.
pub const MIN_EXPORT_DIM: u32 = 16;
pub const MAX_EXPORT_DIM: u32 = 8192;
pub const MIN_EXPORT_SPP: u32 = 1;
/// Deliberately generous, not "whatever the current GPU can chew through by lunchtime":
/// at 4K (3840x2160), 32768 spp is ~272 billion spectral paths. Measured GPU throughput
/// on this project's integrated AMD Radeon is ~23 million samples/second, so that
/// combination is roughly 3+ hours there -- much less on a discrete card, far more on
/// CPU-only builds where the `gpu` feature is off. The cap only exists to stop a typo
/// (an extra zero) from locking up the machine, not to second-guess a deliberate choice
/// to run it overnight.
pub const MAX_EXPORT_SPP: u32 = 32768;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportParams {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
}

/// Which engine(s) an export should use -- the export dialog's "Compute" pill, matching
/// `export_dialog.slint`'s `compute_target` property (0/1/2, see that file's own doc
/// comment) and the top-level task's requirement to expose this as an explicit choice
/// rather than only ever combining automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeTarget {
    LocalOnly,
    RemoteOnly,
    /// Both engines trace disjoint sample ranges concurrently -- see
    /// `export_thread::remote`'s and `run_export`'s doc comments. The default whenever a
    /// worker advertising render capacity is configured (decided in
    /// `gui::render_export`, mirrored by `export_dialog.slint`'s own default binding).
    Both,
}

/// Validates raw (Slint-supplied, hence signed) export request fields, rejecting zero,
/// negative, and absurdly large values before anything is allocated or a thread is
/// spawned. Pure and side-effect-free so it's directly unit-testable.
pub fn validate_export_params(
    width: i32,
    height: i32,
    samples_per_pixel: i32,
) -> Result<ExportParams, String> {
    if width <= 0 || height <= 0 || samples_per_pixel <= 0 {
        return Err("Width, height, and sample count must all be positive.".to_string());
    }
    let width = width as u32;
    let height = height as u32;
    let samples_per_pixel = samples_per_pixel as u32;

    if width < MIN_EXPORT_DIM || height < MIN_EXPORT_DIM {
        return Err(format!(
            "Minimum export size is {MIN_EXPORT_DIM}x{MIN_EXPORT_DIM} px."
        ));
    }
    if width > MAX_EXPORT_DIM || height > MAX_EXPORT_DIM {
        return Err(format!(
            "Maximum export size is {MAX_EXPORT_DIM}x{MAX_EXPORT_DIM} px (requested {width}x{height})."
        ));
    }
    if samples_per_pixel < MIN_EXPORT_SPP {
        return Err("Sample count must be at least 1.".to_string());
    }
    if samples_per_pixel > MAX_EXPORT_SPP {
        return Err(format!(
            "Maximum sample count is {MAX_EXPORT_SPP} samples per pixel."
        ));
    }

    Ok(ExportParams {
        width,
        height,
        samples_per_pixel,
    })
}

/// Builds the default export output path: `exports/gem_export_{width}x{height}_{spp}spp_{unix_seconds}.png`,
/// next to the executable's working directory -- matching the existing convention in
/// `gui::detail::export_diagram_file`, which writes attachment exports to `./exports/`.
#[must_use]
pub fn default_export_path(params: ExportParams) -> PathBuf {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    PathBuf::from("exports").join(format!(
        "gem_export_{}x{}_{}spp_{timestamp}.png",
        params.width, params.height, params.samples_per_pixel
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_export_params_rejects_zero_and_negative() {
        assert!(validate_export_params(0, 1080, 64).is_err());
        assert!(validate_export_params(1920, 0, 64).is_err());
        assert!(validate_export_params(1920, 1080, 0).is_err());
        assert!(validate_export_params(-1920, 1080, 64).is_err());
        assert!(validate_export_params(1920, -1080, 64).is_err());
        assert!(validate_export_params(1920, 1080, -1).is_err());
    }

    #[test]
    fn validate_export_params_rejects_absurdly_large_requests() {
        let err = validate_export_params(100_000, 100_000, 64).unwrap_err();
        assert!(err.contains("Maximum export size"));
    }

    #[test]
    fn validate_export_params_rejects_excessive_sample_counts() {
        let err = validate_export_params(1920, 1080, 1_000_000).unwrap_err();
        assert!(err.contains("Maximum sample count"));
    }

    #[test]
    fn validate_export_params_rejects_tiny_dimensions_below_minimum() {
        assert!(validate_export_params(1, 1, 64).is_err());
    }

    #[test]
    fn validate_export_params_accepts_sensible_presets() {
        assert_eq!(
            validate_export_params(1920, 1080, 256).unwrap(),
            ExportParams {
                width: 1920,
                height: 1080,
                samples_per_pixel: 256
            }
        );
        assert_eq!(
            validate_export_params(3840, 2160, 64).unwrap(),
            ExportParams {
                width: 3840,
                height: 2160,
                samples_per_pixel: 64
            }
        );
    }

    #[test]
    fn validate_export_params_accepts_boundary_values() {
        assert!(
            validate_export_params(
                MIN_EXPORT_DIM as i32,
                MIN_EXPORT_DIM as i32,
                MIN_EXPORT_SPP as i32
            )
            .is_ok()
        );
        assert!(
            validate_export_params(
                MAX_EXPORT_DIM as i32,
                MAX_EXPORT_DIM as i32,
                MAX_EXPORT_SPP as i32
            )
            .is_ok()
        );
    }

    #[test]
    fn default_export_path_lives_under_exports_and_names_dimensions() {
        let path = default_export_path(ExportParams {
            width: 1920,
            height: 1080,
            samples_per_pixel: 64,
        });
        let s = path.to_string_lossy();
        assert!(s.contains("exports"));
        assert!(s.contains("1920x1080"));
        assert!(s.contains("64spp"));
        assert!(s.ends_with(".png"));
    }
}

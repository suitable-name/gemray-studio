//! Converts a 19-sample tilt-angle curve (one value per 5° step, 0..=90°) into an SVG
//! path-data string for `Path::commands` in `performance_graph_dialog.slint`. Slint has
//! no polyline primitive, so the dialog draws each optical channel (brilliance,
//! windowing, extinction) as a `Path` scaled into a fixed 0..100 x 0..100 viewbox --
//! see that file's "High-Resolution 19-Column Performance Chart Canvas" section.

use std::fmt::Write as _;

/// Number of tilt-angle samples (0°, 5°, ..., 90°) -- matches `graph_brilliance` and
/// friends in `app.slint` / `gem_viewport.slint` / `performance_graph_dialog.slint`.
const SAMPLE_COUNT: usize = 19;

/// Builds an `M x y L x y L x y ...` SVG path string from `values`, laid out in a
/// 0..100 x 0..100 viewbox. `x` sweeps evenly across the 19 samples so the first and
/// last points sit exactly on the box's left/right edges; `y` is `100 - value` so a
/// value of 100 lands at the *top* of the box (`viewbox-height` grows downward in
/// Slint's `Path`, the opposite of the percentage-of-brilliance sense of `values`).
///
/// `values` are clamped to 0..100 before conversion. They're percentages and should
/// already be in range, but a stray out-of-range sample must not draw outside the
/// chart's clipped `Rectangle`.
pub fn tilt_curve_path(values: &[f32; SAMPLE_COUNT]) -> String {
    let mut commands = String::new();
    for (i, &value) in values.iter().enumerate() {
        if i > 0 {
            commands.push(' ');
        }
        let x = i as f32 * 100.0 / (SAMPLE_COUNT - 1) as f32;
        let y = 100.0 - value.clamp(0.0, 100.0);
        let cmd = if i == 0 { 'M' } else { 'L' };
        let _ = write!(commands, "{cmd} {x:.2} {y:.2}");
    }
    commands
}

#[cfg(test)]
mod tests {
    use super::{SAMPLE_COUNT, tilt_curve_path};

    #[test]
    fn known_input_produces_expected_path() {
        let values = [0.0f32; SAMPLE_COUNT];
        let path = tilt_curve_path(&values);
        assert_eq!(
            path,
            "M 0.00 100.00 L 5.56 100.00 L 11.11 100.00 L 16.67 100.00 L 22.22 100.00 \
             L 27.78 100.00 L 33.33 100.00 L 38.89 100.00 L 44.44 100.00 L 50.00 100.00 \
             L 55.56 100.00 L 61.11 100.00 L 66.67 100.00 L 72.22 100.00 L 77.78 100.00 \
             L 83.33 100.00 L 88.89 100.00 L 94.44 100.00 L 100.00 100.00"
        );
    }

    #[test]
    fn first_and_last_points_span_full_width() {
        let values = [42.0f32; SAMPLE_COUNT];
        let path = tilt_curve_path(&values);
        let tokens: Vec<&str> = path.split_whitespace().collect();
        // Each point is 3 tokens (command, x, y); 19 points -> 57 tokens.
        assert_eq!(tokens.len(), SAMPLE_COUNT * 3);
        assert_eq!(tokens[1], "0.00", "first sample must sit at x=0");
        assert_eq!(
            tokens[tokens.len() - 2],
            "100.00",
            "last sample must sit at x=100"
        );
    }

    #[test]
    fn value_100_maps_to_top_and_0_maps_to_bottom() {
        let mut values = [50.0f32; SAMPLE_COUNT];
        values[0] = 100.0;
        values[1] = 0.0;
        let path = tilt_curve_path(&values);
        let tokens: Vec<&str> = path.split_whitespace().collect();
        // First point is tokens[0..3] = ["M", x, y]; second is tokens[3..6] = ["L", x, y].
        assert_eq!(tokens[2], "0.00", "value 100 must map to y=0 (top)");
        assert_eq!(tokens[5], "100.00", "value 0 must map to y=100 (bottom)");
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        let mut values = [50.0f32; SAMPLE_COUNT];
        values[0] = 150.0; // above the 0..100 range
        values[1] = -20.0; // below the 0..100 range
        let path = tilt_curve_path(&values);
        let tokens: Vec<&str> = path.split_whitespace().collect();
        assert_eq!(tokens[2], "0.00", "150 must clamp to 100 -> y=0");
        assert_eq!(tokens[5], "100.00", "-20 must clamp to 0 -> y=100");
    }
}

use crate::{MainWindow, gui::show_toast};
use slint::{ComponentHandle, Model, SharedString};
use std::fmt::Write as _;
use tracing::info;

pub fn copy_to_clipboard(text: &str) -> bool {
    info!("Copying text to clipboard ({} chars)", text.len());

    #[cfg(target_os = "linux")]
    {
        // Try xclip first
        if let Ok(mut child) = std::process::Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(std::process::Stdio::piped())
            .spawn()
            && let Some(mut stdin) = child.stdin.take()
        {
            use std::io::Write;
            let _ = stdin.write_all(text.as_bytes());
            let _ = child.wait();
            return true;
        }

        // Try wl-copy (Wayland)
        if let Ok(mut child) = std::process::Command::new("wl-copy")
            .stdin(std::process::Stdio::piped())
            .spawn()
            && let Some(mut stdin) = child.stdin.take()
        {
            use std::io::Write;
            let _ = stdin.write_all(text.as_bytes());
            let _ = child.wait();
            return true;
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(mut child) = std::process::Command::new("clip")
            .stdin(std::process::Stdio::piped())
            .spawn()
            && let Some(mut stdin) = child.stdin.take()
        {
            use std::io::Write;
            let _ = stdin.write_all(text.as_bytes());
            let _ = child.wait();
            return true;
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(mut child) = std::process::Command::new("pbcopy")
            .stdin(std::process::Stdio::piped())
            .spawn()
            && let Some(mut stdin) = child.stdin.take()
        {
            use std::io::Write;
            let _ = stdin.write_all(text.as_bytes());
            let _ = child.wait();
            return true;
        }
    }

    false
}

/// Axis labels for the three non-canonical rows [`build_curve_data_csv`] appends --
/// same order as `gui::tilt_profile`'s `PROFILE_AZIMUTHS_DEG[1..]` and
/// `performance_graph_dialog.slint`'s `axis_labels[1..]`.
const EXTRA_AXIS_LABELS: [&str; 3] = ["45", "90 (width)", "135"];

/// Builds the "Copy 19-Point Data Table (All 4 Axes)" CSV: the canonical (0°) axis's
/// 19 measured samples, plus the three non-canonical axes' whenever `gui::
/// tilt_profile`'s background sweep has landed (it's lazy -- see that module's doc
/// comment). Split out of [`setup_copy_callbacks`] purely to keep that function under
/// clippy's function-length lint. This is the measured-numbers escape hatch the
/// dialog's honesty caption points to: unlike the chart and hover tooltip, every row
/// here is one of the 19 actually-raytraced 5°-step samples, not an interpolation.
fn build_curve_data_csv(ui: &MainWindow) -> String {
    let gb = ui.get_graph_brilliance();
    let ge = ui.get_graph_extinction();
    let gw = ui.get_graph_windowing();
    let mut csv = String::from("Tilt Angle (°),Axis,Brilliance (%),Windowing (%),Extinction (%)\n");
    for i in 0..19 {
        let angle = i * 5;
        let b = gb.row_data(i).unwrap_or(0.0);
        let w = gw.row_data(i).unwrap_or(0.0);
        let e = ge.row_data(i).unwrap_or(0.0);
        let _ = writeln!(csv, "{angle},0 (length),{b:.1},{w:.1},{e:.1}");
    }

    // Still exports the canonical axis above either way, so the button never produces
    // an empty/partial-looking table while the extra axes compute.
    let extra_brilliance = ui.get_graph_brilliance_extra_axes();
    let extra_extinction = ui.get_graph_extinction_extra_axes();
    let extra_windowing = ui.get_graph_windowing_extra_axes();
    if extra_brilliance.row_count() >= 3
        && extra_extinction.row_count() >= 3
        && extra_windowing.row_count() >= 3
    {
        for (axis_idx, label) in EXTRA_AXIS_LABELS.iter().enumerate() {
            let Some(axis_b) = extra_brilliance.row_data(axis_idx) else {
                continue;
            };
            let Some(axis_w) = extra_windowing.row_data(axis_idx) else {
                continue;
            };
            let Some(axis_e) = extra_extinction.row_data(axis_idx) else {
                continue;
            };
            for i in 0..19 {
                let angle = i * 5;
                let b = axis_b.row_data(i).unwrap_or(0.0);
                let w = axis_w.row_data(i).unwrap_or(0.0);
                let e = axis_e.row_data(i).unwrap_or(0.0);
                let _ = writeln!(csv, "{angle},{label},{b:.1},{w:.1},{e:.1}");
            }
        }
    }

    csv
}

/// Wires up the clipboard-copy callbacks (metrics, curve data, generic text, cutting
/// table, single cutting row). Split out of `run_gui` purely to keep that function
/// under clippy's function-length lint.
pub(super) fn setup_copy_callbacks(ui: &MainWindow) {
    let ui_weak_metrics = ui.as_weak();
    ui.on_copy_metrics(move || {
        if let Some(ui) = ui_weak_metrics.upgrade() {
            let b = ui.get_brilliance_pct();
            let f = ui.get_fire_index();
            let w = ui.get_windowing_pct();
            let text = format!(
                "Optical Metrics:\n• Brilliance: {b:.1}%\n• Fire Index: {f:.2}\n• Windowing: {w:.1}%"
            );
            copy_to_clipboard(&text);
            show_toast(&ui, "Optical metrics copied to clipboard!", "success");
        }
    });

    // Copy Performance Curve Data callback (19-Point Tilt Table, all 4 tilt axes) --
    // see `build_curve_data_csv`.
    let ui_weak_curve = ui.as_weak();
    ui.on_copy_curve_data(move || {
        if let Some(ui) = ui_weak_curve.upgrade() {
            let csv = build_curve_data_csv(&ui);
            copy_to_clipboard(&csv);
            show_toast(
                &ui,
                "19-Point performance curve data (4 axes) copied to clipboard!",
                "success",
            );
        }
    });

    let ui_weak_copy = ui.as_weak();
    ui.on_copy_text(move |text: SharedString| {
        if let Some(ui) = ui_weak_copy.upgrade() {
            copy_to_clipboard(&text);
            show_toast(&ui, "Copied to clipboard!", "success");
        }
    });

    let ui_weak_sched = ui.as_weak();
    ui.on_copy_cutting_table(move || {
        if let Some(ui) = ui_weak_sched.upgrade() {
            let angles_model = ui.get_current_angles();
            let mut text = String::from("#\tFacet\tAngle\tIndex\tNotes\n");
            for i in 0..angles_model.row_count() {
                if let Some(row) = angles_model.row_data(i) {
                    let _ = writeln!(
                        text,
                        "{}\t{}\t{}\t{}\t{}",
                        row.order_idx + 1,
                        row.facet,
                        row.angle,
                        row.index_val,
                        row.notes
                    );
                }
            }
            copy_to_clipboard(&text);
            show_toast(&ui, "Cutting schedule copied as TSV!", "success");
        }
    });

    let ui_weak_row = ui.as_weak();
    ui.on_copy_cutting_row(move |idx: i32| {
        if let Some(ui) = ui_weak_row.upgrade() {
            let angles_model = ui.get_current_angles();
            if let Some(row) = angles_model.row_data(idx as usize) {
                let text = format!(
                    "Facet: {}, Angle: {}, Index: {}, Notes: {}",
                    row.facet, row.angle, row.index_val, row.notes
                );
                copy_to_clipboard(&text);
                show_toast(&ui, &format!("Copied step #{}", idx + 1), "success");
            }
        }
    });
}

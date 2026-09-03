// No console window in a release build. Without this, launching the `.exe` directly
// (from Explorer, a shortcut, or the Start menu) opens an empty console alongside the
// window, because Rust binaries default to the Windows CONSOLE subsystem.
//
// `not(debug_assertions)` rather than unconditional: a debug build keeps its console so
// `cargo run` still shows panics and any `tracing` output during development. Note that
// this crate installs no `tracing` subscriber at all today, so a release build currently
// discards nothing by hiding the console -- but if one is ever added here, it must write
// somewhere other than stdout (a file, or the Windows event log) to survive this.
//
// A no-op on every non-Windows target: the attribute is Windows-specific and simply
// ignored elsewhere.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> anyhow::Result<()> {
    diagram_gui::gui::main()
}

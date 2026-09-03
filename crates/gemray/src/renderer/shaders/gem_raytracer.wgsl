// QUARANTINED -- this shader is dead scaffolding, not a working GPU renderer.
//
// It never compiled: it called `fmod(...)` to wrap a wavelength into range, but `fmod`
// is not a WGSL builtin (WGSL uses the `%` operator). `GemRaytracerPipeline::new`
// (renderer/pipeline.rs) `include_str!`s this file, so `cargo check` never sees the
// parse failure -- it would only surface as a panic out of `create_shader_module` the
// first time something actually constructed the pipeline. Nothing in this workspace
// does: `GemRaytracerPipeline` is never instantiated anywhere, the `gpu` feature is
// `default = []` and no workspace member enables it, and this file was never wired up
// to the CPU renderer's tests.
//
// Beyond not parsing, this was a transcription of an early design document, not of the
// current CPU renderer (`optics::raytracer`), and it encodes several physics bugs that
// have since been fixed on the CPU side:
//   - the old symmetric-Gaussian CIE 1931 CMF fit (see `color::cie1931::cie_1931_cmf`
//     for the corrected Wyman/Sloan/Shirley piecewise-asymmetric fit)
//   - no interior-ray exit-facet handling
//   - Fresnel applied twice on the same interface
//   - inverted Beer-Lambert absorption
//   - no spectral MIS weighting
//
// Do not "fix" this file in place -- e.g. swapping `fmod` for `%` would make it parse
// but would not make it correct, and would relegitimize a design that has since moved
// on. Any real GPU port must be a fresh translation of the CURRENT `optics::raytracer`
// (crates/gemray/src/optics/raytracer.rs), validated against it with a CPU/GPU
// equivalence harness, rather than a repair of this file.
//
// See also: `renderer/buffers.rs`'s `DispersionParams` doc comment for a layout bug
// (Rust struct offsets vs. this file's WGSL uniform layout) that any such port would
// also need to fix by regenerating the buffer structs with a layout self-test.

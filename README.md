# Gemray Studio

A physically-based spectral gemstone renderer and faceting-design library, in
Rust. It's for anyone who wants to see what a real cut gemstone actually looks
like from its real cutting schedule — spectral fire and polarization included,
not the RGB/scalar-IOR approximation every conventional renderer reaches for —
and for anyone building their own library of faceting designs (`.asc` files)
around one.

## What makes it technically interesting

At the center is `gemray`, a spectral path tracer that traces real gemstone
geometry rather than approximating it:

- **8-channel stratified hero-wavelength spectral sampling (HWSS)** — each ray
  carries 8 wavelengths at once (one "hero" plus 7 rotated companions),
  avoiding both the color-banding of naive RGB rendering and the noise of
  fully independent per-wavelength sampling.
- **Full Stokes–Mueller polarized light transport** — 4D Stokes vectors and
  Mueller matrices for Fresnel reflection/transmission, total-internal-
  reflection phase retardation, and Brewster-angle extinction, not a scalar
  reflectance approximation.
- **Sellmeier / Cauchy dispersion** — continuous, per-wavelength refractive
  index from real dispersion equations, not three fixed RGB indices.
- **Beer–Lambert absorption with pleochroism** — directional absorption
  tensors so a pleochroic stone's color genuinely depends on the
  electric-field direction relative to the crystal's optical axes, not just
  on wavelength.
- **An optional GPU megakernel** — `gemray`'s `gpu` feature compiles a
  complete WGSL port of the spectral transport physics, verified against the
  CPU tracer by a tiered equivalence harness: bit-exact integer checks,
  per-function ULP budgets (every ported function currently measures max
  genuine ULP = 0), an analytically-computable furnace anchor, and
  statistical comparison of real rendered images. See
  [`crates/gemray/docs/gpu.md`](crates/gemray/docs/gpu.md).

## Workspace map

| Crate / app | What it is |
|---|---|
| [`crates/gemray`](crates/gemray/README.md) | The spectral path tracer — the physics core. Facet planes + material in, rendered pixels and/or brilliance/fire/scintillation metrics out. |
| [`crates/gemray-net`](crates/gemray-net/README.md) | Wire protocol between a viewer and a remote server: offloading `gemray` sample tracing, and reading a remote design library. Types and framing only — no sockets. |
| [`crates/lapidary`](crates/lapidary/README.md) | Reader/writer for GemCAD-style `.asc` cutting-schedule files. Zero runtime dependencies. |
| [`crates/diagram-catalog`](crates/diagram-catalog/README.md) | Local SQLite-backed design library — models, storage, and local `.asc` import/export. |
| [`apps/diagram-gui`](apps/diagram-gui/README.md) | The desktop viewer, built with [Slint](https://slint.dev/): browse, search, and render your library in 3D. |
| [`apps/gemray-worker`](apps/gemray-worker/README.md) | Headless render CLI and remote server — serves a design library over mutual TLS, and optionally accepts render requests too. |

## Build and run

```
git clone https://github.com/<your-username>/gemray-studio.git
cd gemray-studio
cargo build --workspace
cargo run -p diagram-gui
```

`cargo run -p diagram-gui` opens the viewer; it looks for (and creates, if
missing) `facet_diagrams.sqlite` in whatever directory you launch it from —
see [The design library is yours to build](#the-design-library-is-yours-to-build)
below.

To render a scene headlessly instead of opening the GUI:

```
cargo run -p gemray-worker --features worker -- render --scene scene.json --out render.png --width 1920 --height 1080 --samples 256
```

`--features worker` is required — a default `gemray-worker` build compiles
neither `gemray` nor the render path in at all (see
[Feature flags](#feature-flags) below). The `render` subcommand still exists
without it, but running it just prints an error telling you to rebuild with
`--features worker`. See
[`apps/gemray-worker`'s README](apps/gemray-worker/README.md) for how to
produce a `scene.json`, and for the `serve`/`cert` subcommands that let a
second machine's GPU or CPU help render over the network.

### Testing

```
cargo test --workspace                        # everything EXCEPT gemray's gpu feature
cargo test -p gemray --features gpu            # the gpu-feature tests the line above skips
```

**`cargo test --workspace` does not build or exercise `gemray`'s `gpu` feature
at all** — a real bug once hid in exactly that gap for a long time. If you've
touched anything under `gemray`'s `renderer::gpu`, `renderer::env_map_gpu`, or
`renderer::pipeline`, or anything else `#[cfg(feature = "gpu")]`, run the
second command too. `apps/diagram-gui` and `apps/gemray-worker` each have
their own `gpu` feature that forwards to `gemray/gpu`; the same rule applies
to those (`cargo test -p diagram-gui --features gpu`,
`cargo test -p gemray-worker --features gpu`).

Neither of those substitutes for `gemray`'s GPU equivalence harness, which
needs a real GPU adapter and isn't a `cargo test` target at all:

```
cargo run --profile probe -p gemray --features gpu --example gpu_equivalence_harness
```

See [`crates/gemray/docs/gpu.md`](crates/gemray/docs/gpu.md) for what each of
its verification tiers checks.

### Optimized builds

`cargo build --release` is fine for everyday use. For distribution builds there are
two profile-guided scripts, which train on `gemray`'s `pgo_train` example (CPU tracer,
denoiser, tone-map, meet-point solver) and then rebuild both applications against the
merged profile:

| Script | Platform | Does |
|---|---|---|
| [`scripts/pgo-build.ps1`](scripts/pgo-build.ps1) | Windows (PowerShell) | PGO, over AVX2/scalar x GPU/CPU |
| [`scripts/pgo-bolt-build.sh`](scripts/pgo-bolt-build.sh) | Linux or Windows (bash) | PGO over AVX-512/AVX2/scalar x GPU/CPU, plus BOLT on Linux |

Each build is emitted as `<binary>-<os>-<isa>-<gpu>`, so the variants can coexist in
one output directory. The ISA tier is a compile-time `-C target-cpu` baseline and is
independent of `gemray`'s runtime SIMD dispatch, which detects AVX-512/AVX2 on any
build unless `GEMRAY_SIMD` caps it; the training run sets that variable to match the
tier being built.

BOLT runs on Linux only — it has no PE/COFF backend — and only on `gemray-worker`,
which has a headless, deterministic `render` workload to profile. `diagram-gui` is an
interactive desktop app with no scriptable workload, and a startup-only profile would
give it a worse code layout than no BOLT at all.

## Feature flags

All of the following are **off by default**.

| Crate | Feature | Adds |
|---|---|---|
| `gemray` | `gpu` | `renderer::gpu`: the verified GPU port of the spectral transport physics and its equivalence harness. Pulls in `wgpu`, `pollster`. |
| `gemray` | `hdr` | Radiance `.hdr` equirectangular environment-map *decoding* for `renderer::env_map`. Pulls in `image` (default features off, `hdr` format only). |
| `gemray` | `serde` | `Serialize`/`Deserialize` on the scene-description types (`GemMaterial`, `GpuFacetPlane`, `LightingPreset`) that `gemray-net`'s wire protocol needs. |
| `gemray-net` | `render` | `SceneState`/`RenderRequest` and everything else that needs `gemray`'s resolved scene/material types. Off by default so a library-only `gemray-worker` build never compiles `gemray` in at all. |
| `apps/diagram-gui` | `gpu` | Routes the viewport's progressive accumulation and the high-resolution export worker through `gemray`'s GPU megakernel, falling back to the CPU tracer per frame whenever it declines. See [`apps/diagram-gui/docs/gpu.md`](apps/diagram-gui/docs/gpu.md). |
| `apps/gemray-worker` | `worker` | Render capacity: `RenderRequest` handling on `serve`, and the `render` subcommand actually working. Turns on `gemray-net/render`. Without it, `gemray-worker` serves a read-only design library only. |
| `apps/gemray-worker` | `gpu` | Implies `worker`. Routes `render`/`serve` tracing through `gemray`'s GPU megakernel, falling back to the CPU tracer whenever it declines. |

## Screenshots

...

## The design library is yours to build

The SQLite design catalogue (`facet_diagrams.sqlite`) that `diagram-gui` and
`gemray-worker` open is **your own data, and it is not part of this
repository** — `.gitignore` excludes `*.sqlite`/`*.sqlite2`/`*.sqlite.bak`, so
nothing of the kind is tracked, shipped, or downloadable from here. There is
no starter catalogue bundled with this project. You build your own library
entirely by importing your own `.asc` cutting-schedule files, either through
`diagram-gui`'s Import button (see that app's README) or by calling
`diagram_catalog::local::import_asc` directly.

## Documentation

Per-crate documentation is linked from the workspace map above — start with a
crate's own README, then its `docs/` folder if it has one:

- [`crates/gemray/docs/`](crates/gemray/docs/) — the physics's deliberate
  deviations and known simplifications, and the GPU equivalence harness.
- [`apps/diagram-gui/docs/`](apps/diagram-gui/docs/) — the settings file
  format, the preview-then-handoff remote-rendering model, import/export, and
  the `gpu` feature's fallback rules.
- [`apps/gemray-worker/docs/`](apps/gemray-worker/docs/) — the trust model
  behind its mutual-TLS server, and its internal architecture.

## License

MIT — see the `license` field in the workspace [`Cargo.toml`](Cargo.toml).

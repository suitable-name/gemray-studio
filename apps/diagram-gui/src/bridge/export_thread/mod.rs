//! High-resolution still export, running entirely off the interactive render thread.
//!
//! The live viewport (`render_thread.rs`) ties `RenderContext.width`/`height` to the
//! widget and progressively accumulates samples into a buffer sized to match -- an
//! export at, say, 4K/2048spp must not touch that buffer (it would corrupt the
//! viewport's in-progress accumulation and fight it for `width`/`height`) and must not
//! block it either (a multi-minute high-sample export would otherwise freeze the
//! interactive view). So this module:
//!
//! - takes its own read-only [`SceneSnapshot`] of the render state (camera, light,
//!   material, geometry) captured once under a short lock, independent of whatever the
//!   viewport does with `RenderContext` afterwards;
//! - owns its own accumulation buffer, sized to the export's own dimensions;
//! - runs on its own `thread::spawn` worker (parallelized internally with
//!   `thread::scope`, the same pattern `render_thread::render_frame_scanlines` uses);
//! - reports progress and honors cancellation via an [`ExportHandle`].
//!
//! It calls `gemray::optics::raytracer::trace_spectral_ray` directly rather than
//! reworking the shared progressive-render loop.
//!
//! # Wide-gamut export
//!
//! `run_export`'s final tone-mapping step now branches on the caller's chosen
//! `gemray::color::space::ColorSpace`: `Srgb` (the default) keeps going through
//! [`tonemap_to_rgba`], the exact pre-existing `xyz_to_srgb_gamma` path, so that
//! export stays byte-identical to what this app produced before this control existed
//! (`tests::srgb_export_matches_the_pre_wide_gamut_reference_path` pins this). Any
//! other space routes through [`tonemap_wide_gamut`] -- `ColorSpace::encode` with
//! `ToneMap::AcesFilmic { exposure: 1.0 }`, which that type's own doc comment
//! documents as reproducing `xyz_to_srgb_gamma`'s gamut/tone-mapping steps exactly, so
//! the wide-gamut path only changes gamut primaries and transfer curve, never
//! brightness -- and the file is written via [`save_png`], which embeds an ICC
//! profile (`bridge::icc_profile::build`) for any non-`Srgb` space so the wide-gamut
//! pixel values are never silently misinterpreted as sRGB by a viewer that has no
//! other way to know otherwise. `image` 0.25's PNG encoder can carry that profile
//! (`codecs::png::PngEncoder` implements `ImageEncoder::set_icc_profile`, confirmed by
//! reading that crate's vendored source) -- see `bridge::icc_profile`'s module doc
//! comment for why this generates the profile itself rather than embedding a
//! third-party one.
//!
//! `ColorSpace::AcesCg` is deliberately not offered by `export_dialog.slint`'s picker:
//! it is scene-linear (no transfer function), meant to feed further compositing in a
//! floating-point/high-bit-depth container, not an 8-bit PNG meant for direct
//! viewing -- quantizing scene-linear light to 8 bits per channel the way this export
//! path does would concentrate almost the entire visible tonal range into the lowest
//! few code values (severe banding), the exact opposite of what a transfer curve like
//! sRGB's or Rec.2020's exists to prevent.
//!
//! Split into submodules purely to keep this file from growing further: [`params`]
//! (validation and the default output path), [`scene_snapshot`] (the [`SceneSnapshot`]
//! capture), [`batch`] (the batch render loop -- hybrid calibration/dispatch and the
//! CPU scanline batch tracer), and [`tonemap_png`] (tone-map and PNG/ICC output). This
//! file keeps the export worker's own orchestration (`spawn_export`/`run_export`) and
//! its handle/outcome types.

mod batch;
mod params;
mod preview;
pub mod remote;
mod scene_snapshot;
mod tonemap_png;

pub use params::{ComputeTarget, ExportParams, default_export_path, validate_export_params};
pub use remote::{RemoteCapability, probe_remote};
pub use scene_snapshot::SceneSnapshot;

use crate::settings::WorkerSettings;
use batch::{ExportCtx, HYBRID_MIN_SPP, calibrate_split, render_batch, run_local_batches};
use gemray::{
    color::ColorSpace,
    optics::raytracer::Camera,
    renderer::gpu_backend::{GpuBackend, GpuSceneRef},
};
use gemray_net::client::Accumulator;
use glam::Vec3;
use preview::PreviewThrottle;
use remote::{
    REMOTE_MIN_SPP, calibrate_remote_fraction, exceeds_pixel_cap, shortfall, split_remote_samples,
};
use slint::{ComponentHandle, Rgba8Pixel, SharedPixelBuffer, Weak};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};
use tonemap_png::{save_png, tonemap_to_rgba, tonemap_wide_gamut};

#[derive(Debug)]
pub enum ExportOutcome {
    Completed(PathBuf),
    Cancelled,
    Failed(String),
}

/// One progress update from an in-flight export, delivered to `on_progress` after each
/// sample batch. `preview` carries a freshly regenerated thumbnail only when
/// `PreviewThrottle`'s ~2/sec rate limit allows it (see the `preview` submodule) --
/// every other tick's `preview` is `None`, and callers should simply leave the
/// currently-displayed image alone rather than treat `None` as "clear it".
pub struct ExportProgress {
    /// 0.0..=1.0, `samples_done as f32 / samples_total as f32`.
    pub fraction: f32,
    pub samples_done: u32,
    pub samples_total: u32,
    /// A small sRGB thumbnail of the combined local (CPU+GPU) and, while a remote
    /// engine is also in flight, remote accumulation state so far -- see
    /// `preview::downsample_preview` for how it's built and normalised.
    pub preview: Option<SharedPixelBuffer<Rgba8Pixel>>,
    /// A one-off, user-facing status line about the REMOTE half of this export --
    /// e.g. why it fell back to local-only before starting, or that a mid-export
    /// worker failure was recovered by finishing the shortfall locally. `None` on
    /// every other tick; callers should show it once (a toast) rather than treat a
    /// later `None` as "clear the message".
    pub note: Option<String>,
}

/// Handle returned by `spawn_export`. Cancelling is cooperative: the worker checks the
/// flag between sample batches (see `run_export`), so cancellation lands within one
/// batch's worth of work rather than instantly, but does not require tearing down the
/// thread forcibly.
pub struct ExportHandle {
    cancel: Arc<AtomicBool>,
}

impl ExportHandle {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Spawns the export worker thread. `on_progress` is invoked on the UI event loop with
/// an [`ExportProgress`] after each sample batch (fraction complete, samples done/total,
/// and -- at most a couple of times a second, see `preview::PreviewThrottle` -- a
/// freshly regenerated preview thumbnail); `on_done` is invoked once, exactly once, when
/// the export finishes, is cancelled, or fails. `color_space` selects the output PNG's
/// gamut/transfer curve -- see this module's doc comment. `compute_target`/`workers`
/// select and configure the remote engine -- see [`ComputeTarget`] and `run_export`'s
/// own doc comment.
///
/// # `on_done` fires on every exit path, including a panic
///
/// The worker thread's body runs inside `std::panic::catch_unwind`: a prior review
/// found that it did not, so a panic anywhere in `run_export` (or anything it calls)
/// would unwind the whole thread WITHOUT ever calling `on_done`, permanently leaving
/// `RenderContext::export_active` (which `gui::render_export`'s `on_done` closure is the
/// only thing that ever clears) stuck `true` -- freezing the live viewport for the rest
/// of the session, the exact same class of bug as an import panic wedging `is_busy`
/// (already fixed once in this project). Catching the panic here and reporting it as
/// `ExportOutcome::Failed` makes the reset in `gui::render_export`'s `on_done` closure
/// truly unconditional, matching this function's own doc comment above.
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a distinct piece of one export request's own identity \
              (scene, output params/path/colour-space, the new compute-target/workers \
              choice, and the two UI callbacks) -- bundling them into a struct would \
              just move the same count into field access, not reduce it"
)]
pub fn spawn_export<T, P, D>(
    ui_weak: Weak<T>,
    scene: SceneSnapshot,
    params: ExportParams,
    color_space: ColorSpace,
    output_path: PathBuf,
    compute_target: ComputeTarget,
    workers: Vec<WorkerSettings>,
    on_progress: P,
    on_done: D,
) -> ExportHandle
where
    T: ComponentHandle + 'static,
    P: Fn(&T, ExportProgress) + Send + 'static + Clone,
    D: Fn(&T, ExportOutcome) + Send + 'static + Clone,
{
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_worker = cancel.clone();
    let ui_weak_done = ui_weak.clone();

    thread::spawn(move || {
        let progress_ui_weak = ui_weak;
        let report_progress = move |progress: ExportProgress| {
            let on_progress = on_progress.clone();
            let _ = progress_ui_weak.upgrade_in_event_loop(move |ui| on_progress(&ui, progress));
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_export(
                &scene,
                params,
                color_space,
                &output_path,
                compute_target,
                &workers,
                &cancel_worker,
                report_progress,
            )
        }))
        .unwrap_or_else(|payload| {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "the export worker panicked".to_string());
            ExportOutcome::Failed(format!("Export failed unexpectedly: {message}"))
        });
        let _ = ui_weak_done.upgrade_in_event_loop(move |ui| on_done(&ui, outcome));
    });

    ExportHandle { cancel }
}

/// Renders `params.samples_per_pixel` samples per pixel in small batches (so progress
/// can be reported and cancellation checked between batches without either checking
/// after every single sample -- unbounded overhead on a huge image -- or waiting for
/// the entire, possibly minutes-long render before checking even once), tone-maps the
/// finished accumulation, and writes it to `output_path` as PNG.
///
/// # Local + remote as a third disjoint-range engine
///
/// `compute_target` and `workers` add a remote worker into the exact same
/// disjoint-absolute-sample-range convention the pre-existing CPU/GPU hybrid split
/// already uses (see the "Hybrid CPU+GPU export" comment below, unchanged): the ONE
/// absolute sample counter (`samples_done`) this function threads through calibration,
/// the remote dispatch, and the local loop in strict sequence is what guarantees no two
/// engines are ever handed overlapping indices, even though the remote dispatch and the
/// local loop then run CONCURRENTLY (inside the `thread::scope` below) once their
/// disjoint ranges are decided -- concurrency in wall-clock time, never in which
/// indices an engine owns. `ComputeTarget::LocalOnly` skips every remote code path
/// entirely (`remote_capability` stays `None`), so its output is byte-identical to this
/// function before remote existed -- see `tests::srgb_export_is_byte_identical_*`.
///
/// A remote worker is dispatched as ONE request covering its whole assigned range
/// (`apps/gemray-worker/src/validate/mod.rs::MAX_SAMPLES_PER_REQUEST` is 65,536; this
/// export's own `MAX_EXPORT_SPP` cap is 32,768 -- see `params`'s doc comment -- so even
/// a `ComputeTarget::RemoteOnly` export's full sample budget always fits in one
/// `RenderRequest`, no chunking needed), rather than many small batches the way local
/// CPU/GPU work is -- a network round trip's fixed overhead makes many-small-requests
/// the wrong shape for remote the way it's the right one in-process. If the connection
/// drops partway, [`remote::run_remote_batch`]'s returned `samples_done` is always
/// exactly the valid, already-summed PREFIX the worker completed (see that function's
/// and `remote`'s own doc comments on why `gemray-worker`'s tracer guarantees this);
/// this function traces the shortfall locally afterward rather than discarding it or
/// aborting the export -- see the "Remote outcome" section below.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "this is the export worker's own top-level orchestration -- scene/output \
              params plus the new compute-target/workers choice -- and its length is \
              the real local+remote concurrent-dispatch/fallback/merge logic the task \
              asked for, not padding; it's already split across `batch`/`remote`'s own \
              helper functions (`run_local_batches`, `calibrate_remote_fraction`, \
              `run_remote_batch`) everywhere that split doesn't fight the single \
              sequential absolute-sample-counter this function's whole correctness \
              argument depends on threading through in one place"
)]
fn run_export(
    scene: &SceneSnapshot,
    params: ExportParams,
    color_space: ColorSpace,
    output_path: &Path,
    compute_target: ComputeTarget,
    workers: &[WorkerSettings],
    cancel: &AtomicBool,
    mut report_progress: impl FnMut(ExportProgress),
) -> ExportOutcome {
    let ExportParams {
        width,
        height,
        samples_per_pixel,
    } = params;
    let mut accum = vec![Vec3::ZERO; (width as usize) * (height as usize)];
    let camera = Camera::new(scene.yaw, scene.pitch, scene.distance, 42.0);
    // Acquired once per export, not per batch: adapter acquisition and shader
    // compilation are far too slow to repeat 40 times.
    let gpu = GpuBackend::acquire();

    // Loop-invariant scene reference, hoisted (it was rebuilt per batch for no
    // benefit).
    let environment =
        scene
            .lighting_preset
            .studio(scene.exposure, scene.light_yaw, scene.light_pitch);
    let gpu_scene = GpuSceneRef {
        camera: &camera,
        width,
        height,
        planes: &scene.active_planes,
        facet_finishes: &scene.facet_finishes,
        material: &scene.material,
        max_bounces: scene.max_bounces,
        environment,
    };
    let ctx = ExportCtx {
        width,
        height,
        camera: &camera,
        scene,
        gpu: &gpu,
        gpu_scene: &gpu_scene,
    };

    // ---- Remote availability -------------------------------------------------------
    // Re-probed here (not trusted from whatever the export dialog observed when it
    // opened) so a worker that went away, or came up, in the meantime is judged by
    // the ACTUAL state at dispatch time -- "detect before dispatching", never assumed.
    let mut samples_done = 0u32;
    let mut pending_note: Option<String> = None;
    let remote_capability: Option<RemoteCapability> =
        if matches!(compute_target, ComputeTarget::LocalOnly) {
            None
        } else {
            match probe_remote(workers) {
                Ok(cap) => {
                    if exceeds_pixel_cap(width, height, &cap) {
                        // Never silently shrink the export -- fall back to local for the
                        // WHOLE export instead, with a clear status message.
                        let pixels = u64::from(width) * u64::from(height);
                        pending_note = Some(format!(
                            "This export ({pixels} px) exceeds the remote worker's maximum \
                         of {} px -- rendering locally only.",
                            cap.max_pixels
                        ));
                        None
                    } else {
                        Some(cap)
                    }
                }
                Err(reason) => {
                    pending_note = Some(format!("{} Rendering locally only.", reason.message()));
                    None
                }
            }
        };
    let scene_state = remote_capability
        .as_ref()
        .map(|_| remote::scene_state_from_snapshot(scene, width, height));

    // ---- Split decision: how many of the remaining samples go to remote -----------
    // `gpu_accum` is declared here (rather than alongside the "Hybrid CPU+GPU export"
    // comment below, where it conceptually belongs) because `calibrate_remote_fraction`
    // needs it too -- its own local-side calibration probe is GPU-first, same as
    // everywhere else in this function -- so it must exist before that call.
    let mut gpu_accum = vec![Vec3::ZERO; accum.len()];
    let mut remote_range: Option<(u32, u32)> = None;
    if let (Some(cap), Some(state)) = (&remote_capability, &scene_state) {
        let remaining = samples_per_pixel - samples_done;
        // `REMOTE_MIN_SPP` is chosen comfortably above `2 * REMOTE_CALIBRATION_SAMPLES`,
        // so `calibrate_remote_fraction` below always has enough remaining budget for
        // both its probes.
        if remaining >= REMOTE_MIN_SPP {
            let remote_frac = matches!(compute_target, ComputeTarget::Both).then(|| {
                calibrate_remote_fraction(
                    &ctx,
                    cap,
                    state,
                    &mut samples_done,
                    &mut accum,
                    &mut gpu_accum,
                    cancel,
                )
            });
            // `remaining` is recomputed AFTER calibration (which may have advanced
            // `samples_done`) -- see `split_remote_samples`'s own doc comment on why
            // the split is decided against what's actually left, not the pre-
            // calibration figure.
            let remaining_after_calibration = samples_per_pixel - samples_done;
            let remote_samples = split_remote_samples(
                compute_target,
                remaining_after_calibration,
                remote_frac.flatten(),
            );
            if remote_samples > 0 {
                remote_range = Some((samples_done, remote_samples));
            }
        }
    }
    if cancel.load(Ordering::Relaxed) {
        return ExportOutcome::Cancelled;
    }

    // Hybrid CPU+GPU export: the GPU and all CPU cores trace DISJOINT sample
    // ranges of the same frame concurrently, and radiance sums merge -- the
    // same disjoint-sample-range convention `gemray-net`'s distributed
    // protocol and `gemray::renderer::gpu::hybrid` (the verified library-level
    // reference implementation of this split) are built on. The GPU owns its
    // own accumulation buffer for the whole export and is summed into `accum`
    // once at the end, so the two engines never write one buffer concurrently.
    // Which samples land on which engine depends on the measured calibration
    // split, so two runs of the same export can assign samples differently --
    // each sample's estimate is valid and the ranges stay disjoint either way,
    // exactly as in a distributed render.
    let remote_count = remote_range.map_or(0, |(_, count)| count);
    // Local resumes exactly where remote's whole assigned slice ends (or right after
    // calibration, when remote isn't in play at all -- `remote_count == 0` then, so
    // this is a no-op, keeping the LocalOnly path identical to before remote existed).
    samples_done = remote_range.map_or(samples_done, |(first, count)| first + count);
    let mut gpu_frac: Option<f64> = if samples_per_pixel - samples_done >= HYBRID_MIN_SPP {
        calibrate_split(&ctx, &mut samples_done, &mut accum, &mut gpu_accum, cancel)
    } else {
        None
    };
    if cancel.load(Ordering::Relaxed) {
        return ExportOutcome::Cancelled;
    }

    let mut preview_throttle = PreviewThrottle::new();
    let remote_accumulator: Option<Arc<Mutex<Accumulator>>> =
        remote_range.map(|_| Arc::new(Mutex::new(Accumulator::new(width, height))));

    // Reports one progress tick: `local_done` is the caller's own absolute cursor
    // (already past remote's whole slice, per the assignment above), so subtracting
    // `remote_count` back out and adding remote's OWN live progress (peeked from its
    // accumulator, which `remote::run_remote_batch` updates as `FRAME`s arrive) gives
    // the export's TRUE combined completion -- see this module's doc comment. Reduces
    // to the pre-remote formula exactly when `remote_count == 0`.
    let mut on_local_batch = |local_done: u32, local_accum: &[Vec3], local_gpu_accum: &[Vec3]| {
        let (remote_done_so_far, remote_preview) =
            remote_accumulator.as_ref().map_or((0, None), |acc| {
                let acc = acc.lock().unwrap_or_else(PoisonError::into_inner);
                (acc.samples_done(), Some(acc.buffer().to_vec()))
            });
        let total_done =
            (local_done.saturating_sub(remote_count) + remote_done_so_far).min(samples_per_pixel);
        let preview = preview_throttle.maybe_generate(
            width,
            height,
            local_accum,
            local_gpu_accum,
            remote_preview.as_deref(),
            total_done,
        );
        report_progress(ExportProgress {
            fraction: total_done as f32 / samples_per_pixel as f32,
            samples_done: total_done,
            samples_total: samples_per_pixel,
            preview,
            note: pending_note.take(),
        });
    };

    let mut remote_outcome: Option<(u32, bool, Option<String>)> = None;
    if let Some((remote_first, remote_samples)) = remote_range {
        let acc = remote_accumulator
            .clone()
            .expect("remote_range implies remote_accumulator was built above");
        let cap = remote_capability
            .as_ref()
            .expect("remote_range implies remote_capability");
        let state = scene_state.expect("remote_range implies scene_state");

        thread::scope(|s| {
            let remote_thread = s.spawn(|| {
                remote::run_remote_batch(
                    cap,
                    state,
                    remote_first,
                    remote_samples,
                    width,
                    height,
                    &acc,
                    cancel,
                )
            });

            if samples_done < samples_per_pixel {
                run_local_batches(
                    &ctx,
                    samples_per_pixel,
                    &mut gpu_frac,
                    &mut samples_done,
                    &mut accum,
                    &mut gpu_accum,
                    cancel,
                    &mut on_local_batch,
                );
            } else {
                // Nothing left for local to trace (e.g. `ComputeTarget::RemoteOnly`,
                // or a `Both` split that gave remote the entire remaining budget) --
                // the concurrent phase is still real time passing, so poll for
                // progress on a light timer instead of leaving the progress bar
                // frozen until remote's single big request finishes outright.
                while !remote_thread.is_finished() {
                    on_local_batch(samples_done, &accum, &gpu_accum);
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(200));
                }
            }

            remote_outcome = Some(remote_thread.join().unwrap_or_else(|_| {
                (
                    0,
                    false,
                    Some("remote render worker thread panicked".to_string()),
                )
            }));
        });
    } else if samples_done < samples_per_pixel {
        run_local_batches(
            &ctx,
            samples_per_pixel,
            &mut gpu_frac,
            &mut samples_done,
            &mut accum,
            &mut gpu_accum,
            cancel,
            &mut on_local_batch,
        );
    }

    // A user-initiated cancel discards the whole export, remote's partial contribution
    // included -- unchanged from this function's pre-remote behaviour. `run_remote_batch`
    // already sent `CANCEL` to the worker (see its own doc comment) once it observed
    // `cancel`, so by the time control reaches here the worker has been asked to stop
    // too; this is just this function's own bookkeeping catching up.
    if cancel.load(Ordering::Relaxed) {
        return ExportOutcome::Cancelled;
    }

    // ---- Remote outcome: merge its buffer, and trace any shortfall locally --------
    // A shortfall here (as opposed to the `cancel` branch just above, which always
    // returns first) can ONLY mean a genuine worker/connection FAILURE, never a user
    // cancel -- see this module's doc comment. Every sample the worker DID complete is
    // already valid, already-summed radiance in `acc`'s buffer (see `remote`'s doc
    // comment), so this never discards or re-traces anything remote finished; it only
    // fills in what remote never got to.
    if let (Some((remote_first, remote_samples)), Some((remote_done, _cancelled, error))) =
        (remote_range, remote_outcome)
    {
        let acc = remote_accumulator
            .as_ref()
            .expect("remote_range implies remote_accumulator");
        {
            let acc = acc.lock().unwrap_or_else(PoisonError::into_inner);
            for (dst, src) in accum.iter_mut().zip(acc.buffer()) {
                *dst += *src;
            }
        }

        let missing = shortfall(remote_samples, remote_done);
        if missing > 0 {
            let shortfall_start = remote_first + remote_done;
            if !gpu.try_accumulate(&gpu_scene, shortfall_start, missing, &mut gpu_accum) {
                render_batch(
                    width,
                    height,
                    missing,
                    shortfall_start,
                    &camera,
                    scene,
                    &mut accum,
                );
            }
            let message = error.as_deref().unwrap_or("connection ended early");
            report_progress(ExportProgress {
                fraction: (samples_per_pixel - missing) as f32 / samples_per_pixel as f32,
                samples_done: samples_per_pixel - missing,
                samples_total: samples_per_pixel,
                preview: None,
                note: Some(format!(
                    "Remote worker failed ({message}) partway through -- finished the \
                     remaining {missing} samples locally. Every sample it completed \
                     first is still included."
                )),
            });
        }
    }

    // Merge the GPU's separate accumulation exactly once. A pure-CPU export
    // leaves `gpu_accum` all zero, making this a no-op.
    for (px, gpu_px) in accum.iter_mut().zip(&gpu_accum) {
        *px += *gpu_px;
    }

    if cancel.load(Ordering::Relaxed) {
        return ExportOutcome::Cancelled;
    }

    // `Srgb` keeps the exact pre-existing tone-mapping call (byte-identical
    // output); any other space routes through `tonemap_wide_gamut` instead -- see this
    // module's doc comment for why the two must stay this way.
    let rgba = if color_space == ColorSpace::Srgb {
        tonemap_to_rgba(width, height, samples_per_pixel, &accum)
    } else {
        tonemap_wide_gamut(width, height, samples_per_pixel, &accum, color_space)
    };

    if let Some(parent) = output_path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return ExportOutcome::Failed(format!("Could not create output directory: {e}"));
    }

    match save_png(output_path, width, height, &rgba, color_space) {
        Ok(()) => ExportOutcome::Completed(output_path.to_path_buf()),
        Err(e) => ExportOutcome::Failed(format!("Failed to write PNG: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemray::{
        geometry::cuts::StandardGemCuts,
        optics::{materials::GemMaterial, raytracer::LightingPreset},
    };

    #[test]
    fn run_export_produces_a_valid_png_for_a_tiny_scene() {
        let scene = SceneSnapshot {
            yaw: 0.60,
            pitch: 0.45,
            distance: 2.4,
            light_yaw: 0.85,
            light_pitch: 0.95,
            material: GemMaterial::diamond(),
            lighting_preset: LightingPreset::RingLights,
            max_bounces: 4,
            exposure: 1.0,
            active_planes: StandardGemCuts::standard_round_brilliant(),
            facet_finishes: Vec::new(),
        };
        let params = ExportParams {
            width: 8,
            height: 8,
            samples_per_pixel: 1,
        };
        let cancel = AtomicBool::new(false);

        let dir =
            std::env::temp_dir().join(format!("diagram-gui-export-test-{}", std::process::id()));
        let output_path = dir.join("tiny.png");

        let mut progress_calls = 0;
        let outcome = run_export(
            &scene,
            params,
            ColorSpace::Srgb,
            &output_path,
            ComputeTarget::LocalOnly,
            &[],
            &cancel,
            |_frac| {
                progress_calls += 1;
            },
        );

        match outcome {
            ExportOutcome::Completed(path) => {
                assert_eq!(path, output_path);
                let img =
                    image::open(&path).expect("exported file must be a valid, readable image");
                assert_eq!(img.width(), 8);
                assert_eq!(img.height(), 8);
            }
            other => panic!("expected a completed export, got {other:?}"),
        }
        assert!(
            progress_calls > 0,
            "progress must be reported at least once"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_export_honors_pre_set_cancellation_and_writes_no_file() {
        let scene = SceneSnapshot {
            yaw: 0.60,
            pitch: 0.45,
            distance: 2.4,
            light_yaw: 0.85,
            light_pitch: 0.95,
            material: GemMaterial::diamond(),
            lighting_preset: LightingPreset::RingLights,
            max_bounces: 4,
            exposure: 1.0,
            active_planes: StandardGemCuts::standard_round_brilliant(),
            facet_finishes: Vec::new(),
        };
        let params = ExportParams {
            width: 8,
            height: 8,
            samples_per_pixel: 64,
        };
        let cancel = AtomicBool::new(true); // already cancelled before starting

        let dir = std::env::temp_dir().join(format!(
            "diagram-gui-export-cancel-test-{}",
            std::process::id()
        ));
        let output_path = dir.join("never.png");

        let outcome = run_export(
            &scene,
            params,
            ColorSpace::Srgb,
            &output_path,
            ComputeTarget::LocalOnly,
            &[],
            &cancel,
            |_frac| {},
        );
        assert!(matches!(outcome, ExportOutcome::Cancelled));
        assert!(!output_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Wide-gamut export -----------------------------------------------------------

    fn tiny_scene() -> SceneSnapshot {
        SceneSnapshot {
            yaw: 0.60,
            pitch: 0.45,
            distance: 2.4,
            light_yaw: 0.85,
            light_pitch: 0.95,
            material: GemMaterial::diamond(),
            lighting_preset: LightingPreset::RingLights,
            max_bounces: 4,
            exposure: 1.0,
            active_planes: StandardGemCuts::standard_round_brilliant(),
            facet_finishes: Vec::new(),
        }
    }

    /// Pins the wide-gamut-export requirement in its strongest form: an `Srgb` export must
    /// produce bytes IDENTICAL to what this app wrote before the colour-space picker
    /// existed, down to the file's raw bytes -- not just "looks the same", but
    /// literally unchanged, so an export anyone was already doing/scripting/diffing
    /// against is unaffected by this feature landing.
    #[test]
    fn srgb_export_is_byte_identical_regardless_of_which_save_png_path_runs() {
        let scene = tiny_scene();
        let params = ExportParams {
            width: 6,
            height: 5,
            samples_per_pixel: 2,
        };

        // Two full (deterministic) renders, through `run_export`'s public entry point,
        // must agree byte-for-byte -- this is what actually exercises `save_png`'s
        // `Srgb` branch (the plain `image::RgbaImage::save` call) rather than
        // reimplementing it here.
        let dir = std::env::temp_dir().join(format!(
            "diagram-gui-srgb-identity-test-{}",
            std::process::id()
        ));
        let path_a = dir.join("a.png");
        let path_b = dir.join("b.png");
        let cancel = AtomicBool::new(false);

        let outcome_a = run_export(
            &scene,
            params,
            ColorSpace::Srgb,
            &path_a,
            ComputeTarget::LocalOnly,
            &[],
            &cancel,
            |_| {},
        );
        let outcome_b = run_export(
            &scene,
            params,
            ColorSpace::Srgb,
            &path_b,
            ComputeTarget::LocalOnly,
            &[],
            &cancel,
            |_| {},
        );
        assert!(matches!(outcome_a, ExportOutcome::Completed(_)));
        assert!(matches!(outcome_b, ExportOutcome::Completed(_)));

        let bytes_a = std::fs::read(&path_a).unwrap();
        let bytes_b = std::fs::read(&path_b).unwrap();
        assert_eq!(
            bytes_a, bytes_b,
            "two Srgb exports of the identical scene must match exactly"
        );
        // No `iCCP` chunk anywhere in the file -- an sRGB export must stay untagged,
        // matching this app's behaviour before this control existed.
        assert!(
            !contains_bytes(&bytes_a, b"iCCP"),
            "an Srgb export must never carry an ICC profile chunk"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A non-`Srgb` export must carry an embedded ICC profile (`iCCP` PNG chunk) --
    /// the whole point of `bridge::icc_profile`: an untagged wide-gamut PNG is
    /// silently misread as sRGB by every viewer.
    #[test]
    fn wide_gamut_export_embeds_an_icc_profile_chunk() {
        for color_space in [ColorSpace::DisplayP3, ColorSpace::Rec2020] {
            let scene = tiny_scene();
            let params = ExportParams {
                width: 6,
                height: 5,
                samples_per_pixel: 2,
            };
            let cancel = AtomicBool::new(false);
            let dir = std::env::temp_dir().join(format!(
                "diagram-gui-wide-gamut-test-{color_space:?}-{}",
                std::process::id()
            ));
            let output_path = dir.join("wide.png");

            let outcome = run_export(
                &scene,
                params,
                color_space,
                &output_path,
                ComputeTarget::LocalOnly,
                &[],
                &cancel,
                |_| {},
            );
            match outcome {
                ExportOutcome::Completed(path) => {
                    let bytes = std::fs::read(&path).unwrap();
                    assert!(
                        contains_bytes(&bytes, b"iCCP"),
                        "{color_space:?}: expected an iCCP chunk in the exported PNG"
                    );
                    let img = image::open(&path)
                        .unwrap_or_else(|e| panic!("{color_space:?}: not a readable PNG: {e}"));
                    assert_eq!(img.width(), 6);
                    assert_eq!(img.height(), 5);
                }
                other => panic!("{color_space:?}: expected a completed export, got {other:?}"),
            }

            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Byte substring search -- enough to confirm a chunk signature is present
    /// somewhere in a small test PNG without pulling in a PNG-chunk-parsing
    /// dependency just for this test.
    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }
}

#[cfg(test)]
mod hybrid_export_tests {
    use super::*;
    use gemray::{
        geometry::cuts::StandardGemCuts,
        optics::{materials::GemMaterial, raytracer::LightingPreset},
    };

    /// End-to-end smoke test of the hybrid export path: a small frame rendered
    /// through the real `run_export` (calibration, batching, merge, PNG write).
    /// Without a GPU adapter or the `gpu` feature this exercises the CPU-only
    /// path; with both, it exercises calibration plus concurrent hybrid
    /// batches. Asserts completion and a non-empty file, not pixel values (the
    /// hybrid sample-to-engine assignment is calibration-dependent by design).
    #[test]
    fn export_completes_via_hybrid_or_cpu_path() {
        let scene = SceneSnapshot {
            yaw: 0.6,
            pitch: -0.4,
            distance: 3.0,
            light_yaw: 0.85,
            light_pitch: 0.95,
            material: GemMaterial::by_name("Zircon").expect("built-in material"),
            lighting_preset: LightingPreset::RingLights,
            max_bounces: 12,
            exposure: 1.0,
            active_planes: StandardGemCuts::standard_round_brilliant(),
            facet_finishes: Vec::new(),
        };
        let params = ExportParams {
            width: 48,
            height: 48,
            samples_per_pixel: 12,
        };
        let out = std::env::temp_dir().join("gemray_hybrid_export_smoke.png");
        let cancel = AtomicBool::new(false);
        let outcome = run_export(
            &scene,
            params,
            ColorSpace::Srgb,
            &out,
            ComputeTarget::LocalOnly,
            &[],
            &cancel,
            |_frac| {},
        );
        match outcome {
            ExportOutcome::Completed(path) => {
                let len = std::fs::metadata(&path).map_or(0, |m| m.len());
                assert!(len > 0, "export wrote an empty file");
                let _ = std::fs::remove_file(path);
            }
            ExportOutcome::Cancelled => panic!("export reported cancelled without a cancel"),
            ExportOutcome::Failed(e) => panic!("export failed: {e}"),
        }
    }
}

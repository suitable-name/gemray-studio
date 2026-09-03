//! Local primary-ray-only guide-buffer prepass, so a remote-sourced image (radiance
//! only, no guide buffers over the wire -- see `bridge::remote_render`'s module docs)
//! can still be denoised by the same edge-avoiding À-Trous filter the local path uses.
//!
//! # Why this exists
//!
//! `crates/gemray/src/renderer/denoise` is edge-avoiding: it needs a first-hit
//! depth/normal/facet-id per pixel to decide which neighbours are allowed to blend
//! together. The local render loop (`bridge::render_thread::render_frame_scanlines`)
//! gets those for free as a side effect of tracing real spectral samples (via
//! `trace_spectral_ray`'s `primary_hit_out` parameter). A remote worker's `FRAME`/
//! `PREVIEW` payload carries only summed XYZ radiance -- shipping the guide buffers too
//! would multiply the wire payload every frame for data the viewer can regenerate
//! almost free, so they never travel over the wire at all.
//!
//! Instead, this module casts exactly one un-jittered camera ray per pixel and records
//! its first hit -- no bounces, no spectral sampling, no light transport, just the same
//! [`intersect_polyhedron`] call `trace_spectral_ray` itself makes at bounce 0 before it
//! ever touches wavelengths, Stokes vectors, or Fresnel splitting. That is deliberately
//! the SAME geometry-intersection code the local path's guide capture is built on
//! (`trace_spectral_ray`'s bounce-0 `primary_hit_out` capture, see
//! `render_thread::render_frame_scanlines`'s doc comment) -- this module does not
//! reinvent ray/polyhedron intersection, it just calls the piece of that mechanism that
//! produces a first hit, without paying for everything downstream of it a real sample
//! also computes.
//!
//! # Why caching, and on what key
//!
//! The guide buffers depend only on camera pose (`yaw`/`pitch`/`distance`) and the
//! active facet geometry -- never on which backend (local or remote) produced the
//! radiance, and never on light direction, material, or exposure. So they stay valid
//! for as long as the pose and the gem don't change, and [`GuideCache`] recomputes them
//! only when [`GuideCache::ensure`]'s key (resolution + pose + a hash of the facet
//! planes) actually differs from the last call -- not on every redraw of an
//! in-progress remote accumulation, even though a remote render's `FRAME` events can
//! arrive many times per second.
//!
//! # Measured cost -- not free, but no longer paid on the UI thread
//!
//! "Almost free" only holds relative to a real render, and only for the amortized
//! per-`FRAME` cost once cached. The one-time cost of a single [`generate_guide_buffers`]
//! call (measured on the machine this was developed on, `cargo test --profile probe`,
//! single-threaded-per-chunk across `thread::available_parallelism` cores,
//! `StandardGemCuts::standard_round_brilliant`): ~19ms at 800x600, ~82ms at 1920x1080,
//! and ~288ms at 3840x2160. That 4K figure used to be a genuinely user-visible stall:
//! [`GuideCache::ensure`] was called synchronously from
//! `gui::remote::redraw_from_accumulator`, which itself runs on the Slint UI/event-loop
//! thread (inside `upgrade_in_event_loop`) -- so at 4K, the FIRST redraw after a
//! camera-pose or gem change that triggered a remote render would visibly freeze the UI
//! thread for a quarter of a second.
//!
//! It no longer does. `gui::remote::start_remote_render` now kicks the prepass off on a
//! background thread as soon as the `RenderRequest` is dispatched -- guides depend only
//! on camera pose and gem geometry, both already known at dispatch time -- via
//! [`generate_guide_buffers_cancellable`], so the work overlaps the network round trip
//! and the remote render itself instead of blocking the first frame's display. Cancellation
//! is cooperative, via an `Arc<AtomicBool>` checked between rows, mirroring
//! `bridge::export_thread`'s `ExportHandle` (see that module's doc comment): if the pose
//! changes again before a background generation finishes, `gui::remote` abandons it and
//! starts a fresh one for the new pose rather than letting a stale result land.
//! [`GuideCache::ensure`] itself stays synchronous and still lives here -- it's what a
//! background result gets folded into ([`GuideCache::adopt`]) once `gui::remote` has
//! confirmed, via [`GuideCache::key_for`]/[`GuideCache::matches_key`], that it matches
//! the pose currently on screen. If a frame arrives before the background generation for
//! its pose has finished, `gui::remote` renders that frame with a plain tonemap instead
//! of blocking to regenerate guides synchronously; a later redraw denoises once they're
//! ready.

use crate::bridge::render_thread::hash_planes;
use gemray::{
    geometry::plane::GpuFacetPlane,
    optics::raytracer::{Camera, intersect_polyhedron},
};
use glam::Vec3;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

/// One pixel-per-camera-ray depth/normal/facet-id capture, row-major
/// (`index = y * width + x`) -- exactly the shape `renderer::denoise::GBuffers` expects
/// for its `depth`/`normal`/`facet_id` fields.
#[derive(Debug, Clone)]
pub struct GuideBuffers {
    pub depth: Vec<f32>,
    pub normal: Vec<Vec3>,
    pub facet_id: Vec<i32>,
}

impl GuideBuffers {
    /// An all-miss buffer of the given size: `depth = 1.0e6` (matching
    /// `render_thread::update_accumulation_state`'s own "no hit yet" sentinel),
    /// `normal = ZERO`, `facet_id = -1`.
    fn miss(width: u32, height: u32) -> Self {
        let pixel_count = (width as usize) * (height as usize);
        Self {
            depth: vec![1.0e6; pixel_count],
            normal: vec![Vec3::ZERO; pixel_count],
            facet_id: vec![-1; pixel_count],
        }
    }
}

/// Casts one un-jittered camera ray per pixel and records its first hit's
/// depth/normal/facet index -- no bounces, no spectral sampling, no light transport.
/// Parallel across `thread::available_parallelism` CPU threads, mirroring
/// `render_thread::render_frame_scanlines`'s own chunking (row-contiguous chunks, one
/// worker thread per chunk) so this scales the same way a real render does, just over
/// far cheaper per-pixel work.
///
/// Returns an all-miss [`GuideBuffers`] (still correctly sized) for a zero-area image,
/// matching `render_thread`'s own zero-dimension handling rather than panicking.
#[must_use]
pub fn generate_guide_buffers(
    width: u32,
    height: u32,
    camera: &Camera,
    planes: &[GpuFacetPlane],
) -> GuideBuffers {
    // A freshly-constructed, never-stored `AtomicBool` that nothing ever sets: `cancel`
    // is always observed `false`, so `generate_guide_buffers_cancellable` always runs to
    // completion and never returns `None`. This keeps the actual per-pixel loop defined
    // exactly once (see that function's doc comment) while leaving this function's
    // signature -- used throughout this module's own tests, and by
    // `gui::remote`'s synchronous `GuideCache::ensure` path -- unchanged.
    generate_guide_buffers_cancellable(width, height, camera, planes, &AtomicBool::new(false))
        .expect("a cancel flag that is never set to true never yields a cancelled result")
}

/// The cancellable core [`generate_guide_buffers`] is a thin wrapper around: casts one
/// un-jittered camera ray per pixel and records its first hit's depth/normal/facet
/// index -- no bounces, no spectral sampling, no light transport. Parallel across
/// `thread::available_parallelism` CPU threads, mirroring
/// `render_thread::render_frame_scanlines`'s own chunking (row-contiguous chunks, one
/// worker thread per chunk) so this scales the same way a real render does, just over
/// far cheaper per-pixel work.
///
/// `cancel` is checked cooperatively -- once before a row of pixels, mirroring
/// `bridge::export_thread`'s "check between batches" cancellation (see that module's
/// `run_export`) -- so a generation abandoned mid-flight (the pose changed again before
/// it finished) stops within roughly one row's worth of work rather than running to
/// completion for a pose nobody wants any more. Returns `None` if cancellation was
/// observed at any point during the computation -- the caller (`gui::remote`'s
/// background guide-generation thread) must then discard whatever partial buffers exist
/// rather than publishing them, since they cover only some of the image.
///
/// Returns an all-miss [`GuideBuffers`] (still correctly sized) for a zero-area image,
/// matching `render_thread`'s own zero-dimension handling rather than panicking -- even
/// if `cancel` happens to already be set, since a zero-area result costs nothing to
/// produce and there is no partial-image concern for it.
#[must_use]
pub fn generate_guide_buffers_cancellable(
    width: u32,
    height: u32,
    camera: &Camera,
    planes: &[GpuFacetPlane],
    cancel: &AtomicBool,
) -> Option<GuideBuffers> {
    if width == 0 || height == 0 {
        return Some(GuideBuffers::miss(width, height));
    }
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let mut buffers = GuideBuffers::miss(width, height);
    let num_threads = thread::available_parallelism().map_or(8, std::num::NonZero::get);
    let rows_per_chunk = (height as usize).div_ceil(num_threads);

    thread::scope(|s| {
        let chunks_depth: Vec<&mut [f32]> = buffers
            .depth
            .chunks_mut(rows_per_chunk * width as usize)
            .collect();
        let chunks_normal: Vec<&mut [Vec3]> = buffers
            .normal
            .chunks_mut(rows_per_chunk * width as usize)
            .collect();
        let chunks_facet: Vec<&mut [i32]> = buffers
            .facet_id
            .chunks_mut(rows_per_chunk * width as usize)
            .collect();

        let chunks = chunks_depth
            .into_iter()
            .zip(chunks_normal)
            .zip(chunks_facet);

        for (chunk_idx, ((depth_chunk, normal_chunk), facet_chunk)) in chunks.enumerate() {
            let start_y = chunk_idx * rows_per_chunk;
            let end_y = (start_y + rows_per_chunk).min(height as usize);

            s.spawn(move || {
                for y in start_y..end_y {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let local_y = y - start_y;
                    let row_offset = local_y * width as usize;

                    for x in 0..(width as usize) {
                        let local_idx = row_offset + x;
                        // No jitter: this is a single deterministic prepass, not an
                        // accumulated sample -- there is nothing to anti-alias against.
                        let ray = camera.generate_ray(
                            x as f32,
                            y as f32,
                            width as f32,
                            height as f32,
                            0.0,
                            0.0,
                        );
                        let hit = intersect_polyhedron(ray, planes);

                        depth_chunk[local_idx] = hit.map_or(1.0e6, |h| h.t);
                        normal_chunk[local_idx] = hit.map_or(Vec3::ZERO, |h| h.normal);
                        facet_chunk[local_idx] = hit.map_or(-1, |h| h.facet_idx as i32);
                    }
                }
            });
        }
    });

    if cancel.load(Ordering::Relaxed) {
        None
    } else {
        Some(buffers)
    }
}

/// Everything that determines the guide buffers: resolution, camera pose, and the
/// active facet geometry (see the module doc comment for why light/material/exposure
/// are deliberately excluded). `planes_hash` reuses
/// `render_thread::hash_planes` -- the same cheap Pod-bytes hash the metrics cache in
/// that module already uses to detect a geometry change -- rather than a second,
/// parallel implementation.
///
/// `pub`, not private: `gui::remote`'s background guide-generation path tags its result
/// with this SAME key (via [`GuideCache::key_for`]) rather than inventing a second,
/// possibly-diverging notion of "same pose and geometry" -- see `GuideCache::adopt`'s
/// doc comment. Fields stay private so a `GuideKey` can only ever be constructed via
/// `key_for`, never hand-assembled with a mismatched or stale hash. (Not `pub(crate)` --
/// see `hash_planes`'s doc comment on the identical `redundant_pub_crate` reasoning:
/// `bridge::guide_pass` is itself a `pub mod`.)
#[derive(Debug, Clone, PartialEq)]
pub struct GuideKey {
    width: u32,
    height: u32,
    yaw: f32,
    pitch: f32,
    distance: f32,
    planes_hash: u64,
}

/// Caches one [`GuideBuffers`], regenerating it only when [`GuideKey`] changes. See the
/// module doc comment for why pose + geometry alone (not per-frame) is the right
/// invalidation granularity.
///
/// `buffers` is a plain (never-`Option`) field, deliberately: `key` starts `None`, so
/// the very first [`Self::ensure`] call always sees a "stale" key and populates
/// `buffers` before anything ever reads it -- there is no separate empty/uninitialized
/// state to unwrap out of, which is what lets `ensure` return a plain `&GuideBuffers`
/// with no panicking accessor.
#[derive(Debug)]
pub struct GuideCache {
    key: Option<GuideKey>,
    buffers: GuideBuffers,
    /// Incremented every time [`Self::ensure`] actually regenerates the buffers (as
    /// opposed to reusing the cached ones). Exists purely so tests can observe cache
    /// hits/misses without inspecting buffer contents -- not read by any production
    /// call site.
    generation: u64,
}

impl Default for GuideCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GuideCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            key: None,
            buffers: GuideBuffers::miss(0, 0),
            generation: 0,
        }
    }

    /// How many times [`Self::ensure`] has actually regenerated the guide buffers so
    /// far. Test-only hook (see the field's own doc comment) -- `#[cfg(test)]` rather
    /// than plain `pub` so it doesn't need an `#[allow(dead_code)]` in a build where
    /// nothing outside tests calls it.
    #[cfg(test)]
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the guide buffers valid for `(width, height, yaw, pitch, distance,
    /// planes)`, regenerating the primary-ray prepass only if that key differs from the
    /// last call -- an unchanged pose/gem on a subsequent call (e.g. a later `FRAME`
    /// event from the same in-progress remote render) is a cache hit and costs nothing
    /// beyond the key comparison.
    pub fn ensure(
        &mut self,
        width: u32,
        height: u32,
        yaw: f32,
        pitch: f32,
        distance: f32,
        planes: &[GpuFacetPlane],
    ) -> &GuideBuffers {
        let key = Self::key_for(width, height, yaw, pitch, distance, planes);
        if self.key.as_ref() != Some(&key) {
            let camera = Camera::new(yaw, pitch, distance, 42.0);
            self.buffers = generate_guide_buffers(width, height, &camera, planes);
            self.key = Some(key);
            self.generation += 1;
        }
        &self.buffers
    }

    /// Computes the [`GuideKey`] for `(width, height, yaw, pitch, distance, planes)` --
    /// the SAME identity [`Self::ensure`] uses internally to decide cache hit vs. miss.
    /// Exposed so a caller that generates guide buffers OUTSIDE this cache (`gui::remote`'s
    /// background prepass, kicked off at `RenderRequest` dispatch time rather than on
    /// first redraw -- see the module doc comment) can tag its result with the exact key
    /// [`Self::matches_key`]/[`Self::adopt`] will compare against.
    #[must_use]
    pub fn key_for(
        width: u32,
        height: u32,
        yaw: f32,
        pitch: f32,
        distance: f32,
        planes: &[GpuFacetPlane],
    ) -> GuideKey {
        GuideKey {
            width,
            height,
            yaw,
            pitch,
            distance,
            planes_hash: hash_planes(planes),
        }
    }

    /// True if `key` already matches this cache's current contents -- i.e. a
    /// [`Self::ensure`] call with the same key would be a cache hit, costing nothing
    /// beyond the comparison. Lets a caller confirm it's safe to treat the cache as
    /// "ready" for a given pose/geometry without risking `ensure`'s synchronous
    /// regenerate path.
    #[must_use]
    pub fn matches_key(&self, key: &GuideKey) -> bool {
        self.key.as_ref() == Some(key)
    }

    /// Adopts externally-computed guide buffers -- the result of a background
    /// [`generate_guide_buffers_cancellable`] call kicked off at dispatch time -- as this
    /// cache's contents for `key`, without running the prepass itself and without
    /// touching [`Self::generation`] (no regeneration happened here; that counter tracks
    /// actual prepass runs, and the background thread's own run isn't attributed to this
    /// cache instance's `ensure` call count). The caller is responsible for `buffers`
    /// actually being the result of generating for exactly `key` -- `gui::remote`'s
    /// async guide-generation path only ever calls this after confirming the background
    /// result's key equals the key [`Self::key_for`] computes for the pose/geometry
    /// currently on screen.
    pub fn adopt(&mut self, key: GuideKey, buffers: GuideBuffers) {
        self.key = Some(key);
        self.buffers = buffers;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemray::geometry::cuts::StandardGemCuts;

    #[test]
    fn generate_guide_buffers_is_correctly_sized_and_facet_bounded() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let camera = Camera::new(0.60, 0.45, 2.4, 42.0);
        let guides = generate_guide_buffers(16, 12, &camera, &planes);

        assert_eq!(guides.depth.len(), 16 * 12);
        assert_eq!(guides.normal.len(), 16 * 12);
        assert_eq!(guides.facet_id.len(), 16 * 12);

        // Looking straight at a centred gem from a reasonable distance, the centre
        // pixel must hit *some* facet, and every recorded facet id must be a valid
        // index into `planes` (or the -1 "miss" sentinel).
        let centre = (6 * 16 + 8) as usize;
        assert!(
            guides.facet_id[centre] >= 0,
            "centre pixel should hit the gem"
        );
        for &id in &guides.facet_id {
            assert!(
                id == -1 || (id as usize) < planes.len(),
                "facet id {id} out of range for {} planes",
                planes.len()
            );
        }
    }

    #[test]
    fn generate_guide_buffers_handles_zero_area_without_panicking() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let camera = Camera::new(0.0, 0.0, 2.4, 42.0);
        let guides = generate_guide_buffers(0, 0, &camera, &planes);
        assert_eq!(guides.depth.len(), 0);
        assert_eq!(guides.normal.len(), 0);
        assert_eq!(guides.facet_id.len(), 0);
    }

    #[test]
    fn a_miss_pixel_gets_the_sentinel_depth_and_facet_id() {
        // A ray aimed far off the gem's silhouette misses every plane.
        let planes = StandardGemCuts::standard_round_brilliant();
        // Pull the camera far back and look at a corner of a tiny image so at least
        // the corner pixels miss.
        let camera = Camera::new(0.0, 1.5, 50.0, 5.0);
        let guides = generate_guide_buffers(4, 4, &camera, &planes);
        assert!(
            guides.facet_id.contains(&-1),
            "expected at least one miss pixel at this camera distance/fov"
        );
        for (i, &id) in guides.facet_id.iter().enumerate() {
            if id == -1 {
                assert_eq!(guides.depth[i], 1.0e6);
                assert_eq!(guides.normal[i], Vec3::ZERO);
            }
        }
    }

    #[test]
    fn guide_cache_reuses_buffers_when_the_key_is_unchanged() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let mut cache = GuideCache::new();

        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        assert_eq!(cache.generation(), 1);

        // Same width/height/pose/geometry -- must be a cache hit (generation
        // unchanged), which is the whole "recompute on pose/gem change, not per
        // frame" contract this module exists for.
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        assert_eq!(
            cache.generation(),
            1,
            "an unchanged pose/geometry must reuse the cached guide buffers"
        );
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        assert_eq!(cache.generation(), 1);
    }

    #[test]
    fn guide_cache_regenerates_on_yaw_change() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let mut cache = GuideCache::new();
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        cache.ensure(8, 8, 0.90, 0.45, 2.4, &planes);
        assert_eq!(
            cache.generation(),
            2,
            "a changed yaw must invalidate the cache"
        );
    }

    #[test]
    fn guide_cache_regenerates_on_pitch_or_distance_change() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let mut cache = GuideCache::new();
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        cache.ensure(8, 8, 0.60, 0.80, 2.4, &planes);
        assert_eq!(
            cache.generation(),
            2,
            "a changed pitch must invalidate the cache"
        );

        cache.ensure(8, 8, 0.60, 0.80, 3.0, &planes);
        assert_eq!(
            cache.generation(),
            3,
            "a changed distance must invalidate the cache"
        );
    }

    #[test]
    fn guide_cache_regenerates_when_the_gem_geometry_changes() {
        let srb = StandardGemCuts::standard_round_brilliant();
        let emerald = StandardGemCuts::emerald_cut();
        let mut cache = GuideCache::new();
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &srb);
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &emerald);
        assert_eq!(
            cache.generation(),
            2,
            "a changed cutting schedule must invalidate the cache even with an \
             unchanged camera pose"
        );
    }

    #[test]
    fn guide_cache_regenerates_on_resolution_change() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let mut cache = GuideCache::new();
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        cache.ensure(16, 8, 0.60, 0.45, 2.4, &planes);
        assert_eq!(
            cache.generation(),
            2,
            "a changed output resolution must invalidate the cache"
        );
    }

    /// The cache is keyed on camera pose and geometry ONLY -- light direction is not
    /// part of the key, because a primary-ray-only prepass never samples lighting at
    /// all. This isn't something a caller could get wrong by passing a light angle in
    /// (there is no such parameter), but it's worth pinning as the documented design
    /// decision: reusing guides across a light-only change must never regenerate them.
    #[test]
    fn ensure_signature_has_no_light_parameters() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let mut cache = GuideCache::new();
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        assert_eq!(cache.generation(), 1);
    }

    #[test]
    fn generate_guide_buffers_cancellable_matches_the_non_cancellable_version_when_never_cancelled()
    {
        let planes = StandardGemCuts::standard_round_brilliant();
        let camera = Camera::new(0.60, 0.45, 2.4, 42.0);
        let cancel = AtomicBool::new(false);

        let expected = generate_guide_buffers(16, 12, &camera, &planes);
        let actual = generate_guide_buffers_cancellable(16, 12, &camera, &planes, &cancel)
            .expect("an AtomicBool that's never set true must never yield a cancelled result");

        assert_eq!(actual.depth, expected.depth);
        assert_eq!(actual.facet_id, expected.facet_id);
    }

    #[test]
    fn generate_guide_buffers_cancellable_returns_none_when_pre_cancelled() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let camera = Camera::new(0.60, 0.45, 2.4, 42.0);
        let cancel = AtomicBool::new(true);

        let result = generate_guide_buffers_cancellable(64, 64, &camera, &planes, &cancel);
        assert!(
            result.is_none(),
            "a generation cancelled before it starts must not produce buffers"
        );
    }

    #[test]
    fn guide_cache_key_for_matches_what_ensure_uses_internally() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let mut cache = GuideCache::new();
        let key = GuideCache::key_for(8, 8, 0.60, 0.45, 2.4, &planes);

        assert!(
            !cache.matches_key(&key),
            "a freshly-constructed cache must not match any key yet"
        );
        cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        assert!(
            cache.matches_key(&key),
            "the key ensure() just populated must equal key_for()'s independently \
             computed key for the identical pose/geometry"
        );
    }

    #[test]
    fn guide_cache_adopt_installs_externally_computed_buffers_without_recomputing() {
        let planes = StandardGemCuts::standard_round_brilliant();
        let camera = Camera::new(0.60, 0.45, 2.4, 42.0);
        let buffers = generate_guide_buffers(8, 8, &camera, &planes);
        let key = GuideCache::key_for(8, 8, 0.60, 0.45, 2.4, &planes);

        let mut cache = GuideCache::new();
        cache.adopt(key.clone(), buffers.clone());

        assert_eq!(
            cache.generation(),
            0,
            "adopt() folds in an externally-computed result -- it must not be counted \
             as this cache having run its own prepass"
        );
        assert!(cache.matches_key(&key));

        // A subsequent ensure() for the identical key must be a pure cache hit: same
        // generation, same (adopted) buffers, no recompute triggered.
        let cached = cache.ensure(8, 8, 0.60, 0.45, 2.4, &planes);
        assert_eq!(cached.depth, buffers.depth);
        assert_eq!(cache.generation(), 0);
    }
}

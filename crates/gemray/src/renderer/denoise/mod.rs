//! Edge-avoiding À-Trous wavelet denoiser.
//!
//! # Why this exists
//!
//! Correct spectral dispersion means that once a path refracts dispersively, only the
//! hero wavelength survives -- the other channels' pdf for that path is genuinely zero.
//! Each sample therefore carries one wavelength's colour rather than an average of
//! eight, which is what produces both the fire *and* the chromatic speckle in the
//! viewport: they are the same phenomenon. You cannot remove that noise by making the
//! transport less correct; you remove it by filtering (this module) or by taking far
//! more samples.
//!
//! # What this is, and deliberately is not
//!
//! An edge-avoiding À-Trous wavelet filter (Dammertz, Sewtz, Hachisuka, Hensley
//! Danskin 2010, "Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination
//! Filtering"), not full A-SVGF. À-Trous was chosen deliberately: it runs in a handful
//! of separable-in-spirit passes, needs no temporal history or motion vectors, and
//! captures most of the benefit of a full spatiotemporal filter for a fraction of the
//! complexity. Each pass convolves with a widening (dilated) 5x5 B3-spline kernel while
//! down-weighting neighbours whose auxiliary ("guide") buffers disagree with the centre
//! pixel's.
//!
//! # Wiring
//!
//! This module is a self-contained library operating on plain buffers; it does not
//! import `optics` or `apps/diagram-gui` itself. It IS wired into the render loop,
//! though, by the two files that consume it:
//!
//! - `optics::raytracer::trace_spectral_ray` takes an optional `primary_hit_out:
//!   Option<&mut Option<HitRecord>>` and, at bounce 0 only, copies that ray's
//!   [`crate::optics::raytracer::HitRecord`] (`t`, `normal`, `facet_idx`) into it
//!   instead of letting it fall out of scope at the end of the bounce loop.
//! - `apps/diagram-gui/src/bridge/render_thread.rs` owns the accumulation buffer and
//!   the running sample count, and is where a persistent [`AtrousDenoiser`] lives:
//!   `render_frame_scanlines` captures each pixel's `HitRecord` into three parallel
//!   `first_hit_depth`/`first_hit_normal`/`first_hit_facet_id` buffers alongside the
//!   accumulation buffer, and `denoise_and_tonemap_frame` calls
//!   [`AtrousDenoiser::denoise_into`] with the averaged-XYZ accumulation buffer plus
//!   those three auxiliary buffers and the current sample count, tone-mapping the
//!   filtered result instead of the raw average. The accumulation buffer itself is
//!   never overwritten with filtered data -- filtering only happens on that readback
//!   path, so it never contaminates future accumulation.
//!
//! The mechanical shape of that call:
//!
//! ```ignore
//! // one persistent denoiser, created alongside the accumulation buffer:
//! let mut denoiser = AtrousDenoiser::new();
//!
//! // on every readback:
//! let inputs = GBuffers {
//!     color: &avg_color_buf,        // averaged XYZ, accum_buffer[i] / current_sample_count
//!     depth: &first_hit_depth,
//!     normal: &first_hit_normal,
//!     facet_id: &first_hit_facet_id, // -1 for background/miss pixels
//!     width,
//!     height,
//!     spp: current_sample_count,
//! };
//! denoiser.denoise_into(&inputs, &AtrousParams::default(), &mut filtered_buf);
//! // tone-map `filtered_buf` instead of `avg_color_buf`
//! ```
//!
//! The one design decision the wiring needed to resolve on its own: which of the eight
//! hero wavelengths' facet hit "wins" when they disperse to different facets on the
//! same primary ray -- this filter treats facet identity as ground truth per pixel, so
//! it needs exactly one facet id per pixel, not eight. Resolved as the HERO channel's
//! own first hit: the single traced geometric path (`current_ray` in
//! `trace_spectral_ray`) is always driven by the hero channel, so `HitRecord` at bounce
//! 0 unambiguously *is* the hero's hit already -- there is no separate "which of eight"
//! choice to make at the capture site, only a documentation one (see the bounce-0
//! capture comment in `trace_spectral_ray`).
//!
//! # Guide signals and edge-stopping
//!
//! Four terms gate every neighbour's contribution, multiplied together with the
//! spatial kernel weight:
//!
//! - **Facet identity** (dominant term). Gemstone images are almost entirely hard
//!   facet edges -- a filter that blurs across them destroys exactly the crisp
//!   boundaries that make a stone read as cut rather than melted. Facet index is a
//!   discrete signal, so it gets a hard Kronecker-delta weight: `1.0` if the
//!   neighbour's facet id matches the centre's, `0.0` otherwise (see
//!   [`facet_weight`]). No sigma, no tapering -- this term does not soften with more
//!   samples, because facet identity is deterministic geometry, not a noisy estimator.
//! - **Normal**: `max(0, dot(n_p, n_q))^normal_power` (the standard SVGF-style
//!   cosine-power weight). Mostly a safety net given facets are flat and facet-id
//!   already gates hard geometric edges, but it catches normal discontinuities the
//!   facet id alone might not resolve (e.g. any future smooth-shaded region, or an
//!   internal reflection changing the *effective* surface without changing which facet
//!   the primary ray hit). Constant across sample counts, like facet id, for the same
//!   reason: geometric, not stochastic.
//! - **Depth**: `exp(-|z_p - z_q| / sigma_depth)`. A secondary tie-breaker; since
//!   facets are planar, depth is already near-continuous within a facet and this term
//!   rarely does more than the facet/normal terms already do. Also constant across
//!   sample counts.
//! - **Colour** (the only tapered term): `exp(-||c_p - c_q||^2 / (2 * sigma_color^2))`,
//!   compared as a full XYZ vector distance rather than luminance alone, because the
//!   noise this filter targets is specifically chromatic (single-hero-wavelength
//!   speckle) -- two neighbouring pixels can have identical luminance and wildly
//!   different hue. `sigma_color` is the *only* guide sigma that scales with sample
//!   count (see below): colour is the only one of the four buffers that is actually a
//!   noisy Monte-Carlo estimator here. Depth/normal/facet id are read once at the
//!   primary hit and do not accumulate noise as more samples are taken, so tapering
//!   them would be undirected.
//!
//! # The taper curve
//!
//! Monte-Carlo estimator error scales as `O(1/sqrt(N))` for `N` samples (central limit
//! theorem, constant per-sample variance). `sigma_color` should scale with the
//! *expected magnitude of remaining noise*, not stay fixed -- otherwise at high sample
//! counts, once the true signal is already resolved, the filter keeps smearing real
//! detail (the fine scintillation detail this renderer works hard to produce) that is
//! no longer distinguishable from noise by the edge-stopping function. So:
//!
//! ```text
//! taper(spp) = 1 / sqrt(1 + spp / N0)          (N0 = TAPER_REFERENCE_SPP = 4.0)
//! sigma_color_effective(spp) = sigma_color_base * taper(spp)
//! ```
//!
//! At `spp = 0` (first sample), `taper = 1.0`: full base strength, because there is
//! nothing yet to distinguish signal from noise. At `spp = N0`, `taper ~= 0.71`. At
//! `spp = 4*N0 = 16`, `taper = 0.447`. The curve is smooth and never reaches exactly
//! zero, matching the fact that MC noise never reaches exactly zero either.
//!
//! Below the identity threshold (`taper < TAPER_IDENTITY_EPSILON = 0.02`, reached
//! around `spp ~= 10_000`) the filter short-circuits to a plain copy rather than
//! running the full pass pipeline: at that point every neighbour's colour weight is
//! numerically indistinguishable from the centre pixel's own weight-of-one, so passes
//! would burn cycles reproducing the identity function. This also gives an exact
//! (bit-identical, not just epsilon-close) identity result at high sample counts, which
//! is easy to reason about and to test.
//!
//! # Allocation
//!
//! [`AtrousDenoiser`] owns two `Vec<Vec3>` scratch buffers that it ping-pongs between
//! across passes, resizing only when the frame dimensions change. A render loop that
//! keeps one `AtrousDenoiser` alive across frames (as the intended wiring above does)
//! performs zero steady-state heap allocation in [`AtrousDenoiser::denoise`]. The
//! free function [`atrous_denoise`] is a convenience one-shot wrapper (used by this
//! module's own tests) that allocates a fresh denoiser per call; prefer the struct form
//! on any hot path.

use glam::Vec3;

/// The auxiliary ("guide") buffers the filter needs alongside the noisy colour buffer,
/// one entry per pixel, row-major (`index = y * width + x`).
///
/// All buffer slices must have exactly `width * height` elements; [`AtrousDenoiser::denoise`]
/// treats a mismatched length as a degenerate/empty input (see its docs) rather than
/// panicking.
#[derive(Clone, Copy)]
pub struct GBuffers<'a> {
    /// Averaged CIE XYZ radiance per pixel (i.e. the accumulation buffer already
    /// divided by `spp`), row-major.
    pub color: &'a [Vec3],
    /// First-hit depth (camera-space ray parameter `t`, or any monotonic distance
    /// measure) per pixel. The value used for background/miss pixels does not need to
    /// be finite-consistent with hit pixels -- non-finite depth values are treated as
    /// "no information" (contribute a neutral depth weight) rather than propagating
    /// NaNs, but a large finite sentinel (e.g. `1.0e6`) is still the recommended
    /// convention since it composes better with future gradient-aware refinements.
    pub depth: &'a [f32],
    /// First-hit shading normal per pixel, need not be pre-normalised (the filter
    /// normalises defensively).
    pub normal: &'a [Vec3],
    /// First-hit facet index per pixel, as `i32` so that background/miss pixels can be
    /// encoded as `-1` (matching the `-1` "no hit" convention already used in
    /// `gem_raytracer.wgsl`). Any negative value is treated as "no facet"; two
    /// no-facet pixels are only considered a match if their raw values are equal, so a
    /// caller that wants "all background pixels match each other" should use a single
    /// consistent sentinel such as `-1` for every miss.
    pub facet_id: &'a [i32],
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Accumulated sample count backing `color`. Drives the convergence taper --
    /// larger `spp` means less filtering.
    pub spp: u32,
}

/// Tunable parameters for the À-Trous filter. [`AtrousParams::default`] gives
/// reasonable starting points for a normalised-radiance (roughly `0..~4` XYZ Y)
/// gemstone render; scene-specific tuning is expected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtrousParams {
    /// Number of À-Trous passes. Kernel stride doubles each pass starting at 1 (so 5
    /// passes covers dilation strides 1, 2, 4, 8, 16). 4-5 is the standard range;
    /// values outside `1..=8` are clamped.
    pub num_passes: u32,
    /// Base colour edge-stopping sigma (XYZ Euclidean distance) at `spp = 0`, before
    /// the convergence taper is applied. Smaller = stricter (less blur).
    pub sigma_color: f32,
    /// Depth edge-stopping sigma, in the same units as [`GBuffers::depth`]. Smaller =
    /// stricter.
    pub sigma_depth: f32,
    /// Exponent on the clamped normal dot product (`max(0,dot)^normal_power`). Larger
    /// = stricter (SVGF commonly uses values in the 32-128 range).
    pub normal_power: f32,
    /// Reference sample count `N0` in the taper curve `1 / sqrt(1 + spp / N0)`. Smaller
    /// = the filter backs off faster as samples accumulate.
    pub taper_reference_spp: f32,
    /// Taper value below which the filter short-circuits to an exact identity copy
    /// instead of running the pass pipeline.
    pub taper_identity_epsilon: f32,
}

impl Default for AtrousParams {
    fn default() -> Self {
        Self {
            num_passes: 5,
            sigma_color: 0.35,
            sigma_depth: 0.1,
            normal_power: 64.0,
            taper_reference_spp: 4.0,
            taper_identity_epsilon: 0.02,
        }
    }
}

/// 1D B3-spline kernel taps for offsets `-2, -1, 0, 1, 2`. The 2D 5x5 kernel weight at
/// `(dx, dy)` is the separable product `B3[dx+2] * B3[dy+2]`.
const B3_SPLINE: [f32; 5] = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];

/// Small denominator guard shared by every `exp(-x / sigma)` edge-stopping term, so a
/// caller-supplied sigma of exactly zero degrades to "reject everything but an exact
/// match" instead of dividing by zero.
const SIGMA_EPS: f32 = 1.0e-8;

#[inline]
const fn sanitize(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

#[inline]
const fn sanitize_vec3(v: Vec3) -> Vec3 {
    Vec3::new(sanitize(v.x), sanitize(v.y), sanitize(v.z))
}

/// `taper(spp) = 1 / sqrt(1 + spp / N0)`. See the module docs for the derivation.
#[must_use]
pub fn taper_strength(spp: u32, taper_reference_spp: f32) -> f32 {
    let n0 = taper_reference_spp.max(SIGMA_EPS);
    1.0 / (1.0 + spp as f32 / n0).sqrt()
}

/// Hard Kronecker-delta facet weight: `1.0` on a match, `0.0` otherwise. Negative
/// values (background/miss) only match an equal negative value.
#[inline]
const fn facet_weight(a: i32, b: i32) -> f32 {
    if a == b { 1.0 } else { 0.0 }
}

/// Normalises `n` once (defensively -- see [`GBuffers::normal`]'s doc comment) into a
/// unit vector, or returns [`Vec3::ZERO`] when there is no usable normal information
/// (zero/near-zero length or non-finite components, after [`sanitize_vec3`]).
///
/// `Vec3::ZERO` doubles as that "no usable information" sentinel rather than needing a
/// separate validity buffer: a true unit vector can never have exactly zero magnitude,
/// so the two cases can't collide. Used to pre-normalise [`GBuffers::normal`] once per
/// [`AtrousDenoiser::denoise_into`] call (see that function) instead of twice (both
/// centre and neighbour) on every one of the 25 taps x `num_passes` evaluations
/// [`atrous_pixel`] used to redo this for -- same `length()` sqrt + divide, just run
/// once per pixel per call instead of up to 250 times.
#[inline]
fn unit_normal_or_zero(n: Vec3) -> Vec3 {
    let n = sanitize_vec3(n);
    let len = n.length();
    if len <= SIGMA_EPS {
        Vec3::ZERO
    } else {
        n / len
    }
}

/// `base.powf(power)` for the clamped-nonnegative cosine `base` used by
/// [`normal_weight_unit`], computed by exact repeated squaring when `power` is a
/// nonnegative power-of-two integer that fits a `u32` (in particular the
/// [`AtrousParams::default`] value `64.0`: 6 squarings, never entering `f32::powf`'s
/// general transcendental path) and falling back to [`f32::powf`] for every other value
/// (fractional powers, non-power-of-two integers, values too large to fit a `u32`).
///
/// The fast path is not required to be bit-identical to `powf` -- and generally isn't,
/// past the last couple of ULP -- only close: this weight feeds a *display* filter
/// (never the accumulation buffer itself, see this module's docs), and the fast path
/// only ever runs once the taper curve has already decided this call is above the
/// identity short-circuit (`taper >= taper_identity_epsilon`), where such differences
/// are far below anything visible. `power == 0.0` (or any non-power-of-two/fractional
/// power) falls through to `powf` unchanged, which already handles `x.powf(0.0) == 1.0`
/// for any `x` including `0.0`.
#[inline]
fn cos_pow(base: f32, power: f32) -> f32 {
    if power >= 0.0 && power.fract() == 0.0 && power <= u32::MAX as f32 {
        let int_power = power as u32;
        if int_power.is_power_of_two() {
            let squarings = int_power.trailing_zeros();
            let mut v = base;
            for _ in 0..squarings {
                v *= v;
            }
            return v;
        }
    }
    base.powf(power)
}

/// Same edge-stopping term as the module docs describe for "Normal", but taking
/// pre-normalised unit vectors (or the [`unit_normal_or_zero`] zero-sentinel) instead of
/// raw normals -- see [`AtrousDenoiser::denoise_into_with_threads`]'s precompute step.
#[inline]
fn normal_weight_unit(a_n: Vec3, b_n: Vec3, power: f32) -> f32 {
    if a_n == Vec3::ZERO || b_n == Vec3::ZERO {
        // No usable normal information on one side: treat as neutral rather than
        // rejecting, matching the pre-precompute behaviour this replaced.
        return 1.0;
    }
    let cos_theta = a_n.dot(b_n).clamp(-1.0, 1.0).max(0.0);
    cos_pow(cos_theta, power.max(0.0))
}

/// Same edge-stopping term as the module docs describe for "Depth", but taking a
/// precomputed `-1 / sigma_depth` (see [`AtrousDenoiser::denoise_into_with_threads`])
/// instead of `sigma_depth` itself, trading one `exp`-argument division per tap for one
/// multiply -- `sigma_depth` is constant across every tap of every pass within a call,
/// so the division was always recomputing the same value.
#[inline]
fn depth_weight(a: f32, b: f32, neg_inv_sigma_depth: f32) -> f32 {
    let a = sanitize(a);
    let b = sanitize(b);
    ((a - b).abs() * neg_inv_sigma_depth).exp()
}

/// Same edge-stopping term as the module docs describe for "Colour", but taking a
/// precomputed `-1 / (2 * sigma_color_effective^2)` instead of `sigma_color_effective`
/// itself, for the same reason as [`depth_weight`]'s equivalent parameter.
#[inline]
fn color_weight(a: Vec3, b: Vec3, neg_inv_two_sigma_sq: f32) -> f32 {
    let d = sanitize_vec3(a) - sanitize_vec3(b);
    let dist_sq = d.length_squared();
    (dist_sq * neg_inv_two_sigma_sq).exp()
}

/// Resolves a `--threads`-style argument (`0` meaning "let the OS decide") to an actual
/// thread count. Mirrors `gemray-worker::render_core::effective_thread_count` exactly
/// (same fallback of 8 if the OS cannot report a core count), so the two independent
/// row-chunked `thread::scope` call sites in this codebase agree on what "auto" means.
#[must_use]
fn effective_thread_count(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map_or(8, std::num::NonZero::get)
    } else {
        threads
    }
}

/// Per-call constants and scratch buffers threaded through every tap of every pass,
/// computed once by [`AtrousDenoiser::denoise_into_with_threads`] rather than being
/// re-derived (or, for `normal_n`, re-normalised) on each of the up to 250
/// (25 taps x 8 passes) evaluations a single pixel can see.
struct TapConstants<'a> {
    /// [`GBuffers::normal`], pre-normalised per pixel via [`unit_normal_or_zero`].
    normal_n: &'a [Vec3],
    /// `-1 / max(sigma_depth, SIGMA_EPS)`, folding [`depth_weight`]'s division into a
    /// multiply.
    neg_inv_sigma_depth: f32,
    /// `-1 / (2 * max(sigma_color_effective^2, SIGMA_EPS))`, folding [`color_weight`]'s
    /// division into a multiply.
    neg_inv_two_sigma_color_sq: f32,
    /// [`AtrousParams::normal_power`], unmodified (not a reciprocal -- [`cos_pow`] uses
    /// it directly).
    normal_power: f32,
}

/// Computes the filtered colour for a single output pixel at `(x, y)`. Pure function of
/// `src` and the guide buffers `g`/`tap` -- no accumulation across pixels, no ordering
/// dependency on any other pixel's result. This is the property that makes row-chunked
/// parallelisation in [`atrous_pass`] bit-identical to the single-threaded form: every
/// pixel is computed by exactly this same sequence of floating-point operations
/// regardless of which thread runs it or in what order threads are scheduled.
#[inline]
fn atrous_pixel(
    src: &[Vec3],
    g: &GBuffers<'_>,
    x: usize,
    y: usize,
    stride: i64,
    tap: &TapConstants<'_>,
) -> Vec3 {
    let width = g.width;
    let height = g.height;

    let center_idx = y * width + x;
    let c_color = src[center_idx];
    let c_depth = g.depth[center_idx];
    let c_normal_n = tap.normal_n[center_idx];
    let c_facet = g.facet_id[center_idx];

    let mut sum = Vec3::ZERO;
    let mut weight_sum = 0.0f32;

    for (ky, &hy) in B3_SPLINE.iter().enumerate() {
        let oy = (ky as i64 - 2) * stride;
        let sy = y as i64 + oy;
        if sy < 0 || sy >= height as i64 {
            continue;
        }
        for (kx, &hx) in B3_SPLINE.iter().enumerate() {
            let ox = (kx as i64 - 2) * stride;
            let sx = x as i64 + ox;
            if sx < 0 || sx >= width as i64 {
                continue;
            }

            let idx = sy as usize * width + sx as usize;
            let kernel_w = hy * hx;

            let wf = facet_weight(c_facet, g.facet_id[idx]);
            if wf == 0.0 {
                // Hard rejection: skip the (cheap) remaining term evaluation too.
                continue;
            }
            let wn = normal_weight_unit(c_normal_n, tap.normal_n[idx], tap.normal_power);
            let wd = depth_weight(c_depth, g.depth[idx], tap.neg_inv_sigma_depth);
            let wc = color_weight(c_color, src[idx], tap.neg_inv_two_sigma_color_sq);

            let w = kernel_w * wf * wn * wd * wc;
            sum += src[idx] * w;
            weight_sum += w;
        }
    }

    if weight_sum > SIGMA_EPS {
        sum / weight_sum
    } else {
        c_color
    }
}

/// Fills the rows `[start_y, start_y + dst_rows.len() / width)` of a pass's output.
/// `dst_rows` must hold exactly that many whole rows (`width` elements each) -- the
/// caller (`atrous_pass`) guarantees this via `chunks_mut(rows_per_chunk * width)`, so
/// every chunk except possibly the last divides evenly, and even the last is a multiple
/// of `width` since the total length `width * height` always is.
fn atrous_pass_row_chunk(
    src: &[Vec3],
    dst_rows: &mut [Vec3],
    g: &GBuffers<'_>,
    stride: i64,
    tap: &TapConstants<'_>,
    start_y: usize,
) {
    let width = g.width;
    for (local_y, row) in dst_rows.chunks_mut(width).enumerate() {
        let y = start_y + local_y;
        for (x, out) in row.iter_mut().enumerate() {
            *out = atrous_pixel(src, g, x, y, stride, tap);
        }
    }
}

/// Runs a single À-Trous pass at the given dilation `stride`, reading guide buffers
/// from `g` (fixed across all passes) and colour from `src`, writing the filtered
/// result into `dst`. `src` and `dst` must both have `width * height` elements and may
/// not alias.
///
/// Parallelised across `num_threads` OS threads via `std::thread::scope`, splitting the
/// image into contiguous row chunks (the established pattern in this codebase -- see
/// `apps/gemray-worker/src/render_core.rs::trace_samples` and
/// `renderer::gpu::estimator_check::cpu_samples`). Because [`atrous_pixel`] is a pure
/// function of `src`/`g` with no cross-pixel accumulation, the chunking is purely a
/// scheduling decision: the output is bit-identical for any `num_threads >= 1`,
/// including `num_threads` greater than `height` (chunks simply become smaller, some
/// possibly empty).
fn atrous_pass(
    src: &[Vec3],
    dst: &mut [Vec3],
    g: &GBuffers<'_>,
    stride: i64,
    tap: &TapConstants<'_>,
    num_threads: usize,
) {
    let width = g.width;
    let height = g.height;
    if width == 0 || height == 0 {
        return;
    }

    let num_threads = num_threads.max(1);
    let rows_per_chunk = height.div_ceil(num_threads).max(1);

    if rows_per_chunk >= height {
        // Whole image fits in one chunk (small image, or num_threads == 1): skip the
        // thread::scope machinery entirely rather than spawn a single worker for it.
        atrous_pass_row_chunk(src, dst, g, stride, tap, 0);
        return;
    }

    std::thread::scope(|s| {
        let chunk_len = rows_per_chunk * width;
        for (chunk_idx, chunk) in dst.chunks_mut(chunk_len).enumerate() {
            let start_y = chunk_idx * rows_per_chunk;
            s.spawn(move || {
                atrous_pass_row_chunk(src, chunk, g, stride, tap, start_y);
            });
        }
    });
}

/// A persistent À-Trous denoiser.
///
/// Owns two scratch buffers sized to the last-seen frame dimensions and ping-pongs
/// between them across passes, so steady-state use (one instance kept alive across
/// frames) performs no per-call heap allocation beyond the returned/written output.
#[derive(Default)]
pub struct AtrousDenoiser {
    ping: Vec<Vec3>,
    pong: Vec<Vec3>,
    /// Scratch buffer for the per-call pre-normalised normal buffer -- see
    /// [`Self::denoise_into_with_threads`] and [`unit_normal_or_zero`].
    normal_n: Vec<Vec3>,
}

impl AtrousDenoiser {
    /// Creates a denoiser with no scratch buffers allocated yet; the first call to
    /// [`Self::denoise`] sizes them.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ping: Vec::new(),
            pong: Vec::new(),
            normal_n: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, len: usize) {
        if self.ping.len() != len {
            self.ping.clear();
            self.ping.resize(len, Vec3::ZERO);
        }
        if self.pong.len() != len {
            self.pong.clear();
            self.pong.resize(len, Vec3::ZERO);
        }
        if self.normal_n.len() != len {
            self.normal_n.clear();
            self.normal_n.resize(len, Vec3::ZERO);
        }
    }

    /// Filters `inputs.color` using the auxiliary guide buffers and returns a new
    /// `Vec<Vec3>` of the same length. See [`Self::denoise_into`] for the
    /// allocation-free form.
    ///
    /// Degenerate inputs (zero-size image, mismatched buffer lengths, `num_passes ==
    /// 0`) return a plain clone of `inputs.color` (or an empty vec for a zero-size
    /// image) rather than panicking.
    #[must_use]
    pub fn denoise(&mut self, inputs: &GBuffers<'_>, params: &AtrousParams) -> Vec<Vec3> {
        let mut output = vec![Vec3::ZERO; inputs.color.len()];
        self.denoise_into(inputs, params, &mut output);
        output
    }

    /// Filters `inputs.color` into `output` in place. `output` is resized to
    /// `inputs.color.len()` if it does not already match.
    ///
    /// Robustness guarantees: never panics and never introduces NaN/infinity on its
    /// own, regardless of image size (including 0x0 and 1x1), `spp` (including 0), or
    /// buffer contents (including all-zero), *provided* `output` starts as a `Vec`
    /// (any length) rather than something that cannot be resized. A guide-buffer slice
    /// shorter than `width * height` is treated the same as "no filtering" (falls back
    /// to a copy of `color`), since the filter has no principled per-pixel guide value
    /// to use in that case.
    ///
    /// Runs each pass across all available CPU cores (see [`Self::denoise_into_with_threads`]
    /// for a pinned thread count); a single pass at 3840x2160 touches ~8.3M pixels with
    /// an inherently per-pixel-independent kernel (see [`atrous_pixel`]'s docs), so this
    /// is the difference between multi-second and sub-second passes on a modern
    /// multi-core machine.
    pub fn denoise_into(
        &mut self,
        inputs: &GBuffers<'_>,
        params: &AtrousParams,
        output: &mut Vec<Vec3>,
    ) {
        self.denoise_into_with_threads(inputs, params, output, 0);
    }

    /// Same as [`Self::denoise_into`], but with an explicit thread count instead of the
    /// auto-detected one. `threads == 0` means "let the OS decide" (i.e. what
    /// [`Self::denoise_into`] does internally), matching the `--threads`-style
    /// convention already used by `gemray-worker::render_core::effective_thread_count`.
    ///
    /// Exposed mainly so callers (and this module's own tests) can pin a thread count
    /// -- e.g. to verify thread-count invariance, or to bound worker threads in an
    /// environment that already manages its own thread pool. Every À-Trous pass is a
    /// pure per-pixel function of the previous pass's full output (see
    /// [`atrous_pixel`]), so the result is bit-identical for any `threads >= 1`: this
    /// parameter only changes how the work is scheduled across cores, never what is
    /// computed or in what order floating-point terms are summed within a pixel.
    pub fn denoise_into_with_threads(
        &mut self,
        inputs: &GBuffers<'_>,
        params: &AtrousParams,
        output: &mut Vec<Vec3>,
        threads: usize,
    ) {
        let len = inputs.width * inputs.height;
        output.clear();
        output.resize(len, Vec3::ZERO);

        if len == 0 {
            return;
        }

        // Any missing/short guide buffer -> we cannot safely index it per pixel, so
        // degrade to an identity copy rather than panicking or reading out of bounds.
        let buffers_ok = inputs.color.len() == len
            && inputs.depth.len() == len
            && inputs.normal.len() == len
            && inputs.facet_id.len() == len;
        if !buffers_ok {
            let n = inputs.color.len().min(len);
            output[..n].copy_from_slice(&inputs.color[..n]);
            for v in &mut output[n..] {
                *v = Vec3::ZERO;
            }
            return;
        }

        let taper = taper_strength(inputs.spp, params.taper_reference_spp);
        let num_passes = params.num_passes.clamp(1, 8);

        if taper < params.taper_identity_epsilon {
            output.copy_from_slice(inputs.color);
            return;
        }

        let sigma_color_effective = (params.sigma_color * taper).max(0.0);
        let num_threads = effective_thread_count(threads);

        self.ensure_capacity(len);
        self.ping.copy_from_slice(inputs.color);

        // Pre-normalise the normal buffer once per call rather than up to 250 times
        // (25 taps x up to 8 passes) per pixel -- see `unit_normal_or_zero` and
        // `TapConstants::normal_n`. `buffers_ok` above already guarantees
        // `inputs.normal.len() == len`.
        for (dst, &src_n) in self.normal_n.iter_mut().zip(inputs.normal.iter()) {
            *dst = unit_normal_or_zero(src_n);
        }

        let sigma_depth = params.sigma_depth.max(SIGMA_EPS);
        let sigma_color_sq = (sigma_color_effective * sigma_color_effective).max(SIGMA_EPS);
        let tap = TapConstants {
            normal_n: &self.normal_n,
            neg_inv_sigma_depth: -1.0 / sigma_depth,
            neg_inv_two_sigma_color_sq: -1.0 / (2.0 * sigma_color_sq),
            normal_power: params.normal_power,
        };

        for pass in 0..num_passes {
            let stride = 1i64 << pass;
            atrous_pass(
                &self.ping,
                &mut self.pong,
                inputs,
                stride,
                &tap,
                num_threads,
            );
            std::mem::swap(&mut self.ping, &mut self.pong);
        }

        output.copy_from_slice(&self.ping);
    }
}

/// One-shot convenience wrapper around [`AtrousDenoiser`].
///
/// Allocates a fresh denoiser (and therefore fresh scratch buffers) per call. Prefer
/// keeping an [`AtrousDenoiser`] alive across frames on any real render loop -- see
/// the module docs' "Intended wiring" section.
#[must_use]
pub fn atrous_denoise(inputs: &GBuffers<'_>, params: &AtrousParams) -> Vec<Vec3> {
    AtrousDenoiser::new().denoise(inputs, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift32 PRNG, matching the one in
    /// `crates/gemray/tests/denoise_tests.rs` (kept separate rather than shared since
    /// that file is a different crate as far as visibility is concerned).
    struct Xorshift32(u32);
    impl Xorshift32 {
        const fn new(seed: u32) -> Self {
            Self(if seed == 0 { 0xdead_beef } else { seed })
        }
        const fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
        fn next_f32(&mut self) -> f32 {
            (f64::from(self.next_u32()) / f64::from(u32::MAX)) as f32
        }
        fn next_signed(&mut self) -> f32 {
            self.next_f32().mul_add(2.0, -1.0)
        }
    }

    /// A non-trivial synthetic scene: irregular dimensions (so row chunks split
    /// unevenly across most thread counts), several facet regions, jittered normals,
    /// varying depth, and noisy colour -- meant to exercise every edge-stopping term
    /// rather than degenerate to a uniform image.
    fn irregular_scene(width: usize, height: usize, seed: u32) -> (GBuffers<'static>, Vec<Vec3>) {
        let len = width * height;
        let mut rng = Xorshift32::new(seed);
        let mut color = vec![Vec3::ZERO; len];
        let mut depth = vec![0.0f32; len];
        let mut normal = vec![Vec3::Z; len];
        let mut facet_id = vec![0i32; len];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                facet_id[idx] = ((x / 5 + y / 7) % 6) as i32;
                depth[idx] = rng.next_f32() * 4.0;
                let jitter =
                    Vec3::new(rng.next_signed() * 0.3, rng.next_signed() * 0.3, 1.0).normalize();
                normal[idx] = jitter;
                color[idx] = Vec3::new(rng.next_f32(), rng.next_f32(), rng.next_f32());
            }
        }
        // Leak so we can hand back `&'static` slices from owned Vecs without fighting
        // the borrow checker in a test helper; test-only, freed at process exit.
        let color_s: &'static [Vec3] = Box::leak(color.clone().into_boxed_slice());
        let depth_s: &'static [f32] = Box::leak(depth.into_boxed_slice());
        let normal_s: &'static [Vec3] = Box::leak(normal.into_boxed_slice());
        let facet_s: &'static [i32] = Box::leak(facet_id.into_boxed_slice());
        (
            GBuffers {
                color: color_s,
                depth: depth_s,
                normal: normal_s,
                facet_id: facet_s,
                width,
                height,
                spp: 1,
            },
            color,
        )
    }

    fn assert_bit_identical(a: &[Vec3], b: &[Vec3], context: &str) {
        assert_eq!(a.len(), b.len(), "{context}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.x.to_bits(),
                y.x.to_bits(),
                "{context}: pixel {i} x differs"
            );
            assert_eq!(
                x.y.to_bits(),
                y.y.to_bits(),
                "{context}: pixel {i} y differs"
            );
            assert_eq!(
                x.z.to_bits(),
                y.z.to_bits(),
                "{context}: pixel {i} z differs"
            );
        }
    }

    /// Task's core acceptance test: a single `atrous_pass` call (the loop that was
    /// parallelised) must produce bit-identical output regardless of how many threads
    /// it is given, including thread counts that don't evenly divide the row count and
    /// thread counts that exceed the row count.
    #[test]
    fn atrous_pass_is_thread_count_invariant() {
        let width = 137;
        let height = 91;
        let (g, color) = irregular_scene(width, height, 0x1234_5678);
        let params = AtrousParams::default();
        let sigma = params.sigma_color;

        let normal_n: Vec<Vec3> = g.normal.iter().map(|&n| unit_normal_or_zero(n)).collect();
        let tap = TapConstants {
            normal_n: &normal_n,
            neg_inv_sigma_depth: -1.0 / params.sigma_depth.max(SIGMA_EPS),
            neg_inv_two_sigma_color_sq: -1.0 / (2.0 * (sigma * sigma).max(SIGMA_EPS)),
            normal_power: params.normal_power,
        };

        let mut reference = vec![Vec3::ZERO; width * height];
        atrous_pass(&color, &mut reference, &g, 4, &tap, 1);

        for threads in [2usize, 3, 8, 16, 200] {
            let mut out = vec![Vec3::ZERO; width * height];
            atrous_pass(&color, &mut out, &g, 4, &tap, threads);
            assert_bit_identical(&reference, &out, &format!("threads={threads}"));
        }
    }

    /// [`cos_pow`]'s fast path (repeated squaring for a power-of-two exponent) must
    /// agree with `f32::powf` to within a small ULP tolerance, for the exact exponent
    /// [`AtrousParams::default`] uses (64) as well as a few others, and its fallback
    /// path must be bit-identical to `powf` (it just calls it) for a non-power-of-two
    /// exponent.
    #[test]
    fn cos_pow_matches_powf_within_tolerance() {
        for &power in &[1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0] {
            for &base in &[0.0f32, 0.1, 0.37, 0.5, 0.9, 1.0] {
                let fast = cos_pow(base, power);
                let reference = base.powf(power);
                let diff = (fast - reference).abs();
                assert!(
                    diff <= reference.abs().mul_add(1.0e-5, 1.0e-6),
                    "cos_pow({base}, {power}) = {fast}, powf = {reference}, diff = {diff}"
                );
            }
        }

        // Non-power-of-two exponent: must fall through to `powf` exactly (bit-identical,
        // not just close), since the fast path never engages. Iterated (rather than
        // literal `.powf(5.0)` calls) so clippy's `suboptimal_flops` lint -- which would
        // otherwise suggest `powi` here -- has no literal integer exponent to flag: the
        // point of this assertion is pinning `cos_pow`'s fallback to `powf` specifically.
        for &power in &[5.0f32, 60.0] {
            for &base in &[0.0f32, 0.3, 0.7, 1.0] {
                assert_eq!(cos_pow(base, power), base.powf(power));
            }
        }
    }

    /// Same guarantee, but through the full multi-pass pipeline (`denoise_into_with_threads`)
    /// rather than a single pass in isolation -- this is what the render loop actually
    /// calls, ping-ponging scratch buffers across all 5 default passes.
    #[test]
    fn denoise_into_is_thread_count_invariant() {
        let width = 113;
        let height = 67;
        let (g, _color) = irregular_scene(width, height, 0xabcd_ef01);
        let params = AtrousParams::default();

        let mut denoiser = AtrousDenoiser::new();
        let mut reference = Vec::new();
        denoiser.denoise_into_with_threads(&g, &params, &mut reference, 1);

        for threads in [2usize, 4, 8, 16] {
            let mut denoiser = AtrousDenoiser::new();
            let mut out = Vec::new();
            denoiser.denoise_into_with_threads(&g, &params, &mut out, threads);
            assert_bit_identical(&reference, &out, &format!("threads={threads}"));
        }

        // And the public auto-thread-count entry point must agree with the pinned
        // single-threaded reference too.
        let mut denoiser = AtrousDenoiser::new();
        let mut out = Vec::new();
        denoiser.denoise_into(&g, &params, &mut out);
        assert_bit_identical(&reference, &out, "auto thread count");
    }
}

//! Pure resolution logic for the optional local "easy" preview-then-settle rendering
//! (Task: *"optional 'easy' rendering like for remote, but only local"*).
//!
//! `bridge::handoff::HandoffMachine` already gives the REMOTE path exactly this feel:
//! while the camera moves, the viewport renders locally at low quality; once it settles,
//! a full-quality render is requested from a remote worker. This module gives the same
//! feel with no worker at all -- while the camera is moving, [`effective_dimensions`]
//! returns a fraction of the configured render resolution (per the user's
//! [`LocalPreviewScale`] choice); once it settles, it returns the configured resolution
//! unchanged. `bridge::render_thread::spawn_render_thread`'s loop feeds whatever this
//! returns straight into `update_accumulation_state`, which ALREADY reallocates every
//! accumulation/guide buffer and resets `accum_samples` whenever the dimensions it's
//! given differ from the previous frame's -- so the preview<->full transition costs
//! nothing extra to implement, it falls out of machinery that already exists (traced,
//! not assumed: see `update_accumulation_state`'s own doc comment and this crate's own
//! `render_thread::tests` for that reset condition, and `RenderContext::camera_moving`'s
//! doc comment for why "the camera has settled" is decided by the SAME `HandoffMachine`
//! instance/debounce the remote path already uses, not a second, differently-tuned one).
//!
//! No dependency on Slint or any other crate -- pure integer arithmetic, exercised
//! directly by the unit tests without spinning up a UI or a render thread, matching
//! `gui::sample_scale`/`gui::c_axis`'s own established convention for this kind of
//! helper (this one lives in `bridge` rather than `gui` because its one caller,
//! `render_thread::spawn_render_thread`'s loop, already lives here, and `bridge` has no
//! existing dependency on `gui` to introduce).

use crate::settings::model::LocalPreviewScale;

/// Resolves the `width x height` the render loop should actually trace at THIS frame,
/// from the user's configured `width x height` (the "Render Resolution" pill selector's
/// value), their [`LocalPreviewScale`] choice, and whether the camera is currently
/// `moving`.
///
/// Returns `(width, height)` unchanged whenever `scale` is [`LocalPreviewScale::Off`]
/// (`divisor() == 1`, see that method's own doc comment) or `moving` is `false` -- both
/// are the "reproduce today's behaviour exactly" cases: a user who never opens this
/// control, and a settled camera regardless of the control. Only `moving && scale !=
/// Off` actually reduces the dimensions, floored at `1x1` (via `.max(1)`) so a tiny
/// configured resolution can never resolve to a zero-area frame -- same floor
/// `settings::model::PreviewScale::resolve` already applies for the analogous remote
/// `PREVIEW`-stream case.
#[must_use]
pub fn effective_dimensions(
    width: u32,
    height: u32,
    scale: LocalPreviewScale,
    moving: bool,
) -> (u32, u32) {
    if !moving {
        return (width, height);
    }
    let divisor = scale.divisor();
    if divisor <= 1 {
        return (width, height);
    }
    ((width / divisor).max(1), (height / divisor).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_never_reduces_regardless_of_movement() {
        assert_eq!(
            effective_dimensions(1920, 1080, LocalPreviewScale::Off, true),
            (1920, 1080)
        );
        assert_eq!(
            effective_dimensions(1920, 1080, LocalPreviewScale::Off, false),
            (1920, 1080)
        );
    }

    #[test]
    fn settled_always_returns_the_full_configured_resolution() {
        for scale in [
            LocalPreviewScale::Off,
            LocalPreviewScale::Half,
            LocalPreviewScale::Quarter,
        ] {
            assert_eq!(
                effective_dimensions(1280, 720, scale, false),
                (1280, 720),
                "scale={scale:?}"
            );
        }
    }

    #[test]
    fn half_scale_halves_only_while_moving() {
        assert_eq!(
            effective_dimensions(1280, 720, LocalPreviewScale::Half, true),
            (640, 360)
        );
        assert_eq!(
            effective_dimensions(1280, 720, LocalPreviewScale::Half, false),
            (1280, 720)
        );
    }

    #[test]
    fn quarter_scale_quarters_only_while_moving() {
        assert_eq!(
            effective_dimensions(1280, 720, LocalPreviewScale::Quarter, true),
            (320, 180)
        );
        assert_eq!(
            effective_dimensions(1280, 720, LocalPreviewScale::Quarter, false),
            (1280, 720)
        );
    }

    /// A tiny configured resolution must never resolve to a zero-area preview frame --
    /// same floor `PreviewScale::resolve`'s own tests pin for the analogous remote case.
    #[test]
    fn tiny_configured_dimensions_floor_at_one_by_one_while_moving() {
        assert_eq!(
            effective_dimensions(2, 2, LocalPreviewScale::Quarter, true),
            (1, 1)
        );
        assert_eq!(
            effective_dimensions(1, 1, LocalPreviewScale::Half, true),
            (1, 1)
        );
    }

    /// Non-power-of-two configured dimensions (e.g. a hand-edited resolution) must
    /// floor-divide rather than panic or round up.
    #[test]
    fn odd_configured_dimensions_floor_divide() {
        assert_eq!(
            effective_dimensions(801, 601, LocalPreviewScale::Half, true),
            (400, 300)
        );
    }
}

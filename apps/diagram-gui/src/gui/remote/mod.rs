//! Remote rendering wiring: worker-list CRUD, "Test connection", token-based worker
//! enrollment, the global denoise toggle, and the preview-then-handoff orchestrator that
//! drives `bridge::handoff::HandoffMachine` from real camera-pose polling and dispatches
//! to `bridge::remote_render`.
//!
//! Split out of `gui::mod` purely to keep that module (already sizeable) from growing
//! further -- same reasoning as `gui::detail`/`gui::search`/`gui::sync_worker`. Further
//! split into submodules here for the same reason: [`worker_settings`] (the
//! `WorkerItem`<->`WorkerSettings` conversions), [`worker_callbacks`] (worker-list CRUD,
//! "Test connection", token redemption, the denoise toggle), and [`orchestrator`] (the
//! preview-then-handoff state machine itself, including the async guide/denoise
//! generation it drives).
//!
//! `orchestrator::poll_tick` is also the one place that decides "is the camera currently
//! moving" for `bridge::local_preview` (the local-only preview-then-settle feature) --
//! it writes `RenderContext::camera_moving` from this SAME `HandoffMachine` instance's
//! state every tick, so that feature and this one share one definition of "settled"
//! rather than each running its own, differently-tuned debounce timer.
//!
//! # What's tested vs. what isn't
//!
//! Everything this module CALLS INTO is already unit-tested without a GUI or a socket:
//! `bridge::handoff::HandoffMachine` (the state machine itself), `gemray_net::client`
//! (handshake, the epoch-gated accumulator, session framing), `bridge::enroll`
//! (bundle-directory naming, claim-failure wording), and `settings::WorkerSettings`
//! (persistence, `StreamConfig` construction). This module is the GLUE wiring those
//! tested pieces to real Slint callbacks, a real `slint::Timer`, and (via
//! `bridge::remote_render`/`setup_claim_token_callback`) a real socket -- it compiles
//! but, like the rest of the GUI layer, is not exercised by an automated test here.

mod orchestrator;
mod worker_callbacks;
mod worker_settings;

pub use orchestrator::setup_remote_rendering;
pub use worker_callbacks::setup_worker_callbacks;
pub use worker_settings::refresh_worker_options;

use crate::{gui::sample_scale, settings::LiveComputeTarget};

/// Slider bounds for the user-configurable remote sample budget, exposing what
/// used to be the hardcoded `REMOTE_RENDER_SAMPLES` constant. Wider than the local
/// interactive target's own `sample_scale::MIN_EXPONENT..=MAX_EXPONENT` (`8..=1024`
/// samples): a remote render is a one-shot converge-then-display request on
/// (potentially) much more capable hardware, not an ongoing 30-60fps interactive loop,
/// so its ceiling is deliberately set past the local slider's own -- `2^13 = 8192`
/// samples, 8x the local ceiling, for a demanding final image; the floor, `2^7 = 128`,
/// still a legitimate "full quality" one-shot for a quick connectivity check or a
/// modest worker. `RenderContext::remote_render_samples`'s default (`512`, unchanged
/// from this constant's old hardcoded value) sits comfortably inside this range.
pub const REMOTE_SAMPLES_MIN_EXPONENT: u32 = 7; // 128 samples
pub const REMOTE_SAMPLES_MAX_EXPONENT: u32 = 13; // 8192 samples

/// Converts the "Remote Render Samples" slider's EXPONENT to the actual sample count,
/// reusing `gui::sample_scale`'s bounded exponent<->count mapping (see that module's
/// doc comment for the power-of-two idiom itself) clamped to THIS control's own, wider
/// range instead of the local interactive target's `sample_scale::MIN_EXPONENT..=MAX_EXPONENT`.
#[must_use]
pub const fn remote_samples_exponent_to_count(exponent: u32) -> u32 {
    sample_scale::exponent_to_count_bounded(
        exponent,
        REMOTE_SAMPLES_MIN_EXPONENT,
        REMOTE_SAMPLES_MAX_EXPONENT,
    )
}

/// Inverse of [`remote_samples_exponent_to_count`], used once at startup to turn a
/// persisted `AppSettings::remote_render_samples` count back into the exponent the
/// slider should start at -- same round-trip role `gui::sample_scale::count_to_exponent`
/// plays for the local Target Samples slider.
#[must_use]
pub fn remote_samples_count_to_exponent(count: u32) -> u32 {
    sample_scale::count_to_exponent_bounded(
        count,
        REMOTE_SAMPLES_MIN_EXPONENT,
        REMOTE_SAMPLES_MAX_EXPONENT,
    )
}

/// `settings_dialog.slint`'s "Live Compute" pill index (0/1/2, see that property's own
/// doc comment) -> [`LiveComputeTarget`]. Mirrors `gui::render_export::
/// compute_target_from_index`'s exact int-discriminant convention for the export
/// dialog's own three-way Compute pill, duplicated rather than shared since the two
/// pickers carry different enum types (`LiveComputeTarget` here vs.
/// `bridge::export_thread::ComputeTarget` there) -- see `LiveComputeTarget`'s own doc
/// comment for why the two stay separate types.
#[must_use]
pub const fn live_compute_target_from_index(index: i32) -> LiveComputeTarget {
    match index {
        0 => LiveComputeTarget::LocalOnly,
        1 => LiveComputeTarget::RemoteOnly,
        _ => LiveComputeTarget::Both,
    }
}

/// Inverse of [`live_compute_target_from_index`], used at startup to seed the pill from
/// a persisted `AppSettings::live_compute_target` -- same round-trip role
/// `remote_samples_count_to_exponent` plays for the remote-samples slider above.
#[must_use]
pub const fn live_compute_target_index(target: LiveComputeTarget) -> i32 {
    match target {
        LiveComputeTarget::LocalOnly => 0,
        LiveComputeTarget::RemoteOnly => 1,
        LiveComputeTarget::Both => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Live rendering compute-target picker --------------------------

    #[test]
    fn live_compute_target_index_and_from_index_round_trip() {
        for target in [
            LiveComputeTarget::LocalOnly,
            LiveComputeTarget::RemoteOnly,
            LiveComputeTarget::Both,
        ] {
            let idx = live_compute_target_index(target);
            assert_eq!(live_compute_target_from_index(idx), target);
        }
    }

    #[test]
    fn live_compute_target_from_index_matches_expected_pills() {
        assert_eq!(
            live_compute_target_from_index(0),
            LiveComputeTarget::LocalOnly
        );
        assert_eq!(
            live_compute_target_from_index(1),
            LiveComputeTarget::RemoteOnly
        );
        assert_eq!(live_compute_target_from_index(2), LiveComputeTarget::Both);
    }

    #[test]
    fn live_compute_target_from_index_falls_back_to_both_for_unknown_values() {
        // Unlike `gui::mod::local_preview_scale_from_index`'s fallback to `Off`, this
        // falls back to `Both` -- matching `LiveComputeTarget::default()` and this
        // control's own `2` initial pill state, so an out-of-range value (never actually
        // reachable from the fixed 3-pill selector itself) behaves like "unset" rather
        // than silently downgrading to LocalOnly.
        assert_eq!(live_compute_target_from_index(-1), LiveComputeTarget::Both);
        assert_eq!(live_compute_target_from_index(99), LiveComputeTarget::Both);
    }

    // ---- Remote render sample budget ----------------------------------

    #[test]
    fn remote_samples_round_trip_the_default_and_endpoints() {
        assert_eq!(
            remote_samples_count_to_exponent(remote_samples_exponent_to_count(
                REMOTE_SAMPLES_MIN_EXPONENT
            )),
            REMOTE_SAMPLES_MIN_EXPONENT
        );
        assert_eq!(
            remote_samples_count_to_exponent(remote_samples_exponent_to_count(
                REMOTE_SAMPLES_MAX_EXPONENT
            )),
            REMOTE_SAMPLES_MAX_EXPONENT
        );
        // 512, this control's persisted default (`settings::model::DEFAULT_REMOTE_RENDER_SAMPLES`),
        // must resolve to SOME legal exponent inside the slider's own range and round-trip.
        let exponent = remote_samples_count_to_exponent(512);
        assert_eq!(remote_samples_exponent_to_count(exponent), 512);
    }

    #[test]
    fn remote_samples_exponent_to_count_matches_expected_powers_of_two() {
        assert_eq!(remote_samples_exponent_to_count(7), 128);
        assert_eq!(remote_samples_exponent_to_count(9), 512);
        assert_eq!(remote_samples_exponent_to_count(13), 8192);
    }

    #[test]
    fn remote_samples_range_exceeds_the_local_interactive_targets_own_ceiling() {
        // The remote budget's ceiling must be deliberately
        // larger than the local interactive target's `sample_scale::MAX_EXPONENT`
        // (1024 samples), since a remote render is a one-shot request, not an ongoing
        // interactive loop.
        assert!(remote_samples_exponent_to_count(REMOTE_SAMPLES_MAX_EXPONENT) > 1024);
    }
}

//! Adaptive sub-batch sizing: [`next_batch_size`] adapts the tracer's sub-batch sample
//! count toward [`TARGET_SUBBATCH`], given how long the previous sub-batch actually took.

use std::time::Duration;

/// The wall-clock duration [`next_batch_size`] adapts sub-batch sizes toward.
///
/// Bounds both cancellation latency (the tracer only checks its cancel flag BETWEEN
/// sub-batches -- see `run_tracer`) and scheduling granularity, on hardware ranging from
/// an A100 on a LAN to a 2060 over hotel wifi, without hardcoding a sample count that
/// would be wildly wrong for one end of that range or the other.
pub(super) const TARGET_SUBBATCH: Duration = Duration::from_millis(100);

/// Adapts the next sub-batch's sample count toward [`TARGET_SUBBATCH`], given how long
/// `prev` samples actually took to trace.
///
/// Grows (up to 4x) when the previous batch finished well under budget, shrinks toward
/// 1 when it ran over, and never returns 0 -- a hardware-agnostic controller so the
/// worker converges on a sub-batch size that fits its own actual throughput (a fast GPU
/// vs. a laptop CPU) rather than a single hardcoded sample count.
#[must_use]
pub(super) fn next_batch_size(prev: u32, elapsed: Duration) -> u32 {
    if elapsed.is_zero() {
        return prev.saturating_mul(4).max(1);
    }
    let ratio = TARGET_SUBBATCH.as_secs_f64() / elapsed.as_secs_f64();
    let scaled = f64::from(prev) * ratio;
    let max_growth = f64::from(prev.saturating_mul(4).max(1));
    scaled.clamp(1.0, max_growth).round() as u32
}

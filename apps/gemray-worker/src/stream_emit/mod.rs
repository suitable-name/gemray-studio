//! Progressive streaming: the emitter/tracer split described in `serve`'s module
//! docs.
//!
//! This module holds every piece that's testable as pure logic, independent of any
//! actual socket: [`PendingDelta`] (delta coalescing), [`next_batch_size`] (adaptive
//! sub-batch sizing), and [`downsample_preview`] (the cumulative reduced-resolution
//! snapshot `PREVIEW` sends). [`run_stream`] wires these together with a tracer thread
//! and drives the actual `Read + Write` stream; see its own doc comment and `serve`'s
//! module docs for the full architecture.

use std::{
    io::{Read, Write},
    time::Duration,
};

mod downsample;
mod emitter;
mod sizing;
#[cfg(test)]
mod tests;
mod tracer;

pub use emitter::run_stream;

/// This worker's cadence FLOOR, advertised in `WELCOME::min_cadence_ms` -- see that
/// field's doc comment. Matches [`TARGET_SUBBATCH`], the sub-batch duration
/// [`next_batch_size`] targets: emitting faster than one sub-batch completes is not
/// meaningful, since there is nothing new to send in between.
pub const MIN_CADENCE_FLOOR_MS: u32 = 100;

/// A stream that can have a short read timeout applied, so [`run_stream`]'s emitter loop
/// can poll for an incoming `CANCEL` without ever blocking longer than the timeout --
/// see `serve`'s module docs on why the emitter (not a second thread reading the same
/// socket) is what watches for `CANCEL`, interleaved with its own cadence-paced writes,
/// all from one thread that owns the stream.
///
/// Implemented for the real transports `serve::run` hands `handle_connection`
/// (`TcpStream` directly, and `rustls::StreamOwned` wrapping one, via the blanket impl
/// below) by delegating to the underlying socket. A test double that never blocks on
/// read (an in-memory buffer) can implement this to toggle its own "exhausted input
/// means `WouldBlock`, not EOF" behavior -- see `serve`'s `tests::DuplexHalf`, whose
/// `set_read_timeout` does exactly that, mirroring a real socket closely enough to
/// exercise this module's polling logic without any actual networking.
pub trait TimeoutRead {
    /// # Errors
    ///
    /// Returns whatever the underlying transport's own timeout-setting call returns.
    fn set_read_timeout(&mut self, duration: Option<Duration>) -> std::io::Result<()>;
}

impl TimeoutRead for std::net::TcpStream {
    fn set_read_timeout(&mut self, duration: Option<Duration>) -> std::io::Result<()> {
        Self::set_read_timeout(self, duration)
    }
}

impl<C, T: TimeoutRead + Read + Write> TimeoutRead for rustls::StreamOwned<C, T> {
    fn set_read_timeout(&mut self, duration: Option<Duration>) -> std::io::Result<()> {
        self.sock.set_read_timeout(duration)
    }
}

/// Mirrors `std::io::Read`/`Write`'s own blanket impls for `&mut T`, so a test can drive
/// `run_stream`/`handle_connection` through a `&mut SomeTestDouble` (needed to inspect
/// the double afterward) exactly as it already could for `Read + Write`.
impl<T: TimeoutRead + ?Sized> TimeoutRead for &mut T {
    fn set_read_timeout(&mut self, duration: Option<Duration>) -> std::io::Result<()> {
        (**self).set_read_timeout(duration)
    }
}

/// How [`run_stream`] ended, for `handle_connection`'s caller to decide what (if
/// anything) still needs to be written -- `run_stream` itself has already sent every
/// [`StreamEvent`] for the [`Completed`](StreamOutcome::Completed) case (whether the
/// request finished normally or was cancelled: both produce a `DONE`, just with a
/// different `cancelled` flag -- see [`Done`]), so only [`TracePanicked`](StreamOutcome::TracePanicked)
/// requires the caller to do anything further (send `StreamEvent::Error` with the
/// worker's own error-code convention -- see `serve`'s module docs).
///
/// This is only half of what [`run_stream`] returns -- see its own doc comment for the
/// `Option<RenderRequest>` alongside it, which the caller should start immediately
/// (rather than blocking-reading) when it's `Some`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamOutcome {
    /// The request ran to completion (`DONE { cancelled: false }`) or was cancelled
    /// (`DONE { cancelled: true }`) -- either way, every [`StreamEvent`] this request
    /// will ever produce has already been written.
    Completed,
    /// `gemray`'s tracer panicked on this (validation-passing but pathological) scene.
    /// Nothing beyond whatever `FRAME`/`PREVIEW`/`PROGRESS` had already gone out before
    /// the panic has been written -- no `DONE`, since there is no valid outcome to
    /// report. The caller is expected to send `StreamEvent::Error` itself.
    TracePanicked,
}

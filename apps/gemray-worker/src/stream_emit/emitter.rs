//! The cadence emitter and delta coalescing: [`run_stream`] owns the socket for the
//! life of one `RenderRequest`, interleaving cadence-paced `FRAME`/`PREVIEW`/`PROGRESS`
//! writes with short-timeout polls for an incoming `CANCEL` or pipelined
//! `RenderRequest`. [`PendingDelta`] is the coalescing buffer un-emitted `FRAME` deltas
//! sum into between emissions -- see `crate::serve`'s module docs for the full
//! architecture this participates in.

use super::{
    StreamOutcome, TimeoutRead,
    downsample::downsample_preview,
    tracer::{SharedState, TracerJob, run_tracer},
};
use gemray::renderer::gpu_backend::GpuBackend;
use gemray_net::{
    messages::{
        ClientMessage, Done, FrameHeader, NetError, PreviewConfig, PreviewHeader, Progress,
        RenderRequest, Stats, StreamEvent, TransferMode,
    },
    radiance,
};
use glam::Vec3;
use std::{
    io::{Read, Write},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

/// How often the emitter re-checks the socket for a pending `CANCEL` and re-evaluates
/// whether the cadence has elapsed, when the tracer hasn't signalled fresh progress in
/// the meantime. Independent of [`TARGET_SUBBATCH`] -- this is a poll interval, not a
/// batch size.
const EMITTER_POLL: Duration = Duration::from_millis(20);

/// A full-resolution radiance DELTA accumulated (coalesced) since it was last
/// [`take`](Self::take)n -- the shared buffer `run_tracer`'s sub-batches fold into and
/// `run_stream`'s emitter drains on the cadence.
///
/// Coalescing two adjacent deltas is exactly `PendingDelta::add` twice before one
/// `take`: their contributions sum elementwise and their sample ranges merge into one
/// contiguous union, so a delta that went out as two separate `FRAME`s or one coalesced
/// one carries an IDENTICAL sum either way -- see `tests::coalesced_deltas_sum_identically_to_un_coalesced_ones`.
pub(super) struct PendingDelta {
    buffer: Vec<Vec3>,
    /// `(first_sample, samples)` of the buffer's contents so far, or `None` if nothing
    /// has been folded in since the last `take`.
    range: Option<(u32, u32)>,
}

impl PendingDelta {
    pub(super) fn new(pixel_count: usize) -> Self {
        Self {
            buffer: vec![Vec3::ZERO; pixel_count],
            range: None,
        }
    }

    /// Folds `contribution` (the sum for sample sub-range `[first_sample, first_sample +
    /// samples)`) into this delta. `contribution` must immediately follow whatever range
    /// is already pending -- sub-batches are always produced in increasing sample order,
    /// so this is always true in practice; debug-asserted rather than handled, since a
    /// violation would mean `run_tracer` itself is broken, not caller-supplied data.
    pub(super) fn add(&mut self, first_sample: u32, samples: u32, contribution: &[Vec3]) {
        debug_assert_eq!(contribution.len(), self.buffer.len());
        match self.range {
            None => {
                self.buffer.copy_from_slice(contribution);
                self.range = Some((first_sample, samples));
            }
            Some((range_first, range_samples)) => {
                debug_assert_eq!(
                    first_sample,
                    range_first + range_samples,
                    "sub-batches must coalesce in contiguous sample order"
                );
                for (acc, c) in self.buffer.iter_mut().zip(contribution) {
                    *acc += *c;
                }
                self.range = Some((range_first, range_samples + samples));
            }
        }
    }

    /// Removes and returns whatever has been coalesced since the last `take`, resetting
    /// this delta to empty. `None` if nothing has been added since.
    pub(super) fn take(&mut self) -> Option<(u32, u32, Vec<Vec3>)> {
        let (first_sample, samples) = self.range.take()?;
        let len = self.buffer.len();
        let taken = std::mem::replace(&mut self.buffer, vec![Vec3::ZERO; len]);
        Some((first_sample, samples, taken))
    }
}

/// What one poll of the socket for a pending [`ClientMessage`] found.
enum ClientPoll {
    /// Nothing arrived within the poll window -- not (necessarily) an error, just
    /// "no news yet".
    Pending,
    /// A `CANCEL` for the request currently streaming.
    Cancelled,
    /// A `CANCEL` for some OTHER `request_id` -- e.g. a stale message from a request
    /// that already ended. Not honored; see `gemray_net::messages`' docs on why
    /// `request_id` is what makes this mechanical.
    Stale,
    /// The peer closed the connection.
    Closed,
    /// The client's NEXT `RenderRequest`, pipelined ahead of `DONE` for the request
    /// currently streaming -- see [`run_stream`]'s doc comment for how this is handled.
    /// Boxed for the same reason [`ClientMessage::RenderRequest`] is -- see its doc
    /// comment.
    NextRequest(Box<RenderRequest>),
}

/// Attempts to read one pending message frame, tolerating a `WouldBlock`/`TimedOut`
/// `io::Error` (from the short read timeout `run_stream` has set) as "nothing pending"
/// rather than a fatal error.
///
/// Reads the length prefix with a single `read` call (not `read_exact`) specifically so
/// a timeout can only ever land BEFORE any byte of a new message has arrived, never in
/// the middle of one -- once even one byte has been observed, this commits to blocking
/// (no more timeout tolerance) for the rest of that one message, since a message that
/// has started arriving is assumed to complete promptly. This is what keeps a timeout
/// from ever silently discarding partially-read bytes (see `std::io::Read::read_exact`'s
/// own docs: on error, how much of its buffer was actually filled is unspecified).
///
/// # Pipelining a `RENDER` ahead of `DONE` is supported
///
/// Whatever bytes arrive here while a request is streaming decode as a tagged
/// [`ClientMessage`] -- `Cancel` or `RenderRequest` -- rather than being assumed to be
/// `Cancel`, exactly the way [`StreamEvent`] lets a reader tell a mid-stream `FRAME`
/// apart from an `ERROR` on the reply side. This used to be a real gap: an earlier
/// version of this function decoded every mid-stream byte as `Cancel` unconditionally, so
/// a client that wrote its next `RenderRequest` before reading `DONE` for the current one
/// had those bytes fail to decode and its connection torn down with a spurious error --
/// discovered writing this module's own tests, in an early draft of
/// `stale_request_id_frames_are_identifiable` that queued a second `RenderRequest` right
/// behind the first on one connection.
///
/// [`ClientPoll::NextRequest`] is how that case now surfaces to [`run_stream`], which
/// treats it as an implicit cancel of the request currently streaming -- see its own doc
/// comment for why (matches how the viewer actually behaves, and removes the `DONE`
/// round trip from the drag-to-render responsiveness path entirely).
fn poll_for_client_message<S: Read + TimeoutRead>(
    stream: &mut S,
    request_id: u32,
    poll_timeout: Duration,
) -> Result<ClientPoll, NetError> {
    stream
        .set_read_timeout(Some(poll_timeout))
        .map_err(|e| NetError::Framing(gemray_net::framing::FramingError::Io(e)))?;

    let mut len_bytes = [0u8; gemray_net::framing::LEN_PREFIX_BYTES];
    let n = match stream.read(&mut len_bytes) {
        Ok(0) => return Ok(ClientPoll::Closed),
        Ok(n) => n,
        Err(e)
            if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut =>
        {
            return Ok(ClientPoll::Pending);
        }
        Err(e) => return Err(NetError::Framing(gemray_net::framing::FramingError::Io(e))),
    };

    // At least one byte of a message has arrived -- commit to reading the rest of it
    // with no further timeout tolerance.
    stream
        .set_read_timeout(None)
        .map_err(|e| NetError::Framing(gemray_net::framing::FramingError::Io(e)))?;
    if n < len_bytes.len() {
        stream
            .read_exact(&mut len_bytes[n..])
            .map_err(|e| NetError::Framing(gemray_net::framing::FramingError::Io(e)))?;
    }
    let len = u32::from_le_bytes(len_bytes);
    if len > gemray_net::framing::MAX_FRAME_LEN {
        return Err(NetError::Framing(
            gemray_net::framing::FramingError::FrameTooLarge {
                len,
                max: gemray_net::framing::MAX_FRAME_LEN,
            },
        ));
    }
    let mut payload = vec![0u8; len as usize];
    stream
        .read_exact(&mut payload)
        .map_err(|e| NetError::Framing(gemray_net::framing::FramingError::Io(e)))?;

    let msg: ClientMessage = postcard::from_bytes(&payload)?;
    Ok(match msg {
        ClientMessage::Cancel(cancel) if cancel.request_id == request_id => ClientPoll::Cancelled,
        ClientMessage::Cancel(_) => ClientPoll::Stale,
        ClientMessage::RenderRequest(next) => ClientPoll::NextRequest(next),
        // A `LibraryRequest` pipelined mid-stream, ahead of this request's `DONE`.
        // Only `RenderRequest`-vs-`RenderRequest` pipelining is part of this phase's
        // contract (see the module docs) -- a client sending a library request while a
        // render is actively streaming gets no reply for it (the same way a `Cancel`
        // for a stale `request_id` is silently ignored, just below), rather than this
        // emitter growing a second, unrelated responsibility. A future client should
        // simply wait for `DONE` before sending a `LibraryRequest` on a connection with
        // a render in flight.
        ClientMessage::Library(_) => {
            tracing::debug!(
                "received a LibraryRequest while request_id={request_id} was streaming -- library requests \
                 are not serviced mid-stream in this phase; ignoring"
            );
            ClientPoll::Stale
        }
    })
}

/// Computes [`Stats::effective_cadence_ms`]: the average wall-clock interval between
/// emissions over `elapsed`, or `0` if fewer than two emissions happened (nothing to
/// average -- see that field's doc comment).
pub(super) fn effective_cadence_ms(elapsed: Duration, emission_count: u32) -> u32 {
    if emission_count < 2 {
        0
    } else {
        (elapsed.as_millis() / u128::from(emission_count - 1)) as u32
    }
}

/// Spawns [`run_tracer`] on its own thread against a freshly built [`SharedState`], per
/// [`run_stream`]'s doc comment on why the tracer gets its own thread and shared state
/// rather than running inline.
fn spawn_tracer(
    request: &RenderRequest,
    threads: usize,
    gpu: &Arc<GpuBackend>,
) -> (
    Arc<Mutex<SharedState>>,
    Arc<AtomicBool>,
    mpsc::Receiver<()>,
    std::thread::JoinHandle<()>,
) {
    let pixel_count = request.scene.width as usize * request.scene.height as usize;
    let state = Arc::new(Mutex::new(SharedState {
        pending_delta: PendingDelta::new(pixel_count),
        running_total: vec![Vec3::ZERO; pixel_count],
        samples_done: 0,
        finished: false,
        panicked: false,
    }));
    let cancel = Arc::new(AtomicBool::new(false));
    let (progress_tx, progress_rx) = mpsc::channel::<()>();

    let job = TracerJob {
        scene: request.scene.clone(),
        first_sample: request.first_sample,
        samples: request.samples,
        threads,
    };
    let tracer_state = Arc::clone(&state);
    let tracer_cancel = Arc::clone(&cancel);
    // Cloning the `Arc` (not borrowing `gpu` itself) is what lets this run on a real
    // `std::thread::spawn` (which needs `'static`) rather than a scoped thread -- the
    // renderer it may hold behind a `Mutex` is acquired once at `serve::run` startup and
    // shared by every connection, never rebuilt per request. See `gpu_backend`'s doc
    // comment.
    let tracer_gpu = Arc::clone(gpu);
    let handle = std::thread::spawn(move || {
        run_tracer(
            job,
            &tracer_gpu,
            &tracer_state,
            &tracer_cancel,
            &progress_tx,
        );
    });

    (state, cancel, progress_rx, handle)
}

/// Writes the `DONE { cancelled: true, .. }` reply for a cancelled request -- the sole
/// payload it ever carries, per [`run_stream`]'s doc comment on why an unsent buffer is
/// discarded rather than flushed on cancellation.
fn write_cancelled_done<S: Write>(
    stream: &mut S,
    request: &RenderRequest,
    state: &Arc<Mutex<SharedState>>,
    streaming_start: Instant,
    emission_count: u32,
) -> Result<(), NetError> {
    let stats = {
        let guard = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Stats {
            samples_done: guard.samples_done,
            requested_cadence_ms: request.stream.cadence_ms,
            effective_cadence_ms: effective_cadence_ms(streaming_start.elapsed(), emission_count),
        }
    };
    gemray_net::messages::write_stream_event(
        stream,
        &StreamEvent::Done(Done {
            request_id: request.request_id,
            cancelled: true,
            stats,
        }),
        None,
    )
}

/// Streams one `RenderRequest`'s samples over `stream`, per `serve`'s module docs:
/// spawns a tracer thread (via [`run_tracer`]) that never touches
/// `stream`, then this function itself -- the emitter -- owns `stream` for the rest of
/// the request, interleaving cadence-paced `FRAME`/`PREVIEW`/`PROGRESS` writes with
/// short-timeout polls (via [`poll_for_client_message`]) for an incoming `CANCEL` or
/// pipelined `RenderRequest`, all from the one thread that calls this function. The
/// tracer is decoupled specifically so a slow `write()` (a weak link) never stalls
/// sample production (a fast GPU) -- see `serve`'s module docs.
///
/// On a normal finish, sends a final `FRAME` (the whole request, for
/// [`TransferMode::FinalOnly`]; whatever's left un-coalesced, for
/// [`TransferMode::LiveProgressive`]) followed by `DONE { cancelled: false, .. }`. On
/// cancellation (a `CANCEL` for this `request_id`, the peer closing the connection, or a
/// pipelined `RenderRequest` -- see below), discards whatever hasn't been sent yet -- see
/// [`PendingDelta`] -- and sends `DONE { cancelled: true, .. }` with NO further payload
/// message (via [`write_cancelled_done`]). If the tracer itself panics, sends nothing
/// further at all and returns [`StreamOutcome::TracePanicked`] -- see that variant's
/// docs. Restores `stream`'s read timeout to blocking (`None`) before returning either
/// way, so the caller's next (ordinary, indefinitely-blocking) read behaves exactly as it
/// did before streaming started.
///
/// Every [`StreamEvent`] this function writes carries `request.request_id`, unchanged
/// for the life of this call -- see [`gemray_net::messages`]'s docs on why that's what
/// makes a stale reply identifiable on the reading side.
///
/// # A pipelined `RenderRequest` is queued, not rejected
///
/// [`poll_for_client_message`] can hand back [`ClientPoll::NextRequest`] instead of
/// [`ClientPoll::Cancelled`] -- the client's NEXT `RenderRequest`, sent without waiting
/// for `DONE` on this one. Two ways to handle that were on the table: reject it with an
/// `ERROR` (keeping the connection open but forcing the client back to one-in-flight
/// request), or treat it as an implicit `CANCEL` of the current request immediately
/// followed by the new one. This function does the latter, because that is exactly how
/// the viewer this protocol serves actually behaves: preview-then-handoff means dragging
/// the stone produces rapid cancel-then-new-render cycles, and forcing a `DONE` round
/// trip into that path (or bouncing the pipelined request back as an error the viewer
/// would just retry after its own local `CANCEL`) buys nothing but latency. Queuing
/// costs nothing extra in complexity here since [`ClientPoll::Cancelled`] already
/// discards the pending buffer and tears down the tracer the same way -- the only
/// addition is remembering the new request to hand back to the caller.
///
/// The discard rule holds exactly as it does for an explicit `CANCEL`: the superseded
/// request's [`SharedState`] (its [`PendingDelta`] and running total) is never touched by
/// the new request at all -- it is simply dropped once this call returns, since the
/// caller starts the new request with a brand-new [`run_stream`] call, which spawns a
/// brand-new [`SharedState`] via [`spawn_tracer`]. There is no code path by which a
/// sample contributed toward the cancelled request's accumulation could reach the new
/// one's; the two `SharedState`s never coexist in the same call, and `request_id`
/// continues to tag every `FRAME`/`PREVIEW`/`PROGRESS`/`DONE` this or the next call
/// writes, so even a reply for the old request still in flight on the wire when the new
/// one starts remains mechanically identifiable as stale.
///
/// Returns the pipelined request (if any) as the second element of the tuple, for the
/// caller to start immediately -- as if it had just been read fresh off the wire --
/// instead of blocking on another read.
///
/// `gpu` is threaded straight through to [`spawn_tracer`]/[`run_tracer`], which prefers
/// it over the CPU tracer for each adaptively-sized sub-batch (see
/// `crate::render_core::trace_samples_with_gpu`) -- the same `TARGET_SUBBATCH`-targeting
/// loop and the same between-batches `cancel` check bound a GPU dispatch's cancellation
/// latency exactly as they already bound the CPU tracer's, since a GPU dispatch is just
/// one more (blocking) way to produce one sub-batch. See `gpu_backend`'s doc comment for
/// why `gpu` is an `Arc` here rather than a borrow: the tracer thread this function
/// spawns needs `'static` ownership of its own clone.
///
/// # Errors
///
/// Returns [`NetError`] for a transport-level failure -- the same conditions
/// `handle_connection`'s own request loop already propagates for.
pub fn run_stream<S: Read + Write + TimeoutRead>(
    stream: &mut S,
    request: &RenderRequest,
    threads: usize,
    gpu: &Arc<GpuBackend>,
) -> Result<(StreamOutcome, Option<RenderRequest>), NetError> {
    let (state, cancel, progress_rx, tracer_handle) = spawn_tracer(request, threads, gpu);

    let cadence = Duration::from_millis(u64::from(request.stream.cadence_ms));
    let mut last_emit = Instant::now()
        .checked_sub(cadence)
        .unwrap_or_else(Instant::now);
    let mut emission_count: u32 = 0;
    let streaming_start = Instant::now();
    let mut cancelled = false;

    // A connection-closed or transport-error poll result, a `CANCEL`, or a pipelined
    // `RenderRequest`, all set `cancel` and stop the loop -- `tracer_handle` is joined
    // exactly once, below, after the loop ends, once every exit path has had a chance
    // to request a stop.
    let mut peer_closed = false;
    // The client's next `RenderRequest`, if one arrived pipelined ahead of `DONE` for
    // this request -- see this function's doc comment on why that's queued rather than
    // rejected. Handed back to the caller once this call returns.
    let mut pipelined_next: Option<RenderRequest> = None;

    let result = loop {
        let _ = progress_rx.recv_timeout(EMITTER_POLL);

        match poll_for_client_message(stream, request.request_id, EMITTER_POLL) {
            Ok(ClientPoll::Cancelled) => {
                cancel.store(true, Ordering::Relaxed);
                cancelled = true;
            }
            Ok(ClientPoll::NextRequest(next)) => {
                // Implicit cancel-then-queue: discard this request's unsent buffer
                // exactly as an explicit CANCEL would (below), and remember `next` to
                // hand back to the caller once this call returns.
                cancel.store(true, Ordering::Relaxed);
                cancelled = true;
                pipelined_next = Some(*next);
            }
            Ok(ClientPoll::Closed) => {
                cancel.store(true, Ordering::Relaxed);
                peer_closed = true;
            }
            Ok(ClientPoll::Pending | ClientPoll::Stale) => {}
            Err(e) => {
                cancel.store(true, Ordering::Relaxed);
                break Err(e);
            }
        }

        if cancelled || peer_closed {
            break Ok(StreamOutcome::Completed);
        }

        let finished = {
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .finished
        };
        let due = last_emit.elapsed() >= cadence || finished;

        if due && !finished {
            if let Err(e) = emit_tick(stream, request, &state, &mut emission_count) {
                cancel.store(true, Ordering::Relaxed);
                break Err(e);
            }
            last_emit = Instant::now();
        }

        if finished {
            let panicked = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .panicked;
            if panicked {
                break Ok(StreamOutcome::TracePanicked);
            }
            let r = emit_final(
                stream,
                request,
                &state,
                streaming_start,
                &mut emission_count,
            );
            break r.map(|()| StreamOutcome::Completed);
        }
    };

    let _ = tracer_handle.join();

    // Restore blocking reads before handing control back to `handle_connection`'s
    // ordinary (indefinitely-blocking) read of the next `RenderRequest`.
    let _ = stream.set_read_timeout(None);

    if peer_closed {
        return Ok((StreamOutcome::Completed, None));
    }

    if cancelled {
        write_cancelled_done(stream, request, &state, streaming_start, emission_count)?;
        // `pipelined_next` is `None` for an explicit CANCEL (nothing queued) and
        // `Some` for a pipelined RenderRequest (see `ClientPoll::NextRequest` above) --
        // either way, `state` (this request's PendingDelta/running total) is dropped
        // right here, never handed to whatever comes next.
        return Ok((StreamOutcome::Completed, pipelined_next));
    }

    result.map(|outcome| (outcome, None))
}

/// One periodic (cadence-elapsed) emission: `FRAME` (if [`TransferMode::LiveProgressive`]
/// and there's a pending delta), `PREVIEW` (if configured), then `PROGRESS` -- always,
/// even under [`TransferMode::FinalOnly`], so a viewer has SOME live feedback regardless
/// of transfer mode.
fn emit_tick<S: Write>(
    stream: &mut S,
    request: &RenderRequest,
    state: &Arc<Mutex<SharedState>>,
    emission_count: &mut u32,
) -> Result<(), NetError> {
    let (delta, preview_source, samples_done) = {
        let mut guard = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let delta = if matches!(request.stream.transfer_mode, TransferMode::LiveProgressive) {
            guard.pending_delta.take()
        } else {
            None
        };
        let preview_source = request
            .stream
            .preview
            .map(|cfg| (cfg, guard.running_total.clone()));
        (delta, preview_source, guard.samples_done)
    };

    if let Some((first_sample, samples, buffer)) = delta {
        let xyz_bytes = radiance::encode(&buffer);
        let header =
            FrameHeader::for_payload(request.request_id, first_sample, samples, &xyz_bytes);
        gemray_net::messages::write_stream_event(
            stream,
            &StreamEvent::Frame(header),
            Some(&xyz_bytes),
        )?;
        *emission_count += 1;
    }

    if let Some((cfg, running_total)) = preview_source {
        write_preview(stream, request, cfg, &running_total, samples_done)?;
    }

    gemray_net::messages::write_stream_event(
        stream,
        &StreamEvent::Progress(Progress {
            request_id: request.request_id,
            samples_done,
        }),
        None,
    )?;
    *emission_count += 1;

    Ok(())
}

/// The final emission once the tracer has finished (without cancellation or a panic):
/// the last `FRAME` -- the whole request for [`TransferMode::FinalOnly`], or whatever's
/// left un-coalesced for [`TransferMode::LiveProgressive`] -- then `DONE { cancelled: false }`.
fn emit_final<S: Write>(
    stream: &mut S,
    request: &RenderRequest,
    state: &Arc<Mutex<SharedState>>,
    streaming_start: Instant,
    emission_count: &mut u32,
) -> Result<(), NetError> {
    match request.stream.transfer_mode {
        TransferMode::FinalOnly => {
            let running_total = {
                state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .running_total
                    .clone()
            };
            let xyz_bytes = radiance::encode(&running_total);
            let header = FrameHeader::for_payload(
                request.request_id,
                request.first_sample,
                request.samples,
                &xyz_bytes,
            );
            gemray_net::messages::write_stream_event(
                stream,
                &StreamEvent::Frame(header),
                Some(&xyz_bytes),
            )?;
            *emission_count += 1;
        }
        TransferMode::LiveProgressive => {
            let remaining = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pending_delta
                .take();
            if let Some((first_sample, samples, buffer)) = remaining {
                let xyz_bytes = radiance::encode(&buffer);
                let header =
                    FrameHeader::for_payload(request.request_id, first_sample, samples, &xyz_bytes);
                gemray_net::messages::write_stream_event(
                    stream,
                    &StreamEvent::Frame(header),
                    Some(&xyz_bytes),
                )?;
                *emission_count += 1;
            }
        }
    }

    let stats = Stats {
        samples_done: request.samples,
        requested_cadence_ms: request.stream.cadence_ms,
        effective_cadence_ms: effective_cadence_ms(streaming_start.elapsed(), *emission_count),
    };
    gemray_net::messages::write_stream_event(
        stream,
        &StreamEvent::Done(Done {
            request_id: request.request_id,
            cancelled: false,
            stats,
        }),
        None,
    )?;
    Ok(())
}

fn write_preview<S: Write>(
    stream: &mut S,
    request: &RenderRequest,
    cfg: PreviewConfig,
    running_total: &[Vec3],
    samples_done: u32,
) -> Result<(), NetError> {
    let preview_buffer = downsample_preview(
        running_total,
        request.scene.width,
        request.scene.height,
        cfg.width,
        cfg.height,
    );
    let xyz_bytes = radiance::encode(&preview_buffer);
    let header = PreviewHeader::for_payload(
        request.request_id,
        cfg.width,
        cfg.height,
        samples_done,
        &xyz_bytes,
    );
    gemray_net::messages::write_stream_event(
        stream,
        &StreamEvent::Preview(header),
        Some(&xyz_bytes),
    )
}

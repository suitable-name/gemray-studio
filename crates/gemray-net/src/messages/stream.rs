//! Every message a client may send after the handshake ([`ClientMessage`]), and every
//! reply a worker sends for one `RENDER` request ([`StreamEvent`]) -- plus the small
//! per-message structs both are built from, and [`ErrorMsg`], used well beyond just
//! this file (handshake refusal, library-request errors).
//!
//! ```text
//! -> CANCEL   { request_id }
//! <- FRAME    { request_id, first_sample, samples, payload_len, xyz_bytes }   -- DELTA, full-res
//! <- PREVIEW  { request_id, width, height, samples_done, payload_len, xyz_bytes } -- CUMULATIVE, reduced-res
//! <- PROGRESS { request_id, samples_done }
//! <- DONE     { request_id, cancelled, stats: Stats }
//! <- ERROR    { code, message }
//! ```
//!
//! `FRAME` and `PREVIEW` are different from the rest: their `xyz_bytes` payload is the
//! raw POD radiance buffer from [`crate::radiance`], which must NOT go through
//! `postcard` (see that module's docs for why). So each is sent as two consecutive
//! frames -- a small `postcard`-encoded header, then the raw radiance bytes verbatim --
//! via [`write_frame_message`] / [`read_frame_message`] (for `FRAME`) and
//! [`write_preview_message`] / [`read_preview_message`] (for `PREVIEW`), both of which
//! cross-check the header's declared `payload_len` against the actual byte count read.
//!
//! # `FRAME` (delta, full resolution) vs `PREVIEW` (cumulative, reduced resolution)
//!
//! These are NOT interchangeable, and the distinction is load-bearing for correctness,
//! not just a naming choice:
//!
//! - `FRAME` carries a DELTA -- the summed contribution of exactly the sample sub-range
//!   named by [`FrameHeader::first_sample`]/[`FrameHeader::samples`], at full
//!   resolution. A viewer's accumulator SUMS every `FRAME` it receives (whose
//!   `request_id` matches its current epoch -- see below) straight into its running
//!   total. This is what makes deltas coalesce losslessly under backpressure (see
//!   `gemray-worker::serve`'s streaming docs): two adjacent, not-yet-sent deltas sum to
//!   one delta over their union, and the sum is identical whether it went out as one
//!   `FRAME` or several.
//! - `PREVIEW` carries a CUMULATIVE, reduced-resolution snapshot -- the FULL running
//!   total so far (not a delta), downsampled to [`PreviewHeader::width`] x
//!   [`PreviewHeader::height`]. Each `PREVIEW` SUPERSEDES the previous one; a viewer
//!   never sums two `PREVIEW`s together, and never sums a `PREVIEW` into the full-
//!   resolution accumulator -- a reduced-resolution buffer is not additive with a
//!   full-resolution one under any arithmetic. This is also what makes `PREVIEW`
//!   freely droppable under backpressure: losing one just means the next one (which
//!   already reflects everything the dropped one did, plus more) arrives instead.
//!
//! # `request_id` and cancellation epochs
//!
//! Every message from `RENDER` onward -- `FrameHeader`, `PreviewHeader`, `Progress`,
//! `Cancel`, `Done` -- echoes the `request_id` the client chose for that `RENDER`. This
//! is what makes "never merge a stale partial into the next render" mechanical on the
//! client side: a `CANCEL` can be in flight past a worker that's already mid-batch, so
//! `FRAME`/`PREVIEW` payloads for the just-cancelled request may still be in transit
//! when the client moves on to its next request. The rule is simply "sum/display a
//! payload iff its `request_id` matches the current epoch; drop everything else" --
//! no connection-drop-and-reconnect required (see `gemray-worker::serve`'s module docs
//! for why a `CANCEL` message, not a socket close, is how cancellation works here).
//!
//! # [`ClientMessage`]: why every post-handshake client message is tagged
//!
//! v1 got away with a fixed reply shape and no tag on the request side either. That
//! broke down once a client could pipeline its next `RenderRequest` without first
//! waiting for `DONE` on the one currently streaming (see `gemray-worker::stream_emit`'s
//! module docs for why a well-behaved viewer does exactly that): whatever bytes arrive
//! while a request is streaming could then be EITHER a `CANCEL` for it or the next
//! `RenderRequest`. [`ClientMessage`] resolves that the same way [`StreamEvent`]
//! resolves the analogous ambiguity on the reply side -- a tag says which it is, never
//! left to be inferred from position.
//!
//! [`ClientMessage::Library`] extends this to the read-only design-library protocol
//! (see [`crate::library`]) -- the SAME tagged envelope, so a worker's post-handshake
//! read loop has exactly one message type to decode regardless of which family the
//! bytes turn out to belong to.
//!
//! **Variant order is deliberately NOT source order.** [`ClientMessage::Cancel`] and
//! [`ClientMessage::Library`] come first and are ALWAYS compiled in, at the SAME
//! `postcard` discriminant regardless of whether this crate's `render` feature is on --
//! only [`ClientMessage::RenderRequest`], last and `cfg`-gated, has a discriminant that
//! exists only in a `render`-enabled build. This is what keeps a library-only worker
//! and a full worker wire-compatible for everything BOTH of them support: two peers
//! built with different feature flags still agree on tags 0 and 1 no matter what, and
//! only ever exchange tag 2 with a peer whose `WELCOME` already proved it has render
//! capacity (see `super::hello::Welcome::render`). Reordering these variants, or adding
//! a new always-on one after the `cfg`-gated tail, would silently break that.

use serde::{Deserialize, Serialize};

/// The `postcard`-encoded header half of a `<- FRAME` message: a DELTA, at full
/// resolution -- see the module docs for the contrast with [`PreviewHeader`].
///
/// See the module docs for why the radiance payload itself travels as a second, raw
/// (non-`postcard`) frame rather than being embedded here as a `Vec<u8>` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameHeader {
    pub request_id: u32,
    pub first_sample: u32,
    pub samples: u32,
    pub payload_len: u32,
}

impl FrameHeader {
    /// Builds a header whose `payload_len` is derived from `xyz_bytes`, so it can never
    /// disagree with the payload it's paired with in [`write_frame_message`].
    #[must_use]
    pub const fn for_payload(
        request_id: u32,
        first_sample: u32,
        samples: u32,
        xyz_bytes: &[u8],
    ) -> Self {
        Self {
            request_id,
            first_sample,
            samples,
            payload_len: xyz_bytes.len() as u32,
        }
    }
}

/// The `postcard`-encoded header half of a `<- PREVIEW` message: a CUMULATIVE,
/// reduced-resolution snapshot -- see the module docs for the contrast with [`FrameHeader`].
///
/// `samples_done` is the total sample count the running total behind this snapshot
/// reflects (the whole request's progress so far, not a sub-range), which is what a
/// viewer needs to normalize the sum into a displayable average without depending on a
/// `PROGRESS` message having arrived first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreviewHeader {
    pub request_id: u32,
    pub width: u32,
    pub height: u32,
    pub samples_done: u32,
    pub payload_len: u32,
}

impl PreviewHeader {
    /// Builds a header whose `payload_len` is derived from `xyz_bytes`, so it can never
    /// disagree with the payload it's paired with in [`write_preview_message`].
    #[must_use]
    pub const fn for_payload(
        request_id: u32,
        width: u32,
        height: u32,
        samples_done: u32,
        xyz_bytes: &[u8],
    ) -> Self {
        Self {
            request_id,
            width,
            height,
            samples_done,
            payload_len: xyz_bytes.len() as u32,
        }
    }
}

/// `<- PROGRESS`: a lightweight, cadence-paced progress ping.
///
/// Sent even under `TransferMode::FinalOnly` (where `FRAME` itself only arrives once at
/// the end), so a viewer always has SOME live feedback regardless of transfer mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    pub request_id: u32,
    pub samples_done: u32,
}

/// `-> CANCEL`: asks the worker to stop tracing `request_id` as soon as possible,
/// without dropping the connection.
///
/// A connection drop would force a full mutual-TLS re-handshake (plus `HELLO`/`WELCOME`
/// re-verification) on the very next request -- unacceptable when the triggering
/// gesture is camera manipulation, which can fire several times a second. See
/// `gemray-worker::serve`'s module docs for the worker-side cooperative-cancellation
/// implementation this message drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cancel {
    pub request_id: u32,
}

/// Every message a client may send after the handshake, tagged so a worker always
/// knows which follows -- never left to be inferred from position alone.
///
/// See the module doc comment for the full argument, and for why variant ORDER here is
/// load-bearing across differently-`cfg`-feature-flagged builds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClientMessage {
    Cancel(Cancel),
    /// A read-only design-library request -- see [`crate::library::LibraryRequest`].
    /// Always available, regardless of this crate's `render` feature.
    Library(Box<crate::library::LibraryRequest>),
    /// `RenderRequest` is boxed purely to keep this enum's own stack footprint close to
    /// `Cancel`'s tiny one rather than every `ClientMessage` paying for `RenderRequest`'s
    /// much larger `SceneState` -- serde boxes and unboxes it transparently, so this has
    /// no effect on the wire encoding.
    ///
    /// Only compiled in under this crate's `render` feature, and deliberately the LAST
    /// variant -- see the module doc comment.
    #[cfg(feature = "render")]
    RenderRequest(Box<super::render::RenderRequest>),
}

/// Delivery statistics reported on [`Done`].
///
/// Lets a viewer surface something like "requested 250ms, delivering ~1.4s --
/// link-limited" rather than leaving a user confused about why updates feel slower than
/// what they asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    /// Total samples actually traced before this request ended (by finishing or by
    /// cancellation).
    pub samples_done: u32,
    /// Echoes the request's own `StreamConfig::cadence_ms`, so a viewer doesn't need to
    /// have kept its own request around to compare against.
    pub requested_cadence_ms: u32,
    /// The actual average interval between emissions over the life of this request, in
    /// milliseconds. Larger than `requested_cadence_ms` when the link or hardware
    /// couldn't sustain the requested cadence (deltas coalesced under backpressure --
    /// see `gemray-worker::serve`'s streaming docs); `0` if this request never emitted
    /// more than once (nothing to average).
    pub effective_cadence_ms: u32,
}

/// `<- DONE`: the terminal message for a `RENDER` request, exactly once.
///
/// Either it finished (`cancelled: false`) or a `CANCEL` was honored (`cancelled: true`,
/// in which case no further `FRAME`/`PREVIEW` payload for this `request_id` follows: see
/// `gemray-worker::serve`'s module docs on why an unsent buffer is discarded rather than
/// flushed on cancellation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Done {
    pub request_id: u32,
    pub cancelled: bool,
    pub stats: Stats,
}

/// `<- ERROR`: a worker's rejection of a request (or `HELLO`), with a machine-readable
/// `code` and a human-readable `message`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorMsg {
    pub code: u32,
    pub message: String,
}

/// Every reply a worker sends for one `RENDER` request, from the first one through
/// `Done`/`Error`, tagged so a reader always knows which follows.
///
/// Never left to be inferred from position alone -- see the module docs for the full
/// argument.
///
/// `Frame` and `Preview` carry only the small header here; the raw radiance payload
/// that goes with each follows as a second, separate raw frame -- see
/// [`write_stream_event`] / [`read_stream_event`], the sole way this variant is meant to
/// go over the wire (the module docs' `FRAME`/`PREVIEW` sketch is this enum in
/// practice).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamEvent {
    Frame(FrameHeader),
    Preview(PreviewHeader),
    Progress(Progress),
    Done(Done),
    Error(ErrorMsg),
}

/// Writes one [`StreamEvent`] reply.
///
/// `payload` must be `Some` for [`StreamEvent::Frame`]/[`StreamEvent::Preview`] (the raw
/// radiance bytes the header describes) and `None` for every other variant --
/// debug-asserted, since getting this wrong is a caller bug: [`StreamEvent`]'s own tag
/// is what tells [`read_stream_event`] whether to expect a payload frame at all, so a
/// mismatch here would silently corrupt the stream for whatever's read next.
///
/// # Errors
///
/// Returns [`super::codec::NetError::Postcard`] or [`super::codec::NetError::Framing`]
/// under the same conditions as [`super::codec::write_message`] (for the event) and
/// [`crate::framing::write_frame`] (for the raw payload, when present).
pub fn write_stream_event<W: std::io::Write>(
    writer: &mut W,
    event: &StreamEvent,
    payload: Option<&[u8]>,
) -> Result<(), super::codec::NetError> {
    debug_assert_eq!(
        matches!(event, StreamEvent::Frame(_) | StreamEvent::Preview(_)),
        payload.is_some(),
        "StreamEvent::Frame/Preview must carry a payload; every other variant must not"
    );
    super::codec::write_message(writer, event)?;
    if let Some(bytes) = payload {
        crate::framing::write_frame(writer, bytes)?;
    }
    Ok(())
}

/// Reads one [`StreamEvent`] reply written by [`write_stream_event`]. The inverse of
/// [`write_stream_event`].
///
/// Includes its raw payload frame when the decoded variant is
/// [`StreamEvent::Frame`]/[`StreamEvent::Preview`] (validated against that header's own
/// declared `payload_len`, same as [`read_frame_message`]/[`read_preview_message`]).
///
/// # Errors
///
/// Returns [`super::codec::NetError::Postcard`] or [`super::codec::NetError::Framing`]
/// under the same conditions as [`super::codec::read_message`] (for the event) and
/// [`crate::framing::read_frame`] (for the raw payload, when present), or
/// [`super::codec::NetError::FramePayloadLenMismatch`] if a `Frame`/`Preview` header's
/// declared `payload_len` disagrees with the raw payload frame's actual length.
pub fn read_stream_event<R: std::io::Read>(
    reader: &mut R,
) -> Result<(StreamEvent, Option<Vec<u8>>), super::codec::NetError> {
    let event: StreamEvent = super::codec::read_message(reader)?;
    let expected_len = match &event {
        StreamEvent::Frame(h) => Some(h.payload_len),
        StreamEvent::Preview(h) => Some(h.payload_len),
        StreamEvent::Progress(_) | StreamEvent::Done(_) | StreamEvent::Error(_) => None,
    };
    let payload = match expected_len {
        Some(expected) => {
            let bytes = crate::framing::read_frame(reader)?;
            if bytes.len() as u32 != expected {
                return Err(super::codec::NetError::FramePayloadLenMismatch {
                    declared: expected,
                    actual: bytes.len(),
                });
            }
            Some(bytes)
        }
        None => None,
    };
    Ok((event, payload))
}

/// Writes a `<- FRAME` message: a `postcard`-encoded [`FrameHeader`] frame, followed by
/// a second frame carrying `xyz_bytes` completely raw.
///
/// See the module docs for why the radiance payload bypasses `postcard`.
/// `header.payload_len` must equal `xyz_bytes.len()`; this is asserted by the caller's
/// construction, not re-derived here, so `header` and `xyz_bytes` cannot silently drift
/// apart -- use [`FrameHeader::for_payload`] to build a consistent pair.
///
/// # Errors
///
/// Returns [`super::codec::NetError::Postcard`] or [`super::codec::NetError::Framing`]
/// under the same conditions as [`super::codec::write_message`] (for the header) and
/// [`crate::framing::write_frame`] (for the raw payload).
pub fn write_frame_message<W: std::io::Write>(
    writer: &mut W,
    header: &FrameHeader,
    xyz_bytes: &[u8],
) -> Result<(), super::codec::NetError> {
    super::codec::write_message(writer, header)?;
    crate::framing::write_frame(writer, xyz_bytes)?;
    Ok(())
}

/// Reads a `<- FRAME` message written by [`write_frame_message`], validating that the
/// header's declared `payload_len` matches the number of bytes the raw payload frame
/// actually carried before returning either.
///
/// # Errors
///
/// See [`write_frame_message`]'s errors, plus
/// [`super::codec::NetError::FramePayloadLenMismatch`] if the header's declared
/// `payload_len` disagrees with the raw payload frame's actual length.
pub fn read_frame_message<R: std::io::Read>(
    reader: &mut R,
) -> Result<(FrameHeader, Vec<u8>), super::codec::NetError> {
    let header: FrameHeader = super::codec::read_message(reader)?;
    let payload = crate::framing::read_frame(reader)?;
    if payload.len() as u32 != header.payload_len {
        return Err(super::codec::NetError::FramePayloadLenMismatch {
            declared: header.payload_len,
            actual: payload.len(),
        });
    }
    Ok((header, payload))
}

/// Writes a `<- PREVIEW` message.
///
/// A `postcard`-encoded [`PreviewHeader`] frame, followed by a second frame carrying
/// `xyz_bytes` (a reduced-resolution radiance buffer, encoded the same way as a `FRAME`
/// payload via [`crate::radiance::encode`]) completely raw. See the module docs for why
/// a `PREVIEW` payload is CUMULATIVE, never a delta, and must never be summed the way a
/// `FRAME` payload is.
///
/// # Errors
///
/// See [`write_frame_message`]'s errors.
pub fn write_preview_message<W: std::io::Write>(
    writer: &mut W,
    header: &PreviewHeader,
    xyz_bytes: &[u8],
) -> Result<(), super::codec::NetError> {
    super::codec::write_message(writer, header)?;
    crate::framing::write_frame(writer, xyz_bytes)?;
    Ok(())
}

/// Reads a `<- PREVIEW` message written by [`write_preview_message`]. The inverse of
/// [`write_preview_message`].
///
/// # Errors
///
/// See [`read_frame_message`]'s errors.
pub fn read_preview_message<R: std::io::Read>(
    reader: &mut R,
) -> Result<(PreviewHeader, Vec<u8>), super::codec::NetError> {
    let header: PreviewHeader = super::codec::read_message(reader)?;
    let payload = crate::framing::read_frame(reader)?;
    if payload.len() as u32 != header.payload_len {
        return Err(super::codec::NetError::FramePayloadLenMismatch {
            declared: header.payload_len,
            actual: payload.len(),
        });
    }
    Ok((header, payload))
}

#[cfg(test)]
mod tests {
    use super::{
        super::codec::{read_message, write_message},
        *,
    };

    #[test]
    fn error_round_trips() {
        let err = ErrorMsg {
            code: 42,
            message: "scene exceeds max_pixels".to_string(),
        };
        let mut buf = Vec::new();
        write_message(&mut buf, &err).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let decoded: ErrorMsg = read_message(&mut cursor).unwrap();
        assert_eq!(err, decoded);
    }

    #[test]
    fn frame_message_round_trips_and_validates_payload_len() {
        let xyz_bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let header = FrameHeader::for_payload(7, 64, 32, &xyz_bytes);
        assert_eq!(header.payload_len, xyz_bytes.len() as u32);
        assert_eq!(header.request_id, 7);

        let mut buf = Vec::new();
        write_frame_message(&mut buf, &header, &xyz_bytes).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let (decoded_header, decoded_bytes) = read_frame_message(&mut cursor).unwrap();
        assert_eq!(decoded_header, header);
        assert_eq!(decoded_bytes, xyz_bytes);
    }

    #[test]
    fn frame_message_rejects_a_forged_payload_len() {
        let xyz_bytes = vec![0u8; 12];
        let lying_header = FrameHeader {
            request_id: 1,
            first_sample: 0,
            samples: 1,
            payload_len: 999,
        };

        let mut buf = Vec::new();
        write_frame_message(&mut buf, &lying_header, &xyz_bytes).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let result = read_frame_message(&mut cursor);
        assert!(matches!(
            result,
            Err(super::super::codec::NetError::FramePayloadLenMismatch {
                declared: 999,
                actual: 12
            })
        ));
    }

    #[test]
    fn preview_message_round_trips_and_validates_payload_len() {
        let xyz_bytes = vec![9u8; 24];
        let header = PreviewHeader::for_payload(7, 4, 2, 128, &xyz_bytes);
        assert_eq!(header.payload_len, xyz_bytes.len() as u32);

        let mut buf = Vec::new();
        write_preview_message(&mut buf, &header, &xyz_bytes).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let (decoded_header, decoded_bytes) = read_preview_message(&mut cursor).unwrap();
        assert_eq!(decoded_header, header);
        assert_eq!(decoded_bytes, xyz_bytes);
    }

    #[test]
    fn preview_message_rejects_a_forged_payload_len() {
        let xyz_bytes = vec![0u8; 24];
        let lying_header = PreviewHeader {
            request_id: 1,
            width: 4,
            height: 2,
            samples_done: 8,
            payload_len: 999,
        };

        let mut buf = Vec::new();
        write_preview_message(&mut buf, &lying_header, &xyz_bytes).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let result = read_preview_message(&mut cursor);
        assert!(matches!(
            result,
            Err(super::super::codec::NetError::FramePayloadLenMismatch {
                declared: 999,
                actual: 24
            })
        ));
    }

    #[test]
    fn progress_round_trips() {
        let progress = Progress {
            request_id: 42,
            samples_done: 128,
        };
        let mut buf = Vec::new();
        write_message(&mut buf, &progress).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let decoded: Progress = read_message(&mut cursor).unwrap();
        assert_eq!(progress, decoded);
    }

    #[test]
    fn cancel_round_trips() {
        let cancel = Cancel { request_id: 42 };
        let mut buf = Vec::new();
        write_message(&mut buf, &cancel).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let decoded: Cancel = read_message(&mut cursor).unwrap();
        assert_eq!(cancel, decoded);
    }

    #[test]
    fn done_round_trips_both_cancelled_states() {
        for cancelled in [false, true] {
            let done = Done {
                request_id: 42,
                cancelled,
                stats: Stats {
                    samples_done: 256,
                    requested_cadence_ms: 250,
                    effective_cadence_ms: 1400,
                },
            };
            let mut buf = Vec::new();
            write_message(&mut buf, &done).unwrap();
            let mut cursor = std::io::Cursor::new(buf);
            let decoded: Done = read_message(&mut cursor).unwrap();
            assert_eq!(done, decoded);
        }
    }

    #[test]
    fn client_message_cancel_round_trips() {
        let msg = ClientMessage::Cancel(Cancel { request_id: 7 });
        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let decoded: ClientMessage = read_message(&mut cursor).unwrap();
        assert_eq!(decoded, msg);
    }

    #[cfg(feature = "render")]
    #[test]
    fn client_message_render_request_round_trips() {
        use gemray::{
            geometry::cuts::StandardGemCuts,
            optics::{materials::GemMaterial, raytracer::LightingPreset},
        };

        let scene = crate::scene::SceneState {
            width: 4,
            height: 4,
            yaw: 0.4,
            pitch: 0.3,
            distance: 3.0,
            light_yaw: 0.85,
            light_pitch: 0.95,
            exposure: 1.0,
            max_bounces: 4,
            lighting_preset: LightingPreset::Daylight,
            material: GemMaterial::diamond(),
            planes: StandardGemCuts::standard_round_brilliant(),
            girdle_frosted: false,
        };
        let msg = ClientMessage::RenderRequest(Box::new(super::super::render::RenderRequest {
            request_id: 8,
            scene,
            first_sample: 0,
            samples: 4,
            stream: super::super::render::StreamConfig {
                transfer_mode: super::super::render::TransferMode::FinalOnly,
                cadence_ms: 100,
                preview: None,
            },
        }));
        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let decoded: ClientMessage = read_message(&mut cursor).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn stream_event_frame_and_preview_round_trip_with_their_payload() {
        let xyz_bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let frame_header = FrameHeader::for_payload(7, 0, 4, &xyz_bytes);
        let event = StreamEvent::Frame(frame_header);
        let mut buf = Vec::new();
        write_stream_event(&mut buf, &event, Some(&xyz_bytes)).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let (decoded_event, decoded_payload) = read_stream_event(&mut cursor).unwrap();
        assert_eq!(decoded_event, event);
        assert_eq!(decoded_payload, Some(xyz_bytes.clone()));

        let preview_header = PreviewHeader::for_payload(7, 2, 2, 16, &xyz_bytes);
        let event = StreamEvent::Preview(preview_header);
        let mut buf = Vec::new();
        write_stream_event(&mut buf, &event, Some(&xyz_bytes)).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let (decoded_event, decoded_payload) = read_stream_event(&mut cursor).unwrap();
        assert_eq!(decoded_event, event);
        assert_eq!(decoded_payload, Some(xyz_bytes));
    }

    #[test]
    fn stream_event_progress_done_and_error_round_trip_with_no_payload() {
        for event in [
            StreamEvent::Progress(Progress {
                request_id: 7,
                samples_done: 64,
            }),
            StreamEvent::Done(Done {
                request_id: 7,
                cancelled: false,
                stats: Stats {
                    samples_done: 64,
                    requested_cadence_ms: 250,
                    effective_cadence_ms: 300,
                },
            }),
            StreamEvent::Error(ErrorMsg {
                code: 3,
                message: "internal error while tracing this request".to_string(),
            }),
        ] {
            let mut buf = Vec::new();
            write_stream_event(&mut buf, &event, None).unwrap();
            let mut cursor = std::io::Cursor::new(buf);
            let (decoded_event, decoded_payload) = read_stream_event(&mut cursor).unwrap();
            assert_eq!(decoded_event, event);
            assert_eq!(decoded_payload, None);
        }
    }

    #[test]
    fn stream_event_a_sequence_reads_back_in_order() {
        // Simulates one LiveProgressive request's worth of replies: two FRAMEs, a
        // PROGRESS, then DONE -- exactly the kind of variable-length, interleaved
        // sequence a tagged StreamEvent exists to make readable without the reader
        // having to guess what comes next from position alone.
        let bytes_a = vec![0u8; 12];
        let bytes_b = vec![1u8; 12];
        let mut buf = Vec::new();
        write_stream_event(
            &mut buf,
            &StreamEvent::Frame(FrameHeader::for_payload(1, 0, 4, &bytes_a)),
            Some(&bytes_a),
        )
        .unwrap();
        write_stream_event(
            &mut buf,
            &StreamEvent::Frame(FrameHeader::for_payload(1, 4, 4, &bytes_b)),
            Some(&bytes_b),
        )
        .unwrap();
        write_stream_event(
            &mut buf,
            &StreamEvent::Progress(Progress {
                request_id: 1,
                samples_done: 8,
            }),
            None,
        )
        .unwrap();
        write_stream_event(
            &mut buf,
            &StreamEvent::Done(Done {
                request_id: 1,
                cancelled: false,
                stats: Stats {
                    samples_done: 8,
                    requested_cadence_ms: 0,
                    effective_cadence_ms: 0,
                },
            }),
            None,
        )
        .unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let mut events = Vec::new();
        loop {
            let (event, _payload) = read_stream_event(&mut cursor).unwrap();
            let done = matches!(event, StreamEvent::Done(_));
            events.push(event);
            if done {
                break;
            }
        }
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], StreamEvent::Frame(_)));
        assert!(matches!(events[1], StreamEvent::Frame(_)));
        assert!(matches!(events[2], StreamEvent::Progress(_)));
        assert!(matches!(events[3], StreamEvent::Done(_)));
    }
}

//! The wire messages, and encode/decode for each over a length-prefixed
//! [`crate::framing`] stream.
//!
//! ```text
//! -> HELLO    { protocol_version, build_hash }
//! <- WELCOME  { protocol_version, build_hash, render: Option<RenderCapability>, library }
//! -> <ClientMessage>  Cancel | Library(LibraryRequest) | RenderRequest   (post-handshake, tagged)
//! <- <StreamEvent>    Frame | Preview | Progress | Done | Error          (RENDER replies, tagged)
//! ```
//!
//! Split across a few submodules, grouped by what they're about rather than left as one
//! file (the original `messages.rs` ran past 1000 lines):
//!
//! - [`hello`]: the handshake messages, `HELLO`/`WELCOME` -- see that module's own doc
//!   comment for how [`hello::Welcome`] honestly advertises this worker's capabilities.
//! - [`render`]: `RENDER`'s own request shape (only compiled under this crate's
//!   `render` feature -- see that module's doc comment).
//! - [`stream`]: everything else post-handshake -- [`stream::ClientMessage`] (every
//!   message a client may send), [`stream::StreamEvent`] (every reply a `RENDER`
//!   produces), and [`stream::ErrorMsg`], used well beyond just streaming.
//! - [`codec`]: the generic framed codec ([`codec::NetError`],
//!   [`codec::write_message`]/[`codec::read_message`]) every one of the above builds on.
//!
//! Every item is re-exported here at its ORIGINAL flat `messages::` path (this module's
//! own public API is unchanged by the split -- see each submodule for why it was split
//! out this way).

mod codec;
mod hello;
#[cfg(feature = "render")]
mod render;
mod stream;

pub use codec::{NetError, read_message, write_message};
pub use hello::{Backend, Hello, RenderCapability, Welcome};
#[cfg(feature = "render")]
pub use render::{PreviewConfig, RenderRequest, StreamConfig, TransferMode};
pub use stream::{
    Cancel, ClientMessage, Done, ErrorMsg, FrameHeader, PreviewHeader, Progress, Stats,
    StreamEvent, read_frame_message, read_preview_message, read_stream_event, write_frame_message,
    write_preview_message, write_stream_event,
};

/// The wire protocol version this build of `gemray-net` speaks.
///
/// Bumped only for changes to the MESSAGE SHAPES in this module. A change to `gemray`'s
/// physics does not touch the wire format at all and is caught instead by the
/// build-hash check in [`crate::handshake`].
///
/// This is pre-release software: v1 is the first and only version there has ever been,
/// and there is no compatibility shim between versions.
/// [`crate::handshake::verify_compatible`] simply refuses to pair peers speaking
/// different ones.
///
/// # When you MUST bump this
///
/// The wire codec is [`postcard`] (via [`write_message`]/[`read_message`]), and
/// postcard is **not self-describing**. Two consequences drive every bump:
///
/// - **Enums are encoded by declaration-order index, not by name.** Appending a variant
///   leaves every pre-existing variant's encoding byte-for-byte unchanged, but a peer
///   that predates the new variant cannot decode it -- it sees an unrecognized index and
///   fails with an opaque per-message error. It cannot skip what it does not recognize.
/// - **Structs are encoded as their fields in declaration order, with no field names and
///   no length prefix.** Appending a field is worse than an unknown enum variant: a peer
///   on the other version silently MISALIGNS every field after it rather than failing
///   outright. That applies transitively -- appending a field to a type embedded in
///   [`RenderRequest`] (such as [`crate::scene::SceneState`], or the `GemMaterial` inside
///   it) changes the byte layout of every message carrying it.
///
/// So: append a variant to any wire enum, or a field to any wire struct (or to anything
/// a wire struct embeds), and bump this. The bump is what makes the failure diagnosable
/// -- the handshake refuses the pairing up front with a clear version-mismatch log line,
/// instead of letting a peer decode garbage or fail per message with no useful context.
///
/// **`#[serde(default)]` does not help here.** It matters for self-describing formats
/// (`gemray-worker`'s local `scene.json` files, which never cross this wire protocol),
/// and does nothing whatsoever for postcard's fixed-layout encoding. The version bump,
/// not the serde attribute, is what protects the network path.
pub const PROTOCOL_VERSION: u16 = 1;

#[cfg(test)]
mod tests {
    #[test]
    /// Pins the constant so a bump is always a deliberate, reviewed edit rather than
    /// something that drifts in alongside a message-shape change.
    fn protocol_version_is_1() {
        assert_eq!(super::PROTOCOL_VERSION, 1);
    }
}

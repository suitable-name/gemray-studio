//! Codec for the per-frame radiance buffer -- `Vec<Vec3>`, one running XYZ sum per
//! pixel -- the other half of the wire protocol besides [`crate::scene::SceneState`].
//!
//! # Why raw POD bytes, not `postcard`
//!
//! `SceneState` is a few kilobytes sent once per camera change: serialisation cost is
//! irrelevant there, so [`crate::scene::SceneState`] goes through `postcard` for
//! compactness. The radiance buffer is the opposite case -- potentially megapixels of
//! `Vec3`, sent every batch of samples, on the hot path. `Vec<Vec3>` is already a flat
//! array of plain-old-data floats; wrapping it in a serialization framework would only
//! add per-element framing overhead and copies for no benefit. `bytemuck::cast_slice`
//! gives the byte view for free -- the same technique already used for
//! `GpuFacetPlane` in `gemray::geometry::plane`.
//!
//! # Why this needed the `glam/bytemuck` feature
//!
//! `glam::Vec3` is not `bytemuck::Pod` unless glam's own `bytemuck` cargo feature is
//! enabled -- gemray-net's `Cargo.toml` turns it on for this reason. `Vec3` on this
//! target is exactly `3 * size_of::<f32>()` bytes with `f32` alignment (verified: no
//! padding, unlike the SIMD-aligned `Vec3A` glam also offers), so the cast is a
//! straightforward reinterpretation, not a repacking.

use glam::Vec3;

/// Byte size of one radiance sample on the wire (one pixel's running XYZ sum).
pub const BYTES_PER_PIXEL: usize = size_of::<Vec3>();

/// Encodes a radiance buffer as raw little-endian POD bytes, ready to send as a
/// `FRAME` message's `xyz_bytes` payload. The inverse of [`decode`].
#[must_use]
pub fn encode(buffer: &[Vec3]) -> Vec<u8> {
    bytemuck::cast_slice(buffer).to_vec()
}

/// Why a decoded radiance payload was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadianceError {
    /// `bytes.len()` didn't equal `width * height` times [`BYTES_PER_PIXEL`].
    ///
    /// Covers short, long, AND empty payloads uniformly -- an empty buffer against a
    /// nonzero `width * height` is just the `expected != 0, got == 0` case of the same
    /// check.
    LengthMismatch {
        width: u32,
        height: u32,
        expected_bytes: usize,
        got_bytes: usize,
    },
    /// `bytes` was the right length but not aligned for a `&[Vec3]` reinterpretation
    /// (e.g. a caller handed in a sub-slice starting at an odd offset). Rejected rather
    /// than worked around, since silently reinterpreting misaligned bytes as `Vec3`
    /// would be UB.
    Misaligned,
}

/// Decodes a `FRAME` message's raw `xyz_bytes` payload back into a radiance buffer.
///
/// Validates its length against the frame's declared `width * height` before
/// accumulating a single sample -- see [`RadianceError::LengthMismatch`]. Never
/// accumulates a buffer whose length doesn't match the expected pixel count.
///
/// # Errors
///
/// Returns [`RadianceError::LengthMismatch`] if `bytes.len()` isn't exactly
/// `width * height` times [`BYTES_PER_PIXEL`], or [`RadianceError::Misaligned`] if
/// `bytes` is the right length but not aligned for reinterpretation as `&[Vec3]`.
pub fn decode(bytes: &[u8], width: u32, height: u32) -> Result<Vec<Vec3>, RadianceError> {
    let expected_pixels = width as usize * height as usize;
    let expected_bytes = expected_pixels * BYTES_PER_PIXEL;
    if bytes.len() != expected_bytes {
        return Err(RadianceError::LengthMismatch {
            width,
            height,
            expected_bytes,
            got_bytes: bytes.len(),
        });
    }
    if expected_pixels == 0 {
        // `try_cast_slice` on a zero-length slice can report a (harmless, since
        // nothing is ever dereferenced) alignment mismatch on the slice's dangling
        // pointer -- short-circuit rather than let that obscure the real, meaningful
        // failure mode this function exists to catch (a length mismatch).
        return Ok(Vec::new());
    }
    bytemuck::try_cast_slice::<u8, Vec3>(bytes)
        .map_or(Err(RadianceError::Misaligned), |slice| Ok(slice.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_pixel_matches_three_f32s() {
        assert_eq!(BYTES_PER_PIXEL, 12);
    }

    #[test]
    fn round_trips_bit_exactly() {
        let buffer = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-0.5, f32::MAX, 0.0),
            Vec3::ZERO,
        ];
        let bytes = encode(&buffer);
        let decoded = decode(&bytes, 3, 1).unwrap();
        assert_eq!(buffer, decoded);
    }

    #[test]
    fn rejects_short_payload() {
        let buffer = vec![Vec3::ONE; 4];
        let bytes = encode(&buffer);
        let short = &bytes[..bytes.len() - 1];
        assert!(matches!(
            decode(short, 2, 2),
            Err(RadianceError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_long_payload() {
        let buffer = vec![Vec3::ONE; 4];
        let mut bytes = encode(&buffer);
        bytes.push(0);
        assert!(matches!(
            decode(&bytes, 2, 2),
            Err(RadianceError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_empty_payload_against_nonzero_dimensions() {
        assert!(matches!(
            decode(&[], 2, 2),
            Err(RadianceError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn accepts_empty_payload_for_zero_area() {
        assert_eq!(decode(&[], 0, 0).unwrap(), Vec::<Vec3>::new());
    }
}

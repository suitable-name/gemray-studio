//! Readers and writers for gemstone faceting design file formats.
//!
//! `lapidary` -- the craft of cutting and polishing gems -- is an independent,
//! unaffiliated implementation of file formats originating with `GemCAD` (Robert
//! Strickland's faceting-design software) and, in future, Gem Cut Studio. It is not
//! produced or endorsed by either. Each format lives in its own module so that
//! reading (or writing) a design never requires pulling in a particular renderer,
//! database, or GUI toolkit:
//!
//! - [`asc`]: `GemCAD`'s `.asc` cutting-schedule text format. Read and write support,
//!   verified against a real-world corpus of 5,759 files.
//! - `gem` / `gcs` (not yet implemented): `GemCAD`'s native `.gem` format and Gem Cut
//!   Studio's `.gcs` format are natural future additions here, since some real-world
//!   designs exist only as one of those and have no `.asc` counterpart at all.
//!
//! This crate has **zero runtime dependencies** by design -- a file-format reader
//! should not force a dependency tree onto callers who just want to parse text.
//! Anything genuinely shared across more than one format's module (a common schedule
//! representation, index-list handling, etc.) belongs at this crate root rather than
//! inside a single format module; nothing has met that bar yet, since `asc` is
//! currently the only implemented format.

pub mod asc;

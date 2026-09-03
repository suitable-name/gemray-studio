//! `diagram-catalog`: plain data models, a SQLite-backed store, and local import/export
//! for the user's own faceting-design library.
//!
//! This crate holds, searches, and file-imports/exports a local library of designs.
//! It is the storage layer and nothing more: no network client, no HTML/SVG parsing,
//! no OCR. That is a deliberate boundary, not an omission -- the dependency direction
//! is strictly one way, with anything that acquires designs depending on this crate's
//! [`model`] and [`db`], never the reverse. Keeping the boundary here means there is
//! exactly one implementation of the storage, model, and local-import code, whatever
//! ends up feeding it.
//!
//! - [`db`]: SQLite storage and schema migrations. The schema is deliberately wider
//!   than this crate itself fills -- columns like `page_url`/`pdf_file` are just
//!   columns, and an existing `facet_diagrams.sqlite`, however it was populated,
//!   keeps opening and keeps every value it already holds.
//! - [`model`]: `FacetDiagramDetail`, `AngleSetting`, `AttachedFile`, search/range
//!   filters, and cross-source dedup types.
//! - [`local`]: import/export for the user's own `.asc` files (via `lapidary::asc`),
//!   independent of any online source.

pub mod db;
pub mod local;
pub mod model;

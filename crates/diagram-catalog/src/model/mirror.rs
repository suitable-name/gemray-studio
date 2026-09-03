//! Bookkeeping for a local mirror of a remote design-library server -- see
//! `apps/diagram-gui`'s pull-mirror sync.
//!
//! # Why this needs its own table, and why keyed by `url`
//!
//! `gemray_net::library`'s wire protocol carries a content-hash "version" on both
//! [`crate::model::entry::DiagramListItem`]'s remote counterpart (`DesignSummary`) and
//! the full design record (`DesignRecord`), specifically so a mirroring client can skip
//! re-fetching a design whose hash hasn't moved since the last sync (see that crate's
//! `library` module doc comment, "Staleness" section). But `diagram_entries`/
//! `diagram_details` have no column to remember what hash was last seen -- and, per
//! `gemray_net::library`'s own doc comment, deliberately can't grow one (a
//! `gemray-worker` serving that data is read-only against it). This table is the
//! client-side equivalent: a small, purely local, additive record of what this
//! database's own last sync of a given design saw.
//!
//! Keyed by `url`, not `entry_id` -- because `url` is already this database's identity
//! for "is a synced design the same as a row already here"
//! ([`crate::db::sqlite::Database::save_diagram_entry`]'s `UNIQUE` constraint on
//! `diagram_entries.url` is exactly that judgement call, already made and already
//! proven: a local `.asc` import gets a synthetic `local://...` URL specifically so it
//! can never collide with a real remote page URL, see
//! `crate::local::import_asc`'s doc comment). Reusing that same key here means this
//! table never has to make its own "same design" decision -- it just remembers, for a
//! URL `save_diagram_entry` already treats as one design's identity, what the last
//! remote sync of it looked like.
use serde::{Deserialize, Serialize};

/// One design's last-known remote content hashes, as of this database's most recent
/// successful sync of it.
///
/// [`Self::summary_version`] is the cheap, search-result-level hash
/// (`gemray_net::library::DesignSummary::version`) and [`Self::design_version`] is the
/// authoritative, full-record hash (`gemray_net::library::DesignRecord::version`) -- see
/// that crate's `library` module doc comment for why a mirror sync checks the former
/// first (skip a design whose summary hasn't moved) and the latter before deciding a
/// full re-fetch actually changed anything worth re-saving.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirrorState {
    pub url: String,
    /// Which remote server this state was last synced from (see
    /// `crate::db::sqlite::LEGACY_SOURCE_ID`/`crate::local::LOCAL_SOURCE_ID` for the
    /// sibling conventions this reuses -- a mirror sync's own `source_id` is the
    /// worker's configured address, see `apps/diagram-gui/src/bridge/library_mirror.rs`).
    pub source_id: String,
    pub summary_version: [u8; 32],
    pub design_version: [u8; 32],
}

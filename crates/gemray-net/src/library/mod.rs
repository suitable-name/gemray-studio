//! The read-only design-library sync protocol: a client (a future Phase 2 viewer, or a
//! mobile client with no renderer compiled in at all) mirrors designs from a
//! `gemray-worker`'s catalogue.
//!
//! # Why this exists, and why it's separate from the render protocol
//!
//! Long term, `gemray-worker` becomes the server half of full gem-CAD software: serving
//! the design library, not rendering, is the primary role, and a mobile client would
//! talk to exactly this protocol with no `render` feature compiled in at all (see
//! `gemray-worker`'s crate docs). [`LibraryRequest`]/[`LibraryResponse`] carry that
//! traffic as their own message family, tagged into the SAME [`crate::messages::ClientMessage`]
//! envelope the render protocol uses (`ClientMessage::Library`) -- one post-handshake
//! read loop, one place a worker decides which family a message belongs to -- rather
//! than a second, ambiguous top-level message type a reader would have to guess between
//! (see `crate::messages::ClientMessage`'s own doc comment on why that guessing game is
//! exactly what a tag exists to avoid).
//!
//! This module never depends on `gemray` (unlike `crate::scene`/`crate::render`) and is
//! ALWAYS compiled in, regardless of this crate's `render` feature -- a client that can
//! never render still needs to sync the library.
//!
//! # Pull-mirror, read-only -- for now
//!
//! The server is authoritative; a client mirrors FROM it. This phase implements only
//! the read side ([`LibraryRequest::Search`]/[`SearchPage`]/[`FilterOptions`]/
//! [`FetchDesign`]/[`FetchAttachment`]) -- there is no `Put`/`Delete`/merge request, and
//! `gemray-worker` never writes to its catalogue database in this phase.
//!
//! [`LibraryRequest::Search`] is a single request/response round trip capped at
//! `Database::search_diagrams`'s own result cap -- fine for an interactive search box,
//! but a catalogue with more matching rows than that cap cannot be fully listed by it
//! alone. [`LibraryRequest::SearchPage`]/[`LibraryResponse::SearchResultsPage`] is the
//! keyset-cursor form a caller that needs EVERY matching row (a mirror sync,
//! `apps/diagram-gui`'s `bridge::library_mirror`) uses instead: the exact same filters,
//! plus a cursor, walked page by page until a short page signals the end.
//!
//! **The door stays open for push later, without breaking what's already here:**
//! [`LibraryRequest`]/[`LibraryResponse`] are plain, generically-named enums (not
//! `LibraryReadRequest`, which would frame a later write variant as a foreign
//! afterthought), and a write operation is exactly the shape of thing that appends a
//! new variant at the tail of each -- the same additive pattern
//! [`crate::messages::ClientMessage`] itself uses for `RenderRequest` (see that type's
//! doc comment on why variant order, not variant existence, is what has to stay
//! disciplined). [`DesignSummary::version`]/[`DesignRecord::version`] -- present now
//! purely for read-side staleness detection (see their own doc comments) -- are exactly
//! what a future `Put { entry_id, expected_version, .. }` needs for optimistic-
//! concurrency conflict detection, so that piece doesn't need inventing later either.
//! This crate is pre-release (see [`crate::messages::PROTOCOL_VERSION`]'s doc comment:
//! no compatibility shim between wire versions), so a push extension is free to bump
//! that version and refuse to pair with an older peer the same way any other protocol
//! change here already does -- "non-breaking" is about the SHAPE of the extension
//! (additive, not a redesign), not a promise of wire compatibility across versions,
//! which this crate never makes anywhere else either.
//!
//! # Staleness: a content hash, not a sequence number
//!
//! [`DesignSummary::version`]/[`DesignRecord::version`] is a SHA-256 hash over the
//! fields that response actually carries (excluding the hash field itself, and -- for
//! [`DesignRecord`] -- excluding attachment CONTENT, only their metadata). `gemray-worker`
//! computes it at serve time; this crate only carries the resulting 32 bytes.
//!
//! A content hash, not a per-row sequence number or `updated_at` timestamp, because the
//! underlying `diagram_entries`/`diagram_details` schema (see
//! `diagram_catalog::db::sqlite`) has neither column, and this phase must not write to
//! that database at all (see `gemray-worker`'s crate docs -- read-only, deliberately) --
//! so adding one is off the table, not just avoided for convenience. A content hash
//! needs no schema change, is correct by construction (any change to a hashed field
//! changes the hash; nothing to keep in sync with a separate bump-on-write step someone
//! could forget), and is cheap to compute over the small text fields these responses
//! already carry (a few dozen bytes per design -- SHA-256 over that is negligible next
//! to the SQLite query producing it). Its one real limitation -- a value that changes
//! and later changes BACK looks unchanged -- doesn't matter for what this exists to do:
//! decide whether a client's mirror needs a re-fetch, where a false negative (skipping a
//! fetch for data that's actually still identical) is harmless and a false positive
//! (fetching data that turns out unchanged) only costs one wasted round trip, never
//! incorrectness.
//!
//! [`DesignSummary::version`] covers only what [`DesignSummary`] itself carries (cheap:
//! computed for up to 1000 rows per search); [`DesignRecord::version`] additionally
//! covers the full detail record and the diagram image bytes, so it's the authoritative
//! one for deciding whether a full re-fetch is needed -- a client doing a real mirror
//! sync should treat [`DesignSummary::version`] as a cheap first filter (skip designs
//! whose summary hash hasn't moved) and confirm against [`DesignRecord::version`] before
//! actually skipping a `FetchDesign`.
//!
//! # Attachments: fetched separately, one at a time
//!
//! [`DesignRecord`] carries attachment METADATA only ([`AttachedFileMeta`] -- id, name,
//! url, size), never content: `diagram_catalog`'s `attached_files` table can hold
//! multi-megabyte PDFs, and several designs frequently reference the SAME attachment
//! (see `diagram_catalog`'s own docs on competition-results booklets shared across a
//! whole class of entries) -- inlining content into every `FetchDesign` reply would
//! mean re-sending that PDF's bytes once per design that references it. A client fetches
//! an attachment's bytes only when it actually needs them (lazily, and only once per
//! attachment id, not once per design), via [`LibraryRequest::FetchAttachment`].
//!
//! Per-request memory is bounded to ONE attachment's bytes, not a whole design's
//! attachment set or the whole search result set -- `gemray-worker`'s implementation
//! (see that crate's serving code) loads exactly the requested id's `content` column,
//! nothing else. It is NOT chunked/streamed further within that -- one `FetchAttachment`
//! reply carries one attachment's full bytes in memory before writing them. Acceptable
//! for this phase given the corpus's real attachment sizes (competition PDFs and gem
//! diagrams, not video); genuine incremental streaming (SQLite's own incremental-BLOB-
//! read API, chunked over several wire frames) is a natural, additive follow-up if a
//! future attachment ever makes single-shot loading a real problem, not something this
//! phase's message shapes need to anticipate.

use serde::{Deserialize, Serialize};

use crate::messages::ErrorMsg;

/// Wire counterpart of `diagram_catalog::model::filter::RangeFilter`.
///
/// See that type's own doc comment for what each bound means (a `None` bound is
/// unconstrained). Kept as a separate type (rather than depending on `diagram-catalog`
/// from this crate) so this crate's wire shapes stay independent of that crate's own
/// internal representation -- `gemray-worker` is the one place that maps between them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RangeFilterWire {
    pub ri_min: Option<f64>,
    pub ri_max: Option<f64>,
    pub lw_min: Option<f64>,
    pub lw_max: Option<f64>,
    pub volume_min: Option<f64>,
    pub volume_max: Option<f64>,
    pub facets_min: Option<i64>,
    pub facets_max: Option<i64>,
}

/// Wire counterpart of `diagram_catalog::model::filter::AttributeRanges`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttributeRangesWire {
    pub ri: (f64, f64),
    pub lw_ratio: (f64, f64),
    pub volume: (f64, f64),
    pub facets: (i64, i64),
}

/// One design as it appears in a [`LibraryResponse::SearchResults`] list.
///
/// Wire counterpart of `diagram_catalog::model::entry::DiagramListItem`, plus
/// [`Self::version`] (see the module docs on staleness).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignSummary {
    pub entry_id: i64,
    pub title: String,
    pub url: String,
    pub design_id: Option<String>,
    pub shape: Option<String>,
    pub index_gear: Option<String>,
    pub facets_count: Option<String>,
    pub designer_info: Option<String>,
    pub lw_ratio: Option<String>,
    pub refractive_index: Option<String>,
    pub volume: Option<String>,
    pub competition_diagram: Option<String>,
    /// SHA-256 over every other field above -- see the module docs' "Staleness"
    /// section.
    pub version: [u8; 32],
}

/// One angle-schedule row -- wire counterpart of `diagram_catalog::model::angle::AngleSetting`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AngleSettingWire {
    pub order_index: u32,
    pub facet: String,
    pub angle: String,
    pub index: String,
    pub notes: String,
}

/// One attachment's METADATA -- never its content; see the module docs' "Attachments"
/// section for why content is fetched separately, by [`Self::id`], via
/// [`LibraryRequest::FetchAttachment`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachedFileMeta {
    pub id: i64,
    pub name: String,
    pub url: String,
    /// Byte length of the attachment's content, without ever loading it -- lets a
    /// client show "PDF, 2.4 MB" or decide whether to fetch it at all before spending a
    /// round trip.
    pub size: u64,
}

/// One design, in full -- entry + detail + angle settings + attachment METADATA.
///
/// Never attachment content -- see [`LibraryRequest::FetchAttachment`]. Wire
/// counterpart of `diagram_catalog::model::entry::FullDiagramRecord`, minus attachment
/// content, plus [`Self::version`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignRecord {
    pub entry_id: i64,
    pub title: String,
    pub url: String,
    pub design_id: Option<String>,
    pub page_url: String,
    pub diagram_image_name: Option<String>,
    /// The design's own diagram image (an SVG/PNG central to displaying the design
    /// itself, not a supplementary file) -- kept inline, unlike attachment content; see
    /// the module docs' "Attachments" section for the distinction.
    pub diagram_image_data: Option<Vec<u8>>,
    pub competition_diagram: Option<String>,
    pub lw_ratio: Option<String>,
    pub refractive_index: Option<String>,
    pub index_gear: Option<String>,
    pub volume: Option<String>,
    pub facets_count: Option<String>,
    pub shape: Option<String>,
    pub designer_info: Option<String>,
    pub angle_settings: Vec<AngleSettingWire>,
    pub attachments: Vec<AttachedFileMeta>,
    /// SHA-256 over every other field above (including `diagram_image_data` and each
    /// attachment's metadata, but never attachment content) -- see the module docs'
    /// "Staleness" section. The authoritative version for deciding whether a client's
    /// mirror of this one design needs a re-fetch.
    pub version: [u8; 32],
}

/// One request in the read-only library-sync protocol.
///
/// See the module docs for why this is deliberately request/response (not streamed,
/// unlike `RENDER`) and how the shape leaves room for a push extension later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LibraryRequest {
    /// List/search designs -- mirrors
    /// `diagram_catalog::db::sqlite::Database::search_diagrams`'s own filters exactly,
    /// so a client's search UI maps onto this one-for-one. Capped at that same method's
    /// result cap (currently 1000) -- a catalogue with more matching rows than that
    /// cannot be fully listed via this variant alone; see [`Self::SearchPage`] for the
    /// paginated form a full-catalogue walk (e.g. a mirror sync) needs instead.
    Search {
        query: String,
        shape_filter: String,
        gear_filter: String,
        range: RangeFilterWire,
    },
    /// The scalar catalogue facts a search UI needs to build its filter controls --
    /// distinct shapes, distinct gears, and attribute range bounds -- mirroring
    /// `Database::get_unique_shapes`/`get_unique_gears`/`get_attribute_ranges`.
    FilterOptions,
    /// Fetch one design's entry + detail + angle settings + attachment metadata.
    FetchDesign { entry_id: i64 },
    /// Fetch one attachment's raw bytes by id -- see the module docs' "Attachments"
    /// section.
    FetchAttachment { attachment_id: i64 },
    /// Keyset-paginated counterpart of [`Self::Search`], for a caller that needs the
    /// WHOLE matching result set, not just its first page (mirrors
    /// `Database::search_diagrams_page` exactly, one-for-one -- same filters, plus
    /// `cursor`).
    ///
    /// `cursor` is `None` for the first page, or `Some(entry_id)` -- the `entry_id` of
    /// the last [`DesignSummary`] the previous page returned -- to continue strictly
    /// after it. See [`LibraryResponse::SearchResultsPage::next_cursor`] for how a
    /// caller knows when to stop.
    ///
    /// Added as a NEW variant (appended at the tail, `Search` itself untouched) rather
    /// than a `cursor` field bolted onto `Search` -- see the module docs' "door stays
    /// open for push later" section, and `crate::messages::PROTOCOL_VERSION`'s doc
    /// comment on why an appended variant is still a protocol-breaking change (postcard
    /// encodes a variant by its declaration index, so a peer predating it cannot decode
    /// one), for why that is the shape an additive extension to this enum takes.
    SearchPage {
        query: String,
        shape_filter: String,
        gear_filter: String,
        range: RangeFilterWire,
        cursor: Option<i64>,
    },
}

/// A worker's reply to one [`LibraryRequest`].
///
/// `Design` is boxed purely to keep this enum's own stack footprint close to its other,
/// much smaller variants rather than every [`LibraryResponse`] paying for
/// [`DesignRecord`]'s size -- serde boxes and unboxes it transparently, so this has no
/// effect on the wire encoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LibraryResponse {
    SearchResults(Vec<DesignSummary>),
    FilterOptions {
        shapes: Vec<String>,
        gears: Vec<String>,
        ranges: AttributeRangesWire,
    },
    Design(Box<DesignRecord>),
    Attachment {
        name: String,
        content: Vec<u8>,
    },
    /// [`LibraryRequest::FetchDesign`]/[`FetchAttachment`] named an id this worker's
    /// catalogue has no row for.
    NotFound,
    /// A request-level failure this worker could still form a normal reply for (e.g. a
    /// malformed filter) -- distinct from a transport-level [`crate::messages::NetError`],
    /// which never reaches this type at all.
    Error(ErrorMsg),
    /// Reply to [`LibraryRequest::SearchPage`].
    ///
    /// `next_cursor` is `Some(last_entry_id)` -- the `entry_id` of `results`' last
    /// element -- when `results` came back a full page (there may be more; re-request
    /// with `cursor: next_cursor`), or `None` when this was the final page (`results`
    /// may be empty). A caller loops until it sees `None`.
    SearchResultsPage {
        results: Vec<DesignSummary>,
        next_cursor: Option<i64>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{read_message, write_message};

    fn sample_summary() -> DesignSummary {
        DesignSummary {
            entry_id: 42,
            title: "Round Brilliant".to_string(),
            url: "https://example.test/diagram/42".to_string(),
            design_id: Some("RB-1".to_string()),
            shape: Some("Round".to_string()),
            index_gear: Some("96".to_string()),
            facets_count: Some("57".to_string()),
            designer_info: Some("Capps, Jerry".to_string()),
            lw_ratio: Some("1.00".to_string()),
            refractive_index: Some("2.417".to_string()),
            volume: Some("0.65".to_string()),
            competition_diagram: None,
            version: [7u8; 32],
        }
    }

    #[test]
    fn library_request_variants_round_trip() {
        for req in [
            LibraryRequest::Search {
                query: "round".to_string(),
                shape_filter: "All".to_string(),
                gear_filter: "All".to_string(),
                range: RangeFilterWire::default(),
            },
            LibraryRequest::FilterOptions,
            LibraryRequest::FetchDesign { entry_id: 42 },
            LibraryRequest::FetchAttachment { attachment_id: 9 },
            LibraryRequest::SearchPage {
                query: "round".to_string(),
                shape_filter: "All".to_string(),
                gear_filter: "All".to_string(),
                range: RangeFilterWire::default(),
                cursor: None,
            },
            LibraryRequest::SearchPage {
                query: String::new(),
                shape_filter: "Round".to_string(),
                gear_filter: "96".to_string(),
                range: RangeFilterWire::default(),
                cursor: Some(1000),
            },
        ] {
            let mut buf = Vec::new();
            write_message(&mut buf, &req).unwrap();
            let mut cursor = std::io::Cursor::new(buf);
            let decoded: LibraryRequest = read_message(&mut cursor).unwrap();
            assert_eq!(decoded, req);
        }
    }

    #[test]
    fn library_response_variants_round_trip() {
        let design = DesignRecord {
            entry_id: 42,
            title: "Round Brilliant".to_string(),
            url: "https://example.test/diagram/42".to_string(),
            design_id: Some("RB-1".to_string()),
            page_url: "https://example.test/diagram/42".to_string(),
            diagram_image_name: Some("rb.svg".to_string()),
            diagram_image_data: Some(vec![1, 2, 3]),
            competition_diagram: None,
            lw_ratio: Some("1.00".to_string()),
            refractive_index: Some("2.417".to_string()),
            index_gear: Some("96".to_string()),
            volume: Some("0.65".to_string()),
            facets_count: Some("57".to_string()),
            shape: Some("Round".to_string()),
            designer_info: Some("Capps, Jerry".to_string()),
            angle_settings: vec![AngleSettingWire {
                order_index: 0,
                facet: "P1".to_string(),
                angle: "41.0".to_string(),
                index: "96".to_string(),
                notes: String::new(),
            }],
            attachments: vec![AttachedFileMeta {
                id: 1,
                name: "schedule.pdf".to_string(),
                url: "https://example.test/schedule.pdf".to_string(),
                size: 12_345,
            }],
            version: [1u8; 32],
        };

        for resp in [
            LibraryResponse::SearchResults(vec![sample_summary()]),
            LibraryResponse::FilterOptions {
                shapes: vec!["Round".to_string()],
                gears: vec!["96".to_string()],
                ranges: AttributeRangesWire {
                    ri: (1.4, 2.5),
                    lw_ratio: (0.8, 1.5),
                    volume: (0.1, 1.0),
                    facets: (20, 200),
                },
            },
            LibraryResponse::Design(Box::new(design)),
            LibraryResponse::Attachment {
                name: "schedule.pdf".to_string(),
                content: vec![0xDE, 0xAD, 0xBE, 0xEF],
            },
            LibraryResponse::NotFound,
            LibraryResponse::Error(ErrorMsg {
                code: 1,
                message: "bad filter".to_string(),
            }),
            LibraryResponse::SearchResultsPage {
                results: vec![sample_summary()],
                next_cursor: Some(42),
            },
            LibraryResponse::SearchResultsPage {
                results: Vec::new(),
                next_cursor: None,
            },
        ] {
            let mut buf = Vec::new();
            write_message(&mut buf, &resp).unwrap();
            let mut cursor = std::io::Cursor::new(buf);
            let decoded: LibraryResponse = read_message(&mut cursor).unwrap();
            assert_eq!(decoded, resp);
        }
    }

    #[test]
    fn client_message_library_round_trips() {
        use crate::messages::ClientMessage;
        let msg = ClientMessage::Library(Box::new(LibraryRequest::FilterOptions));
        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let decoded: ClientMessage = read_message(&mut cursor).unwrap();
        assert_eq!(decoded, msg);
    }
}

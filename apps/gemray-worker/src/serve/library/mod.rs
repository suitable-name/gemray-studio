//! Serves `gemray_net::library`'s read-only design-library protocol.
//!
//! Backed by a `diagram_catalog::db::sqlite::Database` -- the shared handler both a
//! library-only build and a `worker` build call for `ClientMessage::Library`, from the
//! exact same already-authenticated connection (see `crate::serve`'s module docs).
//!
//! [`handle_request`] is the one entry point: given one `LibraryRequest` and the
//! `Database` this process opened at startup (see `crate::serve::open_library_database`),
//! it returns exactly one `LibraryResponse` -- request/response, never streamed, unlike
//! `RenderRequest` (see `gemray_net::library`'s module docs for why). It never panics on
//! a database error or an unknown id: a query failure becomes `LibraryResponse::Error`
//! (logged in full server-side via `tracing::warn`, but reported to the peer only as a
//! generic message -- the same "don't hand a remote peer a detailed failure reason"
//! posture `gemray_net::enroll::EnrollResponse::ClaimFailed` already uses, applied here
//! to database errors instead of authentication ones), and a `FetchDesign`/
//! `FetchAttachment` for an id with no matching row becomes `LibraryResponse::NotFound`.
//!
//! # Versioning: a content hash computed here, not stored in the database
//!
//! [`gemray_net::library::DesignSummary::version`]/[`gemray_net::library::DesignRecord::version`]
//! are SHA-256 hashes this module computes at response time (see [`hash_summary`]/
//! [`hash_record`]) over exactly the fields that response carries, in a fixed field
//! order, each length-prefixed so adjacent fields can never collide (`"ab"` + `"c"`
//! hashes differently from `"a"` + `"bc"`) and each `Option` tagged present/absent
//! before its value. `diagram_catalog`'s schema has no `updated_at`/revision column to
//! read instead -- see `gemray_net::library`'s module docs for why a content hash was
//! chosen over adding one (this phase must not write to, let alone migrate, the
//! database at all).

use diagram_catalog::{
    db::sqlite::{Database, SEARCH_RESULT_CAP},
    model::{
        entry::{DiagramListItem, FullDiagramMeta},
        filter::{AttributeRanges, RangeFilter},
    },
};
use gemray_net::{
    library::{
        AngleSettingWire, AttachedFileMeta, AttributeRangesWire, DesignRecord, DesignSummary,
        LibraryRequest, LibraryResponse, RangeFilterWire,
    },
    messages::ErrorMsg,
};
use sha2::{Digest, Sha256};

/// `<- ERROR` code for a `LibraryRequest` that failed on this worker's side.
///
/// Distinct from `crate::serve::connection`'s render-specific codes
/// (`BUILD_MISMATCH_CODE` = 1, `VALIDATION_FAILED_CODE` = 2, `TRACE_PANIC_CODE` = 3,
/// only compiled under `worker`), so a client can tell a library failure apart from a
/// render one even on a build where both exist.
pub const LIBRARY_ERROR_CODE: u32 = 4;

/// Handles one [`LibraryRequest`] against `db`, producing exactly one [`LibraryResponse`].
/// See the module doc comment.
#[must_use]
pub fn handle_request(request: &LibraryRequest, db: &Database) -> LibraryResponse {
    match request {
        LibraryRequest::Search {
            query,
            shape_filter,
            gear_filter,
            range,
        } => match db.search_diagrams(query, shape_filter, gear_filter, &from_range_wire(range)) {
            Ok(items) => LibraryResponse::SearchResults(items.iter().map(to_summary).collect()),
            Err(e) => db_error("search_diagrams", &e),
        },
        LibraryRequest::FilterOptions => filter_options(db),
        LibraryRequest::FetchDesign { entry_id } => match db.get_diagram_full_meta(*entry_id) {
            Ok(Some(meta)) => LibraryResponse::Design(Box::new(to_record(&meta))),
            Ok(None) => LibraryResponse::NotFound,
            Err(e) => db_error("get_diagram_full_meta", &e),
        },
        LibraryRequest::FetchAttachment { attachment_id } => {
            match db.get_attachment_content(*attachment_id) {
                Ok(Some((name, content))) => LibraryResponse::Attachment { name, content },
                Ok(None) => LibraryResponse::NotFound,
                Err(e) => db_error("get_attachment_content", &e),
            }
        }
        LibraryRequest::SearchPage {
            query,
            shape_filter,
            gear_filter,
            range,
            cursor,
        } => search_page(db, query, shape_filter, gear_filter, range, *cursor),
    }
}

/// Handles [`LibraryRequest::SearchPage`]: one keyset-paginated page of
/// `Database::search_diagrams_page`, converted to [`LibraryResponse::SearchResultsPage`].
///
/// Pages at [`SEARCH_RESULT_CAP`] rows -- the same size `Database::search_diagrams`
/// (and, through it, [`LibraryRequest::Search`]) has always capped a single response
/// at, so a `SearchPage` walk costs the same per-page work an unpaginated `Search`
/// already did; it just doesn't stop after page one. `next_cursor` is `Some` (the last
/// row's `entry_id`) exactly when the page came back full -- see
/// `Database::search_diagrams_page`'s own doc comment for why that's the signal a
/// caller re-requests on, and why it's cheap-but-not-exact (an occasional harmless
/// extra empty final page) rather than a second COUNT query to know for certain.
fn search_page(
    db: &Database,
    query: &str,
    shape_filter: &str,
    gear_filter: &str,
    range: &RangeFilterWire,
    cursor: Option<i64>,
) -> LibraryResponse {
    match db.search_diagrams_page(
        query,
        shape_filter,
        gear_filter,
        &from_range_wire(range),
        cursor,
        SEARCH_RESULT_CAP,
    ) {
        Ok(items) => {
            let page_full = items.len() == usize::try_from(SEARCH_RESULT_CAP).unwrap_or(usize::MAX);
            let next_cursor = if page_full {
                items.last().map(|i| i.id)
            } else {
                None
            };
            LibraryResponse::SearchResultsPage {
                results: items.iter().map(to_summary).collect(),
                next_cursor,
            }
        }
        Err(e) => db_error("search_diagrams_page", &e),
    }
}

fn filter_options(db: &Database) -> LibraryResponse {
    let shapes = match db.get_unique_shapes() {
        Ok(v) => v,
        Err(e) => return db_error("get_unique_shapes", &e),
    };
    let gears = match db.get_unique_gears() {
        Ok(v) => v,
        Err(e) => return db_error("get_unique_gears", &e),
    };
    let ranges = match db.get_attribute_ranges() {
        Ok(v) => v,
        Err(e) => return db_error("get_attribute_ranges", &e),
    };
    LibraryResponse::FilterOptions {
        shapes,
        gears,
        ranges: to_ranges_wire(ranges),
    }
}

/// Logs the real reason server-side and returns a generic [`LibraryResponse::Error`] --
/// see the module doc comment on why the peer never sees the underlying database error
/// text.
fn db_error(op: &str, e: &anyhow::Error) -> LibraryResponse {
    tracing::warn!("library request failed ({op}): {e:#}");
    LibraryResponse::Error(ErrorMsg {
        code: LIBRARY_ERROR_CODE,
        message: "internal error serving the design library".to_string(),
    })
}

const fn from_range_wire(r: &RangeFilterWire) -> RangeFilter {
    RangeFilter {
        ri_min: r.ri_min,
        ri_max: r.ri_max,
        lw_min: r.lw_min,
        lw_max: r.lw_max,
        volume_min: r.volume_min,
        volume_max: r.volume_max,
        facets_min: r.facets_min,
        facets_max: r.facets_max,
    }
}

const fn to_ranges_wire(r: AttributeRanges) -> AttributeRangesWire {
    AttributeRangesWire {
        ri: r.ri,
        lw_ratio: r.lw_ratio,
        volume: r.volume,
        facets: r.facets,
    }
}

fn to_summary(item: &DiagramListItem) -> DesignSummary {
    let mut summary = DesignSummary {
        entry_id: item.id,
        title: item.title.clone(),
        url: item.url.clone(),
        design_id: item.design_id.clone(),
        shape: item.shape.clone(),
        index_gear: item.index_gear.clone(),
        facets_count: item.facets_count.clone(),
        designer_info: item.designer_info.clone(),
        lw_ratio: item.lw_ratio.clone(),
        refractive_index: item.refractive_index.clone(),
        volume: item.volume.clone(),
        competition_diagram: item.competition_diagram.clone(),
        version: [0u8; 32],
    };
    summary.version = hash_summary(&summary);
    summary
}

fn to_record(meta: &FullDiagramMeta) -> DesignRecord {
    let angle_settings = meta
        .angle_settings
        .iter()
        .map(|a| AngleSettingWire {
            order_index: a.order_index,
            facet: a.facet.clone(),
            angle: a.angle.clone(),
            index: a.index.clone(),
            notes: a.notes.clone(),
        })
        .collect();
    let attachments = meta
        .attached_files
        .iter()
        .map(|f| AttachedFileMeta {
            id: f.id,
            name: f.name.clone(),
            url: f.url.clone(),
            size: u64::try_from(f.size).unwrap_or(0),
        })
        .collect();

    let mut record = DesignRecord {
        entry_id: meta.entry_id,
        title: meta.title.clone(),
        url: meta.url.clone(),
        design_id: meta.design_id.clone(),
        page_url: meta.page_url.clone(),
        diagram_image_name: meta.diagram_image_name.clone(),
        diagram_image_data: meta.diagram_image_data.clone(),
        competition_diagram: meta.competition_diagram.clone(),
        lw_ratio: meta.lw_ratio.clone(),
        refractive_index: meta.refractive_index.clone(),
        index_gear: meta.index_gear.clone(),
        volume: meta.volume.clone(),
        facets_count: meta.facets_count.clone(),
        shape: meta.shape.clone(),
        designer_info: meta.designer_info.clone(),
        angle_settings,
        attachments,
        version: [0u8; 32],
    };
    record.version = hash_record(&record);
    record
}

fn hash_str(hasher: &mut Sha256, s: &str) {
    hasher.update((s.len() as u64).to_le_bytes());
    hasher.update(s.as_bytes());
}

fn hash_opt_str(hasher: &mut Sha256, s: Option<&str>) {
    match s {
        Some(s) => {
            hasher.update([1u8]);
            hash_str(hasher, s);
        }
        None => hasher.update([0u8]),
    }
}

fn hash_opt_bytes(hasher: &mut Sha256, b: Option<&[u8]>) {
    match b {
        Some(b) => {
            hasher.update([1u8]);
            hasher.update((b.len() as u64).to_le_bytes());
            hasher.update(b);
        }
        None => hasher.update([0u8]),
    }
}

/// SHA-256 over every [`DesignSummary`] field EXCEPT [`DesignSummary::version`] itself
/// -- see the module doc comment.
fn hash_summary(s: &DesignSummary) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(s.entry_id.to_le_bytes());
    hash_str(&mut hasher, &s.title);
    hash_str(&mut hasher, &s.url);
    hash_opt_str(&mut hasher, s.design_id.as_deref());
    hash_opt_str(&mut hasher, s.shape.as_deref());
    hash_opt_str(&mut hasher, s.index_gear.as_deref());
    hash_opt_str(&mut hasher, s.facets_count.as_deref());
    hash_opt_str(&mut hasher, s.designer_info.as_deref());
    hash_opt_str(&mut hasher, s.lw_ratio.as_deref());
    hash_opt_str(&mut hasher, s.refractive_index.as_deref());
    hash_opt_str(&mut hasher, s.volume.as_deref());
    hash_opt_str(&mut hasher, s.competition_diagram.as_deref());
    hasher.finalize().into()
}

/// SHA-256 over every [`DesignRecord`] field EXCEPT [`DesignRecord::version`] itself --
/// including each attachment's METADATA (id/name/url/size), never content (this
/// function never sees attachment content in the first place -- see
/// `diagram_catalog::db::sqlite::Database::get_diagram_full_meta`'s own doc comment).
fn hash_record(r: &DesignRecord) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(r.entry_id.to_le_bytes());
    hash_str(&mut hasher, &r.title);
    hash_str(&mut hasher, &r.url);
    hash_opt_str(&mut hasher, r.design_id.as_deref());
    hash_str(&mut hasher, &r.page_url);
    hash_opt_str(&mut hasher, r.diagram_image_name.as_deref());
    hash_opt_bytes(&mut hasher, r.diagram_image_data.as_deref());
    hash_opt_str(&mut hasher, r.competition_diagram.as_deref());
    hash_opt_str(&mut hasher, r.lw_ratio.as_deref());
    hash_opt_str(&mut hasher, r.refractive_index.as_deref());
    hash_opt_str(&mut hasher, r.index_gear.as_deref());
    hash_opt_str(&mut hasher, r.volume.as_deref());
    hash_opt_str(&mut hasher, r.facets_count.as_deref());
    hash_opt_str(&mut hasher, r.shape.as_deref());
    hash_opt_str(&mut hasher, r.designer_info.as_deref());
    hasher.update((r.angle_settings.len() as u64).to_le_bytes());
    for a in &r.angle_settings {
        hasher.update(a.order_index.to_le_bytes());
        hash_str(&mut hasher, &a.facet);
        hash_str(&mut hasher, &a.angle);
        hash_str(&mut hasher, &a.index);
        hash_str(&mut hasher, &a.notes);
    }
    hasher.update((r.attachments.len() as u64).to_le_bytes());
    for f in &r.attachments {
        hasher.update(f.id.to_le_bytes());
        hash_str(&mut hasher, &f.name);
        hash_str(&mut hasher, &f.url);
        hasher.update(f.size.to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagram_catalog::model::{
        detail::FacetDiagramDetail, entry::FacetDiagramEntry, file::AttachedFile,
    };

    /// Builds a fresh, populated temp database (read-write, via `Database::new`) and
    /// returns the path -- callers reopen it `Database::open_read_only`, matching how
    /// `crate::serve::open_library_database` actually opens the real one. Per this
    /// crate's own hard rule: tests never touch `facet_diagrams.sqlite`, only their own
    /// throwaway temp files.
    fn populated_temp_db() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "gemray-worker-library-test-{}-{}.sqlite",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path_str = path.to_str().unwrap();
        let db = Database::new(Some(path_str)).unwrap();

        let entry_id = db
            .save_diagram_entry(
                &FacetDiagramEntry {
                    title: "Round Brilliant".to_string(),
                    url: "https://example.test/diagram/1".to_string(),
                    design_id: "RB-1".to_string(),
                },
                "facetdiagrams.org",
            )
            .unwrap();

        let mut detail = FacetDiagramDetail {
            page_url: "https://example.test/diagram/1".to_string(),
            shape: Some("Round".to_string()),
            refractive_index: Some("2.417".to_string()),
            attached_files: vec![AttachedFile {
                name: "schedule.pdf".to_string(),
                url: "https://example.test/schedule.pdf".to_string(),
                content: vec![1, 2, 3, 4, 5],
            }],
            ..Default::default()
        };
        detail.angle_settings_table = vec![diagram_catalog::model::angle::AngleSetting {
            order_index: 0,
            facet: "P1".to_string(),
            angle: "41.0".to_string(),
            index: "96".to_string(),
            notes: String::new(),
        }];
        db.save_diagram_detail(&detail, entry_id).unwrap();

        path
    }

    #[test]
    fn search_returns_the_seeded_design_with_a_version_hash() {
        let path = populated_temp_db();
        let db = Database::open_read_only(path.to_str().unwrap()).unwrap();

        let response = handle_request(
            &LibraryRequest::Search {
                query: "Round".to_string(),
                shape_filter: "All".to_string(),
                gear_filter: "All".to_string(),
                range: RangeFilterWire::default(),
            },
            &db,
        );
        let LibraryResponse::SearchResults(results) = response else {
            panic!("expected SearchResults, got {response:?}");
        };
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Round Brilliant");
        assert_ne!(results[0].version, [0u8; 32]);

        drop(db);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fetch_design_returns_metadata_only_never_attachment_content() {
        let path = populated_temp_db();
        let db = Database::open_read_only(path.to_str().unwrap()).unwrap();

        let response = handle_request(&LibraryRequest::FetchDesign { entry_id: 1 }, &db);
        let LibraryResponse::Design(record) = response else {
            panic!("expected Design, got {response:?}");
        };
        assert_eq!(record.attachments.len(), 1);
        assert_eq!(record.attachments[0].name, "schedule.pdf");
        assert_eq!(record.attachments[0].size, 5);
        assert_eq!(record.angle_settings.len(), 1);
        assert_ne!(record.version, [0u8; 32]);

        drop(db);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fetch_design_for_an_unknown_id_is_not_found() {
        let path = populated_temp_db();
        let db = Database::open_read_only(path.to_str().unwrap()).unwrap();

        let response = handle_request(&LibraryRequest::FetchDesign { entry_id: 999 }, &db);
        assert_eq!(response, LibraryResponse::NotFound);

        drop(db);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fetch_attachment_returns_exactly_that_attachments_bytes() {
        let path = populated_temp_db();
        let db = Database::open_read_only(path.to_str().unwrap()).unwrap();

        let response = handle_request(&LibraryRequest::FetchAttachment { attachment_id: 1 }, &db);
        assert_eq!(
            response,
            LibraryResponse::Attachment {
                name: "schedule.pdf".to_string(),
                content: vec![1, 2, 3, 4, 5],
            }
        );

        drop(db);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fetch_attachment_for_an_unknown_id_is_not_found() {
        let path = populated_temp_db();
        let db = Database::open_read_only(path.to_str().unwrap()).unwrap();

        let response = handle_request(&LibraryRequest::FetchAttachment { attachment_id: 999 }, &db);
        assert_eq!(response, LibraryResponse::NotFound);

        drop(db);
        std::fs::remove_file(&path).ok();
    }

    /// `FilterOptions` serves `Database::get_unique_shapes`, which is the UNION of the
    /// seeded canonical vocabulary (`DEFAULT_SHAPES`) and the shapes actually present
    /// in the served library -- not just the latter. This used to assert exactly
    /// `["Round"]`, which was correct only while a fresh database had no vocabulary at
    /// all; a remote client now receives the full picker list, exactly as a local one
    /// does, so it can offer the same choices without a second round trip.
    #[test]
    fn filter_options_reports_the_seeded_shape_alongside_the_canonical_vocabulary() {
        let path = populated_temp_db();
        let db = Database::open_read_only(path.to_str().unwrap()).unwrap();

        let response = handle_request(&LibraryRequest::FilterOptions, &db);
        let LibraryResponse::FilterOptions { shapes, .. } = response else {
            panic!("expected FilterOptions, got {response:?}");
        };

        assert!(
            shapes.contains(&"Round".to_string()),
            "the served library's own shape must still be reported, got {shapes:?}"
        );
        assert!(
            shapes.contains(&"Marquise".to_string()),
            "the seeded canonical vocabulary must reach a remote client too, got {shapes:?}"
        );
        // "Round" is both seeded AND present on the fixture design -- the union must
        // dedupe it rather than offering the same choice twice.
        assert_eq!(
            shapes.iter().filter(|s| *s == "Round").count(),
            1,
            "a shape in both the vocabulary and the data must appear once, got {shapes:?}"
        );

        drop(db);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn two_designs_with_different_fields_get_different_version_hashes() {
        let path = populated_temp_db();
        let db = Database::new(Some(path.to_str().unwrap())).unwrap();
        db.save_diagram_entry(
            &FacetDiagramEntry {
                title: "Emerald Cut".to_string(),
                url: "https://example.test/diagram/2".to_string(),
                design_id: "EC-1".to_string(),
            },
            "facetdiagrams.org",
        )
        .unwrap();
        drop(db);

        let ro = Database::open_read_only(path.to_str().unwrap()).unwrap();
        let response = handle_request(
            &LibraryRequest::Search {
                query: String::new(),
                shape_filter: "All".to_string(),
                gear_filter: "All".to_string(),
                range: RangeFilterWire::default(),
            },
            &ro,
        );
        let LibraryResponse::SearchResults(results) = response else {
            panic!("expected SearchResults, got {response:?}");
        };
        assert_eq!(results.len(), 2);
        assert_ne!(results[0].version, results[1].version);

        drop(ro);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn search_page_returns_no_next_cursor_when_the_page_is_not_full() {
        let path = populated_temp_db();
        let db = Database::open_read_only(path.to_str().unwrap()).unwrap();

        let response = handle_request(
            &LibraryRequest::SearchPage {
                query: String::new(),
                shape_filter: "All".to_string(),
                gear_filter: "All".to_string(),
                range: RangeFilterWire::default(),
                cursor: None,
            },
            &db,
        );
        let LibraryResponse::SearchResultsPage {
            results,
            next_cursor,
        } = response
        else {
            panic!("expected SearchResultsPage, got {response:?}");
        };
        assert_eq!(results.len(), 1);
        assert_eq!(
            next_cursor, None,
            "one row is far short of a full page -- there is nothing more to fetch"
        );

        drop(db);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn search_page_cursor_excludes_rows_already_returned_by_an_earlier_page() {
        let path = populated_temp_db();
        let db = Database::new(Some(path.to_str().unwrap())).unwrap();
        let second_entry_id = db
            .save_diagram_entry(
                &FacetDiagramEntry {
                    title: "Emerald Cut".to_string(),
                    url: "https://example.test/diagram/2".to_string(),
                    design_id: "EC-1".to_string(),
                },
                "facetdiagrams.org",
            )
            .unwrap();
        drop(db);

        let ro = Database::open_read_only(path.to_str().unwrap()).unwrap();

        // A first page with no cursor sees both designs.
        let first = handle_request(
            &LibraryRequest::SearchPage {
                query: String::new(),
                shape_filter: "All".to_string(),
                gear_filter: "All".to_string(),
                range: RangeFilterWire::default(),
                cursor: None,
            },
            &ro,
        );
        let LibraryResponse::SearchResultsPage {
            results: first_results,
            ..
        } = first
        else {
            panic!("expected SearchResultsPage, got {first:?}");
        };
        assert_eq!(first_results.len(), 2);
        let first_id = first_results[0].entry_id;

        // Re-requesting with that first row's id as the cursor must see only what came
        // strictly after it, not re-serve it -- the keyset-pagination property a
        // multi-page walk relies on to never see the same row twice.
        let second = handle_request(
            &LibraryRequest::SearchPage {
                query: String::new(),
                shape_filter: "All".to_string(),
                gear_filter: "All".to_string(),
                range: RangeFilterWire::default(),
                cursor: Some(first_id),
            },
            &ro,
        );
        let LibraryResponse::SearchResultsPage {
            results: second_results,
            next_cursor,
        } = second
        else {
            panic!("expected SearchResultsPage, got {second:?}");
        };
        assert_eq!(second_results.len(), 1);
        assert_eq!(second_results[0].entry_id, second_entry_id);
        assert_eq!(next_cursor, None);

        drop(ro);
        std::fs::remove_file(&path).ok();
    }
}

use crate::model::{angle::AngleSetting, file::AttachedFile};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetDiagramEntry {
    pub title: String,
    pub url: String,
    pub design_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagramListItem {
    pub id: i64,
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
}

/// One attachment's metadata -- id, name, url, and byte size -- WITHOUT its content.
///
/// The counterpart of [`crate::model::file::AttachedFile`] that a caller reaches for
/// when it needs to know WHAT attachments a design has (and how large each is) without
/// paying to load every one's bytes into memory -- see
/// [`crate::db::sqlite::Database::get_diagram_full_meta`], whose whole reason for
/// existing alongside [`crate::db::sqlite::Database::get_diagram_full`] is exactly this
/// distinction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachedFileMeta {
    pub id: i64,
    pub name: String,
    pub url: String,
    pub size: i64,
}

/// The same record as [`FullDiagramRecord`], except [`Self::attached_files`] carries
/// each attachment's METADATA only, never its `content`.
///
/// See [`crate::db::sqlite::Database::get_diagram_full_meta`]'s own doc comment for why
/// this exists as a separate query rather than [`FullDiagramRecord`] with the content
/// field simply discarded after loading (discarding after the fact still pays to load
/// it; this query never selects `content` at all).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullDiagramMeta {
    pub entry_id: i64,
    pub title: String,
    pub url: String,
    pub design_id: Option<String>,
    pub page_url: String,
    pub diagram_image_name: Option<String>,
    pub diagram_image_data: Option<Vec<u8>>,
    pub competition_diagram: Option<String>,
    pub lw_ratio: Option<String>,
    pub refractive_index: Option<String>,
    pub index_gear: Option<String>,
    pub volume: Option<String>,
    pub facets_count: Option<String>,
    pub shape: Option<String>,
    pub designer_info: Option<String>,
    pub angle_settings: Vec<AngleSetting>,
    pub attached_files: Vec<AttachedFileMeta>,
}

/// Widened to carry every `diagram_details` column [`crate::model::detail::
/// FacetDiagramDetail`] has.
///
/// It used to stop at `designer_info`, silently omitting `hw_ratio`/`tw_ratio`/
/// `uw_ratio`/`pw_ratio`/`cw_ratio`/`symmetry_order`/`mirror_symmetry`/`designer`/
/// `source_citation`/`pdf_file`/`gem_file`/`shape_category`. That omission was never
/// just an inconvenience: a caller that
/// built a fresh `FacetDiagramDetail` from a `FullDiagramRecord` (to feed
/// `Database::save_diagram_detail`, which fully replaces a design's detail row) would
/// silently zero every one of those fields on every such save -- see
/// `Database::update_diagram_metadata`'s own doc comment for the full story and the
/// narrow-`UPDATE` method that exists specifically so a metadata edit never needs to
/// take that path at all. Widening this struct closes the trap at its root for every
/// OTHER reader too (this crate has exactly one construction site,
/// `Database::get_diagram_full`) -- a caller that only reads a handful of fields off
/// `full.*` is completely unaffected by the extra ones now being present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullDiagramRecord {
    pub entry_id: i64,
    pub title: String,
    pub url: String,
    pub design_id: Option<String>,
    pub page_url: String,
    pub diagram_image_name: Option<String>,
    pub diagram_image_data: Option<Vec<u8>>,
    pub competition_diagram: Option<String>,
    pub lw_ratio: Option<String>,
    pub refractive_index: Option<String>,
    pub index_gear: Option<String>,
    pub volume: Option<String>,
    pub facets_count: Option<String>,
    pub shape: Option<String>,
    pub designer_info: Option<String>,
    pub hw_ratio: Option<String>,
    pub tw_ratio: Option<String>,
    pub uw_ratio: Option<String>,
    pub pw_ratio: Option<String>,
    pub cw_ratio: Option<String>,
    pub symmetry_order: Option<String>,
    pub mirror_symmetry: Option<bool>,
    pub designer: Option<String>,
    pub source_citation: Option<String>,
    pub pdf_file: Option<String>,
    pub gem_file: Option<String>,
    pub shape_category: Option<String>,
    pub angle_settings: Vec<AngleSetting>,
    pub attached_files: Vec<AttachedFile>,
}

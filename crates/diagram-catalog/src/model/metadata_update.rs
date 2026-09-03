/// Fields a user might legitimately hand-correct on an already-imported design.
///
/// Used by the detail view's metadata editor (see `apps/diagram-gui`'s `gui::detail`/
/// `detail_header.slint`). Deliberately NOT title -- that lives in `diagram_entries`,
/// not `diagram_details`, and already has its own narrow, correct setter
/// ([`crate::db::sqlite::Database::rename_diagram_entry`]) with none of this struct's
/// reason for existing.
///
/// Every field is `Option<...>` because the underlying column is nullable and a blank
/// field in the editor is a valid edit (clears that value) -- there is no separate
/// "leave this field unchanged" state, because [`crate::db::sqlite::Database::
/// update_diagram_metadata`]'s whole point is a single `UPDATE` naming exactly these
/// columns and no others. The editor always pre-fills its form from the record it just
/// read, so "the user didn't touch this field" and "the user resubmitted the same
/// value" are the same thing on the wire -- there is no information lost by not having
/// a third "unchanged" state.
///
/// Deliberately excludes every other `diagram_details` column -- `page_url`,
/// `diagram_image_name`/`diagram_image_data`, `competition_diagram`, `tw_ratio`,
/// `uw_ratio`, the `designer`/`source_citation` split, `pdf_file`, `gem_file`,
/// `shape_category` -- plus `angle_settings`/`attached_files` entirely. That is the
/// whole fix for the trap this struct exists to route around: see
/// `update_diagram_metadata`'s own doc comment for what a naive read-modify-write
/// through [`crate::model::detail::FacetDiagramDetail`]/[`crate::db::sqlite::Database::
/// save_diagram_detail`] would have silently zeroed instead.
///
/// `designer` here is the free-text `designer_info` column (what `detail_header.slint`
/// actually displays as "Designed by ..." and what search matches against) -- not the
/// separate machine-split `designer`/`source_citation` pair `FacetDiagramDetail` also
/// carries. There is no UI for editing that split independently, and re-deriving it
/// from a hand-edited free-text field would risk a worse mismatch than just leaving it
/// as whatever the original import produced, so this update path leaves it alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetadataUpdate {
    pub designer_info: Option<String>,
    pub shape: Option<String>,
    pub refractive_index: Option<String>,
    pub index_gear: Option<String>,
    pub facets_count: Option<String>,
    pub symmetry_order: Option<String>,
    pub mirror_symmetry: Option<bool>,
    pub lw_ratio: Option<String>,
    pub hw_ratio: Option<String>,
    pub cw_ratio: Option<String>,
    pub pw_ratio: Option<String>,
    pub volume: Option<String>,
}

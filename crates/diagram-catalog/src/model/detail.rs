use super::{angle::AngleSetting, file::AttachedFile};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FacetDiagramDetail {
    pub page_url: String,
    pub diagram_image_name: Option<String>,
    pub diagram_image_data: Option<Vec<u8>>,
    pub angle_settings_table: Vec<AngleSetting>,
    pub attached_files: Vec<AttachedFile>,

    // Specific metadata fields
    pub competition_diagram: Option<String>,
    pub lw_ratio: Option<String>,
    pub refractive_index: Option<String>,
    pub index_gear: Option<String>,
    pub volume: Option<String>,
    pub facets_count: Option<String>,
    pub shape: Option<String>,
    pub designer_info: Option<String>,

    // Proportion ratios and symmetry, as printed on a design sheet's metadata block
    // alongside `lw_ratio`/`volume` above. `hw_ratio` is the odd one out: most sheets
    // do not print it, so it usually arrives from a design's own metadata rather than
    // from the schedule. All `None` by default -- whatever fills a `FacetDiagramDetail`
    // populates only the subset it can actually supply, and a missing ratio must stay
    // missing rather than be defaulted to a fabricated number. `apps/diagram-gui`'s
    // importer derives several of these from the design's own geometry; see its
    // `gui::library::apply_measured_metadata`.
    pub hw_ratio: Option<String>,
    pub tw_ratio: Option<String>,
    pub uw_ratio: Option<String>,
    pub pw_ratio: Option<String>,
    pub cw_ratio: Option<String>,
    /// The rotational fold count (e.g. `4` in "4-fold, mirror-image symmetry").
    pub symmetry_order: Option<String>,
    /// Whether the sheet additionally declares mirror-image symmetry.
    pub mirror_symmetry: Option<bool>,

    /// The designer alone (e.g. `"Capps, Jerry"`) -- the first half of what
    /// [`Self::designer_info`] holds as one free-text `"Designer; Publication
    /// citation"` string. Split out into its own field so "every design by X" is a
    /// real equality query against an indexed column rather than a `LIKE '%X%'`
    /// scan over the joined string.
    ///
    /// `designer_info` is deliberately *kept* alongside this and
    /// [`Self::source_citation`] rather than replaced by them: it is what
    /// `db::sqlite::Database::search_diagrams`' free-text `LIKE` matches, what
    /// `find_cross_source_duplicates` compares, and what
    /// `apps/diagram-gui` renders in both its list and detail views -- so it stays
    /// as the display/search convenience, with these two as the queryable halves.
    pub designer: Option<String>,
    /// The publication citation alone (e.g. `"Lapidary Journal, May 1994, p95"`) --
    /// the second half of [`Self::designer_info`]. See [`Self::designer`].
    pub source_citation: Option<String>,

    // Competition-entry pages only. facetdiagrams.org serves two kinds of detail
    // page: regular designs (`/diagram/...`), which carry an inline `<svg>` and a
    // `<span>`-based designer/citation `<li>`, and competition entries
    // (`/diagramus/...`), which have no inline diagram SVG at all and instead label
    // their attachments `PDF:`/`GEM:` in a `div.attachmentPost` list. The three
    // fields below come from that second kind of page and are `None` on the first --
    // see `crate::parser::parse_attachment_labels`.
    /// The `PDF:` attachment's file name (e.g. `"2002SSCMasters.pdf"`), the join key
    /// between a competition entry and the PDF corpus in
    /// [`Self::attached_files`].
    ///
    /// A plain string, *not* a foreign key: several competition designs routinely
    /// name the same PDF, because one file is a multi-design results booklet for a
    /// whole competition class (e.g. `2002SSCMasters.pdf` is shared by every entry
    /// in that year's Masters class) -- a many-to-one relationship a per-design
    /// reference could not express.
    pub pdf_file: Option<String>,
    /// The `GEM:` attachment's file name (e.g. `"USFG-SSC-2020-Novice-1.gem"`), a
    /// `GemCAD` design file. Frequently blank on the page (rendered as a literal
    /// `""`), which reads back here as `None` rather than an empty string.
    pub gem_file: Option<String>,
    /// The numbered shape-category id from the `Shape:` metadata item, e.g. `5` from
    /// `"05. Pear"` -- the stable half of that label, whose text half
    /// ([`Self::shape`]) `crate::util::clean_shape_string` already strips it from.
    ///
    /// Kept as a decimal string here, the same way [`Self::symmetry_order`] is, and
    /// bound straight into an INTEGER column. Not derivable from the article's
    /// `shape-diagram-NN-name` CSS class, which `crate::util::map_shape_class` shows
    /// carries *two* conflicting numberings for several shapes (`04`/`10` both being
    /// Emerald, for instance) and so is not a stable id at all.
    pub shape_category: Option<String>,
}

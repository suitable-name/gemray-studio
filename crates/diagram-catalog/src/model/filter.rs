//! Query-time types for the catalogue's numeric range filters (refractive index, L/W
//! ratio, volume, facet count) and the actual data bounds they're built against.

/// Optional min/max bounds to apply to `search_diagrams`, one independent pair per
/// numeric attribute.
///
/// Each bound is applied only when `Some`; a `None` bound leaves that side of that
/// attribute unconstrained (and, if *both* sides of an attribute are `None`, no SQL
/// predicate is added for it at all -- rows with no value for that attribute are still
/// returned, matching the unfiltered "All" behavior of the existing shape/gear
/// dropdowns).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RangeFilter {
    pub ri_min: Option<f64>,
    pub ri_max: Option<f64>,
    pub lw_min: Option<f64>,
    pub lw_max: Option<f64>,
    pub volume_min: Option<f64>,
    pub volume_max: Option<f64>,
    pub facets_min: Option<i64>,
    pub facets_max: Option<i64>,
}

impl RangeFilter {
    /// `true` when every bound is unset -- the query this produces is identical to
    /// omitting range filtering entirely.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ri_min.is_none()
            && self.ri_max.is_none()
            && self.lw_min.is_none()
            && self.lw_max.is_none()
            && self.volume_min.is_none()
            && self.volume_max.is_none()
            && self.facets_min.is_none()
            && self.facets_max.is_none()
    }
}

/// The *usable* min/max bounds for each range-filterable attribute, as of the last
/// time the catalogue was queried -- drives the sliders' scale in the UI.
///
/// The minimum is the real minimum present in the data. The maximum is **not** the raw
/// maximum: it's a robust percentile (see `Database::get_attribute_ranges` /
/// `RANGE_BOUND_PERCENTILE`) chosen so a handful of data-entry errors (e.g. one design
/// with an impossible `volume` two orders of magnitude beyond every other row) can't
/// compress the entire rest of the catalogue into a sliver of a slider's travel. Rows
/// beyond this bound are not excluded from search -- they just don't get to set the
/// control's scale; see `search_diagrams`/`RangeFilter` for how an unmoved slider side
/// stays unfiltered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttributeRanges {
    pub ri: (f64, f64),
    pub lw_ratio: (f64, f64),
    pub volume: (f64, f64),
    pub facets: (i64, i64),
}

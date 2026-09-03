use super::Database;
use crate::model::filter::{AttributeRanges, RangeFilter};
use anyhow::Result;
use std::fmt::Write as _;
use tracing::warn;

/// The percentile used as the range-filter sliders' *usable* upper bound (see
/// `Database::get_attribute_ranges`). p99 was chosen by checking the real catalogue's
/// distribution for all four range-filterable attributes: refractive index, L/W ratio,
/// and facet count all have the same long-right-tail shape as volume (p99 sits well
/// short of the raw max in every case), so the same percentile is applied uniformly
/// rather than special-casing volume.
const RANGE_BOUND_PERCENTILE: f64 = 99.0;

/// A raw maximum more than this many times the derived bound is treated as a probable
/// data-quality outlier worth logging (see `warn_if_outlier`) rather than a
/// legitimately wide distribution. Chosen loosely: on the real catalogue, p99-to-max
/// ratios for genuine long-tail attributes (RI, L/W, facets) stay under ~3x, while the
/// one confirmed data error (a `volume` of 195 against a p99 of 0.88) is on the order
/// of 200x -- so 5x sits comfortably between "wide but real" and "obviously wrong"
/// without needing per-attribute tuning.
const OUTLIER_WARNING_RATIO: f64 = 5.0;

impl Database {
    /// Returns the total number of diagram entries stored.
    ///
    /// # Errors
    ///
    /// Never actually fails: a failed `COUNT` query is caught internally and treated
    /// as a count of 0 rather than propagated. Returns `Result` to match the shape of
    /// the other query methods on this type.
    pub fn get_total_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM diagram_entries", [], |r| r.get(0))
            .unwrap_or(0);
        Ok(count as usize)
    }

    /// Returns the UNION of the seeded `shape_vocabulary` (see `DEFAULT_SHAPES`) and
    /// every distinct non-empty `shape` value actually present in `diagram_details`,
    /// deduplicated, alphabetically sorted.
    ///
    /// Alphabetical across the union -- not canonical-list-first-then-appended --
    /// deliberately: this crate's real ~3,187-design catalogue contributes real
    /// scraped shape strings `DEFAULT_SHAPES` doesn't cover (and never will
    /// exhaustively), so any fixed split between "seeded" and "discovered" entries
    /// would put an arbitrary subset of a filter dropdown out of alphabetical order
    /// for no reason a user could infer. Plain alphabetical is the one ordering a
    /// dropdown reader can always predict, regardless of which side of the union any
    /// given entry came from.
    ///
    /// On a fresh database this still returns the full seeded vocabulary (unlike the
    /// old plain `SELECT DISTINCT`, which returned nothing until some design had
    /// contributed a `shape` value) -- see `Database::migrate_shape_vocabulary`.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or running the underlying `SELECT` query fails.
    /// A row whose value fails to decode as a `String` is silently skipped (via
    /// `.flatten()`) rather than failing the whole call.
    pub fn get_unique_shapes(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT name FROM (
                 SELECT name FROM shape_vocabulary
                 UNION
                 SELECT shape AS name FROM diagram_details WHERE shape IS NOT NULL AND shape != ''
             ) ORDER BY name ASC",
        )?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        let mut shapes = Vec::new();
        for s in rows.flatten() {
            shapes.push(s);
        }
        Ok(shapes)
    }

    /// Returns every distinct non-empty `index_gear` value across all diagram
    /// details, numerically sorted.
    ///
    /// `index_gear` is stored as INTEGER (see `migrate_numeric_columns`); it's cast
    /// back to TEXT here so the return type -- and every caller, which treats gear as
    /// a display/dropdown string, not a number -- stays unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or running the underlying `SELECT DISTINCT`
    /// query fails. A row whose value fails to decode as a `String` is silently
    /// skipped (via `.flatten()`) rather than failing the whole call.
    pub fn get_unique_gears(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT CAST(index_gear AS TEXT) FROM diagram_details WHERE index_gear IS NOT NULL ORDER BY index_gear ASC",
        )?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        let mut gears = Vec::new();
        for g in rows.flatten() {
            gears.push(g);
        }
        Ok(gears)
    }

    /// Returns *usable* min/max bounds across the catalogue for each range-filterable
    /// attribute (refractive index, L/W ratio, volume, facet count) -- the scale the
    /// range-filter sliders are sized against.
    ///
    /// The lower bound is still the real minimum (every attribute's low end sits where
    /// legitimate data actually starts; see the report for the distributions checked).
    /// The upper bound is [`RANGE_BOUND_PERCENTILE`] rather than the raw maximum: real
    /// scraped data has occasional single-row data-entry errors (e.g. a `volume` of
    /// `195` when `vol/w³` cannot physically exceed roughly 1.8 for a real faceted
    /// stone) that would otherwise compress 99%+ of the catalogue into a sliver of the
    /// slider's travel. A raw maximum far beyond the percentile bound is logged as a
    /// probable data-quality issue -- nothing is altered or dropped; the offending
    /// row(s) stay in the database and stay reachable by search whenever that slider
    /// side is left at its default (unfiltered) position, since `search_diagrams` only
    /// adds a bound predicate when a slider has actually moved off its bounds edge (see
    /// `gui::search::active_bound` in `diagram-gui`). They just no longer dictate the
    /// control's scale.
    ///
    /// Returns `(0.0, 0.0)` / `(0, 0)` for an attribute with no non-null values at all
    /// (an empty database) rather than failing.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the underlying per-attribute queries fail.
    pub fn get_attribute_ranges(&self) -> Result<AttributeRanges> {
        Ok(AttributeRanges {
            ri: self.attribute_bounds_f64("refractive_index")?,
            lw_ratio: self.attribute_bounds_f64("lw_ratio")?,
            volume: self.attribute_bounds_f64("volume")?,
            facets: self.attribute_bounds_i64("facets")?,
        })
    }

    /// Every non-null value of `diagram_details.{column}`, sorted ascending. `column`
    /// is always one of this module's own hardcoded field names (never user input), so
    /// building the query via `format!` is safe here.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or running the underlying `SELECT` fails. A row
    /// that fails to decode is silently skipped (via `.flatten()`), matching this
    /// file's existing convention for read queries.
    fn sorted_non_null_reals(&self, column: &str) -> Result<Vec<f64>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {column} FROM diagram_details WHERE {column} IS NOT NULL ORDER BY {column} ASC"
        ))?;
        let rows = stmt.query_map([], |r| r.get::<_, f64>(0))?;
        Ok(rows.flatten().collect())
    }

    /// Integer counterpart of [`Self::sorted_non_null_reals`] -- `facets` is stored as
    /// `INTEGER`, and rusqlite's `f64` decoder doesn't auto-widen an `INTEGER` column,
    /// so it needs its own query rather than reusing the REAL one.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or running the underlying `SELECT` fails. A row
    /// that fails to decode is silently skipped (via `.flatten()`).
    fn sorted_non_null_ints(&self, column: &str) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {column} FROM diagram_details WHERE {column} IS NOT NULL ORDER BY {column} ASC"
        ))?;
        let rows = stmt.query_map([], |r| r.get::<_, i64>(0))?;
        Ok(rows.flatten().collect())
    }

    /// `(min, usable_max)` for a `REAL` attribute column -- see
    /// [`Self::get_attribute_ranges`]'s doc comment for what "usable" means. Logs a
    /// warning (does not error, and touches no data) when the raw maximum sits far
    /// beyond the derived bound, since that gap is the signature of a data-entry error
    /// rather than a legitimately wide distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::sorted_non_null_reals`] fails.
    fn attribute_bounds_f64(&self, column: &str) -> Result<(f64, f64)> {
        let values = self.sorted_non_null_reals(column)?;
        let (Some(&min), Some(&raw_max)) = (values.first(), values.last()) else {
            return Ok((0.0, 0.0));
        };
        let bound_max = percentile_of_sorted(&values, RANGE_BOUND_PERCENTILE).max(min);

        warn_if_outlier(column, raw_max, bound_max);
        Ok((min, bound_max))
    }

    /// Integer counterpart of [`Self::attribute_bounds_f64`], for `facets` (`INTEGER`).
    /// The percentile is computed in `f64` (the same interpolated method as the REAL
    /// columns) and rounded up, so the bound never sits inside a facet count no design
    /// actually has.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::sorted_non_null_ints`] fails.
    fn attribute_bounds_i64(&self, column: &str) -> Result<(i64, i64)> {
        let values = self.sorted_non_null_ints(column)?;
        let (Some(&min), Some(&raw_max)) = (values.first(), values.last()) else {
            return Ok((0, 0));
        };
        let floats: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let bound_max =
            (percentile_of_sorted(&floats, RANGE_BOUND_PERCENTILE).ceil() as i64).max(min);

        warn_if_outlier(column, raw_max as f64, bound_max as f64);
        Ok((min, bound_max))
    }

    /// Searches diagram entries by free-text `query` (matched against title,
    /// designer, and design ID) with optional exact `shape_filter` / `gear_filter`
    /// and optional min/max bounds on refractive index, L/W ratio, volume, and facet
    /// count (`range`; any bound left `None` is unconstrained -- see
    /// [`RangeFilter`]'s doc comment). Capped at [`SEARCH_RESULT_CAP`] results ordered
    /// by entry ID.
    ///
    /// All filtering -- text, shape, gear, and every range bound -- happens in this
    /// one query; nothing is filtered back in Rust.
    ///
    /// This is exactly `self.search_diagrams_page(query, shape_filter, gear_filter,
    /// range, None, SEARCH_RESULT_CAP)` -- kept as its own method (rather than making
    /// every existing caller spell out the extra two arguments) so the local browser's
    /// interactive search box, and every other pre-existing call site, keeps working
    /// unchanged. See [`Self::search_diagrams_page`] for the keyset-paginated form a
    /// mirror walking the whole catalogue needs instead.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or running the assembled `SELECT` query fails,
    /// or if a row fails to decode into a `DiagramListItem`.
    pub fn search_diagrams(
        &self,
        query: &str,
        shape_filter: &str,
        gear_filter: &str,
        range: &RangeFilter,
    ) -> Result<Vec<crate::model::entry::DiagramListItem>> {
        self.search_diagrams_page(
            query,
            shape_filter,
            gear_filter,
            range,
            None,
            SEARCH_RESULT_CAP,
        )
    }

    /// Keyset-paginated counterpart of [`Self::search_diagrams`]: the exact same
    /// filters (`query`/`shape_filter`/`gear_filter`/`range`), plus `after_id` and
    /// `limit`.
    ///
    /// `after_id` is `None` for the first page, or `Some(id)` -- the `id` of the last
    /// row a previous page returned -- to continue strictly after it. Rows are
    /// ordered `ORDER BY de.id ASC`, over `diagram_entries.id` (`INTEGER PRIMARY KEY
    /// AUTOINCREMENT`): unique and strictly increasing, so this ordering is genuinely
    /// total -- no two rows ever tie -- which is what makes paging by `id > after_id`
    /// safe: unlike `OFFSET`, it can never skip or duplicate a row when rows are
    /// inserted between two pages of the same walk, and it stays O(page size) per page
    /// rather than `OFFSET`'s O(offset + page size) table scan.
    ///
    /// A returned page shorter than `limit` (including empty) means every matching row
    /// at or before it has now been seen; a page exactly `limit` long means there may
    /// be more -- the caller re-requests with `after_id` set to this page's last row's
    /// `id`. (A catalogue whose matching-row count happens to be an exact multiple of
    /// `limit` costs one extra, empty final round trip under this rule -- cheap, and
    /// simpler than a second query to check "is there really more" up front.)
    ///
    /// Shares its entire filter-predicate construction and row mapping with
    /// [`Self::search_diagrams`] via [`build_search_predicate`] -- there is exactly one
    /// place the WHERE clause is assembled, so the two can never drift apart.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or running the assembled `SELECT` query fails,
    /// or if a row fails to decode into a `DiagramListItem`.
    pub fn search_diagrams_page(
        &self,
        query: &str,
        shape_filter: &str,
        gear_filter: &str,
        range: &RangeFilter,
        after_id: Option<i64>,
        limit: i64,
    ) -> Result<Vec<crate::model::entry::DiagramListItem>> {
        let (mut sql, mut params) = build_search_predicate(query, shape_filter, gear_filter, range);

        if let Some(after) = after_id {
            sql.push_str(" AND de.id > ? ");
            params.push(Box::new(after));
        }
        sql.push_str(" ORDER BY de.id ASC LIMIT ? ");
        params.push(Box::new(limit));

        let mut stmt = self.conn.prepare(&sql)?;
        let bound: Vec<&dyn rusqlite::ToSql> =
            params.iter().map(std::convert::AsRef::as_ref).collect();

        let rows = stmt.query_map(bound.as_slice(), |row| {
            Ok(crate::model::entry::DiagramListItem {
                id: row.get(0)?,
                title: row.get(1)?,
                url: row.get(2)?,
                design_id: row.get(3)?,
                shape: row.get(4)?,
                index_gear: row.get(5)?,
                facets_count: row.get(6)?,
                designer_info: row.get(7)?,
                lw_ratio: row.get(8)?,
                refractive_index: row.get(9)?,
                volume: row.get(10)?,
                competition_diagram: row.get(11)?,
            })
        })?;

        let mut list = Vec::new();
        for item in rows.flatten() {
            list.push(item);
        }
        Ok(list)
    }
}

/// The result cap [`Database::search_diagrams`] has always applied.
///
/// Also the default page size a mirror walking [`Database::search_diagrams_page`] is
/// sized around (`apps/gemray-worker`'s `SearchPage` handler uses the same value; see
/// that crate's `serve::library` module).
pub const SEARCH_RESULT_CAP: i64 = 1000;

/// Builds the `SELECT ... WHERE ...` predicate (everything through the last filter
/// clause, but not `ORDER BY`/`LIMIT`) shared by [`Database::search_diagrams_page`]
/// (and, through it, [`Database::search_diagrams`]), plus the bound parameters that
/// predicate's `?` placeholders need, in the order they appear in the SQL text.
///
/// `index_gear`/`lw_ratio`/`refractive_index`/`volume` are stored as REAL/INTEGER (see
/// `migrate_numeric_columns`) and cast back to TEXT here so `DiagramListItem`'s fields
/// -- and every caller, which treats them as display strings -- stay unchanged.
fn build_search_predicate(
    query: &str,
    shape_filter: &str,
    gear_filter: &str,
    range: &RangeFilter,
) -> (String, Vec<Box<dyn rusqlite::ToSql>>) {
    let q_pattern = format!("%{}%", query.trim());
    let mut sql = String::from(
        "SELECT de.id, de.title, de.url, de.design_id,
                dd.shape, CAST(dd.index_gear AS TEXT), dd.facets_count, dd.designer_info,
                CAST(dd.lw_ratio AS TEXT), CAST(dd.refractive_index AS TEXT),
                CAST(dd.volume AS TEXT), dd.competition_diagram
         FROM diagram_entries de
         LEFT JOIN diagram_details dd ON de.id = dd.entry_id
         WHERE 1=1 ",
    );
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    if query.trim().is_empty() {
        sql.push_str(" AND (1=1 OR ?1 IS NULL) ");
    } else {
        sql.push_str(
            " AND (de.title LIKE ?1 OR dd.designer_info LIKE ?1 OR de.design_id LIKE ?1) ",
        );
    }
    params.push(Box::new(q_pattern));

    if !shape_filter.is_empty() && shape_filter != "All" {
        let _ = write!(
            sql,
            " AND dd.shape = '{}' ",
            shape_filter.replace('\'', "''")
        );
    }

    if !gear_filter.is_empty() && gear_filter != "All" {
        let _ = write!(
            sql,
            " AND dd.index_gear = '{}' ",
            gear_filter.replace('\'', "''")
        );
    }

    // Every bound becomes its own bound parameter (numbered `?2`, `?3`, ... --
    // SQLite auto-numbers plain `?` markers to continue after the explicit `?1`
    // used above for the text search) rather than a string-interpolated literal,
    // since these are numbers a slider hands us, not a short closed vocabulary
    // like shape/gear.
    if let Some(min) = range.ri_min {
        sql.push_str(" AND dd.refractive_index >= ? ");
        params.push(Box::new(min));
    }
    if let Some(max) = range.ri_max {
        sql.push_str(" AND dd.refractive_index <= ? ");
        params.push(Box::new(max));
    }
    if let Some(min) = range.lw_min {
        sql.push_str(" AND dd.lw_ratio >= ? ");
        params.push(Box::new(min));
    }
    if let Some(max) = range.lw_max {
        sql.push_str(" AND dd.lw_ratio <= ? ");
        params.push(Box::new(max));
    }
    if let Some(min) = range.volume_min {
        sql.push_str(" AND dd.volume >= ? ");
        params.push(Box::new(min));
    }
    if let Some(max) = range.volume_max {
        sql.push_str(" AND dd.volume <= ? ");
        params.push(Box::new(max));
    }
    if let Some(min) = range.facets_min {
        sql.push_str(" AND dd.facets >= ? ");
        params.push(Box::new(min));
    }
    if let Some(max) = range.facets_max {
        sql.push_str(" AND dd.facets <= ? ");
        params.push(Box::new(max));
    }

    (sql, params)
}

/// Linear-interpolation percentile (the same "linear" method `numpy.percentile`
/// defaults to): walks to fractional index `p/100 * (n-1)` in the sorted slice and
/// interpolates between the two bracketing values. `values` must be sorted ascending
/// and non-empty (callers only reach this after checking `.first()`).
pub(super) fn percentile_of_sorted(values: &[f64], p: f64) -> f64 {
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    let idx = (p / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    frac.mul_add(values[hi] - values[lo], values[lo])
}

/// Logs a warning when `raw_max` sits more than [`OUTLIER_WARNING_RATIO`] times beyond
/// `bound`, the signature of a data-entry error rather than a legitimately wide
/// distribution (see `RANGE_BOUND_PERCENTILE`'s doc comment). Purely observational --
/// never errors, never touches any row.
fn warn_if_outlier(column: &str, raw_max: f64, bound: f64) {
    if bound > 0.0 && raw_max > bound * OUTLIER_WARNING_RATIO {
        warn!(
            "diagram_details.{column}: raw max {raw_max} is {:.0}x its p{RANGE_BOUND_PERCENTILE} \
             range-filter bound of {bound} -- likely a data-entry error in the source catalogue, \
             not a real outlier. The row is left untouched and still matches searches while that \
             slider side is unfiltered; it just no longer sets the slider's scale.",
            raw_max / bound
        );
    }
}

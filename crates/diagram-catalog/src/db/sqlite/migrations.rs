use super::{DEFAULT_SHAPES, Database, LEGACY_SOURCE_ID};
use crate::model::facets::parse_facets_count;
use anyhow::{Context, Result};
use rusqlite::{Connection, Transaction, params};
use tracing::{debug, info};

impl Database {
    /// Retypes `diagram_details`'s numeric-but-stored-as-TEXT columns
    /// (`refractive_index`, `lw_ratio`, `volume` -> REAL; `index_gear` -> INTEGER) and
    /// splits `facets_count` (e.g. `"55+6"`) into new `facets`/`girdle_facets` INTEGER
    /// columns, leaving the original `facets_count` TEXT column in place for display.
    ///
    /// Idempotent: gated on whether the `facets` column already exists (the last thing
    /// this migration adds), so a second call on an already-migrated database is a
    /// cheap no-op rather than re-running the retype. Runs inside one transaction --
    /// each retyped column's *replacement* is fully populated (`ADD COLUMN` + `UPDATE`)
    /// before its original TEXT column is dropped, and the whole thing rolls back
    /// atomically on any failure -- so a crash or error partway through can never leave
    /// the database missing a column's data.
    ///
    /// # Errors
    ///
    /// Returns an error if checking for the `facets` column, starting/committing the
    /// transaction, or any of the retype/split steps within it fails.
    pub(super) fn migrate_numeric_columns(&self) -> Result<()> {
        if Self::column_exists(&self.conn, "diagram_details", "facets")? {
            debug!("Numeric column migration already applied; skipping.");
            return Ok(());
        }

        info!("Migrating diagram_details TEXT columns to numeric types...");
        let tx = self
            .conn
            .unchecked_transaction()
            .context("Failed to start numeric-column migration transaction")?;

        retype_text_column_to_numeric(&tx, "refractive_index", "REAL")?;
        retype_text_column_to_numeric(&tx, "lw_ratio", "REAL")?;
        retype_text_column_to_numeric(&tx, "volume", "REAL")?;
        retype_text_column_to_numeric(&tx, "index_gear", "INTEGER")?;

        tx.execute_batch(
            "ALTER TABLE diagram_details ADD COLUMN facets INTEGER;
             ALTER TABLE diagram_details ADD COLUMN girdle_facets INTEGER;",
        )
        .context("Failed to add facets/girdle_facets columns")?;
        split_facets_count_column(&tx)?;

        tx.commit()
            .context("Failed to commit numeric-column migration")?;
        info!("Numeric column migration complete.");
        Ok(())
    }

    /// Adds `diagram_entries.source_id` (`TEXT NOT NULL DEFAULT` [`LEGACY_SOURCE_ID`])
    /// for a database created before multi-source support existed. Needed so an
    /// existing catalogue can be re-synced, attributed, or selectively cleaned up per
    /// source (see `crate::source::DiagramSource`) without every pre-existing row
    /// silently reading back as `NULL`/unattributed.
    ///
    /// Idempotent, the same way as [`Self::migrate_numeric_columns`]: gated on whether
    /// the column already exists, so a second call (including on a database that was
    /// created fresh, where `create_tables_if_not_exist`'s own `CREATE TABLE` already
    /// includes this column) is a cheap no-op rather than an error from re-adding an
    /// existing column.
    ///
    /// # Errors
    ///
    /// Returns an error if checking for the column, or adding it, fails.
    pub(super) fn migrate_source_id_column(&self) -> Result<()> {
        if Self::column_exists(&self.conn, "diagram_entries", "source_id")? {
            debug!("source_id column migration already applied; skipping.");
            return Ok(());
        }

        info!("Adding diagram_entries.source_id column...");
        self.conn
            .execute_batch(&format!(
                "ALTER TABLE diagram_entries ADD COLUMN source_id TEXT NOT NULL DEFAULT '{LEGACY_SOURCE_ID}';"
            ))
            .context("Failed to add diagram_entries.source_id column")?;
        info!("source_id column migration complete.");
        Ok(())
    }

    /// Adds `diagram_details`' proportion-ratio and symmetry columns (`hw_ratio`,
    /// `tw_ratio`, `uw_ratio`, `pw_ratio`, `cw_ratio`, `symmetry_order`,
    /// `mirror_symmetry`) for a database created before this crate captured them.
    ///
    /// All nullable, all brand-new (no existing TEXT column to retype the way
    /// [`Self::migrate_numeric_columns`] does) -- so unlike that migration, this
    /// one is a plain `ADD COLUMN` per field, same shape as
    /// [`Self::migrate_source_id_column`]. Idempotent the same way: gated on
    /// whether `hw_ratio` already exists (the last of the seven, chosen
    /// arbitrarily), so a second call -- including on a freshly created database,
    /// where `create_tables_if_not_exist`'s own `CREATE TABLE` already includes
    /// all seven -- is a cheap no-op rather than an error from re-adding an
    /// existing column.
    ///
    /// # Errors
    ///
    /// Returns an error if checking for the column, or adding any of the seven,
    /// fails.
    pub(super) fn migrate_proportions_columns(&self) -> Result<()> {
        if Self::column_exists(&self.conn, "diagram_details", "hw_ratio")? {
            debug!("Proportions/symmetry column migration already applied; skipping.");
            return Ok(());
        }

        info!("Adding diagram_details proportion-ratio and symmetry columns...");
        self.conn
            .execute_batch(
                "ALTER TABLE diagram_details ADD COLUMN hw_ratio REAL;
                 ALTER TABLE diagram_details ADD COLUMN tw_ratio REAL;
                 ALTER TABLE diagram_details ADD COLUMN uw_ratio REAL;
                 ALTER TABLE diagram_details ADD COLUMN pw_ratio REAL;
                 ALTER TABLE diagram_details ADD COLUMN cw_ratio REAL;
                 ALTER TABLE diagram_details ADD COLUMN symmetry_order INTEGER;
                 ALTER TABLE diagram_details ADD COLUMN mirror_symmetry BOOLEAN;",
            )
            .context("Failed to add proportion-ratio/symmetry columns")?;
        info!("Proportions/symmetry column migration complete.");
        Ok(())
    }

    /// Adds `diagram_details`' split designer/citation columns (`designer`,
    /// `source_citation`) and the competition-entry columns (`pdf_file`, `gem_file`,
    /// `shape_category`) for a database created before this crate captured them,
    /// plus the index that makes "every design by X" an indexed lookup rather than
    /// the `LIKE '%X%'` scan over `designer_info` it is today.
    ///
    /// `designer_info` is deliberately left in place rather than dropped in favour of
    /// the two halves that now sit beside it -- see
    /// `FacetDiagramDetail::designer`'s doc comment for the callers that still read
    /// it. Nothing here backfills the new columns from it either: they are populated
    /// by the parser on the next sync, the same way
    /// [`Self::migrate_proportions_columns`]' columns were.
    ///
    /// All five nullable and brand-new, so like that migration this is a plain
    /// `ADD COLUMN` per field. Idempotent the same way: gated on whether `designer`
    /// already exists (the first of the five, chosen arbitrarily), so a second call --
    /// including on a freshly created database, where `create_tables_if_not_exist`'s
    /// own `CREATE TABLE` already includes all five -- is a cheap no-op rather than an
    /// error from re-adding an existing column. The index sits outside that gate; see
    /// the comment on it for why it cannot live in either of the two other places.
    ///
    /// # Errors
    ///
    /// Returns an error if checking for the column, adding any of the five, or
    /// creating the index fails.
    pub(super) fn migrate_designer_and_attachment_columns(&self) -> Result<()> {
        if Self::column_exists(&self.conn, "diagram_details", "designer")? {
            debug!("Designer/attachment columns already present; skipping the ADD COLUMN step.");
        } else {
            info!("Adding diagram_details designer-split and competition-entry columns...");
            self.conn
                .execute_batch(
                    "ALTER TABLE diagram_details ADD COLUMN designer TEXT;
                     ALTER TABLE diagram_details ADD COLUMN source_citation TEXT;
                     ALTER TABLE diagram_details ADD COLUMN pdf_file TEXT;
                     ALTER TABLE diagram_details ADD COLUMN gem_file TEXT;
                     ALTER TABLE diagram_details ADD COLUMN shape_category INTEGER;",
                )
                .context("Failed to add designer-split/competition-entry columns")?;
            info!("Designer/attachment column migration complete.");
        }

        // Outside the gate above, and not in `create_tables_if_not_exist` either.
        // A freshly created database already has all five columns from that
        // function's own `CREATE TABLE`, so it takes the skip branch above and would
        // never reach an index created inside it -- while a pre-migration database
        // cannot have the index created in `create_tables_if_not_exist`, because that
        // runs *before* this migration and its `CREATE TABLE IF NOT EXISTS` is a
        // no-op there, leaving `designer` nonexistent at that point. Here, after the
        // ADD COLUMNs, is the one place both cases have the column. `IF NOT EXISTS`
        // keeps the unconditional call a no-op on every subsequent open.
        self.conn
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_diagram_details_designer
                 ON diagram_details (designer);",
            )
            .context("Failed to create the diagram_details.designer index")?;
        Ok(())
    }

    /// Adds `custom_gem_materials`' crystal-classification columns for the
    /// custom-material editor: `crystal_system`, `optical_character`, both
    /// TEXT (a `gemray` enum variant name, e.g. `"Trigonal"`), and
    /// `biaxial_delta_beta_alpha` REAL. See `CustomMaterialRow`'s field doc comments
    /// for what each stores and why as plain text/`f32` rather than the `gemray`
    /// enums themselves.
    ///
    /// All three nullable and brand-new (same shape as
    /// `migrate_designer_and_attachment_columns` above, whose idiom this copies): a
    /// plain `ADD COLUMN` per field, gated on whether `crystal_system` (the first of
    /// the three, chosen arbitrarily) already exists, so a second call -- including on
    /// a freshly created database, where `create_tables_if_not_exist`'s own `CREATE
    /// TABLE` already includes all three -- is a cheap no-op rather than an error from
    /// re-adding an existing column. `NULL` on every pre-existing row is exactly the
    /// "not stored, infer as `GemMaterial::new_custom` already does" state --
    /// `CustomMaterialRow`'s fields are the same `Option` shape for the same reason, so
    /// no backfill step is needed here the way `migrate_source_id_column` needed one.
    ///
    /// # Errors
    ///
    /// Returns an error if checking for the `crystal_system` column or adding any of
    /// the three fails.
    pub(super) fn migrate_crystal_optics_columns(&self) -> Result<()> {
        if Self::column_exists(&self.conn, "custom_gem_materials", "crystal_system")? {
            debug!("Crystal-optics columns already present; skipping the ADD COLUMN step.");
            return Ok(());
        }

        info!("Adding custom_gem_materials crystal-classification columns...");
        self.conn
            .execute_batch(
                "ALTER TABLE custom_gem_materials ADD COLUMN crystal_system TEXT;
                 ALTER TABLE custom_gem_materials ADD COLUMN optical_character TEXT;
                 ALTER TABLE custom_gem_materials ADD COLUMN biaxial_delta_beta_alpha REAL;",
            )
            .context("Failed to add custom_gem_materials crystal-classification columns")?;
        info!("Crystal-optics column migration complete.");
        Ok(())
    }

    /// Creates `shape_vocabulary` for a database created before it existed, and seeds
    /// (or re-seeds) it from [`DEFAULT_SHAPES`] -- the fix for `get_unique_shapes`
    /// returning nothing on a fresh database (no imported design has contributed a
    /// `shape` value yet, so its old plain `SELECT DISTINCT` had nothing to select).
    ///
    /// Idempotent by a different mechanism than this file's other migrations: those
    /// gate on `column_exists` and skip entirely on a second run, because an
    /// `ALTER TABLE ADD COLUMN` on an existing column errors. Here there is nothing
    /// that errors on repetition -- `CREATE TABLE IF NOT EXISTS` and `INSERT OR
    /// IGNORE` (keyed on `name`, the primary key) are each naturally idempotent -- so
    /// this always runs both steps rather than checking first. That also means a
    /// second run always re-seeds: harmless, since seeding never touches a row
    /// already present (`OR IGNORE` skips the conflicting insert rather than
    /// overwriting it), so any future hand-edit to this table -- an added or edited
    /// row -- survives every later open untouched, and no user data lives in this
    /// table for a reseed to ever endanger in the first place.
    ///
    /// A freshly created database already has the table from
    /// `create_tables_if_not_exist`'s own `CREATE TABLE IF NOT EXISTS`, so the first
    /// statement here is a no-op there too -- only the seeding step does real work on
    /// a fresh database, which is exactly the case this migration exists for.
    ///
    /// # Errors
    ///
    /// Returns an error if creating the table or inserting any of
    /// [`DEFAULT_SHAPES`]'s entries fails.
    pub(super) fn migrate_shape_vocabulary(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS shape_vocabulary (
                     name TEXT PRIMARY KEY,
                     sort_order INTEGER NOT NULL
                 );",
            )
            .context("Failed to create shape_vocabulary table")?;

        let mut stmt = self
            .conn
            .prepare("INSERT OR IGNORE INTO shape_vocabulary (name, sort_order) VALUES (?1, ?2)")
            .context("Failed to prepare shape_vocabulary seed insert")?;
        for (order, shape) in DEFAULT_SHAPES.iter().enumerate() {
            stmt.execute(params![shape, order as i64])
                .with_context(|| format!("Failed to seed shape_vocabulary entry '{shape}'"))?;
        }
        Ok(())
    }

    /// Whether `table` currently has a column named `column`, via `PRAGMA table_info`.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or running the `PRAGMA table_info` query fails.
    pub(super) fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let name: String = row.get("name")?;
            if name == column {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Retypes `diagram_details.{column}` from TEXT to `sql_type` (`"REAL"` or
/// `"INTEGER"`) in place, via SQLite's standard "add the replacement, populate it,
/// drop the original, rename the replacement into its place" sequence -- SQLite has no
/// `ALTER COLUMN ... TYPE`. Non-numeric-looking values become `NULL` rather than
/// `CAST`'s silent `0` (SQLite's `CAST(x AS REAL)` on text that doesn't look like a
/// number returns `0.0`, which would otherwise quietly fabricate data for the
/// handful of NULL/empty rows this project has).
///
/// Split out of `Database::migrate_numeric_columns` purely so the same four-line
/// sequence isn't repeated for each of the four columns it retypes.
///
/// # Errors
///
/// Returns an error if adding the replacement column, populating it, or
/// dropping/renaming the original fails.
fn retype_text_column_to_numeric(tx: &Transaction<'_>, column: &str, sql_type: &str) -> Result<()> {
    let staging = format!("{column}__migrated");
    tx.execute_batch(&format!(
        "ALTER TABLE diagram_details ADD COLUMN {staging} {sql_type};"
    ))
    .with_context(|| format!("Failed to add staging column for '{column}'"))?;

    tx.execute(
        &format!(
            "UPDATE diagram_details
             SET {staging} = CASE
                 WHEN {column} IS NULL OR TRIM({column}) = '' THEN NULL
                 ELSE CAST({column} AS {sql_type})
             END"
        ),
        [],
    )
    .with_context(|| format!("Failed to populate staging column for '{column}'"))?;

    tx.execute_batch(&format!(
        "ALTER TABLE diagram_details DROP COLUMN {column};
         ALTER TABLE diagram_details RENAME COLUMN {staging} TO {column};"
    ))
    .with_context(|| format!("Failed to swap staging column into place for '{column}'"))?;

    Ok(())
}

/// Populates the new `facets`/`girdle_facets` INTEGER columns from the existing
/// `facets_count` TEXT column (e.g. `"55+6"` -> `facets = 55, girdle_facets = 6`),
/// via [`parse_facets_count`]. `facets_count` itself is left untouched -- it stays the
/// display value (see `migrate_numeric_columns`'s doc comment for why).
///
/// Done row-by-row in Rust rather than with SQL string functions: the real data has
/// more shapes than the common `"N+M"` pattern (a `-` separator, a non-numeric girdle
/// suffix like `"R"`/`"FC"`, a doubled separator), and `parse_facets_count` is the
/// single, unit-tested place that already knows how to handle all of them.
///
/// # Errors
///
/// Returns an error if reading `(id, facets_count)` rows or writing any
/// `facets`/`girdle_facets` update fails.
fn split_facets_count_column(tx: &Transaction<'_>) -> Result<()> {
    let rows: Vec<(i64, Option<String>)> = {
        let mut select_stmt = tx
            .prepare("SELECT id, facets_count FROM diagram_details")
            .context("Failed to prepare facets_count read for splitting")?;
        select_stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .context("Failed to run facets_count read for splitting")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("Failed to decode a row while reading facets_count for splitting")?
    };

    let mut update_stmt = tx
        .prepare("UPDATE diagram_details SET facets = ?1, girdle_facets = ?2 WHERE id = ?3")
        .context("Failed to prepare facets/girdle_facets update")?;
    for (id, raw) in rows {
        let (facets, girdle_facets) = parse_facets_count(raw.as_deref());
        update_stmt
            .execute(params![facets, girdle_facets, id])
            .with_context(|| format!("Failed to write facets/girdle_facets for id {id}"))?;
    }
    Ok(())
}

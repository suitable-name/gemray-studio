use anyhow::{Context, Result};
use rusqlite::Connection;
use tracing::{debug, info};

mod entries;
mod materials;
mod migrations;
mod mirror_state;
mod search;
#[cfg(test)]
mod tests;

pub use materials::CustomMaterialParams;
pub use search::SEARCH_RESULT_CAP;

// The following imports have no production use *in this file* -- they exist solely so
// that `tests`' `use super::*;` (see `tests.rs`, moved unaltered) resolves. Every name
// below is used by production code in one of this module's submodules already; this is
// an extra, test-only binding of the same items into this module's own namespace, gated
// out of non-test builds so it can never trigger an unused-import warning there.
#[cfg(test)]
use crate::model::{
    detail::FacetDiagramDetail, entry::FacetDiagramEntry, filter::RangeFilter,
    metadata_update::MetadataUpdate,
};
#[cfg(test)]
use rusqlite::params;
#[cfg(test)]
use search::percentile_of_sorted;

/// The default database file [`Database::new`] opens when given `None`.
///
/// A path resolved relative to the process's current working directory. `pub` so a
/// caller that wants to report or reuse this default (e.g. `gemray-worker`'s
/// `serve --db`, which falls back to this exact path) doesn't have to duplicate the
/// literal.
pub const DEFAULT_DB_FILE: &str = "facet_diagrams.sqlite";

/// The `source_id` every row synced before the `source_id` column existed is
/// backfilled with -- see `Database::migrate_source_id_column`.
///
/// Every design predating the `source_id` column has exactly one possible origin, so
/// backfilling them all to this value is a historical fact about that data, not a
/// guess.
///
/// Deliberately a plain string literal: `db` is the lower layer and must not depend on
/// whatever produces any particular `source_id`. `pub` so that a crate which CAN see
/// both this constant and the identifier it has to match is able to assert the two
/// stay equal -- that assertion cannot live here, because from here only one side of
/// it is visible.
pub const LEGACY_SOURCE_ID: &str = "facetdiagrams.org";

/// The canonical faceting-design shape vocabulary, most-common first and freeform
/// last (deliberate order -- e.g. a GUI shape picker can present it as-is without
/// re-sorting).
///
/// This is what seeds the `shape_vocabulary` table on every [`Database::new`] (see
/// `Database::migrate_shape_vocabulary`), and it's `pub` so another crate building an
/// import/assignment flow (e.g. `apps/diagram-gui`) can offer the same list as a
/// picker without a round trip through the database -- there is exactly one
/// definition of this list, here.
///
/// This is a *starting* vocabulary, not an exhaustive one: the real catalogue
/// contains free-text scraped shape strings this list doesn't cover (see
/// `Database::get_unique_shapes`, which unions this list with whatever actually
/// appears in `diagram_details.shape` so neither source drops the other's values).
pub const DEFAULT_SHAPES: &[&str] = &[
    "Round",
    "Oval",
    "Cushion",
    "Square",
    "Rectangle",
    "Emerald",
    "Pear",
    "Marquise",
    "Heart",
    "Triangle",
    "Trillion",
    "Hexagon",
    "Octagon",
    "Pentagon",
    "Kite",
    "Rhombus",
    "Shield",
    "Star",
    "Barion",
    "Briolette",
    "Freeform",
];

pub struct Database {
    conn: Connection,
}

impl Database {
    /// Creates a new Database instance, connecting to the SQLite database file.
    /// Creates the file and tables if they don't exist.
    ///
    /// # Arguments
    /// * `db_path` - Optional path to the database file. Defaults to "`facet_diagrams.sqlite`".
    ///
    /// # Errors
    ///
    /// Returns an error if the SQLite file at `db_path` cannot be opened (e.g. bad
    /// path, permissions, or a file that isn't a valid SQLite database), if enabling
    /// the `foreign_keys` pragma fails, or if creating the schema (tables/indexes)
    /// fails.
    pub fn new(db_path: Option<&str>) -> Result<Self> {
        let path = db_path.unwrap_or(DEFAULT_DB_FILE);
        info!("Connecting to database: {}", path);
        let conn = Connection::open(path).context(format!("Failed to open database at {path}"))?;

        // Enable foreign key constraints. Crucial for data integrity.
        conn.execute("PRAGMA foreign_keys = ON;", [])
            .context("Failed to enable foreign keys")?;

        let db = Self { conn };
        db.create_tables_if_not_exist()?;
        db.migrate_numeric_columns()?;
        db.migrate_source_id_column()?;
        db.migrate_proportions_columns()?;
        db.migrate_designer_and_attachment_columns()?;
        db.migrate_crystal_optics_columns()?;
        db.migrate_shape_vocabulary()?;
        Ok(db)
    }

    /// Opens `db_path` READ-ONLY at the SQLite connection level (`SQLITE_OPEN_READ_ONLY`,
    /// no `SQLITE_OPEN_CREATE`) -- for a caller that must never write to this database,
    /// ever, not even to create it if missing.
    ///
    /// Unlike [`Self::new`], this does NOT run schema creation or any migration: a
    /// read-only connection cannot perform the `CREATE TABLE`/`ALTER TABLE` statements
    /// those need (and, per this method's own contract, must not even try). It assumes
    /// `db_path` already has the schema [`Self::new`] would have brought it to --
    /// appropriate for pointing this at an existing, already-populated catalogue (e.g.
    /// a long-running server reading the user's own library), never for provisioning a
    /// fresh one.
    ///
    /// # Errors
    ///
    /// Returns an error if `db_path` doesn't exist, isn't a valid SQLite database, or
    /// can't be opened read-only for any other reason (permissions, a bad path), or if
    /// enabling the `foreign_keys` pragma fails.
    pub fn open_read_only(db_path: &str) -> Result<Self> {
        info!("Connecting to database (read-only): {}", db_path);
        let conn = Connection::open_with_flags(
            db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context(format!("Failed to open database read-only at {db_path}"))?;

        conn.execute("PRAGMA foreign_keys = ON;", [])
            .context("Failed to enable foreign keys")?;

        Ok(Self { conn })
    }

    fn create_tables_if_not_exist(&self) -> Result<()> {
        debug!("Ensuring database tables exist...");
        self.conn
            .execute_batch(
                "BEGIN;

            CREATE TABLE IF NOT EXISTS diagram_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                design_id TEXT,
                source_id TEXT NOT NULL DEFAULT 'facetdiagrams.org'
            );

            CREATE TABLE IF NOT EXISTS diagram_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL UNIQUE, -- Each entry should have only one detail record
                page_url TEXT NOT NULL,
                diagram_image_name TEXT,
                diagram_image_data BLOB,
                competition_diagram TEXT,
                lw_ratio TEXT,
                refractive_index TEXT,
                index_gear TEXT,
                volume TEXT,
                facets_count TEXT,
                shape TEXT,
                designer_info TEXT,
                hw_ratio REAL,
                tw_ratio REAL,
                uw_ratio REAL,
                pw_ratio REAL,
                cw_ratio REAL,
                symmetry_order INTEGER,
                mirror_symmetry BOOLEAN,
                designer TEXT,
                source_citation TEXT,
                pdf_file TEXT,
                gem_file TEXT,
                shape_category INTEGER,
                FOREIGN KEY (entry_id) REFERENCES diagram_entries (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS angle_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_idx INTEGER NOT NULL, 
                detail_id INTEGER NOT NULL,
                facet TEXT NOT NULL,
                angle TEXT NOT NULL,
                index_val TEXT NOT NULL, -- 'index' is a reserved keyword in SQL
                notes TEXT NOT NULL,
                FOREIGN KEY (detail_id) REFERENCES diagram_details (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attached_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detail_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                content BLOB NOT NULL,
                FOREIGN KEY (detail_id) REFERENCES diagram_details (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS custom_gem_materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                refractive_index REAL NOT NULL,
                dispersion REAL NOT NULL,
                birefringence REAL NOT NULL,
                absorption_r REAL NOT NULL,
                absorption_g REAL NOT NULL,
                absorption_b REAL NOT NULL,
                crystal_system TEXT,
                optical_character TEXT,
                biaxial_delta_beta_alpha REAL
            );

            -- Pull-mirror sync (see `crate::model::mirror`): the last remote
            -- content hashes seen for a design mirrored from a remote library server,
            -- keyed by the same `url` that already governs `diagram_entries`' own
            -- cross-sync identity. Purely additive bookkeeping -- an install with no
            -- remote configured never gains a row here, and this table's absence of
            -- data never changes any query against `diagram_entries`/`diagram_details`.
            CREATE TABLE IF NOT EXISTS library_mirror_state (
                url TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                summary_version BLOB NOT NULL,
                design_version BLOB NOT NULL
            );

            -- Canonical shape vocabulary (see `DEFAULT_SHAPES`), seeded by
            -- `migrate_shape_vocabulary` -- created here too (`IF NOT EXISTS`) so a
            -- fresh database already has the table before that migration runs, the
            -- same convention every other table on this list follows. A plain lookup
            -- list, not a FK target for `diagram_details.shape`: the real catalogue
            -- holds free-text scraped shape strings no fixed vocabulary covers, and a
            -- FK constraint would either reject them or force a lossy migration of
            -- real data (see `Database::get_unique_shapes`).
            CREATE TABLE IF NOT EXISTS shape_vocabulary (
                name TEXT PRIMARY KEY,
                sort_order INTEGER NOT NULL
            );

            COMMIT;",
            )
            .context("Failed to create database tables")?;
        info!("Database tables checked/created successfully.");
        Ok(())
    }
}

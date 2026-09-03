use super::Database;
use crate::model::{
    dedup::{CrossSourceDuplicate, normalize_for_dedup},
    detail::FacetDiagramDetail,
    entry::FacetDiagramEntry,
    facets::parse_facets_count,
    metadata_update::MetadataUpdate,
};
use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use tracing::{debug, info};

/// One `(id, source_id, title, designer_info)` row from
/// [`Database::find_cross_source_duplicates`]'s narrowed candidate query, before
/// title/designer normalisation is applied -- named here purely so the query's
/// `Vec<...>` type doesn't trip `clippy::type_complexity`.
type DuplicateCandidateRow = (i64, String, String, Option<String>);

impl Database {
    /// Saves a diagram entry as having come from `source_id` (see
    /// `crate::source::DiagramSource::id`).
    /// If an entry with the same URL already exists, it updates the title, `design_id`,
    /// and `source_id`.
    /// Returns the ID of the inserted or updated entry.
    ///
    /// This only dedupes *within* `url` -- two different sources describing the same
    /// physical design under different URLs both get their own row here. See
    /// [`Self::find_cross_source_duplicates`] for the separate check that surfaces
    /// (without merging) that kind of cross-source collision.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `INSERT`, `UPDATE`, or (when updating) the
    /// follow-up `SELECT` of the existing row's ID fails.
    pub fn save_diagram_entry(&self, entry: &FacetDiagramEntry, source_id: &str) -> Result<i64> {
        // Try to insert. If the URL is unique, this will succeed.
        // `INSERT OR IGNORE` will not update, so we handle update explicitly.
        let mut stmt_insert = self.conn.prepare_cached(
            "INSERT OR IGNORE INTO diagram_entries (title, url, design_id, source_id) VALUES (?1, ?2, ?3, ?4)",
        )?;
        let changes = stmt_insert
            .execute(params![entry.title, entry.url, entry.design_id, source_id])
            .context(format!(
                "Failed to INSERT OR IGNORE diagram entry with URL: {}",
                entry.url
            ))?;

        if changes > 0 {
            // A new row was inserted
            let id = self.conn.last_insert_rowid();
            debug!(
                "Inserted new diagram entry '{}' (URL: {}, source: {}) with ID: {}",
                entry.title, entry.url, source_id, id
            );
            Ok(id)
        } else {
            // No new row was inserted, meaning an entry with this URL already exists.
            // Update the existing entry's title, design_id, and source_id.
            debug!(
                "Diagram entry with URL '{}' already exists. Updating title, design_id, and source_id.",
                entry.url
            );
            let mut stmt_update = self.conn.prepare_cached(
                "UPDATE diagram_entries SET title = ?1, design_id = ?2, source_id = ?3 WHERE url = ?4",
            )?;
            stmt_update
                .execute(params![entry.title, entry.design_id, source_id, entry.url])
                .context(format!(
                    "Failed to UPDATE existing diagram entry with URL: {}",
                    entry.url
                ))?;

            // Retrieve the ID of the existing (now updated) entry.
            let mut stmt_select = self
                .conn
                .prepare_cached("SELECT id FROM diagram_entries WHERE url = ?1")?;
            let id: i64 = stmt_select
                .query_row(params![entry.url], |row| row.get(0))
                .context(format!(
                    "Failed to SELECT ID of existing diagram entry with URL: {}",
                    entry.url
                ))?;
            debug!(
                "Updated existing diagram entry '{}' (URL: {}), existing ID: {}",
                entry.title, entry.url, id
            );
            Ok(id)
        }
    }

    /// Looks for entries already in the catalogue, synced from a source *other than*
    /// `new_source_id`, whose normalised title (and, when both sides have one,
    /// normalised designer) matches `title`/`designer_info`, and whose facet count
    /// matches `facets` when both are known. See `crate::model::dedup`'s module doc
    /// comment for why this only detects and surfaces candidates -- it never merges
    /// or alters anything.
    ///
    /// A missing designer on either side is not treated as a mismatch: without a
    /// designer to compare, there isn't enough information to *rule out* a
    /// collision, and a false positive here just means a human reviews one extra
    /// candidate, while a false negative means a real duplicate silently goes
    /// unflagged. When both sides do have a designer, they must match -- this is what
    /// keeps two different designers' genuinely same-named designs from being flagged
    /// against each other.
    ///
    /// Narrowed first in SQL by `source_id != new_source_id` (and by facet count, when
    /// known) to keep the candidate set the normalisation/comparison loop below has to
    /// examine small -- the catalogue is a few thousand rows, so a linear scan of a
    /// filtered slice is plenty fast for something that runs once per newly-synced
    /// entry, not once per search.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying query fails.
    pub fn find_cross_source_duplicates(
        &self,
        new_source_id: &str,
        title: &str,
        designer_info: Option<&str>,
        facets: Option<i64>,
    ) -> Result<Vec<CrossSourceDuplicate>> {
        let normalized_title = normalize_for_dedup(title);
        if normalized_title.is_empty() {
            return Ok(Vec::new());
        }
        let normalized_designer = designer_info.map(normalize_for_dedup);

        let mut sql = String::from(
            "SELECT de.id, de.source_id, de.title, dd.designer_info
             FROM diagram_entries de
             LEFT JOIN diagram_details dd ON de.id = dd.entry_id
             WHERE de.source_id != ?1",
        );
        let mut sql_params: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(new_source_id.to_string())];
        if let Some(f) = facets {
            sql.push_str(" AND dd.facets = ?2");
            sql_params.push(Box::new(f));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let bound: Vec<&dyn rusqlite::ToSql> =
            sql_params.iter().map(std::convert::AsRef::as_ref).collect();
        let rows: Vec<DuplicateCandidateRow> = stmt
            .query_map(bound.as_slice(), |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut matches = Vec::new();
        for (existing_entry_id, existing_source_id, existing_title, existing_designer_info) in rows
        {
            if normalize_for_dedup(&existing_title) != normalized_title {
                continue;
            }
            if let (Some(want), Some(have)) =
                (&normalized_designer, existing_designer_info.as_deref())
                && normalize_for_dedup(have) != *want
            {
                continue;
            }
            matches.push(CrossSourceDuplicate {
                existing_entry_id,
                existing_source_id,
                existing_title,
                existing_designer_info,
            });
        }
        Ok(matches)
    }

    /// Saves the details of a facet diagram.
    /// This will first delete any existing details, angle settings, and attached files
    /// associated with the `entry_id` to ensure data is fresh and prevent duplicates.
    ///
    /// # Performance: one transaction per design, not one per row
    ///
    /// The lookup, delete, detail insert, and every angle-setting/attached-file insert
    /// below all run inside a single [`Connection::unchecked_transaction`] (same
    /// pattern [`Self::migrate_numeric_columns`] already uses for `&self`-taking
    /// methods -- `Transaction::commit`/its `Drop` rollback are what actually bound
    /// the unit of work, "unchecked" only means the *type system* doesn't stop a
    /// second nested transaction, which nothing here attempts). Without this, every
    /// individual `execute` -- one per angle row, one per attached file, easily 50+
    /// for a single competition design -- was its own implicit autocommit
    /// transaction, each with its own fsync; wrapping the whole design in one
    /// transaction turns that into a single fsync on commit.
    ///
    /// Measured on a fresh temp database loaded with this database's real
    /// per-design angle/attachment row-count distribution (3027 designs, 44259
    /// angle rows, 6428 attachments): the pre-fix code (no explicit transaction,
    /// one `execute` per row) took **625.99s**; this transaction-per-design
    /// version (still one `execute` per row in [`Self::save_angle_settings`]/
    /// [`Self::save_attached_files`] -- see their doc comment for why that stays a
    /// plain loop rather than a batched multi-row insert) took **26.17s**, a
    /// **23.9x** speedup. The transaction is the entire fix; there is no
    /// second-order batching win layered on top of it here (measured, not assumed
    /// -- see the two functions above for what was tried and why it lost).
    ///
    /// On any error partway through (the delete, the detail insert, or any child
    /// row), the transaction is never committed and rolls back automatically on
    /// drop -- a failed save can never leave a design's old data deleted but its new
    /// data only half-written, nor a new design with some but not all of its angle
    /// rows or attachments.
    ///
    /// # Errors
    ///
    /// Returns an error if starting/committing the transaction fails, or if looking
    /// up or deleting an existing detail row fails, or if inserting the new detail
    /// row, its angle settings, or its attached files fails -- in every case the
    /// transaction rolls back and no partial data is left behind.
    pub fn save_diagram_detail(&self, detail: &FacetDiagramDetail, entry_id: i64) -> Result<()> {
        let tx = self.conn.unchecked_transaction().context(format!(
            "Failed to start save transaction for entry_id: {entry_id}"
        ))?;

        // Check if detail for this entry_id already exists.
        // If so, delete it. ON DELETE CASCADE will handle child records in
        // angle_settings and attached_files.
        let existing_detail_id: Option<i64> = tx
            .query_row(
                "SELECT id FROM diagram_details WHERE entry_id = ?1",
                params![entry_id],
                |row| row.get(0),
            )
            .optional()
            .context(format!(
                "Failed to check for existing diagram detail for entry_id: {entry_id}"
            ))?;

        if let Some(old_detail_id) = existing_detail_id {
            debug!(
                "Deleting existing detail (ID: {}) and its associated data for entry_id: {}",
                old_detail_id, entry_id
            );
            tx.execute(
                "DELETE FROM diagram_details WHERE id = ?1",
                params![old_detail_id],
            )
            .context(format!(
                "Failed to delete old diagram detail (ID: {old_detail_id})"
            ))?;
        }

        // Insert the new detail record. `facets`/`girdle_facets` are derived from
        // `facets_count` at write time (via the same `parse_facets_count` the schema
        // migration uses) so every newly-saved design is immediately range-filterable
        // by facet count, not just the ones already in the database when the numeric
        // columns were retyped.
        let (facets, girdle_facets) = parse_facets_count(detail.facets_count.as_deref());
        let mut stmt_detail = tx.prepare_cached(
            "INSERT INTO diagram_details (
                entry_id, page_url, diagram_image_name, diagram_image_data,
                competition_diagram, lw_ratio, refractive_index, index_gear,
                volume, facets_count, facets, girdle_facets, shape, designer_info,
                hw_ratio, tw_ratio, uw_ratio, pw_ratio, cw_ratio, symmetry_order, mirror_symmetry,
                designer, source_citation, pdf_file, gem_file, shape_category
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
                      ?22, ?23, ?24, ?25, ?26)",
        )?;

        stmt_detail
            .execute(params![
                entry_id,
                detail.page_url,
                detail.diagram_image_name,
                detail.diagram_image_data,
                detail.competition_diagram,
                detail.lw_ratio,
                detail.refractive_index,
                detail.index_gear,
                detail.volume,
                detail.facets_count,
                facets,
                girdle_facets,
                detail.shape,
                detail.designer_info,
                detail.hw_ratio,
                detail.tw_ratio,
                detail.uw_ratio,
                detail.pw_ratio,
                detail.cw_ratio,
                detail.symmetry_order,
                detail.mirror_symmetry,
                detail.designer,
                detail.source_citation,
                detail.pdf_file,
                detail.gem_file,
                detail.shape_category,
            ])
            .context(format!(
                "Failed to insert diagram detail for entry_id: {entry_id}"
            ))?;
        // Statement must be dropped before `tx.last_insert_rowid()` below can borrow
        // `tx` again -- `prepare_cached` borrows it for the statement's lifetime.
        drop(stmt_detail);

        let detail_id = tx.last_insert_rowid();
        debug!(
            "Inserted diagram detail for entry_id {} with new detail_id: {}",
            entry_id, detail_id
        );

        Self::save_angle_settings(&tx, detail_id, &detail.angle_settings_table)?;
        Self::save_attached_files(&tx, detail_id, &detail.attached_files)?;

        tx.commit().context(format!(
            "Failed to commit save transaction for entry_id: {entry_id}"
        ))?;

        info!(
            "Successfully saved diagram detail and associated data for entry_id: {}",
            entry_id
        );
        Ok(())
    }

    /// Inserts every angle-setting row for `detail_id`. Split out of
    /// `save_diagram_detail` purely to keep that function under clippy's
    /// function-length lint; takes `conn: &Connection` rather than `&self` so it
    /// can run against an in-progress `Transaction` (which derefs to `Connection`),
    /// not just a bare connection -- see [`Self::save_diagram_detail`]'s doc
    /// comment for why it's called with one.
    ///
    /// One `execute` per row against a `prepare_cached` statement, deliberately
    /// *not* batched into a multi-row `INSERT ... VALUES (...), (...), ...`: tried
    /// exactly that first, measured it against this database's real per-design
    /// angle-row-count distribution (3027 designs, 44259 rows, 64 distinct counts),
    /// and it was *slower* than this simple loop -- 40-42s vs 26.17s, both already
    /// inside [`Self::save_diagram_detail`]'s transaction. Root cause: SQLite is an
    /// in-process embedded engine, not a client/server database -- there is no
    /// network round trip for batching to amortize away, so a per-row `execute`
    /// against an already-cached, fixed-shape prepared statement is already about
    /// as cheap as this gets; a batched insert's SQL text (and so
    /// `prepare_cached`'s cache key) instead varies with the chunk's row count, so
    /// across a real corpus with dozens of distinct row counts it either thrashes
    /// the statement cache (its default capacity is 16) or, even with that raised,
    /// still pays for building a new placeholder string and a fresh
    /// `Vec<&dyn ToSql>` on every design for no offsetting win. The transaction
    /// wrap is where the real, order-of-magnitude gain already measured comes
    /// from; multi-row batching is a client/server-database optimization that
    /// doesn't transfer to SQLite's execution model, so it stays out rather than
    /// staying in as unverified "best practice" that this crate's own benchmark
    /// contradicts.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing the insert statement fails, or if inserting
    /// any individual row fails.
    fn save_angle_settings(
        conn: &Connection,
        detail_id: i64,
        angle_settings: &[crate::model::angle::AngleSetting],
    ) -> Result<()> {
        let mut stmt_angle = conn.prepare_cached(
            "INSERT INTO angle_settings (detail_id, order_idx, facet, angle, index_val, notes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for setting in angle_settings {
            stmt_angle
                .execute(params![
                    detail_id,
                    setting.order_index,
                    setting.facet,
                    setting.angle,
                    setting.index,
                    setting.notes,
                ])
                .context(format!(
                    "Failed to insert angle setting for detail_id: {detail_id}"
                ))?;
        }
        debug!(
            "Inserted {} angle settings for detail_id: {}",
            angle_settings.len(),
            detail_id
        );
        Ok(())
    }

    /// Inserts every attached-file row for `detail_id`. Same policy as
    /// [`Self::save_angle_settings`] -- see that function's doc comment for why
    /// this is a plain per-row loop rather than a batched multi-row insert, and why
    /// `conn: &Connection` rather than `&self`. Split out of `save_diagram_detail`
    /// purely to keep that function under clippy's function-length lint.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing the insert statement fails, or if inserting
    /// any individual row fails.
    fn save_attached_files(
        conn: &Connection,
        detail_id: i64,
        files: &[crate::model::file::AttachedFile],
    ) -> Result<()> {
        let mut stmt_file = conn.prepare_cached(
            "INSERT INTO attached_files (detail_id, name, url, content)
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        for file in files {
            stmt_file
                .execute(params![
                    detail_id,
                    file.name,
                    file.url,
                    file.content, // This is Vec<u8>, will be stored as BLOB
                ])
                .context(format!(
                    "Failed to insert attached file '{}' for detail_id: {}",
                    file.name, detail_id
                ))?;
        }
        debug!(
            "Inserted {} attached files for detail_id: {}",
            files.len(),
            detail_id
        );
        Ok(())
    }

    /// Checks if details for a given diagram entry URL already exist in the database.
    /// Can be used to optionally skip re-fetching/processing if data is already present.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `COUNT` query fails.
    pub fn has_detail_for_entry_url(&self, entry_url: &str) -> Result<bool> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(dd.id)
             FROM diagram_details dd
             JOIN diagram_entries de ON dd.entry_id = de.id
             WHERE de.url = ?1",
                params![entry_url],
                |row| row.get(0),
            )
            .context(format!(
                "Failed to check if detail exists for entry URL: {entry_url}"
            ))?;
        Ok(count > 0)
    }

    /// Loads the full record for one diagram entry -- its entry/detail row plus all
    /// associated angle settings and attached files -- or `None` if `entry_id`
    /// doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if preparing or running any of the underlying queries
    /// (entry/detail, angle settings, attached files) fails, or if a row fails to
    /// decode into its target type.
    pub fn get_diagram_full(
        &self,
        entry_id: i64,
    ) -> Result<Option<crate::model::entry::FullDiagramRecord>> {
        // lw_ratio/refractive_index/index_gear/volume are stored as REAL/INTEGER (see
        // `migrate_numeric_columns`), and hw_ratio/tw_ratio/uw_ratio/pw_ratio/cw_ratio
        // are REAL, symmetry_order/shape_category INTEGER (see
        // `create_tables_if_not_exist`) -- all cast back to TEXT here so
        // `FullDiagramRecord`'s fields, which mirror `FacetDiagramDetail`'s `Option<
        // String>` convention for every one of these, stay `Option<String>` rather
        // than leaking the storage type. `mirror_symmetry` (BOOLEAN) and the plain-TEXT
        // designer/source_citation/pdf_file/gem_file columns need no cast.
        let mut stmt = self.conn.prepare(
            "SELECT de.id, de.title, de.url, de.design_id,
                    dd.id, dd.page_url, dd.diagram_image_name, dd.diagram_image_data,
                    dd.competition_diagram, CAST(dd.lw_ratio AS TEXT), CAST(dd.refractive_index AS TEXT),
                    CAST(dd.index_gear AS TEXT), CAST(dd.volume AS TEXT), dd.facets_count, dd.shape, dd.designer_info,
                    CAST(dd.hw_ratio AS TEXT), CAST(dd.tw_ratio AS TEXT), CAST(dd.uw_ratio AS TEXT),
                    CAST(dd.pw_ratio AS TEXT), CAST(dd.cw_ratio AS TEXT), CAST(dd.symmetry_order AS TEXT),
                    dd.mirror_symmetry, dd.designer, dd.source_citation, dd.pdf_file, dd.gem_file,
                    CAST(dd.shape_category AS TEXT)
             FROM diagram_entries de
             LEFT JOIN diagram_details dd ON de.id = dd.entry_id
             WHERE de.id = ?1",
        )?;

        let mut rows = stmt.query(params![entry_id])?;
        if let Some(row) = rows.next()? {
            let detail_id_opt: Option<i64> = row.get(4)?;

            let mut angles = Vec::new();
            let mut files = Vec::new();

            if let Some(detail_id) = detail_id_opt {
                let mut stmt_angles = self.conn.prepare(
                    "SELECT facet, angle, index_val, notes, order_idx
                     FROM angle_settings
                     WHERE detail_id = ?1
                     ORDER BY order_idx ASC",
                )?;
                let a_rows = stmt_angles.query_map(params![detail_id], |arow| {
                    Ok(crate::model::angle::AngleSetting {
                        facet: arow.get(0)?,
                        angle: arow.get(1)?,
                        index: arow.get(2)?,
                        notes: arow.get(3)?,
                        order_index: arow.get(4)?,
                    })
                })?;
                for a in a_rows.flatten() {
                    angles.push(a);
                }

                let mut stmt_files = self.conn.prepare(
                    "SELECT name, url, content
                     FROM attached_files
                     WHERE detail_id = ?1",
                )?;
                let f_rows = stmt_files.query_map(params![detail_id], |frow| {
                    Ok(crate::model::file::AttachedFile {
                        name: frow.get(0)?,
                        url: frow.get(1)?,
                        content: frow.get(2)?,
                    })
                })?;
                for f in f_rows.flatten() {
                    files.push(f);
                }
            }

            return Ok(Some(crate::model::entry::FullDiagramRecord {
                entry_id: row.get(0)?,
                title: row.get(1)?,
                url: row.get(2)?,
                design_id: row.get(3)?,
                page_url: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                diagram_image_name: row.get(6)?,
                diagram_image_data: row.get(7)?,
                competition_diagram: row.get(8)?,
                lw_ratio: row.get(9)?,
                refractive_index: row.get(10)?,
                index_gear: row.get(11)?,
                volume: row.get(12)?,
                facets_count: row.get(13)?,
                shape: row.get(14)?,
                designer_info: row.get(15)?,
                hw_ratio: row.get(16)?,
                tw_ratio: row.get(17)?,
                uw_ratio: row.get(18)?,
                pw_ratio: row.get(19)?,
                cw_ratio: row.get(20)?,
                symmetry_order: row.get(21)?,
                mirror_symmetry: row.get(22)?,
                designer: row.get(23)?,
                source_citation: row.get(24)?,
                pdf_file: row.get(25)?,
                gem_file: row.get(26)?,
                shape_category: row.get(27)?,
                angle_settings: angles,
                attached_files: files,
            }));
        }

        Ok(None)
    }

    /// Loads the same record [`Self::get_diagram_full`] would, except every attached
    /// file comes back as [`crate::model::entry::AttachedFileMeta`] (id/name/url/size)
    /// instead of [`crate::model::file::AttachedFile`] -- its `content` column is never
    /// selected, so this never loads an attachment's bytes into memory at all, not even
    /// to discard them afterward.
    ///
    /// For a remote-serving caller (e.g. `gemray-worker`'s library protocol) fetching
    /// one design's full record: attachments can be individually large, and several
    /// designs often reference the SAME attachment (a shared competition-results PDF),
    /// so paying to load every one's content on every design fetch -- what
    /// [`Self::get_diagram_full`] does, and rightly so for its own caller, the LOCAL
    /// library UI, which genuinely wants the bytes immediately -- would be wasteful for
    /// a caller that only wants attachment bytes lazily, one at a time, by id (see
    /// [`Self::get_attachment_content`]).
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Self::get_diagram_full`].
    pub fn get_diagram_full_meta(
        &self,
        entry_id: i64,
    ) -> Result<Option<crate::model::entry::FullDiagramMeta>> {
        let mut stmt = self.conn.prepare(
            "SELECT de.id, de.title, de.url, de.design_id,
                    dd.id, dd.page_url, dd.diagram_image_name, dd.diagram_image_data,
                    dd.competition_diagram, CAST(dd.lw_ratio AS TEXT), CAST(dd.refractive_index AS TEXT),
                    CAST(dd.index_gear AS TEXT), CAST(dd.volume AS TEXT), dd.facets_count, dd.shape, dd.designer_info
             FROM diagram_entries de
             LEFT JOIN diagram_details dd ON de.id = dd.entry_id
             WHERE de.id = ?1",
        )?;

        let mut rows = stmt.query(params![entry_id])?;
        if let Some(row) = rows.next()? {
            let detail_id_opt: Option<i64> = row.get(4)?;

            let mut angles = Vec::new();
            let mut files = Vec::new();

            if let Some(detail_id) = detail_id_opt {
                let mut stmt_angles = self.conn.prepare(
                    "SELECT facet, angle, index_val, notes, order_idx
                     FROM angle_settings
                     WHERE detail_id = ?1
                     ORDER BY order_idx ASC",
                )?;
                let a_rows = stmt_angles.query_map(params![detail_id], |arow| {
                    Ok(crate::model::angle::AngleSetting {
                        facet: arow.get(0)?,
                        angle: arow.get(1)?,
                        index: arow.get(2)?,
                        notes: arow.get(3)?,
                        order_index: arow.get(4)?,
                    })
                })?;
                for a in a_rows.flatten() {
                    angles.push(a);
                }

                // `length(content)` -- never `content` itself -- is what keeps this a
                // metadata-only query; see this method's own doc comment.
                let mut stmt_files = self.conn.prepare(
                    "SELECT id, name, url, length(content)
                     FROM attached_files
                     WHERE detail_id = ?1",
                )?;
                let f_rows = stmt_files.query_map(params![detail_id], |frow| {
                    Ok(crate::model::entry::AttachedFileMeta {
                        id: frow.get(0)?,
                        name: frow.get(1)?,
                        url: frow.get(2)?,
                        size: frow.get(3)?,
                    })
                })?;
                for f in f_rows.flatten() {
                    files.push(f);
                }
            }

            return Ok(Some(crate::model::entry::FullDiagramMeta {
                entry_id: row.get(0)?,
                title: row.get(1)?,
                url: row.get(2)?,
                design_id: row.get(3)?,
                page_url: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                diagram_image_name: row.get(6)?,
                diagram_image_data: row.get(7)?,
                competition_diagram: row.get(8)?,
                lw_ratio: row.get(9)?,
                refractive_index: row.get(10)?,
                index_gear: row.get(11)?,
                volume: row.get(12)?,
                facets_count: row.get(13)?,
                shape: row.get(14)?,
                designer_info: row.get(15)?,
                angle_settings: angles,
                attached_files: files,
            }));
        }

        Ok(None)
    }

    /// Loads exactly one attachment's name and content by id -- never a whole design's
    /// attachment set (contrast [`Self::get_diagram_full`]'s `attached_files`, and see
    /// [`Self::get_diagram_full_meta`]'s doc comment for why that split exists). Bounds
    /// a caller's per-request memory use to one attachment's size, not a design's worth
    /// of them.
    ///
    /// `None` if `attachment_id` doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `SELECT` fails.
    pub fn get_attachment_content(&self, attachment_id: i64) -> Result<Option<(String, Vec<u8>)>> {
        self.conn
            .query_row(
                "SELECT name, content FROM attached_files WHERE id = ?1",
                params![attachment_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .context(format!(
                "Failed to load attachment content for attachment_id: {attachment_id}"
            ))
    }

    /// Renames a diagram entry -- the "Organize" library operation for the user's own
    /// designs (works on any entry regardless of `source_id`, not just local imports).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `UPDATE` fails, or if `entry_id` does not
    /// match any row (zero rows affected).
    pub fn rename_diagram_entry(&self, entry_id: i64, new_title: &str) -> Result<()> {
        let trimmed = new_title.trim();
        if trimmed.is_empty() {
            return Err(anyhow::anyhow!("Title cannot be empty."));
        }
        let changed = self
            .conn
            .execute(
                "UPDATE diagram_entries SET title = ?1 WHERE id = ?2",
                params![trimmed, entry_id],
            )
            .context(format!("Failed to rename diagram entry {entry_id}"))?;
        if changed == 0 {
            return Err(anyhow::anyhow!("No diagram entry with id {entry_id}."));
        }
        Ok(())
    }

    /// Updates exactly the metadata fields a user might legitimately hand-correct on an
    /// already-imported design -- see [`MetadataUpdate`]'s own doc comment for exactly
    /// which fields that is and why title isn't one of them.
    ///
    /// # The trap this exists to avoid
    ///
    /// `Database::get_diagram_full` returns a [`crate::model::entry::FullDiagramRecord`],
    /// which is a STRICT SUBSET of [`FacetDiagramDetail`] -- it has no `hw_ratio`/
    /// `tw_ratio`/`uw_ratio`/`pw_ratio`/`cw_ratio`/`symmetry_order`/`mirror_symmetry`/
    /// `designer`/`source_citation`/`pdf_file`/`gem_file`/`shape_category` fields at
    /// all. [`Self::save_diagram_detail`] fully REPLACES a design's entire detail row
    /// (delete, then reinsert, plus reinserting every angle-setting/attached-file
    /// child row), so a naive "read a `FullDiagramRecord`, edit one field, build a
    /// fresh `FacetDiagramDetail` from it, save" would silently zero every field above
    /// on every metadata edit -- for a locally-imported design that means erasing the
    /// very proportions its own import step just measured (see `apps/diagram-gui`'s
    /// `gui::library::apply_measured_metadata`).
    ///
    /// This method is the real fix: one `UPDATE` naming exactly `MetadataUpdate`'s
    /// fields (plus `facets`/`girdle_facets`, kept in sync with `facets_count` below --
    /// see that field's own note) and nothing else. Every other `diagram_details`
    /// column, and `angle_settings`/`attached_files` in their entirety, are never
    /// touched by this statement -- there is no delete, so a child row's own id (and
    /// an attachment's bytes) survive a metadata edit completely unchanged.
    ///
    /// `facets`/`girdle_facets`: these INTEGER columns exist purely as a queryable
    /// split of `facets_count` (see [`parse_facets_count`] and
    /// `Database::search_diagrams`'s range filter, which reads `facets`/`girdle_facets`
    /// directly, never `facets_count` text) -- re-deriving them here from the SAME
    /// `facets_count` string this call is already writing is a deterministic parse of
    /// data already in hand, not the geometry recomputation the user's own instruction
    /// (store everything, discard nothing, never recalculate) is about. Leaving them
    /// stale after a facet-count edit would silently desync the range-filter slider
    /// from the text the detail view now shows.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `UPDATE` fails, or if `entry_id` has no
    /// `diagram_details` row (zero rows affected -- e.g. an entry whose import never
    /// got as far as writing one).
    pub fn update_diagram_metadata(&self, entry_id: i64, update: &MetadataUpdate) -> Result<()> {
        let (facets, girdle_facets) = parse_facets_count(update.facets_count.as_deref());
        let changed = self
            .conn
            .execute(
                "UPDATE diagram_details SET
                    designer_info = ?1, shape = ?2, refractive_index = ?3, index_gear = ?4,
                    facets_count = ?5, facets = ?6, girdle_facets = ?7, symmetry_order = ?8,
                    mirror_symmetry = ?9, lw_ratio = ?10, hw_ratio = ?11, cw_ratio = ?12,
                    pw_ratio = ?13, volume = ?14
                 WHERE entry_id = ?15",
                params![
                    update.designer_info,
                    update.shape,
                    update.refractive_index,
                    update.index_gear,
                    update.facets_count,
                    facets,
                    girdle_facets,
                    update.symmetry_order,
                    update.mirror_symmetry,
                    update.lw_ratio,
                    update.hw_ratio,
                    update.cw_ratio,
                    update.pw_ratio,
                    update.volume,
                    entry_id,
                ],
            )
            .context(format!(
                "Failed to update diagram metadata for entry_id: {entry_id}"
            ))?;
        if changed == 0 {
            return Err(anyhow::anyhow!(
                "No diagram detail row for entry_id {entry_id}."
            ));
        }
        Ok(())
    }

    /// Permanently deletes a diagram entry and everything attached to it (detail,
    /// angle settings, attached files -- all cascade via `ON DELETE CASCADE`, see
    /// `create_tables_if_not_exist`). The "Organize" library operation's delete --
    /// works on any entry regardless of `source_id`.
    ///
    /// Deletes stored data the user already has; it does not police or interpret what
    /// that data is (see this crate's doc comment on the public/private boundary being
    /// about code, not data).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `DELETE` fails, or if `entry_id` does not
    /// match any row (zero rows affected).
    pub fn delete_diagram_entry(&self, entry_id: i64) -> Result<()> {
        let changed = self
            .conn
            .execute(
                "DELETE FROM diagram_entries WHERE id = ?1",
                params![entry_id],
            )
            .context(format!("Failed to delete diagram entry {entry_id}"))?;
        if changed == 0 {
            return Err(anyhow::anyhow!("No diagram entry with id {entry_id}."));
        }
        Ok(())
    }
}

use super::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Returns a fresh path under the OS temp dir for a throwaway SQLite file --
/// tests need a real file (not `:memory:`) whenever they reopen the database to
/// check idempotency or persistence across two `Database::new` calls, since each
/// `:memory:` connection is its own private, empty database.
fn temp_db_path(label: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "diagram_catalog_test_{label}_{n}_{}.sqlite",
        std::process::id()
    ))
}

/// One `(title, url, ri, lw, volume, gear, facets_count)` row for
/// [`seed_pre_migration_db`].
type SeedRow<'a> = (
    &'a str,
    &'a str,
    Option<&'a str>,
    Option<&'a str>,
    Option<&'a str>,
    Option<&'a str>,
    Option<&'a str>,
);

/// One `(title, refractive_index, lw_ratio, volume, index_gear, facets, girdle_facets)`
/// row from `migration_retypes_columns_and_splits_facets_count`'s post-migration
/// read-back query -- named here purely so that query's `Vec<...>` type doesn't trip
/// `clippy::type_complexity`.
type MigratedDetailRow = (
    String,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<i64>,
    Option<i64>,
    Option<i64>,
);

/// Creates the *pre-migration* schema directly (bypassing `Database::new`, which
/// always migrates on open) and inserts one `diagram_entries`/`diagram_details`
/// pair per `SeedRow`, exactly as the old TEXT-column schema would have held them.
/// Used to test the migration against data shaped like what's actually in
/// `facet_diagrams.sqlite` today.
fn seed_pre_migration_db(path: &std::path::Path, rows: &[SeedRow<'_>]) {
    let conn = Connection::open(path).expect("open raw connection for seeding");
    conn.execute_batch(
        "CREATE TABLE diagram_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                design_id TEXT
            );
            CREATE TABLE diagram_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL UNIQUE,
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
                FOREIGN KEY (entry_id) REFERENCES diagram_entries (id) ON DELETE CASCADE
            );
            CREATE TABLE angle_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_idx INTEGER NOT NULL,
                detail_id INTEGER NOT NULL,
                facet TEXT NOT NULL,
                angle TEXT NOT NULL,
                index_val TEXT NOT NULL,
                notes TEXT NOT NULL,
                FOREIGN KEY (detail_id) REFERENCES diagram_details (id) ON DELETE CASCADE
            );
            CREATE TABLE attached_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detail_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                content BLOB NOT NULL,
                FOREIGN KEY (detail_id) REFERENCES diagram_details (id) ON DELETE CASCADE
            );
            CREATE TABLE custom_gem_materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                refractive_index REAL NOT NULL,
                dispersion REAL NOT NULL,
                birefringence REAL NOT NULL,
                absorption_r REAL NOT NULL,
                absorption_g REAL NOT NULL,
                absorption_b REAL NOT NULL
            );",
    )
    .expect("create pre-migration schema");

    for (title, url, ri, lw, volume, gear, facets_count) in rows {
        conn.execute(
            "INSERT INTO diagram_entries (title, url, design_id) VALUES (?1, ?2, NULL)",
            params![title, url],
        )
        .expect("insert entry");
        let entry_id = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO diagram_details (
                    entry_id, page_url, competition_diagram, lw_ratio, refractive_index,
                    index_gear, volume, facets_count, shape, designer_info
                 ) VALUES (?1, '', NULL, ?2, ?3, ?4, ?5, ?6, NULL, NULL)",
            params![entry_id, lw, ri, gear, volume, facets_count],
        )
        .expect("insert detail");
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "a single migration-correctness test walking several seeded rows \
                  through the migration and asserting each one; splitting it would \
                  separate the setup from the assertions it's checking"
)]
fn migration_retypes_columns_and_splits_facets_count() {
    let path = temp_db_path("retype");
    seed_pre_migration_db(
        &path,
        &[
            (
                "A",
                "url-a",
                Some("1.540"),
                Some("1.009"),
                Some("0.171"),
                Some("96"),
                Some("55+6"),
            ),
            (
                "B",
                "url-b",
                Some("2.16"),
                Some("1.001"),
                Some("0.257"),
                Some("96"),
                Some("57"),
            ),
            ("C", "url-c", None, None, None, None, None),
            (
                "D",
                "url-d",
                Some(""),
                Some(""),
                Some(""),
                Some(""),
                Some(""),
            ),
            (
                "E",
                "url-e",
                Some("1.76"),
                Some("1.63"),
                Some("0.412"),
                Some("84"),
                Some("78-7"),
            ),
        ],
    );

    let db = Database::new(Some(path.to_str().unwrap())).expect("open + migrate");

    // Schema: the four columns are now numeric, facets_count survives untouched,
    // and the two new split-out columns exist.
    assert!(Database::column_exists(&db.conn, "diagram_details", "facets").unwrap());
    assert!(Database::column_exists(&db.conn, "diagram_details", "girdle_facets").unwrap());
    assert!(Database::column_exists(&db.conn, "diagram_details", "facets_count").unwrap());

    let mut type_stmt = db
        .conn
        .prepare("PRAGMA table_info(diagram_details)")
        .unwrap();
    let types: std::collections::HashMap<String, String> = type_stmt
        .query_map([], |r| {
            Ok((r.get::<_, String>("name")?, r.get::<_, String>("type")?))
        })
        .unwrap()
        .flatten()
        .collect();
    assert_eq!(types["refractive_index"], "REAL");
    assert_eq!(types["lw_ratio"], "REAL");
    assert_eq!(types["volume"], "REAL");
    assert_eq!(types["index_gear"], "INTEGER");
    assert_eq!(types["facets_count"], "TEXT");
    assert_eq!(types["facets"], "INTEGER");
    assert_eq!(types["girdle_facets"], "INTEGER");

    // Row count preserved.
    let count: i64 = db
        .conn
        .query_row("SELECT COUNT(*) FROM diagram_details", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 5);

    // Values round-trip correctly, including the "55+6" / "57" / "78-7" splits.
    let mut stmt = db
        .conn
        .prepare(
            "SELECT de.title, dd.refractive_index, dd.lw_ratio, dd.volume, dd.index_gear,
                        dd.facets, dd.girdle_facets
                 FROM diagram_details dd JOIN diagram_entries de ON de.id = dd.entry_id
                 ORDER BY de.title",
        )
        .unwrap();
    let got: Vec<MigratedDetailRow> = stmt
        .query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
                r.get(6)?,
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();

    assert_eq!(
        got[0],
        (
            "A".to_string(),
            Some(1.540),
            Some(1.009),
            Some(0.171),
            Some(96),
            Some(55),
            Some(6)
        )
    );
    assert_eq!(
        got[1],
        (
            "B".to_string(),
            Some(2.16),
            Some(1.001),
            Some(0.257),
            Some(96),
            Some(57),
            Some(0)
        )
    );
    // NULL stays NULL through the retype, and facets_count = NULL splits to (None, None).
    assert_eq!(
        got[2],
        ("C".to_string(), None, None, None, None, None, None)
    );
    // empty string also becomes NULL (not 0.0/0), matching "handle the handful of NULL/empty rows".
    assert_eq!(
        got[3],
        ("D".to_string(), None, None, None, None, None, None)
    );
    assert_eq!(
        got[4],
        (
            "E".to_string(),
            Some(1.76),
            Some(1.63),
            Some(0.412),
            Some(84),
            Some(78),
            Some(7)
        )
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn migration_is_idempotent_across_two_opens() {
    let path = temp_db_path("idempotent");
    seed_pre_migration_db(
        &path,
        &[
            (
                "A",
                "url-a",
                Some("1.540"),
                Some("1.009"),
                Some("0.171"),
                Some("96"),
                Some("55+6"),
            ),
            (
                "B",
                "url-b",
                Some("2.16"),
                Some("1.001"),
                Some("0.257"),
                Some("96"),
                Some("57"),
            ),
        ],
    );

    {
        let db = Database::new(Some(path.to_str().unwrap())).expect("first open migrates");
        let count: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM diagram_details", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    // Second open must be a no-op migration-wise: same schema, same row count,
    // same values -- not a second retype attempt (which would error trying to add
    // a column that already exists) and not data loss.
    let db2 = Database::new(Some(path.to_str().unwrap())).expect("second open is idempotent");
    let count: i64 = db2
        .conn
        .query_row("SELECT COUNT(*) FROM diagram_details", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2);

    let ri: f64 = db2
            .conn
            .query_row(
                "SELECT refractive_index FROM diagram_details dd JOIN diagram_entries de ON de.id = dd.entry_id WHERE de.title = 'A'",
                [],
                |r| r.get(0),
            )
            .unwrap();
    assert!((ri - 1.540).abs() < 1e-9);

    let mut type_stmt = db2
        .conn
        .prepare("PRAGMA table_info(diagram_details)")
        .unwrap();
    let column_count = type_stmt
        .query_map([], |r| r.get::<_, String>("name"))
        .unwrap()
        .flatten()
        .count();
    // Exactly the original 13 columns, plus facets/girdle_facets (+2 = 15,
    // from `migrate_numeric_columns`), plus the seven proportion-ratio/
    // symmetry columns (+7 = 22, from `migrate_proportions_columns`), plus the
    // designer-split and competition-entry columns (+5 = 27, from
    // `migrate_designer_and_attachment_columns`) -- no duplicated `__migrated`
    // staging columns left behind by a re-run of any of them.
    assert_eq!(column_count, 27);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn source_id_migration_backfills_legacy_rows_and_is_idempotent_across_two_opens() {
    let path = temp_db_path("source_id_migration");
    // `seed_pre_migration_db`'s schema predates `source_id` entirely (see its own
    // doc comment) -- exactly the shape a real pre-multi-source database has.
    seed_pre_migration_db(
        &path,
        &[
            (
                "A",
                "url-a",
                Some("1.540"),
                Some("1.009"),
                Some("0.171"),
                Some("96"),
                Some("55+6"),
            ),
            (
                "B",
                "url-b",
                Some("2.16"),
                Some("1.001"),
                Some("0.257"),
                Some("96"),
                Some("57"),
            ),
        ],
    );

    {
        let db = Database::new(Some(path.to_str().unwrap())).expect("first open migrates");
        assert!(Database::column_exists(&db.conn, "diagram_entries", "source_id").unwrap());
        let ids: Vec<String> = {
            let mut stmt = db
                .conn
                .prepare("SELECT source_id FROM diagram_entries ORDER BY title")
                .unwrap();
            stmt.query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .flatten()
                .collect()
        };
        // Every pre-existing row is backfilled with the legacy source -- it could
        // only ever have come from facetdiagrams.org.
        assert_eq!(
            ids,
            vec![LEGACY_SOURCE_ID.to_string(), LEGACY_SOURCE_ID.to_string()]
        );
    }

    // Second open must be a no-op: no error from re-adding an existing column, and
    // the backfilled values survive untouched.
    let db2 = Database::new(Some(path.to_str().unwrap())).expect("second open is idempotent");
    let count: i64 = db2
        .conn
        .query_row(
            "SELECT COUNT(*) FROM diagram_entries WHERE source_id = ?1",
            params![LEGACY_SOURCE_ID],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 2);

    let mut type_stmt = db2
        .conn
        .prepare("PRAGMA table_info(diagram_entries)")
        .unwrap();
    let column_count = type_stmt
        .query_map([], |r| r.get::<_, String>("name"))
        .unwrap()
        .flatten()
        .count();
    // title, url, design_id, id, source_id -- exactly 5, no leftover duplicate.
    assert_eq!(column_count, 5);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_fresh_database_already_has_the_source_id_column() {
    let path = temp_db_path("fresh_source_id");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create fresh db");
    assert!(Database::column_exists(&db.conn, "diagram_entries", "source_id").unwrap());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn designer_and_attachment_migration_adds_the_columns_and_is_idempotent() {
    let path = temp_db_path("designer_split_migration");
    // `seed_pre_migration_db`'s schema predates all five of these columns (see
    // its own doc comment) -- exactly the shape a real catalogue has today.
    seed_pre_migration_db(
        &path,
        &[
            (
                "A",
                "url-a",
                Some("1.540"),
                Some("1.009"),
                Some("0.171"),
                Some("96"),
                Some("55+6"),
            ),
            (
                "B",
                "url-b",
                Some("2.16"),
                Some("1.001"),
                Some("0.257"),
                Some("96"),
                Some("57"),
            ),
        ],
    );

    {
        let db = Database::new(Some(path.to_str().unwrap())).expect("first open migrates");
        for column in [
            "designer",
            "source_citation",
            "pdf_file",
            "gem_file",
            "shape_category",
        ] {
            assert!(
                Database::column_exists(&db.conn, "diagram_details", column).unwrap(),
                "migration must add {column}"
            );
        }
        // `designer_info` is deliberately kept rather than dropped -- callers in
        // `search_diagrams`, `find_cross_source_duplicates` and `apps/diagram-gui`
        // still read it (see `FacetDiagramDetail::designer`).
        assert!(Database::column_exists(&db.conn, "diagram_details", "designer_info").unwrap());
        // The index is what turns "every design by X" into a lookup rather than a
        // `LIKE` scan; without it the split would buy nothing.
        let indexed: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'index' AND name = 'idx_diagram_details_designer'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(indexed, 1, "designer index must exist after migrating");
    }

    // Second open must be a no-op: no error from re-adding an existing column or
    // index, and the pre-existing rows survive untouched.
    let db2 = Database::new(Some(path.to_str().unwrap())).expect("second open is idempotent");
    let count: i64 = db2
        .conn
        .query_row("SELECT COUNT(*) FROM diagram_details", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn a_fresh_database_already_has_the_designer_and_attachment_columns() {
    let path = temp_db_path("fresh_designer_split");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create fresh db");
    for column in [
        "designer",
        "source_citation",
        "pdf_file",
        "gem_file",
        "shape_category",
    ] {
        assert!(
            Database::column_exists(&db.conn, "diagram_details", column).unwrap(),
            "a fresh database's CREATE TABLE must already include {column}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// The five new fields must survive a `save_diagram_detail` round trip into their
/// typed columns -- `shape_category` in particular is an `Option<String>` on
/// `FacetDiagramDetail` bound straight into an INTEGER column (the same shape
/// `symmetry_order` already has), so this pins down that it lands as a number and
/// not as text.
#[test]
fn save_diagram_detail_persists_the_designer_split_and_competition_fields() {
    let path = temp_db_path("designer_split_roundtrip");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");
    let entry = FacetDiagramEntry {
        title: "Utopia".to_string(),
        url: "https://facetdiagrams.org/diagramus/utopia/".to_string(),
        design_id: String::new(),
    };
    let entry_id = db
        .save_diagram_entry(&entry, LEGACY_SOURCE_ID)
        .expect("save entry");
    let detail = FacetDiagramDetail {
        designer_info: Some("Capps, Jerry; Lapidary Journal, May 1994, p95".to_string()),
        designer: Some("Capps, Jerry".to_string()),
        source_citation: Some("Lapidary Journal, May 1994, p95".to_string()),
        pdf_file: Some("2002SSCMasters.pdf".to_string()),
        gem_file: None,
        shape_category: Some("5".to_string()),
        ..Default::default()
    };
    db.save_diagram_detail(&detail, entry_id)
        .expect("save detail");

    let (designer, citation, pdf, gem, category) = db
        .conn
        .query_row(
            "SELECT designer, source_citation, pdf_file, gem_file, shape_category
                 FROM diagram_details WHERE entry_id = ?1",
            params![entry_id],
            |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, Option<i64>>(4)?,
                ))
            },
        )
        .expect("read back detail");

    assert_eq!(designer.as_deref(), Some("Capps, Jerry"));
    assert_eq!(citation.as_deref(), Some("Lapidary Journal, May 1994, p95"));
    assert_eq!(pdf.as_deref(), Some("2002SSCMasters.pdf"));
    assert_eq!(gem, None);
    // Read back as an integer, not the "5" string that went in -- INTEGER
    // affinity converted it on the way into the column.
    assert_eq!(category, Some(5));

    let _ = std::fs::remove_file(&path);
}

/// Every `diagram_details` column [`MetadataUpdate`] does NOT cover, read straight off
/// the row -- exactly the set that must survive a metadata edit byte-for-byte, INCLUDING
/// the fields `FullDiagramRecord` cannot even see (`tw_ratio`/`uw_ratio`/`designer`/
/// `source_citation`/`pdf_file`/`gem_file`/`shape_category`), which is exactly the trap
/// `update_diagram_metadata` exists to route around -- see its own doc comment. Named
/// here purely so the query's tuple type doesn't trip `clippy::type_complexity`.
type UntouchedDetailColumnsRow = (
    String,          // page_url
    Option<String>,  // diagram_image_name
    Option<Vec<u8>>, // diagram_image_data
    Option<String>,  // competition_diagram
    Option<f64>,     // tw_ratio
    Option<f64>,     // uw_ratio
    Option<String>,  // designer
    Option<String>,  // source_citation
    Option<String>,  // pdf_file
    Option<String>,  // gem_file
    Option<i64>,     // shape_category
);

/// One `(hw_ratio, cw_ratio, pw_ratio, symmetry_order, mirror_symmetry)` spot-check row
/// from `update_diagram_metadata_touches_only_its_own_fields_and_nothing_else` -- named
/// here purely so that query's tuple type doesn't trip `clippy::type_complexity`.
type RatioAndSymmetrySpotCheckRow = (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<i64>,
    Option<bool>,
);

fn read_untouched_detail_columns(db: &Database, entry_id: i64) -> UntouchedDetailColumnsRow {
    db.conn
        .query_row(
            "SELECT page_url, diagram_image_name, diagram_image_data, competition_diagram,
                    tw_ratio, uw_ratio, designer, source_citation, pdf_file, gem_file, shape_category
             FROM diagram_details WHERE entry_id = ?1",
            params![entry_id],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                    r.get(7)?,
                    r.get(8)?,
                    r.get(9)?,
                    r.get(10)?,
                ))
            },
        )
        .expect("read untouched diagram_details columns")
}

/// The trap this whole method exists to avoid, pinned down directly: a fully-populated
/// detail row (every field `FacetDiagramDetail` has, including the ones
/// `FullDiagramRecord` cannot see at all -- see that struct's own doc comment) gets one
/// narrow `update_diagram_metadata` call that changes exactly two fields (`shape`,
/// `refractive_index`) and resubmits every other `MetadataUpdate` field at its current
/// value, exactly the shape a pre-filled editor form would submit. Every column outside
/// `MetadataUpdate` -- and `angle_settings`/`attached_files` entirely -- must come back
/// byte-for-byte identical; a regression back to a `save_diagram_detail`-style delete
/// and reinsert would either zero those columns or change the child rows' own ids, both
/// of which this test would catch.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one round trip through a fully-populated detail row, a narrow edit, and \
                  every 'must still equal what it started as' assertion; splitting it \
                  would separate the setup from the assertions it's checking"
)]
fn update_diagram_metadata_touches_only_its_own_fields_and_nothing_else() {
    let path = temp_db_path("metadata_update_narrow");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");
    let entry = FacetDiagramEntry {
        title: "Utopia".to_string(),
        url: "https://facetdiagrams.org/diagramus/utopia-narrow/".to_string(),
        design_id: String::new(),
    };
    let entry_id = db
        .save_diagram_entry(&entry, LEGACY_SOURCE_ID)
        .expect("save entry");

    let original = FacetDiagramDetail {
        page_url: "https://facetdiagrams.org/diagramus/utopia-narrow/".to_string(),
        diagram_image_name: Some("utopia.svg".to_string()),
        diagram_image_data: Some(vec![1, 2, 3, 4]),
        angle_settings_table: vec![crate::model::angle::AngleSetting {
            order_index: 0,
            facet: "T".to_string(),
            angle: "0".to_string(),
            index: "-".to_string(),
            notes: String::new(),
        }],
        attached_files: vec![crate::model::file::AttachedFile {
            name: "utopia.asc".to_string(),
            url: String::new(),
            content: b"original bytes, must survive untouched".to_vec(),
        }],
        competition_diagram: Some("2002SSCMasters".to_string()),
        lw_ratio: Some("1.05".to_string()),
        refractive_index: Some("2.417".to_string()),
        index_gear: Some("96".to_string()),
        volume: Some("0.42".to_string()),
        facets_count: Some("57+8".to_string()),
        shape: Some("Round".to_string()),
        designer_info: Some("Capps, Jerry; Lapidary Journal, May 1994, p95".to_string()),
        hw_ratio: Some("0.61".to_string()),
        tw_ratio: Some("0.55".to_string()),
        uw_ratio: Some("0.12".to_string()),
        pw_ratio: Some("0.44".to_string()),
        cw_ratio: Some("0.17".to_string()),
        symmetry_order: Some("8".to_string()),
        mirror_symmetry: Some(true),
        designer: Some("Capps, Jerry".to_string()),
        source_citation: Some("Lapidary Journal, May 1994, p95".to_string()),
        pdf_file: Some("2002SSCMasters.pdf".to_string()),
        gem_file: Some("utopia.gem".to_string()),
        shape_category: Some("5".to_string()),
    };
    db.save_diagram_detail(&original, entry_id)
        .expect("save original detail");

    let before = read_untouched_detail_columns(&db, entry_id);
    let detail_id: i64 = db
        .conn
        .query_row(
            "SELECT id FROM diagram_details WHERE entry_id = ?1",
            params![entry_id],
            |r| r.get(0),
        )
        .unwrap();
    let angle_count_before: i64 = db
        .conn
        .query_row(
            "SELECT COUNT(*) FROM angle_settings WHERE detail_id = ?1",
            params![detail_id],
            |r| r.get(0),
        )
        .unwrap();
    let (attachment_id_before, attachment_content_before): (i64, Vec<u8>) = db
        .conn
        .query_row(
            "SELECT id, content FROM attached_files WHERE detail_id = ?1",
            params![detail_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();

    let update = MetadataUpdate {
        designer_info: original.designer_info.clone(),
        shape: Some("Oval".to_string()),            // the actual edit
        refractive_index: Some("1.76".to_string()), // the actual edit
        index_gear: original.index_gear.clone(),
        facets_count: original.facets_count.clone(),
        symmetry_order: original.symmetry_order.clone(),
        mirror_symmetry: original.mirror_symmetry,
        lw_ratio: original.lw_ratio.clone(),
        hw_ratio: original.hw_ratio.clone(),
        cw_ratio: original.cw_ratio.clone(),
        pw_ratio: original.pw_ratio.clone(),
        volume: original.volume.clone(),
    };
    db.update_diagram_metadata(entry_id, &update)
        .expect("update metadata");

    // The two edited fields actually changed.
    let full = db.get_diagram_full(entry_id).unwrap().unwrap();
    assert_eq!(full.shape.as_deref(), Some("Oval"));
    assert_eq!(full.refractive_index.as_deref(), Some("1.76"));

    // Every `MetadataUpdate` field that was resubmitted unchanged must still read back
    // unchanged.
    assert_eq!(full.designer_info, original.designer_info);
    assert_eq!(full.index_gear, original.index_gear);
    assert_eq!(full.facets_count, original.facets_count);
    assert_eq!(full.lw_ratio, original.lw_ratio);
    let (hw, cw, pw, sym, mirror): RatioAndSymmetrySpotCheckRow = db
        .conn
        .query_row(
            "SELECT hw_ratio, cw_ratio, pw_ratio, symmetry_order, mirror_symmetry
             FROM diagram_details WHERE entry_id = ?1",
            params![entry_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    assert!((hw.unwrap() - 0.61).abs() < 1e-9);
    assert!((cw.unwrap() - 0.17).abs() < 1e-9);
    assert!((pw.unwrap() - 0.44).abs() < 1e-9);
    assert_eq!(sym, Some(8));
    assert_eq!(mirror, Some(true));

    // Every column OUTSIDE `MetadataUpdate` -- including every field
    // `FullDiagramRecord` cannot even see -- must be byte-for-byte identical to
    // before this call. This is the assertion that actually catches the trap.
    assert_eq!(
        read_untouched_detail_columns(&db, entry_id),
        before,
        "update_diagram_metadata must not touch any diagram_details column outside MetadataUpdate"
    );

    // Never a delete-and-reinsert of the detail row's children: same row count, same
    // attachment id, same bytes.
    let angle_count_after: i64 = db
        .conn
        .query_row(
            "SELECT COUNT(*) FROM angle_settings WHERE detail_id = ?1",
            params![detail_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(angle_count_after, angle_count_before);
    let (attachment_id_after, attachment_content_after): (i64, Vec<u8>) = db
        .conn
        .query_row(
            "SELECT id, content FROM attached_files WHERE detail_id = ?1",
            params![detail_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        attachment_id_after, attachment_id_before,
        "attached_files row must not be deleted and reinserted (its id would change)"
    );
    assert_eq!(attachment_content_after, attachment_content_before);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn update_diagram_metadata_rejects_unknown_entry_id() {
    let path = temp_db_path("metadata_update_unknown");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");
    let result = db.update_diagram_metadata(999_999, &MetadataUpdate::default());
    assert!(result.is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_fresh_database_already_has_the_crystal_optics_columns() {
    let path = temp_db_path("fresh_crystal_optics");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create fresh db");
    for column in [
        "crystal_system",
        "optical_character",
        "biaxial_delta_beta_alpha",
    ] {
        assert!(
            Database::column_exists(&db.conn, "custom_gem_materials", column).unwrap(),
            "a fresh database's CREATE TABLE must already include {column}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// Seeds a `custom_gem_materials` row via the *pre-crystal-optics* schema directly
/// (bypassing `Database::new`, which always migrates on open) -- exactly the shape
/// a custom material saved before crystal classification existed has today: no
/// `crystal_system`/`optical_character`/`biaxial_delta_beta_alpha` columns at all.
fn seed_pre_crystal_optics_custom_material(path: &std::path::Path) {
    let conn = Connection::open(path).expect("open raw connection for seeding");
    conn.execute_batch(
            "CREATE TABLE custom_gem_materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                refractive_index REAL NOT NULL,
                dispersion REAL NOT NULL,
                birefringence REAL NOT NULL,
                absorption_r REAL NOT NULL,
                absorption_g REAL NOT NULL,
                absorption_b REAL NOT NULL
            );
            INSERT INTO custom_gem_materials
                (name, refractive_index, dispersion, birefringence, absorption_r, absorption_g, absorption_b)
                VALUES ('Legacy Custom Sapphire', 1.768, 0.018, -0.008, 2.8, 1.2, 0.1);",
        )
        .expect("create pre-crystal-optics custom_gem_materials and seed a row");
}

#[test]
fn crystal_optics_migration_adds_the_columns_and_leaves_existing_rows_nullable() {
    let path = temp_db_path("crystal_optics_migration");
    seed_pre_crystal_optics_custom_material(&path);

    {
        let db = Database::new(Some(path.to_str().unwrap())).expect("first open migrates");
        for column in [
            "crystal_system",
            "optical_character",
            "biaxial_delta_beta_alpha",
        ] {
            assert!(
                Database::column_exists(&db.conn, "custom_gem_materials", column).unwrap(),
                "migration must add {column}"
            );
        }

        // The pre-existing row must survive, with all three new fields absent
        // (`None`) rather than defaulted to some guessed value -- this is exactly
        // the state `CustomMaterialRow`'s doc comments say means "fall back to
        // whatever `GemMaterial::new_custom` infers", which is what keeps this
        // material rendering bit-identically to before the migration.
        let materials = db.get_custom_materials().expect("read back materials");
        assert_eq!(materials.len(), 1);
        let m = &materials[0];
        assert_eq!(m.name, "Legacy Custom Sapphire");
        assert!((m.refractive_index - 1.768).abs() < 1e-6);
        assert_eq!(m.crystal_system, None);
        assert_eq!(m.optical_character, None);
        assert_eq!(m.biaxial_delta_beta_alpha, None);
    }

    // Second open must be a no-op: no error from re-adding an existing column, and
    // the pre-existing row survives untouched.
    let db2 = Database::new(Some(path.to_str().unwrap())).expect("second open is idempotent");
    let materials = db2
        .get_custom_materials()
        .expect("read back materials again");
    assert_eq!(materials.len(), 1);
    assert_eq!(materials[0].crystal_system, None);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn save_custom_material_round_trips_crystal_optics_fields() {
    let path = temp_db_path("crystal_optics_roundtrip");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");

    // A biaxial material: all three new fields set.
    db.save_custom_material(&CustomMaterialParams {
        name: "Custom Tanzanite",
        refractive_index: 1.691,
        dispersion: 0.030,
        birefringence: 0.0130,
        absorption_rgb: [1.8, 1.6, 0.2],
        crystal_system: Some("Orthorhombic"),
        optical_character: Some("BiaxialPositive"),
        biaxial_delta_beta_alpha: Some(0.0070),
    })
    .expect("save biaxial custom material");

    // A uniaxial material: crystal_system/optical_character set, biaxial delta
    // left `None` -- the shape every non-biaxial save takes.
    db.save_custom_material(&CustomMaterialParams {
        name: "Custom Sapphire",
        refractive_index: 1.768,
        dispersion: 0.018,
        birefringence: -0.008,
        absorption_rgb: [2.8, 1.2, 0.1],
        crystal_system: Some("Trigonal"),
        optical_character: Some("UniaxialNegative"),
        biaxial_delta_beta_alpha: None,
    })
    .expect("save uniaxial custom material");

    let materials = db.get_custom_materials().expect("read back materials");
    assert_eq!(materials.len(), 2);

    let tanzanite = materials
        .iter()
        .find(|m| m.name == "Custom Tanzanite")
        .expect("Custom Tanzanite present");
    assert_eq!(tanzanite.crystal_system.as_deref(), Some("Orthorhombic"));
    assert_eq!(
        tanzanite.optical_character.as_deref(),
        Some("BiaxialPositive")
    );
    assert!((tanzanite.biaxial_delta_beta_alpha.unwrap() - 0.0070).abs() < 1e-6);

    let sapphire = materials
        .iter()
        .find(|m| m.name == "Custom Sapphire")
        .expect("Custom Sapphire present");
    assert_eq!(sapphire.crystal_system.as_deref(), Some("Trigonal"));
    assert_eq!(
        sapphire.optical_character.as_deref(),
        Some("UniaxialNegative")
    );
    assert_eq!(sapphire.biaxial_delta_beta_alpha, None);

    // Re-saving over the same name (upsert) must update the crystal-optics
    // columns too, not just the original five.
    db.save_custom_material(&CustomMaterialParams {
        name: "Custom Sapphire",
        refractive_index: 1.768,
        dispersion: 0.018,
        birefringence: -0.008,
        absorption_rgb: [2.8, 1.2, 0.1],
        crystal_system: None,
        optical_character: None,
        biaxial_delta_beta_alpha: None,
    })
    .expect("re-save clears crystal-optics fields");
    let materials = db.get_custom_materials().expect("read back after re-save");
    let sapphire = materials
        .iter()
        .find(|m| m.name == "Custom Sapphire")
        .expect("Custom Sapphire still present");
    assert_eq!(sapphire.crystal_system, None);
    assert_eq!(sapphire.optical_character, None);

    let _ = std::fs::remove_file(&path);
}

// `LEGACY_SOURCE_ID` has no guard test here on purpose: what it must stay equal to
// is not visible from this crate, so the assertion belongs wherever both sides ARE
// visible. See that constant's own doc comment.

/// Builds a `Database` (fresh temp file, already-migrated schema) and inserts one
/// diagram per `(title, shape, ri, lw, volume, facets_count)` tuple via the public
/// `save_diagram_entry`/`save_diagram_detail` API -- the same path real writers
/// use -- for exercising `search_diagrams`/`get_attribute_ranges`.
fn seeded_db(rows: &[(&str, &str, &str, &str, &str, &str)]) -> Database {
    let path = temp_db_path("search");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");
    for (title, shape, ri, lw, volume, facets_count) in rows {
        let entry = FacetDiagramEntry {
            title: (*title).to_string(),
            url: format!("https://example.test/{title}"),
            design_id: String::new(),
        };
        let entry_id = db
            .save_diagram_entry(&entry, LEGACY_SOURCE_ID)
            .expect("save entry");
        let detail = FacetDiagramDetail {
            shape: Some((*shape).to_string()),
            refractive_index: Some((*ri).to_string()),
            lw_ratio: Some((*lw).to_string()),
            volume: Some((*volume).to_string()),
            facets_count: Some((*facets_count).to_string()),
            index_gear: Some("96".to_string()),
            ..Default::default()
        };
        db.save_diagram_detail(&detail, entry_id)
            .expect("save detail");
    }
    db
}

#[test]
fn range_query_returns_only_rows_within_known_bounds() {
    let db = seeded_db(&[
        ("Low", "Round", "1.50", "1.00", "0.10", "50"),
        ("Mid", "Round", "1.76", "1.10", "0.20", "60+8"),
        ("High", "Round", "2.40", "1.20", "0.30", "70"),
    ]);

    // RI in [1.6, 2.0] should match only "Mid" (1.76).
    let range = RangeFilter {
        ri_min: Some(1.6),
        ri_max: Some(2.0),
        ..Default::default()
    };
    let results = db.search_diagrams("", "All", "All", &range).unwrap();
    assert_eq!(
        results.iter().map(|r| r.title.as_str()).collect::<Vec<_>>(),
        vec!["Mid"]
    );

    // facets in [55, 65] should match only "Mid" (facets = 60).
    let range = RangeFilter {
        facets_min: Some(55),
        facets_max: Some(65),
        ..Default::default()
    };
    let results = db.search_diagrams("", "All", "All", &range).unwrap();
    assert_eq!(
        results.iter().map(|r| r.title.as_str()).collect::<Vec<_>>(),
        vec!["Mid"]
    );

    // No range filter at all returns everything, same as before this feature.
    let results = db
        .search_diagrams("", "All", "All", &RangeFilter::default())
        .unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn inverted_range_bounds_return_nothing_not_everything() {
    let db = seeded_db(&[
        ("Low", "Round", "1.50", "1.00", "0.10", "50"),
        ("Mid", "Round", "1.76", "1.10", "0.20", "60+8"),
        ("High", "Round", "2.40", "1.20", "0.30", "70"),
    ]);

    // min > max can never be satisfied -- must return an empty list, not silently
    // behave as if no filter were applied.
    let range = RangeFilter {
        ri_min: Some(2.0),
        ri_max: Some(1.0),
        ..Default::default()
    };
    let results = db.search_diagrams("", "All", "All", &range).unwrap();
    assert!(
        results.is_empty(),
        "inverted RI bounds must return no rows, got {results:?}"
    );

    let range = RangeFilter {
        facets_min: Some(90),
        facets_max: Some(10),
        ..Default::default()
    };
    let results = db.search_diagrams("", "All", "All", &range).unwrap();
    assert!(
        results.is_empty(),
        "inverted facets bounds must return no rows, got {results:?}"
    );
}

#[test]
fn search_diagrams_page_walks_the_whole_result_set_via_keyset_cursor() {
    let db = seeded_db(&[
        ("A", "Round", "1.50", "1.00", "0.10", "50"),
        ("B", "Round", "1.55", "1.00", "0.10", "50"),
        ("C", "Round", "1.60", "1.00", "0.10", "50"),
        ("D", "Round", "1.65", "1.00", "0.10", "50"),
        ("E", "Round", "1.70", "1.00", "0.10", "50"),
    ]);

    let mut collected = Vec::new();
    let mut after_id = None;
    loop {
        let page = db
            .search_diagrams_page("", "All", "All", &RangeFilter::default(), after_id, 2)
            .unwrap();
        if page.is_empty() {
            break;
        }
        let full_page = page.len() == 2;
        after_id = page.last().map(|r| r.id);
        collected.extend(page);
        if !full_page {
            break;
        }
    }

    assert_eq!(
        collected
            .iter()
            .map(|r| r.title.as_str())
            .collect::<Vec<_>>(),
        vec!["A", "B", "C", "D", "E"],
        "a multi-page walk must reach every row, in order, including ones past the \
             first page"
    );
    // Strictly increasing ids across the whole walk -- no page boundary skipped or
    // duplicated a row, exactly what a total ORDER BY over a unique, monotonic key
    // guarantees.
    for pair in collected.windows(2) {
        assert!(pair[0].id < pair[1].id);
    }

    // Must match the single-page call exactly (same filters, same order) --
    // pagination composes with, not changes, what search_diagrams already returns.
    let unpaged = db
        .search_diagrams("", "All", "All", &RangeFilter::default())
        .unwrap();
    assert_eq!(
        collected.iter().map(|r| r.id).collect::<Vec<_>>(),
        unpaged.iter().map(|r| r.id).collect::<Vec<_>>()
    );
}

#[test]
fn search_diagrams_delegates_to_the_first_unpaginated_page() {
    let db = seeded_db(&[
        ("A", "Round", "1.50", "1.00", "0.10", "50"),
        ("B", "Round", "1.55", "1.00", "0.10", "50"),
    ]);

    let via_search = db
        .search_diagrams("", "All", "All", &RangeFilter::default())
        .unwrap();
    let via_page = db
        .search_diagrams_page("", "All", "All", &RangeFilter::default(), None, 1000)
        .unwrap();
    assert_eq!(
        via_search.iter().map(|r| r.id).collect::<Vec<_>>(),
        via_page.iter().map(|r| r.id).collect::<Vec<_>>(),
        "search_diagrams must return exactly what search_diagrams_page(.., None, 1000) does"
    );
}

#[test]
fn get_attribute_ranges_uses_real_min_but_a_percentile_based_max() {
    let db = seeded_db(&[
        ("Low", "Round", "1.50", "1.00", "0.10", "50"),
        ("Mid", "Round", "1.76", "1.10", "0.20", "60+8"),
        ("High", "Round", "2.40", "1.20", "0.30", "70"),
    ]);

    let ranges = db.get_attribute_ranges().unwrap();
    // Minimums are still the real minimums -- only the upper bound changed.
    assert!((ranges.ri.0 - 1.50).abs() < 1e-9);
    assert!((ranges.lw_ratio.0 - 1.00).abs() < 1e-9);
    // p99 (linear-interpolation method) of 3 sorted points [a, b, c] sits at
    // index 1.98, i.e. 98% of the way from b to c.
    assert!((ranges.ri.1 - 0.98_f64.mul_add(2.40 - 1.76, 1.76)).abs() < 1e-6);
    assert!((ranges.lw_ratio.1 - 0.98_f64.mul_add(1.20 - 1.10, 1.10)).abs() < 1e-6);
    // The upper bound must not equal the raw maximum for a right-skewed sample --
    // this is the whole point of the fix.
    assert!(
        ranges.ri.1 < 2.40,
        "p99 bound must sit below the raw max, got {}",
        ranges.ri.1
    );
    assert!(
        ranges.lw_ratio.1 < 1.20,
        "p99 bound must sit below the raw max, got {}",
        ranges.lw_ratio.1
    );
}

#[test]
fn get_attribute_ranges_does_not_let_a_single_outlier_set_the_scale() {
    // Mirrors the shape of the real catalogue's volume column (see the report): a
    // tight cluster of "normal" values plus one physically-impossible outlier that
    // must not drag the slider's usable bound anywhere near it.
    // 200 normal rows keeps the single outlier under 1% of the sample -- matching
    // the real catalogue, where 1 outlier sits among 3,025 rows (0.03%) and even
    // the 18 rows above the physical-plausibility threshold are under 1%.
    let mut owned: Vec<(String, String, String, String, String, String)> = (0..200)
        .map(|i| {
            let vol = f64::from(i).mul_add(0.20 / 199.0, 0.10);
            (
                format!("Normal{i}"),
                "Round".to_string(),
                "1.50".to_string(),
                "1.00".to_string(),
                format!("{vol:.4}"),
                "50".to_string(),
            )
        })
        .collect();
    owned.push((
        "Outlier".to_string(),
        "Round".to_string(),
        "1.50".to_string(),
        "1.00".to_string(),
        "195.0".to_string(),
        "50".to_string(),
    ));
    let rows: Vec<(&str, &str, &str, &str, &str, &str)> = owned
        .iter()
        .map(|(a, b, c, d, e, f)| {
            (
                a.as_str(),
                b.as_str(),
                c.as_str(),
                d.as_str(),
                e.as_str(),
                f.as_str(),
            )
        })
        .collect();
    let db = seeded_db(&rows);

    let ranges = db.get_attribute_ranges().unwrap();
    assert!(
        (ranges.volume.0 - 0.10).abs() < 1e-6,
        "min should be the real min, got {}",
        ranges.volume.0
    );
    assert!(
        ranges.volume.1 < 1.0,
        "a single outlier of 195 must not set the slider's usable max bound, got {}",
        ranges.volume.1
    );

    // The outlier row must still be reachable by search when both volume sides are
    // left at their default (unfiltered) position -- being excluded from the
    // slider's *scale* must not mean being excluded from *results*.
    let results = db
        .search_diagrams("", "All", "All", &RangeFilter::default())
        .unwrap();
    assert!(
        results.iter().any(|r| r.title == "Outlier"),
        "outlier row must still be reachable via unfiltered search"
    );
}

#[test]
fn percentile_of_sorted_interpolates_linearly() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((percentile_of_sorted(&values, 0.0) - 1.0).abs() < 1e-9);
    assert!((percentile_of_sorted(&values, 50.0) - 3.0).abs() < 1e-9);
    assert!((percentile_of_sorted(&values, 100.0) - 5.0).abs() < 1e-9);
    // idx = 0.99 * 4 = 3.96 -> 96% of the way from values[3]=4.0 to values[4]=5.0.
    assert!((percentile_of_sorted(&values, 99.0) - 4.96).abs() < 1e-9);
}

#[test]
fn percentile_of_sorted_single_value_returns_that_value() {
    assert!((percentile_of_sorted(&[42.0], 99.0) - 42.0).abs() < 1e-9);
}

#[test]
fn get_unique_gears_still_returns_display_strings_after_retype() {
    let db = seeded_db(&[("A", "Round", "1.50", "1.00", "0.10", "50")]);
    let gears = db.get_unique_gears().unwrap();
    assert_eq!(gears, vec!["96".to_string()]);
}

#[test]
fn find_cross_source_duplicates_detects_same_design_from_two_sources() {
    let path = temp_db_path("dedup_same_design");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");

    // Same physical design, synced from facetdiagrams.org first...
    let entry_a = FacetDiagramEntry {
        title: "Barion Heart".to_string(),
        url: "https://facetdiagrams.org/a".to_string(),
        design_id: String::new(),
    };
    let id_a = db
        .save_diagram_entry(&entry_a, "facetdiagrams.org")
        .unwrap();
    db.save_diagram_detail(
        &FacetDiagramDetail {
            designer_info: Some("Long, Bob".to_string()),
            facets_count: Some("60+9".to_string()),
            ..Default::default()
        },
        id_a,
    )
    .unwrap();

    // ...then encountered again under a different source with a slightly
    // differently-cased/whitespaced title, same designer, same facet count.
    let dupes = db
        .find_cross_source_duplicates(
            "gemologyproject.com",
            "  barion   heart ",
            Some("Long, Bob"),
            Some(60),
        )
        .unwrap();
    assert_eq!(dupes.len(), 1);
    assert_eq!(dupes[0].existing_entry_id, id_a);
    assert_eq!(dupes[0].existing_source_id, "facetdiagrams.org");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn find_cross_source_duplicates_does_not_flag_different_designers_sharing_a_title() {
    let path = temp_db_path("dedup_diff_designer");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");

    let entry_a = FacetDiagramEntry {
        title: "Sunburst".to_string(),
        url: "https://facetdiagrams.org/sunburst-a".to_string(),
        design_id: String::new(),
    };
    let id_a = db
        .save_diagram_entry(&entry_a, "facetdiagrams.org")
        .unwrap();
    db.save_diagram_detail(
        &FacetDiagramDetail {
            designer_info: Some("Alice Designer".to_string()),
            facets_count: Some("50".to_string()),
            ..Default::default()
        },
        id_a,
    )
    .unwrap();

    // Two designers genuinely producing same-named designs: same title, same
    // facet count, DIFFERENT designer -- must not be flagged as a duplicate.
    let dupes = db
        .find_cross_source_duplicates(
            "gemologyproject.com",
            "Sunburst",
            Some("Bob Other Designer"),
            Some(50),
        )
        .unwrap();
    assert!(
        dupes.is_empty(),
        "different designers sharing a title must not be flagged as duplicates, got {dupes:?}"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn find_cross_source_duplicates_ignores_matches_within_the_same_source() {
    let path = temp_db_path("dedup_same_source");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");

    let entry_a = FacetDiagramEntry {
        title: "Sunburst".to_string(),
        url: "https://facetdiagrams.org/sunburst-a".to_string(),
        design_id: String::new(),
    };
    let id_a = db
        .save_diagram_entry(&entry_a, "facetdiagrams.org")
        .unwrap();
    db.save_diagram_detail(
        &FacetDiagramDetail {
            facets_count: Some("50".to_string()),
            ..Default::default()
        },
        id_a,
    )
    .unwrap();

    // The same source re-syncing/re-scraping the same title must not be reported
    // as a cross-source collision -- `url` uniqueness already handles that case.
    let dupes = db
        .find_cross_source_duplicates("facetdiagrams.org", "Sunburst", None, Some(50))
        .unwrap();
    assert_eq!(dupes, []);

    let _ = std::fs::remove_file(&path);
}

// ---- Organize: rename_diagram_entry / delete_diagram_entry --------------------

#[test]
fn rename_diagram_entry_updates_the_title() {
    let path = temp_db_path("rename");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");
    let entry = FacetDiagramEntry {
        title: "Old Title".to_string(),
        url: "local://old.asc".to_string(),
        design_id: String::new(),
    };
    let id = db.save_diagram_entry(&entry, "local-import").unwrap();

    db.rename_diagram_entry(id, "  New Title  ").unwrap();

    let full = db.get_diagram_full(id).unwrap().unwrap();
    // Trimmed, per `rename_diagram_entry`'s doc comment.
    assert_eq!(full.title, "New Title");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn rename_diagram_entry_rejects_blank_titles_and_unknown_ids() {
    let path = temp_db_path("rename_errors");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");
    let entry = FacetDiagramEntry {
        title: "Keep Me".to_string(),
        url: "local://keep.asc".to_string(),
        design_id: String::new(),
    };
    let id = db.save_diagram_entry(&entry, "local-import").unwrap();

    assert!(db.rename_diagram_entry(id, "   ").is_err());
    assert!(db.rename_diagram_entry(id + 999, "Anything").is_err());

    // The blank-title attempt must not have touched the existing row.
    let full = db.get_diagram_full(id).unwrap().unwrap();
    assert_eq!(full.title, "Keep Me");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn delete_diagram_entry_cascades_to_detail_angles_and_files() {
    let path = temp_db_path("delete");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");
    let entry = FacetDiagramEntry {
        title: "Doomed Design".to_string(),
        url: "local://doomed.asc".to_string(),
        design_id: String::new(),
    };
    let id = db.save_diagram_entry(&entry, "local-import").unwrap();
    db.save_diagram_detail(
        &FacetDiagramDetail {
            angle_settings_table: vec![crate::model::angle::AngleSetting {
                order_index: 0,
                facet: "T".to_string(),
                angle: "0\u{b0}".to_string(),
                index: "0".to_string(),
                notes: String::new(),
            }],
            attached_files: vec![crate::model::file::AttachedFile {
                name: "doomed.asc".to_string(),
                url: String::new(),
                content: b"GemCad 5.0\n".to_vec(),
            }],
            ..Default::default()
        },
        id,
    )
    .unwrap();

    db.delete_diagram_entry(id).unwrap();

    assert!(db.get_diagram_full(id).unwrap().is_none());
    let remaining_angles: i64 = db
        .conn
        .query_row("SELECT COUNT(*) FROM angle_settings", [], |r| r.get(0))
        .unwrap();
    let remaining_files: i64 = db
        .conn
        .query_row("SELECT COUNT(*) FROM attached_files", [], |r| r.get(0))
        .unwrap();
    assert_eq!(remaining_angles, 0);
    assert_eq!(remaining_files, 0);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn delete_diagram_entry_rejects_unknown_ids() {
    let path = temp_db_path("delete_errors");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create migrated db");
    assert!(db.delete_diagram_entry(123_456).is_err());
    let _ = std::fs::remove_file(&path);
}

// ---- shape_vocabulary --------------------------------------------------------

#[test]
fn a_fresh_database_has_the_full_seeded_shape_vocabulary() {
    let path = temp_db_path("fresh_shape_vocab");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create fresh db");

    let names: Vec<String> = {
        let mut stmt = db
            .conn
            .prepare("SELECT name FROM shape_vocabulary ORDER BY sort_order ASC")
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .flatten()
            .collect()
    };
    assert_eq!(
        names,
        DEFAULT_SHAPES
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<_>>(),
        "a fresh database must be seeded with the full canonical list, in order"
    );

    // Before any design is imported, get_unique_shapes must still return the whole
    // seeded vocabulary -- this is the actual bug being fixed: the old plain
    // `SELECT DISTINCT shape FROM diagram_details` returned nothing here.
    let shapes = db.get_unique_shapes().unwrap();
    let mut expected: Vec<String> = DEFAULT_SHAPES.iter().map(|s| (*s).to_string()).collect();
    expected.sort();
    assert_eq!(shapes, expected);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn shape_vocabulary_migration_is_idempotent_across_two_opens() {
    let path = temp_db_path("shape_vocab_idempotent");

    {
        let db = Database::new(Some(path.to_str().unwrap())).expect("first open seeds");
        let count: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM shape_vocabulary", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, DEFAULT_SHAPES.len() as i64);
    }

    // Second open (re-running the migration) must not duplicate rows or error --
    // INSERT OR IGNORE against the `name` primary key must skip every entry that's
    // already there.
    let db2 = Database::new(Some(path.to_str().unwrap())).expect("second open is idempotent");
    let count: i64 = db2
        .conn
        .query_row("SELECT COUNT(*) FROM shape_vocabulary", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, DEFAULT_SHAPES.len() as i64);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn shape_vocabulary_migration_never_overwrites_or_drops_existing_rows() {
    let path = temp_db_path("shape_vocab_preserves_data");
    let db = Database::new(Some(path.to_str().unwrap())).expect("create fresh db");

    // Simulate a user/GUI having added a custom vocabulary entry, and having
    // re-pointed a seeded entry's sort_order.
    db.conn
        .execute(
            "INSERT INTO shape_vocabulary (name, sort_order) VALUES ('Portuguese', 999)",
            [],
        )
        .unwrap();
    db.conn
        .execute(
            "UPDATE shape_vocabulary SET sort_order = 12345 WHERE name = 'Round'",
            [],
        )
        .unwrap();

    // Re-running the migration (as every `Database::new` open does) must not touch
    // either: not delete the custom row, and not reset the hand-edited sort_order.
    db.migrate_shape_vocabulary().expect("re-run migration");

    let custom_present: i64 = db
        .conn
        .query_row(
            "SELECT COUNT(*) FROM shape_vocabulary WHERE name = 'Portuguese'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(custom_present, 1, "custom row must survive a re-seed");

    let round_order: i64 = db
        .conn
        .query_row(
            "SELECT sort_order FROM shape_vocabulary WHERE name = 'Round'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        round_order, 12345,
        "re-seeding must not overwrite an existing row's data"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn get_unique_shapes_unions_the_vocabulary_with_real_catalogue_data() {
    let db = seeded_db(&[
        // A shape already in DEFAULT_SHAPES.
        ("A", "Round", "1.50", "1.00", "0.10", "50"),
        // A real scraped shape string that is NOT in the canonical list -- this
        // must not be dropped by the union.
        ("B", "Portuguese Round", "1.55", "1.00", "0.10", "50"),
    ]);

    let shapes = db.get_unique_shapes().unwrap();

    // Every canonical entry is still present...
    for shape in DEFAULT_SHAPES {
        assert!(
            shapes.iter().any(|s| s == shape),
            "seeded vocabulary entry '{shape}' must appear in the union, got {shapes:?}"
        );
    }
    // ...and so is the real, non-canonical shape actually in the catalogue.
    assert!(
        shapes.iter().any(|s| s == "Portuguese Round"),
        "a real shape string outside the canonical list must still appear, got {shapes:?}"
    );
    // "Round" must not be duplicated just because it's in both sources.
    assert_eq!(shapes.iter().filter(|s| s.as_str() == "Round").count(), 1);
    // Alphabetically sorted across the whole union.
    let mut sorted = shapes.clone();
    sorted.sort();
    assert_eq!(shapes, sorted);
}

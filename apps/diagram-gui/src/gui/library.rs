//! Local-library management: import `.asc` files, rename, delete, and export the
//! user's own designs -- "Organize" plus "Import"/"Export" in the
//! UI (see `apps/diagram-gui/ui/components/{import_dialog,detail_header}.slint`).
//!
//! Nothing here talks to the network or parses HTML/PDF; it operates only on files
//! the user handed the app directly (`import_path`) or on rows already in the local
//! SQLite catalogue. See `diagram_catalog::local`'s doc comment for the underlying
//! parse/reconstruct logic this module wires up to the UI.

use crate::{
    DiagramDetailData, MainWindow,
    bridge::{library_source::LibrarySource, render_thread::RenderContext},
    gui::{
        detail::reconstruct_planes,
        diagram_list::{apply_attribute_ranges_to_ui, fetch_attribute_ranges},
        search::{apply_diagram_list_to_ui, fetch_diagram_list, read_range_filter},
        show_toast,
    },
};
use diagram_catalog::{
    db::sqlite::{DEFAULT_SHAPES, Database},
    local,
    model::{detail::FacetDiagramDetail, metadata_update::MetadataUpdate},
};
use gemray::geometry::{
    GpuFacetPlane,
    cuts::{FacetSpec, StandardGemCuts},
    girdle, stone_metrics,
};
use glam::DVec3;
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel, Weak};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};
use tracing::warn;

/// Re-reads the search/shape/gear filters currently showing in `ui` and re-runs the
/// catalogue query -- the same "refresh everything that could have changed" step
/// `gui::mod`'s old sync-complete handler used to do, now shared by import/
/// rename/delete/shape-change.
///
/// Import/rename/delete only ever touch the LOCAL database (rename/delete refuse
/// outright while browsing remote -- see `setup_rename_callback`/`setup_delete_callback`'s
/// own doc comments; import always writes locally regardless of which library is being
/// browsed). So the attribute-range refresh -- which reads local attribute ranges --
/// only happens here while [`LibrarySource::Local`] is actually active; a currently-Remote
/// view instead goes through [`crate::gui::search::refresh_diagram_list_via_source`]
/// (already off the UI thread -- see `search::refresh_diagram_list_remote`) so it stays
/// showing remote results rather than silently switching to local ones after a
/// local-only write.
///
/// The LOCAL path itself runs the two `Database` reads (`fetch_attribute_ranges`,
/// `fetch_diagram_list`) on a background thread and only marshals the resulting
/// `ui.set_*` calls back through `upgrade_in_event_loop` -- unlike every OTHER caller
/// of those same reads (a search-box keystroke, a filter-slider drag), which stay
/// synchronous because that responsiveness is the point. This call site is different:
/// it always follows a database WRITE (import/rename/delete/shape-change), so the
/// small extra latency of a thread hop is invisible, while running the read
/// synchronously here is not -- against the real ~3,187-design catalogue this
/// crate's own `perf_probe_refresh_after_library_change_cost` test measures
/// `get_attribute_ranges` + an unfiltered `search_diagrams` at tens of milliseconds
/// combined, long enough on the UI thread to read as a second freeze right after a big
/// import finishes (see this task's BUG 1 write-up).
fn refresh_after_library_change(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
) {
    let current = source
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    match current {
        LibrarySource::Local => {
            let search = ui.get_search_text().to_string();
            let shape_idx = ui.get_selected_shape_index() as usize;
            let shape = ui
                .get_shape_options()
                .row_data(shape_idx)
                .unwrap_or_default()
                .to_string();
            let gear_idx = ui.get_selected_gear_index() as usize;
            let gear = ui
                .get_gear_options()
                .row_data(gear_idx)
                .unwrap_or_default()
                .to_string();
            let range = read_range_filter(ui);
            spawn_local_refresh(ui.as_weak(), Arc::clone(db), search, shape, gear, range);
        }
        LibrarySource::Remote(_) => {
            let search = ui.get_search_text();
            let shape_idx = ui.get_selected_shape_index() as usize;
            let shape = ui
                .get_shape_options()
                .row_data(shape_idx)
                .unwrap_or_default();
            let gear_idx = ui.get_selected_gear_index() as usize;
            let gear = ui.get_gear_options().row_data(gear_idx).unwrap_or_default();
            crate::gui::search::refresh_diagram_list_via_source(
                ui, db, source, &search, &shape, &gear,
            );
        }
    }
}

/// Runs [`fetch_attribute_ranges`] + [`fetch_diagram_list`] on their own worker
/// thread and applies both results in one hop back onto the UI thread -- the
/// background half of [`refresh_after_library_change`]'s `LibrarySource::Local`
/// branch; see that function's own doc comment for why this one call site needs to be
/// async where `sync_range_bounds_to_ui`/`refresh_diagram_list`'s other callers don't.
fn spawn_local_refresh(
    ui_weak: Weak<MainWindow>,
    db: Arc<Mutex<Database>>,
    search: String,
    shape: String,
    gear: String,
    range: diagram_catalog::model::filter::RangeFilter,
) {
    thread::spawn(move || {
        let ranges = fetch_attribute_ranges(&db);
        let list_result = fetch_diagram_list(&db, &search, &shape, &gear, &range);
        let _ = ui_weak.upgrade_in_event_loop(move |ui| {
            if let Some(ranges) = ranges {
                apply_attribute_ranges_to_ui(&ui, &ranges);
            }
            match list_result {
                Ok((items, total)) => apply_diagram_list_to_ui(&ui, items, total),
                Err(e) => {
                    warn!("Failed to search diagrams: {e:?}");
                    ui.set_status_message(format!("Error searching database: {e}").into());
                }
            }
        });
    });
}

/// Fills in the proportion fields `diagram_catalog::local::import_asc` always leaves
/// `None` -- it only ever reads the `.asc` file's own header/tiers, which don't carry
/// these. Reconstructs the same 3D facet planes the viewport would show for this
/// design ([`reconstruct_planes`], the exact path `gui::detail::load_diagram_detail`
/// uses when a design is opened -- reused here rather than re-parsed) and measures
/// them with `gemray::geometry::stone_metrics::measure_solid`, which the meet-solver
/// corpus work measured at ~0.01% median error on L/W and ~0.09% on Vol/W^3 against
/// the real ~2,881-design `.asc` corpus, so it's trustworthy. `measure_solid` still
/// returns `None` for a degenerate/unbounded arrangement (e.g. a schedule missing its
/// closing planes), and every field below is left `None` rather than fabricated when
/// that happens -- never store a number this couldn't actually measure.
///
/// Values are written with plain `f64::to_string()` -- the exact convention
/// `local::import_asc` already uses one call site up, for `refractive_index`/
/// `index_gear`/`facets_count`. That choice matters less than it might: the real
/// `facet_diagrams.sqlite`'s `lw_ratio`/`volume`/`hw_ratio`/`cw_ratio`/`pw_ratio`
/// columns are all REAL-affinity (confirmed against the live file -- they were added
/// via `ALTER TABLE ... ADD COLUMN ... REAL`, not the plain `CREATE TABLE`'s `TEXT`
/// declaration a brand-new database's `diagram_details` would get), so SQLite's own
/// type-affinity conversion normalises whatever numeric text is written here the same
/// way regardless of decimal-place convention -- what actually has to match is that it
/// parses as a plain number at all.
fn apply_measured_metadata(detail: &mut FacetDiagramDetail) {
    // `reconstruct_planes` falls back to `standard_round_brilliant()` when given no
    // facet specs at all -- exactly right for the viewport (something must render),
    // but measuring THAT fallback shape here would attribute a fabricated design's
    // proportions to this one. A `.asc` file that parsed with zero tiers is
    // degenerate enough that this case should be rare to nonexistent in practice;
    // guarding it explicitly costs nothing and keeps this function's own "never
    // fabricate a number" promise honest even then.
    if detail.angle_settings_table.is_empty() {
        return;
    }

    let facet_specs: Vec<FacetSpec> = detail
        .angle_settings_table
        .iter()
        .map(|a| FacetSpec {
            facet: a.facet.clone(),
            angle: a.angle.clone(),
            index: a.index.clone(),
            notes: a.notes.clone(),
        })
        .collect();
    let planes = reconstruct_planes(None, detail.index_gear.as_deref(), &facet_specs);
    let dvec_planes: Vec<(DVec3, f64)> = planes
        .iter()
        .map(|p| {
            (
                DVec3::new(
                    f64::from(p.normal[0]),
                    f64::from(p.normal[1]),
                    f64::from(p.normal[2]),
                ),
                // `measure_solid` wants `n.x <= m` (unit outward normals); `GpuFacetPlane`'s
                // own convention is `n.x + d = 0` (see
                // `optics::raytracer::intersect`'s ray-plane test, `p.d + n.dot(origin)`),
                // i.e. `m = -d` -- confirmed against
                // `StandardGemCuts::standard_round_brilliant`'s table plane
                // (`GpuFacetPlane::new((0,1,0), -0.32)`, physically the table at `y = +0.32`).
                -f64::from(p.d),
            )
        })
        .collect();

    let Some(m) = stone_metrics::measure_solid(&dvec_planes) else {
        return;
    };
    let w = m.width_axis;
    if w < 1e-9 {
        return;
    }

    detail.lw_ratio = Some((m.length_axis / w).to_string());
    detail.hw_ratio = Some((m.total_height / w).to_string());
    detail.cw_ratio = m.crown_height.map(|c| (c / w).to_string());
    detail.pw_ratio = m.pavilion_depth.map(|p| (p / w).to_string());
    // Stored dimensionless, matching how this same figure is printed on a real
    // diagram sheet and how `FacetDiagramDetail::volume`'s own doc comment describes
    // it (mirrors a printed sheet's `Vol/W^3`) -- NOT the raw mast-unit volume.
    detail.volume = Some((m.volume / (w * w * w)).to_string());

    // Classified from the SAME planes, so the girdle outline is measured rather than
    // inferred from the schedule's fold count -- see `classify_shape`.
    detail.shape = classify_shape(&planes, m.length_axis / w);
}

/// Assigns [`FacetDiagramDetail::shape`], but only where the design's own girdle
/// outline makes the call unambiguous. Returns `None` otherwise -- a blank shape is
/// honestly correctable by the user, whereas a confidently wrong one looks
/// authoritative and would quietly poison the library's own shape filter.
///
/// # Why the girdle facet count, and not the schedule's fold count
///
/// A shape name describes the stone's SILHOUETTE, and the girdle facets are that
/// silhouette: `classify_girdle_plane_indices` returns every near-vertical plane, so
/// its length is the outline's side count directly.
///
/// Fold count cannot do this job. `local::import_asc`'s own "Round Trichecker-12"
/// fixture parses to `symmetry_order = 6` while being a ROUND cut built from six
/// repeats -- a fold-count rule calls it a Hexagon. Its girdle, by contrast, has far
/// more than six facets, so this rule reads it correctly. For reference,
/// `StandardGemCuts::standard_round_brilliant` has 16 girdle facets (plane indices
/// 33..49, pinned by `geometry::cuts::STANDARD_ROUND_BRILLIANT_GIRDLE_FACETS`).
///
/// # The rule
///
/// `sides` is the girdle facet count; `lw` the measured length/width ratio (>= 1.0):
///
/// - `sides >= ROUND_MIN_SIDES` and `lw <= ROUND_LW_MAX` -> Round. Enough sides to
///   approximate a circle, and actually circular rather than elongated.
/// - exactly 3 / 4 / 6 / 8 sides -> Triangle / Square / Hexagon / Octagon.
/// - anything else -> `None`.
///
/// A regular n-gon whose side count is a multiple of 4 has an exactly square
/// axis-aligned bounding box (a 90 degree rotation is one of its own symmetries), so
/// Square and Octagon measure `lw == 1.0` by construction and get the tight
/// `ROUND_LW_MAX`. Triangle and Hexagon do not, and carry an inherent
/// `2/sqrt(3) ~= 1.1547`, hence the looser `POLYGON_LW_MAX`. Elongated designs
/// (Oval, Rectangle, Marquise, Pear) commonly measure well above 1.3 and are never
/// guessed among -- distinguishing those four needs outline curvature this does not
/// measure.
///
/// Looks the chosen name up in [`DEFAULT_SHAPES`] rather than returning the literal,
/// so a rename there can never leave this returning a label the vocabulary no longer
/// recognises.
fn classify_shape(planes: &[GpuFacetPlane], lw: f64) -> Option<String> {
    const ROUND_LW_MAX: f64 = 1.03;
    const POLYGON_LW_MAX: f64 = 1.20;
    /// Below this, a many-sided outline is not confidently "round" -- a 10-sided
    /// outline is as plausibly a decagon as a coarse circle, so it gets no shape.
    const ROUND_MIN_SIDES: usize = 12;

    let sides = girdle::classify_girdle_plane_indices(planes).len();
    let name = match sides {
        3 if lw <= POLYGON_LW_MAX => "Triangle",
        4 if lw <= ROUND_LW_MAX => "Square",
        6 if lw <= POLYGON_LW_MAX => "Hexagon",
        8 if lw <= ROUND_LW_MAX => "Octagon",
        n if n >= ROUND_MIN_SIDES && lw <= ROUND_LW_MAX => "Round",
        _ => return None,
    };
    DEFAULT_SHAPES
        .iter()
        .find(|s| **s == name)
        .map(|s| (*s).to_string())
}

/// Runs `f`, converting a panic into an error message instead of letting it unwind
/// past this call. Used to wrap the one per-file step (`local::import_asc` +
/// `apply_measured_metadata`) that reaches into `gemray`'s geometry code -- a crate
/// this module doesn't own and can't guarantee is panic-free on every
/// malformed-but-parseable `.asc`. Without this, a single such file would take down
/// the whole worker thread: `import_path`'s loop would stop dead partway through the
/// batch (silently dropping every file after it), and -- see [`spawn_import`]'s own
/// doc comment -- the completion closure that clears `is_busy` would never run
/// either, wedging the UI exactly as this task's BUG 1(b) describes. With this, it's
/// just one more entry in `failed`, same as an ordinary parse error, and the batch
/// keeps going.
fn catch_file_panic<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> Result<T, String> {
    std::panic::catch_unwind(f).map_err(|payload| {
        payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".to_string())
    })
}

/// Backstop against a pathological filesystem when [`import_path`]'s optional
/// subfolder recursion (Task: FEATURE 3) is enabled -- a symlink/junction loop is
/// already caught by `collect_asc_files_recursive`'s `visited` set regardless of
/// depth, so this only guards an absurdly deep (but non-cyclic) real tree. Ordinary
/// design libraries are a handful of levels deep at most.
const MAX_RECURSE_DEPTH: usize = 32;

/// Walks one directory for `.asc` files, descending into subdirectories when
/// `recurse` is true. `visited` records the canonicalized (symlink-resolved) path of
/// every directory already entered THIS call tree -- a symlink or Windows junction
/// that loops back to an ancestor canonicalizes to a path already in that set, so the
/// second visit is skipped rather than recursing forever. Comparing raw paths
/// wouldn't catch this: a symlink's own path text never repeats even though what it
/// POINTS TO does. `depth` is capped at [`MAX_RECURSE_DEPTH`] as a second, independent
/// backstop for a deep-but-non-cyclic tree. Both guards fail open (skip and log via
/// `warn!`, not abort the whole import) -- one bad subfolder shouldn't stop `.asc`
/// files elsewhere in the tree from being found.
fn collect_asc_files_recursive(
    dir: &Path,
    recurse: bool,
    depth: usize,
    max_depth: usize,
    visited: &mut HashSet<PathBuf>,
    out: &mut Vec<PathBuf>,
) {
    if depth > max_depth {
        warn!(
            "Import: not descending into '{}' -- exceeded max recursion depth ({max_depth})",
            dir.display()
        );
        return;
    }
    let canon = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    if !visited.insert(canon) {
        warn!(
            "Import: skipping '{}' -- already visited (symlink loop?)",
            dir.display()
        );
        return;
    }

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("asc")) {
            out.push(p);
        } else if recurse && p.is_dir() {
            collect_asc_files_recursive(&p, recurse, depth + 1, max_depth, visited, out);
        }
    }
}

/// Imports every `.asc` file at `path`: the file itself if `path` names one, or every
/// `.asc` file inside it if `path` names a folder -- directly inside only, unless
/// `recurse` is true, in which case every subfolder is walked too (Task: FEATURE 3;
/// see [`collect_asc_files_recursive`] for the symlink-loop/depth guards that apply
/// either way). Returns a human-readable summary for the toast/status line.
/// `on_progress(done, total)` fires after each file, letting the caller keep the UI
/// honest about a long folder import without waiting for it to finish (this always
/// runs off the UI thread -- see [`spawn_import`]).
///
/// `db` is locked only around each file's actual database work (the collision check
/// and the two writes), never around the file I/O, `.asc` parsing, or plane
/// reconstruction/`measure_solid` geometry that happens first -- Task: BUG 1(a). Under
/// the OLD code, one `db.lock()` was held for the entire loop, so any UI-thread
/// callback that also needs `db` (opening a design, renaming one, moving a filter
/// slider that re-queries the catalogue) blocked for the full import; now it can only
/// ever block for a single row's write.
/// Resolves what `import_path` should actually import: the single file `path` names,
/// or every `.asc` directly inside it (plus, when `recurse`, its subfolders).
///
/// `Err` carries the user-facing message `import_path` returns verbatim -- an
/// unreadable folder, a path that is neither file nor folder, or a folder with no
/// `.asc` in it. Split out purely so `import_path` stays under clippy's
/// `too_many_lines` limit; the two phases (decide what to import, then import it) were
/// already independent.
fn collect_import_candidates(path: &Path, recurse: bool) -> Result<Vec<PathBuf>, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if path.is_dir() {
        match std::fs::read_dir(path) {
            Ok(entries) => {
                // The top-level folder the user picked isn't itself loop-guarded (it
                // can't be reached via a symlink pointing back to itself before this
                // point) -- only directories walked below it are, via `visited`.
                let mut visited: HashSet<PathBuf> = HashSet::new();
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("asc")) {
                        candidates.push(p);
                    } else if recurse && p.is_dir() {
                        collect_asc_files_recursive(
                            &p,
                            recurse,
                            1,
                            MAX_RECURSE_DEPTH,
                            &mut visited,
                            &mut candidates,
                        );
                    }
                }
            }
            Err(e) => return Err(format!("Could not read folder '{}': {e}", path.display())),
        }
    } else if path.is_file() {
        candidates.push(path.to_path_buf());
    } else {
        return Err(format!("'{}' is not a file or folder.", path.display()));
    }

    if candidates.is_empty() {
        return Err(format!("No .asc files found at '{}'.", path.display()));
    }
    Ok(candidates)
}

fn import_path(
    db: &Arc<Mutex<Database>>,
    path: &Path,
    recurse: bool,
    mut on_progress: impl FnMut(usize, usize),
) -> String {
    let candidates = match collect_import_candidates(path, recurse) {
        Ok(c) => c,
        Err(message) => return message,
    };

    let total = candidates.len();
    let mut imported = 0usize;
    let mut failed: Vec<String> = Vec::new();
    // BUG (found, reported, not fixed here -- `diagram_catalog::local::import_asc`
    // dedupes purely on `local://<file_name>`, the bare filename rather than the
    // source path -- see that function's own doc comment). `diagram_entries.url` is
    // UNIQUE, so importing `round.asc` from two different folders makes the second
    // silently REPLACE the first. This crate can't change that key (it lives in
    // `diagram_catalog`, out of scope here), so this loop can only detect the
    // collision -- either against another file already seen in THIS batch, or
    // against a design already saved from a past import, any folder -- and surface
    // it in the result summary below instead of letting it pass silently. The real
    // fix belongs in `diagram_catalog::local::import_asc`: key on something derived
    // from content or the full source path, not the bare filename. Recursive import
    // (Task: FEATURE 3) makes this collision far more likely in practice -- a
    // same-named file living in two different subfolders is now completely ordinary.
    let mut seen_in_batch: HashSet<String> = HashSet::new();
    let mut replaced: Vec<String> = Vec::new();

    for (i, file_path) in candidates.into_iter().enumerate() {
        let file_name = file_path.file_name().map_or_else(
            || "unknown.asc".to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        on_progress(i + 1, total);

        let content = match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                failed.push(format!("{file_name} (read error: {e})"));
                continue;
            }
        };

        let url = format!("local://{file_name}");
        // Recorded before parsing, unconditionally, so a batch-internal name
        // collision is still caught even if THIS occurrence goes on to fail to parse
        // or panic below.
        let seen_before_in_batch = !seen_in_batch.insert(file_name.clone());

        // Parsing and measuring run OUTSIDE the database lock (BUG 1(a)) and inside
        // `catch_file_panic` (BUG 1(b)) -- see both functions' own doc comments.
        let parse_result = catch_file_panic(std::panic::AssertUnwindSafe(|| {
            local::import_asc(&file_name, &content).map(|mut parsed| {
                // Fills the measured proportions AND the shape -- both come from the
                // same reconstructed planes, so they are measured once, not twice.
                apply_measured_metadata(&mut parsed.detail);
                parsed
            })
        }));

        let parsed = match parse_result {
            Ok(Ok(parsed)) => parsed,
            Ok(Err(e)) => {
                failed.push(format!("{file_name} (parse error: {e})"));
                continue;
            }
            Err(panic_msg) => {
                warn!("Import panicked while processing '{file_name}': {panic_msg}");
                failed.push(format!("{file_name} (internal error: {panic_msg})"));
                continue;
            }
        };

        let save_result = {
            let db = db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let is_collision =
                seen_before_in_batch || db.has_detail_for_entry_url(&url).unwrap_or(false);
            db.save_diagram_entry(&parsed.entry, local::LOCAL_SOURCE_ID)
                .and_then(|id| db.save_diagram_detail(&parsed.detail, id))
                .map(|()| is_collision)
        };
        match save_result {
            Ok(is_collision) => {
                imported += 1;
                if is_collision {
                    replaced.push(file_name);
                }
            }
            Err(e) => failed.push(format!("{file_name} (save error: {e})")),
        }
    }

    let mut summary = if failed.is_empty() {
        format!("Imported {imported} .asc file(s).")
    } else {
        warn!(
            "Import from {}: {} failed: {:?}",
            path.display(),
            failed.len(),
            failed
        );
        format!(
            "Imported {imported} .asc file(s); {} skipped ({}).",
            failed.len(),
            failed.join("; ")
        )
    };
    if !replaced.is_empty() {
        use std::fmt::Write as _;
        let _ = write!(
            summary,
            " {} design(s) replaced an existing entry of the same name ({}) -- \
             filename-only dedup, see this app's import report.",
            replaced.len(),
            replaced.join(", ")
        );
    }
    summary
}

/// Runs [`import_path`] on its own worker thread and marshals every UI update back
/// through `upgrade_in_event_loop` -- the same idiom
/// `bridge::export_thread::spawn_export` + `gui::render_export` already use for a
/// long-running operation with progress, not a second one invented for import.
/// `db`/`source` are cloned `Arc`s, cheap to move onto the thread; [`import_path`]
/// itself now locks `db` only around each file's DB writes (Task: BUG 1(a)), so the UI
/// thread is never blocked waiting on this thread's lock for more than one row at a
/// time.
///
/// A `BusyGuard` local to this thread's closure clears `is_busy` in its `Drop` impl,
/// unconditionally -- Task: BUG 1(b). Under the OLD code, `is_busy` was only cleared
/// from the success closure below, so a panic ANYWHERE in this thread (before
/// `catch_file_panic` existed, an ordinary panic inside `gemray`'s geometry code
/// during a single file's processing was enough) skipped straight past it and left
/// `is_busy` stuck true forever -- exactly the user's reported symptom: the design
/// commits before the panic point, but the UI never un-freezes because the
/// completion closure that would have told it to never runs. A `Drop` guard was
/// chosen over `catch_unwind`-ing this whole closure or an explicit reset on every
/// return path because it can't be skipped by a `continue`/early `return`/panic added
/// later without anyone noticing -- there is exactly one exit path (the guard drops),
/// not one per branch to keep in sync. (`catch_file_panic` inside `import_path` is a
/// separate, complementary fix: it stops a single bad file from reaching this point
/// as a panic at all, so the batch keeps processing the REST of the files too, not
/// just so `is_busy` recovers after the fact.)
fn spawn_import(
    ui_weak: Weak<MainWindow>,
    db: Arc<Mutex<Database>>,
    source: Arc<Mutex<LibrarySource>>,
    path: PathBuf,
    recurse: bool,
) {
    thread::spawn(move || {
        struct BusyGuard(Weak<MainWindow>);
        impl Drop for BusyGuard {
            fn drop(&mut self) {
                let ui_weak = self.0.clone();
                let _ = ui_weak.upgrade_in_event_loop(move |ui| ui.set_is_busy(false));
            }
        }
        let _busy_guard = BusyGuard(ui_weak.clone());

        let progress_ui_weak = ui_weak.clone();
        let message = import_path(&db, &path, recurse, move |done, total| {
            let _ = progress_ui_weak.upgrade_in_event_loop(move |ui| {
                ui.set_import_done(done as i32);
                ui.set_import_total(total as i32);
                ui.set_import_progress(done as f32 / total as f32);
                ui.set_status_message(format!("Importing {done} / {total}...").into());
            });
        });
        let _ = ui_weak.upgrade_in_event_loop(move |ui| {
            ui.set_import_result_text(message.clone().into());
            ui.set_status_message(message.clone().into());
            let toast_kind = if message.starts_with("Imported 0") || !message.contains("Imported") {
                "error"
            } else {
                "success"
            };
            show_toast(&ui, &message, toast_kind);
            refresh_after_library_change(&ui, &db, &source);
        });
        // `_busy_guard` drops here: on a normal return, right after the completion
        // closure above has been HANDED TO the event loop (not necessarily run yet --
        // `upgrade_in_event_loop` only enqueues it), so the `is_busy` reset is queued
        // right behind it and both apply in order; on an unwind from anywhere in this
        // function, it drops during that unwind instead, which is the whole point.
    });
}

/// Wires up the "Import" popup's native pickers (`import_dialog.slint`'s "Choose
/// file..."/"Choose folder..." buttons): picking a target now runs the import
/// immediately, off the UI thread (see [`spawn_import`]) -- there is no longer a
/// separate path field or "Import" button to confirm through (BUG 2/3 of this task).
/// Cancelling the native picker does nothing at all: no import, no error, no UI
/// change, the popup stays open exactly as it was (`rfd::FileDialog::pick_file`/
/// `pick_folder` return `None`, and both handlers below just fall through). The
/// folder picker additionally carries a `bool` -- whether to recurse into
/// subfolders (Task: FEATURE 3) -- decided by `import_dialog.slint`'s toggle at the
/// moment the button was clicked; see `on_pick_asc_folder`'s own comment below.
///
/// Always writes to the LOCAL database regardless of which library is currently being
/// browsed -- Import is inherently a local-only operation (the user's own `.asc`
/// file), so it needs no `source` guard the way rename/delete do; `source` is only
/// threaded through here so [`refresh_after_library_change`] keeps showing whichever
/// library was already on screen afterwards.
pub fn setup_import_callback(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
) {
    let db_file = Arc::clone(db);
    let source_file = Arc::clone(source);
    let ui_weak_file = ui.as_weak();
    ui.on_pick_asc_file(move || {
        let Some(ui) = ui_weak_file.upgrade() else {
            return;
        };
        // Blocking `rfd::FileDialog`, invoked directly on the Slint UI/event-loop
        // thread -- see `apps/diagram-gui/Cargo.toml`'s `rfd` dependency comment for
        // why that's the supported way to call it here. Only the picker itself
        // blocks; the import that follows runs on its own thread.
        let dialog = rfd::FileDialog::new().add_filter(".asc design", &["asc"]);
        let Some(path) = dialog.pick_file() else {
            return;
        };
        ui.set_is_busy(true);
        ui.set_status_message("Importing...".into());
        reset_import_progress(&ui);
        spawn_import(
            ui_weak_file.clone(),
            Arc::clone(&db_file),
            Arc::clone(&source_file),
            path,
            false,
        );
    });

    let db_folder = Arc::clone(db);
    let source_folder = Arc::clone(source);
    let ui_weak_folder = ui.as_weak();
    // `recurse` is `import_dialog.slint`'s "Include subfolders" toggle, read and
    // handed to this callback at the exact moment "Choose folder..." was clicked --
    // see that component's own doc comment on `recurse_subfolders` for why it MUST be
    // set before the click (the picker opens and the import starts in this same
    // callback; toggling it afterwards has nothing left to affect).
    ui.on_pick_asc_folder(move |recurse: bool| {
        let Some(ui) = ui_weak_folder.upgrade() else {
            return;
        };
        let dialog = rfd::FileDialog::new();
        let Some(path) = dialog.pick_folder() else {
            return;
        };
        ui.set_is_busy(true);
        ui.set_status_message("Importing...".into());
        reset_import_progress(&ui);
        spawn_import(
            ui_weak_folder.clone(),
            Arc::clone(&db_folder),
            Arc::clone(&source_folder),
            path,
            recurse,
        );
    });
}

/// Clears the previous import's progress readout before starting a new one -- without
/// this, `import_done`/`import_total`/`import_progress` would keep showing the LAST
/// import's finished state (e.g. "file 210 / 210", 100%) for the brief moment before
/// this import's own first [`spawn_import`] progress callback fires.
fn reset_import_progress(ui: &MainWindow) {
    ui.set_import_done(0);
    ui.set_import_total(0);
    ui.set_import_progress(0.0);
}

/// Wires up the detail header's inline rename (pencil icon -> `rename_diagram`).
///
/// Refuses outright while a remote library is being browsed: the selected entry id in
/// that case names a row in the REMOTE server's catalogue, not this process's local
/// database, and the library protocol is read-only in any case (see
/// `gemray_net::library`'s module doc comment -- there is no write request to send even
/// if this wanted to act on the remote row instead). Renaming a LOCAL row while
/// browsing remote is not offered either -- the visible list, and so the selection,
/// is the remote one.
pub fn setup_rename_callback(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
) {
    let db_rename = Arc::clone(db);
    let source_rename = Arc::clone(source);
    let ui_weak = ui.as_weak();
    ui.on_rename_diagram(move |new_title: SharedString| {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };
        if source_rename
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_remote()
        {
            show_toast(
                &ui,
                "Switch to the local library to rename a design.",
                "error",
            );
            return;
        }
        let entry_id = ui.get_selected_entry_id();
        if entry_id < 0 {
            return;
        }
        let result = {
            let db = db_rename
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            db.rename_diagram_entry(i64::from(entry_id), &new_title)
        };
        match result {
            Ok(()) => {
                let mut detail = ui.get_current_detail();
                detail.title = new_title;
                ui.set_current_detail(detail);
                show_toast(&ui, "Renamed.", "success");
                refresh_after_library_change(&ui, &db_rename, &source_rename);
            }
            Err(e) => show_toast(&ui, &format!("Rename failed: {e}"), "error"),
        }
    });
}

/// Sets `entry_id`'s shape to `new_shape`, called by [`setup_set_shape_callback`].
///
/// Edits exactly one column. `Database::update_diagram_metadata` issues a narrow
/// `UPDATE diagram_details SET ...` naming only the fields handed to it, so every
/// other column keeps whatever it already held -- unlike
/// `Database::save_diagram_detail`, which fully REPLACES a design's detail row and
/// would therefore zero anything the caller failed to carry across.
///
/// The rest of the [`MetadataUpdate`] below is read straight back out of the design's
/// current record and passed through unchanged, so the net effect is `shape` and
/// nothing else. In particular the proportions are NOT recomputed: they are measured
/// data (see [`apply_measured_metadata`], which derives them once at import from the
/// design's own geometry), and re-deriving them on an unrelated shape edit would be
/// both wasted work and a chance to disagree with the stored value.
///
/// This used to be considerably worse. `FullDiagramRecord` was once a strict subset
/// of `FacetDiagramDetail`, missing a dozen columns, so a read-modify-write through
/// `save_diagram_detail` silently erased them -- including the very proportions and
/// symmetry values the import step had just computed. The workaround was to re-parse
/// the design's original `.asc` attachment on every shape edit to recover them, with a
/// lossy fallback for designs that had no attachment. Both the subset gap and the
/// workaround are gone: the record now carries every column, and this method writes
/// only what it is given.
fn set_diagram_shape(db: &Database, entry_id: i64, new_shape: &str) -> Result<(), String> {
    let full = db
        .get_diagram_full(entry_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Diagram detail not found.".to_string())?;

    let update = MetadataUpdate {
        shape: Some(new_shape.to_string()),
        // Everything below is this design's existing value, round-tripped unchanged.
        designer_info: full.designer_info,
        refractive_index: full.refractive_index,
        index_gear: full.index_gear,
        facets_count: full.facets_count,
        symmetry_order: full.symmetry_order,
        mirror_symmetry: full.mirror_symmetry,
        lw_ratio: full.lw_ratio,
        hw_ratio: full.hw_ratio,
        cw_ratio: full.cw_ratio,
        pw_ratio: full.pw_ratio,
        volume: full.volume,
    };
    db.update_diagram_metadata(entry_id, &update)
        .map_err(|e| e.to_string())
}

/// Builds the shape picker's option list and the index within it that matches
/// `current` (the selected design's own `shape`, `None`/empty for an unclassified
/// import). Called from `gui::detail::load_diagram_detail` every time a LOCAL design
/// is opened, so the dropdown always reflects this design and the library's present
/// vocabulary.
///
/// `Database::get_unique_shapes` already unions the seeded `DEFAULT_SHAPES` with
/// every distinct scraped value actually present, alphabetically -- this only adds
/// one more thing: if `current` itself isn't in that union for some reason (should be
/// rare -- it would mean this design's own `shape` value was written by something
/// other than `save_diagram_detail`/the seeded vocabulary), it's inserted too, so
/// opening the picker can never silently drop the design's existing value out of the
/// list before the user has touched anything.
pub fn build_shape_picker_options(
    db: &Database,
    current: Option<&str>,
) -> (Vec<SharedString>, i32) {
    let mut shapes = db.get_unique_shapes().unwrap_or_default();
    let current = current.unwrap_or("").trim();
    if !current.is_empty() && !shapes.iter().any(|s| s == current) {
        shapes.push(current.to_string());
        shapes.sort();
    }
    let index = if current.is_empty() {
        -1
    } else {
        shapes
            .iter()
            .position(|s| s == current)
            .map_or(-1, |i| i as i32)
    };
    (shapes.into_iter().map(SharedString::from).collect(), index)
}

/// Wires up the detail header's shape picker (pencil icon next to the "Shape:" chip,
/// same idiom as [`setup_rename_callback`]'s title pencil) -> `set_shape`.
///
/// Refuses outright while a remote library is being browsed -- see
/// `setup_rename_callback`'s own doc comment; `detail_header.slint` additionally only
/// shows the pencil for `root.detail.is_local`, so this guard is a backstop, not the
/// only thing standing between a remote-sourced id and the local database.
pub fn setup_set_shape_callback(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
) {
    let db_shape = Arc::clone(db);
    let source_shape = Arc::clone(source);
    let ui_weak = ui.as_weak();
    ui.on_set_shape(move |new_shape: SharedString| {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };
        if source_shape
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_remote()
        {
            show_toast(
                &ui,
                "Switch to the local library to change a design's shape.",
                "error",
            );
            return;
        }
        let entry_id = ui.get_selected_entry_id();
        if entry_id < 0 {
            return;
        }
        let result = {
            let db = db_shape
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            set_diagram_shape(&db, i64::from(entry_id), new_shape.as_str())
        };
        match result {
            Ok(()) => {
                let mut detail = ui.get_current_detail();
                detail.shape = new_shape;
                ui.set_current_detail(detail);
                show_toast(&ui, "Shape updated.", "success");
                refresh_after_library_change(&ui, &db_shape, &source_shape);
            }
            Err(e) => show_toast(&ui, &format!("Shape update failed: {e}"), "error"),
        }
    });
}

/// Wires up the detail header's delete confirm -> `delete_diagram`: removes the
/// entry (cascading to its detail/angle-settings/attached-files rows, see
/// `Database::delete_diagram_entry`) and clears the now-stale detail/3D-viewport
/// state.
///
/// Refuses outright while a remote library is being browsed -- see
/// `setup_rename_callback`'s own doc comment; the same reasoning applies verbatim
/// (and matters even more here, since a mistaken delete against the wrong local row by
/// a remote-sourced id would be destructive, not just cosmetically wrong).
pub fn setup_delete_callback(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
    render_ctx: &Arc<Mutex<RenderContext>>,
) {
    let db_delete = Arc::clone(db);
    let source_delete = Arc::clone(source);
    let render_ctx_delete = render_ctx.clone();
    let ui_weak = ui.as_weak();
    ui.on_delete_diagram(move || {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };
        if source_delete
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_remote()
        {
            show_toast(
                &ui,
                "Switch to the local library to delete a design.",
                "error",
            );
            return;
        }
        let entry_id = ui.get_selected_entry_id();
        if entry_id < 0 {
            return;
        }
        let result = {
            let db = db_delete
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            db.delete_diagram_entry(i64::from(entry_id))
        };
        match result {
            Ok(()) => {
                ui.set_current_detail(DiagramDetailData {
                    id: -1,
                    title: SharedString::default(),
                    url: SharedString::default(),
                    designer: SharedString::default(),
                    shape: SharedString::default(),
                    gear: SharedString::default(),
                    facets: SharedString::default(),
                    lw_ratio: SharedString::default(),
                    ri: SharedString::default(),
                    volume: SharedString::default(),
                    competition: SharedString::default(),
                    image_name: SharedString::default(),
                    has_image: false,
                    is_local: false,
                    hw_ratio: SharedString::default(),
                    cw_ratio: SharedString::default(),
                    pw_ratio: SharedString::default(),
                    symmetry_order: SharedString::default(),
                    mirror_symmetry: false,
                });
                ui.set_selected_entry_id(-1);
                ui.set_current_angles(ModelRc::new(VecModel::from(Vec::new())));
                ui.set_current_files(ModelRc::new(VecModel::from(Vec::new())));
                {
                    let mut ctx = render_ctx_delete
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    ctx.active_planes = StandardGemCuts::standard_round_brilliant();
                    ctx.dirty = true;
                }
                show_toast(&ui, "Diagram deleted.", "info");
                refresh_after_library_change(&ui, &db_delete, &source_delete);
            }
            Err(e) => show_toast(&ui, &format!("Delete failed: {e}"), "error"),
        }
    });
}

/// Wires up "Export .asc": for a design with an original `.asc` attachment, exports
/// that file byte-for-byte (delegates to `gui::detail::export_diagram_file`, the same
/// path the Attachments tab's per-file export button already uses). Otherwise
/// rebuilds one from the stored angle-settings table via
/// `diagram_catalog::local::reconstruct_asc_schedule` + `lapidary::asc::to_asc_string`
/// -- see that function's doc comment on why the result is marked `RECONSTRUCTED`.
///
/// Refuses outright while a remote library is being browsed -- same reasoning as
/// `setup_rename_callback`/`setup_delete_callback`: the selected entry id names a row
/// in the remote catalogue, and `Database::get_diagram_full` below must never be
/// called with it against the LOCAL database.
pub fn setup_export_asc_callback(
    ui: &MainWindow,
    db: &Arc<Mutex<Database>>,
    source: &Arc<Mutex<LibrarySource>>,
) {
    let db_export = Arc::clone(db);
    let source_export = Arc::clone(source);
    let ui_weak = ui.as_weak();
    ui.on_export_asc(move || {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };
        if source_export
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_remote()
        {
            show_toast(
                &ui,
                "Switch to the local library to export a reconstructed .asc (use the Attachments tab to download a remote design's own files).",
                "error",
            );
            return;
        }
        let entry_id = ui.get_selected_entry_id();
        if entry_id < 0 {
            show_toast(&ui, "No diagram selected for export.", "error");
            return;
        }

        let full = {
            let db = db_export
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            db.get_diagram_full(i64::from(entry_id))
        };
        let Ok(Some(full)) = full else {
            show_toast(&ui, "Diagram detail not found.", "error");
            return;
        };

        // Whether an already-parsed `.asc` attachment exists (byte-for-byte export,
        // same path `gui::detail::export_diagram_file` uses for the Attachments tab)
        // or one has to be reconstructed below, both branches now write to a
        // user-chosen destination rather than silently into `./exports/` -- one Save
        // As dialog shared by both, prompted BEFORE either does any work, so
        // cancelling writes nothing either way (same cancel rule as
        // `gui::detail::export_diagram_file_via_source`).
        let existing_name = full
            .attached_files
            .iter()
            .find(|f| f.name.to_lowercase().ends_with(".asc"))
            .map(|f| f.name.clone());
        let default_file_name =
            existing_name.clone().unwrap_or_else(|| format!("{}.asc", sanitize_filename(&full.title)));

        let mut dialog = rfd::FileDialog::new()
            .set_file_name(&default_file_name)
            .add_filter(".asc design", &["asc"]);
        let default_dir = std::path::Path::new("exports");
        if default_dir.is_dir() {
            dialog = dialog.set_directory(default_dir);
        }
        // Blocking `rfd::FileDialog`, invoked directly on the Slint UI/event-loop
        // thread -- see `apps/diagram-gui/Cargo.toml`'s `rfd` dependency comment for
        // why that's the supported way to call it here.
        let Some(dest_path) = dialog.save_file() else {
            let msg = "Export cancelled.".to_string();
            ui.set_status_message(msg.clone().into());
            show_toast(&ui, &msg, "info");
            return;
        };

        if let Some(name) = existing_name {
            match crate::gui::detail::export_diagram_file(
                &db_export,
                i64::from(entry_id),
                &name,
                &dest_path,
            ) {
                Ok(msg) => {
                    ui.set_status_message(msg.clone().into());
                    show_toast(&ui, &msg, "success");
                }
                Err(e) => show_toast(&ui, &e, "error"),
            }
            return;
        }

        let Some(schedule) = local::reconstruct_asc_schedule(
            &full.title,
            full.refractive_index.as_deref(),
            full.index_gear.as_deref(),
            &full.angle_settings,
        ) else {
            show_toast(&ui, "No cutting-schedule data to export for this diagram.", "error");
            return;
        };

        let text = lapidary::asc::to_asc_string(&schedule);
        if let Some(parent) = dest_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&dest_path, text) {
            Ok(()) => {
                let msg = format!(
                    "Exported reconstructed schedule to {} \
                     (mast distances are placeholders -- see the file's RECONSTRUCTED header).",
                    dest_path.display()
                );
                ui.set_status_message(msg.clone().into());
                show_toast(&ui, &msg, "success");
            }
            Err(e) => show_toast(
                &ui,
                &format!("Failed to write {}: {e}", dest_path.display()),
                "error",
            ),
        }
    });
}

/// Strips characters Windows (and, incidentally, every other common filesystem)
/// disallows in a file name, so an arbitrary design title is always a safe file name.
fn sanitize_filename(title: &str) -> String {
    let cleaned: String = title
        .chars()
        .map(|c| {
            if matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "design".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Manual perf probe for BUG 1's third suspect: `refresh_after_library_change`
    /// re-queries the whole catalogue synchronously on the UI thread after every
    /// import. Opens the user's real ~3,187-design `facet_diagrams.sqlite`
    /// READ-ONLY (never as a test fixture to write into -- see this crate's own
    /// domain rules) and times exactly the two calls that completion path makes:
    /// `get_attribute_ranges` (`sync_range_bounds_to_ui`) and an unfiltered
    /// `search_diagrams` (`refresh_diagram_list`). `#[ignore]`d: it depends on a file
    /// that only exists on this workstation, so it must never run in CI; run
    /// explicitly with `cargo test -p diagram-gui -- --ignored perf_probe --nocapture`.
    #[test]
    #[ignore = "manual perf probe against the real local catalogue, not for CI"]
    fn perf_probe_refresh_after_library_change_cost() {
        let real_db_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../facet_diagrams.sqlite");
        if !Path::new(real_db_path).is_file() {
            eprintln!("skipping: {real_db_path} not found on this machine");
            return;
        }
        let db = Database::open_read_only(real_db_path).expect("open real catalogue read-only");

        let t0 = std::time::Instant::now();
        let ranges = db.get_attribute_ranges().expect("get_attribute_ranges");
        let ranges_elapsed = t0.elapsed();

        let range = diagram_catalog::model::filter::RangeFilter::default();
        let t1 = std::time::Instant::now();
        let items = db
            .search_diagrams("", "All", "All", &range)
            .expect("search_diagrams");
        let search_elapsed = t1.elapsed();

        let t2 = std::time::Instant::now();
        let total = db.get_total_count().expect("get_total_count");
        let count_elapsed = t2.elapsed();

        eprintln!(
            "get_attribute_ranges: {ranges_elapsed:?} ({ranges:?})\n\
             search_diagrams (unfiltered): {search_elapsed:?} ({} rows)\n\
             get_total_count: {count_elapsed:?} ({total} designs)",
            items.len(),
        );
        // Not a hard perf assertion (machine-dependent) -- this is a measurement
        // probe, not a regression gate. See this test's own report in the task
        // write-up for the numbers actually observed.
    }

    #[test]
    fn sanitize_filename_replaces_reserved_characters() {
        assert_eq!(
            sanitize_filename("Round: Trichecker/12 \"Special\""),
            "Round_ Trichecker_12 _Special_"
        );
    }

    #[test]
    fn sanitize_filename_falls_back_for_empty_or_blank_titles() {
        assert_eq!(sanitize_filename(""), "design");
        assert_eq!(sanitize_filename("   "), "design");
    }

    /// The strongest available anchor: the built-in standard round brilliant has 16
    /// girdle facets, so the outline-based rule must call it Round.
    #[test]
    fn classify_shape_calls_the_standard_round_brilliant_round() {
        let planes = StandardGemCuts::standard_round_brilliant();
        assert_eq!(
            classify_shape(&planes, 1.0).as_deref(),
            Some("Round"),
            "16 girdle facets at lw == 1.0 must classify as Round"
        );
    }

    /// The regression this rule exists for. A fold-count rule called "Round
    /// Trichecker-12" a Hexagon, because its schedule declares 6-fold symmetry while
    /// the cut is round. Classification keys on the girdle OUTLINE instead, so a
    /// round outline stays Round no matter what fold count the schedule declares --
    /// `classify_shape` no longer receives `symmetry_order` at all, which is what
    /// makes that misreading unrepresentable rather than merely unlikely.
    #[test]
    fn classify_shape_ignores_fold_count_entirely() {
        let planes = StandardGemCuts::standard_round_brilliant();
        // Same planes, and no symmetry_order is threaded in from anywhere: the only
        // inputs are the outline and the measured ratio.
        assert_eq!(classify_shape(&planes, 1.0).as_deref(), Some("Round"));
    }

    /// An elongated stone is never guessed at, however round-looking its outline:
    /// Oval/Marquise/Pear cannot be told apart by side count alone, so the honest
    /// answer is no shape rather than a confident wrong one.
    #[test]
    fn classify_shape_refuses_to_guess_for_an_elongated_outline() {
        let planes = StandardGemCuts::standard_round_brilliant();
        assert_eq!(
            classify_shape(&planes, 1.6),
            None,
            "a 1.6 length/width ratio must not be classified Round"
        );
    }

    /// No girdle facets at all (or too few to be confident) yields no shape rather
    /// than a panic or a default.
    #[test]
    fn classify_shape_returns_none_without_a_usable_girdle() {
        assert_eq!(classify_shape(&[], 1.0), None);
    }

    // --- BUG 1 reproduction + regression tests ------------------------------------
    //
    // Same fixture `diagram_catalog::local`'s own tests use (see that crate's
    // `local::tests::SAMPLE_ASC`) -- duplicated here since it's private there. Good
    // enough to exercise the full `import_asc` + `apply_measured_metadata` path
    // without needing a real corpus file (which this crate's own domain rules forbid
    // using as a test fixture in any case).
    const VALID_ASC: &str = "GemCad 5.0\n\
g 96 0.0\n\
y 6 y\n\
I 1.72\n\
H Round Trichecker-12\n\
a -41.000000 0.64991234 92 n 1 84 76 68 60\n\
a 41.000000 0.5 92 n T\n";

    use std::sync::atomic::{AtomicU32, Ordering};

    /// A fresh, empty scratch directory under the OS temp dir -- same naming
    /// convention `diagram_catalog::db::sqlite::tests::temp_db_path` uses for its own
    /// throwaway files, extended to a directory here.
    fn temp_dir_for_test(label: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "diagram_gui_test_{label}_{n}_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp test dir");
        dir
    }

    /// A fresh path for a throwaway SQLite file -- never the real
    /// `facet_diagrams.sqlite` (this crate's domain rules forbid that outright); every
    /// test below builds its own temp database via [`Database::new`].
    fn temp_db_path_for_test(label: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "diagram_gui_test_db_{label}_{n}_{}.sqlite",
            std::process::id()
        ))
    }

    fn open_temp_db(path: &Path) -> Arc<Mutex<Database>> {
        Arc::new(Mutex::new(
            Database::new(Some(path.to_str().expect("temp path is valid UTF-8")))
                .expect("create fresh temp test db"),
        ))
    }

    /// `set_diagram_shape` must edit `shape` and NOTHING else.
    ///
    /// `diagram-catalog` already proves `update_diagram_metadata` writes only the
    /// columns it is handed; this proves the wiring on THIS side hands it all of them.
    /// Omitting one field from the `MetadataUpdate` literal would compile fine and
    /// silently blank that column on every shape edit -- exactly the class of data loss
    /// the re-parse workaround this function replaced existed to avoid.
    #[test]
    fn set_diagram_shape_changes_only_the_shape() {
        let path = temp_db_path_for_test("set_shape");
        let db_arc = open_temp_db(&path);
        let db = db_arc.lock().expect("lock temp db");

        let entry = diagram_catalog::model::entry::FacetDiagramEntry {
            title: "Shape edit probe".to_string(),
            url: "local://shape_probe.asc".to_string(),
            design_id: String::new(),
        };
        let detail = FacetDiagramDetail {
            refractive_index: Some("1.762".to_string()),
            index_gear: Some("96".to_string()),
            facets_count: Some("57".to_string()),
            symmetry_order: Some("8".to_string()),
            mirror_symmetry: Some(true),
            lw_ratio: Some("1.234".to_string()),
            hw_ratio: Some("0.617".to_string()),
            cw_ratio: Some("0.145".to_string()),
            pw_ratio: Some("0.431".to_string()),
            volume: Some("0.187".to_string()),
            designer_info: Some("Somebody; Some Journal".to_string()),
            shape: Some("Oval".to_string()),
            ..FacetDiagramDetail::default()
        };
        let entry_id = db
            .save_diagram_entry(&entry, local::LOCAL_SOURCE_ID)
            .expect("save entry");
        db.save_diagram_detail(&detail, entry_id)
            .expect("save detail");

        set_diagram_shape(&db, entry_id, "Cushion").expect("set shape");

        let after = db
            .get_diagram_full(entry_id)
            .expect("read back")
            .expect("row exists");
        assert_eq!(after.shape.as_deref(), Some("Cushion"), "shape must change");
        // Every other field must be exactly what it was. `pw_ratio`/`cw_ratio`/
        // `hw_ratio`/`symmetry_order`/`mirror_symmetry` are the ones the old
        // `FullDiagramRecord` could not carry at all, so they are the load-bearing
        // assertions here.
        assert_eq!(after.refractive_index.as_deref(), Some("1.762"));
        assert_eq!(after.index_gear.as_deref(), Some("96"));
        assert_eq!(after.facets_count.as_deref(), Some("57"));
        assert_eq!(after.symmetry_order.as_deref(), Some("8"));
        assert_eq!(after.mirror_symmetry, Some(true));
        assert_eq!(after.lw_ratio.as_deref(), Some("1.234"));
        assert_eq!(after.hw_ratio.as_deref(), Some("0.617"));
        assert_eq!(after.cw_ratio.as_deref(), Some("0.145"));
        assert_eq!(after.pw_ratio.as_deref(), Some("0.431"));
        assert_eq!(after.volume.as_deref(), Some("0.187"));
        assert_eq!(
            after.designer_info.as_deref(),
            Some("Somebody; Some Journal")
        );

        drop(db);
        drop(db_arc);
        std::fs::remove_file(&path).ok();
    }

    /// The exact mechanism [`import_path`] leans on to keep one bad file from taking
    /// the whole worker thread down (Task: BUG 1(b)) -- proven directly, independent
    /// of `gemray`'s actual geometry code (which this module doesn't own and can't
    /// force to panic on demand for a test).
    #[test]
    fn catch_file_panic_converts_a_panic_into_an_error_message_instead_of_unwinding() {
        // The panic is deliberate and caught -- suppress the default hook's stderr
        // noise for it so this doesn't look like an unhandled test failure in the log.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = catch_file_panic(std::panic::AssertUnwindSafe(|| -> i32 {
            panic!("deliberate test panic");
        }));
        std::panic::set_hook(prev_hook);
        assert_eq!(result, Err("deliberate test panic".to_string()));
    }

    #[test]
    fn catch_file_panic_returns_ok_when_the_closure_does_not_panic() {
        assert_eq!(
            catch_file_panic(std::panic::AssertUnwindSafe(|| 42)),
            Ok(42)
        );
    }

    /// The reproduction for this task's BUG 1: a batch with one good file and one
    /// file that fails to parse must still import the good one, record the bad one in
    /// `failed` rather than abort, and report progress for both -- not just silently
    /// stop partway (which is what the OLD whole-loop `db.lock()` plus an uncaught
    /// panic anywhere downstream would risk turning into a wedged `is_busy`, per this
    /// task's write-up).
    #[test]
    fn import_path_continues_after_one_file_fails_to_parse() {
        let dir = temp_dir_for_test("parse_fail");
        std::fs::write(dir.join("good.asc"), VALID_ASC).expect("write good.asc");
        std::fs::write(dir.join("bad.asc"), "not an asc file").expect("write bad.asc");

        let db_path = temp_db_path_for_test("parse_fail");
        let db = open_temp_db(&db_path);

        let mut progress_calls = 0usize;
        let summary = import_path(&db, &dir, false, |_done, _total| {
            progress_calls += 1;
        });

        assert!(summary.contains("Imported 1"), "summary was: {summary}");
        assert!(summary.contains("1 skipped"), "summary was: {summary}");
        assert!(summary.contains("bad.asc"), "summary was: {summary}");
        assert_eq!(
            progress_calls, 2,
            "on_progress must still fire once per candidate file, good AND bad"
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&db_path);
    }

    /// FEATURE 3's ordering contract at the `import_path` level: with `recurse:
    /// false`, a file that only exists one level down is invisible (matching the old,
    /// always-flat behaviour); with `recurse: true`, the exact same folder yields it
    /// too.
    #[test]
    fn import_path_recurses_into_subfolders_only_when_requested() {
        let dir = temp_dir_for_test("recurse");
        std::fs::write(dir.join("top.asc"), VALID_ASC).expect("write top.asc");
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).expect("create subfolder");
        std::fs::write(sub.join("nested.asc"), VALID_ASC).expect("write nested.asc");

        let flat_db_path = temp_db_path_for_test("recurse_flat");
        let flat_db = open_temp_db(&flat_db_path);
        let flat_summary = import_path(&flat_db, &dir, false, |_, _| {});
        assert!(
            flat_summary.contains("Imported 1"),
            "non-recursive import must see only top.asc: {flat_summary}"
        );

        let deep_db_path = temp_db_path_for_test("recurse_deep");
        let deep_db = open_temp_db(&deep_db_path);
        let deep_summary = import_path(&deep_db, &dir, true, |_, _| {});
        assert!(
            deep_summary.contains("Imported 2"),
            "recursive import must see both top.asc and sub/nested.asc: {deep_summary}"
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&flat_db_path);
        let _ = std::fs::remove_file(&deep_db_path);
    }

    /// The depth backstop in [`collect_asc_files_recursive`], exercised directly with
    /// a small `max_depth` rather than building 32 real nested folders to hit
    /// [`MAX_RECURSE_DEPTH`].
    #[test]
    fn collect_asc_files_recursive_stops_at_max_depth() {
        let root = temp_dir_for_test("depth");
        let level1 = root.join("level1");
        let level2 = level1.join("level2");
        std::fs::create_dir_all(&level2).expect("create nested folders");
        std::fs::write(level1.join("shallow.asc"), VALID_ASC).expect("write shallow.asc");
        std::fs::write(level2.join("deep.asc"), VALID_ASC).expect("write deep.asc");

        let mut visited = HashSet::new();
        let mut out = Vec::new();
        // Entering at depth 1 with max_depth 1: `level1` itself is walked (its own
        // depth is within budget), but `level2` one level further down is not.
        collect_asc_files_recursive(&level1, true, 1, 1, &mut visited, &mut out);

        assert_eq!(
            out.len(),
            1,
            "only the depth-1 file should be found, not the depth-2 one: {out:?}"
        );
        assert!(out[0].ends_with("shallow.asc"));

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The symlink-loop backstop in [`collect_asc_files_recursive`]: a directory
    /// symlink pointing back to an ancestor must not be followed forever, and must
    /// not make `real.asc` (reachable both directly and through the loop) show up
    /// more than once. Best-effort -- creating a directory symlink/junction on
    /// Windows normally needs Developer Mode or an elevated process, so this skips
    /// itself (rather than failing) wherever that isn't available, same tolerance
    /// this crate's own real-catalogue perf probe gives a missing prerequisite.
    #[test]
    fn collect_asc_files_recursive_guards_against_a_symlink_loop() {
        let root = temp_dir_for_test("symlink_loop");
        std::fs::write(root.join("real.asc"), VALID_ASC).expect("write real.asc");
        let link_path = root.join("loop_back");

        #[cfg(windows)]
        let symlink_result = std::os::windows::fs::symlink_dir(&root, &link_path);
        #[cfg(not(windows))]
        let symlink_result = std::os::unix::fs::symlink(&root, &link_path);

        if let Err(e) = symlink_result {
            eprintln!(
                "skipping collect_asc_files_recursive_guards_against_a_symlink_loop: \
                 could not create a directory symlink on this machine ({e}) -- \
                 likely needs Developer Mode or an elevated process on Windows"
            );
            let _ = std::fs::remove_dir_all(&root);
            return;
        }

        let mut visited = HashSet::new();
        let mut out = Vec::new();
        collect_asc_files_recursive(&root, true, 0, MAX_RECURSE_DEPTH, &mut visited, &mut out);

        assert_eq!(
            out.len(),
            1,
            "the symlink loop must not cause real.asc to be found more than once: {out:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Manual perf probe for FEATURE 2's "per-file step indicator" question: how long
    /// a single file's parse + geometry actually takes, to judge whether "parse ->
    /// geometry -> write" sub-steps would be visible to a human (roughly 100ms is the
    /// usual perceptible threshold) or just UI noise. `#[ignore]`d for the same reason
    /// as `perf_probe_refresh_after_library_change_cost` above: a timing measurement,
    /// not a correctness test, doesn't belong in a normal CI run. Run explicitly with
    /// `cargo test -p diagram-gui -- --ignored perf_probe --nocapture`.
    #[test]
    #[ignore = "manual perf probe, not for CI"]
    fn perf_probe_single_file_parse_and_measure_cost() {
        const ITERATIONS: u32 = 500;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            let mut parsed = local::import_asc("trichecker.asc", VALID_ASC).expect("valid .asc");
            apply_measured_metadata(&mut parsed.detail);
        }
        let elapsed = t0.elapsed();
        eprintln!(
            "parse + apply_measured_metadata: {elapsed:?} total over {ITERATIONS} iterations, \
             {:?} average",
            elapsed / ITERATIONS
        );
    }
}

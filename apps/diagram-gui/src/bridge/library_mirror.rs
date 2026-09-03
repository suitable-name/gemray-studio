//! Pull-mirror sync: copies a remote `gemray-worker`'s design library into the
//! local `diagram_catalog` database so it's available offline.
//!
//! Named `library_mirror`, not `sync`, deliberately: "sync" invites the reading that
//! this fetches designs from somewhere public. It does not. This is
//! machine-to-machine mirroring of the user's OWN library, over the
//! already-authenticated mutual-TLS connection `gemray-net`/`bridge::enroll` set up.
//! Nothing here fetches a URL, parses HTML, or runs OCR.
//!
//! # The safety rule, and it is the whole point of this module
//!
//! **This sync is additive and update-only. It never deletes a local design,
//! attachment, or custom material.** A local design's identity is its
//! `diagram_entries.url` (see [`diagram_catalog::db::sqlite::Database::save_diagram_entry`]'s
//! own doc comment: that's already the column's `UNIQUE` constraint, and already what
//! `diagram_catalog::local::import_asc` relies on -- a locally-imported `.asc` file gets
//! a synthetic `local://<file name>` URL specifically so it can never collide with a
//! remote page URL). This module only ever touches a `diagram_entries` row whose `url`
//! a remote [`DesignSummary`]/[`DesignRecord`] actually names -- a `local://...` row is
//! never named by anything a remote worker returns, so it is structurally unreachable by
//! every code path here, not just conventionally avoided. There is no "delete anything
//! local sync didn't just see" step anywhere in [`run_mirror_sync`] -- unlike, say, a
//! naive rsync-style mirror, "make local match remote" is never attempted; only "pull
//! what remote has" is. See `tests::a_local_only_design_survives_a_mirror_sync` for the
//! test this whole module doc comment is really about.
//!
//! # Identity: `url`, reused rather than reinvented
//!
//! A remote [`DesignSummary`]/[`DesignRecord`]'s `url` is the SAME field
//! `diagram_entries.url`'s `UNIQUE` constraint already uses to decide "is this a design
//! already in the database" (see `Database::save_diagram_entry`, which this module calls
//! unchanged, exactly the way a local `.asc` import already does). No new identity
//! scheme was invented: `diagram-catalog` already has cross-source duplicate detection
//! (`Database::find_cross_source_duplicates`, `crate::model::dedup`) for the DIFFERENT
//! problem of two DIFFERENT sources describing the same physical design under two
//! DIFFERENT urls -- that module's own doc comment is explicit that it only detects and
//! surfaces such collisions for a human to review, never merges automatically, because
//! collapsing two genuinely different designs that happen to share a title would
//! silently destroy data. Reusing `url`-equality here (not title/designer matching) for
//! "is this the SAME sync target as last time" is deliberately the narrower, safer
//! question: it answers "would syncing this design again touch the same local row",
//! never "is this possibly the same design as some unrelated other row" -- that broader
//! judgement call stays exactly where it already lived, entirely unautomated.
//!
//! # Staleness: the content hash, two-tier (see `gemray_net::library`'s own doc comment)
//!
//! [`crate::model::mirror::MirrorState`] (`diagram_catalog`'s additive
//! `library_mirror_state` table) remembers, per `url`, the last
//! [`DesignSummary::version`]/[`DesignRecord::version`] this database saw. Every sync
//! first enumerates the WHOLE remote catalogue via [`LibraryRequest::SearchPage`] (see
//! the "Pagination" section below) -- one or more round trips, each listing up to a
//! page's worth of designs' current summary hashes -- then checks each summary's hash
//! against the locally-remembered one and skips a design entirely -- no `FetchDesign`,
//! no attachment fetches, no local write at all -- when it hasn't moved. Pagination
//! composes with this cleanly because it only changes how the summaries are FETCHED,
//! never what's done with each one afterward: [`run_mirror_sync`]'s per-design loop
//! (hash check, skip-or-sync, progress callback) runs over the concatenation of every
//! page exactly as it used to run over one `Search` reply's list, oblivious to how many
//! round trips it took to assemble. **A no-op second sync of an unchanged catalogue
//! costs exactly one `SearchPage` request per page and one `library_mirror_state` read
//! per already-known design; zero `FetchDesign` or `FetchAttachment` requests.** See
//! this crate's top-level task report for the measured cost against a real
//! multi-thousand-design catalogue.
//!
//! # Pagination: a keyset cursor, walked to exhaustion before syncing begins
//!
//! [`run_mirror_sync`] enumerates the remote catalogue by looping
//! [`LibraryRequest::SearchPage`] -- `cursor: None` for the first call, then
//! `cursor: Some(id)` from each reply's [`LibraryResponse::SearchResultsPage::next_cursor`]
//! -- until a reply comes back with `next_cursor: None`, concatenating every page's
//! [`DesignSummary`] list before the per-design sync loop (hash check, skip-or-sync)
//! begins. `gemray_net::library`'s own module docs cover why keyset (`id > cursor`)
//! rather than `OFFSET`-based paging: `diagram_entries.id` is an `INTEGER PRIMARY KEY
//! AUTOINCREMENT` -- unique and strictly increasing, so the ordering it's paged over
//! has no ties -- and a keyset walk can't skip or duplicate a row when another design is
//! inserted server-side mid-walk, unlike `OFFSET`, which would.
//!
//! This replaces this module's original Phase-1 behavior, which sent a single, capped
//! `LibraryRequest::Search` and so could not see the 1001st-and-later design (by entry
//! id) in a catalogue larger than that cap -- see
//! `tests::a_multi_page_mirror_reaches_designs_beyond_the_first_page` for the test
//! proving a catalogue spanning several pages is now fully reachable.
//!
//! # Attachments: fetched eagerly (up to a size cap), not left for later
//!
//! Unlike interactive browsing (`bridge::library_client`, where an attachment is fetched
//! lazily, only when a user actually opens it), a mirror sync's whole purpose is OFFLINE
//! availability -- a design whose attachment bytes were never pulled is not actually
//! mirrored, just its metadata is. So [`run_mirror_sync`] fetches every attachment's
//! content for a design it decides to save, EXCEPT one whose advertised
//! [`AttachedFileMeta::size`] exceeds [`MirrorOptions::max_attachment_bytes`] (a generous
//! default -- see that field's own doc comment) -- a safety net against one pathological
//! file ballooning a sync, not a general "skip attachments" switch. A skipped
//! attachment does not fail the whole design: everything else about it (entry, detail,
//! every other attachment) is still saved; only that one file's bytes are left for a
//! future sync (if the remote ever serves a smaller version) or the interactive
//! `library_client` lazy-fetch path.
//!
//! # Cancellation leaves the database consistent
//!
//! [`run_mirror_sync`]'s loop checks `cancel` exactly once per design, strictly BEFORE
//! that design's own network calls or local write begin -- never in the middle of one.
//! Once a design's processing starts, it always runs to completion (all its attachment
//! fetches, then one `save_diagram_entry` + `save_diagram_detail` call, in that order,
//! with attachment bytes fetched over the network FIRST so nothing is written locally
//! until every byte this design needs is already in hand) before the next cancellation
//! check. `save_diagram_detail` itself already wraps its own entry/detail/angle-
//! settings/attachment writes in one SQLite transaction (see that method's own doc
//! comment) that rolls back whole on any error, so even a mid-write I/O failure -- a
//! different risk than cooperative cancellation, and not new here -- can never leave a
//! design half-written either.
//!
//! **Invariant:** at any point this sync is interrupted, the local database reflects
//! some prefix of fully-completed per-design syncs, plus whatever was there before the
//! sync started. No design is ever left mid-write. See
//! `tests::cancelling_mid_sync_leaves_earlier_designs_committed_and_the_rest_untouched`.
//! This holds regardless of how many `SearchPage` round trips it took to enumerate the
//! catalogue in the first place: cancellation is only ever checked, and only ever takes
//! effect, between two designs in the per-design loop below -- never while the
//! enumeration itself (the pagination loop this module's own "Pagination" section
//! describes, above) is still running, which is deliberately not cancellable
//! mid-enumeration: it does no local writes at all, so there is nothing for
//! cancellation to protect against there, and it fails outright (`MirrorOutcome::Failed`,
//! nothing written) rather than partially, on any request error, exactly as the
//! original single-`Search` enumeration already did.

use crate::{
    bridge::library_client::{self, LibrarySession},
    settings::WorkerSettings,
};
use diagram_catalog::{
    db::sqlite::Database,
    model::{
        angle::AngleSetting, detail::FacetDiagramDetail, entry::FacetDiagramEntry,
        file::AttachedFile, mirror::MirrorState,
    },
};
use gemray_net::library::{AngleSettingWire, LibraryRequest, LibraryResponse, RangeFilterWire};
use slint::{ComponentHandle, Weak};
use std::{
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

/// The `source_id` a mirror sync attributes every design it saves/updates to -- distinct
/// per configured worker (its address), so `Database::find_cross_source_duplicates`
/// naturally treats two different remote libraries (or a remote library vs. a local
/// import / the legacy scraped catalogue) as different sources, exactly as it already
/// does for the local-import/legacy-scrape distinction.
#[must_use]
pub fn mirror_source_id(worker: &WorkerSettings) -> String {
    format!("remote-library:{}", worker.address)
}

/// Tuning knobs for one mirror sync. `Default` is the sensible "just mirror everything
/// reasonable" choice a UI trigger with no advanced options needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MirrorOptions {
    /// An attachment whose advertised [`gemray_net::library::AttachedFileMeta::size`]
    /// exceeds this is skipped (the design itself is still saved without it) -- a safety
    /// net against one pathological file, not a general attachment-skipping switch. See
    /// the module doc comment's "Attachments" section. `50 MiB` comfortably exceeds any
    /// real competition-results PDF or diagram image in this catalogue (the module doc
    /// comment on `gemray_net::library` itself says "competition PDFs and gem diagrams,
    /// not video") while still catching a genuinely anomalous file.
    pub max_attachment_bytes: u64,
}

impl Default for MirrorOptions {
    fn default() -> Self {
        Self {
            max_attachment_bytes: 50 * 1024 * 1024,
        }
    }
}

/// Running/final tallies for one mirror sync -- both [`MirrorProgress::counts`] (a live
/// snapshot, after each design) and [`MirrorOutcome`]'s completed/cancelled payload (the
/// final snapshot) use this same shape.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MirrorCounts {
    /// Total designs the remote catalogue reported, across every `SearchPage` page --
    /// see the module doc comment's "Pagination" section. Unlike this module's
    /// original single-`Search` enumeration, this no longer undercounts a catalogue
    /// past one page's cap.
    pub total_found: usize,
    pub new_count: usize,
    pub updated_count: usize,
    /// Skipped without a `FetchDesign` at all -- summary hash unchanged since the last
    /// sync. The whole point of the two-tier content-hash check; see the module doc
    /// comment's "Staleness" section.
    pub skipped_unchanged: usize,
    /// A design whose fetch or local save failed for any reason (network error,
    /// database error, or it vanished server-side between `Search` and `FetchDesign`).
    /// Left exactly as it was locally before this sync (if it existed at all) -- never
    /// marked as synced, so the next sync retries it.
    pub failed: usize,
    pub attachments_fetched: usize,
    /// An attachment skipped for exceeding [`MirrorOptions::max_attachment_bytes`] --
    /// its design was still saved, just without this one file's bytes.
    pub attachments_skipped_too_large: usize,
    pub attachment_bytes_fetched: u64,
}

/// One progress update, delivered after each design this sync examines (whether it was
/// skipped, saved, or failed).
#[derive(Debug, Clone)]
pub struct MirrorProgress {
    /// How many of `counts.total_found` designs have been examined so far, including
    /// this one.
    pub processed: usize,
    pub counts: MirrorCounts,
    pub current_title: String,
}

/// How [`run_mirror_sync`] ended.
#[derive(Debug, Clone)]
pub enum MirrorOutcome {
    Completed(MirrorCounts),
    /// Stopped early via [`MirrorHandle::cancel`] -- `MirrorCounts` reflects every
    /// design fully processed before the cancellation was observed (see the module doc
    /// comment's "Cancellation" section for why that's always a clean prefix, never a
    /// partial design).
    Cancelled(MirrorCounts),
    /// Could not even enumerate the remote catalogue (a `SearchPage` request failed,
    /// on the first page or any later one) -- nothing was written locally at all.
    Failed(String),
}

/// Handle returned by [`spawn_mirror_sync`]. Cancelling is cooperative -- see the module
/// doc comment's "Cancellation" section for exactly when it takes effect -- matching
/// `bridge::export_thread::ExportHandle`'s existing shape in this crate.
pub struct MirrorHandle {
    cancel: Arc<AtomicBool>,
}

impl MirrorHandle {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// A source of [`LibraryResponse`]s for exactly one [`LibraryRequest`] at a time.
///
/// Exists so [`run_mirror_sync`] -- the actual mirror algorithm -- can be unit-tested
/// against a scripted in-memory fake (see `tests::FakeTransport`) with no live socket or
/// TLS handshake, the same reason `gemray_net::client` is generic over `Read`/`Write`
/// rather than a concrete `TcpStream`. [`LibrarySession`]'s implementation is the one
/// [`spawn_mirror_sync`] actually uses in production: one connect+handshake for the whole
/// sync, reused across every request, reconnecting-and-retrying once per request should
/// the held connection drop -- see that type's own doc comment for the full policy.
/// [`WorkerSettings`]'s implementation (one real connect+handshake+request+response PER
/// call, via `bridge::library_client::request`) is no longer wired into
/// [`spawn_mirror_sync`], but stays available -- it is, after all, one valid way to
/// satisfy this trait, and matches how the interactive remote-browse path
/// (`bridge::library_source`) already calls the same one-shot `request` directly.
pub trait LibraryTransport {
    /// # Errors
    ///
    /// Whatever the underlying transport failed with -- see
    /// `bridge::library_client::LibraryClientError`'s variants.
    fn request(
        &self,
        req: &LibraryRequest,
    ) -> Result<LibraryResponse, library_client::LibraryClientError>;
}

impl LibraryTransport for WorkerSettings {
    fn request(
        &self,
        req: &LibraryRequest,
    ) -> Result<LibraryResponse, library_client::LibraryClientError> {
        library_client::request(self, req)
    }
}

impl LibraryTransport for LibrarySession {
    fn request(
        &self,
        req: &LibraryRequest,
    ) -> Result<LibraryResponse, library_client::LibraryClientError> {
        // Resolves to `LibrarySession`'s own inherent `request` method (inherent methods
        // take priority over a trait method of the same name during resolution) -- the
        // held-connection, reconnect-on-drop implementation; see that type's doc comment.
        self.request(req)
    }
}

/// Spawns the mirror-sync worker thread against `worker`'s design library, writing into
/// `db`. `on_progress` is invoked on the UI event loop after each design examined;
/// `on_done` is invoked once, exactly once, when the sync completes, is cancelled, or
/// fails outright. Follows `bridge::export_thread::spawn_export`'s exact pattern (a
/// `thread::spawn` worker, an `Arc<AtomicBool>` cancel flag, results marshalled back via
/// `Weak::upgrade_in_event_loop`).
///
/// `db` is locked only for the duration of each individual database call inside the
/// sync loop (see [`run_mirror_sync`]), never for the whole sync -- so the local library
/// UI (search, detail, import) stays responsive on the SAME database while a
/// multi-minute sync runs in the background.
///
/// Drives the sync against one [`LibrarySession`] -- built here, from `worker`, and held
/// for the whole sync -- rather than reconnecting per request, the whole point of this
/// module's held-connection task; see [`LibrarySession`]'s own doc comment for what that
/// buys (one handshake instead of thousands for a real catalogue) and how it behaves when
/// that held connection drops mid-sync.
pub fn spawn_mirror_sync<T, P, D>(
    ui_weak: Weak<T>,
    db: Arc<Mutex<Database>>,
    worker: WorkerSettings,
    options: MirrorOptions,
    on_progress: P,
    on_done: D,
) -> MirrorHandle
where
    T: ComponentHandle + 'static,
    P: Fn(&T, MirrorProgress) + Send + 'static + Clone,
    D: Fn(&T, MirrorOutcome) + Send + 'static,
{
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_worker = cancel.clone();
    let ui_weak_done = ui_weak.clone();

    thread::spawn(move || {
        let source_id = mirror_source_id(&worker);
        let session = LibrarySession::new(worker);
        let progress_ui_weak = ui_weak;
        let outcome = run_mirror_sync(
            &db,
            &session,
            &source_id,
            options,
            &cancel_worker,
            move |progress| {
                let on_progress = on_progress.clone();
                let _ =
                    progress_ui_weak.upgrade_in_event_loop(move |ui| on_progress(&ui, progress));
            },
        );
        let _ = ui_weak_done.upgrade_in_event_loop(move |ui| on_done(&ui, outcome));
    });

    MirrorHandle { cancel }
}

/// Walks [`LibraryRequest::SearchPage`] to exhaustion, concatenating every page's
/// [`DesignSummary`] list into one `Vec` -- the WHOLE remote catalogue matching the
/// (currently always empty/unfiltered) query, not just its first page. See the module
/// doc comment's "Pagination" section for the keyset-cursor scheme this implements
/// (`cursor: None`, then `cursor: Some(next_cursor)` from each reply, until a reply's
/// `next_cursor` is `None`) and why it never needs to skip or re-see a row even if
/// designs are added to the remote catalogue while this runs.
///
/// Returns `Err(MirrorOutcome::Failed(..))` -- ready to propagate straight out of
/// [`run_mirror_sync`] -- on the first request that fails outright or replies with
/// anything other than [`LibraryResponse::SearchResultsPage`]; nothing is written
/// locally by this function (it only ever reads from `transport`), so a failure here
/// leaves the local database exactly as it was, matching this module's existing
/// "nothing written until enumeration succeeds" behavior.
fn enumerate_remote_catalogue(
    transport: &impl LibraryTransport,
) -> Result<Vec<gemray_net::library::DesignSummary>, MirrorOutcome> {
    let mut summaries = Vec::new();
    let mut cursor = None;
    loop {
        let request = LibraryRequest::SearchPage {
            query: String::new(),
            shape_filter: "All".to_string(),
            gear_filter: "All".to_string(),
            range: RangeFilterWire::default(),
            cursor,
        };
        match transport.request(&request) {
            Ok(LibraryResponse::SearchResultsPage {
                results,
                next_cursor,
            }) => {
                summaries.extend(results);
                match next_cursor {
                    Some(c) => cursor = Some(c),
                    None => return Ok(summaries),
                }
            }
            Ok(other) => {
                return Err(MirrorOutcome::Failed(format!(
                    "remote worker replied to SearchPage with an unexpected message: {other:?}"
                )));
            }
            Err(e) => {
                return Err(MirrorOutcome::Failed(format!(
                    "could not list the remote library: {e}"
                )));
            }
        }
    }
}

/// The mirror algorithm itself -- generic over [`LibraryTransport`] so it's testable
/// without a socket (see that trait's doc comment). See the module doc comment for the
/// full design: additive/update-only, `url`-keyed identity, two-tier content-hash
/// skipping, eager (capped) attachment fetching, and per-design cancellation safety.
#[must_use]
pub fn run_mirror_sync(
    db: &Arc<Mutex<Database>>,
    transport: &impl LibraryTransport,
    source_id: &str,
    options: MirrorOptions,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(MirrorProgress),
) -> MirrorOutcome {
    let summaries = match enumerate_remote_catalogue(transport) {
        Ok(list) => list,
        Err(outcome) => return outcome,
    };

    let mut counts = MirrorCounts {
        total_found: summaries.len(),
        ..MirrorCounts::default()
    };

    for (processed, summary) in summaries.into_iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return MirrorOutcome::Cancelled(counts);
        }

        let existing_state = {
            let db = db.lock().unwrap_or_else(PoisonError::into_inner);
            db.get_mirror_state(&summary.url).ok().flatten()
        };

        let unchanged = existing_state
            .as_ref()
            .is_some_and(|state| state.summary_version == summary.version);
        if unchanged {
            counts.skipped_unchanged += 1;
            on_progress(MirrorProgress {
                processed: processed + 1,
                counts,
                current_title: summary.title.clone(),
            });
            continue;
        }

        let is_new = existing_state.is_none();
        match sync_one_design(db, transport, source_id, options, &summary, &mut counts) {
            Ok(()) => {
                if is_new {
                    counts.new_count += 1;
                } else {
                    counts.updated_count += 1;
                }
            }
            Err(()) => counts.failed += 1,
        }

        on_progress(MirrorProgress {
            processed: processed + 1,
            counts,
            current_title: summary.title.clone(),
        });
    }

    MirrorOutcome::Completed(counts)
}

/// Fetches and saves exactly one design: `FetchDesign`, then every non-oversized
/// attachment's bytes, then one local `save_diagram_entry` + `save_diagram_detail` +
/// `upsert_mirror_state` -- all network I/O happens before the first local write, so a
/// failure partway through never touches the database (see the module doc comment's
/// "Cancellation" section, which this same all-network-then-all-local ordering also
/// backs).
///
/// `Err(())` on any failure (fetch, or local save) -- the caller only needs to know
/// pass/fail to update [`MirrorCounts::failed`]; the specific reason isn't surfaced
/// per-design (this sync covers up to a thousand designs at once -- see the module doc
/// comment's protocol-limitation note -- so a per-design error UI would be noise; a
/// failed design is simply retried on the next sync, same as one skipped for looking
/// unchanged is not).
fn sync_one_design(
    db: &Arc<Mutex<Database>>,
    transport: &impl LibraryTransport,
    source_id: &str,
    options: MirrorOptions,
    summary: &gemray_net::library::DesignSummary,
    counts: &mut MirrorCounts,
) -> Result<(), ()> {
    let design = match transport.request(&LibraryRequest::FetchDesign {
        entry_id: summary.entry_id,
    }) {
        Ok(LibraryResponse::Design(d)) => *d,
        _ => return Err(()),
    };

    let mut attached_files = Vec::with_capacity(design.attachments.len());
    for meta in &design.attachments {
        if meta.size > options.max_attachment_bytes {
            counts.attachments_skipped_too_large += 1;
            continue;
        }
        match transport.request(&LibraryRequest::FetchAttachment {
            attachment_id: meta.id,
        }) {
            Ok(LibraryResponse::Attachment { name, content }) => {
                counts.attachment_bytes_fetched += content.len() as u64;
                counts.attachments_fetched += 1;
                attached_files.push(AttachedFile {
                    name,
                    url: meta.url.clone(),
                    content,
                });
            }
            Ok(LibraryResponse::NotFound) => {
                // Vanished server-side between FetchDesign and FetchAttachment --
                // save the rest of the design without this one file rather than
                // failing it outright.
            }
            _ => return Err(()),
        }
    }

    let entry = FacetDiagramEntry {
        title: design.title.clone(),
        url: design.url.clone(),
        design_id: design.design_id.clone().unwrap_or_default(),
    };
    let detail = FacetDiagramDetail {
        page_url: design.page_url.clone(),
        diagram_image_name: design.diagram_image_name.clone(),
        diagram_image_data: design.diagram_image_data.clone(),
        angle_settings_table: design.angle_settings.iter().map(to_angle_setting).collect(),
        attached_files,
        competition_diagram: design.competition_diagram.clone(),
        lw_ratio: design.lw_ratio.clone(),
        refractive_index: design.refractive_index.clone(),
        index_gear: design.index_gear.clone(),
        volume: design.volume.clone(),
        facets_count: design.facets_count.clone(),
        shape: design.shape.clone(),
        designer_info: design.designer_info.clone(),
        ..FacetDiagramDetail::default()
    };

    let local_entry_id = {
        let db = db.lock().unwrap_or_else(PoisonError::into_inner);
        let Ok(id) = db.save_diagram_entry(&entry, source_id) else {
            return Err(());
        };
        if db.save_diagram_detail(&detail, id).is_err() {
            return Err(());
        }
        // Only recorded once the local write above actually succeeded -- a design
        // that failed to save is never marked as synced, so the next sync retries
        // it rather than silently treating a failed write as done. See this
        // function's own doc comment.
        let _ = db.upsert_mirror_state(&MirrorState {
            url: design.url.clone(),
            source_id: source_id.to_string(),
            summary_version: summary.version,
            design_version: design.version,
        });
        id
    };
    let _ = local_entry_id;

    Ok(())
}

fn to_angle_setting(a: &AngleSettingWire) -> AngleSetting {
    AngleSetting {
        order_index: a.order_index,
        facet: a.facet.clone(),
        angle: a.angle.clone(),
        index: a.index.clone(),
        notes: a.notes.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemray_net::library::{AttachedFileMeta, DesignRecord, DesignSummary};
    use std::{cell::RefCell, collections::HashMap, sync::atomic::AtomicUsize};

    /// A scripted, in-memory [`LibraryTransport`] -- no socket, no TLS, no real
    /// `gemray-worker`. Counts how many times each request kind is made so tests can
    /// pin the "no-op second sync costs nothing but one `SearchPage` per page" claim
    /// precisely, and pages `search_result` at [`Self::page_size`] rows per
    /// `SearchPage` reply -- a scripted stand-in for `apps/gemray-worker`'s own
    /// `SEARCH_RESULT_CAP`-per-page behavior, small enough in tests to exercise a real
    /// multi-page walk without seeding a thousand-plus rows.
    struct FakeTransport {
        search_result: Vec<DesignSummary>,
        /// How many `search_result` rows one `SearchPage` reply hands back. Defaults
        /// (via [`Self::new`]) to `usize::MAX` -- one page always covers the whole
        /// `search_result` list -- so every pre-existing single-page test keeps making
        /// exactly one `SearchPage` call, unchanged; [`Self::with_page_size`] shrinks it
        /// for a test that specifically wants to exercise pagination.
        page_size: usize,
        designs: HashMap<i64, DesignRecord>,
        attachments: HashMap<i64, (String, Vec<u8>)>,
        search_calls: AtomicUsize,
        fetch_design_calls: AtomicUsize,
        fetch_attachment_calls: AtomicUsize,
        /// Set of entry ids [`LibraryRequest::FetchDesign`] was actually called for --
        /// lets a test assert WHICH designs were (not) re-fetched, not just a count.
        fetched_entry_ids: RefCell<Vec<i64>>,
        /// When `Some(id)`, [`LibraryRequest::FetchDesign`] for that entry id returns a
        /// transport-level [`library_client::LibraryClientError::Client`] error instead
        /// of a normal reply -- standing in for a `LibrarySession` whose held connection
        /// dropped mid-sync AND whose own reconnect attempt then also failed (see
        /// `library_client::request_with_reconnect`'s "propagates the error when
        /// reconnecting also fails" case): a genuine transport failure reaching
        /// [`run_mirror_sync`], not a logical [`LibraryResponse::NotFound`]. Set via
        /// [`Self::with_fetch_design_failure`].
        fail_fetch_design_for: Option<i64>,
    }

    impl FakeTransport {
        fn new(summaries: Vec<DesignSummary>, designs: Vec<DesignRecord>) -> Self {
            Self {
                search_result: summaries,
                page_size: usize::MAX,
                designs: designs.into_iter().map(|d| (d.entry_id, d)).collect(),
                attachments: HashMap::new(),
                search_calls: AtomicUsize::new(0),
                fetch_design_calls: AtomicUsize::new(0),
                fetch_attachment_calls: AtomicUsize::new(0),
                fetched_entry_ids: RefCell::new(Vec::new()),
                fail_fetch_design_for: None,
            }
        }

        fn with_attachment(mut self, id: i64, name: &str, content: Vec<u8>) -> Self {
            self.attachments.insert(id, (name.to_string(), content));
            self
        }

        /// Shrinks how many rows one `SearchPage` reply hands back, forcing
        /// [`enumerate_remote_catalogue`] into more than one round trip for a
        /// `search_result` longer than `page_size`. See [`Self::page_size`]'s own doc
        /// comment.
        fn with_page_size(mut self, page_size: usize) -> Self {
            self.page_size = page_size;
            self
        }

        /// Makes [`LibraryRequest::FetchDesign`] for `entry_id` fail with a
        /// transport-level error -- see [`Self::fail_fetch_design_for`]'s own doc
        /// comment for what this simulates.
        fn with_fetch_design_failure(mut self, entry_id: i64) -> Self {
            self.fail_fetch_design_for = Some(entry_id);
            self
        }
    }

    impl LibraryTransport for FakeTransport {
        fn request(
            &self,
            req: &LibraryRequest,
        ) -> Result<LibraryResponse, library_client::LibraryClientError> {
            match req {
                LibraryRequest::Search { .. } => {
                    unreachable!(
                        "mirror sync always pages via SearchPage now, never sends Search directly"
                    )
                }
                LibraryRequest::SearchPage { cursor, .. } => {
                    self.search_calls.fetch_add(1, Ordering::Relaxed);
                    let start = cursor.map_or(0, |after| {
                        self.search_result
                            .iter()
                            .position(|s| s.entry_id == after)
                            .map_or(self.search_result.len(), |i| i + 1)
                    });
                    let end = start
                        .saturating_add(self.page_size)
                        .min(self.search_result.len());
                    let page = self.search_result[start..end].to_vec();
                    // Mirrors `apps/gemray-worker`'s own rule (see
                    // `serve::library::search_page`): a full page means "there may be
                    // more", a short page means "this was the last one".
                    let next_cursor = if page.len() == self.page_size {
                        page.last().map(|s| s.entry_id)
                    } else {
                        None
                    };
                    Ok(LibraryResponse::SearchResultsPage {
                        results: page,
                        next_cursor,
                    })
                }
                LibraryRequest::FetchDesign { entry_id } => {
                    self.fetch_design_calls.fetch_add(1, Ordering::Relaxed);
                    self.fetched_entry_ids.borrow_mut().push(*entry_id);
                    if self.fail_fetch_design_for == Some(*entry_id) {
                        return Err(library_client::LibraryClientError::Client(
                            gemray_net::client::ClientError::Net(
                                gemray_net::messages::NetError::Framing(
                                    gemray_net::framing::FramingError::Io(std::io::Error::new(
                                        std::io::ErrorKind::ConnectionReset,
                                        "connection dropped and could not be re-established",
                                    )),
                                ),
                            ),
                        ));
                    }
                    Ok(self
                        .designs
                        .get(entry_id)
                        .map_or(LibraryResponse::NotFound, |d| {
                            LibraryResponse::Design(Box::new(d.clone()))
                        }))
                }
                LibraryRequest::FetchAttachment { attachment_id } => {
                    self.fetch_attachment_calls.fetch_add(1, Ordering::Relaxed);
                    match self.attachments.get(attachment_id) {
                        Some((name, content)) => Ok(LibraryResponse::Attachment {
                            name: name.clone(),
                            content: content.clone(),
                        }),
                        None => Ok(LibraryResponse::NotFound),
                    }
                }
                LibraryRequest::FilterOptions => unreachable!("mirror sync never sends this"),
            }
        }
    }

    fn temp_db() -> (Arc<Mutex<Database>>, std::path::PathBuf) {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "diagram-gui-library-mirror-test-{}-{n}.sqlite",
            std::process::id()
        ));
        let db = Database::new(Some(path.to_str().unwrap())).unwrap();
        (Arc::new(Mutex::new(db)), path)
    }

    fn design_record(entry_id: i64, title: &str, url: &str, extra_field: &str) -> DesignRecord {
        let mut record = DesignRecord {
            entry_id,
            title: title.to_string(),
            url: url.to_string(),
            design_id: Some(format!("D-{entry_id}")),
            page_url: url.to_string(),
            diagram_image_name: None,
            diagram_image_data: None,
            competition_diagram: None,
            lw_ratio: Some(extra_field.to_string()),
            refractive_index: Some("1.72".to_string()),
            index_gear: Some("96".to_string()),
            volume: Some("0.65".to_string()),
            facets_count: Some("57".to_string()),
            shape: Some("Round".to_string()),
            designer_info: Some("Capps, Jerry".to_string()),
            angle_settings: vec![AngleSettingWire {
                order_index: 0,
                facet: "P1".to_string(),
                angle: "41.0".to_string(),
                index: "96".to_string(),
                notes: String::new(),
            }],
            attachments: Vec::new(),
            version: [0u8; 32],
        };
        record.version = content_hash(&[title.as_bytes(), url.as_bytes(), extra_field.as_bytes()]);
        record
    }

    fn design_summary(entry_id: i64, title: &str, url: &str, extra_field: &str) -> DesignSummary {
        DesignSummary {
            entry_id,
            title: title.to_string(),
            url: url.to_string(),
            design_id: Some(format!("D-{entry_id}")),
            shape: Some("Round".to_string()),
            index_gear: Some("96".to_string()),
            facets_count: Some("57".to_string()),
            designer_info: Some("Capps, Jerry".to_string()),
            lw_ratio: Some(extra_field.to_string()),
            refractive_index: Some("1.72".to_string()),
            volume: Some("0.65".to_string()),
            competition_diagram: None,
            version: content_hash(&[title.as_bytes(), url.as_bytes(), extra_field.as_bytes()]),
        }
    }

    /// A trivial, deterministic stand-in for the real server-side SHA-256 hash (see
    /// `apps/gemray-worker/src/serve/library/mod.rs::hash_summary`/`hash_record`) --
    /// this module never computes or checks the hash's own algorithm, only compares two
    /// hashes for equality, so any deterministic function of the "content" a test cares
    /// about is sufficient.
    fn content_hash(parts: &[&[u8]]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for p in parts {
            hasher.update(p);
        }
        hasher.finalize().into()
    }

    #[test]
    fn a_local_only_design_survives_a_mirror_sync() {
        let (db, path) = temp_db();
        // A design that exists ONLY locally (a user's own `.asc` import) -- a synthetic
        // `local://` URL, exactly `diagram_catalog::local::import_asc`'s convention.
        let local_entry_id = {
            let guard = db.lock().unwrap();
            guard
                .save_diagram_entry(
                    &FacetDiagramEntry {
                        title: "My Own Trichecker".to_string(),
                        url: "local://my_trichecker.asc".to_string(),
                        design_id: String::new(),
                    },
                    diagram_catalog::local::LOCAL_SOURCE_ID,
                )
                .unwrap()
        };
        {
            let guard = db.lock().unwrap();
            guard
                .save_diagram_detail(
                    &FacetDiagramDetail {
                        shape: Some("Trichecker".to_string()),
                        attached_files: vec![AttachedFile {
                            name: "my_trichecker.asc".to_string(),
                            url: String::new(),
                            content: b"a real user file".to_vec(),
                        }],
                        ..FacetDiagramDetail::default()
                    },
                    local_entry_id,
                )
                .unwrap();
        }

        let remote_summary = design_summary(1, "Round Brilliant", "https://example.test/1", "v1");
        let remote_design = design_record(1, "Round Brilliant", "https://example.test/1", "v1");
        let transport = FakeTransport::new(vec![remote_summary], vec![remote_design]);

        let outcome = run_mirror_sync(
            &db,
            &transport,
            "remote-library:example.test:9443",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |_| {},
        );
        assert!(matches!(outcome, MirrorOutcome::Completed(c) if c.new_count == 1));

        // The local-only design is completely untouched: still there, same title, same
        // attachment content -- sync never deleted or altered it.
        let guard = db.lock().unwrap();
        let local_full = guard.get_diagram_full(local_entry_id).unwrap().unwrap();
        assert_eq!(local_full.title, "My Own Trichecker");
        assert_eq!(local_full.url, "local://my_trichecker.asc");
        assert_eq!(local_full.attached_files.len(), 1);
        assert_eq!(local_full.attached_files[0].content, b"a real user file");
        // And the remote design was actually added alongside it -- total count is both.
        assert_eq!(guard.get_total_count().unwrap(), 2);
        drop(guard);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_new_design_is_saved_with_its_attachment() {
        let (db, path) = temp_db();
        let mut record = design_record(1, "Round Brilliant", "https://example.test/1", "v1");
        record.attachments = vec![AttachedFileMeta {
            id: 100,
            name: "schedule.pdf".to_string(),
            url: "https://example.test/schedule.pdf".to_string(),
            size: 5,
        }];
        let summary = design_summary(1, "Round Brilliant", "https://example.test/1", "v1");
        let transport = FakeTransport::new(vec![summary], vec![record]).with_attachment(
            100,
            "schedule.pdf",
            vec![1, 2, 3, 4, 5],
        );

        let outcome = run_mirror_sync(
            &db,
            &transport,
            "remote-library:w",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |_| {},
        );
        let MirrorOutcome::Completed(counts) = outcome else {
            panic!("expected Completed, got {outcome:?}");
        };
        assert_eq!(counts.new_count, 1);
        assert_eq!(counts.attachments_fetched, 1);
        assert_eq!(counts.attachment_bytes_fetched, 5);

        let guard = db.lock().unwrap();
        let full = guard
            .search_diagrams(
                "",
                "All",
                "All",
                &diagram_catalog::model::filter::RangeFilter::default(),
            )
            .unwrap();
        assert_eq!(full.len(), 1);
        let entry_id = full[0].id;
        let record = guard.get_diagram_full(entry_id).unwrap().unwrap();
        assert_eq!(record.attached_files.len(), 1);
        assert_eq!(record.attached_files[0].content, vec![1, 2, 3, 4, 5]);
        drop(guard);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_second_sync_of_an_unchanged_catalogue_skips_every_design_and_makes_no_fetch_design_calls()
    {
        let (db, path) = temp_db();
        let summary = design_summary(1, "Round Brilliant", "https://example.test/1", "v1");
        let record = design_record(1, "Round Brilliant", "https://example.test/1", "v1");
        let transport = FakeTransport::new(vec![summary], vec![record]);

        let first = run_mirror_sync(
            &db,
            &transport,
            "remote-library:w",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |_| {},
        );
        assert!(matches!(first, MirrorOutcome::Completed(c) if c.new_count == 1));
        assert_eq!(transport.fetch_design_calls.load(Ordering::Relaxed), 1);

        let second = run_mirror_sync(
            &db,
            &transport,
            "remote-library:w",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |_| {},
        );
        let MirrorOutcome::Completed(counts) = second else {
            panic!("expected Completed, got {second:?}");
        };
        assert_eq!(counts.skipped_unchanged, 1);
        assert_eq!(counts.new_count, 0);
        assert_eq!(counts.updated_count, 0);
        // The whole point: a no-op resync makes exactly one more Search and ZERO
        // FetchDesign/FetchAttachment calls beyond the first sync's.
        assert_eq!(transport.search_calls.load(Ordering::Relaxed), 2);
        assert_eq!(transport.fetch_design_calls.load(Ordering::Relaxed), 1);
        assert_eq!(transport.fetch_attachment_calls.load(Ordering::Relaxed), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_changed_design_is_refetched_and_updated_in_place() {
        let (db, path) = temp_db();
        let summary_v1 = design_summary(1, "Round Brilliant", "https://example.test/1", "v1");
        let record_v1 = design_record(1, "Round Brilliant", "https://example.test/1", "v1");
        let transport_v1 = FakeTransport::new(vec![summary_v1], vec![record_v1]);
        let _ = run_mirror_sync(
            &db,
            &transport_v1,
            "remote-library:w",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |_| {},
        );

        // The remote design's L/W ratio changed -- both its summary and design hashes
        // move (see `design_summary`/`design_record`'s shared `content_hash` input).
        let summary_v2 =
            design_summary(1, "Round Brilliant", "https://example.test/1", "v2-updated");
        let record_v2 = design_record(1, "Round Brilliant", "https://example.test/1", "v2-updated");
        let transport_v2 = FakeTransport::new(vec![summary_v2], vec![record_v2]);

        let outcome = run_mirror_sync(
            &db,
            &transport_v2,
            "remote-library:w",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |_| {},
        );
        let MirrorOutcome::Completed(counts) = outcome else {
            panic!("expected Completed, got {outcome:?}");
        };
        assert_eq!(counts.updated_count, 1);
        assert_eq!(counts.new_count, 0);

        let guard = db.lock().unwrap();
        let items = guard
            .search_diagrams(
                "",
                "All",
                "All",
                &diagram_catalog::model::filter::RangeFilter::default(),
            )
            .unwrap();
        assert_eq!(
            items.len(),
            1,
            "must update the existing row, not add a second one"
        );
        assert_eq!(items[0].lw_ratio.as_deref(), Some("v2-updated"));
        drop(guard);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cancelling_mid_sync_leaves_earlier_designs_committed_and_the_rest_untouched() {
        let (db, path) = temp_db();
        let summaries = vec![
            design_summary(1, "First", "https://example.test/1", "v1"),
            design_summary(2, "Second", "https://example.test/2", "v1"),
            design_summary(3, "Third", "https://example.test/3", "v1"),
        ];
        let designs = vec![
            design_record(1, "First", "https://example.test/1", "v1"),
            design_record(2, "Second", "https://example.test/2", "v1"),
            design_record(3, "Third", "https://example.test/3", "v1"),
        ];
        let transport = FakeTransport::new(summaries, designs);
        let cancel = AtomicBool::new(false);

        let mut processed_count = 0;
        let outcome = run_mirror_sync(
            &db,
            &transport,
            "remote-library:w",
            MirrorOptions::default(),
            &cancel,
            |progress| {
                processed_count = progress.processed;
                // Cancel right after the first design commits, before the second is
                // ever looked at.
                if progress.processed == 1 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
        );
        assert_eq!(processed_count, 1);
        let MirrorOutcome::Cancelled(counts) = outcome else {
            panic!("expected Cancelled, got {outcome:?}");
        };
        assert_eq!(counts.new_count, 1);

        let guard = db.lock().unwrap();
        // Exactly one design landed -- the second and third were never even fetched
        // (FetchDesign was called exactly once), so nothing about them exists locally,
        // not even a half-written row.
        assert_eq!(guard.get_total_count().unwrap(), 1);
        assert_eq!(transport.fetch_design_calls.load(Ordering::Relaxed), 1);
        assert_eq!(*transport.fetched_entry_ids.borrow(), vec![1]);
        drop(guard);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_oversized_attachment_is_skipped_but_the_design_is_still_saved() {
        let (db, path) = temp_db();
        let mut record = design_record(1, "Round Brilliant", "https://example.test/1", "v1");
        record.attachments = vec![
            AttachedFileMeta {
                id: 100,
                name: "huge.pdf".to_string(),
                url: String::new(),
                size: 1000,
            },
            AttachedFileMeta {
                id: 101,
                name: "small.pdf".to_string(),
                url: String::new(),
                size: 5,
            },
        ];
        let summary = design_summary(1, "Round Brilliant", "https://example.test/1", "v1");
        let transport = FakeTransport::new(vec![summary], vec![record])
            .with_attachment(100, "huge.pdf", vec![0u8; 1000])
            .with_attachment(101, "small.pdf", vec![9, 9, 9, 9, 9]);

        let options = MirrorOptions {
            max_attachment_bytes: 500,
        };
        let outcome = run_mirror_sync(
            &db,
            &transport,
            "remote-library:w",
            options,
            &AtomicBool::new(false),
            |_| {},
        );
        let MirrorOutcome::Completed(counts) = outcome else {
            panic!("expected Completed, got {outcome:?}");
        };
        assert_eq!(counts.new_count, 1, "the design itself must still be saved");
        assert_eq!(counts.attachments_fetched, 1);
        assert_eq!(counts.attachments_skipped_too_large, 1);
        assert_eq!(
            transport.fetch_attachment_calls.load(Ordering::Relaxed),
            1,
            "the oversized attachment's bytes must never even be requested"
        );

        let guard = db.lock().unwrap();
        let items = guard
            .search_diagrams(
                "",
                "All",
                "All",
                &diagram_catalog::model::filter::RangeFilter::default(),
            )
            .unwrap();
        let full = guard.get_diagram_full(items[0].id).unwrap().unwrap();
        assert_eq!(full.attached_files.len(), 1);
        assert_eq!(full.attached_files[0].name, "small.pdf");
        drop(guard);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_failed_fetch_is_counted_and_never_marked_synced() {
        let (db, path) = temp_db();
        // The summary claims entry_id 1, but the transport's `designs` map has nothing
        // for it -- simulating a design that vanished between Search and FetchDesign.
        let summary = design_summary(1, "Ghost", "https://example.test/1", "v1");
        let transport = FakeTransport::new(vec![summary], Vec::new());

        let outcome = run_mirror_sync(
            &db,
            &transport,
            "remote-library:w",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |_| {},
        );
        let MirrorOutcome::Completed(counts) = outcome else {
            panic!("expected Completed, got {outcome:?}");
        };
        assert_eq!(counts.failed, 1);
        assert_eq!(counts.new_count, 0);

        let guard = db.lock().unwrap();
        assert_eq!(guard.get_total_count().unwrap(), 0);
        assert_eq!(
            guard.get_mirror_state("https://example.test/1").unwrap(),
            None
        );
        drop(guard);
        std::fs::remove_file(&path).ok();
    }

    /// The scenario the held-connection task exists to handle: a mid-sync connection
    /// drop. In production, `LibrarySession` (see `bridge::library_client`) hides a
    /// RECOVERABLE drop entirely -- it reconnects and retries the same request, so
    /// `run_mirror_sync` never even sees a failure (see
    /// `library_client::tests::request_with_reconnect_reconnects_once_after_a_dead_connection_and_succeeds`
    /// for that half, proven with no live socket). This test covers the other half: what
    /// `run_mirror_sync` itself does when a design's connection drop could NOT be
    /// recovered (the reconnect attempt also failed) -- exactly what
    /// `FakeTransport::with_fetch_design_failure` scripts. The sync must not abort: it
    /// counts that one design as failed, leaves it exactly as it was locally (so the next
    /// sync retries it, same as `a_failed_fetch_is_counted_and_never_marked_synced`), and
    /// keeps going to fetch and save every OTHER design in the catalogue.
    #[test]
    fn a_design_whose_connection_could_not_be_recovered_is_skipped_but_the_rest_of_the_sync_completes()
     {
        let (db, path) = temp_db();
        let summaries = vec![
            design_summary(1, "First", "https://example.test/1", "v1"),
            design_summary(2, "Second", "https://example.test/2", "v1"),
            design_summary(3, "Third", "https://example.test/3", "v1"),
        ];
        let designs = vec![
            design_record(1, "First", "https://example.test/1", "v1"),
            design_record(2, "Second", "https://example.test/2", "v1"),
            design_record(3, "Third", "https://example.test/3", "v1"),
        ];
        let transport = FakeTransport::new(summaries, designs).with_fetch_design_failure(2);

        let mut processed_count = 0;
        let outcome = run_mirror_sync(
            &db,
            &transport,
            "remote-library:w",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |progress| processed_count = progress.processed,
        );
        let MirrorOutcome::Completed(counts) = outcome else {
            panic!(
                "an unrecoverable connection drop for ONE design must not abort the whole \
                 sync, got {outcome:?}"
            );
        };
        assert_eq!(
            processed_count, 3,
            "every design was still examined, including after the drop"
        );
        assert_eq!(counts.total_found, 3);
        assert_eq!(
            counts.new_count, 2,
            "designs 1 and 3 still land despite design 2's failure"
        );
        assert_eq!(counts.failed, 1);

        let guard = db.lock().unwrap();
        assert_eq!(guard.get_total_count().unwrap(), 2);
        let items = guard
            .search_diagrams(
                "",
                "All",
                "All",
                &diagram_catalog::model::filter::RangeFilter::default(),
            )
            .unwrap();
        let titles: std::collections::HashSet<&str> =
            items.iter().map(|i| i.title.as_str()).collect();
        assert!(titles.contains("First"));
        assert!(titles.contains("Third"));
        assert!(
            !titles.contains("Second"),
            "the design whose connection could not be recovered must never be half-saved, \
             titles were: {titles:?}"
        );
        assert_eq!(
            guard.get_mirror_state("https://example.test/2").unwrap(),
            None,
            "never marked synced, so the next sync retries it -- same as any other failed fetch"
        );
        drop(guard);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mirror_source_id_is_distinct_per_worker_address() {
        let a = WorkerSettings {
            address: "10.0.0.5:9443".to_string(),
            ..WorkerSettings::default()
        };
        let b = WorkerSettings {
            address: "10.0.0.6:9443".to_string(),
            ..WorkerSettings::default()
        };
        assert_ne!(mirror_source_id(&a), mirror_source_id(&b));
    }

    #[test]
    fn a_multi_page_mirror_reaches_designs_beyond_the_first_page() {
        let (db, path) = temp_db();
        let entries: Vec<(i64, &str)> = vec![
            (1, "First"),
            (2, "Second"),
            (3, "Third"),
            (4, "Fourth"),
            (5, "Fifth"),
        ];
        let summaries: Vec<DesignSummary> = entries
            .iter()
            .map(|(id, title)| {
                design_summary(*id, title, &format!("https://example.test/{id}"), "v1")
            })
            .collect();
        let designs: Vec<DesignRecord> = entries
            .iter()
            .map(|(id, title)| {
                design_record(*id, title, &format!("https://example.test/{id}"), "v1")
            })
            .collect();
        // 5 designs at 2 rows per SearchPage reply: pages of [1,2], [3,4], [5] -- the
        // last page is short, so exactly 3 round trips, not 5 and not 1.
        let transport = FakeTransport::new(summaries, designs).with_page_size(2);

        let outcome = run_mirror_sync(
            &db,
            &transport,
            "remote-library:w",
            MirrorOptions::default(),
            &AtomicBool::new(false),
            |_| {},
        );
        let MirrorOutcome::Completed(counts) = outcome else {
            panic!("expected Completed, got {outcome:?}");
        };
        assert_eq!(
            counts.total_found, 5,
            "every design across every page must be counted, not just the first page's"
        );
        assert_eq!(counts.new_count, 5);
        assert_eq!(transport.search_calls.load(Ordering::Relaxed), 3);
        assert_eq!(transport.fetch_design_calls.load(Ordering::Relaxed), 5);

        let guard = db.lock().unwrap();
        assert_eq!(guard.get_total_count().unwrap(), 5);
        let items = guard
            .search_diagrams(
                "",
                "All",
                "All",
                &diagram_catalog::model::filter::RangeFilter::default(),
            )
            .unwrap();
        let titles: std::collections::HashSet<&str> =
            items.iter().map(|i| i.title.as_str()).collect();
        // The whole point: "Fourth" and "Fifth" sat past the first page and must have
        // actually landed locally, not just been counted.
        assert!(titles.contains("Fourth"), "titles were: {titles:?}");
        assert!(titles.contains("Fifth"), "titles were: {titles:?}");
        drop(guard);
        std::fs::remove_file(&path).ok();
    }
}

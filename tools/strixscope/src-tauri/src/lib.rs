//! Strixscope pulls the StrixFold pair's logs read-only over ssh into SQLite and serves row sets to the UI.

mod derive;
mod discover;
mod error;
mod parse;
mod query;
mod runlog;
mod store;
mod sync;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use rusqlite::Connection;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use error::{Error, Result};

pub struct AppState {
    db: Arc<Mutex<Connection>>,
    syncing: Arc<AtomicBool>,
    settings_path: PathBuf,
}

/// Settings from disk; an unusable hand-edited host reads as unset, and a legacy two-host file keeps rank 0's host.
fn load_settings(path: &Path) -> sync::Settings {
    let mut s: sync::Settings = std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    if s.host.is_none() {
        s.host = s.rank0_host.clone();
    }
    for h in [&mut s.host, &mut s.rank0_host, &mut s.rank1_host] {
        if h.as_deref().is_some_and(|v| !sync::valid_host(v)) {
            *h = None;
        }
    }
    s
}

/// Reject what a pull could not use, so a typo shows up when saving.
fn check_settings(s: &sync::Settings) -> Result<()> {
    let bad = |what: &str, v: &str| Err(Error::Msg(format!("not a usable {what}: {v:?}")));
    match s.host.as_deref() {
        None => return Err(Error::Msg("the ssh host is required".into())),
        Some(h) if !sync::valid_host(h) => return bad("ssh host", h),
        _ => {}
    }
    for h in [&s.rank0_host, &s.rank1_host].into_iter().flatten() {
        if !sync::valid_host(h) {
            return bad("ssh host", h);
        }
    }
    if let Some(r) = s.run_root.as_deref().filter(|r| !discover::valid_path(r)) {
        return bad("run root", r);
    }
    if let Some(p) = s.prod_port.filter(|p| !(1..=65535).contains(p)) {
        return bad("port", &p.to_string());
    }
    Ok(())
}

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> sync::Settings {
    load_settings(&state.settings_path)
}

#[tauri::command]
fn save_settings(state: State<'_, AppState>, settings: sync::Settings) -> Result<()> {
    check_settings(&settings)?;
    let json = serde_json::to_string_pretty(&settings).map_err(|e| Error::Msg(e.to_string()))?;
    // Write then rename, so a crash never leaves half a file.
    let tmp = state.settings_path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &state.settings_path)?;
    Ok(())
}

/// The last good discovery with the settings that made it; a pull uses it only while those settings are unchanged.
#[derive(serde::Serialize, serde::Deserialize)]
struct CachedTopology {
    settings: sync::Settings,
    topology: discover::Topology,
}

/// Overrides that only rename what the pull charts, not where it reads, do not make the answer stale.
fn same_target(a: &sync::Settings, b: &sync::Settings) -> bool {
    a.host == b.host
        && a.rank0_host == b.rank0_host
        && a.rank1_host == b.rank1_host
        && a.run_root == b.run_root
}

fn cached_topology(conn: &Connection, s: &sync::Settings) -> Option<discover::Topology> {
    let j = store::get_meta(conn, "topology").ok().flatten()?;
    let c: CachedTopology = serde_json::from_str(&j).ok()?;
    same_target(&c.settings, s).then(|| c.topology.with_overrides(s))
}

fn cache_topology(conn: &Connection, s: &sync::Settings, t: &discover::Topology) {
    let c = CachedTopology {
        settings: s.clone(),
        topology: t.clone(),
    };
    if let Ok(j) = serde_json::to_string(&c) {
        let _ = store::set_meta(conn, "topology", &j);
    }
}

/// The last discovery for the saved host, with the saved overrides applied.
#[tauri::command]
fn get_topology(state: State<'_, AppState>) -> Option<discover::Topology> {
    let s = load_settings(&state.settings_path);
    cached_topology(&state.db.lock(), &s)
}

/// Discovers with these unsaved settings as a preview, over ssh and off the main thread; nothing is cached.
#[tauri::command(async)]
fn discover(settings: sync::Settings) -> Result<discover::Topology> {
    check_settings(&settings)?;
    discover::discover(&settings)
}

/// Builds with the `self-update` feature carry the updater; others do not.
#[tauri::command]
fn self_update_enabled() -> bool {
    cfg!(all(desktop, feature = "self-update"))
}

#[tauri::command]
fn get_status(state: State<'_, AppState>) -> Result<query::Status> {
    query::status(&state.db.lock())
}

#[tauri::command]
fn is_syncing(state: State<'_, AppState>) -> bool {
    state.syncing.load(Ordering::SeqCst)
}

#[tauri::command(async)]
fn get_dataset(
    state: State<'_, AppState>,
    from: i64,
    to: i64,
    include_tests: bool,
) -> Result<query::Dataset> {
    query::dataset(&state.db.lock(), from, to, include_tests)
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Progress {
    step: usize,
    steps: usize,
    label: String,
}

#[derive(Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct SyncDone {
    new_lines: usize,
    sar_samples: usize,
    errors: Vec<String>,
    rebuild: Option<derive::RebuildStats>,
}

fn report(app: &AppHandle, step: usize, steps: usize, label: String) {
    let _ = app.emit("sync-progress", Progress { step, steps, label });
}

fn note_skipped(out: &mut SyncDone, label: &str, skipped: usize) {
    if skipped > 0 {
        out.errors
            .push(format!("{label}: {skipped} unreadable line(s) skipped"));
    }
}

/// Walks the pull steps in order, reporting each one to the UI.
struct Runner<'a> {
    app: &'a AppHandle,
    db: &'a Arc<Mutex<Connection>>,
    step: usize,
    steps: usize,
}

impl Runner<'_> {
    fn report(&self, label: String) {
        report(self.app, self.step, self.steps, label);
    }

    fn runlog(
        &mut self,
        label: &str,
        host: &str,
        node: &str,
        root: &str,
        rank: u8,
        out: &mut SyncDone,
    ) {
        self.report(format!("Pulling {label}"));
        match runlog::pull(self.db, host, node, root, rank) {
            Ok(r) => {
                out.new_lines += r.lines;
                if r.unstamped > 0 {
                    out.errors.push(format!(
                        "{label}: {} line(s) without a timestamp skipped; start the pair with \
                         LOG_TIME=1 in its serve.conf for request history",
                        r.unstamped
                    ));
                }
                if r.no_meta > 0 {
                    out.errors.push(format!(
                        "{label}: {} run(s) whose resolved-rank{rank}.json could not be read; tried again next pull",
                        r.no_meta
                    ));
                }
            }
            Err(e) => out.errors.push(format!("{label}: {e}")),
        }
        self.step += 1;
    }

    fn journal(
        &mut self,
        label: &str,
        host: &str,
        source: &str,
        selector: &str,
        out: &mut SyncDone,
    ) {
        self.report(format!("Pulling {label}"));
        match sync::pull_journal(self.db, host, source, selector) {
            Ok(r) => {
                out.new_lines += r.lines;
                note_skipped(out, label, r.skipped);
            }
            Err(e) => out.errors.push(format!("{label}: {e}")),
        }
        self.step += 1;
    }

    fn sar(&mut self, label: &str, host: &str, dir: &str, node: &str, out: &mut SyncDone) {
        self.report(format!("Pulling {label}"));
        match sync::pull_sar(self.db, host, dir, node) {
            Ok(r) => {
                out.sar_samples += r.samples;
                if let Some(w) = r.warning {
                    out.errors.push(w);
                }
            }
            Err(e) => out.errors.push(format!("{label}: {e}")),
        }
        self.step += 1;
    }
}

/// The tag the ranks' journal lines carry when a unit runs them.
const JOURNAL_TAG: &str = "-t strixfold-container";

/// Find the pair, pull everything, then rebuild the derived tables.
fn run_sync(app: &AppHandle, db: &Arc<Mutex<Connection>>, s: &sync::Settings) -> SyncDone {
    let mut out = SyncDone::default();
    let mut run = Runner {
        app,
        db,
        step: 0,
        steps: 2,
    };
    run.report("Finding the pair".into());
    // A failed discovery still pulls what it can with the last good answer.
    let topo = match discover::discover(s) {
        Ok(t) => {
            cache_topology(&db.lock(), s, &t);
            Some(t)
        }
        Err(e) => {
            out.errors.push(format!("finding the pair: {e}"));
            cached_topology(&db.lock(), s)
        }
    };
    run.step += 1;
    if let Some(t) = &topo {
        let sar = [&t.rank0_sa_dir, &t.rank1_sa_dir]
            .iter()
            .filter(|d| d.is_some())
            .count();
        run.steps = 1 + 2 + t.watchdog as usize + sar + 1;
        // Each rank is read where it logs now: the journal while a unit runs it, else its run directories.
        for (rank, host, node, journal) in [
            (0u8, &t.rank0_host, &t.rank0_name, t.rank0_journal),
            (1, &t.rank1_host, &t.rank1_name, t.rank1_journal),
        ] {
            if journal {
                let label = format!("rank {rank} journal");
                run.journal(&label, host, &format!("rank{rank}"), JOURNAL_TAG, &mut out);
            } else {
                run.runlog(
                    &format!("rank {rank} logs"),
                    host,
                    node,
                    &t.run_root,
                    rank,
                    &mut out,
                );
            }
        }
        if t.watchdog {
            run.journal(
                "watchdog",
                &t.rank0_host,
                "watchdog",
                "-u strixfold-watchdog",
                &mut out,
            );
        }
        if let Some(d) = &t.rank0_sa_dir {
            run.sar("rank 0 sar", &t.rank0_host, d, "rank0", &mut out);
        }
        if let Some(d) = &t.rank1_sa_dir {
            run.sar("rank 1 sar", &t.rank1_host, d, "rank1", &mut out);
        }
    }
    run.report("Rebuilding statistics".into());
    let mut conn = db.lock();
    match derive::rebuild(&mut conn, s.prod_port) {
        Ok(r) => {
            if let Some(p) = r.prod_port_unused {
                out.errors.push(format!(
                    "the production port set in the pair settings, {p}, matches no run; using {}",
                    r.prod_port.map_or("every port".into(), |p| p.to_string())
                ));
            }
            out.rebuild = Some(r);
        }
        Err(e) => out.errors.push(format!("rebuild: {e}")),
    }
    let now = chrono::Utc::now().to_rfc3339();
    let _ = store::set_meta(&conn, "last_pull", &now);
    let _ = store::set_meta(
        &conn,
        "last_pull_errors",
        &serde_json::to_string(&out.errors).unwrap_or_default(),
    );
    out
}

/// Clears the "a pull is running" flag however the thread ends, panics included.
struct SyncFlag(Arc<AtomicBool>);

impl Drop for SyncFlag {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// The outcome of a pull that panicked: `sync-done` must still arrive and the message is persisted for the UI's reload.
fn panic_done(db: &Mutex<Connection>) -> SyncDone {
    let errors = vec!["the pull failed unexpectedly; try again".to_string()];
    if let Ok(s) = serde_json::to_string(&errors) {
        let _ = store::set_meta(&db.lock(), "last_pull_errors", &s);
    }
    SyncDone {
        errors,
        ..Default::default()
    }
}

/// Pulls everything on its own thread, emitting `sync-progress` per step and `sync-done` at the end.
#[tauri::command]
fn sync_now(app: AppHandle, state: State<'_, AppState>) -> Result<()> {
    if state.syncing.swap(true, Ordering::SeqCst) {
        return Err(Error::Msg("a pull is already running".into()));
    }
    let db = state.db.clone();
    let flag = state.syncing.clone();
    let settings = load_settings(&state.settings_path);
    if settings.host.is_none() {
        state.syncing.store(false, Ordering::SeqCst);
        return Err(Error::Msg("no host set up".into()));
    }
    std::thread::spawn(move || {
        let guard = SyncFlag(flag);
        let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_sync(&app, &db, &settings)
        }))
        .unwrap_or_else(|_| panic_done(&db));
        drop(guard);
        let _ = app.emit("sync-done", done);
    });
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        // A second launch hands over to the running instance, so two processes never write the same database.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(main) = app.get_webview_window("main") {
                let _ = main.unminimize();
                let _ = main.show();
                let _ = main.set_focus();
            }
        }))
        .plugin(tauri_plugin_window_state::Builder::new().build());

    // Self-update (feature `self-update` only): checks the configured manifest, verifies minisign signatures, restarts.
    #[cfg(all(desktop, feature = "self-update"))]
    let builder = builder
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init());

    builder
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(&data_dir)?;
            let conn = store::open(&data_dir.join("strixscope.db"))
                .map_err(|e| Box::new(std::io::Error::other(e.to_string())))?;
            app.manage(AppState {
                db: Arc::new(Mutex::new(conn)),
                syncing: Arc::new(AtomicBool::new(false)),
                settings_path: data_dir.join("settings.json"),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            is_syncing,
            get_dataset,
            sync_now,
            get_settings,
            save_settings,
            get_topology,
            discover,
            self_update_enabled,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Strixscope");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panic must leave the "pull running" flag clear, or no later pull could start.
    #[test]
    fn panic_clears_the_sync_flag() {
        let flag = Arc::new(AtomicBool::new(true));
        let guard = SyncFlag(flag.clone());
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _held = guard;
            panic!("injected");
        }));
        assert!(caught.is_err());
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// The synthesized panic error must survive the reload after `sync-done`, and the database stay usable.
    #[test]
    fn panic_note_is_persisted_and_the_db_still_works() {
        let db = Mutex::new(crate::store::memory());
        let done = panic_done(&db);
        assert_eq!(done.errors.len(), 1);
        let saved = store::get_meta(&db.lock(), "last_pull_errors")
            .unwrap()
            .unwrap();
        assert!(saved.contains("failed unexpectedly"), "{saved}");
        // A subsequent pull can still write.
        store::set_meta(&db.lock(), "last_pull", "now").unwrap();
    }
}

#[cfg(test)]
mod topology_cache {
    use super::*;

    #[test]
    fn fallback_only_for_the_same_settings() {
        let conn = store::memory();
        let s = sync::Settings {
            host: Some("node0".into()),
            ..Default::default()
        };
        let t = discover::Topology {
            rank0_host: "node0".into(),
            rank1_host: "node1".into(),
            run_root: discover::DEFAULT_RUN_ROOT.into(),
            rank1_journal: true,
            ..Default::default()
        };
        cache_topology(&conn, &s, &t);
        assert_eq!(cached_topology(&conn, &s), Some(t.clone()));
        // Another seed host, rank 1 overridden: not about the same boxes.
        for other in [
            sync::Settings {
                host: Some("box9".into()),
                ..s.clone()
            },
            sync::Settings {
                rank1_host: Some("node7".into()),
                ..s.clone()
            },
        ] {
            assert_eq!(cached_topology(&conn, &other), None);
        }
        // Made with an override that is now cleared: stale too.
        let with = sync::Settings {
            rank1_host: Some("node7".into()),
            ..s.clone()
        };
        cache_topology(&conn, &with, &t);
        assert_eq!(cached_topology(&conn, &s), None);
        // An override of what is charted, not where it reads, still falls back.
        cache_topology(&conn, &s, &t);
        let nics = sync::Settings {
            link_ifaces: Some(vec!["eth1".into()]),
            ..s
        };
        assert_eq!(
            cached_topology(&conn, &nics).unwrap().link_ifaces,
            vec!["eth1"]
        );
    }
}

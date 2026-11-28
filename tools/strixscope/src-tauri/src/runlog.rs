//! Rank logs from StrixFold's run directories, fetched by byte offset; only lines with a `TF_LOG_TIME` stamp are kept.

use std::collections::HashMap;

use parking_lot::Mutex;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::discover::{Resolved, valid_path};
use crate::error::{Error, Result};
use crate::store;
use crate::sync::ssh_gz;

/// Outcome of one rank's pull.
#[derive(Debug, Default, Clone, Copy)]
pub struct RunLogPull {
    pub lines: usize,
    pub unstamped: usize,
    /// Runs whose resolved JSON could not be read (asked again next pull).
    pub no_meta: usize,
}

/// Runs fetched per ssh call, so one command line stays small.
const BATCH: usize = 64;
/// Log bytes fetched per ssh call (a run larger than this goes alone).
const BATCH_BYTES: u64 = 32 << 20;

/// A run dir name we may quote and use inside a cursor.
fn valid_run(r: &str) -> bool {
    !r.is_empty()
        && r.len() <= 128
        && !r.starts_with('-')
        && r.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_-".contains(c))
}

/// A short stable name for (node, run root) that scopes state and cursors, from the box's hostname and resolved root.
fn scope(node: &str, root: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in node.bytes().chain([0]).chain(root.bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

fn state_key(ns: &str, rank: u8, run: &str) -> String {
    format!("runlog:{ns}:rank{rank}:{run}")
}

/// What we hold of one run's log, kept in `meta` as JSON.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct RunState {
    /// Bumped when the run dir is reused.
    generation: u32,
    /// mtime (`s.ns`) of the resolved JSON this generation belongs to.
    stamp: String,
    /// Bytes of the log consumed (complete lines only).
    offset: u64,
    /// The resolved JSON was read and its start stored (or skipped).
    meta: bool,
    /// The journal already holds this run: its lines are skipped.
    #[serde(default)]
    journaled: bool,
}

fn list_cmd(root: &str, rank: u8) -> String {
    // pipefail and `exit 1`: a missing root is an error, not "no runs".
    format!(
        "set -o pipefail; {{ cd '{root}' || exit 1; echo \"@@ROOT $(pwd -P)\"; \
         for d in */; do d=${{d%/}}; \
         [ -L \"$d\" ] && continue; \
         [ -f \"$d/resolved-rank{rank}.json\" ] || continue; echo \"@@RUN $d\"; \
         ls -ln --time-style=+%s.%N \"$d/resolved-rank{rank}.json\" \"$d/rank{rank}.log\" 2>/dev/null; \
         done; true; }} | gzip -1"
    )
}

/// The root as the box resolved it, from the listing.
fn listed_root(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|l| l.strip_prefix("@@ROOT "))
        .map(str::trim)
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Listed {
    run: String,
    /// The log's size (0 when there is none).
    size: u64,
    /// The log exists.
    has_log: bool,
    /// The resolved JSON's mtime, `seconds.nanoseconds`.
    stamp: String,
}

/// Every run dir that has this rank's resolved JSON; symlinks (`latest`) are skipped or the current run would count twice.
fn parse_list(text: &str) -> Vec<Listed> {
    let mut out: Vec<Listed> = Vec::new();
    for line in text.lines() {
        if let Some(run) = line.strip_prefix("@@RUN ") {
            out.push(Listed {
                run: run.trim().to_string(),
                ..Default::default()
            });
        } else if let Some(last) = out.last_mut() {
            // `ls -ln --time-style=+%s.%N`: mode, links, uid, gid, size, mtime, name
            let f: Vec<&str> = line.split_whitespace().collect();
            let (Some(size), Some(mtime), Some(name)) = (f.get(4), f.get(5), f.get(6)) else {
                continue;
            };
            if name.ends_with(".json") {
                last.stamp = mtime.to_string();
            } else {
                last.size = size.parse().unwrap_or(0);
                last.has_log = true;
            }
        }
    }
    out
}

struct Want {
    run: String,
    state: RunState,
    /// The resolved JSON's mtime as listed now.
    stamp: String,
    /// Fetch the log too; not while the run's metadata keeps failing, since its lines are taken only after.
    log: bool,
    /// Bytes this fetch may bring (for batching).
    bytes: u64,
}

/// What to fetch for a listed run given what we hold, or None.
fn want(l: Listed, held: Option<RunState>) -> Option<Want> {
    let (state, log) = match held {
        None => (RunState::default(), l.has_log),
        // Reused run dir (resolved JSON rewritten or log shorter): a new generation; a log that is gone is no new run.
        Some(h) if h.stamp != l.stamp || (l.has_log && l.size < h.offset) => (
            RunState {
                generation: h.generation + 1,
                ..Default::default()
            },
            l.has_log,
        ),
        Some(h) if !h.meta => (h, false),
        Some(h) if !l.has_log || l.size <= h.offset => return None,
        Some(h) => (h, true),
    };
    let bytes = if log {
        l.size.saturating_sub(state.offset)
    } else {
        0
    };
    Some(Want {
        run: l.run,
        state,
        stamp: l.stamp,
        log,
        bytes,
    })
}

/// Batches of at most `BATCH` runs and about `BATCH_BYTES` of log, so one ssh answer stays small.
fn batches(wants: &[Want]) -> Vec<&[Want]> {
    let mut out = Vec::new();
    let (mut start, mut bytes) = (0, 0u64);
    for (i, w) in wants.iter().enumerate() {
        if i > start && (i - start >= BATCH || bytes + w.bytes > BATCH_BYTES) {
            out.push(&wants[start..i]);
            (start, bytes) = (i, 0);
        }
        bytes += w.bytes;
    }
    if start < wants.len() {
        out.push(&wants[start..]);
    }
    out
}

fn fetch_cmd(root: &str, rank: u8, nonce: &str, wants: &[Want]) -> String {
    let mut s = format!("set -o pipefail; {{ cd '{root}' || exit 1; ");
    for w in wants {
        let run = &w.run;
        if !w.state.meta {
            s += &format!(
                "echo '@@RES {run} {nonce}'; cat '{run}/resolved-rank{rank}.json'; echo; echo '@@END {nonce}'; "
            );
        }
        if w.log {
            s += &format!(
                "echo '@@LOG {run} {nonce}'; tail -c +{} '{run}/rank{rank}.log' 2>/dev/null; echo; echo '@@END {nonce}'; ",
                w.state.offset + 1
            );
        }
    }
    s + "} | gzip -1"
}

/// Sections of a fetch: (kind "RES"/"LOG", run, exact bytes).
fn parse_fetch<'a>(data: &'a [u8], nonce: &str) -> Vec<(&'a str, &'a str, &'a [u8])> {
    let end = format!("\n@@END {nonce}\n");
    let mut out = Vec::new();
    let mut rest = data;
    while let Some(nl) = rest.iter().position(|b| *b == b'\n') {
        let head = std::str::from_utf8(&rest[..nl]).unwrap_or("");
        let body = &rest[nl + 1..];
        let mut f = head.split(' ');
        let (kind, run, n) = (f.next().unwrap_or(""), f.next().unwrap_or(""), f.next());
        let Some(stop) = find(body, end.as_bytes()) else {
            break;
        };
        if n == Some(nonce) && matches!(kind, "@@RES" | "@@LOG") {
            out.push((&kind[2..], run, &body[..stop]));
        }
        rest = &body[stop + end.len()..];
    }
    out
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// `2026-11-26T14:03:12.123456Z rest` → (µs, rest).
pub fn split_stamp(line: &str) -> Option<(i64, &str)> {
    if !line.as_bytes().first()?.is_ascii_digit() {
        return None;
    }
    let (stamp, rest) = line.split_once(' ')?;
    let t = chrono::DateTime::parse_from_rfc3339(stamp).ok()?;
    Some((t.timestamp_micros(), rest))
}

fn utc_micros(utc: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(utc)
        .ok()
        .map(|t| t.timestamp_micros())
}

/// The run's start as the banner and argv lines `derive` reads.
fn banner_lines(run: &str, r: &Resolved) -> Option<[(i64, String); 2]> {
    let ts = utc_micros(&r.utc)?;
    let sha: String = r.config_sha256.chars().take(12).collect();
    let sha = if sha.is_empty() { "0".into() } else { sha };
    let deploy = if r.deploy.dir.is_empty() {
        "-"
    } else {
        &r.deploy.dir
    };
    let conf = if r.config.is_empty() { "-" } else { &r.config };
    Some([
        (
            ts,
            format!(
                "== {} strixfold rank {} on {}: run {run}, conf {conf} ({sha}), deploy {deploy}",
                r.utc, r.rank, r.host
            ),
        ),
        (ts + 1, format!("== argv: {}", r.argv.join(" "))),
    ])
}

/// Pulls rank `rank`'s run logs from `host` (whose hostname is `node`) into `raw` (source `rank<R>`).
pub fn pull(
    db: &Mutex<Connection>,
    host: &str,
    node: &str,
    root: &str,
    rank: u8,
) -> Result<RunLogPull> {
    if !valid_path(root) {
        return Err(Error::Msg(format!("not a usable run root: {root:?}")));
    }
    let listing = ssh_gz(host, &list_cmd(root, rank))?;
    let text = String::from_utf8_lossy(&listing);
    let ns = scope(node, listed_root(&text).unwrap_or(root));
    let runs = parse_list(&text);
    let (wants, in_journal) = {
        let conn = db.lock();
        let mut w = Vec::new();
        for l in runs.into_iter().filter(|l| valid_run(&l.run)) {
            let held = store::get_meta(&conn, &state_key(&ns, rank, &l.run))?
                .and_then(|j| serde_json::from_str::<RunState>(&j).ok());
            w.extend(want(l, held));
        }
        (w, journal_runs(&conn, rank)?)
    };
    let mut out = RunLogPull::default();
    let nonce = format!(
        "{:x}",
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
    );
    for batch in batches(&wants) {
        let data = ssh_gz(host, &fetch_cmd(root, rank, &nonce, batch))?;
        let sections = parse_fetch(&data, &nonce);
        let r = insert(&mut db.lock(), &ns, rank, batch, &sections, &in_journal)?;
        out.lines += r.lines;
        out.unstamped += r.unstamped;
        out.no_meta += r.no_meta;
    }
    Ok(out)
}

/// Runs whose start the journal already holds for this rank: name → the journal times of their banners.
fn journal_runs(conn: &Connection, rank: u8) -> Result<HashMap<String, Vec<i64>>> {
    let mut st = conn.prepare(
        "SELECT ts, msg FROM raw WHERE source = ?1 AND cursor NOT LIKE 'run=%' AND msg LIKE '== %'",
    )?;
    let rows = st.query_map([format!("rank{rank}")], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
    })?;
    let mut out: HashMap<String, Vec<i64>> = HashMap::new();
    for row in rows {
        let (ts, msg) = row?;
        if let Some(crate::parse::Line::RunStart { name, .. }) = crate::parse::parse(&msg) {
            out.entry(name).or_default().push(ts);
        }
    }
    Ok(out)
}

/// The journal holds this very run: same name, banner within five minutes of the resolved JSON's start.
fn in_journal(journal: &HashMap<String, Vec<i64>>, run: &str, start: i64) -> bool {
    journal
        .get(run)
        .is_some_and(|ts| ts.iter().any(|t| (t - start).abs() <= 300_000_000))
}

fn insert(
    conn: &mut Connection,
    ns: &str,
    rank: u8,
    batch: &[Want],
    sections: &[(&str, &str, &[u8])],
    journal: &HashMap<String, Vec<i64>>,
) -> Result<RunLogPull> {
    let source = format!("rank{rank}");
    let mut resolved: HashMap<&str, Resolved> = HashMap::new();
    let mut logs: HashMap<&str, &[u8]> = HashMap::new();
    for (kind, run, body) in sections {
        match *kind {
            "RES" => {
                if let Ok(r) = serde_json::from_slice::<Resolved>(body) {
                    resolved.insert(run, r);
                }
            }
            _ => {
                logs.insert(run, body);
            }
        }
    }

    let tx = conn.transaction()?;
    let mut out = RunLogPull::default();
    {
        let mut st = tx.prepare(
            "INSERT OR IGNORE INTO raw(source, ts, cursor, msg) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for w in batch {
            let run = w.run.as_str();
            let mut state = RunState {
                stamp: w.stamp.clone(),
                ..w.state.clone()
            };
            let g = state.generation;
            if !state.meta {
                // Unreadable or not written yet: asked again next pull.
                if let Some(lines) = resolved.get(run).and_then(|r| banner_lines(run, r)) {
                    state.journaled = in_journal(journal, run, lines[0].0);
                    if !state.journaled {
                        for (i, (ts, msg)) in lines.iter().enumerate() {
                            out.lines += st.execute(params![
                                source,
                                ts,
                                format!("run={run};n={ns};r={rank};g={g};b={i}"),
                                msg
                            ])?;
                        }
                    }
                    state.meta = true;
                } else {
                    out.no_meta += 1;
                }
            }
            // No lines before the metadata, or they would come in again under a new generation.
            let body = logs.get(run).filter(|_| state.meta);
            let Some(body) = body else {
                store::set_meta(&tx, &state_key(ns, rank, run), &to_json(&state))?;
                continue;
            };
            let journaled = state.journaled;
            // Complete lines only; a partial last line is read next time.
            let done = body.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
            let mut at = state.offset;
            for raw in body[..done].split_inclusive(|b| *b == b'\n') {
                let line = String::from_utf8_lossy(raw);
                let line = line.trim_end_matches(['\n', '\r']);
                if !line.is_empty() && !journaled {
                    match split_stamp(line) {
                        Some((ts, msg)) => {
                            out.lines += st.execute(params![
                                source,
                                ts,
                                format!("run={run};n={ns};r={rank};g={g};o={at}"),
                                msg
                            ])?;
                        }
                        None => out.unstamped += 1,
                    }
                }
                at += raw.len() as u64;
            }
            state.offset += done as u64;
            store::set_meta(&tx, &state_key(ns, rank, run), &to_json(&state))?;
        }
    }
    tx.commit()?;
    Ok(out)
}

fn to_json(s: &RunState) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listing() {
        let l = "@@RUN strixfold-20261125T230446Z\n\
                 -rw-r--r-- 1 1000 1000 4096 1764111888.123456789 strixfold-20261125T230446Z/resolved-rank0.json\n\
                 -rw-r--r-- 1 1000 1000 80957 1764113000.5 strixfold-20261125T230446Z/rank0.log\n\
                 @@RUN vt-image-sha\n\
                 -rw-r--r-- 1 1000 1000 10 1764000000.000000001 vt-image-sha/resolved-rank0.json\n\
                 @@RUN odd.name\n";
        let runs = parse_list(l);
        assert_eq!(runs.len(), 3);
        assert_eq!(
            runs[0],
            Listed {
                run: "strixfold-20261125T230446Z".into(),
                size: 80957,
                has_log: true,
                stamp: "1764111888.123456789".into(),
            }
        );
        assert_eq!((runs[1].size, runs[1].has_log), (0, false));
        assert_eq!(
            listed_root("@@ROOT /home/u/tfp/run/serve\n@@RUN a\n"),
            Some("/home/u/tfp/run/serve")
        );
        assert_eq!(runs[1].stamp, "1764000000.000000001");
        assert!(valid_run("vt-image-sha"));
        assert!(!valid_run("odd.name"));
        assert!(!valid_run("a'b"));
    }

    #[test]
    fn listing_skips_symlinks() {
        let dir = std::env::temp_dir().join(format!("strixscope-runs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("r1")).unwrap();
        std::fs::write(dir.join("r1/resolved-rank0.json"), "{}").unwrap();
        std::fs::write(dir.join("r1/rank0.log"), "abc\n").unwrap();
        std::os::unix::fs::symlink("r1", dir.join("latest")).unwrap();
        let cmd = list_cmd(dir.to_str().unwrap(), 0);
        let out = std::process::Command::new("sh")
            .args(["-c", &cmd])
            .output()
            .unwrap();
        let mut text = String::new();
        std::io::Read::read_to_string(
            &mut flate2::read::GzDecoder::new(&out.stdout[..]),
            &mut text,
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let runs = parse_list(&text);
        assert_eq!(runs.len(), 1, "{text}");
        assert_eq!((runs[0].run.as_str(), runs[0].size), ("r1", 4));
        // Nanosecond mtimes, so a dir reused within a second still differs.
        assert!(runs[0].stamp.contains('.'), "{}", runs[0].stamp);
    }

    #[test]
    fn what_to_fetch() {
        let l = |size, stamp: &str| Listed {
            run: "r".into(),
            size,
            has_log: true,
            stamp: stamp.into(),
        };
        let held = RunState {
            stamp: "7.5".into(),
            offset: 100,
            meta: true,
            ..Default::default()
        };
        // New run: everything, resolved JSON included.
        let w = want(l(50, "7.5"), None).unwrap();
        assert_eq!(w.state, RunState::default());
        // Nothing new.
        assert!(want(l(100, "7.5"), Some(held.clone())).is_none());
        // Grown: from the offset, no resolved JSON.
        let w = want(l(150, "7.5"), Some(held.clone())).unwrap();
        assert_eq!((w.state.offset, w.state.meta), (100, true));
        // Metadata still missing: asked again, the metadata only.
        let pending = RunState {
            meta: false,
            ..held.clone()
        };
        let w = want(l(5000, "7.5"), Some(pending)).unwrap();
        assert!(!w.state.meta && !w.log && w.bytes == 0);
        // The log is gone (cleaned up): no new generation, nothing to read.
        let gone = Listed {
            has_log: false,
            ..l(0, "7.5")
        };
        assert!(want(gone, Some(held.clone())).is_none());
        // A first look: the log too, with its size for batching.
        let w = want(l(500, "7.5"), None).unwrap();
        assert!(w.log && w.bytes == 500);
        // Reused dir (resolved rewritten, or the log shrank): new generation.
        for listed in [l(500, "7.6"), l(50, "7.5")] {
            let w = want(listed, Some(held.clone())).unwrap();
            assert_eq!(
                (w.state.generation, w.state.offset, w.state.meta),
                (1, 0, false)
            );
        }
    }

    #[test]
    fn stamps() {
        let (ts, rest) = split_stamp("2026-11-26T14:03:12.123456Z info: lanes: x").unwrap();
        assert_eq!(ts, 1_795_701_792_123_456);
        assert_eq!(rest, "info: lanes: x");
        assert!(split_stamp("info: lanes: x").is_none());
        assert!(split_stamp("== 2026-11-25T23:04:48Z strixfold rank 0").is_none());
        assert!(split_stamp("2026 is not a stamp").is_none());
    }

    #[test]
    fn fetch_sections_are_exact() {
        // The log's own bytes may contain "@@END" without the nonce, and the last line may be partial.
        let data =
            b"@@RES r1 n1\n{\"rank\":0}\n\n@@END n1\n@@LOG r1 n1\nline\n@@END x\npart\n@@END n1\n";
        let s = parse_fetch(data, "n1");
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], ("RES", "r1", &b"{\"rank\":0}\n"[..]));
        assert_eq!(s[1], ("LOG", "r1", &b"line\n@@END x\npart"[..]));
    }

    #[test]
    fn batches_by_count_and_bytes() {
        let w = |bytes| Want {
            run: "r".into(),
            state: RunState::default(),
            stamp: String::new(),
            log: true,
            bytes,
        };
        let sizes = |v: &[Want]| batches(v).iter().map(|b| b.len()).collect::<Vec<_>>();
        let many: Vec<Want> = (0..130).map(|_| w(1)).collect();
        assert_eq!(sizes(&many), vec![64, 64, 2]);
        // A run larger than the budget goes alone; small ones share.
        let mixed = [w(1), w(BATCH_BYTES + 5), w(1), w(1)];
        assert_eq!(sizes(&mixed), vec![1, 1, 2]);
        assert!(batches(&[]).is_empty());
    }

    #[test]
    fn scopes_differ() {
        assert_eq!(
            scope("node0", "tfp/run/serve"),
            scope("node0", "tfp/run/serve")
        );
        assert_ne!(
            scope("node0", "tfp/run/serve"),
            scope("node1", "tfp/run/serve")
        );
        assert_ne!(scope("node0", "a"), scope("node0", "b"));
    }

    fn resolved(run_utc: &str) -> String {
        format!(
            r#"{{"rank": 0, "host": "node0", "utc": "{run_utc}", "config": "/c/serve.conf",
               "config_sha256": "070eb2add70b99", "deploy": {{"dir": "/d/tfp/deploy/ad367d8a2ede"}},
               "argv": ["/x/tensorfold-native", "serve", "/m", "--port", "18080", "--name", "m"]}}"#
        )
    }

    fn wanted(state: RunState) -> [Want; 1] {
        [Want {
            run: "r1".into(),
            state,
            stamp: "7.5".into(),
            log: true,
            bytes: 0,
        }]
    }

    fn held(conn: &Connection) -> RunState {
        let j = store::get_meta(conn, "runlog:ns:rank0:r1")
            .unwrap()
            .unwrap();
        serde_json::from_str(&j).unwrap()
    }

    fn count(conn: &Connection) -> i64 {
        conn.query_row("SELECT COUNT(*) FROM raw", [], |r| r.get(0))
            .unwrap()
    }

    type Sections<'a> = Vec<(&'a str, &'a str, &'a [u8])>;

    #[test]
    fn insert_and_resume() {
        let mut conn = store::memory();
        let none = HashMap::new();
        let res = resolved("2026-11-21T22:08:56Z");
        let body = b"[tensorfold] unstamped\n2026-11-21T22:09:00.000001Z info: one\n2026-11-21T22:09:01Z info: partial";
        let secs: Sections = vec![("RES", "r1", res.as_bytes()), ("LOG", "r1", &body[..])];
        let r = insert(
            &mut conn,
            "ns",
            0,
            &wanted(RunState::default()),
            &secs,
            &none,
        )
        .unwrap();
        // banner + argv + one stamped line; one unstamped; the partial waits.
        assert_eq!((r.lines, r.unstamped), (3, 1));
        let done = body.iter().rposition(|b| *b == b'\n').unwrap() + 1;
        let h = held(&conn);
        assert_eq!((h.offset, h.meta), (done as u64, true));
        assert_eq!(h.stamp, "7.5");
        let banner: String = conn
            .query_row("SELECT msg FROM raw ORDER BY id LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert!(
            crate::parse::parse(&banner).is_some(),
            "synthesized banner must parse: {banner}"
        );
        // The next pull resumes at the partial line and finishes it.
        let more = b"2026-11-21T22:09:01Z info: partial done\n";
        let secs: Sections = vec![("LOG", "r1", &more[..])];
        let r = insert(&mut conn, "ns", 0, &wanted(h), &secs, &none).unwrap();
        assert_eq!((r.lines, r.unstamped), (1, 0));
        // A reused dir: generation 1 stores its lines beside the old ones.
        let g1 = RunState {
            generation: 1,
            ..Default::default()
        };
        let res2 = resolved("2026-11-22T08:00:00Z");
        let fresh = b"2026-11-22T08:00:05Z info: one\n";
        let secs: Sections = vec![("RES", "r1", res2.as_bytes()), ("LOG", "r1", &fresh[..])];
        let r = insert(&mut conn, "ns", 0, &wanted(g1), &secs, &none).unwrap();
        assert_eq!(r.lines, 3);
        assert_eq!(count(&conn), 7);
    }

    #[test]
    fn no_lines_before_the_metadata() {
        let mut conn = store::memory();
        let none = HashMap::new();
        let log = b"2026-11-21T22:09:00Z info: one\n";
        let secs: Sections = vec![("RES", "r1", b"{\"rank\""), ("LOG", "r1", &log[..])];
        insert(
            &mut conn,
            "ns",
            0,
            &wanted(RunState::default()),
            &secs,
            &none,
        )
        .unwrap();
        // Unreadable metadata: nothing taken, so nothing can come in twice.
        let h = held(&conn);
        assert_eq!((h.meta, h.offset), (false, 0));
        assert_eq!(count(&conn), 0);
        // Listed again unchanged: wanted again; this time it reads.
        let l = Listed {
            run: "r1".into(),
            size: log.len() as u64,
            has_log: true,
            stamp: "7.5".into(),
        };
        // Asked again for the metadata only; once read, the log follows.
        let w = want(l, Some(h)).unwrap();
        assert!(!w.log);
        let res = resolved("2026-11-21T22:08:56Z");
        let secs: Sections = vec![("RES", "r1", res.as_bytes())];
        let r = insert(&mut conn, "ns", 0, &[w], &secs, &none).unwrap();
        assert_eq!((r.lines, r.no_meta), (2, 0));
        let h = held(&conn);
        assert!(h.meta && h.offset == 0);
        let secs: Sections = vec![("LOG", "r1", &log[..])];
        let r = insert(&mut conn, "ns", 0, &wanted(h), &secs, &none).unwrap();
        assert_eq!(r.lines, 1);
        assert_eq!(held(&conn).offset, log.len() as u64);
    }

    #[test]
    fn runs_in_the_journal_are_skipped() {
        let mut conn = store::memory();
        // The journal's banner for r1, logged at its start.
        conn.execute(
            "INSERT INTO raw(source, ts, cursor, msg) VALUES ('rank0', 1795298936000000, 's=a;i=1', ?1)",
            ["== 2026-11-21T22:08:56Z strixfold rank 0 on node0: run r1, conf /c (070eb2add70b), deploy /d/ad367d8a2ede"],
        )
        .unwrap();
        let journal = journal_runs(&conn, 0).unwrap();
        assert!(in_journal(&journal, "r1", 1_795_298_936_000_000));
        // The same dir reused a day later is another run.
        assert!(!in_journal(&journal, "r1", 1_795_385_336_000_000));
        let res = resolved("2026-11-21T22:08:56Z");
        let log = b"2026-11-21T22:09:00Z info: one\n";
        let secs: Sections = vec![("RES", "r1", res.as_bytes()), ("LOG", "r1", &log[..])];
        let r = insert(
            &mut conn,
            "ns",
            0,
            &wanted(RunState::default()),
            &secs,
            &journal,
        )
        .unwrap();
        assert_eq!((r.lines, r.unstamped), (0, 0));
        let h = held(&conn);
        assert!(h.meta && h.journaled && h.offset == log.len() as u64);
        // A later generation of the same dir, started a day later, is new.
        let g1 = RunState {
            generation: 1,
            ..Default::default()
        };
        let res = resolved("2026-11-22T22:08:56Z");
        let two = b"2026-11-22T22:09:00Z info: two\n";
        let secs: Sections = vec![("RES", "r1", res.as_bytes()), ("LOG", "r1", &two[..])];
        let r = insert(&mut conn, "ns", 0, &wanted(g1), &secs, &journal).unwrap();
        assert_eq!(r.lines, 3);
    }
}

//! Pulls logs from the pair over ssh, read-only and gzipped; journal pulls are incremental via the stored `__CURSOR`.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;
use parking_lot::Mutex;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::store;

/// Saved in settings.json; everything but `host` overrides a discovered value, `None` meaning discover it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    /// ssh destination (alias or user@host) of either node.
    pub host: Option<String>,
    pub rank0_host: Option<String>,
    pub rank1_host: Option<String>,
    /// The production serve port; `None`: the port with the most run time.
    pub prod_port: Option<u32>,
    /// rank 0's NICs that carry the links between the nodes.
    pub link_ifaces: Option<Vec<String>>,
    /// StrixFold's RUN_ROOT, relative to $HOME or absolute.
    pub run_root: Option<String>,
}

/// A pull that takes longer than this is stuck.
const REMOTE_DEADLINE: Duration = Duration::from_secs(180);

/// An ssh destination (alias or user@host), never something ssh could read as an option.
pub fn valid_host(h: &str) -> bool {
    !h.is_empty()
        && h.len() <= 253
        && !h.starts_with('-')
        && h.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-@".contains(c))
}

/// journald cursors are `s=<hex>;i=<hex>;b=<hex>;m=<hex>;t=<hex>;x=<hex>`.
fn valid_cursor(c: &str) -> bool {
    !c.is_empty()
        && c.len() <= 512
        && c.chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "=;_-".contains(ch))
}

/// Runs `remote` on `host` and returns its decompressed stdout; the whole ssh process group is killed at `REMOTE_DEADLINE`.
pub(crate) fn ssh_gz(host: &str, remote: &str) -> Result<Vec<u8>> {
    if !valid_host(host) {
        return Err(Error::Msg(format!("not a usable ssh host: {host:?}")));
    }
    let mut cmd = Command::new("ssh");
    cmd.args([
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=10",
        "--",
        host,
        remote,
    ]);
    let out = run_bounded(cmd, host, REMOTE_DEADLINE)?;
    let mut buf = Vec::new();
    GzDecoder::new(&out[..]).read_to_end(&mut buf)?;
    Ok(buf)
}

/// Runs `cmd` in its own process group, polling both pipes under one deadline and killing the group on timeout.
fn run_bounded(mut cmd: Command, label: &str, limit: Duration) -> Result<Vec<u8>> {
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;

    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|e| Error::Msg(format!("cannot run {label}: {e}")))?;
    let pid = child.id() as libc::pid_t;
    // Every early exit kills the group (the leader is unreaped, so -pid is ours), then the leader, then reaps it.
    let abort = |child: &mut std::process::Child| {
        // SAFETY: kill(2) on the process group we created.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        let _ = child.kill();
        let _ = child.wait();
    };
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let fds = [stdout.as_raw_fd(), stderr.as_raw_fd()];
    for fd in fds {
        // SAFETY: fcntl on pipe fds we own.
        let ok = unsafe {
            let fl = libc::fcntl(fd, libc::F_GETFL);
            fl >= 0 && libc::fcntl(fd, libc::F_SETFL, fl | libc::O_NONBLOCK) >= 0
        };
        if !ok {
            abort(&mut child);
            return Err(Error::Msg(format!("{label}: cannot set up its pipes")));
        }
    }
    // Bytes read per pipe per pass, so a busy pipe cannot starve the other or the deadline check.
    const PASS_CAP: usize = 1 << 20;

    // Exited but not reaped, so the pid and process-group id cannot be reused before any group kill.
    let exited = || -> bool {
        // SAFETY: waitid on our own child, leaving it waitable.
        unsafe {
            let mut info: libc::siginfo_t = std::mem::zeroed();
            let r = libc::waitid(
                libc::P_PID,
                pid as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            );
            r == 0 && info.si_pid() != 0
        }
    };

    let deadline = Instant::now() + limit;
    let mut out = Vec::new();
    let mut err = Vec::new();
    let mut open = [true, true];
    let mut buf = [0u8; 64 * 1024];
    loop {
        if !open[0] && !open[1] && exited() {
            break;
        }
        let now = Instant::now();
        if now >= deadline {
            abort(&mut child);
            return Err(Error::Msg(format!(
                "{label}: no answer within {} s; gave up",
                limit.as_secs()
            )));
        }
        let wait_ms = (deadline - now).min(Duration::from_millis(100)).as_millis() as i32;
        let mut pfds: Vec<libc::pollfd> = fds
            .iter()
            .zip(open)
            .filter(|(_, o)| *o)
            .map(|(fd, _)| libc::pollfd {
                fd: *fd,
                events: libc::POLLIN,
                revents: 0,
            })
            .collect();
        if pfds.is_empty() {
            std::thread::sleep(Duration::from_millis(wait_ms.max(1) as u64));
            continue;
        }
        // SAFETY: pfds is a valid array of pollfd for its length.
        unsafe {
            libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, wait_ms);
        }
        for (i, (reader, sink)) in [
            (&mut stdout as &mut dyn Read, &mut out),
            (&mut stderr as &mut dyn Read, &mut err),
        ]
        .into_iter()
        .enumerate()
        {
            let mut took = 0;
            while open[i] && took < PASS_CAP {
                match reader.read(&mut buf) {
                    Ok(0) => open[i] = false,
                    Ok(n) => {
                        sink.extend_from_slice(&buf[..n]);
                        took += n;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        abort(&mut child);
                        return Err(e.into());
                    }
                }
            }
        }
    }
    let status = child.wait()?;
    if !status.success() {
        let err = String::from_utf8_lossy(&err);
        return Err(Error::Msg(format!(
            "{label}: {}",
            err.lines().last().unwrap_or("failed").trim()
        )));
    }
    Ok(out)
}

fn journal_cmd(selector: &str, after: Option<&Cursor>) -> String {
    let from = match after {
        // Validated to [A-Za-z0-9=;_-] (see valid_cursor), so safe inside single quotes.
        Some(Cursor::Journal(c)) => format!(" --after-cursor='{c}'"),
        Some(Cursor::Since(us)) => format!(" --since=@{}", us / 1_000_000),
        None => String::new(),
    };
    // pipefail makes a journalctl failure (bad cursor) fail the pull instead of looking empty.
    format!(
        "set -o pipefail; journalctl --user {selector} -o json --output-fields=MESSAGE{from} | gzip -1"
    )
}

#[derive(Clone)]
enum Cursor {
    Journal(String),
    Since(i64),
}

#[derive(Deserialize)]
struct Entry {
    #[serde(rename = "__REALTIME_TIMESTAMP")]
    ts: String,
    #[serde(rename = "__CURSOR")]
    cursor: String,
    #[serde(rename = "MESSAGE", default)]
    message: serde_json::Value,
}

fn message_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        // journald exports non-UTF-8 messages as byte arrays.
        serde_json::Value::Array(a) => {
            let bytes: Vec<u8> = a
                .iter()
                .filter_map(|b| b.as_u64().map(|x| x as u8))
                .collect();
            Some(String::from_utf8_lossy(&bytes).into_owned())
        }
        _ => None,
    }
}

/// Outcome of one journal pull: new lines stored and lines too broken to read.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct JournalInsert {
    pub lines: usize,
    pub skipped: usize,
}

/// Outcome of one sar pull: samples upserted, and a warning when sadf reports a non-UTC zone (see `sar_ts`).
#[derive(Debug, Default)]
pub struct SarPull {
    pub samples: usize,
    pub warning: Option<String>,
}

/// Pulls one journal source into `raw` and returns the new lines; the database lock is never held across the transfer.
pub fn pull_journal(
    db: &Mutex<Connection>,
    host: &str,
    source: &str,
    selector: &str,
) -> Result<JournalInsert> {
    let key = format!("cursor:{source}");
    let (cursor, last) = {
        let conn = db.lock();
        (
            store::get_meta(&conn, &key)?
                .filter(|c| valid_cursor(c))
                .map(Cursor::Journal),
            store::last_ts(&conn, source)?,
        )
    };
    // No cursor but lines held: resume from the newest stored second; the cursor index drops the overlap.
    let first = match (cursor, last) {
        (None, Some(ts)) => Some(Cursor::Since(ts)),
        (c, _) => c,
    };
    let data = match ssh_gz(host, &journal_cmd(selector, first.as_ref())) {
        Ok(d) => d,
        // The journal was vacuumed past the cursor: restart from the last stored second; held lines are skipped by cursor.
        Err(_) if matches!(first, Some(Cursor::Journal(_))) => {
            let since = last.unwrap_or(0);
            ssh_gz(host, &journal_cmd(selector, Some(&Cursor::Since(since))))?
        }
        Err(e) => return Err(e),
    };
    insert_journal(&mut db.lock(), source, &data)
}

/// Inserts entries not held yet (cursors are unique); an unreadable line is skipped and counted so it cannot wedge the source.
pub(crate) fn insert_journal(
    conn: &mut Connection,
    source: &str,
    data: &[u8],
) -> Result<JournalInsert> {
    let tx = conn.transaction()?;
    let mut out = JournalInsert::default();
    let mut last_cursor = None;
    {
        let mut st = tx.prepare(
            "INSERT OR IGNORE INTO raw(source, ts, cursor, msg) VALUES (?1, ?2, ?3, ?4)",
        )?;
        // Split on bytes: `lines()` would fail the whole batch on one non-UTF-8 line.
        for raw in data.split(|b| *b == b'\n') {
            if raw.is_empty() {
                continue;
            }
            let line = String::from_utf8_lossy(raw);
            let Ok(e) = serde_json::from_str::<Entry>(&line) else {
                out.skipped += 1;
                continue;
            };
            let Some(ts) = e.ts.parse::<i64>().ok() else {
                out.skipped += 1;
                continue;
            };
            if !valid_cursor(&e.cursor) {
                out.skipped += 1;
                continue;
            }
            let Some(msg) = message_text(&e.message) else {
                last_cursor = Some(e.cursor);
                continue;
            };
            out.lines += st.execute(params![source, ts, e.cursor, msg])?;
            last_cursor = Some(e.cursor);
        }
    }
    if let Some(c) = last_cursor {
        store::set_meta(&tx, &format!("cursor:{source}"), &c)?;
    }
    tx.commit()?;
    Ok(out)
}

/// sar day files in `dir` as JSON, each after a `@@FILE name` line, charted activities only; `dir` passed `valid_path`.
fn sar_cmd(dir: &str) -> String {
    format!(
        "set -o pipefail; cd '{dir}' && for f in sa[0-9][0-9]; do \
         [ -f \"$f\" ] || continue; echo \"@@FILE $f\"; \
         sadf -j \"$f\" -- -u -r -S -q -n DEV -b; done | gzip -1"
    )
}

/// Pulls every sar day file still in `dir` on `host`; samples upsert by timestamp.
pub fn pull_sar(db: &Mutex<Connection>, host: &str, dir: &str, node: &str) -> Result<SarPull> {
    if !crate::discover::valid_path(dir) {
        return Err(Error::Msg(format!(
            "not a usable sysstat directory: {dir:?}"
        )));
    }
    let data = ssh_gz(host, &sar_cmd(dir))?;
    let text = String::from_utf8_lossy(&data);
    let mut conn = db.lock();
    let tx = conn.transaction()?;
    let mut out = SarPull::default();
    let mut non_utc = None;
    for chunk in text.split("@@FILE ").skip(1) {
        let Some((_, json)) = chunk.split_once('\n') else {
            continue;
        };
        let Ok(doc) = serde_json::from_str::<serde_json::Value>(json) else {
            continue;
        };
        let (n, tz) = insert_sar(&tx, node, &doc)?;
        out.samples += n;
        non_utc = non_utc.or(tz);
    }
    tx.commit()?;
    if let Some(tz) = non_utc {
        out.warning = Some(format!(
            "sar on {node} reports timestamps in {tz:?}, not UTC; they are parsed as UTC"
        ));
    }
    Ok(out)
}

fn insert_sar(
    tx: &Connection,
    node: &str,
    doc: &serde_json::Value,
) -> Result<(usize, Option<String>)> {
    let Some(stats) = doc["sysstat"]["hosts"][0]["statistics"].as_array() else {
        return Ok((0, None));
    };
    let mut st = tx.prepare_cached(
        "INSERT OR REPLACE INTO sar(node, ts, interval, mem_total, mem_avail, mem_used, mem_cached,
            swap_used, cpu_user, cpu_system, cpu_iowait, cpu_idle, load1, load5, io_rkb, io_wkb)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
    )?;
    let mut net = tx.prepare_cached(
        "INSERT OR REPLACE INTO sar_net(node, ts, iface, rx_kb, tx_kb) VALUES (?1,?2,?3,?4,?5)",
    )?;
    let mut n = 0;
    let mut non_utc = None;
    for s in stats {
        let Some((ts, tz)) = sar_ts(&s["timestamp"]) else {
            continue;
        };
        non_utc = non_utc.or(tz);
        let mem = &s["memory"];
        let cpu = s["cpu-load"]
            .as_array()
            .and_then(|a| a.iter().find(|c| c["cpu"] == "all"));
        let q = &s["queue"];
        let io = &s["io"];
        let f = |v: &serde_json::Value| v.as_f64();
        let i = |v: &serde_json::Value| v.as_i64();
        let total = match (f(&mem["memused"]), f(&mem["memused-percent"])) {
            (Some(u), Some(p)) if p > 0.0 => Some((u * 100.0 / p).round() as i64),
            _ => None,
        };
        st.execute(params![
            node,
            ts,
            s["timestamp"]["interval"].as_i64().unwrap_or(600),
            total,
            i(&mem["avail"]),
            i(&mem["memused"]),
            i(&mem["cached"]),
            i(&mem["swpused"]),
            cpu.and_then(|c| f(&c["user"])),
            cpu.and_then(|c| f(&c["system"])),
            cpu.and_then(|c| f(&c["iowait"])),
            cpu.and_then(|c| f(&c["idle"])),
            f(&q["ldavg-1"]),
            f(&q["ldavg-5"]),
            // sar reports 512-byte blocks per second.
            f(&io["io-reads"]["bread"]).map(|b| b / 2.0),
            f(&io["io-writes"]["bwrtn"]).map(|b| b / 2.0),
        ])?;
        if let Some(ifaces) = s["network"]["net-dev"].as_array() {
            for d in ifaces {
                let name = d["iface"].as_str().unwrap_or("");
                if name.is_empty() || name == "lo" {
                    continue;
                }
                net.execute(params![node, ts, name, f(&d["rxkB"]), f(&d["txkB"])])?;
            }
        }
        n += 1;
    }
    Ok((n, non_utc))
}

/// sadf -j timestamps in the zone named by `tz`; non-UTC zones still parse but are reported so the pull can warn.
fn sar_ts(t: &serde_json::Value) -> Option<(i64, Option<String>)> {
    let date = t["date"].as_str()?;
    let time = t["time"].as_str()?;
    let dt = chrono::NaiveDateTime::parse_from_str(&format!("{date} {time}"), "%Y-%m-%d %H:%M:%S")
        .ok()?;
    let tz = t["tz"].as_str().unwrap_or("UTC");
    let utc = matches!(tz, "UTC" | "GMT" | "UCT" | "Z");
    Some((
        dt.and_utc().timestamp_micros(),
        (!utc).then(|| tz.to_string()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_insert_and_floor() {
        let mut conn = store::memory();
        let data = br#"{"__REALTIME_TIMESTAMP":"100","__CURSOR":"s=a;i=1","MESSAGE":"one"}
{"__REALTIME_TIMESTAMP":"200","__CURSOR":"s=a;i=2","MESSAGE":[104,105]}
"#;
        assert_eq!(insert_journal(&mut conn, "rank0", data).unwrap().lines, 2);
        assert_eq!(
            store::get_meta(&conn, "cursor:rank0").unwrap().as_deref(),
            Some("s=a;i=2")
        );
        let msg: String = conn
            .query_row("SELECT msg FROM raw WHERE ts = 200", [], |r| r.get(0))
            .unwrap();
        assert_eq!(msg, "hi");
        // A since-based re-pull overlaps: held lines are skipped, a new line with a seen timestamp still lands.
        let more = br#"{"__REALTIME_TIMESTAMP":"200","__CURSOR":"s=a;i=2","MESSAGE":"hi"}
{"__REALTIME_TIMESTAMP":"200","__CURSOR":"s=a;i=3","MESSAGE":"same second"}
"#;
        assert_eq!(insert_journal(&mut conn, "rank0", more).unwrap().lines, 1);
        assert_eq!(
            store::get_meta(&conn, "cursor:rank0").unwrap().as_deref(),
            Some("s=a;i=3")
        );
    }

    #[test]
    fn broken_line_does_not_wedge_the_pull() {
        let mut conn = store::memory();
        // Truncated, non-numeric-timestamp and bad-cursor lines are skipped and counted; good lines land and the cursor advances.
        let data = br#"{"__REALTIME_TIMESTAMP":"100","__CURSOR":"s=a;i=1","MESSAGE":"one"}
{"__REALTIME_TIMESTAMP":
{"__REALTIME_TIMESTAMP":"nope","__CURSOR":"s=a;i=9","MESSAGE":"x"}
{"__REALTIME_TIMESTAMP":"300","__CURSOR":"bad cursor!","MESSAGE":"x"}
{"__REALTIME_TIMESTAMP":"400","__CURSOR":"s=a;i=4","MESSAGE":"four"}
"#;
        let r = insert_journal(&mut conn, "rank0", data).unwrap();
        assert_eq!((r.lines, r.skipped), (2, 3));
        assert_eq!(
            store::get_meta(&conn, "cursor:rank0").unwrap().as_deref(),
            Some("s=a;i=4")
        );
        // The next pull sees the cursor and asks only for what follows.
        assert_eq!(insert_journal(&mut conn, "rank0", data).unwrap().lines, 0);
    }

    #[test]
    fn invalid_utf8_line_does_not_abort_the_batch() {
        let mut conn = store::memory();
        let mut data = Vec::new();
        data.extend_from_slice(
            br#"{"__REALTIME_TIMESTAMP":"100","__CURSOR":"s=a;i=1","MESSAGE":"one"}"#,
        );
        data.push(b'\n');
        // A raw 0xFF byte is invalid UTF-8: lossy reading keeps the line and the cursor still advances.
        data.extend_from_slice(
            b"{\"__REALTIME_TIMESTAMP\":\"200\",\"__CURSOR\":\"s=a;i=2\",\"MESSAGE\":\"",
        );
        data.push(0xFF);
        data.extend_from_slice(b"\"}");
        data.push(b'\n');
        data.extend_from_slice(
            br#"{"__REALTIME_TIMESTAMP":"300","__CURSOR":"s=a;i=3","MESSAGE":"three"}"#,
        );
        let r = insert_journal(&mut conn, "rank0", &data).unwrap();
        assert_eq!((r.lines, r.skipped), (3, 0));
        assert_eq!(
            store::get_meta(&conn, "cursor:rank0").unwrap().as_deref(),
            Some("s=a;i=3")
        );
        let msg: String = conn
            .query_row("SELECT msg FROM raw WHERE ts = 200", [], |r| r.get(0))
            .unwrap();
        assert_eq!(msg, "\u{fffd}");
    }

    #[test]
    fn deadline_covers_inherited_pipes() {
        // The shell exits at once but a background child keeps stdout open: the deadline must still end the wait.
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "sleep 30 & echo started"]);
        let t = Instant::now();
        let r = run_bounded(cmd, "test", Duration::from_secs(1));
        assert!(r.is_err(), "{r:?}");
        assert!(t.elapsed() < Duration::from_secs(5));

        // A descendant that leaves the group (setsid) and keeps stdout is out of the group kill's reach, but we return on time.
        let mut esc = Command::new("sh");
        esc.args(["-c", "setsid sleep 30 & echo started"]);
        let t = Instant::now();
        assert!(run_bounded(esc, "test", Duration::from_secs(1)).is_err());
        assert!(t.elapsed() < Duration::from_secs(5));

        // Output that never ends still hits the deadline.
        let mut flood = Command::new("sh");
        flood.args(["-c", "exec cat /dev/zero"]);
        let t = Instant::now();
        assert!(run_bounded(flood, "test", Duration::from_secs(1)).is_err());
        assert!(t.elapsed() < Duration::from_secs(5));

        // Large output is drained while the child runs.
        let mut big = Command::new("sh");
        big.args(["-c", "head -c 3000000 /dev/zero"]);
        assert_eq!(
            run_bounded(big, "test", Duration::from_secs(10))
                .unwrap()
                .len(),
            3_000_000
        );

        let mut ok = Command::new("sh");
        ok.args(["-c", "printf hi"]);
        assert_eq!(
            run_bounded(ok, "test", Duration::from_secs(5)).unwrap(),
            b"hi"
        );
    }

    #[test]
    fn rejects_odd_hosts_and_cursors() {
        assert!(valid_host("node1"));
        assert!(valid_host("me@10.0.0.1"));
        assert!(!valid_host("-oProxyCommand=x"));
        assert!(!valid_host("a b"));
        assert!(!valid_host(""));
        assert!(valid_cursor("s=1c19;i=24801c;b=af95;m=1c87;t=65fe;x=f307"));
        assert!(!valid_cursor("s=1';touch x;'"));
    }

    #[test]
    fn journal_command_quotes_cursor() {
        let c = journal_cmd(
            "-t strixfold-container",
            Some(&Cursor::Journal("s=1;i=2".into())),
        );
        assert!(c.contains("--after-cursor='s=1;i=2'"));
        assert!(c.starts_with("set -o pipefail;"));
        let s = journal_cmd("-t x", Some(&Cursor::Since(5_000_000)));
        assert!(s.contains("--since=@5"));
    }

    #[test]
    fn sar_document() {
        let conn = store::memory();
        let doc: serde_json::Value = serde_json::from_str(
            r#"{"sysstat":{"hosts":[{"statistics":[{
              "timestamp":{"date":"2026-11-22","time":"06:20:08","tz":"UTC","interval":592},
              "cpu-load":[{"cpu":"all","user":0.03,"system":0.06,"iowait":0.0,"idle":99.9}],
              "io":{"io-reads":{"bread":0.69},"io-writes":{"bwrtn":30.86}},
              "memory":{"avail":11709392,"memused":115798020,"memused-percent":88.74,"cached":12879396,"swpused":171940},
              "queue":{"ldavg-1":0.0,"ldavg-5":0.01},
              "network":{"net-dev":[{"iface":"lo","rxkB":1,"txkB":1},{"iface":"eth1","rxkB":12.5,"txkB":3.0}]}
            }]}]}}"#,
        )
        .unwrap();
        assert_eq!(insert_sar(&conn, "rank0", &doc).unwrap().0, 1);
        let avail: i64 = conn
            .query_row("SELECT mem_avail FROM sar", [], |r| r.get(0))
            .unwrap();
        assert_eq!(avail, 11709392);
        let total: i64 = conn
            .query_row("SELECT mem_total FROM sar", [], |r| r.get(0))
            .unwrap();
        assert!((130_490_000..130_500_000).contains(&total), "{total}");
        let ifaces: i64 = conn
            .query_row("SELECT COUNT(*) FROM sar_net", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ifaces, 1);
        // A non-UTC zone is reported instead of silently shifting times.
        let mut local = doc.clone();
        local["sysstat"]["hosts"][0]["statistics"][0]["timestamp"]["tz"] =
            serde_json::Value::from("CST");
        assert_eq!(
            insert_sar(&conn, "rank1", &local).unwrap().1.as_deref(),
            Some("CST")
        );
    }
}

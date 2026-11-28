//! Finds the pair from one ssh host, read-only: run dirs' `resolved-rank<R>.json`, the conf they name, and `/health`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::sync::{self, Settings};

/// StrixFold's default `RUN_ROOT`, relative to `$HOME`.
pub const DEFAULT_RUN_ROOT: &str = "tfp/run/serve";

/// What discovery found, with the saved overrides applied.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Topology {
    pub rank0_host: String,
    pub rank1_host: String,
    pub rank0_name: String,
    pub rank1_name: String,
    pub serve_port: Option<u32>,
    pub link_ifaces: Vec<String>,
    pub rank0_sa_dir: Option<String>,
    pub rank1_sa_dir: Option<String>,
    pub run_root: String,
    /// Both ranks' lines are read from the journal (see `rank0_journal`).
    pub journal: bool,
    /// Rank 0's lines come from the journal when its newest entry is no older than the latest run, else from run files.
    #[serde(default)]
    pub rank0_journal: bool,
    #[serde(default)]
    pub rank1_journal: bool,
    pub watchdog: bool,
    pub model: Option<String>,
    pub context: Option<u64>,
    pub lanes: Option<u32>,
    pub discovered_at: String,
}

impl Topology {
    /// Saved overrides win over what was discovered.
    pub fn with_overrides(mut self, s: &Settings) -> Topology {
        if let Some(h) = s.rank0_host.as_ref().filter(|h| sync::valid_host(h)) {
            self.rank0_host = h.clone();
        }
        if let Some(h) = s.rank1_host.as_ref().filter(|h| sync::valid_host(h)) {
            self.rank1_host = h.clone();
        }
        if let Some(l) = &s.link_ifaces {
            self.link_ifaces = l.iter().filter(|i| valid_iface(i)).cloned().collect();
        }
        if let Some(r) = s.run_root.as_ref().filter(|r| valid_path(r)) {
            self.run_root = r.clone();
        }
        self
    }
}

/// A path we may put inside single quotes on the remote shell.
pub fn valid_path(p: &str) -> bool {
    !p.is_empty()
        && p.len() <= 512
        && !p.starts_with('-')
        && !p.split('/').any(|c| c == "..")
        && p.chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-+".contains(c))
}

fn valid_iface(i: &str) -> bool {
    !i.is_empty()
        && i.len() <= 15
        && i.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
}

/// One box's answer to the probe.
#[derive(Debug, Default)]
struct Probe {
    latest: Option<String>,
    resolved: Vec<Resolved>,
    sa_dir: Option<String>,
    /// (interface, addresses without prefix length)
    ifaces: Vec<(String, Vec<String>)>,
    /// µs of the newest `strixfold-container` journal entry.
    journal_last: Option<i64>,
    /// journalctl's exit status: 0 also when it simply has no entries.
    journal_rc: i32,
    watchdog: bool,
}

/// Journal lines around a run's start (seconds), message text only; pipefail makes a failed read an error.
fn journal_window_cmd(start: i64) -> String {
    format!(
        "set -o pipefail; journalctl --user -t strixfold-container -o cat -q --since=@{} --until=@{} | gzip -1",
        start - 1800,
        start + 600
    )
}

/// The run's own banner is among these journal lines.
fn banner_in(lines: &str, run: &str) -> bool {
    let tag = format!(": run {run},");
    lines
        .lines()
        .any(|l| l.starts_with("== ") && l.contains(" strixfold rank ") && l.contains(&tag))
}

/// Does `rank` on `host` log its current run to the journal? Only if it holds that run's banner; a failed check is an error.
fn journal_active(host: &str, p: &Probe, rank: u8) -> Result<bool> {
    if p.journal_last.is_none() {
        return Ok(false);
    }
    let run = p.latest.as_deref().filter(|r| {
        r.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_-".contains(c))
    });
    let start = p
        .resolved
        .iter()
        .find(|r| r.rank == rank)
        .and_then(|r| chrono::DateTime::parse_from_rfc3339(&r.utc).ok())
        .map(|t| t.timestamp());
    let (Some(run), Some(start)) = (run, start) else {
        return Ok(true);
    };
    let d = sync::ssh_gz(host, &journal_window_cmd(start))
        .map_err(|e| Error::Msg(format!("rank {rank}'s journal at {host:?}: {e}")))?;
    Ok(banner_in(&String::from_utf8_lossy(&d), run))
}

/// The fields of `resolved-rank<R>.json` (`schema tf-port-serve/1`) we use.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Resolved {
    pub rank: u8,
    pub host: String,
    #[serde(default)]
    pub utc: String,
    #[serde(default)]
    pub config: String,
    #[serde(default)]
    pub config_sha256: String,
    #[serde(default)]
    pub deploy: ResolvedDeploy,
    #[serde(default)]
    pub argv: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ResolvedDeploy {
    #[serde(default)]
    pub dir: String,
}

impl Resolved {
    /// The value after `flag` in argv.
    pub fn arg(&self, flag: &str) -> Option<&str> {
        let i = self.argv.iter().position(|a| a == flag)?;
        self.argv.get(i + 1).map(|s| s.as_str())
    }
}

fn probe_cmd(run_root: &str) -> String {
    // run_root passed valid_path (safe inside single quotes); a relative root is relative to $HOME, where ssh starts.
    format!(
        "R='{run_root}'; {{ \
         echo '@@latest'; readlink \"$R/latest\"; \
         for f in \"$R\"/latest/resolved-rank*.json; do [ -f \"$f\" ] && {{ echo '@@resolved'; cat \"$f\"; echo; }}; done; \
         echo '@@sa'; ls -d /var/log/sysstat /var/log/sa 2>/dev/null; \
         echo '@@ip'; ip -br addr 2>/dev/null; \
         echo '@@journal'; journalctl --user -t strixfold-container -n 1 -q -o short-unix 2>/dev/null; \
         echo \"@@jrc\"; echo $?; \
         echo '@@watchdog'; journalctl --user -u strixfold-watchdog -n 1 -q -o short-unix 2>/dev/null; \
         echo '@@end'; }} | gzip -1"
    )
}

fn parse_probe(text: &str) -> Probe {
    let mut p = Probe::default();
    for section in text.split("@@").skip(1) {
        let (name, body) = section.split_once('\n').unwrap_or((section, ""));
        let body = body.trim();
        match name.trim() {
            "latest" => {
                p.latest = body
                    .lines()
                    .next()
                    .and_then(|l| l.trim_end_matches('/').rsplit('/').next())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            }
            "resolved" => {
                if let Ok(r) = serde_json::from_str::<Resolved>(body) {
                    p.resolved.push(r);
                }
            }
            "sa" => {
                p.sa_dir = body
                    .lines()
                    .next()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            }
            "ip" => {
                p.ifaces = body
                    .lines()
                    .filter_map(|l| {
                        let mut f = l.split_whitespace();
                        let name = f.next()?.split('@').next()?.to_string();
                        let _state = f.next();
                        let addrs = f
                            .map(|a| a.split('/').next().unwrap_or(a).to_string())
                            .collect();
                        Some((name, addrs))
                    })
                    .collect()
            }
            "journal" => {
                // `short-unix`: "<seconds.micros> host tag[pid]: message"
                p.journal_last = body
                    .split_whitespace()
                    .next()
                    .and_then(|t| t.parse::<f64>().ok())
                    .map(|s| (s * 1e6) as i64)
            }
            "jrc" => p.journal_rc = body.parse().unwrap_or(-1),
            "watchdog" => p.watchdog = !body.is_empty(),
            _ => {}
        }
    }
    p
}

/// `KEY=value` lines of a sourced bash conf; comments and quotes stripped.
pub fn parse_conf(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        if k.is_empty()
            || !k
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        {
            continue;
        }
        let v = v.trim_start();
        let v = if let Some(rest) = v.strip_prefix('\'') {
            rest.split('\'').next().unwrap_or("")
        } else if let Some(rest) = v.strip_prefix('"') {
            rest.split('"').next().unwrap_or("")
        } else {
            v.split_whitespace().next().unwrap_or("")
        };
        out.insert(k.to_string(), v.to_string());
    }
    out
}

/// The NICs on rank 0 that hold the link addresses, master first.
fn link_ifaces(ifaces: &[(String, Vec<String>)], addrs: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for a in addrs.iter().filter(|a| !a.is_empty()) {
        if let Some((name, _)) = ifaces.iter().find(|(_, ips)| ips.iter().any(|ip| ip == a))
            && !out.contains(name)
        {
            out.push(name.clone());
        }
    }
    out
}

#[derive(Deserialize)]
struct Health {
    model: Option<String>,
    context_length: Option<u64>,
    max_batch_size: Option<u32>,
}

/// journalctl failing (not just finding nothing) is an error, not "no journal", or the run dirs would be read too.
fn check_journal(host: &str, p: &Probe) -> Result<()> {
    if p.journal_rc != 0 {
        return Err(Error::Msg(format!(
            "{host}: journalctl --user failed (exit {}); cannot tell where the ranks log",
            p.journal_rc
        )));
    }
    Ok(())
}

fn probe(host: &str, run_root: &str) -> Result<Probe> {
    let data = sync::ssh_gz(host, &probe_cmd(run_root))?;
    Ok(parse_probe(&String::from_utf8_lossy(&data)))
}

/// Discover the pair from `settings.host` (either node), honoring overrides.
pub fn discover(settings: &Settings) -> Result<Topology> {
    let host = settings
        .host
        .as_deref()
        .filter(|h| sync::valid_host(h))
        .ok_or_else(|| Error::Msg("no host set up".into()))?;
    let run_root = settings
        .run_root
        .clone()
        .filter(|r| valid_path(r))
        .unwrap_or_else(|| DEFAULT_RUN_ROOT.to_string());

    let first = probe(host, &run_root)?;
    check_journal(host, &first)?;
    let me = first.resolved.first().cloned().ok_or_else(|| {
        Error::Msg(format!(
            "{host}: no StrixFold run found under {run_root}/latest (has the pair been started with first_light.sh or its unit?)"
        ))
    })?;

    // The conf is byte-identical on both boxes; read it where we are.
    let conf = if valid_path(&me.config) && me.config.starts_with('/') {
        sync::ssh_gz(host, &format!("cat '{}' | gzip -1", me.config))
            .map(|d| parse_conf(&String::from_utf8_lossy(&d)))
            .unwrap_or_default()
    } else {
        Default::default()
    };
    let conf_host = |k: &str| conf.get(k).filter(|h| sync::valid_host(h)).cloned();
    let pick = |over: &Option<String>, found: Option<String>| {
        over.clone()
            .filter(|h| sync::valid_host(h))
            .or(found)
            .unwrap_or_default()
    };
    let (found0, found1) = if me.rank == 0 {
        (Some(host.to_string()), conf_host("RANK1_HOST"))
    } else {
        (conf_host("RANK0_HOST"), Some(host.to_string()))
    };
    let rank0_host = pick(&settings.rank0_host, found0);
    let rank1_host = pick(&settings.rank1_host, found1);
    if !rank0_host.is_empty() && rank0_host == rank1_host {
        return Err(Error::Msg(format!(
            "rank 0 and rank 1 both resolve to {rank0_host:?}; set the other node under Overrides"
        )));
    }
    if rank0_host.is_empty() || rank1_host.is_empty() {
        return Err(Error::Msg(format!(
            "{host} is rank {}, but its conf names no ssh host for the other rank; set it under Overrides",
            me.rank
        )));
    }

    // Each rank's box at its effective host; the first answer is reused only where it came from that host.
    let mut probes: HashMap<String, Probe> = HashMap::new();
    probes.insert(host.to_string(), first);
    for (rank, h) in [(0u8, &rank0_host), (1, &rank1_host)] {
        if !probes.contains_key(h.as_str()) {
            let p = probe(h, &run_root)
                .and_then(|p| check_journal(h, &p).map(|_| p))
                .map_err(|e| {
                    Error::Msg(format!(
                        "rank {rank} at {h:?} (from the conf or overrides): {e}"
                    ))
                })?;
            probes.insert(h.clone(), p);
        }
    }
    let p0 = &probes[rank0_host.as_str()];
    let p1 = &probes[rank1_host.as_str()];
    // Each box must hold its own rank's run, or swapped or repeated hosts would pull one box's lines as the other rank's.
    for (rank, h, p) in [(0u8, &rank0_host, p0), (1, &rank1_host, p1)] {
        if !p.resolved.iter().any(|r| r.rank == rank) {
            return Err(Error::Msg(format!(
                "{h:?} has no rank {rank} run under {run_root}/latest; check the rank hosts under Overrides"
            )));
        }
    }
    let res = |p: &Probe, rank: u8| p.resolved.iter().find(|r| r.rank == rank).cloned();
    let r0 = res(p0, 0);
    let r1 = res(p1, 1);

    let serve_port = r0
        .as_ref()
        .and_then(|r| r.arg("--port"))
        .and_then(|p| p.parse().ok());
    // Link addresses: the conf's, else rank 0's stripe env and rank 1's master argument (`tf-dsv41-lanes MODEL CACHE 1 2 MASTER_IP ...`).
    let master = conf
        .get("MASTER_IP")
        .cloned()
        .or_else(|| r1.as_ref().and_then(|r| r.argv.get(5).cloned()))
        .unwrap_or_default();
    let stripe = conf
        .get("TP_STRIPE")
        .cloned()
        .or_else(|| r0.as_ref().and_then(|r| r.env.get("TF_TP_STRIPE").cloned()))
        .unwrap_or_default();

    let j0 = journal_active(&rank0_host, p0, 0)?;
    let j1 = journal_active(&rank1_host, p1, 1)?;
    let mut topo = Topology {
        rank0_name: r0
            .as_ref()
            .map(|r| r.host.clone())
            .unwrap_or_else(|| rank0_host.clone()),
        rank1_name: r1
            .as_ref()
            .map(|r| r.host.clone())
            .unwrap_or_else(|| rank1_host.clone()),
        rank0_host,
        rank1_host,
        serve_port,
        link_ifaces: link_ifaces(&p0.ifaces, &[&master, &stripe]),
        rank0_sa_dir: p0.sa_dir.clone(),
        rank1_sa_dir: p1.sa_dir.clone(),
        run_root,
        journal: j0 && j1,
        rank0_journal: j0,
        rank1_journal: j1,
        watchdog: p0.watchdog,
        model: r0
            .as_ref()
            .and_then(|r| r.arg("--name"))
            .map(str::to_string),
        context: r0
            .as_ref()
            .and_then(|r| r.arg("--context"))
            .and_then(|c| c.parse().ok()),
        lanes: r0
            .as_ref()
            .and_then(|r| r.arg("--parallel"))
            .and_then(|c| c.parse().ok()),
        discovered_at: chrono::Utc::now().to_rfc3339(),
    };

    // The server's own view, when it is up (it may listen on 127.0.0.1 only).
    if let Some(port) = topo.serve_port
        && let Ok(d) = sync::ssh_gz(
            &topo.rank0_host,
            &format!("curl -s -m 3 http://127.0.0.1:{port}/health 2>/dev/null | gzip -1"),
        )
        && let Ok(h) = serde_json::from_slice::<Health>(&d)
    {
        topo.model = h.model.or(topo.model);
        topo.context = h.context_length.or(topo.context);
        topo.lanes = h.max_batch_size.or(topo.lanes);
    }
    Ok(topo.with_overrides(settings))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANK0: &str = r#"{
 "schema": "tf-port-serve/1", "rank": 0, "host": "node0", "utc": "2026-11-25T23:04:48Z",
 "config": "/home/u/.config/strixfold/serve.conf", "config_sha256": "1758b0df32d3141101b4",
 "deploy": {"dir": "/home/u/tfp/deploy/523511c0f877", "commit": "523511c0f877"},
 "argv": ["/x/tensorfold-native", "serve", "/m", "--host", "0.0.0.0", "--port", "8731",
          "--name", "deepseek-v4.1-flash", "--context", "1048527", "--parallel", "4"],
 "env": {"HOME": "/home/u", "TF_TP_STRIPE": "10.40.1.1"}
}"#;

    fn probe_text(resolved: &str) -> String {
        format!(
            "@@latest\ntfp/run/serve/strixfold-20261125T230446Z\n@@resolved\n{resolved}\n\
             @@sa\n/var/log/sysstat\n@@ip\n\
             lo               UNKNOWN        127.0.0.1/8 ::1/128 \n\
             enp97s0          DOWN           \n\
             thunderbolt0     UP             10.40.0.1/24 fe80::3e:47ff:feee:fc69/64 \n\
             thunderbolt1@x   UP             10.40.1.1/24 \n\
             @@journal\n@@jrc\n0\n@@watchdog\n1764112000.123456 node0 strixfold-watchdog[1]: ok\n@@end\n"
        )
    }

    #[test]
    fn probe_sections() {
        let p = parse_probe(&probe_text(RANK0));
        assert_eq!(p.latest.as_deref(), Some("strixfold-20261125T230446Z"));
        assert_eq!(p.resolved.len(), 1);
        let r = &p.resolved[0];
        assert_eq!((r.rank, r.host.as_str()), (0, "node0"));
        assert_eq!(r.arg("--port"), Some("8731"));
        assert_eq!(p.sa_dir.as_deref(), Some("/var/log/sysstat"));
        assert!(p.journal_last.is_none());
        assert!(p.watchdog);
        assert_eq!(
            link_ifaces(&p.ifaces, &["10.40.0.1", "10.40.1.1"]),
            vec!["thunderbolt0", "thunderbolt1"]
        );
        // An empty probe (no run yet) parses to nothing rather than failing.
        assert!(check_journal("node0", &p).is_ok());
        // journalctl failing is an error, not "no journal".
        let bad = parse_probe(&probe_text(RANK0).replace("@@jrc\n0", "@@jrc\n1"));
        assert_eq!(bad.journal_rc, 1);
        assert!(check_journal("node0", &bad).is_err());
        let e = parse_probe("@@latest\n@@sa\n@@ip\n@@journal\n@@watchdog\n@@end\n");
        assert!(e.journal_last.is_none());
        assert!(e.latest.is_none() && e.resolved.is_empty() && e.sa_dir.is_none());
    }

    #[test]
    fn journal_holds_the_run() {
        let lines = "== 2026-11-25T23:04:48Z strixfold rank 0 on node0: run strixfold-20261125T230446Z, conf /c (1758b0df32d3), deploy /d/523511c0f877\n\
                     == argv: /x serve\ninfo: ran\n";
        assert!(banner_in(lines, "strixfold-20261125T230446Z"));
        // Another run's banner, or the name in some other line, is not it.
        assert!(!banner_in(lines, "strixfold-20261125T230446"));
        assert!(!banner_in("info: run strixfold-x, conf", "strixfold-x"));
        let p = parse_probe(&probe_text(RANK0).replace(
            "@@journal\n",
            "@@journal\n1795647890.5 node0 strixfold-container[9]: x\n",
        ));
        assert_eq!(p.journal_last, Some(1_795_647_890_500_000));
        assert!(
            journal_window_cmd(1_795_647_888).contains("--since=@1795646088 --until=@1795648488")
        );
    }

    #[test]
    fn conf_values() {
        let c = parse_conf(
            "# comment\nWORLD=2\nRANK0_HOST=node0                 # ssh name from node1\n\
             export RANK1_HOST=node1\nRUN_ROOT='$HOME/tfp/run/serve'   # each run\nSERVE_NAME=\"a b\"\n\
             TP_DATA_PORT=\n  lower=x\n",
        );
        assert_eq!(c["RANK0_HOST"], "node0");
        assert_eq!(c["RANK1_HOST"], "node1");
        assert_eq!(c["RUN_ROOT"], "$HOME/tfp/run/serve");
        assert_eq!(c["SERVE_NAME"], "a b");
        assert_eq!(c["TP_DATA_PORT"], "");
        assert!(!c.contains_key("lower"));
    }

    #[test]
    fn paths_and_overrides() {
        assert!(valid_path("tfp/run/serve"));
        assert!(valid_path("/home/u/.config/strixfold/serve.conf"));
        assert!(!valid_path("a'b"));
        assert!(!valid_path("../etc"));
        assert!(!valid_path("-x"));
        assert!(!valid_path("$HOME/x"));
        let t = Topology {
            rank0_host: "node0".into(),
            rank1_host: "node1".into(),
            link_ifaces: vec!["thunderbolt0".into()],
            run_root: DEFAULT_RUN_ROOT.into(),
            ..Default::default()
        };
        let s = Settings {
            host: Some("node0".into()),
            rank1_host: Some("me@10.0.0.2".into()),
            link_ifaces: Some(vec!["a".into(), "bad name".into()]),
            ..Default::default()
        };
        let t = t.with_overrides(&s);
        assert_eq!(t.rank1_host, "me@10.0.0.2");
        assert_eq!(t.link_ifaces, vec!["a"]);
        assert_eq!(t.run_root, DEFAULT_RUN_ROOT);
    }
}

/// Against a real pair, read-only: `STRIXSCOPE_DISCOVER_HOST=<ssh host> cargo test real_pair -- --ignored --nocapture`
#[cfg(test)]
mod real {
    #[test]
    #[ignore]
    fn real_pair() {
        let host = std::env::var("STRIXSCOPE_DISCOVER_HOST").expect("set STRIXSCOPE_DISCOVER_HOST");
        let s = crate::sync::Settings {
            host: Some(host),
            ..Default::default()
        };
        let t = super::discover(&s).unwrap();
        println!("{t:#?}");
        // STRIXSCOPE_DB_IN=<a copy of strixscope.db> pulls into that copy.
        let db = parking_lot::Mutex::new(match std::env::var("STRIXSCOPE_DB_IN") {
            Ok(p) => crate::store::open(std::path::Path::new(&p)).unwrap(),
            Err(_) => crate::store::memory(),
        });
        for (rank, h, n) in [
            (0, &t.rank0_host, &t.rank0_name),
            (1, &t.rank1_host, &t.rank1_name),
        ] {
            let t0 = std::time::Instant::now();
            let r = crate::runlog::pull(&db, h, n, &t.run_root, rank).unwrap();
            println!("rank {rank} run logs: {r:?} in {:?}", t0.elapsed());
            let again = crate::runlog::pull(&db, h, n, &t.run_root, rank).unwrap();
            println!("rank {rank} again: {again:?}");
        }
        let s = crate::derive::rebuild(&mut db.lock(), None).unwrap();
        println!("{s:?}");
    }
}

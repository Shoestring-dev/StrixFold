//! Rebuilds runs, requests, events and http from `raw`; request lines carry no id, so they match FIFO on prompt tokens.

use std::collections::{HashMap, VecDeque};

use rusqlite::{Connection, params};

use crate::error::Result;
use crate::parse::{self, Line};

/// Requests this small with at most one token are health probes; views hide them by default.
const PROBE_MAX_PROMPT: u64 = 16;
const PROBE_MAX_GEN: u64 = 1;
/// An access line farther than this from its finish belongs to something else (a 4xx without a `refused` line, say).
const ACCESS_SLACK_US: i64 = 2_000_000;
const COMPLETION_PATHS: [&str; 3] = ["/v1/chat/completions", "/v1/completions", "/v1/messages"];

#[derive(Default, Debug, Clone)]
struct Run {
    rank: u8,
    ts: i64,
    end_ts: Option<i64>,
    name: String,
    conf: String,
    deploy: String,
    port: Option<u32>,
    parallel: Option<u32>,
    context: Option<u64>,
    model: Option<String>,
    loaded_s: Option<f64>,
    exit_status: Option<i32>,
}

struct Pending {
    t0: i64,
    t1: Option<i64>,
    prompt: u64,
    slot: u32,
    others: u32,
    resume: parse::Resume,
    /// Streaming replies log their access line before the prompt pass; it is parked here until the finish.
    access: Option<Access>,
}

#[derive(Clone)]
struct Access {
    ts: i64,
    client: String,
    path: String,
    status: u16,
}

#[derive(Default, Debug, Clone)]
pub struct Request {
    pub run: Option<usize>,
    pub req_id: String,
    pub t_start: Option<i64>,
    pub t_prefill: Option<i64>,
    pub t_end: i64,
    pub prompt: u64,
    pub cached: u64,
    pub generated: u64,
    pub thinking: Option<bool>,
    pub effort: Option<String>,
    pub finish: String,
    pub rounds: Option<u64>,
    pub accepted: Option<u64>,
    pub drafted: Option<u64>,
    pub slot: Option<u32>,
    pub others: Option<u32>,
    pub resume: Option<&'static str>,
    pub client: Option<String>,
    pub path: Option<String>,
    pub status: Option<u16>,
}

struct Event {
    run: Option<usize>,
    ts: i64,
    kind: &'static str,
    v: [Option<f64>; 5],
    text: Option<String>,
}

fn ev(run: Option<usize>, ts: i64, kind: &'static str, v: &[f64]) -> Event {
    let mut a = [None; 5];
    for (i, x) in v.iter().take(5).enumerate() {
        a[i] = Some(*x);
    }
    Event {
        run,
        ts,
        kind,
        v: a,
        text: None,
    }
}

#[derive(Default)]
pub struct Derived {
    runs: Vec<Run>,
    pub requests: Vec<Request>,
    events: Vec<Event>,
}

/// Derives everything from one source's lines (`rank0` or `rank1`), given as (ts µs, message) in journal order.
fn derive_source(out: &mut Derived, lines: &[(i64, String)]) {
    let mut run: Option<usize> = None;
    let mut pending: Vec<Pending> = Vec::new();
    // Finished requests waiting for their access line; None = a refusal.
    let mut awaiting: VecDeque<(Option<usize>, i64)> = VecDeque::new();
    // Access lines with nothing finished to claim them: a streamed reply's opening, claimed by the next prompt pass.
    let mut early: VecDeque<Access> = VecDeque::new();
    let mut last_ts = 0i64;

    let close = |out: &mut Derived, run: Option<usize>, ts: i64| {
        if let Some(r) = run
            && out.runs[r].end_ts.is_none()
        {
            out.runs[r].end_ts = Some(ts);
        }
    };

    for (ts, msg) in lines {
        let ts = *ts;
        let Some(line) = parse::parse(msg) else {
            last_ts = ts;
            continue;
        };
        match line {
            Line::RunStart {
                rank,
                name,
                conf,
                deploy,
            } => {
                close(out, run, last_ts.max(1));
                pending.clear();
                awaiting.clear();
                early.clear();
                out.runs.push(Run {
                    rank,
                    ts,
                    name,
                    conf,
                    deploy: parse::deploy_hash(&deploy),
                    ..Default::default()
                });
                run = Some(out.runs.len() - 1);
            }
            Line::Argv {
                port,
                parallel,
                context,
                ..
            } => {
                if let Some(r) = run {
                    let r = &mut out.runs[r];
                    // Rank 1's argv is positional; keep what rank 0 says.
                    r.port = r.port.or(port);
                    r.parallel = r.parallel.or(parallel);
                    r.context = r.context.or(context);
                }
            }
            Line::Serving { model, loaded_s } => {
                if let Some(r) = run {
                    out.runs[r].model = Some(model);
                    out.runs[r].loaded_s = loaded_s;
                }
            }
            Line::Exit { status, .. } => {
                if let Some(r) = run {
                    out.runs[r].end_ts = Some(ts);
                    out.runs[r].exit_status = Some(status);
                }
                out.events.push(ev(run, ts, "exit", &[status as f64]));
                pending.clear();
                awaiting.clear();
                early.clear();
            }
            Line::PromptPass {
                tokens,
                slot,
                resume,
                others,
            } => {
                while early.front().is_some_and(|a| ts - a.ts > ACCESS_SLACK_US) {
                    early.pop_front();
                }
                pending.push(Pending {
                    t0: ts,
                    t1: None,
                    prompt: tokens,
                    slot,
                    others,
                    resume,
                    access: early.pop_front(),
                })
            }
            Line::PromptCache {
                tokens,
                resumed_at,
                states,
                mib,
                cap_mib,
                hits,
                misses,
                evicted,
                ..
            } => {
                if let Some(p) = pending
                    .iter_mut()
                    .find(|p| p.prompt == tokens && p.t1.is_none())
                {
                    p.t1 = Some(ts);
                }
                let mut e = ev(
                    run,
                    ts,
                    "pcache",
                    &[
                        states as f64,
                        mib as f64,
                        hits as f64,
                        misses as f64,
                        evicted as f64,
                    ],
                );
                e.text = Some(format!("{cap_mib} {resumed_at}"));
                out.events.push(e);
            }
            Line::Done {
                id,
                prompt,
                cached,
                thinking,
                effort,
                tokens,
                finish,
                rounds,
                accepted,
                drafted,
            } => {
                let p = take_pending(&mut pending, prompt);
                let access = p.as_ref().and_then(|p| p.access.clone());
                out.requests.push(Request {
                    run,
                    req_id: id,
                    t_start: p.as_ref().map(|p| p.t0),
                    t_prefill: p.as_ref().and_then(|p| p.t1),
                    t_end: ts,
                    prompt,
                    cached,
                    generated: tokens,
                    thinking: Some(thinking),
                    effort: Some(effort),
                    finish,
                    rounds: Some(rounds),
                    accepted: Some(accepted),
                    drafted: Some(drafted),
                    slot: p.as_ref().map(|p| p.slot),
                    others: p.as_ref().map(|p| p.others),
                    resume: p.as_ref().map(|p| p.resume.as_str()),
                    ..Default::default()
                });
                finish_access(&mut out.requests, access, &mut awaiting, ts);
            }
            Line::Ended {
                id,
                reason,
                prompt,
                tokens,
                after_s,
            } => {
                let p = take_pending(&mut pending, prompt);
                let access = p.as_ref().and_then(|p| p.access.clone());
                let t_start = p
                    .as_ref()
                    .map(|p| p.t0)
                    .unwrap_or(ts - (after_s * 1e6) as i64);
                out.requests.push(Request {
                    run,
                    req_id: id,
                    t_start: Some(t_start),
                    t_prefill: p.as_ref().and_then(|p| p.t1),
                    t_end: ts,
                    prompt,
                    generated: tokens,
                    finish: reason,
                    slot: p.as_ref().map(|p| p.slot),
                    others: p.as_ref().map(|p| p.others),
                    resume: p.as_ref().map(|p| p.resume.as_str()),
                    ..Default::default()
                });
                finish_access(&mut out.requests, access, &mut awaiting, ts);
            }
            Line::Refused { why, .. } => {
                let mut e = ev(run, ts, "refused", &[]);
                e.text = Some(why);
                out.events.push(e);
                awaiting.push_back((None, ts));
            }
            Line::StreamError(why) => {
                let mut e = ev(run, ts, "stream_error", &[]);
                e.text = Some(why);
                out.events.push(e);
            }
            Line::Access {
                client,
                method,
                path,
                status,
            } => {
                if method == "POST" && COMPLETION_PATHS.contains(&path.as_str()) {
                    // Drop finishes that never got an access line.
                    while awaiting
                        .front()
                        .is_some_and(|(_, t)| ts - t > ACCESS_SLACK_US)
                    {
                        awaiting.pop_front();
                    }
                    match awaiting.pop_front() {
                        Some((Some(i), _)) => {
                            let r = &mut out.requests[i];
                            r.client = Some(client.clone());
                            r.path = Some(path.clone());
                            r.status = Some(status);
                        }
                        Some((None, _)) => {} // a refusal's access line
                        // Only a successful response can be a stream opening; an error with nothing to claim it stays unattached.
                        None if (200..300).contains(&status) => early.push_back(Access {
                            ts,
                            client: client.clone(),
                            path: path.clone(),
                            status,
                        }),
                        None => {}
                    }
                }
            }
            Line::DiskSave {
                rank,
                tokens,
                bytes,
                ms,
            } => out.events.push(ev(
                run,
                ts,
                "disk_save",
                &[tokens as f64, bytes as f64, ms, rank as f64],
            )),
            Line::DiskBlocks {
                rank,
                count,
                bytes,
                ms,
            } => out.events.push(ev(
                run,
                ts,
                "disk_blocks",
                &[count as f64, bytes as f64, ms, rank as f64],
            )),
            Line::DiskResume {
                rank,
                hit,
                tokens,
                bytes,
                ms,
            } => out.events.push(ev(
                run,
                ts,
                if hit { "disk_hit" } else { "disk_miss" },
                &[tokens as f64, bytes as f64, ms, rank as f64],
            )),
            Line::Stale { tokens } => out.events.push(ev(run, ts, "stale", &[tokens as f64])),
            Line::NoResume { tokens } => {
                out.events.push(ev(run, ts, "no_resume", &[tokens as f64]))
            }
            Line::CutIn {
                tokens,
                waits_at,
                of,
            } => out.events.push(ev(
                run,
                ts,
                "cut_in",
                &[tokens as f64, waits_at as f64, of as f64],
            )),
            Line::Restore => out.events.push(ev(run, ts, "restore", &[])),
            Line::ImageMemo {
                pictures,
                hits,
                misses,
            } => out.events.push(ev(
                run,
                ts,
                "image_memo",
                &[pictures as f64, hits as f64, misses as f64],
            )),
            Line::Encoder {
                pictures,
                encoded,
                cached,
                ms,
            } => out.events.push(ev(
                run,
                ts,
                "encoder",
                &[pictures as f64, encoded as f64, cached as f64, ms],
            )),
            Line::Structured { rows, us_per_row } => {
                out.events
                    .push(ev(run, ts, "structured", &[rows as f64, us_per_row]))
            }
            Line::MaxTokensThinking => out.events.push(ev(run, ts, "max_tokens_thinking", &[])),
            Line::GpuWait { rank, ms } => {
                out.events.push(ev(run, ts, "gpu_wait", &[ms, rank as f64]))
            }
        }
        last_ts = ts;
    }
    close(out, run, last_ts);
}

/// A finished request takes the access line its stream opened with, or waits for the one a plain reply logs next.
fn finish_access(
    requests: &mut [Request],
    access: Option<Access>,
    awaiting: &mut VecDeque<(Option<usize>, i64)>,
    ts: i64,
) {
    let i = requests.len() - 1;
    match access {
        Some(a) => {
            let r = &mut requests[i];
            r.client = Some(a.client);
            r.path = Some(a.path);
            r.status = Some(a.status);
        }
        None => awaiting.push_back((Some(i), ts)),
    }
}

/// Prefers a pending pass whose prompt pass already ended (the cache line came), else any with the same prompt length.
fn take_pending(pending: &mut Vec<Pending>, prompt: u64) -> Option<Pending> {
    let i = pending
        .iter()
        .position(|p| p.prompt == prompt && p.t1.is_some())
        .or_else(|| pending.iter().position(|p| p.prompt == prompt))?;
    Some(pending.remove(i))
}

/// Derive one source's lines; public for tests and the CLI check.
pub fn derive(lines_rank0: &[(i64, String)], lines_rank1: &[(i64, String)]) -> Derived {
    let mut out = Derived::default();
    derive_source(&mut out, lines_rank0);
    let rank0_runs = out.runs.len();
    derive_source(&mut out, lines_rank1);
    // Rank 1 runs inherit production status from the same-named rank 0 run started nearest in time (rank 1's argv has no --port).
    let mut by_name: HashMap<String, Vec<(i64, Option<u32>)>> = HashMap::new();
    for r in &out.runs[..rank0_runs] {
        by_name
            .entry(r.name.clone())
            .or_default()
            .push((r.ts, r.port));
    }
    for r in &mut out.runs[rank0_runs..] {
        if r.port.is_none() {
            r.port = rank0_port(&by_name, &r.name, r.ts);
        }
    }
    out
}

fn rank0_port(
    by_name: &HashMap<String, Vec<(i64, Option<u32>)>>,
    name: &str,
    ts: i64,
) -> Option<u32> {
    by_name
        .get(name)?
        .iter()
        .min_by_key(|(t, _)| (t - ts).abs())
        .and_then(|(_, p)| *p)
}

fn load(conn: &Connection, source: &str) -> Result<Vec<(i64, String)>> {
    // Time order: run-dir logs arrive run by run, so sort; same-µs lines keep their arrival (journal) order.
    let mut st =
        conn.prepare_cached("SELECT ts, msg FROM raw WHERE source = ?1 ORDER BY ts, id")?;
    let rows = st
        .query_map([source], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// When a run stopped, if its lines show it: an exit line, or lines over a second past its start; else unknown.
fn known_end(r: &Run) -> Option<i64> {
    let end = r.end_ts?;
    (r.exit_status.is_some() || end - r.ts > 1_000_000).then_some(end)
}

/// The rank 0 port in place longest (a run lasts to its known end, else the next start or now; ties go low); `over` wins if used.
fn production_port(runs: &[Run], over: Option<u32>, now: i64) -> (Option<u32>, Option<u32>) {
    let mut r0: Vec<&Run> = runs.iter().filter(|r| r.rank == 0).collect();
    if let Some(p) = over
        && r0.iter().any(|r| r.port == Some(p))
    {
        return (Some(p), None);
    }
    r0.sort_by_key(|r| r.ts);
    let mut time: HashMap<u32, i64> = HashMap::new();
    for (i, r) in r0.iter().enumerate() {
        let next = r0.get(i + 1).map(|n| n.ts);
        let until = match (known_end(r), next) {
            (Some(e), Some(n)) => e.min(n),
            (Some(e), None) => e,
            (None, Some(n)) => n,
            (None, None) => now,
        };
        if let Some(p) = r.port {
            *time.entry(p).or_default() += (until - r.ts).max(0);
        }
    }
    let found = time
        .into_iter()
        .max_by_key(|(p, t)| (*t, std::cmp::Reverse(*p)))
        .map(|(p, _)| p);
    (found, over)
}

/// Recreates runs/requests/events/http from the raw lines; `prod_port` overrides the production port.
pub fn rebuild(conn: &mut Connection, prod_port: Option<u32>) -> Result<RebuildStats> {
    let r0 = load(conn, "rank0")?;
    let r1 = load(conn, "rank1")?;
    let wd = load(conn, "watchdog")?;
    let mut d = derive(&r0, &r1);
    watchdog_events(&mut d, &wd);
    // The pull's own clock: a run from unstamped logs has no lines after its start but has been in place until now.
    let now = chrono::Utc::now().timestamp_micros();
    let (prod, prod_port_unused) = production_port(&d.runs, prod_port, now);

    let tx = conn.transaction()?;
    tx.execute_batch("DELETE FROM runs; DELETE FROM requests; DELETE FROM events;")?;
    {
        let mut st = tx.prepare(
            "INSERT INTO runs(id, node, rank, ts, end_ts, name, conf, deploy, port, parallel,
                              context, production, model, loaded_s, exit_status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        )?;
        for (i, r) in d.runs.iter().enumerate() {
            st.execute(params![
                i as i64,
                if r.rank == 0 { "rank0" } else { "rank1" },
                r.rank,
                r.ts,
                r.end_ts,
                r.name,
                r.conf,
                r.deploy,
                r.port,
                r.parallel,
                r.context.map(|c| c as i64),
                (prod.is_none() || r.port == prod) as i64,
                r.model,
                r.loaded_s,
                r.exit_status,
            ])?;
        }
        let mut st = tx.prepare(
            "INSERT INTO requests(run_id, req_id, t_start, t_prefill, t_end, prompt, cached, generated,
                 thinking, effort, finish, rounds, accepted, drafted, slot, others, resume,
                 client, path, status, probe)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21)",
        )?;
        for r in &d.requests {
            st.execute(params![
                r.run.map(|x| x as i64),
                r.req_id,
                r.t_start,
                r.t_prefill,
                r.t_end,
                r.prompt as i64,
                r.cached as i64,
                r.generated as i64,
                r.thinking,
                r.effort,
                r.finish,
                r.rounds.map(|x| x as i64),
                r.accepted.map(|x| x as i64),
                r.drafted.map(|x| x as i64),
                r.slot,
                r.others,
                r.resume,
                r.client,
                r.path,
                r.status,
                is_probe(r) as i64,
            ])?;
        }
        let mut st = tx.prepare(
            "INSERT INTO events(run_id, ts, kind, a, b, c, d, e, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        for e in &d.events {
            st.execute(params![
                e.run.map(|x| x as i64),
                e.ts,
                e.kind,
                e.v[0],
                e.v[1],
                e.v[2],
                e.v[3],
                e.v[4],
                e.text
            ])?;
        }
    }
    tx.commit()?;
    let matched = d.requests.iter().filter(|r| r.t_prefill.is_some()).count();
    Ok(RebuildStats {
        lines: r0.len() + r1.len() + wd.len(),
        runs: d.runs.len(),
        requests: d.requests.len(),
        matched,
        events: d.events.len(),
        prod_port: prod,
        prod_port_unused,
    })
}

fn is_probe(r: &Request) -> bool {
    r.prompt <= PROBE_MAX_PROMPT && r.generated <= PROBE_MAX_GEN
}

/// Watchdog lines worth keeping: failed checks and restart requests.
fn watchdog_events(d: &mut Derived, lines: &[(i64, String)]) {
    for (ts, msg) in lines {
        let Some(rest) = msg.strip_prefix("strixfold-watchdog: ") else {
            continue;
        };
        let kind = if rest.starts_with("requesting a restart") {
            "wd_restart"
        } else if rest.starts_with("check ") {
            "wd_check_failed"
        } else {
            "wd_note"
        };
        let mut e = ev(None, *ts, kind, &[]);
        e.text = Some(rest.to_string());
        d.events.push(e);
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RebuildStats {
    pub lines: usize,
    pub runs: usize,
    pub requests: usize,
    pub matched: usize,
    pub events: usize,
    pub prod_port: Option<u32>,
    /// The saved production port, when no run uses it (so it was ignored).
    pub prod_port_unused: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<(i64, String)> {
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let (ts, msg) = l.split_once(' ').unwrap();
                (ts.parse().unwrap(), msg.to_string())
            })
            .collect()
    }

    const RUN: &str = r#"
1000000 == 2026-11-21T22:08:56Z strixfold rank 0 on node0: run strixfold-20261121T220855Z, conf /c/serve.conf (070eb2add70b), deploy /d/tfp/deploy/ad367d8a2ede
1000001 == argv: /x serve /m --host 0.0.0.0 --port 8731 --name m --context 1048527 --parallel 4
2000000 [tensorfold] serving deepseek-v4.1-flash at http://0.0.0.0:8731/v1 (sampling: x; loaded in 51.0s)
3000000 info: lanes: step 1 rank 0 prompt pass: 6847 tokens, slot 0, extent at 0, 8192 rows; from 0: no kept state; 0 other stream(s) live; collectives
3100000 info: lanes: step 2 rank 0 prompt pass: 500 tokens, slot 1, extent at 0, 2048 rows; resumes state 1 at 256 in place; 1 other stream(s) live; collectives
3400000 info: prompt cache: 500 tokens, resumed at 256, kept 1; 1 states, 6 MiB and 0 MiB spare of 2048 MiB (hits 1, misses 0, evicted 0, refused 0, failed 0)
8500000 info: prompt cache: 6847 tokens, resumed at 0, kept 1; 2 states, 12 MiB and 0 MiB spare of 2048 MiB (hits 1, misses 1, evicted 0, refused 0, failed 0)
9000000 [tensorfold] done chatcmpl-b prompt=500 cached=256 thinking=False effort=none tokens=40 sha=x finish=stop rounds=10 accepted=30/40
9000000 [tensorfold] 10.0.0.2 "POST /v1/chat/completions HTTP/1.1" 200 -
9100000 [tensorfold] 127.0.0.1 "GET /health HTTP/1.1" 200 -
18500000 [tensorfold] done chatcmpl-a prompt=6847 cached=0 thinking=True effort=high tokens=590 sha=x finish=length rounds=200 accepted=389/600
18500000 [tensorfold] 10.0.0.3 "POST /v1/chat/completions HTTP/1.1" 200 -
20000000 info: lanes: step 3 rank 0 prompt pass: 7 tokens, slot 0, extent at 0, 2048 rows; from 0: no kept state; 0 other stream(s) live; collectives
20070000 info: prompt cache: 7 tokens, resumed at 0, kept 0; 2 states, 12 MiB and 0 MiB spare of 2048 MiB (hits 1, misses 2, evicted 0, refused 0, failed 0)
20070100 [tensorfold] done chatcmpl-p prompt=7 cached=0 thinking=False effort=none tokens=1 sha=x finish=length rounds=0 accepted=0/0
20070100 [tensorfold] 127.0.0.1 "POST /v1/chat/completions HTTP/1.1" 200 -
21000000 info: lanes: step 4 rank 0 prompt pass: 25308 tokens, slot 0, extent at 0, 26624 rows; from 0: no kept state; 0 other stream(s) live; collectives
31970000 [tensorfold] ended chatcmpl-e reason=client-left prompt=25308 tokens=314 after=10.97s
31970000 [tensorfold] 10.0.0.2 "POST /v1/chat/completions HTTP/1.1" 200 -
32990000 [tensorfold] 10.0.0.66 "POST /v1/chat/completions HTTP/1.1" 400 -
33000000 [tensorfold] 10.0.0.9 "POST /v1/chat/completions HTTP/1.1" 200 -
33000080 info: lanes: step 5 rank 0 prompt pass: 11 tokens, slot 0, extent at 0, 2048 rows; from 0: no kept state; 0 other stream(s) live; collectives
33090000 info: prompt cache: 11 tokens, resumed at 0, kept 0; 2 states, 12 MiB and 0 MiB spare of 2048 MiB (hits 1, misses 3, evicted 0, refused 0, failed 0)
33400000 [tensorfold] done chatcmpl-s prompt=11 cached=0 thinking=False effort=none tokens=15 sha=x finish=stop rounds=3 accepted=10/12
33500000 [tensorfold] 10.0.0.4 "POST /v1/chat/completions HTTP/1.1" 200 -
33500100 [tensorfold] done chatcmpl-n prompt=99 cached=0 thinking=False effort=none tokens=2 sha=x finish=stop rounds=1 accepted=1/1
33500100 [tensorfold] 10.0.0.5 "POST /v1/chat/completions HTTP/1.1" 200 -
40000000 == 2026-11-21T22:09:40Z rank 0 exited with status 0
"#;

    #[test]
    fn reconstructs_requests() {
        let d = derive(&lines(RUN), &[]);
        assert_eq!(d.runs.len(), 1);
        let run = &d.runs[0];
        assert_eq!(run.port, Some(8731));
        assert_eq!(run.deploy, "ad367d8a2ede");
        assert_eq!(run.loaded_s, Some(51.0));
        assert_eq!(run.end_ts, Some(40000000));
        assert_eq!(run.exit_status, Some(0));

        assert_eq!(d.requests.len(), 6);
        let b = &d.requests[0];
        assert_eq!(b.req_id, "chatcmpl-b");
        assert_eq!(
            (b.t_start, b.t_prefill, b.t_end),
            (Some(3100000), Some(3400000), 9000000)
        );
        assert_eq!(b.resume, Some("ram"));
        assert_eq!(b.others, Some(1));
        assert_eq!(b.client.as_deref(), Some("10.0.0.2"));

        // Finished out of order: still paired with its own prompt pass.
        let a = &d.requests[1];
        assert_eq!((a.t_start, a.t_prefill), (Some(3000000), Some(8500000)));
        assert_eq!(a.client.as_deref(), Some("10.0.0.3"));
        assert_eq!((a.accepted, a.drafted), (Some(389), Some(600)));

        assert!(is_probe(&d.requests[2]));
        assert!(!is_probe(a));

        let e = &d.requests[3];
        assert_eq!(e.finish, "client-left");
        assert_eq!(e.t_start, Some(21000000));
        assert_eq!(e.t_prefill, None);
        assert_eq!(e.generated, 314);

        // A streamed reply keeps its opening access line; the stray one after it does not steal the next plain reply's.
        assert_eq!(d.requests[4].client.as_deref(), Some("10.0.0.9"));
        assert_eq!(d.requests[4].t_start, Some(33000080));
        assert_eq!(d.requests[5].client.as_deref(), Some("10.0.0.5"));
    }

    #[test]
    fn production_port_is_the_longest_running() {
        const S: i64 = 1_000_000; // µs
        let run = |rank, ts: i64, end: Option<i64>, port| Run {
            rank,
            ts: ts * S,
            end_ts: end.map(|e| e * S),
            port,
            ..Default::default()
        };
        let now = 1000 * S;
        // Long production runs on 8731, short tests on 18731, and the current production run still going.
        let runs = [
            run(0, 0, Some(100), Some(8731)),
            run(0, 100, Some(130), Some(18731)),
            run(0, 130, Some(200), Some(18731)),
            run(0, 200, Some(990), Some(8731)),
            run(1, 0, Some(10_000), Some(18731)),
        ];
        assert_eq!(production_port(&runs, None, now), (Some(8731), None));
        assert_eq!(
            production_port(&runs, Some(18731), now),
            (Some(18731), None)
        );
        // An override no run uses is ignored and reported.
        assert_eq!(
            production_port(&runs, Some(8732), now),
            (Some(8731), Some(8732))
        );
        assert_eq!(production_port(&[run(0, 0, None, None)], None, now).0, None);
        // A two-minute test killed without an exit line, then two idle weeks: the test is credited two minutes, not the gap.
        let killed = [
            run(0, 0, Some(3600), Some(8731)),
            run(0, 4000, Some(4120), Some(18731)),
            run(0, 1_213_600, Some(1_213_700), Some(8731)),
        ];
        assert_eq!(production_port(&killed, None, 1_213_800 * S).0, Some(8731));
        // Unstamped logs: no ends known; the current run weighs until now.
        let bare = [run(0, 0, None, Some(8731)), run(0, 100, None, Some(18731))];
        assert_eq!(production_port(&bare, None, 150 * S).0, Some(8731));
        assert_eq!(production_port(&bare, None, 1000 * S).0, Some(18731));
    }

    /// Through `rebuild`, with only the run starts that a run-dir pull of unstamped logs stores.
    #[test]
    fn production_port_from_bare_run_starts() {
        let start = |ts: i64, run: &str, port: u32| {
            [
                (
                    ts,
                    format!(
                        "== 2026-11-21T22:08:56Z strixfold rank 0 on node0: run {run}, conf /c (070eb2add70b), deploy /d/ad367d8a2ede"
                    ),
                ),
                (
                    ts + 1,
                    format!("== argv: /x serve /m --host 0.0.0.0 --port {port} --name m"),
                ),
            ]
        };
        let fill = |lines: &[(i64, String)]| {
            let mut conn = crate::store::memory();
            for (i, (ts, m)) in lines.iter().enumerate() {
                conn.execute(
                    "INSERT INTO raw(source, ts, cursor, msg) VALUES ('rank0', ?1, ?2, ?3)",
                    params![ts, format!("i={i}"), m],
                )
                .unwrap();
            }
            rebuild(&mut conn, None).unwrap().prod_port
        };
        // A day on 8731, then a test run on 18731 started an hour ago: the test run has been in place since.
        let mut lines: Vec<(i64, String)> = start(1_000_000, "a", 8731).into();
        lines.extend(start(1_000_000 + 86_400_000_000, "b", 18731));
        assert_eq!(fill(&lines), Some(18731));
        // The same test run, but it exited after a minute: 8731 wins.
        lines.push((
            1_000_000 + 86_460_000_000,
            "== 2026-11-22T22:09:56Z rank 0 exited with status 0".into(),
        ));
        assert_eq!(fill(&lines), Some(8731));
    }

    #[test]
    fn rank1_takes_the_nearest_rank0_run_of_its_name() {
        let mut m: HashMap<String, Vec<(i64, Option<u32>)>> = HashMap::new();
        // One run dir used twice: production first, a test a day later.
        m.insert(
            "smoke".into(),
            vec![(1_000, Some(8731)), (86_400_000_000, Some(18731))],
        );
        assert_eq!(rank0_port(&m, "smoke", 1_500), Some(8731));
        assert_eq!(rank0_port(&m, "smoke", 86_400_000_900), Some(18731));
        assert_eq!(rank0_port(&m, "other", 1_500), None);
    }

    #[test]
    fn rebuild_roundtrip() {
        let mut conn = crate::store::memory();
        {
            let mut st = conn
                .prepare("INSERT INTO raw(source, ts, cursor, msg) VALUES ('rank0', ?1, ?2, ?3)")
                .unwrap();
            for (i, (ts, m)) in lines(RUN).into_iter().enumerate() {
                st.execute(params![ts, format!("i={i}"), m]).unwrap();
            }
        }
        let s = rebuild(&mut conn, None).unwrap();
        assert_eq!((s.runs, s.requests, s.matched), (1, 6, 4));
        let prod: i64 = conn
            .query_row("SELECT production FROM runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(prod, 1);
        let probes: i64 = conn
            .query_row("SELECT COUNT(*) FROM requests WHERE probe = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(probes, 1);
    }
}

/// Real-data check: `STRIXSCOPE_JOURNAL_DIR=<dir with r0.json.gz r1.json.gz> cargo test --release real_journal -- --ignored`
#[cfg(test)]
mod real {
    use std::io::Read;

    #[test]
    #[ignore]
    fn real_journal() {
        let dir = std::env::var("STRIXSCOPE_JOURNAL_DIR").expect("set STRIXSCOPE_JOURNAL_DIR");
        let mut conn = crate::store::memory();
        for (src, file) in [
            ("rank0", "r0.json.gz"),
            ("rank1", "r1.json.gz"),
            ("watchdog", "wd.json.gz"),
        ] {
            let mut raw = Vec::new();
            flate2::read::GzDecoder::new(std::fs::File::open(format!("{dir}/{file}")).unwrap())
                .read_to_end(&mut raw)
                .unwrap();
            let n = crate::sync::insert_journal(&mut conn, src, &raw).unwrap();
            println!("{src}: {} lines, {} skipped", n.lines, n.skipped);
        }
        let t = std::time::Instant::now();
        let s = super::rebuild(&mut conn, None).unwrap();
        println!("rebuild {:?} in {:?}", s, t.elapsed());
        // STRIXSCOPE_SAR=<rank 0 ssh host>,<rank 1 ssh host> also pulls sar.
        if let Ok(hosts) = std::env::var("STRIXSCOPE_SAR") {
            let db = parking_lot::Mutex::new(conn);
            for (host, node) in hosts.split(',').zip(["rank0", "rank1"]) {
                println!(
                    "sar {node}: {:?}",
                    crate::sync::pull_sar(&db, host, "/var/log/sysstat", node)
                        .map_err(|e| e.to_string())
                );
            }
            conn = db.into_inner();
        }
        if let Ok(out) = std::env::var("STRIXSCOPE_DATASET_OUT") {
            let d = crate::query::dataset(&conn, 0, i64::MAX, false).unwrap();
            std::fs::write(&out, serde_json::to_vec(&d).unwrap()).unwrap();
            println!(
                "dataset: {} requests, {} sar",
                d.requests.len(),
                d.sar.len()
            );
        }
        if let Ok(out) = std::env::var("STRIXSCOPE_DB_OUT") {
            let _ = std::fs::remove_file(&out);
            conn.execute("VACUUM INTO ?1", [&out]).unwrap();
        }
        let q = |sql: &str| -> String {
            conn.query_row(sql, [], |r| r.get::<_, Option<f64>>(0))
                .unwrap()
                .map(|v| format!("{v:.1}"))
                .unwrap_or("-".into())
        };
        let window = "t_end >= 1795140000000000"; // µs since the epoch
        println!(
            "current run: requests {} matched {}",
            q(&format!("SELECT COUNT(*) FROM requests WHERE {window}")),
            q(&format!(
                "SELECT COUNT(*) FROM requests WHERE {window} AND t_prefill IS NOT NULL"
            ))
        );
        let med = |expr: &str, cond: &str| -> String {
            let mut v: Vec<f64> = conn
                .prepare(&format!("SELECT {expr} FROM requests WHERE {cond}"))
                .unwrap()
                .query_map([], |r| r.get(0))
                .unwrap()
                .map(|x| x.unwrap())
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if v.is_empty() {
                return "-".into();
            }
            format!("n={} median {:.0}", v.len(), v[v.len() / 2])
        };
        println!(
            "prefill tok/s (>2K new): {}",
            med(
                "(prompt - cached) * 1e6 / (t_prefill - t_start)",
                &format!("{window} AND t_prefill > t_start AND prompt - cached > 2000")
            )
        );
        println!(
            "decode tok/s solo (>100): {}",
            med(
                "generated * 1e6 / (t_end - t_prefill)",
                &format!("{window} AND t_end > t_prefill AND generated > 100 AND others = 0")
            )
        );
        println!(
            "production runs {}",
            q("SELECT COUNT(*) FROM runs WHERE production = 1 AND rank = 0")
        );
        println!(
            "clients with null: {}",
            q("SELECT COUNT(*) FROM requests WHERE client IS NULL")
        );
        for kind in [
            "pcache",
            "disk_save",
            "disk_hit",
            "stale",
            "cut_in",
            "refused",
            "exit",
            "wd_restart",
            "gpu_wait",
            "restore",
        ] {
            println!(
                "event {kind}: {}",
                q(&format!(
                    "SELECT COUNT(*) FROM events WHERE kind = '{kind}'"
                ))
            );
        }
    }
}

/// Rebuilds a copy of an app database: `STRIXSCOPE_DB_IN=<copy of strixscope.db> cargo test real_db -- --ignored --nocapture`
#[cfg(test)]
mod real_db {
    #[test]
    #[ignore]
    fn real_db() {
        let path = std::env::var("STRIXSCOPE_DB_IN").expect("set STRIXSCOPE_DB_IN");
        let mut conn = crate::store::open(std::path::Path::new(&path)).unwrap();
        let s = super::rebuild(&mut conn, None).unwrap();
        println!("{s:?}");
        let mut st = conn
            .prepare("SELECT port, production, COUNT(*) FROM runs WHERE rank = 0 GROUP BY 1, 2")
            .unwrap();
        let rows: Vec<(Option<i64>, i64, i64)> = st
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        println!("port, production, runs: {rows:?}");
    }
}

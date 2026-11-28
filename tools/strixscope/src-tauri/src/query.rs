//! Read queries for the UI, which does the binning and statistics; Rust hands it the rows for a range.

use rusqlite::{Connection, params};
use serde::Serialize;

use crate::error::Result;
use crate::store;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestRow {
    pub run: Option<i64>,
    pub t_start: Option<i64>,
    pub t_prefill: Option<i64>,
    pub t_end: i64,
    pub prompt: i64,
    pub cached: i64,
    pub generated: i64,
    pub thinking: Option<bool>,
    pub effort: Option<String>,
    pub finish: Option<String>,
    pub rounds: Option<i64>,
    pub accepted: Option<i64>,
    pub drafted: Option<i64>,
    pub slot: Option<i64>,
    pub others: Option<i64>,
    pub resume: Option<String>,
    pub client: Option<String>,
    pub path: Option<String>,
    pub status: Option<i64>,
    pub probe: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EventRow {
    pub run: Option<i64>,
    pub ts: i64,
    pub kind: String,
    pub v: [Option<f64>; 5],
    pub text: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRow {
    pub id: i64,
    pub node: String,
    pub ts: i64,
    pub end_ts: Option<i64>,
    pub name: Option<String>,
    pub deploy: Option<String>,
    pub port: Option<i64>,
    pub parallel: Option<i64>,
    pub context: Option<i64>,
    pub production: bool,
    pub model: Option<String>,
    pub loaded_s: Option<f64>,
    pub exit_status: Option<i64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarRow {
    pub node: String,
    pub ts: i64,
    pub interval: i64,
    pub mem_total: Option<i64>,
    pub mem_avail: Option<i64>,
    pub mem_used: Option<i64>,
    pub mem_cached: Option<i64>,
    pub swap_used: Option<i64>,
    pub cpu_user: Option<f64>,
    pub cpu_system: Option<f64>,
    pub cpu_iowait: Option<f64>,
    pub cpu_idle: Option<f64>,
    pub load1: Option<f64>,
    pub load5: Option<f64>,
    pub io_rkb: Option<f64>,
    pub io_wkb: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetRow {
    pub node: String,
    pub ts: i64,
    pub iface: String,
    pub rx_kb: Option<f64>,
    pub tx_kb: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Dataset {
    pub from: i64,
    pub to: i64,
    pub runs: Vec<RunRow>,
    pub requests: Vec<RequestRow>,
    pub events: Vec<EventRow>,
    pub sar: Vec<SarRow>,
    pub net: Vec<NetRow>,
}

/// Rows whose run is a production run (or that have no run: watchdog).
const PROD: &str =
    "(?3 = 1 OR run_id IS NULL OR run_id IN (SELECT id FROM runs WHERE production = 1))";

/// Rows in the half-open range [from, to).
pub fn dataset(conn: &Connection, from: i64, to: i64, include_tests: bool) -> Result<Dataset> {
    let tests = include_tests as i64;
    let runs = conn
        .prepare(
            "SELECT id, node, ts, end_ts, name, deploy, port, parallel, context, production,
                    model, loaded_s, exit_status
             FROM runs WHERE ts < ?2 AND COALESCE(end_ts, ?2) >= ?1 AND (?3 = 1 OR production = 1)
             ORDER BY ts",
        )?
        .query_map(params![from, to, tests], |r| {
            Ok(RunRow {
                id: r.get(0)?,
                node: r.get(1)?,
                ts: r.get(2)?,
                end_ts: r.get(3)?,
                name: r.get(4)?,
                deploy: r.get(5)?,
                port: r.get(6)?,
                parallel: r.get(7)?,
                context: r.get(8)?,
                production: r.get::<_, i64>(9)? == 1,
                model: r.get(10)?,
                loaded_s: r.get(11)?,
                exit_status: r.get(12)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let requests = conn
        .prepare(&format!(
            "SELECT run_id, t_start, t_prefill, t_end, prompt, cached, generated, thinking, effort,
                    finish, rounds, accepted, drafted, slot, others, resume, client, path,
                    status, probe
             FROM requests WHERE t_end >= ?1 AND t_end < ?2 AND {PROD} ORDER BY t_end"
        ))?
        .query_map(params![from, to, tests], |r| {
            Ok(RequestRow {
                run: r.get(0)?,
                t_start: r.get(1)?,
                t_prefill: r.get(2)?,
                t_end: r.get(3)?,
                prompt: r.get(4)?,
                cached: r.get(5)?,
                generated: r.get(6)?,
                thinking: r.get(7)?,
                effort: r.get(8)?,
                finish: r.get(9)?,
                rounds: r.get(10)?,
                accepted: r.get(11)?,
                drafted: r.get(12)?,
                slot: r.get(13)?,
                others: r.get(14)?,
                resume: r.get(15)?,
                client: r.get(16)?,
                path: r.get(17)?,
                status: r.get(18)?,
                probe: r.get::<_, i64>(19)? == 1,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let events = conn
        .prepare(&format!(
            "SELECT ts, kind, a, b, c, d, e, text, run_id FROM events
             WHERE ts >= ?1 AND ts < ?2 AND {PROD} ORDER BY ts"
        ))?
        .query_map(params![from, to, tests], |r| {
            Ok(EventRow {
                run: r.get(8)?,
                ts: r.get(0)?,
                kind: r.get(1)?,
                v: [r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?],
                text: r.get(7)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let sar = conn
        .prepare(
            "SELECT node, ts, interval, mem_total, mem_avail, mem_used, mem_cached, swap_used,
                    cpu_user, cpu_system, cpu_iowait, cpu_idle, load1, load5, io_rkb, io_wkb
             FROM sar WHERE ts >= ?1 AND ts < ?2 ORDER BY ts",
        )?
        .query_map(params![from, to], |r| {
            Ok(SarRow {
                node: r.get(0)?,
                ts: r.get(1)?,
                interval: r.get(2)?,
                mem_total: r.get(3)?,
                mem_avail: r.get(4)?,
                mem_used: r.get(5)?,
                mem_cached: r.get(6)?,
                swap_used: r.get(7)?,
                cpu_user: r.get(8)?,
                cpu_system: r.get(9)?,
                cpu_iowait: r.get(10)?,
                cpu_idle: r.get(11)?,
                load1: r.get(12)?,
                load5: r.get(13)?,
                io_rkb: r.get(14)?,
                io_wkb: r.get(15)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let net = conn
        .prepare(
            "SELECT node, ts, iface, rx_kb, tx_kb FROM sar_net
             WHERE ts >= ?1 AND ts < ?2 ORDER BY ts",
        )?
        .query_map(params![from, to], |r| {
            Ok(NetRow {
                node: r.get(0)?,
                ts: r.get(1)?,
                iface: r.get(2)?,
                rx_kb: r.get(3)?,
                tx_kb: r.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(Dataset {
        from,
        to,
        runs,
        requests,
        events,
        sar,
        net,
    })
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    /// First and last log line held (µs), or None before the first pull.
    pub first_ts: Option<i64>,
    pub last_ts: Option<i64>,
    pub last_pull: Option<String>,
    pub last_pull_errors: Vec<String>,
}

pub fn status(conn: &Connection) -> Result<Status> {
    let (first_ts, last_ts): (Option<i64>, Option<i64>) = conn.query_row(
        "SELECT MIN(ts), MAX(ts) FROM raw WHERE source = 'rank0'",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let last_pull_errors = store::get_meta(conn, "last_pull_errors")?
        .map(|s| serde_json::from_str(&s).unwrap_or_default())
        .unwrap_or_default();
    Ok(Status {
        first_ts,
        last_ts,
        last_pull: store::get_meta(conn, "last_pull")?,
        last_pull_errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store;

    /// The "test runs" toggle reaches exactly this branch: by default only production runs and their rows are served.
    #[test]
    fn production_filter_hides_test_runs_and_their_rows() {
        let conn = store::memory();
        conn.execute_batch(
            "INSERT INTO runs(id, node, rank, ts, production) VALUES
                 (0, 'rank0', 0, 100, 1), (1, 'rank0', 0, 200, 0);
             INSERT INTO requests(t_end, prompt, cached, generated, run_id, probe) VALUES
                 (150, 10, 0, 2, 0, 0), (250, 10, 0, 2, 1, 0);
             INSERT INTO events(ts, kind, run_id) VALUES (150, 'pcache', 0), (250, 'pcache', 1);",
        )
        .unwrap();
        let prod = dataset(&conn, 0, 1000, false).unwrap();
        assert_eq!(
            (prod.runs.len(), prod.requests.len(), prod.events.len()),
            (1, 1, 1)
        );
        let all = dataset(&conn, 0, 1000, true).unwrap();
        assert_eq!(
            (all.runs.len(), all.requests.len(), all.events.len()),
            (2, 2, 2)
        );
    }

    #[test]
    fn range_is_half_open() {
        let conn = store::memory();
        conn.execute_batch(
            "INSERT INTO requests(t_end, prompt, cached, generated, probe) VALUES
                 (100, 1, 0, 1, 0), (200, 1, 0, 1, 0);
             INSERT INTO sar(node, ts, interval) VALUES ('rank0', 100, 600), ('rank0', 200, 600);",
        )
        .unwrap();
        let d = dataset(&conn, 100, 200, false).unwrap();
        assert_eq!(d.requests.len(), 1);
        assert_eq!(d.sar.len(), 1);
    }
}

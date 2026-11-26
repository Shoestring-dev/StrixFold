//! SQLite store: raw pulled lines, journal cursors, sar samples, and the derived tables rebuilt after every pull.

use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};

use crate::error::Result;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per journal entry, in journal order (id). `source` is
-- 'rank0', 'rank1' or 'watchdog'.
CREATE TABLE IF NOT EXISTS raw (
    id     INTEGER PRIMARY KEY,
    source TEXT    NOT NULL,
    ts     INTEGER NOT NULL,   -- unix microseconds
    cursor TEXT    NOT NULL,   -- journald __CURSOR: unique per entry
    msg    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS raw_source ON raw(source, id);
CREATE UNIQUE INDEX IF NOT EXISTS raw_cursor ON raw(source, cursor);

-- sar, one row per node and sample (10-minute averages).
CREATE TABLE IF NOT EXISTS sar (
    node      TEXT    NOT NULL,
    ts        INTEGER NOT NULL,  -- unix microseconds, end of the interval
    interval  INTEGER NOT NULL,  -- seconds
    mem_total INTEGER,           -- KiB (memused / memused-percent)
    mem_avail INTEGER,           -- KiB
    mem_used  INTEGER,
    mem_cached INTEGER,
    swap_used INTEGER,
    cpu_user  REAL,
    cpu_system REAL,
    cpu_iowait REAL,
    cpu_idle  REAL,
    load1     REAL,
    load5     REAL,
    io_rkb    REAL,               -- KiB/s read, all disks (from io bread)
    io_wkb    REAL,
    PRIMARY KEY (node, ts)
);
CREATE TABLE IF NOT EXISTS sar_net (
    node  TEXT    NOT NULL,
    ts    INTEGER NOT NULL,
    iface TEXT    NOT NULL,
    rx_kb REAL,                   -- KiB/s
    tx_kb REAL,
    PRIMARY KEY (node, ts, iface)
);

-- Derived tables: dropped and rebuilt from `raw` after each pull.
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    node        TEXT    NOT NULL,
    rank        INTEGER NOT NULL,
    ts          INTEGER NOT NULL,
    end_ts      INTEGER,
    name        TEXT,
    conf        TEXT,
    deploy      TEXT,
    port        INTEGER,
    parallel    INTEGER,
    context     INTEGER,
    production  INTEGER NOT NULL DEFAULT 0,
    model       TEXT,
    loaded_s    REAL,
    exit_status INTEGER
);
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER,
    req_id      TEXT,
    t_start     INTEGER,            -- prompt pass began (µs)
    t_prefill   INTEGER,            -- prompt pass ended
    t_end       INTEGER NOT NULL,   -- reply finished / client left
    prompt      INTEGER NOT NULL,
    cached      INTEGER NOT NULL,
    generated   INTEGER NOT NULL,
    thinking    INTEGER,
    effort      TEXT,
    finish      TEXT,               -- stop / length / client-left / ...
    rounds      INTEGER,
    accepted    INTEGER,
    drafted     INTEGER,
    slot        INTEGER,
    others      INTEGER,
    resume      TEXT,
    client      TEXT,
    path        TEXT,
    status      INTEGER,
    probe       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS requests_end ON requests(t_end);
CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY,
    run_id INTEGER,
    ts     INTEGER NOT NULL,
    kind   TEXT    NOT NULL,
    a      REAL,
    b      REAL,
    c      REAL,
    d      REAL,
    e      REAL,
    text   TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"#;

/// Bump with every schema change and add its step to `migrate`.
const SCHEMA_VERSION: i64 = 2;

pub fn open(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = OFF;",
    )?;
    migrate(&conn)?;
    Ok(conn)
}

/// Versioned via `PRAGMA user_version`; version 0 with tables present is a pre-release copy, dropped and rebuilt.
fn migrate(conn: &Connection) -> Result<()> {
    // One transaction, so an interrupted upgrade leaves the old database intact.
    let tx = conn.unchecked_transaction()?;
    migrate_in(&tx)?;
    tx.commit()?;
    Ok(())
}

fn migrate_in(conn: &Connection) -> Result<()> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version == 0 {
        let has_tables: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'raw'",
            [],
            |r| r.get(0),
        )?;
        // Any table at all means a pre-release copy.
        let has_tables: i64 = has_tables
            + conn.query_row::<i64, _, _>(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'
                 AND name IN ('meta', 'sar', 'sar_net', 'runs', 'requests', 'events', 'http')",
                [],
                |r| r.get(0),
            )?;
        if has_tables > 0 {
            conn.execute_batch(
                "DROP TABLE IF EXISTS raw; DROP TABLE IF EXISTS meta; DROP TABLE IF EXISTS sar;
                 DROP TABLE IF EXISTS sar_net; DROP TABLE IF EXISTS runs;
                 DROP TABLE IF EXISTS requests; DROP TABLE IF EXISTS events;
                 DROP TABLE IF EXISTS http;",
            )?;
        }
    }
    if version > SCHEMA_VERSION {
        return Err(crate::error::Error::Msg(format!(
            "this database is from a newer Strixscope (schema {version}); update the app"
        )));
    }
    // Future steps: `if version < N { ALTER TABLE ... }`, never a drop of data the user cannot pull again.
    if version < 2 {
        conn.execute_batch("DROP TABLE IF EXISTS http;")?;
    }
    conn.execute_batch(SCHEMA)?;
    conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))?;
    Ok(())
}

pub fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
        .optional()?)
}

pub fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

/// Newest raw timestamp for a source (µs), used when a cursor went stale.
pub fn last_ts(conn: &Connection, source: &str) -> Result<Option<i64>> {
    Ok(
        conn.query_row("SELECT MAX(ts) FROM raw WHERE source = ?1", [source], |r| {
            r.get(0)
        })?,
    )
}

#[cfg(test)]
pub fn memory() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    migrate(&conn).unwrap();
    conn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_copy_loses_the_http_table() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE http (run_id INTEGER, ts INTEGER NOT NULL, client TEXT,
                                method TEXT, path TEXT, status INTEGER);
             INSERT INTO http(ts) VALUES (1);
             CREATE TABLE raw (id INTEGER PRIMARY KEY, source TEXT NOT NULL, ts INTEGER NOT NULL,
                               cursor TEXT NOT NULL, msg TEXT NOT NULL);
             INSERT INTO raw(source, ts, cursor, msg) VALUES ('rank0', 7, 's=a;i=1', 'kept');
             CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO meta(key, value) VALUES ('cursor:rank0', 's=a;i=1');
             PRAGMA user_version = 1;",
        )
        .unwrap();
        migrate(&conn).unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'http'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0);
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        // The archive and its cursors survive the upgrade.
        assert_eq!(
            get_meta(&conn, "cursor:rank0").unwrap().as_deref(),
            Some("s=a;i=1")
        );
        assert_eq!(last_ts(&conn, "rank0").unwrap(), Some(7));
    }

    /// A failed migration must leave the database as it was: version, `raw` and the `http` table intact.
    #[test]
    fn failed_migration_rolls_back() {
        let conn = Connection::open_in_memory().unwrap();
        // A mismatched `raw` makes the schema's CREATE INDEX fail.
        conn.execute_batch(
            "CREATE TABLE raw (id TEXT);
             CREATE TABLE http (ts INTEGER);
             INSERT INTO http(ts) VALUES (1);
             PRAGMA user_version = 1;",
        )
        .unwrap();
        assert!(migrate(&conn).is_err());
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, 1);
        let http_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM http", [], |r| r.get(0))
            .unwrap();
        assert_eq!(http_rows, 1);
        // `raw` is untouched: still the mismatched one-column table.
        let raw_cols: i64 = conn
            .query_row("SELECT COUNT(*) FROM pragma_table_info('raw')", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(raw_cols, 1);
    }

    #[test]
    fn half_upgraded_copy_is_rebuilt() {
        let conn = Connection::open_in_memory().unwrap();
        // An interrupted upgrade dropped raw before recreating sar.
        conn.execute_batch(
            "CREATE TABLE sar (node TEXT, ts INTEGER, interval INTEGER, mem_avail INTEGER,
                               PRIMARY KEY (node, ts));",
        )
        .unwrap();
        migrate(&conn).unwrap();
        conn.execute(
            "INSERT INTO sar(node, ts, interval, mem_total) VALUES ('rank0', 1, 600, 5)",
            [],
        )
        .unwrap();
    }

    #[test]
    fn prerelease_copy_is_rebuilt() {
        let conn = Connection::open_in_memory().unwrap();
        // The first pre-release schema: raw without cursor, sar without mem_total.
        conn.execute_batch(
            "CREATE TABLE raw (id INTEGER PRIMARY KEY, source TEXT, ts INTEGER, msg TEXT);
             CREATE TABLE sar (node TEXT, ts INTEGER, interval INTEGER, mem_avail INTEGER,
                               PRIMARY KEY (node, ts));",
        )
        .unwrap();
        migrate(&conn).unwrap();
        conn.execute(
            "INSERT INTO raw(source, ts, cursor, msg) VALUES ('rank0', 1, 'c', 'm')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sar(node, ts, interval, mem_total) VALUES ('rank0', 1, 600, 5)",
            [],
        )
        .unwrap();
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        // Idempotent.
        migrate(&conn).unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM raw", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }
}

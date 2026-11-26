# Strixscope

A visual log analyzer for a StrixFold pair: requests and tokens,
reading/writing speed, caches, memory and restarts, per day and per hour.

It is opened only occasionally and never adds anything to the servers. On
open (and on the refresh button) it finds the pair and pulls its logs
**read-only over ssh**, keeping them in a local SQLite file; all parsing and
statistics run on the desktop.

## Finding the pair

Give it the ssh host of either node. Every StrixFold run leaves
`~/tfp/run/serve/<run>/resolved-rank<R>.json` (rank, hostname, start, conf,
argv) and `~/tfp/run/serve/latest`; from there the app reads the conf
(`RANK0_HOST`, `RANK1_HOST`, `MASTER_IP`, `TP_STRIPE`), probes the other
node, and finds:

- both nodes' ssh hosts and hostnames;
- the serve port, model, context and lanes (also from `GET /health` on the
  node's 127.0.0.1);
- the NICs carrying the links (the ones holding `MASTER_IP` / `TP_STRIPE`);
- sysstat's directory, if sysstat is installed (`/var/log/sysstat` or
  `/var/log/sa`);
- where the rank logs are: the journal, when a unit runs the ranks and their
  lines carry the `strixfold-container` tag, else the run directories.

Every value can be overridden in the app's pair settings. The production
port is the one whose runs were in place longest; others count as test runs.

| Source | Remote command (read-only) | Incremental by |
|---|---|---|
| discovery | `readlink`, `cat`, `ls`, `ip -br addr`, `journalctl -n 1`, `curl 127.0.0.1:PORT/health` | re-run on every pull |
| rank log lines (unit layout) | `journalctl --user -t strixfold-container -o json` on each node | journal cursor |
| rank log lines (run dirs) | `ls -ln`, `cat resolved-rank<R>.json`, `tail -c +N rank<R>.log` on each node | byte offset per run |
| watchdog notes, if that unit exists | `journalctl --user -u strixfold-watchdog -o json` | journal cursor |
| memory / CPU / network (10-minute samples) | `sadf -j <sysstat dir>/saDD` on both | re-pulled; upsert by time |

The journal stamps each line with µs. Lines in the run directories carry a
time only when the ranks run with `LOG_TIME=1` in StrixFold's serve.conf
(`TF_LOG_TIME=1`: an RFC 3339 UTC stamp before each line); without it a
pull still records each run (start, conf, deploy, model) and the system
charts, but no request history, and says so. sar keeps about 7 days on the
nodes; samples stay in the local copy once pulled. Nothing records GPU
memory, power or temperatures, so the app has none.

## Views

- **Overview** — tokens written, busy / prompt reuse / draft acceptance
  gauges, day × hour heatmap (click a cell to open the day), the pair
  diagram, KPI tiles with deltas vs the previous period.
- **Day** — every request in its slot lane (prefill then decode), tokens and
  speed per hour, memory per node.
- **Performance** — 3D writing-speed landscape and day × hour skyline
  (echarts-gl; 2D charts below if WebGL is missing), speed scatters, speed by
  deploy, wait for the first token, draft acceptance.
- **Caches** — RAM prompt cache, where prompts resumed from, disk prefix
  cache, image memo.
- **Requests** — sortable table with filters by client and finish.
- **System** — runs by deploy with crash-loop detection, memory, CPU, the
  links between the nodes, watchdog notes.

Every view shows production runs only. The header's **test runs** chip
also includes runs on other ports.

## Setup

The desktop needs key-based ssh to both nodes (key or agent;
`BatchMode=yes`, so no password prompts). On first open the app asks for one
node's ssh host (an alias or user@host) and shows what it found. Settings
live in `<app data>/settings.json` (`~/.local/share/dev.strixscope.app/` on
Linux):

```json
{ "host": "node0", "rank1Host": null, "prodPort": null, "linkIfaces": null, "runRoot": null }
```

## Build

From this directory, with Node 22.12 or newer, Rust 1.90 or newer and
[Tauri's system packages](https://v2.tauri.app/start/prerequisites/):

```sh
npm install
npm run tauri build
```

### Self-update (optional)

Off by default: a plain build contains no updater code. To ship an
auto-updating build, compile with the `self-update` Cargo feature and a
`--config` overlay that carries `plugins.updater` (`endpoints` + `pubkey`),
`bundle.createUpdaterArtifacts: true` and an inline `updater` capability
(`updater:default`, `process:allow-restart`) listed next to `"default"` in
`app.security.capabilities`:

```sh
npm run tauri -- build --features self-update --config updater.conf.json
```

## Development

```sh
npm install
npm run tauri:dev      # the desktop app
npm run dev            # browser only: serves dev-data/dataset.json
npm test               # frontend statistics tests (vitest)
cargo test --manifest-path src-tauri/Cargo.toml
```

`dev-data/dataset.json` is a private real-data snapshot (gitignored, served
by the Vite dev server only, never bundled). To
refresh it from journal exports (`r0.json.gz`, `r1.json.gz`, `wd.json.gz`):

```sh
STRIXSCOPE_SAR=<rank0 host>,<rank1 host> STRIXSCOPE_DATASET_OUT=$PWD/dev-data/dataset.json \
STRIXSCOPE_JOURNAL_DIR=<dir> cargo test --manifest-path src-tauri/Cargo.toml \
  --release real_journal -- --ignored --nocapture
```

The icon source is `design/icon.svg` (`npx tauri icon design/icon-1024.png`).

## License

Apache License 2.0; see [LICENSE](LICENSE).

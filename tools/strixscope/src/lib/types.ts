/** Rows returned by the Rust `get_dataset` command (see src-tauri/src/query.rs). */

export interface RequestRow {
  run: number | null;
  tStart: number | null; // µs: prompt pass began
  tPrefill: number | null; // µs: prompt pass ended
  tEnd: number; // µs: reply finished / client left
  prompt: number;
  cached: number;
  generated: number;
  thinking: boolean | null;
  effort: string | null;
  finish: string | null;
  rounds: number | null;
  accepted: number | null;
  drafted: number | null;
  slot: number | null;
  others: number | null;
  resume: string | null;
  client: string | null;
  path: string | null;
  status: number | null;
  probe: boolean;
}

export interface EventRow {
  run: number | null;
  ts: number;
  kind: string;
  v: [number | null, number | null, number | null, number | null, number | null];
  text: string | null;
}

export interface RunRow {
  id: number;
  node: "rank0" | "rank1";
  ts: number;
  endTs: number | null;
  name: string | null;
  deploy: string | null;
  port: number | null;
  parallel: number | null;
  context: number | null;
  production: boolean;
  model: string | null;
  loadedS: number | null;
  exitStatus: number | null;
}

export interface SarRow {
  node: "rank0" | "rank1";
  ts: number;
  interval: number;
  memTotal: number | null; // KiB
  memAvail: number | null; // KiB
  memUsed: number | null;
  memCached: number | null;
  swapUsed: number | null;
  cpuUser: number | null;
  cpuSystem: number | null;
  cpuIowait: number | null;
  cpuIdle: number | null;
  load1: number | null;
  load5: number | null;
  ioRkb: number | null;
  ioWkb: number | null;
}

export interface NetRow {
  node: "rank0" | "rank1";
  ts: number;
  iface: string;
  rxKb: number | null;
  txKb: number | null;
}

export interface Dataset {
  from: number;
  to: number;
  runs: RunRow[];
  requests: RequestRow[];
  events: EventRow[];
  sar: SarRow[];
  net: NetRow[];
}

export interface Status {
  firstTs: number | null;
  lastTs: number | null;
  lastPull: string | null;
  lastPullErrors: string[];
}

export interface SyncProgress {
  step: number;
  steps: number;
  label: string;
}

export interface SyncDone {
  newLines: number;
  sarSamples: number;
  errors: string[];
  rebuild: { lines: number; runs: number; requests: number; matched: number; events: number } | null;
}

/** Saved in settings.json; only `host` is required, other fields override discovery (null = use the discovered value). */
export interface Settings {
  host: string | null; // ssh destination of either node; null = not set up yet
  rank0Host: string | null;
  rank1Host: string | null;
  prodPort: number | null; // null = the port with the most run time
  linkIfaces: string[] | null; // rank 0's NICs carrying the links between the nodes
  runRoot: string | null; // StrixFold's RUN_ROOT, relative to $HOME or absolute
}

/** What discovery found, with the overrides applied (src-tauri/src/discover.rs). */
export interface Topology {
  rank0Host: string; // ssh destination
  rank1Host: string;
  rank0Name: string; // hostname, for labels
  rank1Name: string;
  servePort: number | null;
  linkIfaces: string[]; // may be empty
  rank0SaDir: string | null; // null = no sysstat on that node
  rank1SaDir: string | null;
  runRoot: string;
  journal: boolean; // true: both ranks' lines come from the journal
  rank0Journal: boolean; // per rank: journal, else run-dir log files
  rank1Journal: boolean;
  watchdog: boolean;
  model: string | null; // from /health, when it answered
  context: number | null;
  lanes: number | null;
  discoveredAt: string; // RFC 3339
}

/** Browser-only helpers that slice a dev snapshot like the Rust query does; pure, so unit-tested without a DOM. */
import { rank0Ifaces } from "./topology";
import type { Dataset, Settings, Topology } from "./types";

/** Rows whose run is a production run, or that have no run (watchdog). */
export function filterDataset(d: Dataset, from: number, to: number, includeTests: boolean): Dataset {
  const inR = (t: number) => t >= from && t < to;
  const prod = new Set(d.runs.filter((r) => r.production).map((r) => r.id));
  const keep = (run: number | null) => includeTests || run == null || prod.has(run);
  return {
    ...d,
    from,
    to,
    requests: d.requests.filter((r) => inR(r.tEnd) && keep(r.run)),
    events: d.events.filter((e) => inR(e.ts) && keep(e.run)),
    sar: d.sar.filter((s) => inR(s.ts)),
    net: d.net.filter((s) => inR(s.ts)),
    runs: d.runs.filter((r) => r.ts < to && (r.endTs ?? to) >= from && (includeTests || r.production)),
  };
}

/** A set-up pair, so the browser skips the setup screen. */
export const DEV_SETTINGS: Settings = {
  host: "node0",
  rank0Host: null,
  rank1Host: null,
  prodPort: null,
  linkIfaces: null,
  runRoot: null,
};

/** What discovery would find for the snapshot: generic names, rank 0's NICs. */
export function devTopology(d: Dataset): Topology {
  const prod = d.runs.filter((r) => r.node === "rank0" && r.production).sort((a, b) => b.ts - a.ts)[0];
  return {
    rank0Host: "node0",
    rank1Host: "node1",
    rank0Name: "node0",
    rank1Name: "node1",
    servePort: prod?.port ?? null,
    linkIfaces: rank0Ifaces(d.net),
    rank0SaDir: "/var/log/sysstat",
    rank1SaDir: "/var/log/sysstat",
    runRoot: "tfp/run/serve",
    journal: true,
    rank0Journal: true,
    rank1Journal: true,
    watchdog: true,
    model: prod?.model ?? null,
    context: prod?.context ?? null,
    lanes: prod?.parallel ?? null,
    discoveredAt: new Date().toISOString(),
  };
}

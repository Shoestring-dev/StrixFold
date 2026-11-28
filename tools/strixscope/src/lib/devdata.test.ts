/** The browser adapter must match the Rust query: half-open range and production-run rule. */
import { describe, expect, it } from "vitest";

import { devTopology, filterDataset } from "./devdata";
import type { Dataset, EventRow, RequestRow, RunRow } from "./types";

function run(id: number, ts: number, production: boolean, node: RunRow["node"] = "rank0"): RunRow {
  return {
    id,
    node,
    ts,
    endTs: null,
    name: null,
    deploy: null,
    port: production ? 8731 : 18731,
    parallel: 4,
    context: null,
    production,
    model: null,
    loadedS: null,
    exitStatus: null,
  };
}

function req(runId: number | null, tEnd: number): RequestRow {
  return {
    run: runId,
    tStart: null,
    tPrefill: null,
    tEnd,
    prompt: 10,
    cached: 0,
    generated: 2,
    thinking: null,
    effort: null,
    finish: "stop",
    rounds: null,
    accepted: null,
    drafted: null,
    slot: null,
    others: null,
    resume: null,
    client: null,
    path: null,
    status: null,
    probe: false,
  };
}

function ev(runId: number | null, ts: number): EventRow {
  return { run: runId, ts, kind: "pcache", v: [null, null, null, null, null], text: null };
}

const snapshot: Dataset = {
  from: 0,
  to: 0,
  runs: [run(1, 0, true), run(2, 0, false), run(3, 0, true, "rank1")],
  requests: [req(1, 10), req(2, 20), req(null, 30)],
  events: [ev(1, 10), ev(2, 20)],
  sar: [{ node: "rank0", ts: 10, interval: 600, memTotal: null, memAvail: 1, memUsed: null, memCached: null, swapUsed: null, cpuUser: null, cpuSystem: null, cpuIowait: null, cpuIdle: null, load1: null, load5: null, ioRkb: null, ioWkb: null }],
  net: [{ node: "rank0", ts: 10, iface: "eth1", rxKb: 1, txKb: 2 }],
};

describe("dev snapshot filter", () => {
  it("keeps production runs and rows without a run by default", () => {
    const d = filterDataset(snapshot, 0, 100, false);
    expect(d.runs.map((r) => r.id)).toEqual([1, 3]);
    expect(d.requests.map((r) => r.tEnd)).toEqual([10, 30]);
    expect(d.events.map((e) => e.ts)).toEqual([10]);
  });

  it("keeps test runs and their rows when asked", () => {
    const d = filterDataset(snapshot, 0, 100, true);
    expect(d.runs.map((r) => r.id)).toEqual([1, 2, 3]);
    expect(d.requests.map((r) => r.tEnd)).toEqual([10, 20, 30]);
    expect(d.events.map((e) => e.ts)).toEqual([10, 20]);
  });

  it("applies the range as half-open to every row set", () => {
    const d = filterDataset(snapshot, 10, 20, true);
    expect(d.requests.map((r) => r.tEnd)).toEqual([10]);
    expect(d.sar.map((s) => s.ts)).toEqual([10]);
    expect(d.net.map((s) => s.ts)).toEqual([10]);
  });
});

describe("dev topology", () => {
  it("names the nodes generically and takes rank 0's NICs from sar", () => {
    const t = devTopology({
      ...snapshot,
      net: [
        { node: "rank0", ts: 10, iface: "eth0", rxKb: 1, txKb: 1 },
        { node: "rank0", ts: 10, iface: "link0", rxKb: 50, txKb: 50 },
        { node: "rank1", ts: 10, iface: "link9", rxKb: 99, txKb: 99 },
      ],
    });
    expect([t.rank0Name, t.rank1Name, t.rank0Host]).toEqual(["node0", "node1", "node0"]);
    expect(t.linkIfaces).toEqual(["link0", "eth0"]);
    expect(t.journal).toBe(true);
    expect(t.servePort).toBe(8731);
  });
});

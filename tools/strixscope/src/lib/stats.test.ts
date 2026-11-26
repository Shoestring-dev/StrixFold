/// <reference types="node" />
process.env.TZ = "America/Chicago"; // a fixed zone with DST, so the day-boundary cases are deterministic

import { describe, expect, it } from "vitest";

import {
  bucketize,
  busyShare,
  busyUs,
  byLocalHour,
  compact,
  counterGrowth,
  crashLoops,
  dayEdges,
  dayStart,
  enrich,
  gib,
  hourEdges,
  kpis,
  nextDay,
  overlapping,
  pct,
  prepare,
  promptCacheHitRate,
  quantile,
  sortedOf,
  type Bucket,
  type Req,
} from "./stats";
import type { EventRow, RequestRow, RunRow } from "./types";

const HOUR = 3600e6;

/** A raw row with no timings; every test overrides what it needs. */
function row(over: Partial<RequestRow> = {}): RequestRow {
  return {
    run: 1,
    tStart: null,
    tPrefill: null,
    tEnd: 0,
    prompt: 0,
    cached: 0,
    generated: 0,
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
    ...over,
  };
}

/** The enriched form the series functions take. */
function r(over: Partial<RequestRow> = {}): Req {
  return enrich(row(over));
}

function bucket(over: Partial<Bucket> = {}): Bucket {
  return {
    start: 0,
    end: 0,
    requests: 0,
    aborted: 0,
    tokensIn: 0,
    tokensFresh: 0,
    tokensOut: 0,
    cachedShare: null,
    prefillMed: null,
    prefillP10: null,
    prefillP90: null,
    decodeMed: null,
    decodeP10: null,
    decodeP90: null,
    acceptance: null,
    accepted: 0,
    drafted: 0,
    busy: 0,
    maxConcurrency: 0,
    ...over,
  };
}

function run(over: Partial<RunRow> = {}): RunRow {
  return {
    id: 1,
    node: "rank0",
    ts: 0,
    endTs: 100,
    name: null,
    deploy: null,
    port: 8731,
    parallel: 4,
    context: null,
    production: true,
    model: null,
    loadedS: null,
    exitStatus: 1,
    ...over,
  };
}

function event(kind: string, run: number | null, ts: number, v: number[]): EventRow {
  const xs: EventRow["v"] = [null, null, null, null, null];
  v.forEach((x, i) => (xs[i] = x));
  return { run, ts, kind, v: xs, text: null };
}

describe("quantile", () => {
  it("interpolates and handles the ends", () => {
    expect(quantile([], 0.5)).toBeNull();
    expect(quantile([1, 2, 3, 4], 0)).toBe(1);
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(quantile([1, 2, 3, 4], 1)).toBe(4);
    expect(quantile([7], 0.9)).toBe(7);
  });

  it("drops nulls and non-finite values before sorting", () => {
    expect(sortedOf([3, null, 1, Number.NaN, 2])).toEqual([1, 2, 3]);
  });
});

describe("busyUs", () => {
  it("counts the union of overlapping intervals, clipped to the range", () => {
    expect(busyUs([[0, 10], [5, 15]], 0, 20)).toBe(15);
    expect(busyUs([[-5, 5], [15, 25]], 0, 20)).toBe(10);
    expect(busyUs([[0, 2], [5, 7]], 0, 20)).toBe(4);
    expect(busyUs([], 0, 20)).toBe(0);
    // Touching intervals must not be double counted.
    expect(busyUs([[0, 5], [5, 10]], 0, 10)).toBe(10);
  });
});

describe("bucketize", () => {
  it("is half-open: a request finishing on an edge belongs to the next bucket", () => {
    const reqs = [r({ tEnd: 10 }), r({ tEnd: 20 }), r({ tEnd: 30 })];
    const bs = bucketize(reqs, [0, 10, 20, 30]);
    expect(bs.map((b) => b.requests)).toEqual([0, 1, 1]);
    expect(bs[0].cachedShare).toBeNull();
    expect(bs[0].acceptance).toBeNull();
  });

  it("counts busy time from every request overlapping the span", () => {
    const busy = [r({ tStart: 0, tEnd: 10 }), r({ tStart: 10, tEnd: 20 })];
    const finishing = [r({ tEnd: 10 })];
    const bs = bucketize(finishing, [0, 10, 20], busy);
    expect(bs[0].requests).toBe(0);
    expect(bs[0].busy).toBe(1);
    expect(bs[1].busy).toBe(1);
  });

  it("sums tokens, cache share and draft counts with their own denominators", () => {
    const bs = bucketize(
      [
        r({ tEnd: 5, prompt: 100, cached: 40, generated: 10, accepted: 1, drafted: 1, others: 2 }),
        r({ tEnd: 6, prompt: 300, cached: 30, generated: 20, accepted: 0, drafted: 99, others: 0 }),
      ],
      [0, 10],
    )[0];
    expect(bs.tokensIn).toBe(400);
    expect(bs.tokensOut).toBe(30);
    expect(bs.cachedShare).toBeCloseTo(70 / 400);
    expect(bs.accepted).toBe(1);
    expect(bs.drafted).toBe(100);
    expect(bs.acceptance).toBeCloseTo(0.01);
    // Others is the number of *other* live streams; this request saw 2.
    expect(bs.maxConcurrency).toBe(3);
  });

  it("weights busy share by the real duration of each bucket", () => {
    const bs = [bucket({ start: 0, end: 10, busy: 1 }), bucket({ start: 10, end: 30, busy: 0 })];
    expect(busyShare(bs)).toBeCloseTo(10 / 30);
    expect(busyShare([])).toBe(0);
  });
});

describe("byLocalHour", () => {
  // 2026-11-01 is the US fall-back day: 01:00 happens twice and both map to local hour 1, so they merge.
  const first = Date.parse("2026-11-01T06:00:00Z"); // 01:00 CDT
  const second = Date.parse("2026-11-01T07:00:00Z"); // 01:00 CST

  it("merges a repeated fall-back hour with the right denominators", () => {
    const a = bucket({
      start: first * 1000,
      end: first * 1000 + HOUR,
      requests: 1,
      tokensIn: 100,
      cachedShare: 0.5,
      accepted: 1,
      drafted: 1,
      busy: 1,
    });
    const b = bucket({
      start: second * 1000,
      end: second * 1000 + HOUR,
      requests: 5,
      tokensIn: 300,
      cachedShare: 0.1,
      accepted: 0,
      drafted: 99,
      busy: 0,
    });
    const slots = byLocalHour([a, b]);
    expect(slots).toHaveLength(24);
    const merged = slots[1]!;
    expect(merged.requests).toBe(6);
    expect(merged.start).toBe(a.start);
    expect(merged.end).toBe(b.end);
    // (0.5*100 + 0.1*300) / 400 — weighted by prompt tokens.
    expect(merged.cachedShare).toBeCloseTo(0.2);
    // 1 of 100 drafts, not the average of 100% and 0%.
    expect(merged.acceptance).toBeCloseTo(0.01);
    expect(merged.busy).toBeCloseTo(0.5);
    // Percentiles come from the fuller bucket; they cannot be merged exactly.
    expect(merged.prefillMed).toBeNull();
  });

  it("leaves the missing spring-forward hour empty", () => {
    const day = dayStart(Date.parse("2026-03-08T12:00:00Z") * 1000); // local 07:00
    const hours = bucketize([], hourEdges(day));
    const slots = byLocalHour(hours);
    expect(slots).toHaveLength(24);
    // 02:00 never happens that day; 03:00 does.
    expect(slots[2]).toBeNull();
    expect(slots[3]).not.toBeNull();
  });

  it("keeps an hour with no requests at all empty rather than zero-filled", () => {
    const slots = byLocalHour([bucket({ start: 0, end: HOUR })]);
    expect(slots.filter((s) => s != null)).toHaveLength(1);
  });
});

describe("day and hour edges", () => {
  it("covers a fall-back day with 25 hour buckets and a spring-forward day with 23", () => {
    const fall = dayStart(Date.parse("2026-11-01T12:00:00Z") * 1000);
    expect(nextDay(fall) - fall).toBe(25 * HOUR);
    expect(hourEdges(fall)).toHaveLength(26);

    const spring = dayStart(Date.parse("2026-03-08T12:00:00Z") * 1000);
    expect(nextDay(spring) - spring).toBe(23 * HOUR);
    expect(hourEdges(spring)).toHaveLength(24);
  });

  it("builds day edges for a range and a single partial day", () => {
    const day = dayStart(Date.parse("2026-11-02T12:00:00Z") * 1000);
    const edges = dayEdges(day + HOUR, day + 3 * HOUR);
    expect(edges).toEqual([day, nextDay(day)]);
    expect(dayEdges(day, nextDay(day) + 1)).toHaveLength(3);
  });
});

describe("counterGrowth", () => {
  it("counts a run that began in the range from zero", () => {
    const events = [event("pcache", 1, 10, [0, 0, 10, 5]), event("pcache", 1, 20, [0, 0, 25, 9])];
    expect(counterGrowth(events, "pcache", 2, new Map([[1, 0]]), 0)).toBe(25);
  });

  it("counts a run that began earlier from its first sample in the range", () => {
    const events = [event("pcache", 2, 10, [0, 0, 100, 5]), event("pcache", 2, 20, [0, 0, 150, 9])];
    expect(counterGrowth(events, "pcache", 2, new Map([[2, -1000]]), 0)).toBe(50);
  });

  it("never adds a negative step when a counter resets", () => {
    const events = [event("pcache", 1, 10, [0, 0, 10, 5]), event("pcache", 1, 20, [0, 0, 4, 6])];
    expect(counterGrowth(events, "pcache", 2, new Map([[1, 0]]), 0)).toBe(10);
  });

  it("ignores events without a run and other kinds", () => {
    const events = [event("pcache", null, 10, [0, 0, 10]), event("stale", 1, 20, [0, 0, 99])];
    expect(counterGrowth(events, "pcache", 2, new Map(), 0)).toBe(0);
  });

  it("divides hits by hits plus misses", () => {
    const events = [event("pcache", 1, 10, [0, 0, 3, 1])];
    expect(promptCacheHitRate(events, new Map([[1, 0]]), 0)).toBeCloseTo(0.75);
    expect(promptCacheHitRate([], new Map(), 0)).toBeNull();
  });
});

describe("crash loops", () => {
  it("flags three consecutive short failed runs", () => {
    const runs = [run({ ts: 0, endTs: 60e6 }), run({ ts: 100e6, endTs: 160e6 }), run({ ts: 200e6, endTs: 260e6 })];
    const loops = crashLoops(runs);
    expect(loops).toHaveLength(1);
    expect(loops[0]).toMatchObject({ from: 0, to: 260e6, runs: 3 });
  });

  it("does not flag two, a long run, or a clean SIGTERM restart", () => {
    expect(crashLoops([run(), run({ ts: 100e6 })])).toHaveLength(0);
    expect(crashLoops([run({ endTs: 30 * 60e6 })])).toHaveLength(0);
    const runs = [run(), run({ ts: 100e6 }), run({ ts: 200e6, exitStatus: 143 })];
    expect(crashLoops(runs)).toHaveLength(0);
  });
});

describe("enrich", () => {
  it("only measures speeds where there is enough work to time", () => {
    const x = enrich(row({ prompt: 100, cached: 40, tStart: 0, tPrefill: 1e6, tEnd: 3e6, generated: 50 }));
    expect(x.fresh).toBe(60);
    expect(x.prefillS).toBe(1);
    expect(x.prefillTps).toBeNull(); // 60 new tokens is below the 256 floor
    expect(x.decodeTps).toBe(25);
    expect(x.totalS).toBe(3);
    expect(x.acceptance).toBeNull();
    expect(x.aborted).toBe(false);
  });

  it("keeps a negative cache count out of fresh tokens", () => {
    expect(enrich(row({ prompt: 10, cached: 40 })).fresh).toBe(0);
  });

  it("computes acceptance, aborts and missing timings", () => {
    const x = enrich(row({ accepted: 5, drafted: 10, finish: "client-left" }));
    expect(x.acceptance).toBe(0.5);
    expect(x.aborted).toBe(true);
    expect(x.prefillS).toBeNull();
    expect(x.totalS).toBeNull();
  });
});

describe("prepare and overlapping", () => {
  it("keeps probes out of the main series", () => {
    const p = prepare({ requests: [row(), row({ probe: true })] } as never);
    expect(p.reqs).toHaveLength(1);
    expect(p.probes).toHaveLength(1);
  });

  it("treats the range as half-open", () => {
    const x = r({ tStart: 0, tEnd: 10 });
    expect(overlapping([x], 5, 20)).toHaveLength(1);
    expect(overlapping([x], 11, 20)).toHaveLength(0);
    // A request starting exactly at `to` has not begun inside the range.
    expect(overlapping([r({ tStart: 10, tEnd: 20 })], 0, 10)).toHaveLength(0);
  });
});

describe("kpis", () => {
  it("counts rank-0 launches in the range and distinct clients", () => {
    const reqs = [r({ tEnd: 5, client: "a" }), r({ tEnd: 6, client: "a" }), r({ tEnd: 7, client: "b" })];
    const runs = [run({ ts: 1 }), run({ ts: 2, node: "rank1" })];
    const k = kpis(reqs, reqs, runs, 0, 10);
    expect(k.requests).toBe(3);
    expect(k.restarts).toBe(1);
    expect(k.clients).toBe(2);
  });
});

describe("formatting", () => {
  it("compacts large numbers and never prints NaN", () => {
    expect(compact(12345)).toBe("12.3K");
    expect(compact(1.2e6)).toBe("1.2M");
    expect(compact(1.5e9)).toBe("1.5B");
    expect(compact(null)).toBe("—");
    expect(compact(Number.POSITIVE_INFINITY)).toBe("—");
    expect(pct(0.1234, 1)).toBe("12.3%");
    expect(pct(null)).toBe("—");
    expect(gib(1048576)).toBe(1);
    expect(gib(null)).toBeNull();
  });
});

/** Statistics over a dataset slice; all binning is in local time, Rust stores unix microseconds. */
import type { Dataset, EventRow, RequestRow, RunRow, SarRow } from "./types";

export const US = 1_000_000;
export const HOUR_US = 3600 * US;
export const DAY_US = 24 * HOUR_US;

/** Prompts with fewer new tokens than this time the overhead, not reading. */
const MIN_PREFILL_TOKENS = 256;
/** Replies shorter than this time the first round, not decoding. */
const MIN_DECODE_TOKENS = 32;

export interface Req extends RequestRow {
  /** New prompt tokens computed (prompt − cached). */
  fresh: number;
  /** Prompt tokens per second over the prompt pass, when measurable. */
  prefillTps: number | null;
  /** Generated tokens per second after the prompt pass, when measurable. */
  decodeTps: number | null;
  /** Seconds in the prompt pass (≈ time to first token, without queueing). */
  prefillS: number | null;
  /** Seconds from pass start to finish. */
  totalS: number | null;
  /** Draft acceptance 0..1. */
  acceptance: number | null;
  aborted: boolean;
}

export function enrich(r: RequestRow): Req {
  const fresh = Math.max(0, r.prompt - r.cached);
  const prefillS = r.tStart != null && r.tPrefill != null && r.tPrefill > r.tStart ? (r.tPrefill - r.tStart) / US : null;
  const decodeS = r.tPrefill != null && r.tEnd > r.tPrefill ? (r.tEnd - r.tPrefill) / US : null;
  return {
    ...r,
    fresh,
    prefillS,
    prefillTps: prefillS && fresh >= MIN_PREFILL_TOKENS ? fresh / prefillS : null,
    decodeTps: decodeS && r.generated >= MIN_DECODE_TOKENS ? r.generated / decodeS : null,
    totalS: r.tStart != null ? (r.tEnd - r.tStart) / US : null,
    acceptance: r.drafted ? (r.accepted ?? 0) / r.drafted : null,
    aborted: r.finish === "client-left",
  };
}

export function quantile(sorted: number[], q: number): number | null {
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function sortedOf(xs: (number | null)[]): number[] {
  return xs.filter((x): x is number => x != null && Number.isFinite(x)).sort((a, b) => a - b);
}

export const median = (xs: (number | null)[]) => quantile(sortedOf(xs), 0.5);
export const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

/** Local midnight (µs) of the day holding ts. */
export function dayStart(tsUs: number): number {
  const d = new Date(tsUs / 1000);
  d.setHours(0, 0, 0, 0);
  return d.getTime() * 1000;
}
export function hourStart(tsUs: number): number {
  const d = new Date(tsUs / 1000);
  d.setMinutes(0, 0, 0);
  return d.getTime() * 1000;
}
/** Next local midnight (handles DST: not always +24h). */
export function nextDay(dayUs: number): number {
  const d = new Date(dayUs / 1000);
  d.setDate(d.getDate() + 1);
  d.setHours(0, 0, 0, 0);
  return d.getTime() * 1000;
}

export function dayLabel(dayUs: number, opts: Intl.DateTimeFormatOptions = { weekday: "short", month: "short", day: "numeric" }) {
  return new Date(dayUs / 1000).toLocaleDateString(undefined, opts);
}
export function timeLabel(tsUs: number) {
  return new Date(tsUs / 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/** Busy microseconds of [a,b) covered by the union of request intervals. */
export function busyUs(intervals: [number, number][], a: number, b: number): number {
  let total = 0;
  let curS = -1;
  let curE = -1;
  for (const [s0, e0] of intervals) {
    const s = Math.max(s0, a);
    const e = Math.min(e0, b);
    if (e <= s) continue;
    if (s > curE) {
      if (curE > curS) total += curE - curS;
      curS = s;
      curE = e;
    } else if (e > curE) curE = e;
  }
  if (curE > curS) total += curE - curS;
  return total;
}

export function intervalsOf(reqs: Req[]): [number, number][] {
  return reqs
    .filter((r) => r.tStart != null)
    .map((r) => [r.tStart as number, r.tEnd] as [number, number])
    .sort((x, y) => x[0] - y[0]);
}

export interface Bucket {
  start: number;
  end: number;
  requests: number;
  aborted: number;
  tokensIn: number; // all prompt tokens
  tokensFresh: number; // computed (not cached)
  tokensOut: number;
  cachedShare: number | null; // cached / prompt
  prefillMed: number | null;
  prefillP10: number | null;
  prefillP90: number | null;
  decodeMed: number | null;
  decodeP10: number | null;
  decodeP90: number | null;
  acceptance: number | null;
  /** Kept / offered MTP drafts, so a merge can weight acceptance exactly. */
  accepted: number;
  drafted: number;
  busy: number; // 0..1
  maxConcurrency: number;
}

/** Buckets over half-open edges: counts and tokens go to the finishing bucket, busy time comes from `busySource`. */
export function bucketize(reqs: Req[], edges: number[], busySource: Req[] = reqs): Bucket[] {
  const iv = intervalsOf(busySource);
  const out: Bucket[] = [];
  let j = 0;
  const sorted = [...reqs].sort((a, b) => a.tEnd - b.tEnd);
  for (let i = 0; i + 1 < edges.length; i++) {
    const a = edges[i];
    const b = edges[i + 1];
    const inB: Req[] = [];
    while (j < sorted.length && sorted[j].tEnd < a) j++;
    let k = j;
    while (k < sorted.length && sorted[k].tEnd < b) inB.push(sorted[k++]);
    const pf = sortedOf(inB.map((r) => r.prefillTps));
    const dc = sortedOf(inB.map((r) => r.decodeTps));
    const prompt = sum(inB.map((r) => r.prompt));
    const acc = sum(inB.map((r) => r.accepted ?? 0));
    const dr = sum(inB.map((r) => r.drafted ?? 0));
    out.push({
      start: a,
      end: b,
      requests: inB.length,
      aborted: inB.filter((r) => r.aborted).length,
      tokensIn: prompt,
      tokensFresh: sum(inB.map((r) => r.fresh)),
      tokensOut: sum(inB.map((r) => r.generated)),
      cachedShare: prompt ? sum(inB.map((r) => r.cached)) / prompt : null,
      prefillMed: quantile(pf, 0.5),
      prefillP10: quantile(pf, 0.1),
      prefillP90: quantile(pf, 0.9),
      decodeMed: quantile(dc, 0.5),
      decodeP10: quantile(dc, 0.1),
      decodeP90: quantile(dc, 0.9),
      acceptance: dr ? acc / dr : null,
      accepted: acc,
      drafted: dr,
      busy: b > a ? busyUs(iv, a, b) / (b - a) : 0,
      maxConcurrency: inB.reduce((m, r) => Math.max(m, (r.others ?? 0) + 1), 0),
    });
  }
  return out;
}

export function dayEdges(from: number, to: number): number[] {
  const edges: number[] = [];
  for (let d = dayStart(from); d < to; d = nextDay(d)) edges.push(d);
  edges.push(nextDay(edges[edges.length - 1]));
  return edges;
}

export function hourEdges(dayUs: number): number[] {
  const edges: number[] = [];
  const end = nextDay(dayUs);
  for (let h = dayUs; h <= end; h += HOUR_US) edges.push(h);
  if (edges[edges.length - 1] !== end) edges.push(end);
  return edges;
}

export interface Kpis {
  requests: number;
  aborted: number;
  tokensIn: number;
  tokensFresh: number;
  tokensOut: number;
  prefillMed: number | null;
  decodeMed: number | null;
  acceptance: number | null;
  cachedShare: number | null;
  busy: number;
  restarts: number;
  clients: number;
}

/** KPIs over [from, to); `busyReqs` are the requests overlapping it. */
export function kpis(reqs: Req[], busyReqs: Req[], runs: RunRow[], from: number, to: number): Kpis {
  const b = bucketize(reqs, [from, to], busyReqs)[0];
  return {
    requests: b.requests,
    aborted: b.aborted,
    tokensIn: b.tokensIn,
    tokensFresh: b.tokensFresh,
    tokensOut: b.tokensOut,
    prefillMed: b.prefillMed,
    decodeMed: b.decodeMed,
    acceptance: b.acceptance,
    cachedShare: b.cachedShare,
    busy: b.busy,
    // Rank 0 run starts: each start is a (re)launch of the pair.
    restarts: runs.filter((r) => r.node === "rank0" && r.ts >= from && r.ts < to).length,
    clients: new Set(reqs.map((r) => r.client).filter(Boolean)).size,
  };
}

/** Growth of a cumulative per-run counter (event `idx`) in [from, to), counting from zero for runs begun in the range. */
export function counterGrowth(events: EventRow[], kind: string, idx: number, runStart: Map<number, number>, from: number): number {
  const byRun = new Map<number, number[]>();
  for (const e of events) {
    if (e.kind !== kind || e.run == null) continue;
    const xs = byRun.get(e.run) ?? [];
    xs.push(e.v[idx] ?? 0);
    byRun.set(e.run, xs);
  }
  let total = 0;
  for (const [run, xs] of byRun) {
    const startedInRange = (runStart.get(run) ?? -Infinity) >= from;
    let prev = startedInRange ? 0 : xs[0];
    for (const x of xs) {
      if (x >= prev) total += x - prev;
      prev = x;
    }
  }
  return total;
}

/** Prompt-cache hit ratio over the range, counted per run. */
export function promptCacheHitRate(events: EventRow[], runStart: Map<number, number>, from: number): number | null {
  const hits = counterGrowth(events, "pcache", 2, runStart, from);
  const misses = counterGrowth(events, "pcache", 3, runStart, from);
  return hits + misses ? hits / (hits + misses) : null;
}

/** 24 local-hour slots: fall-back merges two buckets, spring-forward leaves null; percentiles come from the fuller bucket. */
export function byLocalHour(buckets: Bucket[]): (Bucket | null)[] {
  const slots: (Bucket | null)[] = Array.from({ length: 24 }, () => null);
  for (const b of buckets) {
    const h = new Date(b.start / 1000).getHours();
    const prev = slots[h];
    if (!prev) {
      slots[h] = b;
      continue;
    }
    const d1 = prev.end - prev.start;
    const d2 = b.end - b.start;
    const main = b.requests > prev.requests ? b : prev;
    const prompt = prev.tokensIn + b.tokensIn;
    const drafted = prev.drafted + b.drafted;
    const accepted = prev.accepted + b.accepted;
    slots[h] = {
      ...main,
      start: prev.start,
      end: b.end,
      requests: prev.requests + b.requests,
      aborted: prev.aborted + b.aborted,
      tokensIn: prompt,
      tokensFresh: prev.tokensFresh + b.tokensFresh,
      tokensOut: prev.tokensOut + b.tokensOut,
      cachedShare:
        prompt > 0
          ? ((prev.cachedShare ?? 0) * prev.tokensIn + (b.cachedShare ?? 0) * b.tokensIn) / prompt
          : null,
      accepted,
      drafted,
      acceptance: drafted > 0 ? accepted / drafted : null,
      busy: (prev.busy * d1 + b.busy * d2) / (d1 + d2),
      maxConcurrency: Math.max(prev.maxConcurrency, b.maxConcurrency),
    };
  }
  return slots;
}

/** Busy share over a set of buckets, weighted by their real durations. */
export function busyShare(buckets: Bucket[]): number {
  const dur = sum(buckets.map((b) => b.end - b.start));
  return dur ? sum(buckets.map((b) => b.busy * (b.end - b.start))) / dur : 0;
}

/** Requests overlapping [from, to): for busy time. */
export function overlapping(reqs: Req[], from: number, to: number): Req[] {
  return reqs.filter((r) => r.tEnd >= from && (r.tStart ?? r.tEnd) < to);
}

/** Runs shorter than this, three or more in a row, are a crash loop. */
const SHORT_RUN_US = 10 * 60e6;
/** SIGTERM: the watchdog's deliberate kill, not a crash. */
const SIGTERM = 143;

export interface CrashLoop {
  from: number;
  to: number;
  runs: number;
}

/** Runs shorter than ten minutes that ended badly, three or more in a row; `runs` must be in start order. */
export function crashLoops(runs: RunRow[]): CrashLoop[] {
  const loops: CrashLoop[] = [];
  let cur: RunRow[] = [];
  const flush = () => {
    if (cur.length >= 3) {
      loops.push({ from: cur[0].ts, to: cur[cur.length - 1].endTs ?? cur[cur.length - 1].ts, runs: cur.length });
    }
    cur = [];
  };
  for (const r of runs) {
    const short =
      r.endTs != null && r.endTs - r.ts < SHORT_RUN_US && r.exitStatus !== 0 && r.exitStatus !== SIGTERM;
    if (short) cur.push(r);
    else flush();
  }
  flush();
  return loops;
}

export function sarFor(rows: SarRow[], node: "rank0" | "rank1") {
  return rows.filter((s) => s.node === node);
}

export function prepare(d: Dataset, includeProbes = false) {
  const all = d.requests.map(enrich);
  return { reqs: includeProbes ? all : all.filter((r) => !r.probe), probes: all.filter((r) => r.probe) };
}

/** 1,284 → "1,284"; 12_900 → "12.9K"; 4.2e6 → "4.2M". */
export function compact(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(digits) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(digits) + "M";
  if (a >= 1e4) return (n / 1e3).toFixed(digits) + "K";
  return Math.round(n).toLocaleString();
}
export const pct = (x: number | null | undefined, digits = 0) =>
  x == null || !Number.isFinite(x) ? "—" : (x * 100).toFixed(digits) + "%";
export const gib = (kib: number | null | undefined) => (kib == null ? null : kib / 1024 / 1024);

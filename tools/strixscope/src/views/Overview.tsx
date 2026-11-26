import { useMemo, useState } from "react";

import { Card } from "../components/Card";
import { CountUp } from "../components/CountUp";
import { EChart } from "../components/EChart";
import { Gauge } from "../components/Gauge";
import { PairDiagram } from "../components/PairDiagram";
import { StatTile } from "../components/StatTile";
import {
  bucketize,
  byLocalHour,
  overlapping,
  compact,
  dayEdges,
  dayLabel,
  DAY_US,
  dayStart,
  gib,
  hourEdges,
  kpis,
  nextDay,
  type Req,
} from "../lib/stats";
import { alpha, type Palette } from "../lib/theme";
import { linkIfaces, linkRate, nodeName } from "../lib/topology";
import type { EventRow, NetRow, RunRow, SarRow, Topology } from "../lib/types";

export interface Slice {
  from: number;
  to: number;
  /** Requests finishing in [from, to): counts and tokens. */
  reqs: Req[];
  /** Requests overlapping [from, to): busy time. */
  busyReqs: Req[];
  /** Every request held (the heatmap shows at least 7 days). */
  allReqs: Req[];
  events: EventRow[];
  runs: RunRow[];
  /** Every run held, for looking up the run a request belongs to. */
  allRuns: RunRow[];
  sar: SarRow[];
  net: NetRow[];
  /** The same-length period just before, for deltas (null for "all"). */
  prev: { reqs: Req[]; busyReqs: Req[]; runs: RunRow[]; from: number; to: number } | null;
}

type HeatMetric = "busy" | "requests" | "tokensOut";

const HEAT_LABEL: Record<HeatMetric, string> = {
  busy: "Busy time",
  requests: "Requests",
  tokensOut: "Tokens written",
};

export function Overview({
  s,
  topo,
  pal,
  onOpenDay,
}: {
  s: Slice;
  topo: Topology | null;
  pal: Palette;
  onOpenDay: (dayUs: number) => void;
}) {
  const [heat, setHeat] = useState<HeatMetric>("busy");

  const k = useMemo(() => kpis(s.reqs, s.busyReqs, s.runs, s.from, s.to), [s]);
  const kp = useMemo(() => (s.prev ? kpis(s.prev.reqs, s.prev.busyReqs, s.prev.runs, s.prev.from, s.prev.to) : null), [s]);
  const delta = (a: number | null, b: number | null | undefined) => (a != null && b ? a / b - 1 : null);

  // Daily buckets (hourly when the range is a single day) for sparklines.
  const short = s.to - s.from <= 1.01 * DAY_US;
  const buckets = useMemo(() => {
    if (short) {
      const edges: number[] = [];
      for (let t = s.from; t < s.to; t += DAY_US / 24) edges.push(t);
      edges.push(s.to);
      return bucketize(s.reqs, edges, s.busyReqs);
    }
    return bucketize(s.reqs, dayEdges(s.from, s.to), s.busyReqs);
  }, [s, short]);

  // Heatmap: days × hours, at least the last 7 days.
  const heatData = useMemo(() => {
    const from = Math.min(s.from, s.to - 7 * DAY_US);
    const days: number[] = [];
    for (let d = dayStart(from); d < s.to; d = nextDay(d)) days.push(d);
    const cells: [number, number, number][] = [];
    let max = 0;
    // Hours keyed by local hour of day (DST days merge or skip an hour).
    days.forEach((d, yi) => {
      const dayAll = overlapping(s.allReqs, d, nextDay(d));
      const slots = byLocalHour(bucketize(dayAll, hourEdges(d), dayAll));
      slots.forEach((b, h) => {
        if (!b) return;
        const v = heat === "busy" ? b.busy * 100 : heat === "requests" ? b.requests : b.tokensOut;
        max = Math.max(max, v);
        cells.push([h, yi, v]);
      });
    });
    return { days, cells, max };
  }, [s, heat]);

  const heatOption = useMemo(
    () => ({
      tooltip: {
        backgroundColor: pal.surface,
        borderColor: pal.axis,
        textStyle: { color: pal.ink },
        formatter: (p: { value: [number, number, number] }) => {
          const [h, yi, v] = p.value;
          const day = dayLabel(heatData.days[yi]);
          const val = heat === "busy" ? `${v.toFixed(0)}% busy` : `${compact(v)} ${heat === "requests" ? "requests" : "tokens"}`;
          return `<b>${day}</b> · ${String(h).padStart(2, "0")}:00<br/>${val}`;
        },
      },
      grid: { left: 92, right: 16, top: 6, bottom: 24 },
      xAxis: {
        type: "category",
        data: Array.from({ length: 24 }, (_, h) => String(h).padStart(2, "0")),
        splitArea: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: pal.muted, fontSize: 10.5 },
      },
      yAxis: {
        type: "category",
        data: heatData.days.map((d) => dayLabel(d)),
        inverse: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: pal.ink2, fontSize: 11 },
      },
      visualMap: {
        show: false,
        min: 0,
        max: Math.max(1, heatData.max),
        inRange: { color: pal.ramp },
      },
      series: [
        {
          type: "heatmap",
          data: heatData.cells,
          itemStyle: { borderColor: pal.surface, borderWidth: 3, borderRadius: 5 },
          emphasis: { itemStyle: { shadowBlur: 14, shadowColor: alpha(pal.prefill, 0.9) } },
          progressive: 0,
        },
      ],
    }),
    [heatData, heat, pal],
  );

  const heroOption = useMemo(
    () => ({
      grid: { left: 0, right: 0, top: 8, bottom: 18 },
      tooltip: {
        trigger: "axis",
        backgroundColor: pal.surface,
        borderColor: pal.axis,
        textStyle: { color: pal.ink },
        axisPointer: { type: "shadow", shadowStyle: { color: alpha(pal.ink, 0.05) } },
        valueFormatter: (v: number) => compact(v),
      },
      legend: { show: false },
      xAxis: {
        type: "category",
        data: buckets.map((b) =>
          short ? new Date(b.start / 1000).getHours().toString().padStart(2, "0") : dayLabel(b.start, { day: "numeric", month: "short" }),
        ),
        axisLine: { lineStyle: { color: pal.axis } },
        axisTick: { show: false },
        axisLabel: { color: pal.muted, fontSize: 10, interval: "auto" },
      },
      yAxis: { type: "value", show: false },
      series: [
        {
          name: "Tokens written",
          type: "bar",
          data: buckets.map((b) => b.tokensOut),
          barMaxWidth: 18,
          itemStyle: {
            borderRadius: [4, 4, 0, 0],
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: pal.decode },
                { offset: 1, color: alpha(pal.decode, 0.35) },
              ],
            },
            shadowColor: alpha(pal.decode, 0.5),
            shadowBlur: 10,
          },
        },
      ],
    }),
    [buckets, pal, short],
  );

  // Pair diagram: latest memory, average CPU and link throughput.
  const node = (n: "rank0" | "rank1") => {
    const rows = s.sar.filter((x) => x.node === n);
    const last = rows[rows.length - 1];
    const cpu = rows.length ? rows.reduce((a, x) => a + (100 - (x.cpuIdle ?? 100)), 0) / rows.length : null;
    return { avail: gib(last?.memAvail), total: gib(last?.memTotal), cpu };
  };
  const n0 = node("rank0");
  const n1 = node("rank1");
  const ifaces = useMemo(() => linkIfaces(topo, s.net), [topo, s.net]);

  const spark = (f: (b: (typeof buckets)[number]) => number | null) => buckets.map(f);

  return (
    <div className="grid grid-cols-12 gap-4">
      <Card className="col-span-12 xl:col-span-5" title="Tokens written" subtitle={short ? "per hour" : "per day"} delay={0}>
        <div className="flex items-end gap-3">
          <div className="text-[56px] font-semibold leading-none tracking-tight">
            <CountUp value={k.tokensOut} format={(n) => compact(n)} />
          </div>
          <div className="mb-1.5 text-[13px] text-[var(--ink-2)]">
            from <b className="text-[var(--ink)]">{compact(k.requests)}</b> requests ·{" "}
            <b className="text-[var(--ink)]">{compact(k.tokensIn)}</b> read
          </div>
        </div>
        <EChart option={heroOption} className="mt-3 h-36 w-full" />
      </Card>

      <Card className="col-span-12 xl:col-span-7" title="How hard the pair worked" subtitle="share of the period" delay={0.05}>
        <div className="grid h-full grid-cols-3 items-center gap-2">
          <Gauge label="Busy" value={k.busy} color={pal.decode} caption="time with a request in flight" pal={pal} />
          <Gauge label="Prompt reuse" value={k.cachedShare} color={pal.cache} caption="prompt tokens from cache" pal={pal} />
          <Gauge label="Draft acceptance" value={k.acceptance} color={pal.prefill} caption="MTP drafts kept" pal={pal} />
        </div>
      </Card>

      <Card
        className="col-span-12"
        title="Activity"
        subtitle="day × hour · click a cell to open that day"
        delay={0.1}
        right={
          <div className="flex gap-1.5">
            {(Object.keys(HEAT_LABEL) as HeatMetric[]).map((m) => (
              <button key={m} className="chip" data-on={heat === m} onClick={() => setHeat(m)}>
                {HEAT_LABEL[m]}
              </button>
            ))}
          </div>
        }
      >
        <EChart
          option={heatOption}
          replace={false}
          className="w-full"
          style={{ height: 34 + heatData.days.length * 26 }}
          label={`Activity: ${HEAT_LABEL[heat]} by day and hour`}
          onClick={(p) => {
            const v = p.value as [number, number, number];
            onOpenDay(heatData.days[v[1]]);
          }}
        />
        <HeatLegend pal={pal} max={heatData.max} metric={heat} />
      </Card>

      <Card className="col-span-12 xl:col-span-5" title="The pair" subtitle="memory free now · link traffic while busy, and the busiest sar interval" delay={0.15}>
        <div className="h-64">
          <PairDiagram
            pal={pal}
            left={{ name: nodeName(topo, "rank0"), role: "rank 0", memAvailGiB: n0.avail, memTotalGiB: n0.total, cpu: n0.cpu, color: pal.rank0 }}
            right={{ name: nodeName(topo, "rank1"), role: "rank 1", memAvailGiB: n1.avail, memTotalGiB: n1.total, cpu: n1.cpu, color: pal.rank1 }}
            links={ifaces.map((i) => ({ name: i, ...linkRate(s.net, i) }))}
          />
        </div>
      </Card>

      <div className="col-span-12 grid grid-cols-2 gap-4 md:grid-cols-3 xl:col-span-7">
        <StatTile
          pal={pal}
          label="Reading speed (median)"
          value={k.prefillMed}
          format={(n) => compact(n)}
          unit="tok/s"
          color={pal.prefill}
          spark={spark((b) => b.prefillMed)}
          delta={delta(k.prefillMed, kp?.prefillMed)}
        />
        <StatTile
          pal={pal}
          label="Writing speed (median)"
          value={k.decodeMed}
          format={(n) => (n == null ? "—" : n.toFixed(1))}
          unit="tok/s"
          color={pal.decode}
          spark={spark((b) => b.decodeMed)}
          delta={delta(k.decodeMed, kp?.decodeMed)}
        />
        <StatTile
          pal={pal}
          label="Tokens read"
          value={k.tokensIn}
          format={(n) => compact(n)}
          color={pal.prefill}
          spark={spark((b) => b.tokensIn)}
          delta={delta(k.tokensIn, kp?.tokensIn)}
        />
        <StatTile
          pal={pal}
          label="Requests"
          value={k.requests}
          format={(n) => compact(n)}
          color={pal.rank0}
          spark={spark((b) => b.requests)}
          delta={delta(k.requests, kp?.requests)}
        />
        <StatTile
          pal={pal}
          label="Clients that left early"
          value={k.aborted}
          format={(n) => compact(n)}
          color={pal.rank1}
          spark={spark((b) => b.aborted)}
          delta={delta(k.aborted, kp?.aborted)}
          goodWhenUp={false}
        />
        <StatTile
          pal={pal}
          label="Restarts"
          value={k.restarts}
          format={(n) => compact(n)}
          color={pal.memory}
          spark={buckets.map((b) => s.runs.filter((r) => r.node === "rank0" && r.ts >= b.start && r.ts < b.end).length)}
          delta={delta(k.restarts, kp?.restarts)}
          goodWhenUp={false}
        />
      </div>
    </div>
  );
}

function HeatLegend({ pal, max, metric }: { pal: Palette; max: number; metric: HeatMetric }) {
  const fmt = (v: number) => (metric === "busy" ? `${v.toFixed(0)}%` : compact(v));
  return (
    <div className="mt-1 flex items-center justify-end gap-2 text-[11px] text-[var(--muted)]">
      <span>{fmt(0)}</span>
      <span
        className="h-2 w-40 rounded-full"
        style={{ background: `linear-gradient(90deg, ${pal.ramp.join(", ")})` }}
      />
      <span>{fmt(max)}</span>
    </div>
  );
}

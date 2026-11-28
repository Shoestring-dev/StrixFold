import { useMemo } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { Card } from "../components/Card";
import { NoData } from "../components/NoData";
import { EChart } from "../components/EChart";
import { bucketize, busyShare, byLocalHour, compact, dayLabel, gib, hourEdges, nextDay, timeLabel, type Req } from "../lib/stats";
import { axisCommon, esc, tooltipBase } from "../lib/chartkit";
import { alpha, type Palette } from "../lib/theme";
import { nodeName, sarNote } from "../lib/topology";
import type { SarRow, Topology } from "../lib/types";

interface Props {
  day: number; // local midnight, µs
  days: number[]; // every day with data, ascending
  reqs: Req[]; // all requests (the view slices its day)
  sar: SarRow[];
  topo: Topology | null;
  pal: Palette;
  onDay: (d: number) => void;
}

/** Slots a run can serve in parallel. */
const MIN_LANES = 4;

export function Day({ day, days, reqs, sar, topo, pal, onDay }: Props) {
  const end = nextDay(day);
  const dayReqs = useMemo(() => reqs.filter((r) => r.tEnd >= day && (r.tStart ?? r.tEnd) < end), [reqs, day, end]);
  // Lanes and busy time use every request overlapping the day; counts and tokens go to the hour a request finished in.
  const hours = useMemo(() => bucketize(dayReqs, hourEdges(day), dayReqs), [dayReqs, day]);
  // Charts always show 24 local-hour slots: fall-back days merge the repeated hour, spring-forward days leave one empty.
  const slots = useMemo(() => byLocalHour(hours), [hours]);
  // One lane per slot the day actually used, never fewer than four.
  const lanes = useMemo(() => dayReqs.reduce((m, r) => Math.max(m, (r.slot ?? 0) + 1), MIN_LANES), [dayReqs]);
  const i = days.indexOf(day);

  const lanesOption = useMemo(() => {
    const data = dayReqs
      .filter((r) => r.slot != null && r.tStart != null)
      .map((r, idx) => [r.slot, r.tStart! / 1000, (r.tPrefill ?? r.tStart!) / 1000, r.tEnd / 1000, idx]);
    const byIdx = dayReqs.filter((r) => r.slot != null && r.tStart != null);
    return {
      tooltip: {
        ...tooltipBase(pal),
        formatter: (p: { value: number[] }) => {
          const r = byIdx[p.value[4]];
          const rows = [
            `<b>${timeLabel(r.tStart!)} → ${timeLabel(r.tEnd)}</b> · slot ${r.slot}`,
            `${compact(r.prompt)} prompt tokens (${compact(r.cached)} cached) · ${compact(r.generated)} written`,
            r.prefillTps ? `reading ${compact(r.prefillTps)} tok/s` : "",
            r.decodeTps ? `writing ${r.decodeTps.toFixed(1)} tok/s` : "",
            `finish: ${esc(r.finish ?? "?")}${r.client ? ` · ${esc(r.client)}` : ""}`,
          ];
          return rows.filter(Boolean).join("<br/>");
        },
      },
      grid: { left: 56, right: 16, top: 8, bottom: 58 },
      dataZoom: [
        { type: "inside", xAxisIndex: 0, filterMode: "weakFilter" },
        {
          type: "slider",
          xAxisIndex: 0,
          height: 22,
          bottom: 8,
          filterMode: "weakFilter",
          borderColor: pal.axis,
          fillerColor: alpha(pal.prefill, 0.15),
          handleStyle: { color: pal.prefill, borderColor: pal.prefill },
          moveHandleStyle: { color: pal.prefill },
          textStyle: { color: pal.muted },
          dataBackground: { lineStyle: { color: pal.axis }, areaStyle: { color: alpha(pal.prefill, 0.08) } },
          labelFormatter: (v: number) => timeLabel(v * 1000),
        },
      ],
      xAxis: {
        type: "time",
        min: day / 1000,
        max: end / 1000,
        ...axisCommon(pal),
        splitLine: { show: true, lineStyle: { color: pal.grid } },
        axisLabel: { color: pal.muted, fontSize: 10.5, formatter: (v: number) => timeLabel(v * 1000) },
      },
      yAxis: {
        type: "category",
        data: Array.from({ length: lanes }, (_, k) => `slot ${k}`),
        inverse: true,
        ...axisCommon(pal),
        splitLine: { show: false },
      },
      series: [
        {
          type: "custom",
          clip: true,
          encode: { x: [1, 3], y: 0 },
          data,
          renderItem: (
            _params: unknown,
            api: {
              value: (i: number) => number;
              coord: (v: [number, number]) => [number, number];
              size: (v: [number, number]) => [number, number];
            },
          ) => {
            const lane = api.value(0);
            const [x0, y] = api.coord([api.value(1), lane]);
            const [x1] = api.coord([api.value(2), lane]);
            const [x2] = api.coord([api.value(3), lane]);
            const h = api.size([0, 1])[1] * 0.56;
            const w1 = Math.max(1.5, x1 - x0);
            const w2 = Math.max(1.5, x2 - x1);
            return {
              type: "group",
              children: [
                {
                  type: "rect",
                  shape: { x: x0, y: y - h / 2, width: w1, height: h, r: [4, 0, 0, 4] },
                  style: { fill: pal.prefill, shadowBlur: 8, shadowColor: alpha(pal.prefill, 0.6) },
                },
                {
                  type: "rect",
                  shape: { x: x0 + w1, y: y - h / 2, width: w2, height: h, r: [0, 4, 4, 0] },
                  style: { fill: pal.decode, shadowBlur: 8, shadowColor: alpha(pal.decode, 0.6) },
                },
              ],
            };
          },
        },
      ],
    };
  }, [dayReqs, pal, day, end, lanes]);

  const hourLabels = Array.from({ length: 24 }, (_, h) => String(h).padStart(2, "0"));

  const columns = (title: string, data: (number | null)[], color: string) => ({
    tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => compact(v) },
    grid: { left: 44, right: 8, top: 10, bottom: 22 },
    xAxis: { type: "category", data: hourLabels, ...axisCommon(pal) },
    yAxis: { type: "value", ...axisCommon(pal), axisLabel: { color: pal.muted, fontSize: 10, formatter: (v: number) => compact(v, 0) } },
    series: [
      {
        name: title,
        type: "bar",
        data,
        barMaxWidth: 14,
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color },
              { offset: 1, color: alpha(color, 0.3) },
            ],
          },
          shadowColor: alpha(color, 0.55),
          shadowBlur: 10,
        },
      },
    ],
  });

  const band = (title: string, med: (number | null)[], lo: (number | null)[], hi: (number | null)[], color: string, digits: number) => ({
    tooltip: {
      ...tooltipBase(pal),
      trigger: "axis",
      formatter: (ps: { dataIndex: number }[]) => {
        const i = ps[0].dataIndex;
        const f = (v: number | null) => (v == null ? "—" : v.toFixed(digits));
        return `<b>${hourLabels[i]}:00</b><br/>median ${f(med[i])} tok/s<br/>p10–p90 ${f(lo[i])}–${f(hi[i])}`;
      },
    },
    grid: { left: 44, right: 8, top: 10, bottom: 22 },
    xAxis: { type: "category", data: hourLabels, boundaryGap: false, ...axisCommon(pal) },
    yAxis: { type: "value", ...axisCommon(pal), scale: true },
    series: [
      // p10–p90 band: a transparent base plus a stacked wash.
      { type: "line", data: lo, stack: "band", lineStyle: { opacity: 0 }, showSymbol: false, connectNulls: false, silent: true },
      {
        type: "line",
        data: hi.map((h, k) => (h == null || lo[k] == null ? null : h - (lo[k] as number))),
        stack: "band",
        lineStyle: { opacity: 0 },
        showSymbol: false,
        areaStyle: { color: alpha(color, 0.16) },
        silent: true,
      },
      {
        name: title,
        type: "line",
        data: med,
        smooth: 0.3,
        symbol: "circle",
        symbolSize: 8,
        itemStyle: { color, borderColor: pal.surface, borderWidth: 2 },
        lineStyle: { width: 2.5, color, shadowColor: alpha(color, 0.7), shadowBlur: 12 },
      },
    ],
  });

  const memOption = useMemo(() => {
    const series = (["rank0", "rank1"] as const).map((n) => {
      const color = n === "rank0" ? pal.rank0 : pal.rank1;
      return {
        name: nodeName(topo, n),
        type: "line",
        smooth: 0.3,
        showSymbol: false,
        data: sar.filter((x) => x.node === n && x.ts >= day && x.ts < end).map((x) => [x.ts / 1000, gib(x.memAvail)]),
        lineStyle: { width: 2.5, color, shadowColor: alpha(color, 0.6), shadowBlur: 10 },
        itemStyle: { color },
        areaStyle: { color: alpha(color, 0.08) },
        endLabel: { show: true, color: pal.ink2, fontSize: 11, formatter: (p: { seriesName: string }) => p.seriesName },
      };
    });
    (series[0] as Record<string, unknown>).markLine = {
      silent: true,
      symbol: "none",
      lineStyle: { color: pal.warning, type: "solid", width: 1 },
      label: { color: pal.muted, formatter: "8 GiB floor", position: "insideStartTop", fontSize: 10 },
      data: [{ yAxis: 8 }],
    };
    return {
      tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => (v == null ? "—" : `${v.toFixed(1)} GiB`) },
      legend: { top: 0, right: 0, textStyle: { color: pal.ink2 }, icon: "circle", itemWidth: 8, itemHeight: 8 },
      grid: { left: 44, right: 70, top: 28, bottom: 22 },
      xAxis: {
        type: "time",
        min: day / 1000,
        max: end / 1000,
        ...axisCommon(pal),
        splitLine: { show: false },
        axisLabel: { color: pal.muted, fontSize: 10.5, formatter: (v: number) => timeLabel(v * 1000) },
      },
      yAxis: { type: "value", name: "GiB free", ...axisCommon(pal), min: 0 },
      series,
    };
  }, [sar, day, end, topo, pal]);
  const sarEmpty = sarNote(topo, sar.filter((x) => x.ts >= day && x.ts < end).length);

  const totals = {
    req: hours.reduce((a, h) => a + h.requests, 0),
    out: hours.reduce((a, h) => a + h.tokensOut, 0),
    busy: busyShare(hours),
  };

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 flex items-center gap-3">
        <button className="chip !px-2" disabled={i <= 0} onClick={() => onDay(days[i - 1])} aria-label="Previous day">
          <ChevronLeft size={16} />
        </button>
        <h1 className="text-[26px] font-semibold tracking-tight">{dayLabel(day, { weekday: "long", month: "long", day: "numeric" })}</h1>
        <button className="chip !px-2" disabled={i < 0 || i >= days.length - 1} onClick={() => onDay(days[i + 1])} aria-label="Next day">
          <ChevronRight size={16} />
        </button>
        <div className="ml-4 flex gap-5 text-[13px] text-[var(--ink-2)]">
          <span>
            <b className="text-[var(--ink)]">{compact(totals.req)}</b> requests
          </span>
          <span>
            <b className="text-[var(--ink)]">{compact(totals.out)}</b> tokens written
          </span>
          <span>
            <b className="text-[var(--ink)]">{(totals.busy * 100).toFixed(0)}%</b> busy
          </span>
        </div>
        <div className="ml-auto flex max-w-[45%] gap-1 overflow-x-auto">
          {days.slice(-10).map((d) => (
            <button key={d} className="chip whitespace-nowrap" data-on={d === day} onClick={() => onDay(d)}>
              {dayLabel(d, { month: "short", day: "numeric" })}
            </button>
          ))}
        </div>
      </div>

      <Card
        className="col-span-12"
        title="Lanes"
        subtitle="every request in its slot: reading the prompt, then writing the reply · scroll or drag to zoom"
        right={<Legend items={[{ c: pal.prefill, t: "reading (prefill)" }, { c: pal.decode, t: "writing (decode)" }]} />}
      >
        <EChart option={lanesOption} className="h-72 w-full" label="Requests per slot: prompt pass then reply" />
      </Card>

      <Card className="col-span-12 lg:col-span-6" title="Tokens read" subtitle="prompt tokens per hour" delay={0.05}>
        <EChart option={columns("Tokens read", slots.map((s) => s?.tokensIn ?? null), pal.prefill)} className="h-48 w-full" label="Prompt tokens per hour" />
      </Card>
      <Card className="col-span-12 lg:col-span-6" title="Tokens written" subtitle="reply tokens per hour" delay={0.08}>
        <EChart option={columns("Tokens written", slots.map((s) => s?.tokensOut ?? null), pal.decode)} className="h-48 w-full" label="Reply tokens per hour" />
      </Card>

      <Card className="col-span-12 lg:col-span-6" title="Reading speed" subtitle="tok/s per hour · median with p10–p90 band" delay={0.1}>
        <EChart
          option={band("Reading", slots.map((s) => s?.prefillMed ?? null), slots.map((s) => s?.prefillP10 ?? null), slots.map((s) => s?.prefillP90 ?? null), pal.prefill, 0)}
          className="h-48 w-full"
          label="Reading speed per hour, median with p10–p90"
        />
      </Card>
      <Card className="col-span-12 lg:col-span-6" title="Writing speed" subtitle="tok/s per hour · median with p10–p90 band" delay={0.12}>
        <EChart
          option={band("Writing", slots.map((s) => s?.decodeMed ?? null), slots.map((s) => s?.decodeP10 ?? null), slots.map((s) => s?.decodeP90 ?? null), pal.decode, 1)}
          className="h-48 w-full"
          label="Writing speed per hour, median with p10–p90"
        />
      </Card>

      <Card className="col-span-12" title="Memory free" subtitle="MemAvailable per node (sar, 10-minute samples)" delay={0.15}>
        {sarEmpty ? (
          <NoData note={sarEmpty} className="h-56 w-full" />
        ) : (
          <EChart option={memOption} className="h-56 w-full" label="Memory available on both nodes" />
        )}
      </Card>
    </div>
  );
}

function Legend({ items }: { items: { c: string; t: string }[] }) {
  return (
    <div className="flex gap-4 text-[12px] text-[var(--ink-2)]">
      {items.map((it) => (
        <span key={it.t} className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: it.c, boxShadow: `0 0 8px ${it.c}` }} />
          {it.t}
        </span>
      ))}
    </div>
  );
}


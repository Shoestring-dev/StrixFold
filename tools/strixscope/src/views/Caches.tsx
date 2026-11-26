import { useMemo } from "react";

import { Card } from "../components/Card";
import { EChart } from "../components/EChart";
import { Gauge } from "../components/Gauge";
import { StatTile } from "../components/StatTile";
import { axisCommon, barFill, glowLine, gradientFill, legendTop, timeAxis, tooltipBase, zoomSlider } from "../lib/chartkit";
import { compact, counterGrowth, dayEdges, dayLabel, promptCacheHitRate, sum } from "../lib/stats";
import { alpha, type Palette } from "../lib/theme";
import type { Slice } from "./Overview";

const RESUME = [
  { key: "none", label: "computed from scratch" },
  { key: "ram", label: "RAM state, in place" },
  { key: "ram-moved", label: "RAM state, moved" },
  { key: "disk", label: "disk prefix cache" },
] as const;

export function Caches({ s, pal }: { s: Slice; pal: Palette }) {
  const pc = useMemo(() => s.events.filter((e) => e.kind === "pcache"), [s]);
  const last = pc[pc.length - 1];
  const capMib = last?.text ? Number(last.text.split(" ")[0]) : 2048;
  const fill = last ? (last.v[1] ?? 0) / capMib : null;
  const runStart = useMemo(() => new Map(s.allRuns.map((r) => [r.id, r.ts])), [s]);
  const hitRate = useMemo(() => promptCacheHitRate(pc, runStart, s.from), [pc, runStart, s]);
  const disk = useMemo(() => {
    const saves = s.events.filter((e) => e.kind === "disk_save" && (e.v[3] ?? 0) === 0);
    const hits = s.events.filter((e) => e.kind === "disk_hit");
    const misses = s.events.filter((e) => e.kind === "disk_miss");
    const bytes = sum(saves.map((e) => e.v[1] ?? 0));
    const ms = sum(saves.map((e) => e.v[2] ?? 0));
    return { saves, hits, misses, bytes, mbps: ms ? bytes / 1048576 / (ms / 1000) : null };
  }, [s]);
  const stale = s.events.filter((e) => e.kind === "stale").length;
  const evicted = useMemo(() => counterGrowth(pc, "pcache", 4, runStart, s.from), [pc, runStart, s]);

  // Prompt cache MiB over time (with the cap).
  const mibOption = useMemo(
    () => ({
      tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => `${compact(v)} MiB` },
      grid: { left: 48, right: 34, top: 16, bottom: 44 },
      dataZoom: zoomSlider(pal),
      xAxis: timeAxis(pal, s.from, s.to),
      yAxis: { type: "value", name: "MiB", ...axisCommon(pal), min: 0, max: Math.max(capMib, 1) },
      series: [
        {
          name: "RAM prompt cache",
          type: "line",
          step: "end",
          showSymbol: false,
          data: pc.map((e) => [e.ts / 1000, e.v[1]]),
          lineStyle: glowLine(pal.cache, 2),
          areaStyle: gradientFill(pal.cache, 0.3),
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: pal.muted, type: "solid", width: 1 },
            label: { color: pal.muted, formatter: "cap", fontSize: 10 },
            data: [{ yAxis: capMib }],
          },
        },
      ],
    }),
    [pc, pal, s, capMib],
  );

  // Where each prompt resumed from, per day (100% stacked).
  const resumeOption = useMemo(() => {
    const edges = dayEdges(s.from, s.to);
    const days = edges.slice(0, -1);
    const shares = RESUME.map((k) =>
      days.map((d, i) => {
        const inDay = s.reqs.filter((r) => r.tEnd >= d && r.tEnd < edges[i + 1] && r.resume);
        return inDay.length ? inDay.filter((r) => r.resume === k.key).length / inDay.length : 0;
      }),
    );
    const colors = [pal.other, pal.cache, alpha(pal.cache, 0.55), pal.memory];
    return {
      tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => `${(v * 100).toFixed(0)}%`, axisPointer: { type: "shadow", shadowStyle: { color: alpha(pal.ink, 0.05) } } },
      legend: legendTop(pal),
      grid: { left: 40, right: 8, top: 34, bottom: 24 },
      xAxis: { type: "category", data: days.map((d) => dayLabel(d, { month: "short", day: "numeric" })), ...axisCommon(pal) },
      yAxis: { type: "value", max: 1, ...axisCommon(pal), axisLabel: { color: pal.muted, fontSize: 10.5, formatter: (v: number) => `${v * 100}%` } },
      series: RESUME.map((k, i) => ({
        name: k.label,
        type: "bar",
        stack: "r",
        barMaxWidth: 24,
        data: shares[i],
        itemStyle: { color: colors[i], borderColor: pal.surface, borderWidth: 1, borderRadius: i === RESUME.length - 1 ? [4, 4, 0, 0] : 0 },
      })),
    };
  }, [s, pal]);

  // Disk cache: bytes written per day.
  const diskOption = useMemo(() => {
    const edges = dayEdges(s.from, s.to);
    const days = edges.slice(0, -1);
    const per = days.map((d, i) => sum(disk.saves.filter((e) => e.ts >= d && e.ts < edges[i + 1]).map((e) => e.v[1] ?? 0)) / 1048576);
    return {
      tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => `${compact(v)} MiB` },
      grid: { left: 48, right: 8, top: 12, bottom: 24 },
      xAxis: { type: "category", data: days.map((d) => dayLabel(d, { month: "short", day: "numeric" })), ...axisCommon(pal) },
      yAxis: { type: "value", name: "MiB", ...axisCommon(pal) },
      series: [{ name: "written to disk", type: "bar", data: per, barMaxWidth: 22, itemStyle: barFill(pal.memory) }],
    };
  }, [disk, s, pal]);

  // Disk write throughput per save (MiB/s) over time.
  const diskRate = useMemo(
    () => ({
      tooltip: { ...tooltipBase(pal), formatter: (p: { value: number[] }) => `${p.value[1].toFixed(0)} MiB/s` },
      grid: { left: 48, right: 12, top: 12, bottom: 24 },
      xAxis: timeAxis(pal, s.from, s.to),
      yAxis: { type: "value", name: "MiB/s", ...axisCommon(pal) },
      series: [
        {
          type: "scatter",
          symbolSize: 6,
          data: disk.saves.filter((e) => (e.v[2] ?? 0) > 0).map((e) => [e.ts / 1000, (e.v[1] ?? 0) / 1048576 / ((e.v[2] ?? 1) / 1000)]),
          itemStyle: { color: alpha(pal.memory, 0.8), shadowBlur: 6, shadowColor: alpha(pal.memory, 0.6) },
        },
      ],
    }),
    [disk, s, pal],
  );

  const images = useMemo(() => {
    const memo = s.events.filter((e) => e.kind === "image_memo");
    const enc = s.events.filter((e) => e.kind === "encoder");
    const hits = sum(memo.map((e) => e.v[1] ?? 0));
    const misses = sum(memo.map((e) => e.v[2] ?? 0));
    const encodedMs = enc.filter((e) => (e.v[1] ?? 0) > 0);
    return {
      pictures: sum(memo.map((e) => e.v[0] ?? 0)),
      hitRate: hits + misses ? hits / (hits + misses) : null,
      msPerPicture: encodedMs.length ? sum(encodedMs.map((e) => e.v[3] ?? 0)) / sum(encodedMs.map((e) => e.v[1] ?? 0)) : null,
    };
  }, [s]);

  return (
    <div className="grid grid-cols-12 gap-4">
      <Card className="col-span-12 xl:col-span-6" title="RAM prompt cache" subtitle="kept states that let a prompt resume instead of re-reading">
        <div className="grid grid-cols-3 items-center">
          <Gauge label="Filled" value={fill} color={pal.cache} caption={last ? `${compact(last.v[1])} of ${compact(capMib)} MiB · ${compact(last.v[0])} states` : "no data"} pal={pal} />
          <Gauge label="Hit rate" value={hitRate} color={pal.cache} caption="prompts that found a kept state" pal={pal} />
          <Gauge
            label="Disk hits"
            value={disk.hits.length + disk.misses.length ? disk.hits.length / (disk.hits.length + disk.misses.length) : null}
            color={pal.memory}
            caption={`${compact(disk.hits.length)} resumes from disk`}
            pal={pal}
          />
        </div>
      </Card>
      <div className="col-span-12 grid grid-cols-2 gap-4 xl:col-span-6">
        <StatTile pal={pal} label="States evicted (cap reached)" value={evicted} format={(n) => compact(n)} color={pal.cache} />
        <StatTile pal={pal} label="Kept states gone stale" value={stale} format={(n) => compact(n)} color={pal.rank1} />
        <StatTile pal={pal} label="Written to the disk cache" value={disk.bytes / 1048576 / 1024} format={(n) => (n == null ? "—" : n.toFixed(1))} unit="GiB" color={pal.memory} />
        <StatTile pal={pal} label="Disk write speed" value={disk.mbps} format={(n) => compact(n)} unit="MiB/s" color={pal.memory} />
      </div>

      <Card className="col-span-12" title="RAM prompt cache over time" subtitle="MiB held after each prompt · drag the slider to zoom" delay={0.05}>
        <EChart option={mibOption} className="h-60 w-full" />
      </Card>

      <Card className="col-span-12 xl:col-span-7" title="Where prompts resumed from" subtitle="share of requests per day" delay={0.08}>
        <EChart option={resumeOption} className="h-64 w-full" />
      </Card>
      <Card className="col-span-12 xl:col-span-5" title="Disk prefix cache" subtitle="MiB written per day" delay={0.1}>
        <EChart option={diskOption} className="h-64 w-full" />
      </Card>

      <Card className="col-span-12 xl:col-span-7" title="Disk write throughput" subtitle="MiB/s per saved state" delay={0.12}>
        <EChart option={diskRate} className="h-52 w-full" />
      </Card>
      <Card className="col-span-12 xl:col-span-5" title="Images" subtitle="vision memo and encoder" delay={0.14}>
        <div className="grid h-full grid-cols-3 items-center gap-3 text-center">
          <Big label="pictures" value={compact(images.pictures)} />
          <Big label="memo hit rate" value={images.hitRate == null ? "—" : `${(images.hitRate * 100).toFixed(0)}%`} />
          <Big label="ms per encode" value={compact(images.msPerPicture)} />
        </div>
      </Card>
    </div>
  );
}

function Big({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[32px] font-semibold tracking-tight">{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";

import { Card } from "../components/Card";
import { EChart } from "../components/EChart";
import { axisCommon, esc, glowLine, gradientFill, legendTop, tooltipBase } from "../lib/chartkit";
import { loadGL, webglAvailable } from "../lib/echarts";
import { bucketize, byLocalHour, compact, dayEdges, dayLabel, dayStart, hourEdges, nextDay, quantile, sortedOf, type Req } from "../lib/stats";
import { alpha, type Palette } from "../lib/theme";
import type { RunRow } from "../lib/types";
import type { Slice } from "./Overview";

/** Ordinal ramp of one hue for 1 / 2 / 3+ live streams (light → strong). */
function streamColors(pal: Palette) {
  return pal.mode === "dark" ? ["#f6b493", pal.decode, "#9c3a12"] : ["#f0a27f", pal.decode, "#8a3208"];
}
const STREAMS = ["1 stream", "2 streams", "3–4 streams"];
const streamIdx = (r: Req) => Math.min(2, r.others ?? 0);

const log10 = (x: number) => Math.log10(Math.max(1, x));
const logTicks = (v: number) => compact(10 ** v, 0);

export function Performance({ s, pal }: { s: Slice; pal: Palette }) {
  const [gl, setGl] = useState<"loading" | "ready" | "none">(() => (webglAvailable() ? "loading" : "none"));
  useEffect(() => {
    if (gl !== "loading") return;
    loadGL().then(
      () => setGl("ready"),
      () => setGl("none"),
    );
  }, [gl]);

  const dec = useMemo(() => s.reqs.filter((r) => r.decodeTps != null && r.decodeTps < 400), [s]);
  // Above ~4K tok/s is a mismatched pass/finish pair, not reading: leave those out of the speed charts.
  const pre = useMemo(() => s.reqs.filter((r) => r.prefillTps != null && r.prefillTps < 4000), [s]);

  // 3D landscape: context (log) × live streams × writing speed.
  const landscape = useMemo(() => {
    const sc = streamColors(pal);
    return {
      tooltip: {
        ...tooltipBase(pal),
        formatter: (p: { value: number[] }) =>
          `${compact(10 ** p.value[0])} tokens of context<br/>${p.value[1]} stream(s) live<br/><b>${p.value[2].toFixed(1)} tok/s</b>`,
      },
      grid3D: {
        boxWidth: 110,
        boxDepth: 70,
        boxHeight: 70,
        environment: "none",
        viewControl: { autoRotate: true, autoRotateSpeed: 6, distance: 150, alpha: 20, beta: 35 },
        axisLine: { lineStyle: { color: pal.axis } },
        axisPointer: { lineStyle: { color: alpha(pal.decode, 0.7) } },
        splitLine: { lineStyle: { color: pal.grid } },
        axisLabel: { textStyle: { color: pal.muted, fontSize: 10 } },
        axisTick: { show: false },
        light: { main: { intensity: 1.1 }, ambient: { intensity: 0.5 } },
      },
      xAxis3D: { type: "value", name: "context", nameTextStyle: { color: pal.muted }, axisLabel: { formatter: logTicks, textStyle: { color: pal.muted } } },
      yAxis3D: { type: "value", name: "streams", min: 1, max: 4, interval: 1, nameTextStyle: { color: pal.muted } },
      zAxis3D: { type: "value", name: "tok/s", nameTextStyle: { color: pal.muted } },
      series: [0, 1, 2].map((k) => ({
        type: "scatter3D",
        name: STREAMS[k],
        symbolSize: 5,
        itemStyle: { color: sc[k], opacity: 0.9 },
        emphasis: { itemStyle: { color: pal.ink } },
        data: dec.filter((r) => streamIdx(r) === k).map((r) => [log10(r.prompt + r.generated), (r.others ?? 0) + 1, r.decodeTps]),
      })),
      legend: { ...legendTop(pal), data: STREAMS },
    };
  }, [dec, pal]);

  // 3D skyline: days × hours, height = tokens written.
  const skyline = useMemo(() => {
    const days: number[] = [];
    for (let d = dayStart(s.from); d < s.to; d = nextDay(d)) days.push(d);
    const shown = days.slice(-14);
    const data: [number, number, number][] = [];
    let max = 1;
    shown.forEach((d, di) => {
      const inDay = s.reqs.filter((r) => r.tEnd >= d && r.tEnd < nextDay(d));
      byLocalHour(bucketize(inDay, hourEdges(d))).forEach((b, h) => {
        if (!b) return;
        max = Math.max(max, b.tokensOut);
        data.push([h, di, b.tokensOut]);
      });
    });
    return {
      tooltip: {
        ...tooltipBase(pal),
        formatter: (p: { value: number[] }) =>
          `<b>${dayLabel(shown[p.value[1]])}</b> · ${String(p.value[0]).padStart(2, "0")}:00<br/>${compact(p.value[2])} tokens written`,
      },
      visualMap: { show: false, min: 0, max, inRange: { color: pal.ramp.slice(2) } },
      grid3D: {
        boxWidth: 120,
        boxDepth: 80,
        boxHeight: 50,
        environment: "none",
        viewControl: { autoRotate: true, autoRotateSpeed: 4, distance: 170, alpha: 28, beta: 20 },
        axisLine: { lineStyle: { color: pal.axis } },
        splitLine: { lineStyle: { color: pal.grid } },
        axisLabel: { textStyle: { color: pal.muted, fontSize: 9.5 } },
        axisTick: { show: false },
        axisPointer: { show: false },
        // No shadow maps: WebKitGTK (the Linux webview) draws no bars with them on.
        light: { main: { intensity: 1.2 }, ambient: { intensity: 0.4 } },
      },
      xAxis3D: { type: "category", data: Array.from({ length: 24 }, (_, h) => String(h).padStart(2, "0")), name: "" },
      yAxis3D: { type: "category", data: shown.map((d) => dayLabel(d, { month: "short", day: "numeric" })), name: "" },
      zAxis3D: { type: "value", name: "", axisLabel: { formatter: (v: number) => compact(v, 0), textStyle: { color: pal.muted } } },
      series: [
        {
          type: "bar3D",
          data,
          shading: "lambert",
          bevelSize: 0.3,
          itemStyle: { opacity: 0.95 },
          emphasis: { itemStyle: { color: pal.decode } },
        },
      ],
    };
  }, [s, pal]);

  // 2D: writing speed vs context, by live streams (ordinal ramp).
  const decScatter = useMemo(() => {
    const sc = streamColors(pal);
    return {
      tooltip: {
        ...tooltipBase(pal),
        formatter: (p: { value: number[]; seriesName: string }) =>
          `${compact(10 ** p.value[0])} tokens of context · ${p.seriesName}<br/><b>${p.value[1].toFixed(1)} tok/s</b>`,
      },
      legend: legendTop(pal),
      grid: { left: 44, right: 12, top: 30, bottom: 30 },
      xAxis: { type: "value", name: "context tokens", nameLocation: "middle", nameGap: 22, ...axisCommon(pal), min: 1, axisLabel: { color: pal.muted, fontSize: 10.5, formatter: logTicks } },
      yAxis: { type: "value", name: "tok/s", ...axisCommon(pal) },
      series: [0, 1, 2].map((k) => ({
        type: "scatter",
        name: STREAMS[k],
        symbolSize: 6,
        large: true,
        itemStyle: { color: alpha(sc[k], 0.75), borderColor: pal.surface, borderWidth: 0.5 },
        data: dec.filter((r) => streamIdx(r) === k).map((r) => [log10(r.prompt + r.generated), r.decodeTps]),
      })),
    };
  }, [dec, pal]);

  const preScatter = useMemo(
    () => ({
      tooltip: {
        ...tooltipBase(pal),
        formatter: (p: { value: number[] }) => `${compact(10 ** p.value[0])} new prompt tokens<br/><b>${compact(p.value[1])} tok/s</b>`,
      },
      grid: { left: 50, right: 12, top: 14, bottom: 30 },
      xAxis: { type: "value", name: "new prompt tokens", nameLocation: "middle", nameGap: 22, ...axisCommon(pal), min: 2, axisLabel: { color: pal.muted, fontSize: 10.5, formatter: logTicks } },
      yAxis: { type: "value", name: "tok/s", ...axisCommon(pal), axisLabel: { color: pal.muted, fontSize: 10.5, formatter: (v: number) => compact(v, 0) } },
      series: [
        {
          type: "scatter",
          symbolSize: 6,
          large: true,
          itemStyle: { color: alpha(pal.prefill, 0.7), shadowBlur: 6, shadowColor: alpha(pal.prefill, 0.5) },
          data: pre.map((r) => [log10(r.fresh), r.prefillTps]),
        },
      ],
    }),
    [pre, pal],
  );

  // Per deploy: median writing / reading speed with p10–p90 whiskers.
  const byDeploy = useMemo(() => {
    const runDeploy = new Map<number, RunRow>(s.allRuns.map((r) => [r.id, r]));
    const groups = new Map<string, { first: number; dec: number[]; pre: number[]; n: number }>();
    for (const r of s.reqs) {
      const run = r.run != null ? runDeploy.get(r.run) : undefined;
      const key = run?.deploy ?? "unknown";
      const g = groups.get(key) ?? { first: run?.ts ?? r.tEnd, dec: [], pre: [], n: 0 };
      g.n++;
      if (r.decodeTps != null) g.dec.push(r.decodeTps);
      if (r.prefillTps != null) g.pre.push(r.prefillTps);
      groups.set(key, g);
    }
    const rows = [...groups.entries()]
      .filter(([, g]) => g.dec.length >= 5)
      .sort((a, b) => a[1].first - b[1].first)
      .map(([k, g]) => {
        const d = sortedOf(g.dec);
        return { deploy: k.slice(0, 12), first: g.first, n: g.n, p10: quantile(d, 0.1)!, med: quantile(d, 0.5)!, p90: quantile(d, 0.9)! };
      });
    return {
      tooltip: {
        ...tooltipBase(pal),
        trigger: "axis",
        axisPointer: { type: "shadow", shadowStyle: { color: alpha(pal.ink, 0.05) } },
        formatter: (ps: { dataIndex: number }[]) => {
          const r = rows[ps[0].dataIndex];
          return `<b>${esc(r.deploy)}</b> · from ${dayLabel(r.first)}<br/>${r.n} requests<br/>writing median <b>${r.med.toFixed(1)}</b> tok/s<br/>p10–p90 ${r.p10.toFixed(1)}–${r.p90.toFixed(1)}`;
        },
      },
      grid: { left: 104, right: 24, top: 8, bottom: 26 },
      xAxis: { type: "value", name: "tok/s", ...axisCommon(pal) },
      yAxis: { type: "category", data: rows.map((r) => r.deploy), inverse: true, ...axisCommon(pal), axisLabel: { color: pal.ink2, fontFamily: "ui-monospace, monospace", fontSize: 10.5 } },
      series: [
        {
          type: "custom",
          encode: { x: [1, 2, 3], y: 0 },
          data: rows.map((r, i) => [i, r.p10, r.med, r.p90]),
          renderItem: (
            _p: unknown,
            api: { value: (i: number) => number; coord: (v: [number, number]) => [number, number]; size: (v: [number, number]) => [number, number] },
          ) => {
            const i = api.value(0);
            const [x10, y] = api.coord([api.value(1), i]);
            const [xm] = api.coord([api.value(2), i]);
            const [x90] = api.coord([api.value(3), i]);
            const h = Math.min(14, api.size([0, 1])[1] * 0.5);
            return {
              type: "group",
              children: [
                { type: "rect", shape: { x: x10, y: y - h / 2, width: Math.max(2, x90 - x10), height: h, r: 7 }, style: { fill: alpha(pal.decode, 0.22) } },
                { type: "circle", shape: { cx: xm, cy: y, r: 6 }, style: { fill: pal.decode, stroke: pal.surface, lineWidth: 2, shadowBlur: 10, shadowColor: alpha(pal.decode, 0.8) } },
              ],
            };
          },
        },
      ],
    };
  }, [s, pal]);

  // Wait for the first token: prompt pass seconds, log-spaced bins.
  const ttft = useMemo(() => {
    const xs = sortedOf(s.reqs.map((r) => r.prefillS));
    const edges = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, Infinity];
    const labels = ["<0.1s", "0.1–0.25", "0.25–0.5", "0.5–1", "1–2", "2–5", "5–10", "10–30", "30–60", "1–2m", "2–5m", ">5m"];
    const counts = labels.map((_, i) => xs.filter((x) => x >= edges[i] && x < edges[i + 1]).length);
    const p50 = quantile(xs, 0.5);
    const p90 = quantile(xs, 0.9);
    return {
      option: {
        tooltip: { ...tooltipBase(pal), trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: alpha(pal.ink, 0.05) } } },
        grid: { left: 40, right: 8, top: 10, bottom: 40 },
        xAxis: { type: "category", data: labels, ...axisCommon(pal), axisLabel: { color: pal.muted, fontSize: 10, rotate: 35 } },
        yAxis: { type: "value", ...axisCommon(pal) },
        series: [
          {
            name: "requests",
            type: "bar",
            data: counts,
            barMaxWidth: 22,
            itemStyle: {
              borderRadius: [4, 4, 0, 0],
              color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: pal.prefill }, { offset: 1, color: alpha(pal.prefill, 0.3) }] },
              shadowBlur: 10,
              shadowColor: alpha(pal.prefill, 0.5),
            },
          },
        ],
      },
      p50,
      p90,
    };
  }, [s, pal]);

  const acceptance = useMemo(() => {
    const bs = bucketize(s.reqs, dayEdges(s.from, s.to), s.busyReqs);
    return {
      tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`) },
      grid: { left: 40, right: 12, top: 10, bottom: 24 },
      xAxis: { type: "category", data: bs.map((b) => dayLabel(b.start, { month: "short", day: "numeric" })), boundaryGap: false, ...axisCommon(pal) },
      yAxis: { type: "value", min: 0, max: 1, ...axisCommon(pal), axisLabel: { color: pal.muted, fontSize: 10.5, formatter: (v: number) => `${v * 100}%` } },
      series: [
        {
          name: "drafts kept",
          type: "line",
          smooth: 0.3,
          data: bs.map((b) => b.acceptance),
          symbol: "circle",
          symbolSize: 8,
          itemStyle: { color: pal.prefill, borderColor: pal.surface, borderWidth: 2 },
          lineStyle: glowLine(pal.prefill),
          areaStyle: gradientFill(pal.prefill, 0.25),
          connectNulls: true,
        },
      ],
    };
  }, [s, pal]);

  const noGL = (
    <div className="grid h-full place-items-center text-center text-[13px] text-[var(--muted)]">
      {gl === "loading" ? "Loading 3D…" : "3D needs WebGL, which this window does not offer; the 2D charts below show the same data."}
    </div>
  );

  return (
    <div className="grid grid-cols-12 gap-4">
      <Card className="col-span-12 xl:col-span-7" title="Writing speed landscape" subtitle="every reply: context length × streams live × tok/s · drag to orbit, scroll to zoom">
        <div className="h-[380px]">{gl === "ready" ? <EChart option={landscape} className="h-full w-full" /> : noGL}</div>
      </Card>
      <Card className="col-span-12 xl:col-span-5" title="Skyline" subtitle="tokens written, day × hour (last 14 days)" delay={0.05}>
        <div className="h-[380px]">{gl === "ready" ? <EChart option={skyline} className="h-full w-full" /> : noGL}</div>
      </Card>

      <Card className="col-span-12 lg:col-span-6" title="Writing speed vs context" subtitle="tok/s per reply; darker = more streams sharing the pair" delay={0.08}>
        <EChart option={decScatter} className="h-64 w-full" />
      </Card>
      <Card className="col-span-12 lg:col-span-6" title="Reading speed vs prompt size" subtitle="tok/s over the prompt pass, per request" delay={0.1}>
        <EChart option={preScatter} className="h-64 w-full" />
      </Card>

      <Card className="col-span-12 xl:col-span-7" title="By deploy" subtitle="writing speed median (dot) and p10–p90 (bar), oldest first" delay={0.12}>
        <EChart option={byDeploy} className="w-full" style={{ height: Math.max(160, 40 + 30 * (byDeploy.yAxis.data as string[]).length) }} />
      </Card>
      <Card
        className="col-span-12 xl:col-span-5"
        title="Wait for the first token"
        subtitle={`prompt pass time · median ${ttft.p50 == null ? "—" : ttft.p50.toFixed(2) + " s"} · p90 ${ttft.p90 == null ? "—" : ttft.p90.toFixed(1) + " s"}`}
        delay={0.14}
      >
        <EChart option={ttft.option} className="h-56 w-full" />
      </Card>

      <Card className="col-span-12" title="Draft acceptance" subtitle="share of MTP drafts kept, per day" delay={0.16}>
        <EChart option={acceptance} className="h-44 w-full" />
      </Card>
    </div>
  );
}

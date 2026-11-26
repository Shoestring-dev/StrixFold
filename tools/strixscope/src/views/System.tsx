import { useMemo } from "react";

import { Card } from "../components/Card";
import { NoData } from "../components/NoData";
import { EChart } from "../components/EChart";
import { StatTile } from "../components/StatTile";
import { axisCommon, esc, glowLine, gradientFill, legendTop, timeAxis, tooltipBase, zoomSlider } from "../lib/chartkit";
import { compact, crashLoops, dayLabel, gib, timeLabel } from "../lib/stats";
import { alpha, type Palette } from "../lib/theme";
import { linkIfaces, nodeName, sarNote } from "../lib/topology";
import type { RunRow, Topology } from "../lib/types";
import type { Slice } from "./Overview";

const fmtDur = (us: number) => {
  const m = us / 60e6;
  if (m < 90) return `${m.toFixed(0)} min`;
  const h = m / 60;
  return h < 48 ? `${h.toFixed(1)} h` : `${(h / 24).toFixed(1)} days`;
};

export function System({ s, allRuns, topo, pal }: { s: Slice; allRuns: RunRow[]; topo: Topology | null; pal: Palette }) {
  // The dataset already respects the "test runs" toggle; filtering again would hide test runs when it is on.
  const runs = useMemo(
    () => allRuns.filter((r) => r.node === "rank0" && r.ts <= s.to && (r.endTs ?? s.to) >= s.from).sort((a, b) => a.ts - b.ts),
    [allRuns, s],
  );
  const loops = useMemo(() => crashLoops(runs), [runs]);
  // The newest run with no exit line is still serving; its end is just the last line pulled.
  const last = runs[runs.length - 1];
  const newest = useMemo(
    () => allRuns.filter((r) => r.node === "rank0").reduce<RunRow | null>((m, r) => (!m || r.ts > m.ts ? r : m), null),
    [allRuns],
  );
  const running = !!last && last.exitStatus == null && last === newest;
  const deploys = useMemo(() => [...new Set(runs.map((r) => r.deploy ?? "?"))], [runs]);
  const watchdog = s.events.filter((e) => e.kind.startsWith("wd_"));

  const gantt = useMemo(() => {
    const data = runs.map((r) => [deploys.indexOf(r.deploy ?? "?"), r.ts / 1000, (r.endTs ?? s.to) / 1000, r.exitStatus ?? -999, r.id]);
    return {
      tooltip: {
        ...tooltipBase(pal),
        formatter: (p: { value: number[]; seriesType: string; name?: string }) => {
          if (p.seriesType !== "custom") return esc(p.name ?? "");
          const r = runs.find((x) => x.id === p.value[4])!;
          return `<b>${esc(r.deploy)}</b><br/>${dayLabel(r.ts)} ${timeLabel(r.ts)} → ${r.endTs ? timeLabel(r.endTs) : "running"}<br/>up ${fmtDur((r.endTs ?? s.to) - r.ts)}${r.exitStatus != null ? ` · exit ${r.exitStatus}` : ""}${r.loadedS ? `<br/>loaded in ${r.loadedS.toFixed(0)} s` : ""}`;
        },
      },
      grid: { left: 104, right: 16, top: 8, bottom: 44 },
      dataZoom: zoomSlider(pal),
      xAxis: timeAxis(pal, s.from, s.to),
      yAxis: {
        type: "category",
        data: deploys.map((d) => d.slice(0, 12)),
        inverse: true,
        ...axisCommon(pal),
        splitLine: { show: true, lineStyle: { color: pal.grid } },
        axisLabel: { color: pal.ink2, fontFamily: "ui-monospace, monospace", fontSize: 10.5 },
      },
      series: [
        {
          type: "custom",
          clip: true,
          encode: { x: [1, 2], y: 0 },
          data,
          renderItem: (
            _p: unknown,
            api: { value: (i: number) => number; coord: (v: [number, number]) => [number, number]; size: (v: [number, number]) => [number, number] },
          ) => {
            const lane = api.value(0);
            const [x0, y] = api.coord([api.value(1), lane]);
            const [x1] = api.coord([api.value(2), lane]);
            const h = Math.min(16, api.size([0, 1])[1] * 0.55);
            const crashed = api.value(3) !== 0 && api.value(3) !== -999;
            const c = crashed ? pal.critical : pal.rank0;
            return {
              type: "rect",
              shape: { x: x0, y: y - h / 2, width: Math.max(2, x1 - x0 - 1), height: h, r: 4 },
              style: { fill: alpha(c, crashed ? 0.85 : 0.7), shadowBlur: 8, shadowColor: alpha(c, 0.6) },
            };
          },
        },
        {
          type: "scatter",
          name: "watchdog",
          symbol: "triangle",
          symbolSize: 11,
          itemStyle: { color: pal.warning },
          data: watchdog.filter((e) => e.kind === "wd_restart").map((e) => ({ value: [e.ts / 1000, 0], name: e.text ?? "watchdog restart" })),
        },
      ],
    };
  }, [runs, deploys, s, pal, watchdog]);

  const memOption = useMemo(() => {
    const ser = (["rank0", "rank1"] as const).map((n) => {
      const color = n === "rank0" ? pal.rank0 : pal.rank1;
      return {
        name: nodeName(topo, n),
        type: "line",
        showSymbol: false,
        smooth: 0.2,
        data: s.sar.filter((x) => x.node === n).map((x) => [x.ts / 1000, gib(x.memAvail)]),
        lineStyle: glowLine(color, 2),
        itemStyle: { color },
        areaStyle: gradientFill(color, 0.12),
      };
    });
    (ser[0] as Record<string, unknown>).markLine = {
      silent: true,
      symbol: "none",
      lineStyle: { color: pal.warning, type: "solid", width: 1 },
      label: { color: pal.muted, formatter: "8 GiB floor", fontSize: 10, position: "insideStartTop" },
      data: [{ yAxis: 8 }],
    };
    return {
      tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => (v == null ? "—" : `${v.toFixed(1)} GiB`) },
      legend: legendTop(pal),
      grid: { left: 40, right: 16, top: 30, bottom: 44 },
      dataZoom: zoomSlider(pal),
      xAxis: timeAxis(pal, s.from, s.to),
      yAxis: { type: "value", name: "GiB free", ...axisCommon(pal), min: 0 },
      series: ser,
    };
  }, [s, topo, pal]);

  const cpuOption = useMemo(
    () => ({
      tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => (v == null ? "—" : `${v.toFixed(1)}%`) },
      legend: legendTop(pal),
      grid: { left: 40, right: 16, top: 30, bottom: 24 },
      xAxis: timeAxis(pal, s.from, s.to),
      yAxis: { type: "value", name: "% busy", ...axisCommon(pal), min: 0 },
      series: (["rank0", "rank1"] as const).map((n) => {
        const color = n === "rank0" ? pal.rank0 : pal.rank1;
        return {
          name: nodeName(topo, n),
          type: "line",
          showSymbol: false,
          data: s.sar.filter((x) => x.node === n).map((x) => [x.ts / 1000, 100 - (x.cpuIdle ?? 100)]),
          lineStyle: glowLine(color, 1.5),
          itemStyle: { color },
        };
      }),
    }),
    [s, topo, pal],
  );

  const ifaces = useMemo(() => linkIfaces(topo, s.net), [topo, s.net]);
  const linkRows = useMemo(() => s.net.filter((x) => x.node === "rank0" && ifaces.includes(x.iface)).length, [s.net, ifaces]);
  const netOption = useMemo(() => {
    const colors = [pal.prefill, pal.decode, pal.cache, pal.memory, pal.rank1, pal.rank0];
    return {
      tooltip: { ...tooltipBase(pal), trigger: "axis", valueFormatter: (v: number) => (v == null ? "—" : `${v.toFixed(1)} MiB/s`) },
      legend: legendTop(pal),
      grid: { left: 40, right: 16, top: 30, bottom: 24 },
      xAxis: timeAxis(pal, s.from, s.to),
      yAxis: { type: "value", name: "MiB/s", ...axisCommon(pal), min: 0 },
      series: ifaces.map((i, k) => ({
        name: i,
        type: "line",
        showSymbol: false,
        data: s.net.filter((x) => x.node === "rank0" && x.iface === i).map((x) => [x.ts / 1000, ((x.rxKb ?? 0) + (x.txKb ?? 0)) / 1024]),
        lineStyle: glowLine(colors[k % colors.length], 1.5),
        itemStyle: { color: colors[k % colors.length] },
        areaStyle: gradientFill(colors[k % colors.length], 0.15),
      })),
    };
  }, [s, ifaces, pal]);
  const sarEmpty = sarNote(topo, s.sar.length);
  const netEmpty = ifaces.length === 0 ? "No link NICs known; name them under Pair → Overrides" : sarNote(topo, linkRows);

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile
          pal={pal}
          label="Up since the last launch"
          value={running ? (s.to - last.ts) / 3600e6 : null}
          format={(n) => (n == null ? "stopped" : n < 48 ? `${n.toFixed(1)} h` : `${(n / 24).toFixed(1)} d`)}
          unit={running ? `on ${last.deploy?.slice(0, 7)}` : undefined}
          color={pal.rank0}
        />
        <StatTile pal={pal} label="Launches in the period" value={runs.filter((r) => r.ts >= s.from).length} format={(n) => compact(n)} color={pal.rank0} />
        <StatTile pal={pal} label="Crash loops" value={loops.length} format={(n) => compact(n)} color={pal.rank1} />
        <StatTile pal={pal} label="Watchdog restarts" value={watchdog.filter((e) => e.kind === "wd_restart").length} format={(n) => compact(n)} color={pal.memory} />
      </div>

      <Card
        className="col-span-12"
        title="Runs by deploy"
        subtitle="each launch of the pair; red = exited with an error · ▲ watchdog restart"
        right={
          loops.length ? (
            <div className="max-w-[50%] text-right text-[12px]" style={{ color: pal.ink2 }}>
              {loops.slice(-3).map((l) => (
                <div key={l.from}>
                  <span style={{ color: pal.critical }}>●</span> crash loop {dayLabel(l.from, { month: "short", day: "numeric" })} {timeLabel(l.from)}–{timeLabel(l.to)} · {l.runs} launches
                </div>
              ))}
            </div>
          ) : undefined
        }
      >
        <EChart option={gantt} className="w-full" style={{ height: Math.max(170, 70 + deploys.length * 26) }} />
      </Card>

      <Card className="col-span-12" title="Memory free" subtitle="MemAvailable per node (sar keeps 7 days on the nodes; older samples stay here once pulled)" delay={0.05}>
        {sarEmpty ? <NoData note={sarEmpty} className="h-64 w-full" /> : <EChart option={memOption} className="h-64 w-full" />}
      </Card>
      <Card className="col-span-12 lg:col-span-6" title="CPU" subtitle="per node, 10-minute averages" delay={0.08}>
        {sarEmpty ? <NoData note={sarEmpty} className="h-52 w-full" /> : <EChart option={cpuOption} className="h-52 w-full" />}
      </Card>
      <Card className="col-span-12 lg:col-span-6" title="Links between the nodes" subtitle={`${nodeName(topo, "rank0")}, both directions`} delay={0.1}>
        {netEmpty ? <NoData note={netEmpty} className="h-52 w-full" /> : <EChart option={netOption} className="h-52 w-full" />}
      </Card>

      {watchdog.length > 0 && (
        <Card className="col-span-12" title="Watchdog" subtitle="what it noticed" delay={0.12}>
          <div className="scroll-area max-h-48 space-y-1 text-[12.5px]">
            {watchdog
              .slice()
              .reverse()
              .map((e) => (
                <div key={`${e.ts}${e.kind}`} className="flex gap-3">
                  <span className="tnum w-32 shrink-0 text-[var(--muted)]">
                    {dayLabel(e.ts, { month: "short", day: "numeric" })} {timeLabel(e.ts)}
                  </span>
                  <span style={{ color: e.kind === "wd_restart" ? pal.warning : pal.ink2 }}>{e.text}</span>
                </div>
              ))}
          </div>
        </Card>
      )}
    </div>
  );
}

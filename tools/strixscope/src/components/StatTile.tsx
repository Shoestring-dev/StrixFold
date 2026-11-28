import { useMemo } from "react";

import { EChart } from "./EChart";
import { CountUp } from "./CountUp";
import { alpha, type Palette } from "../lib/theme";

interface Props {
  label: string;
  value: number | null;
  format: (n: number | null) => string;
  unit?: string;
  color: string;
  spark?: (number | null)[];
  /** Change vs the previous period, as a fraction; sign × goodWhenUp picks the color. */
  delta?: number | null;
  goodWhenUp?: boolean;
  pal: Palette;
}

export function StatTile({ label, value, format, unit, color, spark, delta, goodWhenUp = true, pal }: Props) {
  const option = useMemo(
    () => ({
      grid: { left: 0, right: 0, top: 4, bottom: 0 },
      xAxis: { type: "category", show: false, boundaryGap: false },
      yAxis: { type: "value", show: false, scale: true },
      tooltip: { show: false },
      series: [
        {
          type: "line",
          data: spark ?? [],
          smooth: 0.35,
          showSymbol: false,
          connectNulls: true,
          lineStyle: { width: 2, color, shadowColor: alpha(color, 0.6), shadowBlur: 10 },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: alpha(color, 0.28) },
                { offset: 1, color: alpha(color, 0) },
              ],
            },
          },
        },
      ],
    }),
    [spark, color],
  );
  const up = (delta ?? 0) > 0;
  const good = up === goodWhenUp;
  return (
    <div className="card flex flex-col justify-between overflow-hidden p-4">
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full" style={{ background: color, boxShadow: `0 0 10px ${color}` }} />
        <span className="label">{label}</span>
      </div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className="text-[28px] font-semibold leading-none tracking-tight">
          <CountUp value={value} format={format} />
        </span>
        {unit && <span className="text-[13px] text-[var(--muted)]">{unit}</span>}
        {delta != null && Number.isFinite(delta) && Math.abs(delta) >= 0.005 && (
          <span
            className="ml-auto rounded-full px-2 py-0.5 text-[11.5px] font-medium tnum"
            style={{
              color: good ? pal.good : pal.serious,
              background: alpha(good ? pal.good : pal.serious, 0.12),
            }}
          >
            {up ? "▲" : "▼"} {Math.abs(delta * 100).toFixed(0)}%
          </span>
        )}
      </div>
      {spark && <EChart option={option} className="-mx-1 mt-2 h-11" />}
    </div>
  );
}

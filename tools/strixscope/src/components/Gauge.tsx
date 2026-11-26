import { useMemo } from "react";

import { EChart } from "./EChart";
import { alpha, type Palette } from "../lib/theme";

interface Props {
  label: string;
  /** 0..1, or null when there is no data. */
  value: number | null;
  color: string;
  caption?: string;
  pal: Palette;
}

/** A 270° ring with a glowing progress arc and a fine tick scale. */
export function Gauge({ label, value, color, caption, pal }: Props) {
  const option = useMemo(() => {
    const v = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100;
    return {
      series: [
        {
          // Tick scale, just outside the arc.
          type: "gauge",
          startAngle: 225,
          endAngle: -45,
          radius: "96%",
          min: 0,
          max: 100,
          splitNumber: 10,
          axisLine: { lineStyle: { width: 0 } },
          splitLine: { distance: 2, length: 6, lineStyle: { color: pal.axis, width: 1.5 } },
          axisTick: { distance: 2, length: 3, splitNumber: 4, lineStyle: { color: pal.grid, width: 1 } },
          axisLabel: { show: false },
          pointer: { show: false },
          detail: { show: false },
          silent: true,
        },
        {
          type: "gauge",
          startAngle: 225,
          endAngle: -45,
          radius: "84%",
          min: 0,
          max: 100,
          progress: {
            show: true,
            roundCap: true,
            width: 14,
            itemStyle: {
              color: {
                type: "linear",
                x: 0,
                y: 1,
                x2: 1,
                y2: 0,
                colorStops: [
                  { offset: 0, color: alpha(color, 0.55) },
                  { offset: 1, color },
                ],
              },
              shadowColor: alpha(color, 0.75),
              shadowBlur: 18,
            },
          },
          axisLine: { roundCap: true, lineStyle: { width: 14, color: [[1, alpha(color, 0.1)]] } },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { show: false },
          pointer: { show: false },
          anchor: { show: false },
          title: { show: false },
          detail: {
            valueAnimation: true,
            offsetCenter: [0, "4%"],
            fontSize: 30,
            fontWeight: 600,
            color: pal.ink,
            formatter: () => (value == null ? "—" : `${Math.round(v)}%`),
          },
          data: [{ value: v }],
        },
      ],
    };
  }, [value, color, pal]);
  return (
    <div className="flex min-w-0 flex-col items-center">
      <EChart option={option} replace={false} className="h-40 w-full max-w-[200px]" />
      <div className="-mt-6 text-center">
        <div className="text-[13px] font-medium">{label}</div>
        {caption && <div className="label mt-0.5">{caption}</div>}
      </div>
    </div>
  );
}

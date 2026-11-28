/** Shared ECharts pieces so every view has the same chrome. */
import { alpha, type Palette } from "./theme";
import { compact, timeLabel, dayLabel } from "./stats";

/** HTML-escape log-derived text before a tooltip formatter returns it: ECharts assigns it with `innerHTML`. */
export function esc(s: unknown): string {
  return String(s ?? "").replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

export function tooltipBase(pal: Palette) {
  return {
    backgroundColor: pal.surface,
    borderColor: pal.axis,
    textStyle: { color: pal.ink, fontSize: 12 },
    extraCssText: "border-radius:12px;box-shadow:0 12px 32px -12px rgba(0,0,0,.6);",
  };
}

export function axisCommon(pal: Palette) {
  return {
    axisLine: { lineStyle: { color: pal.axis } },
    axisTick: { show: false },
    axisLabel: { color: pal.muted, fontSize: 10.5 },
    splitLine: { lineStyle: { color: pal.grid } },
    nameTextStyle: { color: pal.muted, fontSize: 10.5 },
  };
}

/** A time x-axis that labels hours within a day and dates across days. */
export function timeAxis(pal: Palette, fromUs: number, toUs: number) {
  const multiDay = toUs - fromUs > 36 * 3600e6;
  return {
    type: "time",
    min: fromUs / 1000,
    max: toUs / 1000,
    ...axisCommon(pal),
    splitLine: { show: false },
    axisLabel: {
      color: pal.muted,
      fontSize: 10.5,
      hideOverlap: true,
      formatter: (v: number) => (multiDay ? dayLabel(v * 1000, { month: "short", day: "numeric" }) : timeLabel(v * 1000)),
    },
  };
}

export function valueAxis(pal: Palette, name?: string, extra: Record<string, unknown> = {}) {
  return {
    type: "value",
    name,
    ...axisCommon(pal),
    axisLabel: { color: pal.muted, fontSize: 10.5, formatter: (v: number) => compact(v, 0) },
    ...extra,
  };
}

export function glowLine(color: string, width = 2.5) {
  return { width, color, shadowColor: alpha(color, 0.65), shadowBlur: 12 };
}

export function gradientFill(color: string, top = 0.3) {
  return {
    color: {
      type: "linear",
      x: 0,
      y: 0,
      x2: 0,
      y2: 1,
      colorStops: [
        { offset: 0, color: alpha(color, top) },
        { offset: 1, color: alpha(color, 0) },
      ],
    },
  };
}

export function barFill(color: string) {
  return {
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
    shadowColor: alpha(color, 0.5),
    shadowBlur: 10,
  };
}

export function legendTop(pal: Palette) {
  return { top: 0, right: 0, textStyle: { color: pal.ink2, fontSize: 11.5 }, icon: "circle", itemWidth: 8, itemHeight: 8 };
}

export function zoomSlider(pal: Palette) {
  return [
    { type: "inside", filterMode: "weakFilter" },
    {
      type: "slider",
      height: 18,
      bottom: 4,
      filterMode: "weakFilter",
      borderColor: pal.axis,
      fillerColor: alpha(pal.prefill, 0.15),
      handleStyle: { color: pal.prefill, borderColor: pal.prefill },
      moveHandleStyle: { color: pal.prefill },
      textStyle: { color: pal.muted, fontSize: 10 },
      dataBackground: { lineStyle: { color: pal.axis }, areaStyle: { color: alpha(pal.prefill, 0.08) } },
      showDetail: false,
    },
  ];
}

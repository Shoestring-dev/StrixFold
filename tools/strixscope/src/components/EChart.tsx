import { useEffect, useRef, type CSSProperties } from "react";
import { useReducedMotion } from "motion/react";

import { echarts, type EChartsOption } from "../lib/echarts";

interface Props {
  option: EChartsOption;
  className?: string;
  style?: CSSProperties;
  /** Replace instead of merge (needed when series come and go). */
  replace?: boolean;
  onClick?: (params: echarts.ECElementEvent) => void;
  /** Accessible name; charts are canvases, so this is their only text form. */
  label?: string;
}

/** One ECharts instance per mount; option changes animate in place. */
export function EChart({ option, className, style, replace = true, onClick, label }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const click = useRef(onClick);
  click.current = onClick;
  const reduced = useReducedMotion();

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, undefined, { renderer: "canvas" });
    chart.current = c;
    c.on("click", (p) => click.current?.(p as echarts.ECElementEvent));
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    const base = reduced ? { animation: false } : { animationDuration: 700, animationEasing: "cubicOut" as const };
    try {
      chart.current?.setOption({ ...base, ...option }, { notMerge: replace, lazyUpdate: true });
    } catch (e) {
      // A chart that cannot render (say, a 3D view without eval) must not take the whole window down.
      console.error("chart option failed", e);
    }
  }, [option, replace, reduced]);

  return (
    <div
      ref={el}
      className={className ?? "h-64 w-full"}
      style={style}
      role="img"
      aria-label={label ?? "chart"}
    />
  );
}

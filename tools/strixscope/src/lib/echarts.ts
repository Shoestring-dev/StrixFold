/** Tree-shaken ECharts; the CSP needs 'unsafe-eval' for echarts-gl's `new Function`, while inline handlers stay blocked. */
import * as echarts from "echarts/core";
import { BarChart, CustomChart, GaugeChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  AxisPointerComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart,
  CustomChart,
  GaugeChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  AxisPointerComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

let gl: Promise<void> | null = null;
/** echarts-gl (3D) loads on first use: it is large and only one view needs it. */
export function loadGL(): Promise<void> {
  gl ??= Promise.all([
    import("echarts-gl/lib/component/grid3D"),
    import("echarts-gl/lib/chart/bar3D"),
    import("echarts-gl/lib/chart/scatter3D"),
  ]).then(() => undefined);
  return gl;
}

export function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

export { echarts };
export type EChartsOption = echarts.EChartsCoreOption;

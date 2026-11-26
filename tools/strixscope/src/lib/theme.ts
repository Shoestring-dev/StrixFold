/** Concept colors, fixed across every view; scatter plots use at most three plus a gray "other" to stay distinguishable. */
export type Mode = "dark" | "light";

const SERIES = {
  dark: ["#508dff", "#e86333", "#00ae7e", "#be8800", "#e65695", "#8e7ef0"],
  light: ["#3e7aeb", "#d3501c", "#009b6c", "#ab7500", "#d04283", "#7d6cdb"],
} as const;

export interface Palette {
  mode: Mode;
  prefill: string; // tokens in / reading
  decode: string; // tokens out / writing
  cache: string;
  memory: string;
  rank1: string; // rank 1 (the build box)
  rank0: string; // rank 0 (serves HTTP)
  other: string;
  surface: string;
  page: string;
  grid: string;
  axis: string;
  ink: string;
  ink2: string;
  muted: string;
  good: string;
  warning: string;
  serious: string;
  critical: string;
  /** Sequential ramp (one hue: blue), surface → bright. */
  ramp: string[];
}

export function palette(mode: Mode): Palette {
  const s = SERIES[mode];
  const dark = mode === "dark";
  return {
    mode,
    prefill: s[0],
    decode: s[1],
    cache: s[2],
    memory: s[3],
    rank1: s[4],
    rank0: s[5],
    other: dark ? "#5b6272" : "#9aa1ad",
    surface: dark ? "#10141c" : "#ffffff",
    page: dark ? "#07090d" : "#f3f5f8",
    grid: dark ? "#1d2330" : "#e6e9ef",
    axis: dark ? "#2a3140" : "#c9ced8",
    ink: dark ? "#eef1f6" : "#0d1117",
    ink2: dark ? "#b9c1cf" : "#4a5260",
    muted: dark ? "#7d8696" : "#7a8291",
    good: "#0ca30c",
    warning: "#fab219",
    serious: "#ec835a",
    critical: "#d03b3b",
    ramp: dark
      ? ["#141b28", "#16294a", "#1a3a70", "#1f4f9b", "#2a6bd0", "#508dff", "#8db4ff", "#cfe0ff"]
      : ["#f2f6fd", "#d6e4fb", "#aecaf6", "#7eaaf0", "#4f8ae8", "#2f6fd6", "#1f55ad", "#143c80"],
  };
}

export function systemMode(): Mode {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

/** Hex + alpha → rgba() for glows and washes. */
export function alpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

import { motion, useReducedMotion } from "motion/react";

import { alpha, type Palette } from "../lib/theme";

export interface NodeInfo {
  name: string;
  role: string;
  memAvailGiB: number | null;
  memTotalGiB: number | null;
  cpu: number | null; // 0..100 busy
  color: string;
}

interface Props {
  left: NodeInfo;
  right: NodeInfo;
  /** Per link, KiB/s both directions (see `linkRate`): mean while busy and busiest sar interval. */
  links: { name: string; busy: number | null; peak: number | null }[];
  pal: Palette;
}

function fmtRate(kbps: number | null) {
  if (kbps == null) return "—";
  if (kbps >= 1024 * 1024) return `${(kbps / 1024 / 1024).toFixed(1)} GiB/s`;
  if (kbps >= 1024) return `${(kbps / 1024).toFixed(1)} MiB/s`;
  return `${kbps.toFixed(0)} KiB/s`;
}

function linkTitle(l: Props["links"][number]) {
  if (l.peak == null) return `${l.name}: no sar samples in this period`;
  if (l.busy == null) return `${l.name}: idle in every sar interval of this period`;
  return `${l.name}: ${fmtRate(l.busy)} averaged over the busy sar intervals, ${fmtRate(l.peak)} in the busiest (both directions; bursts inside an interval are averaged out)`;
}

/** Two node chips with memory rings joined by up to four links; pulses scale with throughput and alternate direction. */
export function PairDiagram({ left, right, links, pal }: Props) {
  const reduced = useReducedMotion();
  const W = 560;
  const H = 250;
  const nodeX = [110, W - 110];
  const cy = 118;
  const shown = links.slice(0, 4);
  const gap = shown.length > 2 ? 36 : 52;
  const linkY = (i: number) => cy + (i - (shown.length - 1) / 2) * gap;
  const colors = [pal.prefill, pal.decode, pal.cache, pal.memory];
  const maxK = Math.max(1, ...shown.map((l) => l.busy ?? 0));
  const x0 = nodeX[0] + 62;
  const x1 = nodeX[1] - 62;
  const label = `${left.name} and ${right.name}, ${shown.length ? `joined by ${shown.length} link${shown.length > 1 ? "s" : ""}` : "links unknown"}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" role="img" aria-label={label}>
      <defs>
        {[left, right].map((n, i) => (
          <radialGradient key={i} id={`glow${i}`} cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={alpha(n.color, 0.35)} />
            <stop offset="100%" stopColor={alpha(n.color, 0)} />
          </radialGradient>
        ))}
        <linearGradient id="linkgrad" x1="0" x2="1">
          <stop offset="0%" stopColor={left.color} />
          <stop offset="100%" stopColor={right.color} />
        </linearGradient>
      </defs>
      {shown.length === 0 && (
        <g>
          <line x1={x0} y1={cy} x2={x1} y2={cy} stroke={pal.axis} strokeWidth={2} strokeDasharray="4 6" strokeLinecap="round" />
          <text x={W / 2} y={cy - 9} textAnchor="middle" fontSize={11} fill={pal.muted}>
            no link NICs known
          </text>
        </g>
      )}
      {shown.map((l, i) => {
        const y = linkY(i);
        const share = (l.busy ?? 0) / maxK;
        const dur = 3.2 - 2.2 * share; // busier link → faster pulses
        const c = colors[i];
        const fwd = i % 2 === 0;
        return (
          <g key={l.name}>
            <line x1={x0} y1={y} x2={x1} y2={y} stroke={pal.axis} strokeWidth={6} strokeLinecap="round" />
            <line x1={x0} y1={y} x2={x1} y2={y} stroke="url(#linkgrad)" strokeOpacity={0.35 + 0.5 * share} strokeWidth={2} strokeLinecap="round" />
            {!reduced &&
              [0, 1, 2].map((k) => (
                <motion.circle
                  key={k}
                  r={3.5}
                  cy={y}
                  fill={c}
                  style={{ filter: `drop-shadow(0 0 6px ${c})` }}
                  initial={{ cx: fwd ? x0 : x1, opacity: 0 }}
                  animate={{ cx: fwd ? [x0, x1] : [x1, x0], opacity: [0, 1, 1, 0] }}
                  transition={{ duration: dur, repeat: Infinity, delay: (k * dur) / 3, ease: "linear" }}
                />
              ))}
            <text x={W / 2} y={y - 9} textAnchor="middle" fontSize={11} fill={pal.muted}>
              <title>{linkTitle(l)}</title>
              {l.name} · {l.peak == null ? "—" : l.busy == null ? "idle" : `${fmtRate(l.busy)} · peak ${fmtRate(l.peak)}`}
            </text>
          </g>
        );
      })}
      {[left, right].map((n, i) => {
        const x = nodeX[i];
        const r = 48;
        const used = n.memAvailGiB != null && n.memTotalGiB ? 1 - n.memAvailGiB / n.memTotalGiB : null;
        const circ = 2 * Math.PI * r;
        return (
          <g key={i}>
            <circle cx={x} cy={cy} r={92} fill={`url(#glow${i})`} />
            <circle cx={x} cy={cy} r={r + 10} fill={pal.surface} stroke={pal.axis} />
            <circle cx={x} cy={cy} r={r} fill="none" stroke={alpha(pal.memory, 0.14)} strokeWidth={7} />
            {used != null && (
              <motion.circle
                cx={x}
                cy={cy}
                r={r}
                fill="none"
                stroke={pal.memory}
                strokeWidth={7}
                strokeLinecap="round"
                transform={`rotate(-90 ${x} ${cy})`}
                strokeDasharray={circ}
                initial={{ strokeDashoffset: circ }}
                animate={{ strokeDashoffset: circ * (1 - used) }}
                transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
                style={{ filter: `drop-shadow(0 0 5px ${alpha(pal.memory, 0.8)})` }}
              />
            )}
            <text x={x} y={cy - 4} textAnchor="middle" fontSize={20} fontWeight={600} fill={pal.ink}>
              {n.memAvailGiB == null ? "—" : n.memAvailGiB.toFixed(1)}
            </text>
            <text x={x} y={cy + 13} textAnchor="middle" fontSize={10.5} fill={pal.muted}>
              GiB free
            </text>

            <circle cx={x - 34} cy={cy + 82} r={4.5} fill={n.color} />
            <text x={x - 24} y={cy + 86} fontSize={13} fontWeight={600} fill={pal.ink}>
              {n.name}
            </text>
            <text x={x} y={cy + 104} textAnchor="middle" fontSize={11} fill={pal.muted}>
              {n.role}
              {used != null && n.memTotalGiB != null ? ` · ${Math.round(used * 100)}% of ${Math.round(n.memTotalGiB)} GiB used` : ""}
              {n.cpu != null ? ` · CPU ${n.cpu.toFixed(0)}%` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

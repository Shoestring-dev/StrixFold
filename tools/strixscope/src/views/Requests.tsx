import { useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowDown, ArrowUp } from "lucide-react";

import { Card } from "../components/Card";
import { compact, dayLabel, timeLabel, type Req } from "../lib/stats";
import { alpha, type Palette } from "../lib/theme";
import type { Slice } from "./Overview";

type Key = "tEnd" | "client" | "prompt" | "cached" | "generated" | "prefillS" | "prefillTps" | "decodeTps" | "acceptance" | "others" | "finish";

const COLS: { key: Key; label: string; w: string; num?: boolean }[] = [
  { key: "tEnd", label: "Finished", w: "9.5rem" },
  { key: "client", label: "Client", w: "8.5rem" },
  { key: "prompt", label: "Prompt", w: "1fr", num: true },
  { key: "cached", label: "Cached", w: "5rem", num: true },
  { key: "generated", label: "Written", w: "1fr", num: true },
  { key: "prefillS", label: "1st token", w: "5.5rem", num: true },
  { key: "prefillTps", label: "Read tok/s", w: "6rem", num: true },
  { key: "decodeTps", label: "Write tok/s", w: "6rem", num: true },
  { key: "acceptance", label: "Drafts", w: "4.5rem", num: true },
  { key: "others", label: "Streams", w: "4.5rem", num: true },
  { key: "finish", label: "Finish", w: "6.5rem" },
];
const GRID = COLS.map((c) => c.w).join(" ");

/** Up to six named parts, the rest folded into "other". */
function topParts(values: (string | null)[], n = 5) {
  const counts = new Map<string, number>();
  for (const v of values) counts.set(v ?? "unknown", (counts.get(v ?? "unknown") ?? 0) + 1);
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const head = sorted.slice(0, n);
  const rest = sorted.slice(n).reduce((a, [, c]) => a + c, 0);
  return rest ? [...head, ["other", rest] as [string, number]] : head;
}

export function Requests({ s, probes, pal }: { s: Slice; probes: Req[]; pal: Palette }) {
  const [sort, setSort] = useState<{ key: Key; desc: boolean }>({ key: "tEnd", desc: true });
  const [client, setClient] = useState<string | null>(null);
  const [finish, setFinish] = useState<string | null>(null);
  const [showProbes, setShowProbes] = useState(false);

  const base = useMemo(() => (showProbes ? [...s.reqs, ...probes.filter((r) => r.tEnd >= s.from && r.tEnd < s.to)] : s.reqs), [s, probes, showProbes]);
  const clients = useMemo(() => topParts(base.map((r) => r.client)), [base]);
  const finishes = useMemo(() => topParts(base.map((r) => r.finish), 4), [base]);
  const rows = useMemo(() => {
    const named = (parts: [string, number][]) => new Set(parts.map(([k]) => k).filter((k) => k !== "other"));
    const match = (sel: string | null, value: string, parts: [string, number][]) =>
      !sel || (sel === "other" ? !named(parts).has(value) : value === sel);
    const f = base.filter(
      (r) => match(client, r.client ?? "unknown", clients) && match(finish, r.finish ?? "unknown", finishes),
    );
    const k = sort.key;
    const dir = sort.desc ? -1 : 1;
    return f.sort((a, b) => {
      const x = a[k] as number | string | null;
      const y = b[k] as number | string | null;
      // Empty values stay last in both directions; everything else sorts.
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * dir;
    });
  }, [base, client, finish, sort, clients, finishes]);

  const maxPrompt = rows.reduce((m, r) => Math.max(m, r.prompt), 1);
  const maxGen = rows.reduce((m, r) => Math.max(m, r.generated), 1);
  const partColors = [pal.rank0, pal.rank1, pal.prefill, pal.cache, pal.memory, pal.other];

  const parent = useRef<HTMLDivElement>(null);
  const virt = useVirtualizer({ count: rows.length, getScrollElement: () => parent.current, estimateSize: () => 34, overscan: 12 });

  return (
    <div className="grid grid-cols-12 gap-4">
      <Card className="col-span-12 lg:col-span-7" title="Clients" subtitle="share of requests · click a part to filter">
        <Segmented parts={clients} colors={partColors} pal={pal} selected={client} onSelect={setClient} />
      </Card>
      <Card className="col-span-12 lg:col-span-5" title="How replies ended" subtitle="click a part to filter" delay={0.05}>
        <Segmented parts={finishes} colors={[pal.cache, pal.memory, pal.rank1, pal.other, pal.prefill]} pal={pal} selected={finish} onSelect={setFinish} />
      </Card>

      <Card
        className="col-span-12"
        title={`${compact(rows.length)} requests`}
        subtitle="click a column to sort"
        delay={0.08}
        right={
          <label className="flex cursor-pointer items-center gap-2 text-[12.5px] text-[var(--ink-2)]">
            <input type="checkbox" checked={showProbes} onChange={(e) => setShowProbes(e.target.checked)} className="accent-[var(--prefill)]" />
            show health probes
          </label>
        }
      >
        <div className="grid border-b border-[var(--line)] pb-2 text-[11.5px] text-[var(--muted)]" style={{ gridTemplateColumns: GRID }}>
          {COLS.map((c) => (
            <button
              key={c.key}
              className={`flex items-center gap-1 px-2 hover:text-[var(--ink)] ${c.num ? "justify-end" : ""}`}
              onClick={() => setSort((p) => ({ key: c.key, desc: p.key === c.key ? !p.desc : true }))}
            >
              {c.label}
              {sort.key === c.key && (sort.desc ? <ArrowDown size={11} /> : <ArrowUp size={11} />)}
            </button>
          ))}
        </div>
        <div ref={parent} className="scroll-area h-[56vh]">
          <div style={{ height: virt.getTotalSize(), position: "relative" }}>
            {virt.getVirtualItems().map((v) => {
              const r = rows[v.index];
              return (
                <div
                  key={v.key}
                  className="tnum absolute left-0 right-0 grid items-center border-b border-[var(--line)] text-[12.5px] hover:bg-[var(--line)]"
                  style={{ gridTemplateColumns: GRID, height: v.size, transform: `translateY(${v.start}px)` }}
                >
                  <span className="px-2 text-[var(--ink-2)]">
                    {dayLabel(r.tEnd, { month: "short", day: "numeric" })} {timeLabel(r.tEnd)}
                  </span>
                  <span className="truncate px-2">{r.client ?? "—"}</span>
                  <Bar value={r.prompt} max={maxPrompt} color={pal.prefill} />
                  <span className="px-2 text-right text-[var(--ink-2)]">{compact(r.cached)}</span>
                  <Bar value={r.generated} max={maxGen} color={pal.decode} />
                  <span className="px-2 text-right">{r.prefillS == null ? "—" : r.prefillS < 10 ? r.prefillS.toFixed(2) + "s" : r.prefillS.toFixed(0) + "s"}</span>
                  <span className="px-2 text-right">{r.prefillTps == null ? "—" : compact(r.prefillTps)}</span>
                  <span className="px-2 text-right">{r.decodeTps == null ? "—" : r.decodeTps.toFixed(1)}</span>
                  <span className="px-2 text-right">{r.acceptance == null ? "—" : `${Math.round(r.acceptance * 100)}%`}</span>
                  <span className="px-2 text-right">{r.others == null ? "—" : r.others + 1}</span>
                  <span className="px-2">
                    <FinishBadge finish={r.finish} pal={pal} />
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </Card>
    </div>
  );
}

/** Log-scaled inline bar with the value at its end. */
function Bar({ value, max, color }: { value: number; max: number; color: string }) {
  const w = value > 0 ? (Math.log10(value + 1) / Math.log10(max + 1)) * 100 : 0;
  return (
    <span className="flex items-center gap-2 px-2">
      <span className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: alpha(color, 0.12) }}>
        <span className="block h-full rounded-full" style={{ width: `${w}%`, background: color, boxShadow: `0 0 8px ${alpha(color, 0.7)}` }} />
      </span>
      <span className="w-12 text-right">{compact(value)}</span>
    </span>
  );
}

function FinishBadge({ finish, pal }: { finish: string | null; pal: Palette }) {
  const c = finish === "stop" ? pal.cache : finish === "length" ? pal.memory : finish === "client-left" ? pal.rank1 : pal.other;
  return (
    <span className="rounded-full px-2 py-0.5 text-[11px]" style={{ color: pal.ink, background: alpha(c, 0.22), border: `1px solid ${alpha(c, 0.5)}` }}>
      {finish ?? "?"}
    </span>
  );
}

/** White or near-black text, by the fill's luminance. */
function textOn(hex: string): string {
  if (!hex.startsWith("#")) return "#fff";
  const n = parseInt(hex.slice(1, 7), 16);
  const lum = (0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
  return lum > 0.55 ? "#0b0b0b" : "#fff";
}

/** The tall segmented bar: one whole split into ≤ 6 parts, share inside. */
function Segmented({
  parts,
  colors,
  pal,
  selected,
  onSelect,
}: {
  parts: [string, number][];
  colors: string[];
  pal: Palette;
  selected: string | null;
  onSelect: (k: string | null) => void;
}) {
  const total = parts.reduce((a, [, c]) => a + c, 0) || 1;
  return (
    <div>
      <div className="flex h-7 gap-[2px]">
        {parts.map(([k, c], i) => {
          const share = c / total;
          const dim = selected != null && selected !== k;
          return (
            <button
              key={k}
              title={`${k}: ${compact(c)} (${(share * 100).toFixed(0)}%)`}
              onClick={() => onSelect(selected === k ? null : k)}
              className="flex min-w-[3px] items-center overflow-hidden rounded px-2 text-[11.5px] font-medium transition-opacity"
              style={{
                flex: `${share} 1 0`,
                background: colors[i % colors.length],
                color: textOn(colors[i % colors.length]),
                opacity: dim ? 0.3 : 1,
                boxShadow: `0 0 14px ${alpha(colors[i % colors.length], 0.45)}`,
              }}
            >
              {share >= 0.07 ? `${Math.round(share * 100)}%` : ""}
            </button>
          );
        })}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-[var(--ink-2)]">
        {parts.map(([k, c], i) => (
          <span key={k} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: colors[i % colors.length] }} />
            {k} <span style={{ color: pal.muted }}>{compact(c)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

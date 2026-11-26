/** Node labels, link NICs and sar notes from the discovered topology, plus the setup form; pure, so unit-tested. */
import type { NetRow, Settings, Topology } from "./types";

export type Node = "rank0" | "rank1";

/** The node's hostname, or its rank when nothing was discovered yet. */
export function nodeName(t: Topology | null, n: Node): string {
  const name = n === "rank0" ? t?.rank0Name : t?.rank1Name;
  return name || (n === "rank0" ? "rank 0" : "rank 1");
}

/** Every NIC sar saw on rank 0, busiest first. */
export function rank0Ifaces(net: NetRow[]): string[] {
  const total = new Map<string, number>();
  for (const x of net) {
    if (x.node !== "rank0") continue;
    total.set(x.iface, (total.get(x.iface) ?? 0) + (x.rxKb ?? 0) + (x.txKb ?? 0));
  }
  return [...total].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([i]) => i);
}

/** The NICs carrying the links: the discovered ones, or (before discovery) every rank 0 NIC in the data. */
export function linkIfaces(t: Topology | null, net: NetRow[]): string[] {
  return t ? t.linkIfaces : rank0Ifaces(net);
}

/** A sar sample below this (KiB/s, both directions) counts as idle. */
export const LINK_IDLE_KBPS = 1024;

/** Link KiB/s on a rank 0 NIC: mean over busy samples and the busiest one; null without samples, `busy` null if all idle. */
export interface LinkRate {
  busy: number | null;
  peak: number | null;
}

export function linkRate(net: NetRow[], iface: string): LinkRate {
  let sum = 0;
  let n = 0;
  let peak: number | null = null;
  for (const x of net) {
    if (x.node !== "rank0" || x.iface !== iface) continue;
    const v = (x.rxKb ?? 0) + (x.txKb ?? 0);
    peak = Math.max(peak ?? 0, v);
    if (v >= LINK_IDLE_KBPS) {
      sum += v;
      n += 1;
    }
  }
  return { busy: n ? sum / n : null, peak };
}

/** Why a sar chart is empty, or null when it has rows. */
export function sarNote(t: Topology | null, rows: number): string | null {
  if (rows > 0) return null;
  if (t && t.rank0SaDir == null && t.rank1SaDir == null) return "sysstat is not installed on the nodes, so there are no samples";
  return "No sar samples in this period";
}

/** The setup form, as typed. */
export interface SettingsForm {
  host: string;
  rank0Host: string;
  rank1Host: string;
  prodPort: string;
  linkIfaces: string; // comma- or space-separated
  runRoot: string;
}

export function settingsForm(s: Settings): SettingsForm {
  return {
    host: s.host ?? "",
    rank0Host: s.rank0Host ?? "",
    rank1Host: s.rank1Host ?? "",
    prodPort: s.prodPort == null ? "" : String(s.prodPort),
    linkIfaces: (s.linkIfaces ?? []).join(", "),
    runRoot: s.runRoot ?? "",
  };
}

/** Form → settings: blanks mean "discover". Throws a readable reason. */
export function parseSettingsForm(f: SettingsForm): Settings {
  const opt = (v: string) => v.trim() || null;
  let prodPort: number | null = null;
  if (f.prodPort.trim()) {
    // Digits only: Number() would also take "1e3" or "0x10".
    prodPort = /^[0-9]+$/.test(f.prodPort.trim()) ? Number(f.prodPort.trim()) : NaN;
    if (!Number.isInteger(prodPort) || prodPort < 1 || prodPort > 65535) throw new Error("the production port must be a number from 1 to 65535");
  }
  const ifaces = f.linkIfaces.split(/[\s,]+/).filter(Boolean);
  return {
    host: opt(f.host),
    rank0Host: opt(f.rank0Host),
    rank1Host: opt(f.rank1Host),
    prodPort,
    linkIfaces: ifaces.length ? ifaces : null,
    runRoot: opt(f.runRoot),
  };
}

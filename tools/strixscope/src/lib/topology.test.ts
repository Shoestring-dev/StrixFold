import { describe, expect, it } from "vitest";

import { linkIfaces, linkRate, nodeName, parseSettingsForm, rank0Ifaces, sarNote, settingsForm } from "./topology";
import type { NetRow, Settings, Topology } from "./types";

const topo: Topology = {
  rank0Host: "node0",
  rank1Host: "user@node1",
  rank0Name: "node0",
  rank1Name: "node1",
  servePort: 18080,
  linkIfaces: ["thunderbolt0", "thunderbolt1"],
  rank0SaDir: null,
  rank1SaDir: null,
  runRoot: "tfp/run/serve",
  journal: false,
  rank0Journal: false,
  rank1Journal: false,
  watchdog: false,
  model: null,
  context: null,
  lanes: null,
  discoveredAt: "2026-11-26T12:00:00Z",
};

const net = (node: NetRow["node"], iface: string, kb: number): NetRow => ({ node, ts: 0, iface, rxKb: kb, txKb: null });

describe("topology labels", () => {
  it("uses the hostnames, or the ranks before discovery", () => {
    expect([nodeName(topo, "rank0"), nodeName(topo, "rank1")]).toEqual(["node0", "node1"]);
    expect([nodeName(null, "rank0"), nodeName(null, "rank1")]).toEqual(["rank 0", "rank 1"]);
    expect(nodeName({ ...topo, rank1Name: "" }, "rank1")).toBe("rank 1");
  });

  it("takes the discovered links, even none; before discovery every rank 0 NIC, busiest first", () => {
    const rows = [net("rank0", "eth0", 1), net("rank0", "eth1", 5), net("rank0", "eth0", 1), net("rank1", "eth9", 99)];
    expect(rank0Ifaces(rows)).toEqual(["eth1", "eth0"]);
    expect(linkIfaces(null, rows)).toEqual(["eth1", "eth0"]);
    expect(linkIfaces(topo, rows)).toEqual(["thunderbolt0", "thunderbolt1"]);
    expect(linkIfaces({ ...topo, linkIfaces: [] }, rows)).toEqual([]);
  });

  it("averages link traffic over busy samples only, with the peak", () => {
    const rows = [net("rank0", "eth1", 0), net("rank0", "eth1", 500), net("rank0", "eth1", 2048), net("rank0", "eth1", 6144), net("rank1", "eth1", 99999)];
    expect(linkRate(rows, "eth1")).toEqual({ busy: 4096, peak: 6144 });
    expect(linkRate([net("rank0", "eth1", 10)], "eth1")).toEqual({ busy: null, peak: 10 });
    expect(linkRate(rows, "eth9")).toEqual({ busy: null, peak: null });
  });

  it("says why a sar chart is empty", () => {
    expect(sarNote(topo, 3)).toBeNull();
    expect(sarNote(topo, 0)).toMatch(/sysstat is not installed/);
    expect(sarNote({ ...topo, rank1SaDir: "/var/log/sa" }, 0)).toMatch(/this period/);
    expect(sarNote(null, 0)).toMatch(/this period/);
  });
});

describe("setup form", () => {
  const blank: Settings = { host: null, rank0Host: null, rank1Host: null, prodPort: null, linkIfaces: null, runRoot: null };

  it("turns blanks into null (discover) and trims", () => {
    const f = { ...settingsForm(blank), host: "  user@node1 " };
    expect(parseSettingsForm(f)).toEqual({ ...blank, host: "user@node1" });
  });

  it("splits NICs on commas or spaces and reads the port", () => {
    const s = parseSettingsForm({ ...settingsForm(blank), host: "node0", prodPort: "18080", linkIfaces: "tb0, tb1  tb2," });
    expect(s.prodPort).toBe(18080);
    expect(s.linkIfaces).toEqual(["tb0", "tb1", "tb2"]);
  });

  it("round-trips saved settings", () => {
    const s: Settings = { host: "node0", rank0Host: "a", rank1Host: "b", prodPort: 8080, linkIfaces: ["x", "y"], runRoot: "/srv/run" };
    expect(parseSettingsForm(settingsForm(s))).toEqual(s);
  });

  it("rejects a bad port", () => {
    for (const p of ["0", "70000", "80.5", "http", "1e3", "0x10", "-1"]) expect(() => parseSettingsForm({ ...settingsForm(blank), host: "h", prodPort: p })).toThrow(/port/);
  });
});

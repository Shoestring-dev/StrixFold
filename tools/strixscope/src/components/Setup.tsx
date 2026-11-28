import { useState, type ChangeEvent, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ChevronRight, Radar, TriangleAlert } from "lucide-react";

import { discover, saveSettings } from "../lib/api";
import { compact } from "../lib/stats";
import { alpha, type Palette } from "../lib/theme";
import { parseSettingsForm, settingsForm, type SettingsForm } from "../lib/topology";
import type { Settings, Topology } from "../lib/types";
import { Card } from "./Card";

interface Props {
  settings: Settings;
  /** The cached discovery, shown until Discover runs again. */
  topo: Topology | null;
  pal: Palette;
  /** Saved: the caller re-reads the topology and pulls. */
  onSaved: (s: Settings) => void;
  /** Null on first run (there is nothing to go back to). */
  onClose: (() => void) | null;
}

const FIELD =
  "w-full rounded-lg border border-[var(--line)] bg-[var(--surface)] px-3 py-1.5 text-[13px] text-[var(--ink)] outline-none transition-colors placeholder:text-[var(--muted)] focus:border-[var(--line-strong)]";

/** One ssh host in, the pair's shape out: discover, adjust, save. */
export function Setup({ settings, topo, pal, onSaved, onClose }: Props) {
  const [form, setForm] = useState<SettingsForm>(() => settingsForm(settings));
  const [overrides, setOverrides] = useState(
    () => !!(settings.rank0Host || settings.rank1Host || settings.prodPort || settings.linkIfaces || settings.runRoot),
  );
  const [found, setFound] = useState<Topology | null>(topo);
  const [error, setError] = useState<{ title: string; text: string } | null>(null);
  const [busy, setBusy] = useState<"discover" | "save" | null>(null);
  const set = (k: keyof SettingsForm) => (e: ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [k]: e.target.value });
    // What was found described the old host; keep it only until it changes.
    if (k !== "linkIfaces" && k !== "prodPort") setFound(null);
  };
  const ready = form.host.trim() !== "" && !busy;

  const run = async (what: "discover" | "save") => {
    setError(null);
    let s: Settings;
    try {
      s = parseSettingsForm(form);
    } catch (e) {
      setError({ title: "Check the overrides", text: (e as Error).message });
      return;
    }
    setBusy(what);
    try {
      if (what === "discover") setFound(await discover(s));
      else {
        await saveSettings(s);
        onSaved(s);
      }
    } catch (e) {
      if (what === "discover") setFound(null);
      setError({ title: what === "discover" ? "Discovery failed" : "Not saved", text: String(e) });
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="mx-auto max-w-4xl pt-6">
      <Card
        title={settings.host ? "The pair" : "Point Strixscope at your StrixFold pair"}
        subtitle="One ssh host is enough: discovery reads the latest run on that node and finds its peer."
        right={
          onClose && (
            <button className="chip" onClick={onClose}>
              Close
            </button>
          )
        }
      >
        <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <form
            className="flex flex-col gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              if (ready) void run("discover");
            }}
          >
            <label className="flex flex-col gap-1.5">
              <span className="label">ssh host of either node</span>
              <div className="flex gap-2">
                <input className={FIELD} value={form.host} onChange={set("host")} placeholder="node0 or user@node0" autoFocus spellCheck={false} />
                <button type="submit" className="chip flex shrink-0 items-center gap-1.5 disabled:opacity-40" data-on={busy === "discover"} disabled={!ready}>
                  <Radar size={13} style={busy === "discover" ? { animation: "spin 1.2s linear infinite" } : undefined} />
                  {busy === "discover" ? "Discovering…" : "Discover"}
                </button>
              </div>
            </label>
            <p className="label leading-relaxed">
              The desktop needs key-based ssh to both nodes (BatchMode: no password prompts); it only reads logs and samples.
            </p>

            <button
              type="button"
              className="flex items-center gap-1 self-start text-[12.5px] text-[var(--ink-2)] hover:text-[var(--ink)]"
              onClick={() => setOverrides(!overrides)}
              aria-expanded={overrides}
            >
              <ChevronRight size={14} style={{ transform: overrides ? "rotate(90deg)" : undefined, transition: "transform 160ms" }} />
              Overrides
              <span className="label ml-1">blank = discovered</span>
            </button>
            <AnimatePresence initial={false}>
              {overrides && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="grid grid-cols-2 gap-x-3 gap-y-2.5 pb-1">
                    <Field label="rank 0 ssh host">
                      <input className={FIELD} value={form.rank0Host} onChange={set("rank0Host")} placeholder={found?.rank0Host ?? "from the conf"} spellCheck={false} />
                    </Field>
                    <Field label="rank 1 ssh host">
                      <input className={FIELD} value={form.rank1Host} onChange={set("rank1Host")} placeholder={found?.rank1Host ?? "from the conf"} spellCheck={false} />
                    </Field>
                    <Field label="production port">
                      <input className={FIELD} value={form.prodPort} onChange={set("prodPort")} placeholder="most run time" inputMode="numeric" />
                    </Field>
                    <Field label="run root">
                      <input className={FIELD} value={form.runRoot} onChange={set("runRoot")} placeholder={found?.runRoot ?? "tfp/run/serve"} spellCheck={false} />
                    </Field>
                    <div className="col-span-2">
                      <Field label="link NICs on rank 0 (comma-separated)">
                        <input
                          className={FIELD}
                          value={form.linkIfaces}
                          onChange={set("linkIfaces")}
                          placeholder={found?.linkIfaces.join(", ") || "the NICs holding the link addresses"}
                          spellCheck={false}
                        />
                      </Field>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="mt-auto flex gap-2 pt-2">
              <button type="button" className="chip disabled:opacity-40" data-on="true" disabled={!ready} onClick={() => void run("save")}>
                {busy === "save" ? "Saving…" : "Save and pull"}
              </button>
              {onClose && (
                <button type="button" className="chip" onClick={onClose}>
                  Cancel
                </button>
              )}
            </div>
          </form>

          <div className="min-w-0 rounded-2xl border border-[var(--line)] p-4" style={{ background: alpha(pal.page, 0.4) }}>
            {error ? (
              <div className="flex items-start gap-2 text-[12.5px]">
                <TriangleAlert size={15} color={pal.critical} className="mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <b>{error.title}</b>
                  <div className="mt-1 whitespace-pre-wrap break-words text-[var(--ink-2)]">{error.text}</div>
                </div>
              </div>
            ) : found ? (
              <Found t={found} pal={pal} />
            ) : (
              <div className="grid h-full min-h-40 place-items-center text-center">
                <p className="label max-w-64">Enter a host and press Discover to see the pair it belongs to.</p>
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label">{label}</span>
      {children}
    </label>
  );
}

/** What discovery found: both nodes side by side, then the pair's facts. */
function Found({ t, pal }: { t: Topology; pal: Palette }) {
  const sa = (d: string | null) => (d ? <span title={d}>yes</span> : <span style={{ color: pal.warning }}>no</span>);
  const at = Date.parse(t.discoveredAt);
  const rows: [string, ReactNode, ReactNode][] = [
    ["ssh host", t.rank0Host, t.rank1Host],
    ["hostname", t.rank0Name, t.rank1Name],
    ["sysstat", sa(t.rank0SaDir), sa(t.rank1SaDir)],
    ["logs from", t.rank0Journal ? "journal" : "run files", t.rank1Journal ? "journal" : "run files"],
  ];
  const facts: [string, ReactNode][] = [
    ["serve port", t.servePort ?? "—"],
    ["link NICs", t.linkIfaces.length ? t.linkIfaces.join(", ") : <span style={{ color: pal.warning }}>none found</span>],
    ["watchdog", t.watchdog ? "yes" : "no"],
  ];
  if (t.model) facts.push(["model", t.model]);
  if (t.context != null) facts.push(["context", `${compact(t.context)} tokens`]);
  if (t.lanes != null) facts.push(["lanes", t.lanes]);
  return (
    <div className="text-[12.5px]">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="font-semibold">Found</span>
        {Number.isFinite(at) && <span className="label">{new Date(at).toLocaleString()}</span>}
      </div>
      <table className="w-full table-fixed border-separate border-spacing-y-1">
        <thead>
          <tr className="text-left">
            <th className="w-24" />
            {(["rank 0", "rank 1"] as const).map((r, i) => (
              <th key={r} className="label font-normal">
                <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: i ? pal.rank1 : pal.rank0, boxShadow: `0 0 8px ${i ? pal.rank1 : pal.rank0}` }} />
                {r}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, a, b]) => (
            <tr key={k}>
              <td className="label">{k}</td>
              <td className="truncate pr-2">{a}</td>
              <td className="truncate">{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 grid grid-cols-[6rem_minmax(0,1fr)] gap-y-1 border-t border-[var(--line)] pt-2">
        {facts.map(([k, v]) => (
          <div key={k} className="contents">
            <span className="label">{k}</span>
            <span className="truncate">{v}</span>
          </div>
        ))}
      </div>
      {!(t.rank0Journal && t.rank1Journal) && (
        <p className="mt-3 rounded-lg px-2.5 py-2 leading-relaxed" style={{ background: alpha(pal.warning, 0.1), color: pal.ink2 }}>
          Rank logs from run dirs need per-line timestamps (StrixFold <code>LOG_TIME=1</code>) for request history; without them only runs and
          system charts fill in.
        </p>
      )}
      {t.rank0SaDir == null && t.rank1SaDir == null && (
        <p className="label mt-2 leading-relaxed">No sysstat on either node: memory, CPU and link charts stay empty.</p>
      )}
    </div>
  );
}

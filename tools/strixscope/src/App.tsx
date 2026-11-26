import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Download, Network, RefreshCw, TriangleAlert } from "lucide-react";

import { Setup } from "./components/Setup";
import { getDataset, getSettings, getStatus, getTopology, IN_TAURI, isSyncing, onSyncDone, onSyncProgress, syncNow } from "./lib/api";
import { useAppUpdate } from "./lib/appUpdate";
import { dayStart, DAY_US, overlapping, prepare } from "./lib/stats";
import { palette, systemMode, type Mode } from "./lib/theme";
import { nodeName } from "./lib/topology";
import type { Dataset, Settings, Status, SyncProgress, Topology } from "./lib/types";
import { Caches } from "./views/Caches";
import { Day } from "./views/Day";
import { Overview, type Slice } from "./views/Overview";
import { Performance } from "./views/Performance";
import { Requests } from "./views/Requests";
import { System } from "./views/System";

type View = "overview" | "day" | "performance" | "caches" | "requests" | "system";
type Range = "24h" | "7d" | "30d" | "all";

const VIEWS: { id: View; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "day", label: "Day" },
  { id: "performance", label: "Performance" },
  { id: "caches", label: "Caches" },
  { id: "requests", label: "Requests" },
  { id: "system", label: "System" },
];
const RANGES: Record<Range, number | null> = { "24h": DAY_US, "7d": 7 * DAY_US, "30d": 30 * DAY_US, all: null };

function useMode(): Mode {
  const [mode, setMode] = useState<Mode>(systemMode());
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const on = () => setMode(mq.matches ? "light" : "dark");
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return mode;
}

export default function App() {
  const mode = useMode();
  const pal = useMemo(() => palette(mode), [mode]);
  const update = useAppUpdate();
  const [view, setViewState] = useState<View>(() => {
    const h = window.location.hash.slice(1) as View;
    return VIEWS.some((v) => v.id === h) ? h : "overview";
  });
  const setView = useCallback((v: View) => {
    setViewState(v);
    history.replaceState(null, "", `#${v}`);
  }, []);
  const [range, setRange] = useState<Range>("7d");
  const [status, setStatus] = useState<Status | null>(null);
  const [data, setData] = useState<Dataset | null>(null);
  const [progress, setProgress] = useState<SyncProgress | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [day, setDay] = useState<number | null>(null);
  const [includeTests, setIncludeTests] = useState(false);
  // Null settings = not read yet; a null host = not set up (setup screen shows, nothing is pulled).
  const [settings, setSettings] = useState<Settings | null>(null);
  const [topo, setTopo] = useState<Topology | null>(null);
  const [setupOpen, setSetupOpen] = useState(false);
  const setupShown = settings != null && (settings.host == null || setupOpen);

  // The filter lives in a ref so `load` stays stable, and `seq` makes the newest load win if two overlap.
  const tests = useRef(false);
  const seq = useRef(0);

  // Every pull re-discovers, so the topology is re-read after each one.
  const loadTopology = useCallback(async () => {
    try {
      setTopo(await getTopology());
    } catch (e) {
      console.error("topology", e);
    }
  }, []);

  const load = useCallback(async () => {
    const mine = ++seq.current;
    const st = await getStatus();
    if (mine !== seq.current) return;
    setStatus(st);
    setErrors(st.lastPullErrors);
    if (st.firstTs == null) return;
    const d = await getDataset(0, Number.MAX_SAFE_INTEGER, tests.current);
    if (mine !== seq.current) return;
    setData(d);
  }, []);

  const showTests = useCallback((on: boolean) => {
    tests.current = on;
    setIncludeTests(on);
    void load();
  }, [load]);

  const pull = useCallback(async () => {
    setProgress({ step: 0, steps: 0, label: "Connecting" });
    try {
      await syncNow();
    } catch (e) {
      setProgress(null);
      setErrors([String(e)]);
    }
    if (!IN_TAURI) setProgress(null);
  }, []);

  // Settings saved during a pull: that pull uses the old ones, so pull again once it is done.
  const again = useRef(false);

  // Pull on open: nothing is collected between visits.
  useEffect(() => {
    const subs = [
      onSyncProgress(setProgress),
      onSyncDone((d) => {
        setProgress(null);
        setErrors(d.errors);
        void load();
        void loadTopology();
        if (again.current) {
          again.current = false;
          void pull();
        }
      }),
    ];
    void (async () => {
      // Unreadable settings: start from none so the setup screen opens and saving writes a good file.
      const st = await getSettings().catch((e): Settings => {
        setErrors([`settings: ${String(e)}`]);
        return { host: null, rank0Host: null, rank1Host: null, prodPort: null, linkIfaces: null, runRoot: null };
      });
      setSettings(st);
      await Promise.all([load(), loadTopology()]);
      if (IN_TAURI && st?.host != null && !(await isSyncing())) void pull();
    })();
    return () => subs.forEach((s) => s.then((u) => u()));
  }, [load, loadTopology, pull]);

  const saved = useCallback(
    async (st: Settings) => {
      setSettings(st);
      setSetupOpen(false);
      void loadTopology();
      if (!IN_TAURI || !(await isSyncing())) return void pull();
      again.current = true;
      // The running pull may have ended between the check and the flag.
      if (!(await isSyncing()) && again.current) {
        again.current = false;
        void pull();
      }
    },
    [loadTopology, pull],
  );

  // "Now" is the newest line held, so a stale copy still reads sensibly.
  const now = status?.lastTs ?? Date.now() * 1000;
  const prepared = useMemo(() => (data ? prepare(data) : { reqs: [], probes: [] }), [data]);
  // Every range is half-open, [from, to); `to` is just past the newest line.
  const slice: Slice | null = useMemo(() => {
    if (!data) return null;
    const span = RANGES[range];
    const to = now + 1;
    // "All" starts at the oldest thing held, whatever kind (a loop: spreading many values into Math.min overflows the stack).
    let oldest = to - 1;
    const older = (t: number) => {
      if (t < oldest) oldest = t;
    };
    for (const r of prepared.reqs) older(r.tStart ?? r.tEnd);
    for (const r of prepared.probes) older(r.tStart ?? r.tEnd);
    for (const r of data.runs) older(r.ts);
    for (const e of data.events) older(e.ts);
    for (const x of data.sar) older(x.ts);
    for (const x of data.net) older(x.ts);
    const from = span ? to - span : oldest;
    const reqs = prepared.reqs;
    const inR = (a: number, b: number) => (t: number) => t >= a && t < b;
    const cur = inR(from, to);
    const prev = span ? inR(from - span, from) : null;
    return {
      from,
      to,
      reqs: reqs.filter((r) => cur(r.tEnd)),
      busyReqs: overlapping(reqs, from, to),
      allReqs: reqs,
      events: data.events.filter((e) => cur(e.ts)),
      runs: data.runs.filter((r) => cur(r.ts)),
      allRuns: data.runs,
      sar: data.sar.filter((x) => cur(x.ts)),
      net: data.net.filter((x) => cur(x.ts)),
      prev:
        prev && span
          ? {
              from: from - span,
              to: from,
              reqs: reqs.filter((r) => prev(r.tEnd)),
              busyReqs: overlapping(reqs, from - span, from),
              runs: data.runs.filter((r) => prev(r.ts)),
            }
          : null,
    };
  }, [data, prepared, range, now]);

  const allReqs = prepared.reqs;
  const days = useMemo(() => {
    const set = new Set(allReqs.map((r) => dayStart(r.tEnd)));
    return [...set].sort((a, b) => a - b);
  }, [allReqs]);
  // Default to the latest day with real traffic (today may have barely begun).
  const busyDay = useMemo(() => {
    const counts = new Map<number, number>();
    for (const r of allReqs) counts.set(dayStart(r.tEnd), (counts.get(dayStart(r.tEnd)) ?? 0) + 1);
    return [...days].reverse().find((d) => (counts.get(d) ?? 0) >= 50) ?? days[days.length - 1];
  }, [allReqs, days]);
  const shownDay = day != null && days.includes(day) ? day : (busyDay ?? dayStart(now));

  return (
    <div className="relative flex h-full flex-col">
      <div className="aurora" />
      <header className="relative z-10 flex items-center gap-6 px-6 pb-3 pt-4">
        <Brand pal={pal} />
        <nav className="flex gap-1 rounded-full border border-[var(--line)] bg-[var(--surface)]/60 p-1">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              onClick={() => {
                setView(v.id);
                setSetupOpen(false);
              }}
              className="relative rounded-full px-3.5 py-1.5 text-[13px] text-[var(--ink-2)] transition-colors hover:text-[var(--ink)] data-[on=true]:text-[var(--ink)]"
              data-on={view === v.id && !setupShown}
            >
              {view === v.id && !setupShown && (
                <motion.span
                  layoutId="tab"
                  className="absolute inset-0 rounded-full"
                  style={{ background: `linear-gradient(135deg, ${pal.prefill}33, ${pal.rank0}33)`, border: `1px solid ${pal.prefill}55` }}
                  transition={{ type: "spring", stiffness: 420, damping: 34 }}
                />
              )}
              <span className="relative">{v.label}</span>
            </button>
          ))}
        </nav>
        {view !== "day" && !setupShown && (
          <div className="flex gap-1">
            {(Object.keys(RANGES) as Range[]).map((r) => (
              <button key={r} className="chip" data-on={range === r} onClick={() => setRange(r)}>
                {r === "all" ? "All" : r}
              </button>
            ))}
          </div>
        )}
        <button
          className="chip"
          data-on={includeTests}
          onClick={() => showTests(!includeTests)}
          title="Include runs on other ports (test servers); production only otherwise"
        >
          test runs
        </button>
        <div className="ml-auto flex items-center gap-3">
          {update.error && (
            <span className="label max-w-64 truncate" style={{ color: pal.critical }} title={update.error}>
              update: {update.error}
            </span>
          )}
          {update.phase === "ready" && (
            <button className="chip flex items-center gap-1.5" onClick={() => void update.install()}>
              <Download size={13} /> Update to {update.version}
            </button>
          )}
          {settings && (
            <button
              className="chip flex items-center gap-1.5"
              data-on={setupShown}
              disabled={settings.host == null}
              onClick={() => setSetupOpen(!setupOpen)}
              title="The pair: ssh host, discovered nodes and overrides"
            >
              <Network size={13} />
              {topo ? `${nodeName(topo, "rank0")} · ${nodeName(topo, "rank1")}` : "Pair"}
            </button>
          )}
          {settings?.host != null && <PullButton pal={pal} progress={progress} lastPull={status?.lastPull ?? null} onPull={pull} />}
        </div>
      </header>

      <AnimatePresence>
        {errors.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="relative z-10 mx-6 mb-2 flex items-start gap-2 rounded-xl border px-3 py-2 text-[12.5px]"
            style={{ borderColor: `${pal.warning}55`, background: `${pal.warning}14` }}
          >
            <TriangleAlert size={15} color={pal.warning} className="mt-0.5 shrink-0" />
            <div>
              <b>Some sources did not answer; showing what is stored.</b>
              {errors.map((e, i) => (
                <div key={i} className="text-[var(--ink-2)]">
                  {e}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="scroll-area relative z-10 flex-1 px-6 pb-8">
        {setupShown ? (
          <Setup
            key={settings.host ?? ""}
            settings={settings}
            topo={topo}
            pal={pal}
            onSaved={saved}
            onClose={settings.host == null ? null : () => setSetupOpen(false)}
          />
        ) : !slice ? (
          <Empty pulling={!!progress} />
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={view}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.22 }}
            >
              {view === "overview" && (
                <Overview
                  s={slice}
                  topo={topo}
                  pal={pal}
                  onOpenDay={(d) => {
                    setDay(d);
                    setView("day");
                  }}
                />
              )}
              {view === "day" && data && (
                <Day day={shownDay} days={days} reqs={allReqs} sar={data.sar} topo={topo} pal={pal} onDay={setDay} />
              )}
              {view === "performance" && <Performance s={slice} pal={pal} />}
              {view === "caches" && <Caches s={slice} pal={pal} />}
              {view === "requests" && <Requests s={slice} probes={prepared.probes} pal={pal} />}
              {view === "system" && <System s={slice} allRuns={slice.allRuns} topo={topo} pal={pal} />}
            </motion.div>
          </AnimatePresence>
        )}
      </main>
    </div>
  );
}

/** Two glowing rings — the pair, and an owl's eyes (Strix is the owl genus). */
function Brand({ pal }: { pal: ReturnType<typeof palette> }) {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="38" height="22" viewBox="0 0 38 22" aria-hidden>
        {[
          [11, pal.rank0],
          [27, pal.rank1],
        ].map(([cx, c]) => (
          <g key={cx as number}>
            <circle cx={cx as number} cy={11} r={8.5} fill="none" stroke={c as string} strokeWidth={2.5} style={{ filter: `drop-shadow(0 0 4px ${c})` }} />
            <circle cx={cx as number} cy={11} r={3} fill={pal.ink} />
          </g>
        ))}
      </svg>
      <span className="text-[17px] font-semibold tracking-tight">Strixscope</span>
    </div>
  );
}

function PullButton({
  pal,
  progress,
  lastPull,
  onPull,
}: {
  pal: ReturnType<typeof palette>;
  progress: SyncProgress | null;
  lastPull: string | null;
  onPull: () => void;
}) {
  const frac = progress && progress.steps > 0 ? (progress.step + 0.5) / progress.steps : 0;
  const ago = lastPull ? Math.max(0, Math.round((Date.now() - Date.parse(lastPull)) / 60000)) : null;
  return (
    <div className="flex items-center gap-3">
      <span className="label text-right">
        {progress ? progress.label : ago == null ? "never pulled" : ago < 1 ? "pulled just now" : `pulled ${ago} min ago`}
      </span>
      <button
        onClick={onPull}
        disabled={!!progress}
        className="relative grid h-9 w-9 place-items-center rounded-full border border-[var(--line)] bg-[var(--surface)] transition-colors hover:border-[var(--line-strong)]"
        aria-label="Pull logs now"
      >
        {progress && (
          <svg className="absolute inset-0" viewBox="0 0 36 36">
            <circle cx="18" cy="18" r="16" fill="none" stroke={`${pal.prefill}30`} strokeWidth="2.5" />
            <motion.circle
              cx="18"
              cy="18"
              r="16"
              fill="none"
              stroke={pal.prefill}
              strokeWidth="2.5"
              strokeLinecap="round"
              transform="rotate(-90 18 18)"
              strokeDasharray={2 * Math.PI * 16}
              animate={{ strokeDashoffset: 2 * Math.PI * 16 * (1 - frac) }}
            />
          </svg>
        )}
        <RefreshCw size={15} style={progress ? { animation: "spin 1.2s linear infinite" } : undefined} />
      </button>
    </div>
  );
}

function Empty({ pulling }: { pulling: boolean }) {
  return (
    <div className="grid h-[60vh] place-items-center text-center">
      <div>
        <div className="text-[20px] font-semibold">{pulling ? "Pulling the pair's logs…" : "No logs yet"}</div>
        <p className="label mt-2">
          {pulling ? "The first pull copies every log over ssh; later ones only fetch new lines." : "Press the refresh button to pull."}
        </p>
      </div>
    </div>
  );
}



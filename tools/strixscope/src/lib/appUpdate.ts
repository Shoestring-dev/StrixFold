import { useEffect, useEffectEvent, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { Update } from "@tauri-apps/plugin-updater";

import { IN_TAURI } from "./api";
import { IS_LINUX } from "./platform";

/** Self-update state machine; the plugin verifies the minisign signature and a failed check keeps a finished download. */
export type UpdatePhase = "idle" | "checking" | "downloading" | "ready" | "installing";

export interface AppUpdate {
  phase: UpdatePhase;
  /** The version that was found (and downloaded, once phase is "ready"). */
  version: string | null;
  error: string | null;
  /** Install the downloaded update and relaunch. */
  install: () => Promise<void>;
}

/** Re-check the manifest every six hours. */
const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;

/** The updater exists only in builds with the `self-update` feature; also require Tauri's bridge so plain `npm run dev` skips it. */
let supported: Promise<boolean> | null = null;
function updateSupported(): Promise<boolean> {
  supported ??= !(IS_LINUX && IN_TAURI)
    ? Promise.resolve(false)
    : invoke<boolean>("self_update_enabled").catch(() => false);
  return supported;
}

/** Invoke/plugin errors are usually strings, but normalize everything. */
function describeError(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error && e.message) return e.message;
  return String(e);
}

export function useAppUpdate(): AppUpdate {
  const [phase, setPhase] = useState<UpdatePhase>("idle");
  const [version, setVersion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pending = useRef<Update | null>(null);
  const busy = useRef(false);

  // One check at a time: a startup check may still be downloading when the six-hour timer fires.
  async function runCheck() {
    if (busy.current) return;
    busy.current = true;
    try {
      if (!(await updateSupported())) return;
      setError(null);
      setPhase("checking");
      const { check } = await import("@tauri-apps/plugin-updater");
      const found = await check();
      if (!found) {
        // Keep a downloaded update visible; otherwise go quiet.
        setPhase(pending.current ? "ready" : "idle");
        setVersion(pending.current?.version ?? null);
        return;
      }
      if (pending.current?.version === found.version) {
        // Already downloaded: no reason to download it again.
        setPhase("ready");
        setVersion(found.version);
        return;
      }
      setPhase("downloading");
      setVersion(found.version);
      await found.download();
      // Only a completed download counts as pending install.
      pending.current = found;
      setPhase("ready");
    } catch (e) {
      setError(describeError(e));
      if (pending.current) {
        // The failed check was for a newer version; keep naming the one actually downloaded.
        setPhase("ready");
        setVersion(pending.current.version);
      } else {
        setPhase("idle");
        setVersion(null);
      }
    } finally {
      busy.current = false;
    }
  }

  // Silent check on startup, then every six hours; an effect event reads the latest `runCheck` without re-running.
  const silentCheck = useEffectEvent(() => {
    void runCheck();
  });

  useEffect(() => {
    let id: number | undefined;
    let cancelled = false;
    void updateSupported().then((ok) => {
      if (!ok || cancelled) return;
      silentCheck();
      id = window.setInterval(() => silentCheck(), CHECK_INTERVAL_MS);
    });
    return () => {
      cancelled = true;
      if (id !== undefined) window.clearInterval(id);
    };
  }, []);

  async function install() {
    const update = pending.current;
    if (!update || busy.current) return;
    busy.current = true;
    setPhase("installing");
    setError(null);
    try {
      // install() runs the system installer then relaunches; a refused install returns to "ready" and can be retried.
      await update.install();
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (e) {
      setPhase("ready");
      setError(describeError(e));
    } finally {
      busy.current = false;
    }
  }

  return { phase, version, error, install };
}

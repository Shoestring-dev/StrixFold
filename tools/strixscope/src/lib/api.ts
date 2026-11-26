/** Typed wrappers over the Rust commands; a plain-browser `npm run dev` serves a dev snapshot and a generic set-up pair. */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type { Dataset, Settings, Status, SyncDone, SyncProgress, Topology } from "./types";
import { DEV_SETTINGS, devTopology, filterDataset } from "./devdata";

export const IN_TAURI = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

let devData: Promise<Dataset> | null = null;
function devDataset(): Promise<Dataset> {
  devData ??= fetch("/dev/dataset.json").then((r) => {
    if (!r.ok) throw new Error("no dev snapshot at /dev/dataset.json");
    return r.json() as Promise<Dataset>;
  });
  return devData;
}

export async function getDataset(from: number, to: number, includeTests: boolean): Promise<Dataset> {
  if (!IN_TAURI) return filterDataset(await devDataset(), from, to, includeTests);
  return invoke<Dataset>("get_dataset", { from, to, includeTests });
}

export async function getStatus(): Promise<Status> {
  if (!IN_TAURI) {
    const d = await devDataset();
    const span = (xs: number[]) => xs.reduce(([lo, hi], x) => [Math.min(lo, x), Math.max(hi, x)], [Infinity, -Infinity]);
    const [firstTs, lastTs] = span(d.requests.map((r) => r.tEnd));
    return {
      firstTs,
      lastTs,
      lastPull: new Date().toISOString(),
      lastPullErrors: [],
    };
  }
  return invoke<Status>("get_status");
}

export const getSettings = () => (IN_TAURI ? invoke<Settings>("get_settings") : Promise.resolve(DEV_SETTINGS));
/** Rejects with the reason when a host is malformed. */
export const saveSettings = (settings: Settings) =>
  IN_TAURI ? invoke<void>("save_settings", { settings }) : Promise.resolve();
/** The cached discovery with the saved overrides; null before the first one. */
export async function getTopology(): Promise<Topology | null> {
  if (!IN_TAURI) return devTopology(await devDataset());
  return invoke<Topology | null>("get_topology");
}
/** Discovers now with these (unsaved) settings; rejects with the reason. */
export async function discover(settings: Settings): Promise<Topology> {
  if (!IN_TAURI) return devTopology(await devDataset());
  return invoke<Topology>("discover", { settings });
}

export const isSyncing = () => (IN_TAURI ? invoke<boolean>("is_syncing") : Promise.resolve(false));
export const syncNow = () => (IN_TAURI ? invoke<void>("sync_now") : Promise.resolve());

export function onSyncProgress(cb: (p: SyncProgress) => void): Promise<UnlistenFn> {
  if (!IN_TAURI) return Promise.resolve(() => {});
  return listen<SyncProgress>("sync-progress", (e) => cb(e.payload));
}
export function onSyncDone(cb: (d: SyncDone) => void): Promise<UnlistenFn> {
  if (!IN_TAURI) return Promise.resolve(() => {});
  return listen<SyncDone>("sync-done", (e) => cb(e.payload));
}

#!/usr/bin/env python3
"""memledger.py sample OUT.jsonl PHASE_FILE STOP_FILE [PIDFILE...] | summary OUT.jsonl [PLAN_GIB]

The allocation and residency ledger of one box during a served run (handoff section 5, gate M3.allocation-residency).
`sample` writes one JSON line per second until STOP_FILE exists: the phase the driver wrote into PHASE_FILE (load,
warmup, prefill, verify, steady, ...), host memory (/proc/meminfo), the rank processes' RSS split (anon/file/shmem)
and swap, the GPU's GTT/VRAM use (on this UMA part GTT is host RAM the GPU maps: counted once, inside MemTotal), swap
traffic, major faults, memory PSI, the user slice's memory.events, clocks, temperatures and package power.
`summary` prints per phase the peak and the end value of each quantity, the growth of swap and of resident weights
after load, and checks: no swap-out by the engine, no OOM/max events, MemAvailable never under 8 GiB, GTT at load end
within PLAN_GIB (the memory plan's device figure) and no loss of resident weights (rank RSS not falling by more than
1 GiB after load while the process lives).
Standard library only.
"""
import glob
import json
import os
import sys
import time

GIB = 1 << 30


def read(p, default=None):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return default


def meminfo():
    out = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) * 1024
    return {k: out.get(k) for k in ("MemTotal", "MemAvailable", "MemFree", "Cached", "AnonPages", "Shmem", "SwapTotal", "SwapFree", "Mlocked", "Unevictable")}


def vmstat():
    want = ("pswpin", "pswpout", "pgmajfault", "oom_kill", "pgscan_direct", "pgsteal_direct")
    out = {}
    for line in (read("/proc/vmstat") or "").splitlines():
        k, v = line.split()
        if k in want:
            out[k] = int(v)
    return out


def proc(pid):
    st = read(f"/proc/{pid}/status")
    if st is None:
        return None
    out = {"pid": pid}
    for line in st.splitlines():
        k, _, v = line.partition(":")
        if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "VmHWM", "VmLck", "VmPin"):
            out[k] = int(v.split()[0]) * 1024
    io = read(f"/proc/{pid}/io")
    if io:
        for line in io.splitlines():
            k, _, v = line.partition(":")
            if k in ("read_bytes", "write_bytes"):
                out[k] = int(v)
    return out


def gpu():
    d = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"))
    if not d:
        return {}
    base = os.path.dirname(d[0])
    out = {}
    for k in ("mem_info_gtt_used", "mem_info_vram_used", "gpu_busy_percent"):
        v = read(os.path.join(base, k))
        out[k] = int(v) if v and v.strip().isdigit() else None
    sclk = read(os.path.join(base, "pp_dpm_sclk")) or ""
    cur = [l for l in sclk.splitlines() if l.strip().endswith("*")]
    out["sclk_mhz"] = int(cur[0].split()[1].rstrip("Mhz")) if cur else None
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        name = (read(os.path.join(h, "name")) or "").strip()
        if name == "amdgpu":
            t, p = read(os.path.join(h, "temp1_input")), read(os.path.join(h, "power1_average")) or read(os.path.join(h, "power1_input"))
            out["gpu_temp_c"] = int(t) / 1000 if t else None
            out["pkg_power_w"] = int(p) / 1e6 if p else None
        elif name == "k10temp":
            t = read(os.path.join(h, "temp1_input"))
            out["cpu_tctl_c"] = int(t) / 1000 if t else None
    return out


def psi():
    out = {}
    for line in (read("/proc/pressure/memory") or "").splitlines():
        kind, *kv = line.split()
        out[kind] = {k: float(v) for k, v in (x.split("=") for x in kv)}
    return out


def events():
    out = {}
    for p in ("/sys/fs/cgroup/user.slice/memory.events",):
        for line in (read(p) or "").splitlines():
            k, v = line.split()
            out[k] = int(v)
    return out


def pids(files):
    out = []
    for f in files:
        v = (read(f) or "").strip()
        if v.isdigit():
            out.append(int(v))
    return out


def sample(out, phase_file, stop_file, pidfiles):
    with open(out, "a") as f:
        while not os.path.exists(stop_file):
            t = time.time()
            rec = {"t": round(t, 3), "phase": (read(phase_file) or "").strip() or None, "mem": meminfo(), "vm": vmstat(),
                   "gpu": gpu(), "psi": psi(), "events": events(), "load": (read("/proc/loadavg") or "").split()[:3],
                   "procs": [p for p in (proc(pid) for pid in pids(pidfiles)) if p]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(max(0.0, 1.0 - (time.time() - t)))


def summary(path, plan_gib=None):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        print(json.dumps({"error": "no samples"}))
        return 1
    phases = []
    for r in recs:
        if not phases or phases[-1]["phase"] != r["phase"]:
            phases.append({"phase": r["phase"], "n": 0, "first_t": r["t"]})
        ph = phases[-1]
        ph["n"] += 1
        ph["last_t"] = r["t"]
        rss = sum(p.get("VmRSS", 0) for p in r["procs"])
        vals = {"mem_available_min_gib": r["mem"]["MemAvailable"] / GIB, "gtt_used_max_gib": (r["gpu"].get("mem_info_gtt_used") or 0) / GIB,
                "rank_rss_max_gib": rss / GIB, "cached_max_gib": r["mem"]["Cached"] / GIB,
                "gpu_temp_max_c": r["gpu"].get("gpu_temp_c"), "cpu_tctl_max_c": r["gpu"].get("cpu_tctl_c"),
                "pkg_power_max_w": r["gpu"].get("pkg_power_w"), "psi_full_avg10_max": r["psi"].get("full", {}).get("avg10")}
        for k, v in vals.items():
            if v is None:
                continue
            if k.endswith("_min_gib"):
                ph[k] = min(ph.get(k, v), v)
            else:
                ph[k] = max(ph.get(k, v), v)
        ph["end"] = {"mem_available_gib": vals["mem_available_min_gib"], "gtt_used_gib": vals["gtt_used_max_gib"], "rank_rss_gib": vals["rank_rss_max_gib"]}
    first, last = recs[0], recs[-1]
    d = lambda k: last["vm"].get(k, 0) - first["vm"].get(k, 0)
    swap_used = lambda r: (r["mem"]["SwapTotal"] or 0) - (r["mem"]["SwapFree"] or 0)
    ev = {k: last["events"].get(k, 0) - first["events"].get(k, 0) for k in ("oom", "oom_kill", "max")}
    min_avail = min(r["mem"]["MemAvailable"] for r in recs) / GIB
    # resident weights: the rank's RSS once loaded (first sample after the load phase) vs the lowest later while it lives
    # (the first start only: from the end of its load to the first stop, while the rank lives)
    after, seen_load = [], False
    for r in recs:
        if r["phase"] == "load":
            seen_load = True
        elif seen_load and r["phase"] in ("stop", "done"):
            break
        elif seen_load and r["procs"]:
            after.append(r)
    loaded = sum(p.get("VmRSS", 0) for p in after[0]["procs"]) / GIB if after else None
    low = min(sum(p.get("VmRSS", 0) for p in r["procs"]) for r in after) / GIB if after else None
    gtt_loaded = (after[0]["gpu"].get("mem_info_gtt_used") or 0) / GIB if after else None
    gtt_low = min((r["gpu"].get("mem_info_gtt_used") or 0) for r in after) / GIB if after else None
    checks = {
        "no_swap_out": d("pswpout") == 0 and swap_used(last) <= swap_used(first),
        "no_oom_or_max_events": all(v == 0 for v in ev.values()) and d("oom_kill") == 0,
        "mem_available_floor_8gib": min_avail >= 8.0,
        # the weights live in GPU-mapped (GTT) memory: it must not shrink while the ranks run; host RSS likewise
        "resident_weights_kept": (gtt_loaded is not None and gtt_low is not None and gtt_loaded - gtt_low <= 1.0
                                  and loaded is not None and low is not None and loaded - low <= 1.0),
    }
    if plan_gib is not None and gtt_loaded is not None:
        checks["gtt_within_plan"] = gtt_loaded <= plan_gib
    out = {"samples": len(recs), "seconds": round(last["t"] - first["t"], 1), "phases": phases, "swap_out_pages": d("pswpout"),
           "swap_in_pages": d("pswpin"), "major_faults": d("pgmajfault"), "direct_reclaim_scans": d("pgscan_direct"),
           "memory_events": ev, "mem_available_min_gib": round(min_avail, 2), "rank_rss_loaded_gib": loaded, "rank_rss_low_gib": low,
           "gtt_loaded_gib": gtt_loaded, "gtt_low_after_load_gib": gtt_low, "plan_gib": plan_gib, "checks": checks, "pass": all(checks.values())}
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "sample":
        sample(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "summary":
        sys.exit(summary(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None))
    else:
        print(__doc__)
        sys.exit(2)

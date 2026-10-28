#!/usr/bin/env python3
"""sclk_sample.py RUN_ID OUT [PERIOD_MS]: the GPU clock (amdgpu hwmon freq1_input, MHz) every PERIOD_MS (default 1)
while first_light's quiet window for RUN_ID is open (the timed bench), stamped with CLOCK_BOOTTIME and CLOCK_MONOTONIC
(ns; rocprofv3's kernel timestamps are compared against them); package power (power1_average, W) every 50th sample.
OUT: one JSON line a sample. OUT.stop ends a wait for a window that never opened (or the sampling, 15 min max)."""
import glob, json, os, sys, time
run, out = sys.argv[1], sys.argv[2]
period = float(sys.argv[3]) / 1e3 if len(sys.argv) > 3 else 1e-3
want = f"/var/tmp/halo-quiet.d/want.first-light-{run}"
hw = [h for h in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*") if open(h + "/name").read().strip() == "amdgpu"][0]
fq = os.open(hw + "/freq1_input", os.O_RDONLY)
pw = os.open(hw + "/power1_average", os.O_RDONLY)
t_end = time.time() + 3600
while not os.path.exists(want):
    if time.time() > t_end or os.path.exists(out + ".stop"):
        sys.exit("no quiet window")
    time.sleep(0.05)
BT, MO = time.CLOCK_BOOTTIME, time.CLOCK_MONOTONIC
n = 0
t_stop = time.time() + 900
with open(out, "w") as f:
    nxt = time.clock_gettime_ns(MO)
    while os.path.exists(want) and not os.path.exists(out + ".stop") and time.time() < t_stop:
        t0 = time.clock_gettime_ns(BT)
        v = int(os.pread(fq, 32, 0)) / 1e6
        t1 = time.clock_gettime_ns(BT)
        rec = {"t": (t0 + t1) // 2, "tm": time.clock_gettime_ns(MO), "rd_us": (t1 - t0) / 1e3, "sclk": v}
        if n % 50 == 0:
            rec["w"] = int(os.pread(pw, 32, 0)) / 1e6
        f.write(json.dumps(rec) + "\n")
        n += 1
        nxt += int(period * 1e9)
        d = nxt - time.clock_gettime_ns(MO)
        if d > 0:
            time.sleep(d / 1e9)
        else:
            nxt = time.clock_gettime_ns(MO)
print(f"{os.uname().nodename}: {n} samples")

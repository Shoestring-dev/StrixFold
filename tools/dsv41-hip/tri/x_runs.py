"""Consecutive launches of one kernel and grid in a rocprofv3 kernel trace: name, grid, count, median/min device ms."""
import csv, glob, statistics, sys
trace = glob.glob(sys.argv[1] + "/**/*kernel_trace*.csv", recursive=True)[0]
rows = sorted(csv.DictReader(open(trace)), key=lambda r: int(r["Start_Timestamp"]))
runs = []
for r in rows:
    if r["Kernel_Name"].startswith("__amd"):
        continue
    key = (r["Kernel_Name"][:40], r.get("Grid_Size_X", r.get("Grid_Size")), r.get("Grid_Size_Y", ""))
    d = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6
    if runs and runs[-1][0] == key:
        runs[-1][1].append(d)
    else:
        runs.append([key, [d]])
for key, ds in runs:
    print(f"{key[0]:40s} grid={key[1]}x{key[2]} n={len(ds)} median={statistics.median(ds):.3f} ms min={min(ds):.3f} ms")

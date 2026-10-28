"""Device durations from rocprofv3's kernel trace, run by run (consecutive launches of one kernel and grid), aligned
with t_perf's log lines in order: prints each case with the host-measured and the profiled device time."""
import csv, glob, json, re, statistics, sys
trace = glob.glob(sys.argv[1] + "/**/*kernel_trace*.csv", recursive=True)[0]
rows = list(csv.DictReader(open(trace)))
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
runs = []
for r in rows:
    key = (r["Kernel_Name"], r.get("Grid_Size_X", r.get("Grid_Size")), r.get("Grid_Size_Y", ""), r.get("Grid_Size_Z", ""))
    d = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000.0
    if runs and runs[-1][0] == key:
        runs[-1][1].append(d)
    else:
        runs.append([key, [d]])
runs = [r for r in runs if len(r[1]) >= 3 and not r[0][0].startswith("__amd")]   # timeit's batches
cases = [l for l in open(sys.argv[2]) if l.startswith("[INFO]")]
i = 0
for line in cases:
    m = re.match(r"\[INFO\] (\S+) (.*?): (\{.*\})", line.strip())
    info = json.loads(m.group(3))
    while i < len(runs) and runs[i][0][0] != m.group(1):
        i += 1
    if i >= len(runs):
        break
    key, ds = runs[i]
    i += 1
    print(f"{m.group(1):26s} {m.group(2)[:48]:48s} rows={info['rows']:<5d} host={info['us']:>9.1f}us  device(median)={statistics.median(ds):>9.1f}us  [{key[0][:24]} n={len(ds)}]")

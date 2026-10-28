#!/bin/bash
# qual_bench.sh bench|soak CONF RUN_ID [client args...]: a timed served run under handoff section 9's rules (gates
# M5.benchmark, M5.soak). One run under both GPU locks (node0's first): both ranks from CONF, a probe, the run
# conditions (the prompt set and client by hash, clocks, power profile, temperatures, load on both boxes), then inside
# quiet windows on both boxes (node0 first; refreshed every 5 minutes):
#   bench  benchmark.py on rank 0's box (creator prompt set v1; extra args, e.g. --reps 5 --prefill 2000,8000,16000)
#   soak   soak.py for SOAK_S seconds (default 900; extra args, e.g. --long-every 5), with the USB4 links' missed
#          frames, TCP retransmissions, RTOs and loss probes counted on both boxes before and after
# while memledger.py samples memory, clocks, temperatures and power on both boxes. Afterwards run_manifest.py writes
# manifest.json: the deployment's full identity (binaries, kit objects, bundle, both ranks' resolved.json, toolchain
# per box, the request token ids). Then the ranks stop, the windows
# close and the locks are released on every exit path. Writes ~/tfp/qual/out/<bench|soak>-RUN_ID/<bench|soak>.json for
# qualify.sh (deploy commit and configuration hash recorded; status pass only if the client saw no error and, for the
# soak, the duration was reached).
set -uo pipefail
MODE=${1:?bench|soak}; QCONF=${2:?CONF}; RID=${3:?RUN_ID}; shift 3
QRUN=$MODE-$RID
QLOCK_MAX_S=${QLOCK_MAX_S:-$([ "$MODE" = soak ] && echo $(( ${SOAK_S:-900} + 900 )) || echo 2700)}
source "$(dirname "$0")/qual_lib.sh"
ql_init
CREATOR=$HOME/tfp/qual/bench/creator
[ -f "$CREATOR/prompts-v1.json" ] || { echo "no $CREATOR/prompts-v1.json (recipe 9fa230d3 bench/prompts-v1.json)" >&2; exit 2; }
scp -q "$QTOOLS/benchmark.py" "$QTOOLS/soak.py" "$CREATOR/prompts-v1.json" node0:tfp/qual/bin/
QTAG=q-qual-$MODE-$RID

netcount() {  # netcount BOX: the links' and TCP's loss counters as JSON
  local cmd='python3 - <<"PY"
import json, subprocess
out = {}
for dev in ("thunderbolt0", "thunderbolt1"):
    t = subprocess.run(["ip", "-s", "link", "show", dev], capture_output=True, text=True).stdout.split("\n")
    i = next((k for k, l in enumerate(t) if l.strip().startswith("RX:")), None)
    if i is not None:
        h, v = t[i].split(), t[i + 1].split()
        out[dev] = dict(zip(h[1:], map(int, v)))
snmp = open("/proc/net/snmp").read().split("\n")
tcp = [l.split() for l in snmp if l.startswith("Tcp:")]
out["tcp"] = dict(zip(tcp[0][1:], map(int, tcp[1][1:])))
ns = [l.split() for l in open("/proc/net/netstat").read().split("\n") if l.startswith("TcpExt:")]
ext = dict(zip(ns[0][1:], map(int, ns[1][1:])))
out["tcpext"] = {k: ext.get(k) for k in ("TCPTimeouts", "TCPLossProbes", "TCPSpuriousRTOs", "TCPLostRetransmit", "TCPFastRetrans")}
print(json.dumps(out))
PY'
  if [ "$1" = node0 ]; then ssh -n node0 "$cmd"; else bash -c "$cmd"; fi
}

conditions() {  # the run conditions: prompt set, client, box state at the start (manifest.json, written at the end by
  # run_manifest.py, holds the deployment's full identity)
  python3 - "$QOUT/conditions.json" "$QCONF1" "${QDEPLOY//\$HOME/$HOME}" "$CREATOR" "$QTOOLS" "$MODE" "$*" \
    "$(ssh -n node0 'cat /proc/loadavg; grep "\*" /sys/class/drm/card0/device/pp_dpm_sclk; cat /sys/class/drm/card0/device/power_dpm_force_performance_level; cat /sys/firmware/acpi/platform_profile 2>/dev/null; cat /sys/class/hwmon/hwmon*/temp1_input 2>/dev/null | head -3 | tr "\n" " "')" \
    "$(cat /proc/loadavg; grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk; cat /sys/class/drm/card0/device/power_dpm_force_performance_level; cat /sys/firmware/acpi/platform_profile 2>/dev/null; cat /sys/class/hwmon/hwmon*/temp1_input 2>/dev/null | head -3 | tr '\n' ' ')" \
    "$(hipcc --version 2>/dev/null | head -1)" "$(uname -r)" <<'PY'
import hashlib, json, os, sys
out, conf, dep, creator, tools, mode, args, s0, s1, hip, kern = sys.argv[1:12]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
d = json.load(open(os.path.join(dep, "DEPLOY.json")))
json.dump({"schema": "tf-port-bench-manifest/1", "mode": mode, "client_args": args,
           "deploy": {"dir": dep, "commit": d.get("commit"), "tree": d.get("tree"), "sha256sums": sha(os.path.join(dep, "SHA256SUMS"))},
           "config": conf, "config_sha256": sha(conf), "config_text": open(conf).read(),
           "prompt_set": {"file": "recipe 9fa230d3 bench/prompts-v1.json", "git_blob": "15f511f840bd946189264f22fdacaf088a2497c9",
                          "sha256": sha(os.path.join(creator, "prompts-v1.json"))},
           "client": {"benchmark.py": sha(os.path.join(tools, "benchmark.py")), "soak.py": sha(os.path.join(tools, "soak.py"))},
           "checkpoint": "LibertAI EXL3 2.9 bpw @ c178a01284fc5c0e859d8668cffaac81ca8f4042; Engram 1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170",
           "sampling": "temperature 0, thinking off", "toolchain": {"hipcc": hip, "kernel": kern},
           "box_state": {"node0": s0.split("\n"), "node1": s1.split("\n")}}, open(out, "w"), indent=1)
PY
}

quiet() { ql_quiet "$1" "$QTAG" >/dev/null; }  # quiet on|off: both boxes (qual_lib.sh)

cleanup() { set +e; [ -n "${REFRESH:-}" ] && kill "$REFRESH" 2>/dev/null; quiet off; ql_stop; ql_sampler_stop; ql_copy_logs; ql_unlock; }
trap cleanup EXIT
ql_lock || exit 3
ql_sampler_start
if ! ql_start; then
  if ql_env_refusal; then ql_log "node0 not idle: rank 0 refused to load (memory budget, U25); nothing measured"; exit 4; fi
  ql_log "the server did not come up"; exit 1
fi
ql_phase warmup
ssh -n node0 "python3 ~/tfp/qual/bin/serve_qual.py http://127.0.0.1:$QPORT $QRUN0/probe.json --model $QNAME --cases probe" >> "$QOUT/driver.log" 2>&1
conditions "$@"
quiet on
( while sleep 300; do quiet on; done ) &
REFRESH=$!
sleep 5
ql_phase "$MODE"
t0=$(date +%s)
if [ "$MODE" = bench ]; then
  ssh -n node0 "mkdir -p $QRUN0/out && python3 ~/tfp/qual/bin/benchmark.py http://127.0.0.1:$QPORT $QRUN0/out --prompts ~/tfp/qual/bin/prompts-v1.json --label $RID --model $QNAME $*" | tee "$QOUT/client.log"
  crc=${PIPESTATUS[0]}
else
  netcount node0 > "$QOUT/net-before-node0.json"; netcount node1 > "$QOUT/net-before-node1.json"
  ssh -n node0 "mkdir -p $QRUN0/out && python3 ~/tfp/qual/bin/soak.py http://127.0.0.1:$QPORT $QRUN0/out --prompts ~/tfp/qual/bin/prompts-v1.json --seconds ${SOAK_S:-900} --model $QNAME $*" | tee "$QOUT/client.log"
  crc=${PIPESTATUS[0]}
  netcount node0 > "$QOUT/net-after-node0.json"; netcount node1 > "$QOUT/net-after-node1.json"
fi
secs=$(( $(date +%s) - t0 ))
kill "$REFRESH" 2>/dev/null; REFRESH=
quiet off
ql_phase steady
scp -q -r "node0:$QRUN0/out/." "$QOUT/" 2>/dev/null
cleanup
trap - EXIT
python3 "$QTOOLS/run_manifest.py" "$QOUT/manifest.json" --deploy "${QDEPLOY//\$HOME/$HOME}" --conf "$QCONF1" \
  --resolved "$QOUT/resolved-rank0.json" --resolved "$QOUT/resolved-rank1.json" --prompt-ids "$QOUT/prompt_ids.json" \
  --note "qual_bench.sh $MODE $RID; conditions.json: prompt set, client hashes, box state at the start"
python3 "$QTOOLS/memledger.py" summary "$QOUT/mem-node0.jsonl" 104.6 > "$QOUT/mem-node0.summary.json"; m0=$?
python3 "$QTOOLS/memledger.py" summary "$QOUT/mem-node1.jsonl" 104.6 > "$QOUT/mem-node1.summary.json"; m1=$?
python3 - "$QOUT" "$MODE" "$crc" "$secs" "$m0" "$m1" "${SOAK_S:-900}" <<'PY'
import glob, json, os, sys
out, mode, crc, secs, m0, m1, soak_s = sys.argv[1:8]
j = lambda p: json.load(open(p)) if os.path.exists(p) else None
man = j(os.path.join(out, "manifest.json")) or {}
doc = {"schema": f"tf-port-{mode}/1", "deploy_commit": (man.get("deploy") or {}).get("commit"),
       "config_sha256": (man.get("config") or {}).get("sha256") or man.get("config_sha256"), "manifest_complete": man.get("complete"),
       "manifest": "manifest.json", "client_exit": int(crc), "seconds": int(secs),
       "memory": {"node0": j(os.path.join(out, "mem-node0.summary.json")), "node1": j(os.path.join(out, "mem-node1.summary.json"))}}
reasons = []
if not man.get("complete"):
    reasons.append("the run manifest is incomplete (manifest.json: binaries, kit objects, bundle hash, both ranks)")
if crc != "0":
    reasons.append(f"client exit {crc} (errors in the requests)")
if mode == "bench":
    b = [j(p) for p in glob.glob(os.path.join(out, "bench-*.json"))]
    b = b[0] if b else None
    doc["summary"] = b and b.get("summary")
    if not b:
        reasons.append("no benchmark output")
    elif b["reps"] < 5:
        reasons.append(f"{b['reps']} repetitions (< 5)")
else:
    s = j(os.path.join(out, "soak-summary.json"))
    doc["summary"] = s
    net = {}
    for box in ("node0", "node1"):
        a, z = j(os.path.join(out, f"net-before-{box}.json")), j(os.path.join(out, f"net-after-{box}.json"))
        if a and z:
            net[box] = {dev: {k: z[dev][k] - a[dev][k] for k in z[dev] if isinstance(z[dev].get(k), int) and isinstance(a[dev].get(k), int)}
                        for dev in z if dev in a}
    doc["network_deltas"] = net
    if not s or s["seconds"] < int(soak_s) - 5:
        reasons.append("the soak did not reach its duration")
    if m0 != "0" or m1 != "0":
        reasons.append("memory ledger check failed (see memory)")
doc["status"] = "pass" if not reasons else "fail"
doc["reason"] = "; ".join(reasons) or ("benchmark complete, no request errors" if mode == "bench" else "soak complete: no request errors, memory checks pass")
json.dump(doc, open(os.path.join(out, f"{mode}.json"), "w"), indent=1)
print(json.dumps({"status": doc["status"], "reason": doc["reason"]}))
PY

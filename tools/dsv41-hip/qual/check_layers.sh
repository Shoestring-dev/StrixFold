#!/bin/bash
# check_layers.sh CONF TAG [SET]: the independent teacher-forced trace gate (handoff section 7; gates M3.trace-target,
# M2.layer-classes) on a deployment: the layer gate (tf-dsv41-layer, both ranks over USB4, every layer anchored on the
# reference's own layer input) against w-ref's FP64 fixture set SET (default gpufix2-p160-rounds: P=160, which emits
# every compressed-cache class, then decode rounds of 1, 1, 6 (the verify window) and 1 rows on the reference's greedy
# tokens), graded on node0 by w-ref's gpufmt/grade.py under numerics.json (FP64 router re-evaluation with --pack).
# CONF supplies the deployment and the engine settings the trace runs with (GPU_STRICT, AUX): the advertised ones.
# One run under both GPU locks (node0's first); the grading runs afterwards as capped background CPU work on node0.
# Writes ~/tfp/qual/out/trace-TAG/trace.json (status pass only if the grader says PASS for every step, the rounds
# include the verify window and both ranks exited 0) for qualify.sh's trace step.
# LAYER_EXTRA: more tf-dsv41-layer arguments for both ranks, appended to the gate's (e.g. "--side 1" or "--aio 1":
# the gate reads no environment for them); recorded in trace.json.
set -uo pipefail
QCONF=${1:?CONF}; TAG=${2:?TAG}; SET=${3:-gpufix2-p160-rounds}
QRUN=trace-$TAG
QLOCK_MAX_S=${QLOCK_MAX_S:-1200}   # both ranks' load and 40 layers x 5 steps
source "$(dirname "$0")/qual_lib.sh"
ql_init
GPORT=${GPORT:-29660}
B1=$HOME/models/tf-port/inputs-tp2-c178a012
D1=${QDEPLOY//\$HOME/$HOME}
(source "$QCONF1"; echo "$GPU_STRICT" "$AUX" "$KIT_SUBDIR" "$TP_STRIPE") > /tmp/cl-$$.txt
read -r STRICT AUXV KITSUB STRIPE < /tmp/cl-$$.txt; rm -f /tmp/cl-$$.txt
G1=$HOME/tfp/qual/trace/$TAG; G0="\$HOME/tfp/qual/trace/$TAG"
FIX1=$HOME/tfp/run/gate/fix/$SET
[ -d "$FIX1" ] || { echo "no rank-1 fixture copy $FIX1" >&2; exit 2; }
ssh -n node0 "test -d \$HOME/tfp/ref/out/$SET" || { echo "no fixture set $SET on node0" >&2; exit 2; }
mkdir -p "$G1"; ssh -n node0 "mkdir -p $G0"

layer_cmd() {  # layer_cmd RANK FIXDIR GATEDIR BOXHOME: the rank's command line (paths in that box's terms)
  local r=$1 fix=$2 g=$3 h=$4 dep=${QDEPLOY//\$HOME/$4}
  local b=$h/models/tf-port/inputs-tp2-c178a012
  echo "env TF_TP_STRIPE=$STRIPE TF_HIP_DEVICE=0 TF_GPU_STRICT=$STRICT TF_DS_AUX=$AUXV" \
    "$dep/native-hip/bin/tf-dsv41-layer $h/models/tf-port/libertai-2.9bpw $b/rank$r/rank-cache $r 2 10.40.0.1 $GPORT $fix" \
    "--layers 99 --chunks 1 --check-from 0 --rounds 4 --engram $h/models/tf-port/ds1017-1bae0e10 --token-map $b/dsv41_token_map.json" \
    "--dump $g/dump --anchor 1 --keep-going 1 --hip-kit $dep/$KITSUB --kit-dir $b/kit ${LAYER_EXTRA:-}"
}

cleanup() { set +e; [ -n "${P1:-}" ] && kill "$P1" 2>/dev/null; ssh -n node0 "[ -s $G0/rank0.pid ] && kill \$(cat $G0/rank0.pid) 2>/dev/null"; ql_unlock; }
trap cleanup EXIT
ql_lock || exit 3
C1=$(layer_cmd 1 "$FIX1" "$G1" "$HOME")
C0=$(layer_cmd 0 "$QHOME0/tfp/ref/out/$SET" "$QHOME0/tfp/qual/trace/$TAG" "$QHOME0")
echo "$C1" > "$G1/rank1.cmd"; ssh -n node0 "cat > $G0/rank0.cmd" <<< "$C0"
ql_log "trace $TAG: set $SET, deploy $QDEPLOY, strict $STRICT, aux $AUXV, extra '${LAYER_EXTRA:-}'"
t0=$(date +%s)
(cd "$G1" && exec $C1 > rank1.jsonl 2> rank1.err 9>&-) &
P1=$!
sleep 1
ssh -n node0 "cd $G0 && setsid -f sh -c 'echo \$\$ > rank0.pid; exec $C0 > rank0.jsonl 2> rank0.err' </dev/null >/dev/null 2>&1"
wait "$P1"; rc1=$?
P1=
for _ in $(seq 1 600); do ssh -n node0 "[ -s $G0/rank0.pid ] && kill -0 \$(cat $G0/rank0.pid) 2>/dev/null" || break; sleep 2; done
rc0=$(ssh -n node0 "tail -c 300 $G0/rank0.jsonl | grep -o '\"all_equal\": [a-z]*' | tail -1")
secs=$(( $(date +%s) - t0 ))
ql_unlock
trap - EXIT
ql_log "layer gate done in ${secs}s: rank 1 exit $rc1; rank 0 $rc0"

# rank 1's dump next to rank 0's on node0 (the grader reads DUMP/rank*/layers.jsonl and DUMP/layers/)
ssh -n node0 "mkdir -p $G0/dump"
scp -q -r "$G1/dump/." "node0:tfp/qual/trace/$TAG/dump/"
# the grade: w-ref's grader and numerics.json as they are on node0 (recorded by hash), capped background CPU work
ssh -n node0 "cd \$HOME/tfp/ref && sha256sum numerics.json gpufmt/grade.py > $G0/grader.sha256 && git -C \$HOME/tfp/ref rev-parse HEAD >> $G0/grader.sha256 &&
  systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryHigh=12G -p MemoryMax=16G taskset -c 8-15,24-31 nice -n 19 ionice -c3 \
  \$HOME/venvs/ref/bin/python gpufmt/grade.py out/$SET $G0/dump --stream bf16 --pack data/libertai-2.9bpw --json $G0/grade.json > $G0/grade.txt 2>&1"
grc=$?
OUT=$HOME/tfp/qual/out/trace-$TAG; mkdir -p "$OUT"
scp -q "node0:tfp/qual/trace/$TAG/grade.json" "node0:tfp/qual/trace/$TAG/grade.txt" "node0:tfp/qual/trace/$TAG/grader.sha256" \
  "node0:tfp/qual/trace/$TAG/rank0.jsonl" "node0:tfp/qual/trace/$TAG/rank0.err" "$OUT/" 2>/dev/null
cp "$G1/rank1.jsonl" "$G1/rank1.err" "$G1/rank1.cmd" "$OUT/" 2>/dev/null
scp -q "node0:tfp/qual/trace/$TAG/rank0.cmd" "$OUT/rank0.cmd" 2>/dev/null
python3 "$QTOOLS/run_manifest.py" "$OUT/manifest.json" --deploy "$D1" --conf "$QCONF1" --rank-cmd "0=$OUT/rank0.cmd" --rank-cmd "1=$OUT/rank1.cmd" \
  --note "check_layers.sh $TAG $SET; grader hashes in grader.sha256"
LAYER_EXTRA=${LAYER_EXTRA:-} python3 "$QTOOLS/trace_verdict.py" "$OUT" "$SET" "$QCONF1" "$D1" "$rc1" "$grc" "$secs"

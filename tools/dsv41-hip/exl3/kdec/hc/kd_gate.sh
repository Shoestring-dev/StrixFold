#!/bin/bash
# kd_gate.sh TAG DEPLOY FUSE [SET]: the two-rank layer gate (a-arch's gate_rank.sh flow: tf-dsv41-layer on w-ref's v2
# fixtures, rank 0 node0, rank 1 node1, tcp_coll on both links, TF_GPU_STRICT=1, TF_DS_AUX=hip, --rounds 4 --anchor 1
# --keep-going 1, every checked point dumped) of DEPLOY (a dir under ~/tfp) with TF_HC_FUSE=FUSE. Both GPU locks
# (node0 first, then node1 within 60 s, else neither and retry), released when both ranks end. Output:
# ~/tfp/kdec/gate/TAG/rank<R>.{cmd,jsonl,err} and dump/ on each box. Run on node1.
set -uo pipefail
TAG=$1 D=$2 FUSE=$3 SET=${4:-gpufix2-p24-rounds}
K=$HOME/tfp/kdec
H0=$(ssh -n node0 'echo $HOME')
L1=$K/gate/$TAG.lock.pid L0=$H0/tfp/kdec/gate/$TAG.lock.pid
mkdir -p $K/gate; ssh -n node0 "mkdir -p ~/tfp/kdec/gate"
unlock() { [ -s "$L1" ] && kill "$(cat "$L1")" 2>/dev/null; rm -f "$L1"; ssh -n node0 "[ -s $L0 ] && kill \$(cat $L0) 2>/dev/null; rm -f $L0"; }
take() {
  for try in $(seq 1 20); do
    ssh -n node0 "rm -f $L0; setsid -f flock -w 1800 /var/tmp/halo-gpu.lock sh -c 'echo \$\$ > $L0; exec sleep 1800' </dev/null >/dev/null 2>&1;
      for i in \$(seq 1 1800); do [ -s $L0 ] && exit 0; sleep 1; done; exit 1" || { unlock; return 1; }
    rm -f "$L1"; setsid -f flock -w 60 /var/tmp/halo-gpu.lock sh -c "echo \$\$ > $L1; exec sleep 1800" </dev/null >/dev/null 2>&1
    for i in $(seq 1 60); do [ -s "$L1" ] && return 0; sleep 1; done
    unlock; sleep 30
  done
  return 1
}
take || { echo "no locks"; exit 3; }
trap unlock EXIT
run_rank() {  # RANK
  local R=$1 B='$HOME/models/tf-port/inputs-tp2-c178a012'
  local FIX; [ "$R" = 0 ] && FIX='$HOME/tfp/ref/out/'$SET || FIX='$HOME/tfp/run/gate/fix/'$SET
  local O='$HOME/tfp/kdec/gate/'$TAG DD='$HOME/tfp/'$D
  local CMD="mkdir -p $O; echo \"\$(date -u +%FT%TZ) \$(hostname) rank $R set $SET deploy \$(readlink -f $DD) fuse $FUSE\" > $O/rank$R.cmd;
    timeout -k 30 900 env TF_TP_STRIPE=10.40.1.1 TF_HIP_DEVICE=0 TF_GPU_STRICT=1 TF_DS_AUX=hip TF_HC_FUSE=$FUSE \
    $DD/native-hip/bin/tf-dsv41-layer \$HOME/models/tf-port/libertai-2.9bpw $B/rank$R/rank-cache $R 2 10.40.0.1 29640 $FIX \
    --layers 99 --chunks 1 --check-from 0 --rounds 4 --engram \$HOME/models/tf-port/ds1017-1bae0e10 --token-map $B/dsv41_token_map.json \
    --dump $O/dump --anchor 1 --keep-going 1 --hip-kit $DD/native-hip/share/tensorfold/hip/gfx1151/dsv41 --kit-dir $B/kit \
    > $O/rank$R.jsonl 2> $O/rank$R.err 9>&-; echo exit=\$? >> $O/rank$R.cmd"
  if [ "$R" = 0 ]; then ssh -n node0 "$CMD"; else bash -c "$CMD"; fi
}
run_rank 1 & r1=$!
run_rank 0
wait $r1
unlock; trap - EXIT
O=$K/gate/$TAG
tail -1 $O/rank1.cmd; ssh -n node0 "tail -1 ~/tfp/kdec/gate/$TAG/rank0.cmd"
echo "points rank1 $(wc -l < $O/dump/rank1/layers.jsonl 2>/dev/null) rank0 $(ssh -n node0 "wc -l < ~/tfp/kdec/gate/$TAG/dump/rank0/layers.jsonl" 2>/dev/null)"

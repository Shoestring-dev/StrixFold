#!/bin/bash
# runq.sh NAME CMD...: one GPU run through gpu.sh: results/NAME.jsonl, logs/NAME.err, a line in logs/runq.log; a GPU
# lock wait that times out (gpu.sh exit 75) is retried. While ~/tfp/kdec/PAUSE exists (a served run of mine needs both
# boxes' locks), no new run starts.
cd ~/tfp/kdec
name=$1; shift
while [ -e PAUSE ]; do sleep 10; done
echo "$(date -Is) start $name: $*" >> logs/runq.log
while :; do
  tools/gpu.sh "$@" > "results/$name.jsonl" 2> "logs/$name.err"
  rc=$?
  [ $rc -ne 75 ] && break
  echo "$(date -Is) lock wait timed out for $name; retrying" >> logs/runq.log
done
echo "$(date -Is) done $name rc=$rc" >> logs/runq.log
exit $rc

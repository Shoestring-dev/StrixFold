#!/bin/bash
# calibrate.sh CONF RUN_ID [ROWS...]: the measured AMD DSpark cost profile (calibrate.py; gate M4.cost-profile). CONF is
# a drafts-on configuration ('$HOME/...', the same file on both boxes). For each R (default 1 2 3 4 5 6) one run under both
# GPU locks (node0's first, released after it): a variant of CONF whose ROUND_MS is flat for 1 .. R rows and prohibitive
# above (DRAFT_MS 0), so every round verifies R rows; both ranks started; calibrate.py client inside quiet windows on both
# boxes; the ranks stopped. Then calibrate.py fit -> ~/tfp/qual/out/cal-RUN_ID/profile.json (ROUND_MS and DRAFT_MS to
# put in serve.conf) with each run's manifest.json. Nothing is left running on any exit path.
set -uo pipefail
QCONF_BASE=${1:?CONF}; RID=${2:?RUN_ID}; shift 2
ROWS=("$@"); [ ${#ROWS[@]} -gt 0 ] || ROWS=(1 2 3 4 5 6)
QDIR=$(cd "$(dirname "$0")" && pwd)
OUTROOT=$HOME/tfp/qual/out/cal-$RID
mkdir -p "$OUTROOT"
export QOUT_ROOT=$OUTROOT
export QLOCK_MAX_S=${QLOCK_MAX_S:-600}
dirs=()
for R in "${ROWS[@]}"; do
  flat=$(python3 -c "import sys; R=int(sys.argv[1]); print(','.join(['10'] * R + ['1000000000']))" "$R")
  base1=${QCONF_BASE//\$HOME/$HOME}
  python3 "$QDIR/mkconf.py" "$base1" "$HOME/tfp/qual/conf/q-cal-$R.conf" "ROUND_MS=$flat" "DRAFT_MS=0" || exit 1
  ssh -n node0 "mkdir -p ~/tfp/qual/conf"; scp -q "$HOME/tfp/qual/conf/q-cal-$R.conf" "node0:tfp/qual/conf/q-cal-$R.conf"
  QCONF="\$HOME/tfp/qual/conf/q-cal-$R.conf"; QRUN=cal-$RID-r$R
  # shellcheck source=qual_lib.sh
  source "$QDIR/qual_lib.sh"
  ql_init
  scp -q "$QDIR/calibrate.py" node0:tfp/qual/bin/
  QTAG=q-qual-cal-$RID-$R
  trap 'set +e; ql_quiet off "$QTAG" >/dev/null; ql_stop; ql_copy_logs; ql_unlock' EXIT
  ql_lock || { echo "R=$R: locks not free" >&2; trap - EXIT; continue; }
  if ql_start; then
    ql_quiet on "$QTAG" >/dev/null; sleep 5
    ssh -n node0 "python3 ~/tfp/qual/bin/calibrate.py client http://127.0.0.1:$QPORT $QRUN0/calibrate.json --rows $R --model $QNAME" | tee "$QOUT/client.log"
    ql_quiet off "$QTAG" >/dev/null
    scp -q "node0:$QRUN0/calibrate.json" "$QOUT/" && dirs+=("$QOUT")
  elif ql_env_refusal; then
    echo "R=$R: node0 not idle (rank 0 refused to load); not measured" >&2
  fi
  ql_stop; ql_copy_logs; ql_unlock
  trap - EXIT
  python3 "$QDIR/run_manifest.py" "$QOUT/manifest.json" --deploy "${QDEPLOY//\$HOME/$HOME}" --conf "$QCONF1" \
    --resolved "$QOUT/resolved-rank0.json" --resolved "$QOUT/resolved-rank1.json" --note "calibrate.sh R=$R" > /dev/null
done
python3 "$QDIR/calibrate.py" fit "$OUTROOT/profile.json" "${dirs[@]}"

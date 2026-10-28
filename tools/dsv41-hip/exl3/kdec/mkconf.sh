#!/bin/bash
# mkconf.sh NAME DEPLOY [KEY=VALUE...]: ~/tfp/kdec/conf/NAME.conf on both boxes = DEPLOY's serve.conf (DEPLOY: a directory
# under ~/tfp, e.g. deploy-kdec/840f280abcde or deploy/current) with DEPLOY_DIR = that deployment and the given values
# replaced (after a-arch's mkconf.sh). Run on node1.
set -euo pipefail
NAME=$1 D=$2; shift 2
SRC=$HOME/tfp/$D/tools/dsv41-hip/serve.conf
OUT=$HOME/tfp/kdec/conf/$NAME.conf
python3 - "$SRC" "$OUT" "DEPLOY_DIR='\$HOME/tfp/$D'" "$@" <<'PY'
import re, sys
src, out, *sets = sys.argv[1:]
sets = dict(a.split("=", 1) for a in sets)
lines = []
for line in open(src):
    m = re.match(r"^([A-Z0-9_]+)=", line)
    if m and m.group(1) in sets:
        line = f"{m.group(1)}={sets.pop(m.group(1))}   # k-dec run conf\n"
    lines.append(line)
assert not sets, f"not in serve.conf: {sets}"
open(out, "w").write("".join(lines))
PY
scp -q "$OUT" "node0:tfp/kdec/conf/$NAME.conf"
echo "$OUT (both boxes): $(sha256sum "$OUT" | cut -c1-12)"

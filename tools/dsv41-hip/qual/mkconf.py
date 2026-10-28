#!/usr/bin/env python3
"""mkconf.py BASE OUT KEY=VALUE...: a copy of a serve.conf-format configuration with the given values replaced (comments
kept, every key must already exist), for the qualification's configuration variants."""
import os
import re
import sys

base, out, sets = sys.argv[1], sys.argv[2], dict(a.split("=", 1) for a in sys.argv[3:])
lines = []
for line in open(base):
    m = re.match(r"^([A-Z0-9_]+)=", line)
    if m and m.group(1) in sets:
        line = f"{m.group(1)}={sets.pop(m.group(1))}   # qualification variant ({os.path.basename(out)})\n"
    lines.append(line)
if sets:
    sys.exit(f"not in {base}: {sorted(sets)}")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
open(out, "w").write("".join(lines))

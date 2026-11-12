#!/usr/bin/env python3
"""mkconf.py BASE OUT KEY=VALUE...: a copy of a serve.conf-format configuration with the given values replaced (comments
kept, every key must already exist, active or commented out), for the qualification's configuration variants."""
import os
import re
import sys

base, out, sets = sys.argv[1], sys.argv[2], dict(a.split("=", 1) for a in sys.argv[3:])
lines = list(open(base))
# an active assignment first, else a commented-out one (an optional key such as POOL ships commented)
for pattern in (r"^([A-Z0-9_]+)=", r"^#([A-Z0-9_]+)="):
    for i, line in enumerate(lines):
        m = re.match(pattern, line)
        if m and m.group(1) in sets:
            lines[i] = f"{m.group(1)}={sets.pop(m.group(1))}   # qualification variant ({os.path.basename(out)})\n"
if sets:
    sys.exit(f"not in {base}: {sorted(sets)}")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
open(out, "w").write("".join(lines))

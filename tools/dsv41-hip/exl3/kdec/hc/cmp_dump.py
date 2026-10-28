import json, sys
def load(p):
    d = {}
    for l in open(p):
        if l.startswith("{"):
            x = json.loads(l)
            d[x.get("key") or json.dumps(x, sort_keys=True)[:200]] = x.get("sha256")
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
same = sum(1 for k in a if k in b and a[k] == b[k])
diff = [k for k in a if k in b and a[k] != b[k]]
print(f"points {len(a)} / {len(b)}, identical {same}, different {len(diff)}, only in one {len(set(a) ^ set(b))}" + (f"; first different: {diff[:3]}" if diff else ""))

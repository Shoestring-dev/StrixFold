#!/usr/bin/env python3
"""rdma_check.py: the RDMA preconditions of a rank with serve.conf TRANSPORT=rdma (rdma_proxy.zig), one box.

  rdma_check.py [--dev IBDEV | --ifname IFNAME] [--gid N] [--memlock-mib N] [--memlock-source shell|unit] [--json]

Checks (exit 1 on any failure, every check printed):
  device   the ibdev (--dev), the one whose device/net carries --ifname, or the box's only one (/sys/class/infiniband);
           a box with more than one ibdev and neither flag is a failure (the engine refuses it too)
  port     ports/1 state ACTIVE and phys_state LinkUp (rate and the PCIe link speed/width recorded beside it)
  gid      ports/1/gids/N is set (not all zeros) and its type is RoCE v2 (the rail address: TF_RDMA_GID_INDEX)
  memlock  RLIMIT_MEMLOCK soft >= --memlock-mib (default 256; the slab needs 2 x TF_COLL_SLOTS x TF_COLL_SLOT_KIB):
           --memlock-source shell (this process's limit) or unit (the user manager's DefaultLimitMEMLOCK, what a
           rank started as a transient user unit gets)
  uverbs   the device's /dev/infiniband/uverbsN is writable by this user
preflight.sh records the same inventory (its "rdma" section) and `preflight.sh --rdma-check ...` runs this.
"""
import glob, json, os, resource, subprocess, sys

IB = "/sys/class/infiniband"


def rd(p, default=None):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def pick_device(dev=None, ifname=None):
    names = sorted(os.path.basename(p) for p in glob.glob(IB + "/*"))
    if dev:
        return (dev, None) if dev in names else (None, f"no rdma device {dev} among {names or 'none'}")
    if ifname:
        for n in names:
            if ifname in os.listdir(f"{IB}/{n}/device/net") if os.path.isdir(f"{IB}/{n}/device/net") else []:
                return n, None
        return None, f"no rdma device carries interface {ifname} among {names or 'none'}"
    if len(names) == 1:
        return names[0], None
    return None, f"{len(names)} rdma devices and neither --dev nor --ifname: {names}"


def memlock_bytes(source):
    if source == "unit":
        v = subprocess.run(["systemctl", "--user", "show", "-p", "DefaultLimitMEMLOCK", "--value"], capture_output=True, text=True).stdout.strip()
        if v in ("", "infinity"):
            return None if v == "" else resource.RLIM_INFINITY, v or "(no user manager)"
        return int(v), v
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return soft, str(soft)


def inventory(dev=None, ifname=None, gid=3, memlock_mib=256, memlock_source="shell"):
    r = {"devices": sorted(os.path.basename(p) for p in glob.glob(IB + "/*")), "checks": {}, "ok": True}
    name, err = pick_device(dev, ifname)
    r["device"] = name
    r["checks"]["device"] = {"ok": name is not None, "detail": err or name}
    if name:
        d = f"{IB}/{name}"
        state, phys, rate = rd(f"{d}/ports/1/state", ""), rd(f"{d}/ports/1/phys_state", ""), rd(f"{d}/ports/1/rate", "")
        r["port"] = {"state": state, "phys_state": phys, "rate": rate, "netdev": sorted(os.listdir(f"{d}/device/net")) if os.path.isdir(f"{d}/device/net") else [],
                     "pcie_link": f"{rd(d + '/device/current_link_speed', '?')} x{rd(d + '/device/current_link_width', '?')}"}
        r["checks"]["port"] = {"ok": state.endswith("ACTIVE") and phys.endswith("LinkUp"), "detail": f"{state}, {phys}, {rate}, pcie {r['port']['pcie_link']}"}
        g = rd(f"{d}/ports/1/gids/{gid}", "")
        gtype = rd(f"{d}/ports/1/gid_attrs/types/{gid}", "")
        r["gid"] = {"index": gid, "gid": g, "type": gtype, "ndev": rd(f"{d}/ports/1/gid_attrs/ndevs/{gid}", "")}
        r["checks"]["gid"] = {"ok": bool(g) and g.replace(":", "").strip("0") != "" and gtype == "RoCE v2", "detail": f"gid {gid} = {g or '(none)'} {gtype or '(no type)'}"}
        uverbs = None
        for u in glob.glob("/sys/class/infiniband_verbs/uverbs*"):
            if rd(u + "/ibdev") == name:
                uverbs = "/dev/infiniband/" + os.path.basename(u)
        r["uverbs"] = uverbs
        r["checks"]["uverbs"] = {"ok": uverbs is not None and os.access(uverbs, os.W_OK), "detail": f"{uverbs or '(no uverbs device)'} {'writable' if uverbs and os.access(uverbs, os.W_OK) else 'not writable'}"}
    soft, text = memlock_bytes(memlock_source)
    need = memlock_mib << 20
    ok = soft is not None and (soft == resource.RLIM_INFINITY or soft >= need)
    r["memlock"] = {"source": memlock_source, "soft_bytes": None if soft is None else (-1 if soft == resource.RLIM_INFINITY else soft), "need_mib": memlock_mib}
    r["checks"]["memlock"] = {"ok": ok, "detail": f"{memlock_source} soft {text} >= {memlock_mib} MiB" if ok else f"{memlock_source} soft {text} < {memlock_mib} MiB"}
    r["ok"] = all(c["ok"] for c in r["checks"].values())
    return r


def main(argv):
    dev = ifname = None
    gid, mib, source, as_json = 3, 256, "shell", False
    a = list(argv)
    while a:
        x = a.pop(0)
        if x == "--dev": dev = a.pop(0)
        elif x == "--ifname": ifname = a.pop(0)
        elif x == "--gid": gid = int(a.pop(0))
        elif x == "--memlock-mib": mib = int(a.pop(0))
        elif x == "--memlock-source": source = a.pop(0)
        elif x == "--json": as_json = True
        elif x in ("-h", "--help"): print(__doc__); return 0
        else: print(f"rdma_check.py: unknown argument {x}", file=sys.stderr); return 2
    r = inventory(dev, ifname, gid, mib, source)
    if as_json:
        print(json.dumps(r, indent=1))
    else:
        for k, c in r["checks"].items():
            print(f"rdma check {k}: {'ok' if c['ok'] else 'FAIL'}: {c['detail']}")
        print(f"rdma check on {os.uname().nodename}: {'ok' if r['ok'] else 'FAILED'} ({r['device'] or 'no device'})")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

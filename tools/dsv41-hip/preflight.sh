#!/bin/bash
# preflight.sh [OUT_DIR] [--baseline OLD.json] [--no-hip-probe]
# preflight.sh --diff OLD.json NEW.json
#
# Host, toolchain, memory, disk, network and power inventory for the TensorFold DeepSeek -> Strix Halo port
# (handoff section 4). Writes OUT_DIR/preflight-<host>.json and OUT_DIR/raw-<host>/ (every command's raw output).
# Default OUT_DIR: ~/tfp/manifest/preflight. The JSON records this script's path, SHA-256 and git commit
# (provenance.py; on a deploy_rank0.sh copy, the DEPLOY.json source commit).
#
# Read-only except for two bounded probes:
# - O_DIRECT: aligned 16 MiB O_DIRECT reads of a rank-cache extent and of an Engram shard extent (btrfs), into a
#   pre-faulted and into a fresh buffer, each checked against a buffered read and for page-cache residency (mincore),
#   plus one misaligned O_DIRECT read;
# - HIP (skip with --no-hip-probe): hip_probe.hip, built with hipcc for gfx1151, run under the GPU lock: device
#   properties, hipMemGetInfo, and one allocation of PREFLIGHT_PROBE_BYTES (default the plan's largest single device
#   allocation, 2,559,098,880 B) written and checked word for word, with a 64 MiB host round trip; then freed.
# The GPU commands (rocminfo, the HIP probe) wait up to PREFLIGHT_LOCK_WAIT seconds (default 120) for
# /var/tmp/halo-gpu.lock; if it stays busy they are recorded as skipped and the ISA comes from KFD sysfs.
# No installs, no sysctl/governor writes. `sudo -n` only for read-only queries, skipped when it would prompt.
#
# --baseline OLD.json embeds a drift report against an earlier preflight (packages, holds, kernel, BIOS, toolchain
# versions and binary hashes); --diff OLD NEW prints that report for two existing files and exits 1 on any drift.
# Run it at every milestone review and diff against the previous milestone's JSON.
set -u
SCRIPT=$(readlink -f "$0")
HERE=$(dirname "$SCRIPT")
OUT=$HOME/tfp/manifest/preflight BASELINE= DIFF_OLD= DIFF_NEW= HIP_PROBE=1
while [ $# -gt 0 ]; do
  case $1 in
    --baseline) BASELINE=$(readlink -f "$2"); shift 2 ;;
    --diff) DIFF_OLD=$2 DIFF_NEW=$3; shift 3 ;;
    --no-hip-probe) HIP_PROBE=0; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    -*) echo "preflight.sh: unknown option $1" >&2; exit 2 ;;
    *) OUT=$1; shift ;;
  esac
done
export SCRIPT HERE BASELINE DIFF_OLD DIFF_NEW HIP_PROBE PREFLIGHT_LOCK_WAIT=${PREFLIGHT_LOCK_WAIT:-120}
export PREFLIGHT_PROBE_BYTES=${PREFLIGHT_PROBE_BYTES:-2559098880}
if [ -n "$DIFF_OLD" ]; then exec python3 - <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["HERE"])
from preflight_drift import drift
d = drift(json.load(open(os.environ["DIFF_OLD"])), json.load(open(os.environ["DIFF_NEW"])))
print(json.dumps(d, indent=1))
sys.exit(1 if d["changed"] else 0)
PY
fi
H=$(hostname)
RAW=$OUT/raw-$H
mkdir -p "$RAW"
export OUT RAW H
exec python3 - <<'PY'
import glob, json, os, re, shutil, subprocess, sys, time

OUT, RAW, H, HERE = os.environ["OUT"], os.environ["RAW"], os.environ["H"], os.environ["HERE"]
LOCK_WAIT = os.environ["PREFLIGHT_LOCK_WAIT"]
sys.path.insert(0, HERE)
from provenance import record, sha256_file  # noqa: E402
R = {"schema": "tf-port-preflight/2", "host": H, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "script": os.environ["SCRIPT"], "provenance": record(os.environ["SCRIPT"]),
     "helpers": {n: sha256_file(os.path.join(HERE, n)) for n in ("provenance.py", "preflight_drift.py", "hip_probe.hip")},
     "unavailable": [], "notes": []}

def run(name, cmd, timeout=30, shell=False):
    """Run a read-only command, keep raw stdout+stderr in RAW/name.txt, return (rc, stdout)."""
    try:
        p = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout)
        out, rc = p.stdout, p.returncode
        err = p.stderr
    except FileNotFoundError:
        out, err, rc = "", "not installed", 127
    except subprocess.TimeoutExpired:
        out, err, rc = "", f"timeout {timeout}s", 124
    with open(os.path.join(RAW, name + ".txt"), "w") as f:
        f.write(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n# rc={rc}\n{out}")
        if err: f.write(f"\n# stderr:\n{err}")
    if rc == 127: R["unavailable"].append(name)
    return rc, out

def rd(path, default=None):
    try:
        with open(path) as f: return f.read().strip()
    except Exception: return default

def num(s):
    try: return int(s)
    except Exception:
        try: return float(s)
        except Exception: return s

# ---- machine, CPU, kernel, distro
dmi = {k: rd(f"/sys/class/dmi/id/{k}") for k in ("sys_vendor", "product_name", "product_version", "board_vendor", "board_name", "bios_vendor", "bios_version", "bios_date")}
rc, lscpu = run("lscpu", ["lscpu", "-J"])
cpu = {}
if rc == 0:
    for e in json.loads(lscpu)["lscpu"]:
        cpu[e["field"].rstrip(":")] = e["data"]
osr = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', rd("/etc/os-release", ""), re.M))
R["machine"] = {"dmi": dmi, "arch": os.uname().machine}
R["cpu"] = {k: cpu.get(k) for k in ("Model name", "Architecture", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "CPU max MHz", "CPU min MHz", "NUMA node(s)", "L3", "Flags") if k in cpu}
if "Flags" in R["cpu"]:
    fl = R["cpu"].pop("Flags").split()
    R["cpu"]["flags_of_interest"] = [f for f in ("avx2", "avx512f", "avx512_bf16", "avx512_fp16", "avx512vnni", "sha_ni", "vaes") if f in fl]
R["kernel"] = {"uname": " ".join(os.uname()), "cmdline": rd("/proc/cmdline"), "release": os.uname().release}
R["distro"] = {k: osr.get(k) for k in ("PRETTY_NAME", "ID", "VERSION_ID", "VERSION_CODENAME", "ID_LIKE")}
R["uptime_s"] = num(rd("/proc/uptime", "0 0").split()[0])

# ---- memory, limits, cgroups
mi = {}
for line in rd("/proc/meminfo", "").splitlines():
    k, v = line.split(":", 1)
    v = v.split()
    mi[k] = int(v[0]) * (1024 if len(v) > 1 else 1)
R["memory"] = {"meminfo_bytes": {k: mi.get(k) for k in ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal", "SwapFree", "Shmem", "Unevictable", "Mlocked", "HugePages_Total", "Hugepagesize", "CommitLimit", "Committed_AS")},
               "swaps": rd("/proc/swaps"),
               "numa_nodes": sorted(os.path.basename(p) for p in glob.glob("/sys/devices/system/node/node[0-9]*")),
               "thp": {"enabled": rd("/sys/kernel/mm/transparent_hugepage/enabled"), "defrag": rd("/sys/kernel/mm/transparent_hugepage/defrag")},
               "sysctl": {k: rd("/proc/sys/" + k.replace(".", "/")) for k in ("vm.overcommit_memory", "vm.overcommit_ratio", "vm.swappiness", "vm.max_map_count", "vm.min_free_kbytes", "vm.dirty_ratio", "vm.dirty_background_ratio", "vm.zone_reclaim_mode", "kernel.numa_balancing")}}
rc, ul = run("ulimit", "ulimit -a", shell=True)
lim = {}
for line in rd("/proc/self/limits", "").splitlines()[1:]:
    m = re.match(r"(.+?)\s{2,}(\S+)\s+(\S+)\s*(\S*)", line)
    if m: lim[m.group(1).strip()] = {"soft": m.group(2), "hard": m.group(3), "units": m.group(4)}
cg = rd("/proc/self/cgroup", "")
cgpath = cg.split("::", 1)[1].strip() if "::" in cg else None
cgroups = {"self": cgpath}
def cgread(p):
    d = {}
    for k in ("memory.current", "memory.max", "memory.high", "memory.low", "memory.min", "memory.swap.max", "memory.swap.current", "memory.events", "memory.peak", "cpuset.cpus.effective", "cpuset.mems.effective", "pids.max"):
        v = rd(os.path.join("/sys/fs/cgroup", p.lstrip("/"), k))
        if v is not None: d[k] = v.replace("\n", "; ")
    return d
if cgpath:
    parts = cgpath.strip("/").split("/")
    cgroups["chain"] = {"/" + "/".join(parts[:i]): cgread("/".join(parts[:i])) for i in range(1, len(parts) + 1)}
uid = os.getuid()
cgroups["user_slice"] = cgread(f"user.slice/user-{uid}.slice")
cgroups["user_manager"] = cgread(f"user.slice/user-{uid}.slice/user@{uid}.service")
R["limits"] = {"process": lim, "memlock_soft": lim.get("Max locked memory", {}).get("soft"), "address_space_soft": lim.get("Max address space", {}).get("soft"), "cgroups": cgroups,
               "limits_conf_memlock": [l for l in (rd("/etc/security/limits.conf", "") + "\n" + "\n".join(rd(p, "") for p in glob.glob("/etc/security/limits.d/*"))).splitlines() if "memlock" in l and not l.lstrip().startswith("#")]}
rc, _ = run("systemd_user_defaults", "systemctl --user show -p DefaultLimitMEMLOCK -p DefaultLimitAS 2>&1; systemctl show -p DefaultLimitMEMLOCK 2>&1", shell=True)

# ---- GPU: KFD topology, amdgpu sysfs, module params, rocminfo (under the shared GPU lock)
nodes = []
for nd in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*")):
    props = dict(l.split(" ", 1) for l in (rd(nd + "/properties", "") or "").splitlines() if " " in l)
    if props.get("gfx_target_version", "0") == "0": continue
    v = int(props["gfx_target_version"])
    isa = f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}"
    banks = []
    for b in sorted(glob.glob(nd + "/mem_banks/*")):
        bp = dict(l.split(" ", 1) for l in (rd(b + "/properties", "") or "").splitlines() if " " in l)
        banks.append({k: num(bp.get(k)) for k in ("heap_type", "size_in_bytes", "flags", "width", "mem_clk_max")})
    nodes.append({"node": os.path.basename(nd), "isa_from_gfx_target_version": isa, "gfx_target_version": v,
                  **{k: num(props.get(k)) for k in ("vendor_id", "device_id", "simd_count", "array_count", "cu_per_simd_array", "simd_per_cu", "wave_front_size", "max_waves_per_simd", "lds_size_in_kb", "max_engine_clk_fcompute", "max_engine_clk_ccompute", "num_sdma_engines", "drm_render_minor", "local_mem_size")},
                  "compute_units": num(props.get("array_count", 0)) * num(props.get("cu_per_simd_array", 0)) if props.get("array_count") else None,
                  "mem_banks": banks})
cards = []
for dev in sorted(glob.glob("/sys/class/drm/card[0-9]")):
    d = dev + "/device"
    if rd(d + "/vendor") != "0x1002": continue
    c = {"card": os.path.basename(dev), "pci": os.path.basename(os.path.realpath(d)), "device_id": rd(d + "/device"), "revision": rd(d + "/revision")}
    for k in ("mem_info_vram_total", "mem_info_vram_used", "mem_info_vis_vram_total", "mem_info_gtt_total", "mem_info_gtt_used", "gpu_busy_percent", "power_dpm_force_performance_level", "power_dpm_state", "current_link_speed", "current_link_width"):
        v = rd(d + "/" + k)
        if v is not None: c[k] = num(v)
    for k in ("pp_dpm_sclk", "pp_dpm_mclk", "pp_dpm_fclk", "pp_dpm_socclk", "pp_power_profile_mode"):
        v = rd(d + "/" + k)
        if v is not None: c[k] = v.splitlines()[:12]
    hw = {}
    for hp in glob.glob(d + "/hwmon/hwmon*"):
        for f in sorted(glob.glob(hp + "/*_input")) + sorted(glob.glob(hp + "/power1_*")) + sorted(glob.glob(hp + "/freq*_input")):
            hw[os.path.basename(f)] = num(rd(f))
    c["hwmon"] = hw
    cards.append(c)
amdgpu_params = {os.path.basename(p): rd(p) for p in sorted(glob.glob("/sys/module/amdgpu/parameters/*")) if os.path.basename(p) in ("gttsize", "no_system_mem_limit", "vramlimit", "vm_size", "noretry", "mes", "cwsr_enable", "runpm", "gpu_recovery", "lockup_timeout", "ppfeaturemask", "sched_policy", "dcfeaturemask")}
ttm_params = {os.path.basename(p): rd(p) for p in sorted(glob.glob("/sys/module/ttm/parameters/*"))}
lock_cmd = f"flock -w {LOCK_WAIT} /var/tmp/halo-gpu.lock timeout 60 rocminfo"
t0 = time.time()
rc, rinfo = run("rocminfo", lock_cmd, timeout=int(LOCK_WAIT) + 90, shell=True)
rocm_agents = []
if rc == 0:
    for blk in rinfo.split("*******")[1:]:
        name = re.search(r"^\s*Name:\s+(\S+)", blk, re.M)
        if name and name.group(1).startswith("gfx"):
            g = lambda pat: (re.search(pat, blk, re.M) or [None, None])[1]
            rocm_agents.append({"name": name.group(1), "marketing_name": (g(r"Marketing Name:\s+(.+)$") or "").strip(), "compute_units": num(g(r"Compute Unit:\s+(\d+)")),
                                "wavefront_size": num(g(r"Wavefront Size:\s+(\d+)")), "max_clock_mhz": num(g(r"Max Clock Freq\. \(MHz\):\s+(\d+)")),
                                "isas": re.findall(r"Name:\s+(amdgcn-amd-amdhsa--\S+)", blk)})
gpu_users = []
for fd in glob.glob("/proc/[0-9]*/fd/*"):
    try:
        if os.readlink(fd) == "/dev/kfd":
            pid = fd.split("/")[2]
            gpu_users.append({"pid": int(pid), "comm": rd(f"/proc/{pid}/comm"), "cmd": (rd(f"/proc/{pid}/cmdline", "") or "").replace("\0", " ")[:200]})
    except OSError:
        pass
R["gpu"] = {"kfd_nodes": nodes, "drm_cards": cards, "amdgpu_params": amdgpu_params, "ttm_params": ttm_params,
            "ttm_pages_limit_bytes": int(ttm_params["pages_limit"]) * 4096 if ttm_params.get("pages_limit", "").isdigit() else None,
            "rocminfo": {"rc": rc, "cmd": lock_cmd, "seconds_incl_lock_wait": round(time.time() - t0, 1), "agents": rocm_agents,
                         "status": "ok" if rc == 0 else ("lock busy or timeout: ISA from KFD topology only" if rc in (1, 124) else f"failed rc={rc}")},
            "isa": sorted({a["name"] for a in rocm_agents} or {n["isa_from_gfx_target_version"] for n in nodes}),
            "processes_holding_kfd_visible_to_user": sorted(gpu_users, key=lambda x: x["pid"])}
devs = {}
for p in ["/dev/kfd"] + sorted(glob.glob("/dev/dri/renderD*")) + sorted(glob.glob("/dev/dri/card*")):
    try:
        st = os.stat(p)
        import grp
        devs[p] = {"mode": oct(st.st_mode & 0o777), "group": grp.getgrgid(st.st_gid).gr_name, "rw_for_user": os.access(p, os.R_OK | os.W_OK)}
    except Exception as e:
        devs[p] = {"error": str(e)}
rc, groups = run("id", ["id"])
R["gpu"]["device_permissions"] = {"devices": devs, "id": groups.strip()}

# ---- toolchain versions
def first(cmd, name, n=3):
    rc, o = run(name, cmd, shell=isinstance(cmd, str))
    return {"rc": rc, "out": "\n".join(o.strip().splitlines()[:n])} if rc != 127 else None
pk = {}
rc, dpkg = run("dpkg_rocm", "dpkg-query -W -f='${Package} ${Version}\\n' 2>/dev/null | grep -iE 'rocm|hip|rocblas|hipblas|rccl|comgr|hsa|amdgpu|llvm-2|clang-2|libdrm-amdgpu|firmware-amd|python3-torch|triton' ", shell=True)
for line in dpkg.splitlines():
    a = line.split(" ", 1)
    if len(a) == 2: pk[a[0]] = a[1]
# drift guard (review m1): the GPU stack, kernel and firmware package versions, the apt holds, the dpkg history of
# those packages, and a hash of the whole installed set; --baseline/--diff compare them between milestones
DRIFT_RE = r"rocm|hip|hsa|rocblas|hipblas|rccl|comgr|amdgpu|rocprof|rocminfo|roctracer|miopen|llvm|clang|lld|libdrm|mesa|linux-image|linux-headers|linux-modules|linux-firmware|firmware|libtorch-rocm|python3-torch|triton"
rc, allpk = run("dpkg_all", ["dpkg-query", "-W", "-f=${Package}\t${Version}\t${db:Status-Abbrev}\n"])
# installed = dpkg status "?i?": "ii", and "hi" for a held package (the GPU stack was put on hold after M0, and held
# packages then read as removed). The want flag is written as "i" so that a hold alone leaves the set hash unchanged;
# apt_holds records the holds.
installed = sorted("\t".join(l.split("\t")[:2] + ["i" + l.split("\t")[-1][1:]]) for l in allpk.splitlines()
                   if l.split("\t")[-1][1:2] == "i")
gpu_kernel_fw = {l.split("\t")[0]: l.split("\t")[1] for l in installed if re.search(DRIFT_RE, l.split("\t")[0], re.I)}
rc, holds = run("apt_mark_showhold", ["apt-mark", "showhold"])
rc, dpkg_hist = run("dpkg_log_gpu_kernel_fw", f"grep -hE ' (install|upgrade|remove) ' /var/log/dpkg.log /var/log/dpkg.log.1 2>/dev/null | grep -iE '{DRIFT_RE}' | tail -400", shell=True)
R["packages"] = {"gpu_kernel_firmware": gpu_kernel_fw, "gpu_kernel_firmware_count": len(gpu_kernel_fw),
                 "apt_holds": holds.split(), "installed_count": len(installed),
                 "installed_set_sha256": __import__("hashlib").sha256("\n".join(installed).encode()).hexdigest(),
                 "dpkg_log_last_changes": dpkg_hist.strip().splitlines()[-40:],
                 "note": "with no apt hold on the GPU stack or kernel, drift is detected, not prevented: diff this section "
                         "at every milestone review (preflight.sh --diff OLD NEW); dpkg_log_last_changes dates each install"}
libs = {}
for pat in ("librocblas.so*", "libhipblas.so*", "libhipblaslt.so*", "librccl.so*", "libamdhip64.so*", "libhsa-runtime64.so*", "libamd_comgr.so*", "libibverbs.so*"):
    hits = sorted(set(glob.glob("/opt/rocm*/lib/" + pat) + glob.glob("/opt/rocm*/*/lib/" + pat) + glob.glob("/usr/lib/x86_64-linux-gnu/" + pat)))
    libs[pat.split(".so")[0]] = [h + (" -> " + os.path.realpath(h) if os.path.islink(h) else "") for h in hits][:8]
# co-installed HIP stacks (review m1): every libamdhip64 / libhsa-runtime64 real file with its SONAME and owning
# package, and what the dynamic loader's cache resolves each SONAME to (ld.so.conf.d lists /opt/rocm/lib before the
# multiarch directory, so ROCm's copy wins for a SONAME both provide)
def soname(p):
    rc, o = run("readelf_" + os.path.basename(p), ["readelf", "-d", p])
    m = re.search(r"\(SONAME\).*\[(.+)\]", o)
    return m.group(1) if m else None
def owner(p):
    try:
        o = subprocess.run(["dpkg", "-S", p], capture_output=True, text=True).stdout.strip()
        return o.split(":")[0] if o else None
    except FileNotFoundError:
        return None
stacks = {}
for base in ("libamdhip64", "libhsa-runtime64"):
    files = sorted({os.path.realpath(h) for h in glob.glob(f"/opt/rocm*/lib/{base}.so*") + glob.glob(f"/usr/lib/x86_64-linux-gnu/{base}.so*")})
    stacks[base] = [{"file": f, "soname": soname(f), "package": owner(f), "sha256": sha256_file(f)} for f in files if os.path.isfile(f)]
rc, ldc = run("ldconfig_p", "/sbin/ldconfig -p 2>/dev/null || ldconfig -p", shell=True)
stacks["ld_cache_resolution"] = {s: [l.split("=>")[-1].strip() for l in ldc.splitlines() if l.strip().startswith(s + " ")]
                                 for s in ("libamdhip64.so.7", "libamdhip64.so.5", "libhsa-runtime64.so.1")}
stacks["ld_so_conf_d"] = sorted(os.listdir("/etc/ld.so.conf.d")) if os.path.isdir("/etc/ld.so.conf.d") else []
stacks["note"] = ("Debian's HIP 5.7 (libamdhip64.so.5) and HSA 1.13 (libhsa-runtime64.so.1) are installed beside ROCm's HIP 7.15 "
                  "(libamdhip64.so.7) and HSA 1.21, which shares the HSA SONAME. The first ld cache entry is the one ld.so "
                  "loads. The port's adapter dlopens libamdhip64.so.7 by that SONAME (or the full /opt/rocm path), never .so.5.")
# HIP header directories: only ROCm's. /usr/include/hip belongs to libtorch-rocm-dev (ATen HIP headers), not to HIP
hip_inc = sorted(set(os.path.realpath(p) for p in glob.glob("/opt/rocm*/include/hip") + glob.glob("/opt/rocm*/*/include/hip")))
other_hip_inc = {p: owner(p) for p in ("/usr/include/hip",) if os.path.isdir(p)}
# Zig 0.17.0 (review M1): the user-local install and the kit's toolchain copy, whichever exist; the first found is
# the one the build uses (STRIX_ZIG, then ~/opt/zig, then the kit's .strix-toolchain, then PATH)
zig_cands = [os.environ.get("STRIX_ZIG"), os.path.expanduser("~/opt/zig/zig")] + \
            sorted(glob.glob(os.path.expanduser("~/tfp/kit/tensorfold-strix/.strix-toolchain/zig-*/zig"))) + [shutil.which("zig")]
zigs, seen = [], set()
for z in zig_cands:
    if not z or not os.path.isfile(z) or os.path.realpath(z) in seen: continue
    seen.add(os.path.realpath(z))
    v = subprocess.run([z, "version"], capture_output=True, text=True)
    zigs.append({"path": z, "realpath": os.path.realpath(z), "version": v.stdout.strip() if v.returncode == 0 else None, "sha256": sha256_file(z)})
with open(os.path.join(RAW, "zig_version.txt"), "w") as f: json.dump(zigs, f, indent=1)
if not zigs: R["unavailable"].append("zig_version")
# compiler and runtime binaries by content, so a silent package rebuild shows up in the drift report
def which_real(c):
    w = shutil.which(c)
    return os.path.realpath(w) if w else None
rc, clangdir = run("hipclangpath", ["hipconfig", "--hipclangpath"])
binaries = {}
for label, p in (("hipcc", which_real("hipcc")), ("amdclang", os.path.join(clangdir.strip(), "clang") if clangdir.strip() else None),
                 ("ld.lld", os.path.join(clangdir.strip(), "ld.lld") if clangdir.strip() else None),
                 ("libamdhip64.so.7", os.path.realpath("/opt/rocm/lib/libamdhip64.so.7")),
                 ("libhsa-runtime64.so.1(rocm)", os.path.realpath("/opt/rocm/lib/libhsa-runtime64.so.1")),
                 ("libamd_comgr", next(iter(sorted(glob.glob("/opt/rocm/lib/libamd_comgr.so.*.*"))), None)),
                 ("rocminfo", which_real("rocminfo"))):
    if p and os.path.isfile(p):
        p = os.path.realpath(p)
        binaries[label] = {"path": p, "sha256": sha256_file(p)}
R["toolchain"] = {
    "rocm_path": os.path.realpath("/opt/rocm") if os.path.exists("/opt/rocm") else None,
    "rocm_alternatives": {p: os.path.realpath(p) for p in sorted(glob.glob("/opt/rocm/*")) if os.path.islink(p)},
    "rocm_version_files": {p: rd(p) for p in glob.glob("/opt/rocm*/.info/version*") + glob.glob("/opt/rocm*/*/.info/version*")},
    "hipconfig_version": first(["hipconfig", "--version"], "hipconfig_version", 1),
    "hipconfig_full": first(["hipconfig", "--full"], "hipconfig_full", 12),
    "hipcc_version": first(["hipcc", "--version"], "hipcc_version", 4),
    "amd_clang": first("$(hipconfig --hipclangpath 2>/dev/null)/clang --version", "amd_clang", 2),
    "device_lib_bitcode_dirs": sorted(set(os.path.realpath(p) for p in glob.glob("/opt/rocm*/amdgcn/bitcode") + glob.glob("/opt/rocm*/*/amdgcn/bitcode") + glob.glob("/opt/rocm*/lib/llvm/amdgcn/bitcode") + glob.glob("/opt/rocm*/*/lib/llvm/amdgcn/bitcode") + glob.glob("/usr/lib/llvm-*/lib/clang/*/amdgcn/bitcode"))),
    "hip_include_dirs": hip_inc, "other_hip_named_include_dirs_not_hip": other_hip_inc,
    "packages": pk, "libraries": libs, "hip_stacks": stacks, "binaries_sha256": binaries,
    "amd_smi": first(["amd-smi", "version"], "amd_smi_version", 3),
    "python3": first(["python3", "--version"], "python3_version", 1),
    "python_modules": first(["python3", "-c", "import importlib.util as u;print({m: bool(u.find_spec(m)) for m in ('torch','triton','tokenizers','numpy','safetensors','cryptography')})"], "python_modules", 1),
    "gcc": first(["gcc", "--version"], "gcc_version", 1),
    "git": first(["git", "--version"], "git_version", 1),
    "ninja": first(["ninja", "--version"], "ninja_version", 1),
    "zig": {"rc": 0, "out": zigs[0]["version"]} if zigs else "not installed",
    "zig_path": zigs[0]["path"] if zigs else None, "zig_all": zigs,
}

# ---- bounded HIP probe (handoff section 4: HIP free/total memory, a bounded allocation/copy/kernel test, support
# for the largest planned allocation). hip_probe.hip, built here for gfx1151, run under the GPU lock.
env_override = {"process_env": os.environ.get("HSA_OVERRIDE_GFX_VERSION")}
for f in ("/etc/environment", os.path.expanduser("~/.bashrc"), os.path.expanduser("~/.profile"), os.path.expanduser("~/.bash_profile")):
    env_override[f] = [l.strip() for l in (rd(f, "") or "").splitlines() if "HSA_OVERRIDE_GFX_VERSION" in l]
rc, sysd_env = run("systemd_user_environment", ["systemctl", "--user", "show-environment"])
env_override["systemd_user"] = [l for l in sysd_env.splitlines() if "HSA_OVERRIDE_GFX_VERSION" in l]
R["gpu"]["hsa_override_gfx_version"] = env_override
R["gpu"]["hsa_override_gfx_version_set"] = bool(env_override["process_env"] or any(v for k, v in env_override.items() if k != "process_env"))
probe = {"status": "not run (--no-hip-probe)"}
if os.environ.get("HIP_PROBE") == "1":
    exe = os.path.join(RAW, "hip_probe")
    src = os.path.join(HERE, "hip_probe.hip")
    rc, _ = run("hip_probe_build", ["nice", "-n", "19", "taskset", "-c", "8-15,24-31", "hipcc", "--offload-arch=gfx1151", "-O2", "-o", exe, src], timeout=300)
    if rc != 0:
        probe = {"status": f"build failed rc={rc} (see raw hip_probe_build.txt)"}
    else:
        rc, lddo = run("hip_probe_ldd", ["ldd", exe])
        nbytes = os.environ["PREFLIGHT_PROBE_BYTES"]
        cmd = f"flock -w {LOCK_WAIT} /var/tmp/halo-gpu.lock timeout 120 {exe} {nbytes}"
        t0 = time.time()
        rc, po = run("hip_probe", cmd, timeout=int(LOCK_WAIT) + 180, shell=True)
        try:
            res = json.loads(po)
        except json.JSONDecodeError:
            res = None
        probe = {"status": "pass" if rc == 0 and res and res.get("ok") else ("lock busy or timeout" if rc in (1, 124) and not res else f"fail rc={rc}"),
                 "cmd": cmd, "seconds_incl_lock_wait": round(time.time() - t0, 1), "source_sha256": sha256_file(src),
                 "binary_sha256": sha256_file(exe), "allocation_bytes_requested": int(nbytes),
                 "allocation_basis": "the plan's largest single device allocation: one layer's expert trellis buffer (weights.zig experts(), 2,559,098,880 B at K=3; memory_plan.md)",
                 "resolved_libraries": [l.strip() for l in lddo.splitlines() if re.search(r"amdhip|hsa-runtime|comgr|drm", l)],
                 "result": res}
        if res:
            probe["cu_note"] = (f"hipDeviceProp.multiProcessorCount = {res.get('multiProcessorCount')} while rocminfo/KFD report "
                                f"{(rocm_agents[0]['compute_units'] if rocm_agents else nodes[0]['compute_units'] if nodes else '?')} compute units: "
                                "on gfx1151 HIP counts WGPs (2 CUs each). Size grids and occupancy from the CU count and wave32.")
R["gpu"]["hip_probe"] = probe
R["gpu"]["hip_mem_get_info"] = (probe.get("result") or {}).get("mem_get_info") or "see hip_probe status"

# ---- disk and filesystems
rc, df = run("df", ["df", "-B1", "--output=source,fstype,size,used,avail,target", "/", os.path.expanduser("~"), os.path.expanduser("~/models")])
rc, fm = run("findmnt", ["findmnt", "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS", "/"])
rc, lb = run("lsblk", ["lsblk", "-J", "-b", "-o", "NAME,MODEL,SIZE,ROTA,TYPE,FSTYPE,MOUNTPOINTS,PHY-SEC,LOG-SEC,MIN-IO,OPT-IO,TRAN"])
nv = []
for c in sorted(glob.glob("/sys/class/nvme/nvme*")):
    nv.append({"ctrl": os.path.basename(c), "model": rd(c + "/model"), "firmware": rd(c + "/firmware_rev"), "transport": rd(c + "/transport"),
               "pcie_link": f"{rd(c + '/device/current_link_speed')} x{rd(c + '/device/current_link_width')}"})
blk = {}
for q in sorted(glob.glob("/sys/block/nvme*n*/queue")):
    dev = q.split("/")[3]
    blk[dev] = {k: rd(q + "/" + k) for k in ("logical_block_size", "physical_block_size", "dma_alignment", "max_hw_sectors_kb", "max_sectors_kb", "read_ahead_kb", "scheduler", "nr_requests", "io_poll", "rotational")}
st = os.statvfs(os.path.expanduser("~"))
rc, subv = run("btrfs_subvolumes", "sudo -n btrfs subvolume list -t / 2>&1 || echo 'sudo -n unavailable'", shell=True)
rc, snapt = run("snapshot_timers", "systemctl list-timers --all --no-pager 2>/dev/null | grep -iE 'snap|btrfs|timeshift|backup' ; ls /etc/snapper/configs 2>/dev/null; ls /etc/timeshift* 2>/dev/null; systemctl list-unit-files --no-pager 2>/dev/null | grep -iE 'snap|btrfs|timeshift'", shell=True)
rc, attrs = run("lsattr_models", "for d in ~/models ~/models/*/ ~/models/tf-port/*/; do [ -d \"$d\" ] && lsattr -d \"$d\" 2>&1; done", shell=True)
R["disk"] = {"df": df.strip().splitlines(), "root_mount": json.loads(fm)["filesystems"] if fm else None, "lsblk": json.loads(lb)["blockdevices"] if lb else None,
             "nvme": nv, "block_queues": blk, "home_statvfs": {"f_bsize": st.f_bsize, "f_frsize": st.f_frsize, "avail_bytes": st.f_bavail * st.f_frsize},
             "btrfs_subvolumes": subv.strip().splitlines()[:40], "snapshot_services": snapt.strip().splitlines(), "lsattr_models": attrs.strip().splitlines(),
             "direct_io_note": "btrfs: O_DIRECT is honoured on non-compressed extents; compressed extents and misaligned requests fall back to buffered I/O silently. Alignment = logical_block_size of the NVMe (see block_queues); see odirect_probe for the measured behaviour."}

# ---- O_DIRECT probe (review m1): the loader reads expert spans and Engram AIO reads table rows with O_DIRECT. For a
# rank-cache extent and an Engram shard extent: drop the range from the page cache (fadvise DONTNEED: clean pages
# only), read 16 MiB with O_DIRECT, check with mincore whether the range became resident (i.e. the read went through the
# page cache) and compare the bytes with a buffered read. Done twice: into a pre-faulted buffer (how a loader with a
# reused staging buffer reads) and into a fresh, never-touched mapping. Found 2026-10-19 on kernel 7.2.7 btrfs: the
# pre-faulted read bypasses the cache; the fresh one is served through it (plus readahead). Then one misaligned O_DIRECT
# read. Bounded: 2 x 2 x 16 MiB of reads per file.
import ctypes, ctypes.util, errno, hashlib, mmap
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype, libc.mmap.argtypes = ctypes.c_void_p, (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
def resident_pages(path, off, length):
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, off)
        if addr in (None, ctypes.c_void_p(-1).value): return None
        vec = (ctypes.c_ubyte * ((length + 4095) // 4096))()
        rc = libc.mincore(addr, length, vec)
        libc.munmap(addr, length)
        return sum(v & 1 for v in vec) if rc == 0 else None
    finally:
        os.close(fd)
def odirect_read(path, off, length, prefault):
    r = {"offset": off, "length": length, "buffer": "pre-faulted" if prefault else "fresh (never touched)"}
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    r["resident_pages_before"] = before = resident_pages(path, off, length)
    buf = mmap.mmap(-1, length)
    if prefault: buf.write(bytes(length))
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        n = os.preadv(fd, [buf], off)
    finally:
        os.close(fd)
    r["resident_pages_after"] = after = resident_pages(path, off, length)
    with open(path, "rb") as f:
        f.seek(off)
        ref = f.read(length)
    r["read_bytes"], r["matches_buffered_read"] = n, n == length and hashlib.sha256(buf[:n]).digest() == hashlib.sha256(ref).digest()
    r["bypassed_page_cache"] = (after == 0) if before == 0 and after is not None else "inconclusive (range still cached before the read)"
    return r, buf
def odirect_probe(path, length=16 << 20, align=4096):
    r = {"file": path}
    try:
        size = os.path.getsize(path)
        rc, la = run("lsattr_" + os.path.basename(path), ["lsattr", path])
        off = (size // 2) // (1 << 20) * (1 << 20)
        r.update(bytes=size, lsattr=(la.split() or [None])[0], alignment=align)
        r["prefaulted"], buf = odirect_read(path, off, length, True)
        r["fresh"], _ = odirect_read(path, off + 4 * length, length, False)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            os.preadv(fd, [memoryview(buf)[:align]], off + 1)
            r["misaligned_read"] = "accepted (no EINVAL): this filesystem does not reject misaligned O_DIRECT, so the reader must enforce alignment itself"
        except OSError as e:
            r["misaligned_read"] = f"refused: {errno.errorcode.get(e.errno, e.errno)}"
        finally:
            os.close(fd)
        pf, fr = r["prefaulted"], r["fresh"]
        r["status"] = "pass" if pf["matches_buffered_read"] and fr["matches_buffered_read"] and pf["bypassed_page_cache"] is True else "fail"
        r["finding"] = ("O_DIRECT into a pre-faulted buffer bypasses the page cache" if pf["bypassed_page_cache"] is True else "O_DIRECT into a pre-faulted buffer was cached") + \
                       ("; into a fresh, never-touched buffer it is served through the page cache: pre-fault O_DIRECT destination buffers" if fr["bypassed_page_cache"] is False else
                        "; a fresh buffer also bypasses it" if fr["bypassed_page_cache"] is True else "")
    except OSError as e:
        r["status"] = f"error: {e}"
    return r
od_files = [p for p in (os.environ.get("PREFLIGHT_ODIRECT_FILES") or "").split(":") if p] or \
           [g[0] for g in (sorted(glob.glob(os.path.expanduser("~/models/tf-port/inputs-tp2-*/rank*/rank-cache/rank*of2-*.bin"))),
                           sorted(glob.glob(os.path.expanduser("~/models/tf-port/ds1017-*/model-00048-of-00048.safetensors")))) if g]
R["disk"]["odirect_probe"] = [odirect_probe(p) for p in od_files] or "no rank cache or Engram shard found on this box"

# ---- network
rc, ipa = run("ip_addr", ["ip", "-j", "addr"])
rc, ipr = run("ip_route", ["ip", "-j", "route"])
ifs = []
for a in (json.loads(ipa) if ipa else []):
    n = a["ifname"]
    if n == "lo": continue
    base = f"/sys/class/net/{n}"
    dev = os.path.realpath(base + "/device") if os.path.exists(base + "/device") else None
    ifs.append({"ifname": n, "operstate": a.get("operstate"), "mtu": a.get("mtu"), "addr4": [x["local"] + "/" + str(x["prefixlen"]) for x in a.get("addr_info", []) if x.get("family") == "inet"],
                "driver": os.path.basename(os.path.realpath(base + "/device/driver")) if dev else None, "speed_mbps": num(rd(base + "/speed")),
                "pcie_link": f"{rd(dev + '/current_link_speed')} x{rd(dev + '/current_link_width')}" if dev and rd(dev + "/current_link_speed") else None,
                "numa_node": num(rd(base + "/device/numa_node")) if dev else None,
                "tx_queue_len": num(rd(base + "/tx_queue_len")), "gro_flush_timeout": rd(base + "/gro_flush_timeout"), "napi_defer_hard_irqs": rd(base + "/napi_defer_hard_irqs")})
tb = []
for d in sorted(glob.glob("/sys/bus/thunderbolt/devices/*")):
    if os.path.exists(d + "/rx_speed"):
        tb.append({"dev": os.path.basename(d), "device_name": rd(d + "/device_name"), "vendor_name": rd(d + "/vendor_name"), "generation": rd(d + "/generation"),
                   "rx": f"{rd(d + '/rx_speed')} x{rd(d + '/rx_lanes')}", "tx": f"{rd(d + '/tx_speed')} x{rd(d + '/tx_lanes')}", "authorized": rd(d + "/authorized")})
rc, bolt = run("boltctl", ["boltctl", "list"])
rc, rdma = run("rdma_link", ["rdma", "link", "show"])
peers = {}
for i in ifs:
    if i["ifname"].startswith("thunderbolt") and i["addr4"]:
        mine = i["addr4"][0].split("/")[0]
        o = mine.split(".")
        peer = ".".join(o[:3] + ["1" if o[3] == "2" else "2"])
        rc, pg = run(f"ping_{i['ifname']}", ["ping", "-c", "5", "-i", "0.2", "-q", "-I", i["ifname"], peer], timeout=15)
        m = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", pg)
        peers[i["ifname"]] = {"peer": peer, "reachable": rc == 0, "rtt_ms_min_avg_max_mdev": [float(x) for x in m.groups()] if m else None}
R["network"] = {"interfaces": ifs, "routes": json.loads(ipr) if ipr else None, "thunderbolt_links": tb, "boltctl": bolt.strip().splitlines()[:30],
                "rdma_links": rdma.strip() or "none", "infiniband_class": sorted(os.listdir("/sys/class/infiniband")) if os.path.isdir("/sys/class/infiniband") else [],
                "peer_ping": peers,
                "sysctl": {k: rd("/proc/sys/" + k.replace(".", "/")) for k in ("net.core.busy_poll", "net.core.busy_read", "net.core.rmem_max", "net.core.wmem_max", "net.core.netdev_max_backlog", "net.ipv4.tcp_rmem", "net.ipv4.tcp_wmem", "net.ipv4.tcp_low_latency", "net.ipv4.tcp_congestion_control")}}

# ---- clocks, power, temperatures
govs, epps, drivers = {}, {}, {}
for c in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq"):
    for d, k in ((govs, "scaling_governor"), (epps, "energy_performance_preference"), (drivers, "scaling_driver")):
        v = rd(c + "/" + k)
        if v: d[v] = d.get(v, 0) + 1
cur = [num(rd(c + "/scaling_cur_freq")) for c in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")]
temps = {}
for hp in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
    name = rd(hp + "/name")
    for f in sorted(glob.glob(hp + "/temp*_input")):
        lab = rd(f.replace("_input", "_label")) or os.path.basename(f)
        temps[f"{name}:{lab}"] = num(rd(f)) / 1000 if rd(f) else None
    for f in sorted(glob.glob(hp + "/power*_average")) + sorted(glob.glob(hp + "/power*_input")):
        temps[f"{name}:{os.path.basename(f)}_W"] = num(rd(f)) / 1e6 if rd(f) else None
rc, pp = run("powerprofilesctl", ["powerprofilesctl", "get"])
rc, sleep = run("sleep_targets", "systemctl is-enabled sleep.target suspend.target hibernate.target hybrid-sleep.target 2>&1; systemctl show -p IdleAction logind 2>/dev/null; grep -hE '^[^#]*(IdleAction|HandleSuspend|HandleLidSwitch)' /etc/systemd/logind.conf /etc/systemd/logind.conf.d/* 2>/dev/null", shell=True)
R["power"] = {"cpufreq": {"governors": govs, "epp": epps, "drivers": drivers, "boost": rd("/sys/devices/system/cpu/cpufreq/boost"),
                          "amd_pstate_status": rd("/sys/devices/system/cpu/amd_pstate/status"), "cur_khz_min_max": [min(cur), max(cur)] if cur else None},
              "platform_profile": rd("/sys/firmware/acpi/platform_profile"), "platform_profile_choices": rd("/sys/firmware/acpi/platform_profile_choices"),
              "powerprofilesctl": pp.strip(), "sensors": temps, "sleep_targets": sleep.strip().splitlines(),
              "gpu_lock": {"path": "/var/tmp/halo-gpu.lock", "exists": os.path.exists("/var/tmp/halo-gpu.lock")}}
rc, lk = run("lslocks", ["lslocks", "-o", "COMMAND,PID,PATH", "--noheadings"])
R["power"]["gpu_lock"]["holders"] = [l for l in lk.splitlines() if "halo-gpu.lock" in l]
R["load"] = {"loadavg": rd("/proc/loadavg"), "memory_pressure": (rd("/proc/pressure/memory", "") or "").splitlines(),
             "note": "box activity while preflight ran (other workers' jobs may be running; preflight takes no timings)"}

if os.environ.get("BASELINE"):
    from preflight_drift import drift
    R["drift_vs_baseline"] = {"baseline": os.environ["BASELINE"], **drift(json.load(open(os.environ["BASELINE"])), R)}
out = os.path.join(OUT, f"preflight-{H}.json")
with open(out, "w") as f: json.dump(R, f, indent=1, default=str)
print(out, "hip_probe:", R["gpu"]["hip_probe"]["status"], "odirect:",
      [p.get("status") for p in R["disk"]["odirect_probe"]] if isinstance(R["disk"]["odirect_probe"], list) else R["disk"]["odirect_probe"],
      "zig:", [z["version"] for z in zigs], "unavailable:", R["unavailable"])
PY

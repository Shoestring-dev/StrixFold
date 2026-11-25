#!/usr/bin/env python3
"""strixfold doctor: is this box ready for StrixFold, and what comes next?

    ./strixfold doctor [--json] [--role node0|node1] [--conf FILE]

It only reads: no sudo, nothing installed, written, started or stopped. The commands it runs are the ones in COMMANDS
(over ssh to the other box only `true`, with the host key already known and nothing remembered). Its output names no
host, user or MAC address, so it can be pasted into an issue or the Discord. Python 3.9 or later, standard library only.

Each line is one check, in the order of docs/GETTING-STARTED.md:
  ok    as it should be
  todo  not done yet
  wait  done here as far as it goes: the other box has to do something first
  bad   wrong: it stops StrixFold from working
  note  differs from the boxes we measured on (it may still work), or good to know
What comes next is the first todo, wait or bad. On a Ryzen AI Halo's own system and on Debian 13 that leads from new boxes
to a ready pair. Any other system has its own guide, which leads; the lines here only check. Exit status: 0 nothing is bad,
1 something is."""
import argparse
import glob
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import urllib.error
import urllib.request

try:
    import grp
    import pwd
except ImportError:  # Windows: the_system says so
    grp = pwd = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMANDS = ("dpkg-query", "/sbin/ldconfig", "mokutil", "ip", "ping", "ssh", "systemctl")
KERNEL = "7.2.7+rex-amd64"
KERNEL_KEY = "Collabora Rex secure boot - CA 2026"
ROCM = "10.0.0-4"
GTT = 133143986176  # 124 GiB
ZIG = "0.17.0"
VENV = ("numpy-2.3.5", "tokenizers-0.22.1", "torch-2.9.0+cpu")  # prepare_inputs.sh asserts these
TRITON = "triton-3.8.0+git675c5987.rocm10.1.0a20260822"
GB = 1000 ** 3
CACHE = 106 * GB  # one rank's
ROOM = 1.05  # builds, a download's partial files
LOCK = "/var/tmp/halo-gpu.lock"
ENGINES, OTHERS = ("tensorfold-nati", "tf-dsv41-lanes", "tf-dsv41-layer"), ("ds4", "ds4-server", "ds4-bench")  # together: serve_lib.sh's serve_idle
SSH = ["ssh"] + [x for o in ("BatchMode=yes", "ConnectTimeout=5", "StrictHostKeyChecking=yes", "UpdateHostKeys=no", "ControlMaster=no", "ControlPath=none",
                             "AddKeysToAgent=no", "PermitLocalCommand=no", "ForwardAgent=no", "ForwardX11=no", "ClearAllForwardings=yes") for x in ("-o", o)]
ORDER = ("The box", "Software", "The link", "Tools", "Model files", "Inputs", "Serving")
START = "docs/GETTING-STARTED.md"
PLAIN = ("box", "localhost", "node0", "node1", "debian", "ubuntu", "fedora", "rex", "strixfold", "video", "render")  # host names that say nothing
# where each system's steps are: rex (the Ryzen AI Halo's own), Debian 13 (any other box), Ubuntu, Fedora, our container
WHERE = {
    "rex": {"kernel": f"{START}, step 2a", "gpu memory": "docs/BOSGAME-M5.md, step 5", "rocm": "docs/BOSGAME-M5.md, step 7", "link": f"{START}, step 3",
            "python": f"{START}, step 4", "groups": f"{START}, step 2a"},
    "debian": {"kernel": "docs/BOSGAME-M5.md, steps 3 to 6", "gpu memory": "docs/BOSGAME-M5.md, step 5", "rocm": "docs/BOSGAME-M5.md, step 7",
               "link": f"{START}, step 3", "python": f"{START}, step 4", "groups": "docs/BOSGAME-M5.md, step 7"},
    "ubuntu": {"kernel": "docs/FRAMEWORK-DESKTOP.md, section 2", "gpu memory": "docs/FRAMEWORK-DESKTOP.md, section 2", "rocm": "docs/FRAMEWORK-DESKTOP.md, section 5",
               "link": "docs/FRAMEWORK-DESKTOP.md, sections 3 and 4", "python": "docs/FRAMEWORK-DESKTOP.md, section 6", "groups": "docs/FRAMEWORK-DESKTOP.md, section 4"},
    "fedora": {"kernel": "docs/FEDORA-PODMAN.md, section 1", "gpu memory": "docs/FEDORA-PODMAN.md, section 1", "rocm": "docs/FEDORA-PODMAN.md, section 3",
               "link": "docs/FEDORA-PODMAN.md, section 2", "python": "docs/FEDORA-PODMAN.md, section 4", "groups": "docs/FEDORA-PODMAN.md, section 1"},
    "other": {"kernel": "docs/HARDWARE.md, Operating systems", "gpu memory": "docs/HARDWARE.md, Operating systems", "rocm": "docs/HARDWARE.md, Operating systems",
              "link": "docs/HARDWARE.md, USB4 ports", "python": f"{START}, step 4", "groups": f"{START}, step 2a"},
}
WHERE["container"] = dict(WHERE["fedora"])
LED = ("rex", "debian")  # the systems whose steps these lines lead through
OWN = {"ubuntu": "docs/FRAMEWORK-DESKTOP.md", "fedora": "docs/FEDORA-PODMAN.md", "container": "docs/FEDORA-PODMAN.md", "other": "docs/HARDWARE.md"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class Box:
    """What the checks read. The tests give a made-up one."""
    def __init__(self):
        self.home = os.path.expanduser("~")
        self.host = socket.gethostname()
        self.root = ROOT

    def read(self, path):
        try:
            if not stat.S_ISREG(os.stat(path).st_mode):
                return None
            with open(path, "rb") as f:
                return f.read(4 << 20).decode("utf-8", "replace")
        except OSError:
            return None

    def raw(self, path, n=64):
        try:
            with open(path, "rb") as f:
                return f.read(n)
        except OSError:
            return None

    def sha256(self, path):
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
        except OSError:
            return None
        return h.hexdigest()

    def glob(self, pattern):
        return sorted(glob.glob(pattern))

    def size(self, path):
        try:
            return os.stat(path).st_size
        except OSError:
            return None

    def mtime(self, path):
        try:
            return os.stat(path).st_mtime
        except OSError:
            return None

    def real(self, path):
        return os.path.realpath(path)

    def root_user(self):
        return hasattr(os, "geteuid") and os.geteuid() == 0

    def inode(self, path):
        try:
            return os.stat(path).st_ino
        except OSError:
            return None

    def free(self, path):
        while path and not os.path.exists(path):
            path = os.path.dirname(path)
        try:
            s = os.statvfs(path or "/")
        except OSError:
            return None
        return s.f_bavail * s.f_frsize

    def usable(self, path):
        return os.access(path, os.R_OK | os.W_OK)

    def groups(self):
        """(the groups this process has, the ones the account is in)."""
        have, member = set(), set()
        try:
            user = pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            user = None
        for g in os.getgroups():
            try:
                have.add(grp.getgrgid(g).gr_name)
            except KeyError:
                pass
        for g in grp.getgrall():
            if user in g.gr_mem:
                member.add(g.gr_name)
        return have, member | have

    def kernel(self):
        return os.uname().release if hasattr(os, "uname") else ""

    def windows(self):
        return os.name == "nt"

    def run(self, argv, timeout=8):
        """(exit status, output); 127 when the command isn't there. Only COMMANDS."""
        if argv[0] not in COMMANDS:
            raise ValueError(argv[0])
        try:
            p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                               env=dict(os.environ, LC_ALL="C"))
        except (OSError, subprocess.TimeoutExpired):
            return 127, ""
        return p.returncode, p.stdout.decode("utf-8", "replace")

    def get(self, url):
        """The HTTP status of a GET on this box (past any proxy), or None."""
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(url, timeout=4) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except (OSError, ValueError):
            return None


class Doctor:
    def __init__(self, box, role=None, conf=None):
        self.b, self.role, self.conf_path = box, role, conf
        self.out = []
        self.vars = {}
        self.system = "other"

    def say(self, section, status, title, detail="", fix="", guide=""):
        self.out.append({"section": section, "status": status, "title": title, "detail": self.hide(detail), "fix": fix, "guide": guide})
        return status

    def hide(self, text):
        """Nothing of whose box it is: the home folder as ~, the host's name out."""
        text = str(text)
        if len(self.b.home) > 1:
            text = text.replace(self.b.home, "~")
        return text if not self.b.host or self.b.host.lower() in PLAIN else re.sub(rf"\b{re.escape(self.b.host)}\b", "<this box>", text)

    def h(self, *parts):
        return os.path.join(self.b.home, *parts)

    def where(self, what):
        return WHERE[self.system][what]

    # ---------------------------------------------------------------- the box
    def the_system(self):
        rel = dict(re.findall(r'(?m)^(\w+)="?([^"\n]*)"?$', self.b.read("/etc/os-release") or ""))
        name, ident = rel.get("PRETTY_NAME", "unknown"), rel.get("ID")
        ours = "framework" in (self.b.read("/sys/class/dmi/id/sys_vendor") or "").lower() and rel.get("VERSION_ID") == "26.10"
        k = self.b.kernel()
        if "microsoft" in k.lower() or self.b.windows():
            return self.say("The box", "bad", "System", "Windows (or Linux inside it)", "StrixFold needs Linux on the box itself: install Debian 13.", "docs/BOSGAME-M5.md, steps 1 and 2")
        if self.b.root_user():
            self.say("The box", "note", "User", "you are root", "Run this as the user who will run StrixFold, without sudo.")
        self.system = ("container" if self.b.size("/run/.containerenv") is not None else "rex" if "rex" in name.lower() else
                       "debian" if (ident, rel.get("VERSION_ID")) == ("debian", "13") else ident if ident == "fedora" or (ident == "ubuntu" and ours) else "other")
        if self.system in ("rex", "debian"):
            self.say("The box", "ok", "System", name)
        else:
            self.say("The box", "note", "System", name + (", in a container" if self.system == "container" else ""),
                     "Our numbers come from Debian 13; its package commands are for Debian 13 only. Use this system's guide.", self.where("kernel").split(",")[0])
        ver, model = self.b.read("/sys/class/dmi/id/bios_version"), self.b.read("/sys/class/dmi/id/product_name")
        self.say("The box", "note", "Model", f"{(model or 'unknown').strip()}, BIOS {(ver or 'unknown').strip()}")

    def chip(self):
        m = re.search(r"(?m)^model name\s*:\s*(.+)$", self.b.read("/proc/cpuinfo") or "")
        name = m.group(1).strip() if m else "unknown processor"
        gfx = {m for f in self.b.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")
               for m in re.findall(r"(?m)^gfx_target_version (\d+)$", self.b.read(f) or "")}
        if "110501" in gfx:
            if "395" in name:
                return self.say("The box", "ok", "Chip", f"{name}, gfx1151")
            return self.say("The box", "note", "Chip", f"{name}, gfx1151", "StrixFold is measured on the Ryzen AI Max+ 395 (16 cores); serve.conf's ENGINE_CPUS expects them.", "docs/HARDWARE.md")
        if re.search(r"ryzen ai max", name, re.I):
            if self.old_kernel():
                return self.say("The box", "note", "Chip", f"{name}; its GPU comes with the kernel (below)")
            return self.say("The box", "todo", "Chip", f"{name}, but the kernel shows no gfx1151 GPU to ROCm", "Install the kernel's firmware, or a newer kernel, then restart.", self.where("kernel"))
        self.say("The box", "bad", "Chip", name, "StrixFold needs a Ryzen AI Max+ 395 with its gfx1151 GPU.", "docs/HARDWARE.md")

    def memory(self):
        m = re.search(r"(?m)^MemTotal:\s+(\d+) kB", self.b.read("/proc/meminfo") or "")
        gib = int(m.group(1)) / 1048576 if m else 0
        known = self.numbers("/sys/class/drm/card*/device/mem_info_vram_total")
        vram = sum(known) / 2 ** 30
        if gib >= 118:
            return self.say("The box", "ok", "Memory", f"{gib:.0f} GiB")
        if gib + vram >= 118:
            return self.say("The box", "bad", "Memory", f"Linux has {gib:.0f} GiB; the BIOS sets {vram:.0f} aside for the GPU",
                            "In the BIOS, set the GPU's memory (UMA Frame Buffer Size) to its smallest value.", "docs/BOSGAME-M5.md, step 1")
        if not known and gib > 24:
            return self.say("The box", "bad", "Memory", f"Linux has {gib:.0f} GiB", "Each box needs 128 GB. If this one has it, its BIOS sets the rest aside for the GPU: set that (UMA Frame Buffer Size) to its "
                            "smallest value.", "docs/BOSGAME-M5.md, step 1")
        self.say("The box", "bad", "Memory", f"{gib + vram:.0f} GiB", "Each box needs 128 GB.", "docs/HARDWARE.md")

    def numbers(self, pattern):
        return [int(v) for v in (self.b.read(f) for f in self.b.glob(pattern)) if (v or "").strip().isdigit()]

    def secure_boot_on(self):
        f = self.b.glob("/sys/firmware/efi/efivars/SecureBoot-*")
        raw = self.b.raw(f[0]) if f else None
        return raw[-1] == 1 if raw else None if self.b.glob("/sys/firmware/efi") else False  # (no EFI: an old-style start, without it)

    def kernel_key(self):
        """Whether the firmware trusts the kernel's key (enrolled, or built in as on a Ryzen AI Halo); None: unknown."""
        seen = False
        for what in ("--list-enrolled", "--db"):
            rc, out = self.b.run(["mokutil", what])
            seen = seen or (rc == 0 and "Subject:" in out)
            if KERNEL_KEY in out:
                return True
        return False if seen else None

    def old_kernel(self):
        """Older than ours, where ours can be installed."""
        three = lambda v: tuple(int(x) for x in (re.findall(r"\d+", v) + ["0"] * 3)[:3])  # noqa: E731
        return self.system in ("rex", "debian") and self.b.kernel() != KERNEL and three(self.b.kernel()) <= three(KERNEL)

    def kernel(self):
        k, sb = self.b.kernel(), self.secure_boot_on()
        if k == KERNEL:
            self.say("The box", "ok", "Kernel", k)
        elif not self.old_kernel():
            self.say("The box", "note", "Kernel", k, f"Ours is {KERNEL}; owners run others.", self.where("kernel"))
        elif self.b.glob(f"/usr/lib/modules/{KERNEL}"):
            key = True if sb is False else self.kernel_key()
            if key:
                self.say("The box", "todo", "Kernel", f"{KERNEL} is installed, {k} is running", "Restart the box.", self.where("kernel"))
            else:
                self.say("The box", "todo", "Kernel", f"{KERNEL} is installed, but Secure Boot won't start it yet", "Step 5, then step 6, whose last command is the only restart (with your user at the "
                         "box's monitor and keyboard).", "docs/BOSGAME-M5.md, steps 5 and 6")
        elif self.system == "debian" and not any("debs.ryai.dev" in (self.b.read(f) or "") for f in self.b.glob("/etc/apt/sources.list.d/*")):
            self.say("The box", "todo", "Kernel", k, f"Ours is {KERNEL}: add its package repository first. Then steps 4, 5 and 6; the only restart is at the end of step 6.", "docs/BOSGAME-M5.md, step 3")
        else:
            self.say("The box", "todo", "Kernel", k, f"Install {KERNEL} and its firmware." + (" Then steps 5 and 6; the only restart is at the end of step 6." if self.system == "debian" else ""),
                     "docs/BOSGAME-M5.md, step 4")
        if sb is None:
            self.say("The box", "note", "Secure Boot", "not readable")
        else:
            self.say("The box", "ok" if sb else "note", "Secure Boot", "on" if sb else "off", "" if sb else "It works without, but the kernel is no longer checked at start.")

    def gpu_memory(self):
        most = max(self.numbers("/sys/class/drm/card*/device/mem_info_gtt_total"), default=0)
        if most >= GTT:
            return self.say("The box", "ok", "GPU memory", f"{most / 2 ** 30:.0f} GiB")
        late = self.system == "debian" and self.old_kernel()
        self.say("The box", "todo", "GPU memory", f"{most / 2 ** 30:.0f} GiB of the 124 it needs" if most else "not shown by the kernel",
                 "Add amdgpu.gttsize=126976 ttm.pages_limit=32505856 to the kernel options (the step's sed replaces the whole line: if it holds more than quiet, keep that in it)" + ("; they count from the restart at the end of step 6." if late else
                 ", then restart."), self.where("gpu memory"))

    def sleep(self):
        rc, out = self.b.run(["systemctl", "is-enabled", "sleep.target", "suspend.target"])
        states = out.split()
        if rc == 127 or not states:
            return
        if all(s == "masked" for s in states):
            return self.say("The box", "ok", "Sleep", "off")
        self.say("The box", "todo", "Sleep", "the box may go to sleep in a long step", "Mask the sleep targets.", self.where("gpu memory"))

    # ---------------------------------------------------------------- software
    def rocm(self):
        if self.system == "fedora":
            return self.say("Software", "note", "ROCm and the tools", "are in the container on Fedora", "Set up the link, then the container, and run this again inside it.", self.where("rocm"))
        _, out = self.b.run(["dpkg-query", "-W", "-f", "${db:Status-Abbrev}|${Package}|${Version}\n", "amdrocm-core*", "therock-gfx1151"])
        got = {p: v for st, p, v in (l.split("|") for l in out.splitlines() if l.count("|") == 2) if st.strip()[-1:] == "i"}
        core = {p: v for p, v in got.items() if re.match(r"amdrocm-core[\d.]+-gfx1151$", p)}
        sdk = [p for p in got if re.match(r"amdrocm-core-sdk[\d.]+-gfx1151$", p)]
        loose = [p for st, p, v in (l.split("|") for l in out.splitlines() if l.count("|") == 2) if p in (*core, *sdk, "therock-gfx1151") and st[:1] == "i"]
        _, libs = self.b.run(["/sbin/ldconfig", "-p"])
        found = "libamdhip64.so.7" in libs
        if self.system in ("rex", "debian") and core and not sdk:
            self.say("Software", "todo", "ROCm", "its SDK isn't installed (node1 builds with it)", "Install the three packages of this step.", self.where("rocm"))
        elif any(v == ROCM for v in core.values()):
            self.say("Software", "ok", "ROCm", ROCM)
        elif core or found:
            self.say("Software", "note", "ROCm", ", ".join(sorted(core.values())) or "there, not from packages", f"Ours is {ROCM}; owners also run 10.1.", self.where("rocm"))
        else:
            self.say("Software", "todo", "ROCm", "not installed", "Install it as this step says." if self.system != "other" else "Install ROCm 10 for gfx1151 your system's way.", self.where("rocm"))
        if self.system in LED and core and loose:
            self.say("Software", "note", "ROCm is not held", "an upgrade can replace it", "The step holds it with apt-mark hold.", self.where("groups"))
        if found:
            self.say("Software", "ok", "ROCm's libraries", "found by the system")
        elif core:
            self.say("Software", "todo", "ROCm's libraries", "installed, but the system can't find libamdhip64.so.7",
                     "Install therock-gfx1151 as this step does, then sudo ldconfig." if self.system in ("rex", "debian") else "Register /opt/rocm/lib with ldconfig.", self.where("rocm"))

    def access(self):
        have, member = self.b.groups()
        want, missing = {"render", "video"}, False
        if self.b.size("/dev/kfd") is not None and self.b.usable("/dev/kfd"):
            return self.say("Software", "ok", "The GPU device", "you can use it")
        if not want <= have:
            missing = True
            if want <= member:
                self.say("Software", "todo", "Groups", "you are in render and video, but not in this session yet", "Log out and in again (an AI tool: start it again in a new login).")
            else:
                self.say("Software", "todo", "Groups", f"not in {' and '.join(sorted(want - member))}", 'sudo usermod -aG render,video "$USER", then log out and in (an AI tool: start it again in a new login).', self.where("groups"))
        if self.b.size("/dev/kfd") is None:
            self.say("Software", "todo", "The GPU device", "/dev/kfd is not there", "The kernel and its firmware come first.", self.where("kernel"))
        elif missing:
            self.say("Software", "todo", "The GPU device", "comes with the groups")
        else:
            self.say("Software", "bad", "The GPU device", "/dev/kfd can't be opened by you, though you are in its groups", "Show your user `ls -l /dev/kfd`.")

    # ---------------------------------------------------------------- the link
    def nets(self):
        """The two cables' networks and node0's address on each: serve.conf's MASTER_IP and TP_STRIPE (empty: one cable)."""
        a = self.vars.get("MASTER_IP") or "10.40.0.1"
        b = self.vars.get("TP_STRIPE", "10.40.1.1")
        return {x.rsplit(".", 1)[0] + ".": x for x in (a, b) if re.fullmatch(r"\d+(\.\d+){3}", x)}

    def link(self):
        rc, out = self.b.run(["ip", "-j", "-4", "addr", "show"])
        try:
            addr = {x["ifname"]: [a["local"] for a in x.get("addr_info") or [] if a.get("local")] for x in json.loads(out) if x.get("ifname")}
        except (ValueError, AttributeError, TypeError):
            addr = None
        nets, guide = self.nets(), self.where("link")
        here = {net: next(((n, a) for n, l in (addr or {}).items() for a in l if a.startswith(net)), None) for net in nets}
        ends = {"node0" if x[1] == nets[net] else "node1" for net, x in here.items() if x}
        if self.role is None and len(ends) == 1:
            self.role = next(iter(ends))
        elif ends and self.role not in (None, *ends):
            self.say("The link", "bad", "Role", f"you said {self.role}, its addresses are {' and '.join(sorted(ends))}'s", "Say the other role, or give this box the other box's addresses.", guide)
        if self.vars.get("TRANSPORT") == "rdma":
            return self.say("The link", "note", "Cables", "your configuration uses RDMA", "Not looked at here: rdma_check.py checks it when a rank starts.")
        if addr is None:
            return self.say("The link", "note", "Cables", "could not be looked at: no working `ip`", "Install iproute2.")
        cards = [os.path.basename(p) for p in self.b.glob("/sys/class/net/thunderbolt*")]
        if len(cards) < len(nets) and not all(here.values()):
            self.say("The link", "todo", "Cables", f"{len(cards)} of {len(nets)} USB4 links seen", "Join the boxes' USB4 ports with two cables, both boxes on; sudo modprobe thunderbolt_net on both if "
                     "it isn't loaded. Not every port links to another computer: try another pair of ports.", guide)
            if not cards:
                return
        bare = [n for n in cards if n not in {x[0] for x in here.values() if x}]
        for net in nets:
            if not here[net]:
                if bare:
                    n = bare.pop(0)
                    self.say("The link", "todo", n, "no link address yet" if not addr.get(n) else "has an address, not the link's", f"Give the cables their fixed addresses: {nets[net]} on node0, another {net}x on node1 (the guide's: {net}2).", guide)
                continue
            n, a = here[net]
            if (self.b.read(f"/sys/class/net/{n}/operstate") or "").strip() != "up":
                self.say("The link", "wait", n, f"{a}, but the cable has no link yet", "Set up the link on the other box (it must be on). If it is set up there: both ends of the cable, or another pair of ports.", guide)
                continue
            mtu = (self.b.read(f"/sys/class/net/{n}/mtu") or "").strip()
            self.say("The link", "ok" if mtu == "1500" else "note", n, f"up, {a}" + ("" if mtu == "1500" else f", MTU {mtu}"),
                     "" if mtu == "1500" else "Ours is 1500 on both boxes: larger ones lost packets.", "" if mtu == "1500" else "docs/TUNING.md")
        got = {}
        for net, x in here.items():
            if x and (self.b.read(f"/sys/class/net/{x[0]}/operstate") or "").strip() == "up":
                peer = nets[net] if x[1] != nets[net] else net + "2" if nets[net] == net + "1" else None
                if peer is None:  # (a configuration with other addresses: node1's isn't in it)
                    continue
                got[peer] = self.b.run(["ping", "-n", "-c", "1", "-W", "2", peer])[0]
        for peer, rc in got.items():
            if rc == 0:
                self.say("The link", "ok", f"The other box at {peer}", "answers")
            elif rc == 127:
                self.say("The link", "note", f"The other box at {peer}", "not asked: no working `ping`", "Install iputils-ping to check it.")
            elif 0 in got.values():
                self.say("The link", "wait", f"The other box at {peer}", "doesn't answer, and the other cable does", "Run this on the other box: it may lack that address. If both boxes have both addresses, the cables are crossed: swap "
                         "them at one box.", guide)
            else:
                self.say("The link", "wait", f"The other box at {peer}", "doesn't answer yet", "Set up the link on the other box. Set up on both and still nothing: the cables may be crossed (swap them at one box), both boxes may "
                         "have the same addresses, or a firewall is in the way.", guide)
        if len(ends) > 1:
            self.say("The link", "bad", "Addresses", "this box is node0 on one cable and node1 on the other", "Use node0's on both or node1's on both.", guide)
        if self.role is None and not ends:
            self.say("The link", "note", "Role", "not known yet", "Before the link is set up, say which box this is: --role node0 or --role node1.")

    def ssh(self):
        if self.role != "node1":
            return
        master = self.vars.get("MASTER_IP") or "10.40.0.1"
        _, out = self.b.run(["ssh", "-G", "node0"])
        m = re.search(r"(?m)^hostname (\S+)$", out)
        if not m or m.group(1) != master:
            there = m and m.group(1) != "node0"
            return self.say("The link", "todo", "SSH to node0", f"the name node0 isn't {master} yet", f"Change its entry in ~/.ssh/config to HostName {master}; a second entry does nothing." if there else
                            "Make a key, copy it to node0, and add the Host node0 entry.", self.where("link"))
        rc, _ = self.b.run(SSH + ["node0", "true"], timeout=12)
        if rc == 0:
            return self.say("The link", "ok", "SSH to node0", "works, no password")
        self.say("The link", "todo", "SSH to node0", "doesn't work without a password (or node0 is off)", "Your user copies the key to node0 and logs in there once, to accept its host key. A key with a passphrase "
                 "doesn't work here: the key must have none.", self.where("link"))

    # ---------------------------------------------------------------- tools
    def set_up(self):
        """Deployed, with its inputs checked: what built it isn't needed on this box any more."""
        rep = self.report()
        return bool(rep) and rep[0] == "pass" and not self.deployment()[1]

    def deployed(self):
        return self.expand(self.vars.get("DEPLOY_DIR")) or self.h("tfp/deploy/current")

    def deployment(self):
        """(its commit, what it lacks)."""
        dep = self.deployed()
        try:
            commit = str(json.loads(self.b.read(os.path.join(dep, "DEPLOY.json")) or "")["commit"])
        except (ValueError, KeyError, TypeError):
            return None, ["nothing deployed on this box"]
        return commit, [f"no {x}" for x in ("native-hip/bin/tensorfold-native", "native-hip/bin/tf-dsv41-lanes", "ds-kit/engram.json") if self.b.size(os.path.join(dep, x)) is None]

    def venv(self, name, want):
        """None when it holds `want`, else what is wrong with it."""
        base = self.h("venvs", name)
        if self.b.size(f"{base}/bin/python") is None:
            return f"no ~/venvs/{name}"
        if not self.b.glob(f"{base}/lib/python3.13/site-packages"):
            return f"~/venvs/{name} isn't made with Python 3.13"
        lacks = [w for w in want if not self.b.glob(f"{base}/lib/python3.13/site-packages/{w}.dist-info")]
        return f"~/venvs/{name} lacks {', '.join(lacks)}: its install didn't finish" if lacks else None

    def tools(self):
        if self.system == "fedora" or self.elsewhere():
            return
        done = self.set_up()
        todo, tail = ("note", " (not needed to run)") if done else ("todo", "")
        there = lambda t: any(self.b.size(f"{d}/{t}") is not None for d in ("/usr/bin", "/bin", "/usr/local/bin"))  # noqa: E731
        if self.role == "node0" and done and not there("curl"):
            self.say("Tools", "todo", "curl", "missing: a start asks the server's /health with it", "sudo apt install curl", f"{START}, step 4")
        missing = [t for t in ("git", "rsync", "wget", "curl") if not there(t)]
        if not self.b.glob("/usr/lib/python3*/ensurepip") and not self.b.glob("/opt/python/*/lib/python3.13/ensurepip"):
            missing.append("python3-venv")
        if missing:
            self.say("Tools", todo, "Packages", f"{' and '.join(missing)} missing{tail}", f"sudo apt install {' '.join(missing)}", f"{START}, step 4")
        else:
            self.say("Tools", "ok", "Packages", "there")
        wrong = self.venv("tfp-inputs", VENV)
        if wrong and not done and not any(self.b.glob(f"{d}/python3.13") for d in ("/usr/bin", "/usr/local/bin", self.h(".local/bin"))):
            self.say("Tools", "todo", "Python", "no Python 3.13", "The input tools' lock file needs 3.13.", self.where("python"))
        if not wrong:
            self.say("Tools", "ok", "Input tools", "~/venvs/tfp-inputs")
        elif self.role == "node0":
            self.say("Tools", "note" if done else "wait", "Input tools", wrong + tail, "The step's last two commands, on node1, make it here.", f"{START}, step 4")
        else:
            self.say("Tools", todo, "Input tools", wrong + tail, "Make it on both boxes.", f"{START}, step 4")
        if self.role == "node0":
            return
        zig = self.b.real(self.h("opt/zig/zig")) if self.b.size(self.h("opt/zig/zig")) is not None else None
        if zig and ZIG in zig:
            self.say("Tools", "ok", "Zig", ZIG)
        elif zig:
            self.say("Tools", "note", "Zig", "~/opt/zig/zig is there, its version isn't in its folder's name", f"`~/opt/zig/zig version` must say {ZIG}.", f"{START}, step 4")
        else:
            self.say("Tools", todo, "Zig", f"no ~/opt/zig/zig{tail}", f"Install Zig {ZIG} on node1 (with the ~/opt/zig link).", f"{START}, step 4")
        wrong = self.venv("triton", (TRITON,))
        kit = self.b.size(self.h("tfp/tri/kit/gfx1151/aot.json")) is not None  # (written once every variant was tried)
        try:
            failed = sum(not x["ok"] for x in json.loads(self.b.read(self.h("tfp/tri/kit/gfx1151/build_log.json")) or "[]"))
        except (ValueError, KeyError, TypeError):
            failed = 0
        if wrong and not kit:
            self.say("Tools", todo, "Triton", wrong + tail, "Make it on node1.", f"{START}, step 4")
        if kit and failed:
            self.say("Tools", todo, "Triton kit", f"{failed} of its variants failed to build{tail}", "Move it aside (ask your user) and make it again; the lines it prints for FAILED say why.", f"{START}, step 5, The Triton kit")
        else:
            self.say("Tools", "ok" if kit else todo, "Triton kit", "made" if kit else f"not made yet, or not finished{tail}", "" if kit else "Make it once on node1.", "" if kit else f"{START}, step 5, The Triton kit")
        lacks = [x for x in ("bin/tf-dsv41-rank-cache", "bin/tf-dsv41-check", "native-hip/bin/tensorfold-native", "native-hip/bin/tf-dsv41-lanes")
                 if self.b.size(os.path.join(self.b.root, "zig-out", x)) is None]
        if lacks:
            self.say("Tools", todo, "Build", f"this checkout isn't built: no zig-out/{lacks[0]}{tail}", "Build it and run the GPU tests, all three commands.", f"{START}, step 5, Build and GPU tests")
        else:
            built, moved = self.b.mtime(os.path.join(self.b.root, "zig-out/native-hip/bin/tensorfold-native")), self.moved()
            if built and moved and built < moved:  # (a note: a build that changes nothing leaves the programs' times)
                self.say("Tools", "note", "Build", "there, from before this checkout moved to another commit", "Build it again (all three commands) before you deploy: a deployment names the commit it is of.", f"{START}, step 5, Build and GPU tests")
            else:
                self.say("Tools", "ok", "Build", "there")
            self.gpu_tests(todo, tail)

    def moved(self):
        """When this checkout last moved to another commit, after its clone."""
        t = re.findall(r"(?m)> (\d+) [+-]\d{4}\t", self.b.read(os.path.join(self.b.root, ".git/logs/HEAD")) or "")
        return int(t[-1]) if len(t) > 1 else None

    def gpu_tests(self, todo, tail):
        last = self.b.glob(self.h("tfp/arch/receipts/*/run.json"))
        guide = f"{START}, step 5, Build and GPU tests"
        try:
            run = json.loads(self.b.read(last[-1]) or "") if last else None
        except ValueError:
            run = {}
        if run is None:
            self.say("Tools", todo, "GPU tests", f"not run yet{tail}", "run_gpu_tests.sh, a few minutes.", guide)
        elif not isinstance(run, dict) or not run:
            self.say("Tools", todo, "GPU tests", f"the last run didn't finish{tail}", "run_gpu_tests.sh, a few minutes.", guide)
        elif run.get("all_passed"):
            self.say("Tools", "ok", "GPU tests", "passed")
        else:
            failed = [str(p.get("program")) for p in run.get("programs") or [] if isinstance(p, dict) and p.get("exit")]
            if failed == ["tf-dsv41-hip-glue-test"]:
                self.say("Tools", "note", "GPU tests", "the glue test failed", "On a Bosgame M5 one failing clock check in it is expected; anything else: show your user.", guide)
            else:
                self.say("Tools", "note" if tail else "bad", "GPU tests", f"failed: {', '.join(failed) or 'the last run'}{tail}", "Show your user its log, in the newest ~/tfp/arch/receipts folder. Don't go on before it passes.", guide)

    # ---------------------------------------------------------------- model files and inputs
    def config(self):
        """The serving configuration's keys (a plain KEY=VALUE file; only what is looked at, never run)."""
        path = self.conf_path or self.h("tfp/run/conf/my.conf")
        text = self.b.read(path)
        if text is None:
            return None
        self.vars = {k: v.strip("'\"") for k, v in re.findall(r"(?m)^([A-Z][A-Z0-9_]*)=('[^']*'|\"[^\"]*\"|[^\s#]*)", text)}
        return path

    def expand(self, v):
        return (v or "").replace("${HOME}", self.b.home).replace("$HOME", self.b.home)

    def pins(self, name):
        try:
            return json.loads(self.b.read(os.path.join(self.b.root, "tools/dsv41-hip/assets", name)) or "")["files"]
        except (ValueError, KeyError, TypeError):
            return None

    def compare(self, folder, want):
        """(files as pinned, missing, wrong size) of a download, by their sizes (the inputs step checks every byte)."""
        ok, missing, wrong = 0, [], []
        for path, n in want:
            s = self.b.size(os.path.join(folder, path))
            if s is None:
                missing.append(path)
            elif s != n:
                wrong.append(path)
            else:
                ok += 1
        return ok, missing, wrong

    def bundle(self):
        return self.expand(self.vars.get("BUNDLE")) or self.h("models/tf-port/inputs-tp2-c178a012")

    def report(self):
        """This box's verify_inputs.sh report: (status, the manifest it is of) or None."""
        rank = {"node0": 0, "node1": 1}.get(self.role)
        for f in [f"{self.bundle()}.verify-rank{rank}-{self.b.host}.json"] if rank is not None else []:
            try:
                d = json.loads(self.b.read(f) or "")
                return str(d.get("status")), d.get("manifest_sha256")
            except (ValueError, AttributeError):
                pass
        return None

    def models(self):
        if self.system == "fedora" or self.elsewhere():
            return
        rep = self.report()
        done = bool(rep) and rep[0] == "pass"
        model = self.h("models/tf-port/libertai-2.9bpw")
        engram = self.expand(self.vars.get("ENGRAM_DIR")) or self.h("models/tf-port/ds1017-1bae0e10")
        sets = []
        c = self.pins("checkpoint_files.json")
        if c:
            sets.append(("Checkpoint", model, [(x["path"], x["bytes"]) for x in c if x["path"] != ".gitattributes"], not done))
        o = self.pins("original_files.json")
        shards = [(x["path"], x["bytes"]) for x in o or [] if re.search(r"model-0004[78]-of-00048\.safetensors$", x.get("path", ""))]
        if shards:
            sets.append(("Engram shards", engram, shards, True))
        if not sets:
            return self.say("Model files", "note", "Pins", "not found: run this from a StrixFold checkout")
        node0 = self.role == "node0"
        found = [(*s, *self.compare(s[1], s[2])) for s in sets]
        self.disk(done, node0, sum(n for _t, _f, want, needed, _ok, missing, wrong in found for p, n in want if needed and p in set(missing) | set(wrong)))
        for title, folder, want, needed, ok, missing, wrong in found:
            status, fix = ("wait", "Copy them here from node1, with the step's rsync.") if node0 else ("todo", "Download them on node1; stopped, it takes up where it left.")
            if (missing or wrong) and needed and (self.at_work("hf") or (self.at_work("rsync") and self.partial())):
                self.say("Model files", "todo", title, f"{ok} of {len(want)} files; a download or a copy is under way now", "Wait until it has ended. Don't start another.", f"{START}, step 5, Model files")
            elif not missing and not wrong:
                self.say("Model files", "ok", title, f"{ok} files, the right sizes")
            elif not needed:
                self.say("Model files", "note", title, "not whole on this box; its inputs are made and checked, so it isn't needed")
            elif wrong:
                self.say("Model files", status, title, f"{len(wrong)} of {len(want)} files have the wrong size: {', '.join(sorted(wrong)[:3])}", fix, f"{START}, step 5, Model files")
            else:
                self.say("Model files", status, title, f"{ok} of {len(want)} files", fix, f"{START}, step 5, Model files")

    def disk(self, done, node0, need):
        free = self.b.free(self.h("models"))
        if done or free is None:
            return
        made = self.b.size(os.path.join(self.bundle(), "inputs_manifest.json")) is not None  # (written once both caches are)
        ranks = (0,) if node0 else () if made else (0, 1)
        need = int(max(0, need - self.partial()) * ROOM + sum(CACHE for r in ranks if not self.b.glob(os.path.join(self.bundle(), f"rank{r}", "rank-cache", f"rank{r}of2-*.bin"))) * ROOM)
        if free >= need:
            self.say("Model files", "ok", "Disk space", f"{free / GB:.0f} GB free, about {need / GB:.0f} GB more needed")
        else:
            self.say("Model files", "bad", "Disk space", f"{free / GB:.0f} GB free, about {need / GB:.0f} GB more needed", "Ask your user to make room or use a bigger disk.", f"{START}, step 1")

    def partial(self):
        """What a download or a copy under way already holds of the model files."""
        folders = (self.h("models/tf-port/libertai-2.9bpw"), self.expand(self.vars.get("ENGRAM_DIR")) or self.h("models/tf-port/ds1017-1bae0e10"))
        return sum(self.b.size(f) or 0 for d in folders for pat in (".cache/huggingface/download/*.incomplete", ".cache/huggingface/download/*/*.incomplete", ".*.safetensors.*")
                   for f in self.b.glob(os.path.join(d, pat)))

    def inputs(self):
        if self.system == "fedora" or self.elsewhere():
            return
        b, node0 = self.bundle(), self.role == "node0"
        sha = self.b.sha256(os.path.join(b, "inputs_manifest.json"))
        self.sha = sha
        lacks = self.cache_lacks(b) if sha else None
        if lacks:
            self.say("Inputs", "wait" if node0 else "todo", "Inputs", f"not whole: {lacks}", "Copy node0's part here from node1 again, with the step's rsync." if node0 else
                     "Ask your user: the caches were moved, or the preparation stopped.", f"{START}, step 5, Inputs")
        elif sha:
            self.say("Inputs", "ok", "Inputs", f"there: BUNDLE_SHA256 is {sha}" if not node0 else "there")
        elif node0:
            self.say("Inputs", "wait", "Inputs", "not here yet", "Copy node0's part here from node1, with the step's rsync.", f"{START}, step 5, Inputs")
        elif self.at_work("prepare_inputs.sh", "prepare_inputs.py"):
            self.say("Inputs", "todo", "Inputs", "being made now", "Wait until it has ended (it takes more than an hour). Don't start it again and don't touch its folder.", f"{START}, step 5, Inputs")
        elif any(self.b.size(os.path.join(b, x)) is not None for x in ("PREPARING.json", "FAILED.json")):
            self.say("Inputs", "bad", "Inputs", "a preparation that stopped or failed left its folder half made", "Show your user the folder, and FAILED.json if it holds one (the error). Only with their yes, remove that "
                     "folder and prepare again.", f"{START}, step 5, Inputs")
        else:
            self.say("Inputs", "todo", "Inputs", "not made yet", "prepare_inputs.sh on node1 (about 1 TB read, 212 GB written), then the rsync to node0.", f"{START}, step 5, Inputs")
        if not node0:
            self.check_of_inputs(sha)

    def cache_lacks(self, b):
        """What is missing of this box's rank cache, by the manifest's name and size for it."""
        rank = {"node0": 0, "node1": 1}.get(self.role)
        try:
            c = next(c for c in json.loads(self.b.read(os.path.join(b, "inputs_manifest.json")) or "")["rank_caches"] if c["rank"] == rank)
            name, size = os.path.basename(c["file"]), c["bytes"]
        except (ValueError, KeyError, TypeError, StopIteration):
            return None
        folder = self.expand(self.vars.get("RANK_CACHE", "")).replace("$BUNDLE", b).replace("$RANK", str(rank)) or os.path.join(b, f"rank{rank}", "rank-cache")
        got = self.b.size(os.path.join(folder, name))
        return None if got == size else f"rank {rank}'s cache is missing" if got is None else f"rank {rank}'s cache has the wrong size"

    def check_of_inputs(self, sha):
        rep, node0 = self.report(), self.role == "node0"
        guide = f"{START}, step 5, " + ("Deploy (after it: node0's check)" if node0 else "Inputs")
        status, fix = ("wait", "On node1, run the step's check of node0's copy.") if node0 else ("todo", "Run verify_inputs.sh on this box (about 15 minutes).")
        if self.vars.get("VERIFY_REPORT") == "0":
            self.say("Inputs", "note", "Check of the inputs", "your configuration doesn't ask for it (VERIFY_REPORT=0)")
        elif self.role is None:
            self.say("Inputs", "todo", "Check of the inputs", "which box this is isn't known yet", "Set up the link first, or say --role node0 or node1.")
        elif rep is None:
            self.say("Inputs", status, "Check of the inputs", "not done on this box", fix, guide)
        elif rep[0] == "pass-with-skips":
            self.say("Inputs", status, "Check of the inputs", "passed with parts skipped: a rank won't start on it", fix + " Without the --skip options.", guide)
        elif rep[0] != "pass":
            self.say("Inputs", "bad", "Check of the inputs", f"the report says {rep[0]}", "Run verify_inputs.sh again and show your user what it names.", guide)
        elif sha and rep[1] != sha:
            self.say("Inputs", status, "Check of the inputs", "the report is of other inputs", fix, guide)
        else:
            self.say("Inputs", "ok", "Check of the inputs", "passed")

    # ---------------------------------------------------------------- serving
    def elsewhere(self):
        """An engine runs here from a configuration we weren't told of: what it needs isn't ours to list."""
        return self.path is None and not self.conf_path and bool(self.running())

    def serving(self):
        if self.system == "fedora":
            return
        node0 = self.role == "node0"
        if self.elsewhere():
            self.say("Serving", "note", "Configuration", "an engine is running, from another configuration than ~/tfp/run/conf/my.conf", "To check its set-up, name that one with --conf.")
            return self.engine(node0)
        commit, lacks = self.deployment()
        fix = "Build it, then deploy_rank0.sh for both boxes with --native and --ds-kit, on node1."
        if not lacks:
            self.say("Serving", "ok", "Deployed", f"commit {commit[:12]} (the other box must have the same)")
        else:
            self.say("Serving", "wait" if node0 else "todo", "Deployed", lacks[0] if commit is None else f"without the engine: {', '.join(lacks)}", fix, f"{START}, step 5, Deploy")
        if node0:
            self.check_of_inputs(self.sha)
        self.configuration(node0)
        self.engine(node0)

    def configuration(self, node0):
        guide = f"{START}, step 5, Serve"
        pin, sha = self.vars.get("BUNDLE_SHA256"), self.sha
        copy = "Copy it here from node1, with the step's scp."
        if self.path is None:
            if self.conf_path:
                return self.say("Serving", "bad", "Configuration", "the file you named with --conf can't be read", "Check its path.")
            return self.say("Serving", "wait" if node0 else "todo", "Configuration", "no ~/tfp/run/conf/my.conf", copy if node0 else "Make it on node1 from the deployed serve.conf: the step's commands up to the scp, not yet `up`. Or name yours "
                            "with --conf.", guide)
        if sha and pin == sha:
            return self.say("Serving", "ok", "Configuration", "names these inputs")
        if not sha:
            return self.say("Serving", "wait" if node0 else "todo", "Configuration", "its inputs aren't here yet")
        what = f"BUNDLE_SHA256 is {pin[:8]}..., these inputs are {sha[:8]}..." if pin else "BUNDLE_SHA256 is empty"
        self.say("Serving", "wait" if node0 else "todo", "Configuration", what, "The inputs and the configuration come from node1: copy both again." if node0 else
                 f"Set BUNDLE_SHA256={sha} in it, and copy it to node0.", guide)

    def programs(self, names):
        return [n for n in ((self.b.read(f) or "").strip() for f in self.b.glob("/proc/[0-9]*/comm")) if n in names]

    def at_work(self, *names):
        """Whether a program or script of one of these names runs on this box."""
        for f in self.b.glob("/proc/[0-9]*/cmdline"):
            args = [os.path.basename(a) for a in (self.b.read(f) or "").split("\0")]
            if args[0] in names or (re.match(r"(python|bash|sh\b)", args[0]) and len(args) > 1 and args[1] in names):
                return True
        return False

    def running(self):
        return self.programs(ENGINES)

    def engine(self, node0):
        running = self.running()
        ino = self.b.inode(LOCK)
        held = ino is not None and any(re.search(rf":{ino} ", l) and "->" not in l for l in (self.b.read("/proc/locks") or "").splitlines())
        others = self.programs(OTHERS)
        if running:
            self.say("Serving", "ok", "Engine", "running on this box")
        elif others:
            self.say("Serving", "bad", "Engine", f"another engine is running ({others[0]}): a rank won't start beside it", "Ask your user whether it may be stopped.", f"{START}, section 6")
        elif held:
            self.say("Serving", "note", "GPU lock", "held by something, with no engine running", "A benchmark or another GPU job: wait for it, or ask your user.")
        else:
            m = re.search(r"(?m)^MemAvailable:\s+(\d+) kB", self.b.read("/proc/meminfo") or "")
            avail, peak = int(m.group(1)) // 1048576 if m else 0, self.vars.get("PEAK_GIB", "101")
            need = int(peak) + 8 if re.fullmatch(r"[0-9]+", peak) else 109
            if avail >= need:
                self.say("Serving", "ok", "Free memory", f"{avail} GiB, a rank needs {need}")
            else:
                self.say("Serving", "bad", "Free memory", f"{avail} GiB, a rank needs {need}", "Ask your user what may be closed, or to restart the box.", f"{START}, section 6")
        if node0 and running:
            port = self.vars.get("SERVE_PORT", "18080")
            port = port if re.fullmatch(r"[0-9]+", port) else "18080"
            st = self.b.get(f"http://127.0.0.1:{port}/health")
            if st == 200:
                self.say("Serving", "ok", "Server", f"answers on 127.0.0.1:{port}")
            else:
                self.say("Serving", "note", "Server", f"no answer on 127.0.0.1:{port} yet" if st is None else f"/health says {st}", "Still loading, or on another port: see the run's rank0.log.")

    def run(self):
        self.sha = None
        self.path = self.config()
        for check in (self.the_system, self.chip, self.memory, self.kernel, self.gpu_memory, self.sleep, self.rocm, self.access, self.link, self.ssh,
                      self.tools, self.models, self.inputs, self.serving):
            try:
                check()
            except Exception as e:  # noqa: BLE001 - one check that can't be made never hides the others
                self.say(ORDER[0] if not self.out else self.out[-1]["section"], "note", check.__name__.replace("_", " "), f"could not be checked ({type(e).__name__})")
        return self.out

    def next(self):
        """The first thing to do, in the guide's order; on a system with its own guide: that guide, unless something is wrong."""
        if self.system not in LED:
            bad = next((r for r in self.out if r["status"] == "bad"), None)
            if bad or self.running():
                return bad
            return {"section": "The box", "status": "guide", "title": "Your system's own guide", "detail": "it leads; the lines here only check", "guide": OWN[self.system],
                    "fix": {"ubuntu": "Follow its sections 1 to 8, in order; section 9 only if a start wedges as it says, section 10 only for a longer context. Run this again after each.",
                            "fedora": "Follow its sections 1 to 4, in order. From section 4 on, run this inside the container.",
                            "container": "Follow its section 4. Run this again after each step."}.get(self.system, "Nobody has tested this system on this box. Read it, and start from the guide for "
                                                                                              "the nearest one (Ubuntu: FRAMEWORK-DESKTOP.md). Tell your user.")}
        return next((r for r in self.out if r["status"] in ("todo", "wait", "bad")), None)

    def last_words(self):
        n = self.next()
        if n is None:
            if any(r["title"] == "Engine" for r in self.out):
                return "Nothing to do on this box."
            return "Nothing to do on this box. When the other one has nothing to do either: first_light.sh up, on node1 (docs/GETTING-STARTED.md, step 5, Serve)."
        return (f"Next{', on the other box' if n['status'] == 'wait' else ''}: {n['title']}: {n['detail']}. {n['fix']}" + (f" ({n['guide']})" if n["guide"] else "")).strip()


def text(d):
    marks = {"ok": " ok ", "todo": "todo", "wait": "wait", "bad": "BAD ", "note": "note"}
    lines = ["StrixFold doctor: reads only, changes nothing." + (f" This box is {d.role}." if d.role else "")]
    for s in ORDER:
        rows = [r for r in d.out if r["section"] == s]
        if rows:
            lines += ["", s]
        for r in rows:
            lines.append(f"  [{marks[r['status']]}] {r['title']}: {r['detail']}")
            if r["status"] != "ok" and (r["fix"] or r["guide"]):
                lines.append("         -> " + " ".join(x for x in (r["fix"], f"({r['guide']})" if r["guide"] else "") if x))
    n = {k: sum(r["status"] == k for r in d.out) for k in marks}
    lines += ["", f"{n['ok']} ok, {n['todo']} to do, {n['wait']} waiting for the other box, {n['bad']} bad, {n['note']} notes.", d.last_words()]
    return "\n".join(lines)


def main(argv=None, box=None):
    ap = argparse.ArgumentParser(prog="strixfold doctor", description="Is this box ready for StrixFold, and what comes next? Reads only.")
    ap.add_argument("--json", action="store_true", help="the checks as JSON")
    ap.add_argument("--role", choices=("node0", "node1"), help="which box this is, when its link isn't set up yet")
    ap.add_argument("--conf", help="your serving configuration (default ~/tfp/run/conf/my.conf)")
    a = ap.parse_args(argv)
    d = Doctor(box or Box(), a.role, os.path.expanduser(a.conf) if a.conf else None)
    d.run()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(json.dumps({"role": d.role, "system": d.system, "next": d.next(), "words": d.last_words(), "checks": d.out}, indent=1) if a.json else text(d))
    return 1 if any(r["status"] == "bad" for r in d.out) else 0


if __name__ == "__main__":
    sys.exit(main())

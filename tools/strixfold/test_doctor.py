"""doctor.py on made-up boxes: each thing that goes wrong for a newcomer is named, with what to do, and nothing is changed;
two fresh boxes get to a ready pair by doing only what comes next; what it names is in the docs and the scripts.
Run: python3 -m unittest discover -s tools/strixfold (no box, no network)."""
import contextlib
import http.server
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doctor  # noqa: E402

REPO = doctor.ROOT
HOME = "/home/pat"
SHA = "a" * 64
G = 1000 ** 3
PINS = {"checkpoint_files.json": [{"path": ".gitattributes", "bytes": 9}, {"path": "model-1.safetensors", "bytes": 200 * G}, {"path": "config.json", "bytes": 10}],
        "original_files.json": [{"path": "model-00001-of-00048.safetensors", "bytes": 5}, {"path": "model-00047-of-00048.safetensors", "bytes": 100 * G},
                                {"path": "model-00048-of-00048.safetensors", "bytes": 100 * G}]}
MODEL, ENGRAM, BUNDLE = f"{HOME}/models/tf-port/libertai-2.9bpw", f"{HOME}/models/tf-port/ds1017-1bae0e10", f"{HOME}/models/tf-port/inputs-tp2-c178a012"
DEP, CONF = f"{HOME}/tfp/deploy/current", f"{HOME}/tfp/run/conf/my.conf"
SITE = f"{HOME}/venvs/tfp-inputs/lib/python3.13/site-packages"
TSITE = f"{HOME}/venvs/triton/lib/python3.13/site-packages"
CHECKPOINT = {f"{MODEL}/model-1.safetensors": 200 * G, f"{MODEL}/config.json": 10}
SHARDS = {f"{ENGRAM}/model-00047-of-00048.safetensors": 100 * G, f"{ENGRAM}/model-00048-of-00048.safetensors": 100 * G}
PACKAGES = ("/usr/bin/git", "/usr/bin/rsync", "/usr/bin/wget", "/usr/bin/curl", "/usr/lib/python3.13/ensurepip")
INPUT_VENV = [f"{HOME}/venvs/tfp-inputs/bin/python", SITE] + [f"{SITE}/{w}.dist-info" for w in doctor.VENV]
TRITON_VENV = [f"{HOME}/venvs/triton/bin/python", TSITE, f"{TSITE}/{doctor.TRITON}.dist-info"]
BUILT = [f"/repo/zig-out/{x}" for x in ("bin/tf-dsv41-rank-cache", "bin/tf-dsv41-check", "native-hip/bin/tensorfold-native", "native-hip/bin/tf-dsv41-lanes")]
DEPLOYED = [f"{DEP}/{x}" for x in ("native-hip/bin/tensorfold-native", "native-hip/bin/tf-dsv41-lanes", "ds-kit/engram.json")]
HALO = 'PRETTY_NAME="AMD Ryzen AI Developer Platform 1 (rex)"\nVERSION_ID="1"\nID=debian\n'
GLUE = json.dumps({"programs": [{"program": "tf-dsv41-hip-smoke", "exit": 0}, {"program": "tf-dsv41-hip-glue-test", "exit": 1}], "all_passed": False})
UBUNTU = 'PRETTY_NAME="Ubuntu 26.10"\nVERSION_ID="26.10"\nID=ubuntu\n'
FEDORA = 'PRETTY_NAME="Fedora Linux 44 (Server)"\nVERSION_ID=44\nID=fedora\n'
KEYS = "        Subject: CN=Debian Secure Boot CA\n"
MANIFEST = json.dumps({"rank_caches": [{"rank": r, "file": f"rank{r}of2-x.bin", "bytes": 106 * G} for r in (0, 1)]})
CACHES = [f"{BUNDLE}/rank{r}/rank-cache/rank{r}of2-x.bin" for r in (0, 1)]  # as prepare_inputs.py writes them (AsTheDocsSay)
RECEIPT = f"{HOME}/tfp/arch/receipts/20261101T000000Z/run.json"
MOVED = "0" * 40 + " " + "c" * 40 + " Pat <pat@x> 1700000000 +0000\tclone: from x\n"


class Fake:
    """A box that is ready, node1; each test breaks one thing."""
    def __init__(self, n=2):
        self.home, self.host, self.root, self.ran, self.n = HOME, "pats-box", "/repo", [], n
        peer = 1 if n == 2 else 2
        self.files = {
            "/proc/cpuinfo": "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n", "/proc/meminfo": "MemTotal: 130495064 kB\nMemAvailable: 128536208 kB\n",
            "/etc/os-release": 'PRETTY_NAME="Debian GNU/Linux 13 (trixie)"\nVERSION_ID="13"\nID=debian\n', "/proc/locks": "1: FLOCK ADVISORY WRITE 5 00:1c:999 0 EOF\n",
            "/sys/class/kfd/kfd/topology/nodes/1/properties": "gfx_target_version 110501\n", "/sys/class/drm/card0/device/mem_info_gtt_total": "133143986176\n",
            "/sys/class/drm/card0/device/mem_info_vram_total": "1073741824\n", "/sys/class/dmi/id/product_name": "Box\n", "/sys/class/dmi/id/bios_version": "1.0\n", "/dev/kfd": "",
            "/etc/apt/sources.list.d/ryai.sources": "URIs: https://debs.ryai.dev/\n", f"/usr/lib/modules/{doctor.KERNEL}": "",
            f"{HOME}/opt/zig/zig": "x", f"{HOME}/tfp/tri/kit/gfx1151/aot.json": "{}", f"{BUNDLE}/inputs_manifest.json": MANIFEST, "/proc/7/comm": "bash\n",
            f"{HOME}/tfp/tri/kit/gfx1151/build_log.json": '[{"ok": true}]', RECEIPT: '{"all_passed": true}', "/repo/.git/logs/HEAD": MOVED, "/sys/firmware/efi": "",
            **{f: "x" for f in CACHES},
            f"{BUNDLE}.verify-rank{n - 1}-pats-box.json": json.dumps({"status": "pass", "manifest_sha256": SHA}),
            CONF: f"MASTER_IP=10.40.0.1   # x\nTP_STRIPE=10.40.1.1\nBUNDLE='$HOME/models/tf-port/inputs-tp2-c178a012'\nBUNDLE_SHA256={SHA}   # mine\nPEAK_GIB=101\nSERVE_PORT=18080\n",
            f"{DEP}/DEPLOY.json": json.dumps({"commit": "c" * 40})}
        self.files.update({f: "x" for f in (*PACKAGES, "/usr/bin/python3.13", *INPUT_VENV, *TRITON_VENV, *BUILT, *DEPLOYED)})
        for name, files in PINS.items():
            self.files[f"/repo/tools/dsv41-hip/assets/{name}"] = json.dumps({"files": files})
        self.sizes, self.built = {**CHECKPOINT, **SHARDS, **{f: 106 * G for f in CACHES}}, 1700000100
        for k in (0, 1):
            self.files.update({f"/sys/class/net/thunderbolt{k}": "", f"/sys/class/net/thunderbolt{k}/operstate": "up\n", f"/sys/class/net/thunderbolt{k}/mtu": "1500\n"})
        self.addr = {"thunderbolt0": f"10.40.0.{n}", "thunderbolt1": f"10.40.1.{n}"}
        self.alive, self.ssh_ok, self.ssh_host = {f"10.40.0.{peer}", f"10.40.1.{peer}"}, True, "10.40.0.1"
        self.have, self.member, self.space, self.sb, self.kern = {"render", "video"}, {"render", "video"}, 900 * G, b"\6\0\0\0\1", doctor.KERNEL
        self.pkgs = "hi |amdrocm-core10.0-gfx1151|10.0.0-4\nhi |amdrocm-core-sdk10.0-gfx1151|10.0.0-4\nhi |therock-gfx1151|10.0\n"
        self.lib, self.asleep, self.kfd, self.health = "libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7", "masked\nmasked\n", True, 200
        self.keys, self.zig, self.gone, self.root_now, self.win = KEYS + f"        Subject: CN={doctor.KERNEL_KEY}\n", f"{HOME}/opt/zig-x86_64-linux-0.17.0/zig", set(), False, False

    def read(self, p): return self.files.get(p)
    def raw(self, p, n=64): return self.sb
    def sha256(self, p): return SHA if p in self.files else None
    def size(self, p): return self.sizes.get(p, 1) if p in self.files or p in {**CHECKPOINT, **SHARDS} and p in self.sizes else None
    def inode(self, p): return 999 + 1 if p == doctor.LOCK else None
    def real(self, p): return self.zig
    def mtime(self, p): return self.built if p in self.files else None
    def free(self, p): return self.space
    def usable(self, p): return self.kfd
    def groups(self): return self.have, self.member
    def kernel(self): return self.kern
    def windows(self): return self.win
    def root_user(self): return self.root_now
    def get(self, url): return self.health

    def glob(self, pat):
        import fnmatch
        if "SecureBoot" in pat:
            return ["sb"] if self.sb else []
        return sorted(p for p in self.files if fnmatch.fnmatch(p, pat) and p.count("/") == pat.count("/"))

    def run(self, argv, timeout=8):
        assert argv[0] in doctor.COMMANDS, argv
        self.ran.append(argv)
        if argv[0] in self.gone:
            return 127, ""
        if argv[0] == "ip":
            return 0, json.dumps([{"ifname": k, "addr_info": [{"local": v}] if v else []} for k, v in self.addr.items()])
        if argv[0] == "ping":
            return (0 if argv[-1] in self.alive else 1), ""
        if argv[0] == "ssh":
            if argv[1] == "-G":
                return 0, f"user pat\nhostname {self.ssh_host}\nport 22\n"
            assert argv[-2:] == ["node0", "true"] and argv[:-2] == doctor.SSH
            return (0 if self.ssh_ok else 255), ""
        if argv[0] == "mokutil":
            return 0, self.keys if argv[1] == "--list-enrolled" else KEYS
        return 0, {"dpkg-query": self.pkgs, "/sbin/ldconfig": self.lib, "systemctl": self.asleep}[argv[0]]


def look(box, **k):
    d = doctor.Doctor(box, **k)
    d.run()
    return d, {r["title"]: r for r in d.out}


class Ready(unittest.TestCase):
    def test_a_ready_box_has_nothing_to_do(self):
        for n, role in ((2, "node1"), (1, "node0")):
            for system in (None, HALO):
                b = Fake(n)
                if system:
                    b.files["/etc/os-release"] = system
                d, r = look(b)
                self.assertEqual(d.role, role)
                self.assertIsNone(d.next(), d.next())
                self.assertIn("first_light.sh up", doctor.text(d))
                self.assertFalse([x for x in d.out if "could not be checked" in x["detail"]], d.out)
        self.assertIn("SSH to node0", look(Fake(2))[1])
        self.assertNotIn("SSH to node0", look(Fake(1))[1])

    def test_it_names_nobody_and_runs_only_what_reads(self):
        b = Fake()
        b.files["/etc/os-release"] = 'PRETTY_NAME="pats-box edition"\n'
        t = doctor.text(look(b)[0])
        for private in ("pats-box", "/home/pat", "pat "):
            self.assertNotIn(private, t)
        self.assertTrue({a[0] for a in b.ran} <= set(doctor.COMMANDS))
        with open(doctor.__file__) as f:
            src = f.read()
        for word in ("sudo -", "os.remove", "os.mkdir", "makedirs", "shutil", "\"wb\"", "\"w\"", "'w'", "shell=True", "os.system"):
            self.assertNotIn(word, src, word)
        self.assertRaises(ValueError, doctor.Box().run, ["rm", "-rf", "x"])
        for o in ("UpdateHostKeys=no", "ControlMaster=no", "AddKeysToAgent=no", "PermitLocalCommand=no", "BatchMode=yes", "StrictHostKeyChecking=yes"):
            self.assertIn(o, doctor.SSH)

    def test_a_host_name_never_changes_a_command_it_gives(self):
        for host in ("node1", "rex", "video", "render"):
            b = Fake()
            b.host, b.kern, b.kfd, b.have, b.member = host, "6.12.57+deb13-amd64", False, set(), set()
            d, r = look(b, role="node1")
            self.assertIn('usermod -aG render,video "$USER"', r["Groups"]["fix"])
            self.assertFalse([x for x in d.out if "<this box>" in x["fix"]])

    def test_json_and_exit_status(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(doctor.main(["--json"], Fake()), 0)
        j = json.loads(out.getvalue())
        self.assertEqual((j["role"], j["system"], j["next"]), ("node1", "debian", None))
        b = Fake()
        b.files["/proc/meminfo"] = "MemTotal: 33554432 kB\n"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(doctor.main([], b), 1)


class Walk(unittest.TestCase):
    """Two fresh boxes and somebody who only ever does what comes next: each thing once, in the guide's order, to a ready pair."""
    def fresh(self, n, halo):
        b = Fake(n)
        keep = {f for f in b.files if f.startswith(("/proc", "/sys/class/kfd", "/sys/class/drm", "/sys/class/dmi", "/sys/firmware", "/repo/tools", "/repo/.git", "/etc/os-release"))}
        b.files = {f: v for f, v in b.files.items() if f in keep}
        b.files["/sys/class/drm/card0/device/mem_info_gtt_total"] = "66571993088\n"
        b.files["/dev/kfd"] = ""
        b.sizes, b.addr, b.alive, b.ssh_ok, b.ssh_host = {}, {}, set(), False, "node0"
        b.have, b.member, b.kfd, b.pkgs, b.lib, b.asleep, b.keys = set(), set(), False, "", "", "static\nstatic\n", KEYS
        if halo:
            b.files["/etc/os-release"] = HALO
            ready = Fake(n)
            b.pkgs, b.lib = ready.pkgs, ready.lib
            b.files["/sys/class/drm/card0/device/mem_info_gtt_total"] = "133143986176\n"
        else:
            b.kern = "6.12.57+deb13-amd64"
            b.files.pop("/sys/class/kfd/kfd/topology/nodes/1/properties")
            b.files.pop("/dev/kfd")
        for k in (0, 1):  # the cables are in; thunderbolt_net makes the cards, down until an address is set
            b.files.update({f"/sys/class/net/thunderbolt{k}": "", f"/sys/class/net/thunderbolt{k}/operstate": "down\n", f"/sys/class/net/thunderbolt{k}/mtu": "1500\n"})
        return b

    def restart(self, b):
        sb_lets = doctor.KERNEL_KEY in b.keys or "rex" in b.files["/etc/os-release"]
        if f"/usr/lib/modules/{doctor.KERNEL}" in b.files:
            self.assertTrue(sb_lets, "restarted into a kernel Secure Boot refuses")
            b.kern = doctor.KERNEL
            b.files.update({"/sys/class/kfd/kfd/topology/nodes/1/properties": "gfx_target_version 110501\n", "/dev/kfd": ""})
        if b.grub:
            b.files["/sys/class/drm/card0/device/mem_info_gtt_total"] = "133143986176\n"

    def up(self, boxes):
        both = all(len([a for a in b.addr.values() if a]) == 2 for b in boxes.values())
        for n, b in boxes.items():
            for k in (0, 1):
                b.files[f"/sys/class/net/thunderbolt{k}/operstate"] = "up\n" if both else "down\n"
            b.alive = {f"10.40.{k}.{3 - b.n}" for k in (0, 1)} if both else set()

    def do(self, b, boxes, r):
        """What a careful reader does about `r` on box `b`: the named step's commands."""
        t, g, node0, node1 = r["title"], r["guide"], boxes["node0"], boxes["node1"]
        if t == "Kernel":
            if "step 3" in g:
                b.files["/etc/apt/sources.list.d/ryai.sources"] = "URIs: https://debs.ryai.dev/\n"
            elif "step 4" in g:
                b.files[f"/usr/lib/modules/{doctor.KERNEL}"] = ""
            elif "steps 5 and 6" in g:
                b.grub, b.asleep = True, "masked\nmasked\n"
                b.keys += f"        Subject: CN={doctor.KERNEL_KEY}\n"
                self.restart(b)
            else:
                self.restart(b)
        elif t == "GPU memory":
            b.grub = True
            if "end of step 6" not in r["fix"]:
                self.restart(b)
        elif t == "Sleep":
            b.asleep = "masked\nmasked\n"
        elif t == "ROCm":
            ready = Fake(b.n)
            b.pkgs, b.lib = ready.pkgs, ready.lib
        elif t == "Groups":
            b.have, b.member, b.kfd = {"render", "video"}, {"render", "video"}, True
        elif t.startswith("thunderbolt") and r["status"] == "todo":
            b.addr = {"thunderbolt0": f"10.40.0.{b.n}", "thunderbolt1": f"10.40.1.{b.n}"}
            self.up(boxes)
        elif t == "SSH to node0":
            b.ssh_host, b.ssh_ok = "10.40.0.1", True
        elif t == "Packages":
            b.files.update({f: "x" for f in (*PACKAGES, "/usr/bin/python3.13")})
        elif t == "Input tools":
            for x in (node0, node1):
                x.files.update({f: "x" for f in INPUT_VENV})
        elif t == "Zig":
            b.files[f"{HOME}/opt/zig/zig"] = "x"
        elif t == "Triton":
            b.files.update({f: "x" for f in TRITON_VENV})
        elif t == "Triton kit":
            b.files[f"{HOME}/tfp/tri/kit/gfx1151/aot.json"] = "{}"
        elif t == "Build":
            b.files.update({f: "x" for f in BUILT})
        elif t == "GPU tests":
            b.files[RECEIPT] = '{"all_passed": true}'
        elif t in ("Checkpoint", "Engram shards"):
            for x in (node1, node0):
                x.sizes.update({**CHECKPOINT, **SHARDS})
        elif t == "Inputs":
            self.assertIn(BUILT[0], node1.files, "prepare_inputs.sh before the build")
            for x, caches in ((node1, CACHES), (node0, CACHES[:1])):
                x.files[f"{BUNDLE}/inputs_manifest.json"] = MANIFEST
                x.files.update({f: "x" for f in caches})
                x.sizes.update({f: 106 * G for f in caches})
        elif t == "Check of the inputs":
            if b.n == 1:
                self.assertIn(f"{DEP}/DEPLOY.json", b.files, "node0's check before its tools are deployed")
            b.files[f"{BUNDLE}.verify-rank{b.n - 1}-pats-box.json"] = json.dumps({"status": "pass", "manifest_sha256": SHA})
        elif t == "Deployed":
            for x in (node0, node1):
                x.files.update({f: "x" for f in DEPLOYED})
                x.files[f"{DEP}/DEPLOY.json"] = json.dumps({"commit": "c" * 40})
        elif t == "Configuration":
            self.assertIn(f"{DEP}/DEPLOY.json", node1.files, "a configuration before the deployed serve.conf")
            for x in (node0, node1):
                x.files[CONF] = f"BUNDLE_SHA256={SHA}\n"
        else:
            self.fail(f"nothing a reader can do about {r}")

    def walk(self, halo):
        boxes = {"node0": self.fresh(1, halo), "node1": self.fresh(2, halo)}
        for b in boxes.values():
            b.grub = False
        here, done, log = "node1", [], []
        for _ in range(80):
            d, _r = look(boxes[here], role=here)
            n = d.next()
            log.append((here, n and (n["status"], n["title"])))
            if n is None or n["status"] == "wait":
                other = "node0" if here == "node1" else "node1"
                o = look(boxes[other], role=other)[0].next()
                if n is None and o is None:
                    return done, log
                if o is None or (n and o["status"] == "wait"):  # the other has nothing to do: the wait says what to do there
                    key = (here, n["title"], n["detail"], n["guide"])
                    self.assertNotIn(key, done, log)
                    done.append(key)
                    self.do(boxes[here], boxes, n)
                else:
                    here = other
                continue
            self.assertEqual(n["status"], "todo", (n, log))
            self.assertTrue(n["fix"], n)
            key = (here, n["title"], n["detail"], n["guide"])
            self.assertNotIn(key, done, f"twice: {key}")
            done.append(key)
            self.do(boxes[here], boxes, n)
        self.fail(f"never ready: {log}")

    def test_two_fresh_bosgame_m5s(self):
        done, log = self.walk(False)
        kernel = [k for k in done if k[1] == "Kernel" and k[0] == "node1"]
        self.assertEqual(len(kernel), 3)  # its repository, the kernel, then steps 5 and 6 with the one restart
        self.assertLess(len(done), 45)

    def test_two_fresh_ryzen_ai_halos(self):
        done, log = self.walk(True)
        self.assertFalse([k for k in done if k[1] in ("Kernel", "ROCm", "GPU memory")], done)


class Wrong(unittest.TestCase):
    def with_(self, change):
        b = Fake()
        change(b)
        return b

    def one(self, title, status, words, change, role=None, n=2, **k):
        b = Fake(n)
        change(b)
        d, r = look(b, role=role, **k)
        self.assertIn(title, r, sorted(r))
        self.assertEqual(r[title]["status"], status, r[title])
        self.assertIn(words, " ".join((r[title]["detail"], r[title]["fix"], r[title]["guide"])))
        return d, r

    def test_the_box(self):
        self.one("Chip", "bad", "Ryzen AI Max+ 395", lambda b: b.files.update({"/proc/cpuinfo": "model name : Ryzen 7 5800X\n", "/sys/class/kfd/kfd/topology/nodes/1/properties": "gfx_target_version 100300\n"}))
        self.one("Chip", "todo", "no gfx1151 GPU", lambda b: b.files.update({"/sys/class/kfd/kfd/topology/nodes/1/properties": "gfx_target_version 0\n"}))
        d, r = self.one("Chip", "note", "comes with the kernel", lambda b: (b.files.update({"/sys/class/kfd/kfd/topology/nodes/1/properties": "gfx_target_version 0\n"}), setattr(b, "kern", "6.12.57+deb13-amd64")))
        self.assertEqual(d.next()["title"], "Kernel")
        self.one("Chip", "note", "16 cores", lambda b: b.files.update({"/proc/cpuinfo": "model name : AMD RYZEN AI MAX 385 w/ Radeon 8050S\n"}))
        for mem, vram in ((98000000, 32), (65000000, 64), (32000000, 96)):
            self.one("Memory", "bad", "smallest", lambda b: b.files.update({"/proc/meminfo": f"MemTotal: {mem} kB\nMemAvailable: 9 kB\n", "/sys/class/drm/card0/device/mem_info_vram_total": f"{vram * 2 ** 30}\n"}))
        self.one("Memory", "bad", "needs 128 GB", lambda b: b.files.update({"/proc/meminfo": "MemTotal: 98000000 kB\nMemAvailable: 9 kB\n"}))
        self.one("Memory", "bad", "If this one has it", lambda b: (b.files.update({"/proc/meminfo": "MemTotal: 65000000 kB\nMemAvailable: 9 kB\n"}), b.files.pop("/sys/class/drm/card0/device/mem_info_vram_total")))
        self.one("GPU memory", "todo", "step 5", lambda b: b.files.update({"/sys/class/drm/card0/device/mem_info_gtt_total": "66571993088\n"}))
        self.one("Secure Boot", "note", "no longer checked", lambda b: setattr(b, "sb", b"\6\0\0\0\0"))
        self.one("Sleep", "todo", "step 5", lambda b: setattr(b, "asleep", "static\nstatic\n"))
        self.one("System", "bad", "Linux on the box itself", lambda b: setattr(b, "kern", "6.6.87.2-microsoft-standard-WSL2"))
        self.one("User", "note", "without sudo", lambda b: setattr(b, "root_now", True))

    def test_the_kernel_and_secure_boot(self):
        old = "6.12.57+deb13-amd64"
        def none(b):
            b.kern = old
            b.files.pop(f"/usr/lib/modules/{doctor.KERNEL}")
        d, r = self.one("Kernel", "todo", "only restart is at the end of step 6", none)
        self.assertEqual(d.next()["title"], "Kernel")
        self.one("Kernel", "todo", "step 3", lambda b: (none(b), b.files.pop("/etc/apt/sources.list.d/ryai.sources")))
        self.one("Kernel", "todo", "only restart", lambda b: (setattr(b, "kern", old), setattr(b, "keys", KEYS)))
        self.one("Kernel", "todo", "only restart", lambda b: (setattr(b, "kern", old), b.gone.add("mokutil")))  # can't tell: the safe answer
        self.one("Kernel", "todo", "only restart", lambda b: (setattr(b, "kern", old), setattr(b, "sb", None), setattr(b, "keys", KEYS)))
        self.one("Kernel", "todo", "Restart the box", lambda b: (setattr(b, "kern", old), setattr(b, "sb", None), setattr(b, "keys", KEYS), b.files.pop("/sys/firmware/efi")))  # started the old way
        self.one("Kernel", "todo", "Restart the box", lambda b: setattr(b, "kern", old))
        self.one("Kernel", "todo", "Restart the box", lambda b: (setattr(b, "kern", old), setattr(b, "keys", KEYS), setattr(b, "sb", b"\6\0\0\0\0")))
        self.one("Kernel", "note", "owners run others", lambda b: setattr(b, "kern", "7.3.1+rex-amd64"))
        _, r = self.one("GPU memory", "todo", "end of step 6", lambda b: (setattr(b, "kern", old), b.files.update({"/sys/class/drm/card0/device/mem_info_gtt_total": "66571993088\n"})))

    def test_another_system_gets_its_own_guide_and_no_debian_commands(self):
        for rel, guide, vendor in ((UBUNTU, "FRAMEWORK-DESKTOP", "Framework\n"), (FEDORA, "FEDORA-PODMAN", "Framework\n"), ('PRETTY_NAME="Arch Linux"\nID=arch\n', "HARDWARE", "x"),
                                   (UBUNTU, "HARDWARE", "GMKtec\n"), (UBUNTU.replace("26.10", "24.04"), "HARDWARE", "Framework\n")):
            b = Fake()
            b.files.update({"/etc/os-release": rel, "/sys/class/dmi/id/sys_vendor": vendor})
            b.kern, b.pkgs, b.lib, b.asleep = "7.3.0-10-generic", "", "", "static\nstatic\n"
            b.files["/sys/class/drm/card0/device/mem_info_gtt_total"] = "66571993088\n"
            b.files.pop(f"{HOME}/venvs/tfp-inputs/bin/python")
            b.files.pop("/usr/bin/python3.13")
            b.files.pop(f"{DEP}/DEPLOY.json")
            d, r = look(b)
            for x in d.out:
                self.assertNotIn("therock", x["fix"], x)
                if x["title"] != "Memory":
                    self.assertNotIn("BOSGAME", x["guide"] + x["fix"], x)
            for t in ("System", "GPU memory", "Sleep"):
                self.assertIn(guide, r[t]["guide"], r[t])
            self.assertEqual(r["Kernel"]["status"], "note")
            self.assertEqual((d.next()["status"], d.next()["guide"].split("/")[-1].split(".")[0]), ("guide", guide))  # it leads, these lines only check
            b.files["/proc/meminfo"] = "MemTotal: 33554432 kB\n"
            self.assertEqual(look(b)[0].next()["title"], "Memory")
            if "Fedora" in rel:
                self.assertNotIn("Zig", r)
                self.assertEqual(r["ROCm and the tools"]["status"], "note")
            elif guide == "FRAMEWORK-DESKTOP":
                self.assertIn("section 6", r["Python"]["guide"])
                b.files["/proc/meminfo"] = Fake().files["/proc/meminfo"]
                self.assertIn("sections 1 to 8", look(b)[0].next()["fix"])
        b = Fake()
        b.files.update({"/etc/os-release": UBUNTU, "/run/.containerenv": "", "/opt/python/cpython-3.13.18/lib/python3.13/ensurepip": ""})
        b.files.pop("/usr/lib/python3.13/ensurepip")
        b.pkgs = ""
        b.gone.add("systemctl")
        d = look(b)[0]
        self.assertEqual((d.system, [x for x in d.out if x["status"] in ("todo", "wait", "bad")]), ("container", []))

    def test_software(self):
        self.one("ROCm", "todo", "step 7", lambda b: (setattr(b, "pkgs", ""), setattr(b, "lib", "")))
        self.one("ROCm", "note", "10.1", lambda b: setattr(b, "pkgs", "ii |amdrocm-core10.1-gfx1151|10.1.0-2\nii |amdrocm-core-sdk10.1-gfx1151|10.1.0-2\n"))
        self.one("ROCm", "todo", "SDK", lambda b: setattr(b, "pkgs", "ii |amdrocm-core10.0-gfx1151|10.0.0-4\n"))
        self.one("ROCm's libraries", "todo", "step 7", lambda b: setattr(b, "lib", "nothing"))
        self.one("Groups", "todo", "Log out", lambda b: (setattr(b, "have", {"users"}), setattr(b, "kfd", False)))
        d, r = self.one("Groups", "todo", "usermod", lambda b: (setattr(b, "have", set()), setattr(b, "member", set()), setattr(b, "kfd", False)))
        self.assertEqual(r["The GPU device"]["status"], "todo")
        self.one("The GPU device", "bad", "ls -l", lambda b: setattr(b, "kfd", False))
        self.assertNotIn("Groups", look(Fake())[1])  # it can be used: nothing to say about groups
        self.one("ROCm is not held", "note", "apt-mark", lambda b: setattr(b, "pkgs", b.pkgs.replace("hi |", "ii |")))
        self.assertNotIn("ROCm is not held", look(Fake())[1])
        for rel, where in ((HALO, "step 2a"), (None, "BOSGAME-M5.md, step 7")):
            self.one("Groups", "todo", where, lambda b: (setattr(b, "have", set()), setattr(b, "member", set()), setattr(b, "kfd", False), rel and b.files.update({"/etc/os-release": rel})))

    def test_the_link(self):
        def gone(b):
            for p in [p for p in b.files if "thunderbolt" in p]:
                del b.files[p]
            b.addr, b.alive = {}, set()
        self.one("Cables", "todo", "two cables", gone)
        self.one("Cables", "todo", "1 of 2", lambda b: ([b.files.pop(p) for p in [p for p in b.files if "thunderbolt1" in p]], b.addr.pop("thunderbolt1")))
        down = lambda b: [b.files.update({f"/sys/class/net/thunderbolt{k}/operstate": "down\n"}) for k in (0, 1)]  # noqa: E731
        _, r = self.one("thunderbolt0", "todo", "no link address", lambda b: (down(b), b.addr.update(thunderbolt0=None, thunderbolt1=None)), role="node1")  # a fresh M5
        self.assertEqual(r["thunderbolt1"]["status"], "todo")
        self.one("thunderbolt0", "wait", "other box", down)
        self.one("The other box at 10.40.0.1", "wait", "crossed", lambda b: b.alive.clear())
        self.one("thunderbolt0", "todo", "not the link's", lambda b: b.addr.update(thunderbolt0="192.168.7.9"))
        self.one("Role", "bad", "node0's", lambda b: b.addr.update(thunderbolt0="10.40.0.1", thunderbolt1="10.40.1.1"), role="node1")
        own = self.with_(lambda b: (b.files.update({CONF: f"MASTER_IP=192.168.1.10\nTP_STRIPE=192.168.2.10\nBUNDLE_SHA256={SHA}\n"}), b.addr.update(thunderbolt0="192.168.1.10", thunderbolt1="192.168.2.10")))
        d, _r = look(own)
        self.assertEqual((d.role, [a[-1] for a in own.ran if a[0] == "ping"]), ("node0", []))  # its own addresses: nobody's router is pinged
        own = self.with_(lambda b: (b.files.update({CONF: f"MASTER_IP=192.168.1.10\nTP_STRIPE=192.168.2.10\nBUNDLE_SHA256={SHA}\n"}), b.addr.update(thunderbolt0="192.168.1.11", thunderbolt1="192.168.2.11"),
                                    setattr(b, "alive", {"192.168.1.10", "192.168.2.10"}), setattr(b, "ssh_host", "192.168.1.10")))
        self.assertIsNone(look(own)[0].next())
        odd = Fake()
        plain = odd.run
        odd.run = lambda a, timeout=8: (0, json.dumps([{}, {"ifname": "lo"}, {"ifname": "thunderbolt0", "addr_info": [{"local": "10.40.0.2"}, {}]}, {"ifname": "thunderbolt1", "addr_info": [{"local": "10.40.1.2"}]}])) if a[0] == "ip" else plain(a, timeout)
        self.assertIsNone(look(odd)[0].next())
        self.one("The other box at 10.40.1.1", "wait", "may lack that address", lambda b: b.alive.discard("10.40.1.1"))  # never "swap" before both ends are looked at
        self.one("The other box at 10.40.0.1", "note", "ping", lambda b: b.gone.add("ping"))
        self.one("Cables", "note", "iproute2", lambda b: b.gone.add("ip"))
        self.one("thunderbolt0", "note", "MTU 9000", lambda b: b.files.update({"/sys/class/net/thunderbolt0/mtu": "9000\n"}))
        self.one("Addresses", "bad", "node0 on one cable", lambda b: b.addr.update(thunderbolt1="10.40.1.1"))
        swapped, _ = look(self.with_(lambda b: b.addr.update(thunderbolt0="10.40.1.2", thunderbolt1="10.40.0.2")))
        self.assertIsNone(swapped.next())  # named in the order the links came up: fine
        one = self.with_(lambda b: (b.files.update({CONF: f"MASTER_IP=10.40.0.1\nTP_STRIPE=\nBUNDLE_SHA256={SHA}\n"}), [b.files.pop(p) for p in [p for p in b.files if "thunderbolt1" in p]],
                                    b.addr.pop("thunderbolt1")))
        self.assertIsNone(look(one)[0].next())  # one cable, as its configuration says
        self.one("Cables", "note", "RDMA", lambda b: b.files.update({CONF: f"TRANSPORT=rdma\nBUNDLE_SHA256={SHA}\n"}))
        self.one("SSH to node0", "todo", "without a password", lambda b: setattr(b, "ssh_ok", False))
        b = self.with_(lambda b: setattr(b, "ssh_host", "192.168.1.7"))
        self.assertIn("Change its entry", look(b)[1]["SSH to node0"]["fix"])
        b = self.with_(lambda b: setattr(b, "ssh_host", "node0"))
        self.assertIn("add the Host node0 entry", look(b)[1]["SSH to node0"]["fix"])
        self.assertFalse([a for a in b.ran if a[0] == "ssh" and a[-1] == "true"])  # a node0 that isn't the other box is never asked

    def test_tools(self):
        fresh = lambda b: b.files.pop(f"{DEP}/DEPLOY.json")  # noqa: E731
        self.one("Packages", "todo", "python3-venv", lambda b: (fresh(b), b.files.pop("/usr/lib/python3.13/ensurepip")))
        self.one("Packages", "note", "not needed to run", lambda b: b.files.pop("/usr/bin/git"))
        self.one("Packages", "todo", "curl", lambda b: (fresh(b), b.files.pop("/usr/bin/curl")))
        self.one("curl", "todo", "/health", lambda b: b.files.pop("/usr/bin/curl"), n=1)  # node0 needs it to run
        self.one("Triton kit", "todo", "2 of its variants", lambda b: (fresh(b), b.files.update({f"{HOME}/tfp/tri/kit/gfx1151/build_log.json": '[{"ok": true}, {"ok": false}, {"ok": false}]'})))
        pulled = MOVED + "c" * 40 + " " + "d" * 40 + " Pat <pat@x> 1700000200 +0000\tpull: Fast-forward\n"
        self.one("Build", "note", "before you deploy", lambda b: (fresh(b), b.files.update({"/repo/.git/logs/HEAD": pulled})))
        self.one("Build", "ok", "there", lambda b: (fresh(b), setattr(b, "built", 1699999999)))  # never moved since its clone
        self.one("GPU tests", "todo", "not run", lambda b: (fresh(b), b.files.pop(RECEIPT)))
        self.one("GPU tests", "todo", "didn't finish", lambda b: (fresh(b), b.files.update({RECEIPT: '{ "host": "x", "programs": ['})))
        self.one("GPU tests", "note", "clock check", lambda b: (fresh(b), b.files.update({RECEIPT: GLUE})))
        self.one("GPU tests", "bad", "tf-dsv41-hip-smoke", lambda b: (fresh(b), b.files.update({RECEIPT: GLUE.replace('"exit": 0', '"exit": 2')})))
        self.one("GPU tests", "bad", "the last run", lambda b: (fresh(b), b.files.update({RECEIPT: '{"all_passed": false}'})))
        for odd in ("[]", '"x"', '{"programs": [1], "all_passed": false}', "null"):
            d, r = look(self.with_(lambda b: (fresh(b), b.files.update({RECEIPT: odd}))))
            self.assertFalse([x for x in d.out if "could not be checked" in x["detail"]], odd)
        self.one("Input tools", "todo", "didn't finish", lambda b: (fresh(b), b.files.pop(f"{SITE}/torch-2.9.0+cpu.dist-info")))
        self.one("Input tools", "todo", "Python 3.13", lambda b: (fresh(b), [b.files.pop(p) for p in [p for p in b.files if p.startswith(SITE)]]))
        self.one("Input tools", "wait", "on node1", lambda b: (fresh(b), b.files.pop(f"{HOME}/venvs/tfp-inputs/bin/python")), n=1)
        self.one("Zig", "todo", "link", lambda b: (fresh(b), b.files.pop(f"{HOME}/opt/zig/zig")))
        self.one("Zig", "note", "0.17.0", lambda b: (fresh(b), setattr(b, "zig", f"{HOME}/opt/zig-x86_64-linux-0.16.0/zig")))
        self.one("Triton kit", "todo", "not finished", lambda b: (fresh(b), b.files.pop(f"{HOME}/tfp/tri/kit/gfx1151/aot.json")))
        self.one("Triton", "todo", "lacks", lambda b: (fresh(b), b.files.pop(f"{HOME}/tfp/tri/kit/gfx1151/aot.json"), b.files.pop(f"{TSITE}/{doctor.TRITON}.dist-info")))
        for x in BUILT:
            self.one("Build", "todo", "GPU tests", lambda b: (fresh(b), b.files.pop(x)))

    def test_files_and_inputs(self):
        fresh = lambda b: b.files.pop(f"{BUNDLE}.verify-rank{b.n - 1}-pats-box.json")  # noqa: E731
        self.one("Checkpoint", "todo", "wrong size", lambda b: (fresh(b), b.sizes.update({f"{MODEL}/model-1.safetensors": 44})))
        self.one("Checkpoint", "todo", "0 of 2", lambda b: (fresh(b), [b.sizes.pop(p) for p in CHECKPOINT]))
        self.one("Checkpoint", "wait", "rsync", lambda b: (fresh(b), [b.sizes.pop(p) for p in CHECKPOINT]), n=1)
        self.one("Checkpoint", "note", "isn't needed", lambda b: [b.sizes.pop(p) for p in CHECKPOINT])
        nothing = lambda b: (fresh(b), b.sizes.clear(), [b.files.pop(p) for p in [p for p in b.files if p.startswith(BUNDLE + "/")]])  # noqa: E731
        self.one("Disk space", "bad", "make room", lambda b: (nothing(b), setattr(b, "space", 620 * G)))  # 400 + 2 x 106, and some room
        d, r = self.one("Disk space", "ok", "643 GB more", lambda b: (nothing(b), setattr(b, "space", 650 * G)))
        self.assertLess([x["title"] for x in d.out].index("Disk space"), [x["title"] for x in d.out].index("Checkpoint"))  # before the download it has to hold
        half = {f"{ENGRAM}/.cache/huggingface/download/model-0004{k}-of-00048.safetensors.abc.incomplete": 99 * G for k in (7, 8)}
        self.one("Disk space", "ok", "about 435 GB", lambda b: (nothing(b), b.sizes.update(half), b.files.update({f: "x" for f in half}), setattr(b, "space", 450 * G)))  # a download under way: what it holds counts
        copy = {f"{ENGRAM}/.model-00047-of-00048.safetensors.Ab12Cd": 99 * G}
        self.one("Disk space", "ok", "about 427 GB", lambda b: (nothing(b), b.sizes.update(copy), b.files.update({f: "x" for f in copy}), setattr(b, "space", 450 * G)), n=1)
        unmade = lambda b: (fresh(b), [b.files.pop(p) for p in [p for p in b.files if p.startswith(BUNDLE + "/")]])  # noqa: E731
        self.one("Inputs", "todo", "Wait until it has ended", lambda b: (unmade(b), b.files.update({f"{BUNDLE}/PREPARING.json": "{}", "/proc/8/cmdline": "/home/pat/venvs/tfp-inputs/bin/python\0/r/prepare_inputs.py\0x\0"})))
        self.one("Inputs", "todo", "Wait until it has ended", lambda b: (unmade(b), b.files.update({"/proc/8/cmdline": "/bin/bash\0tools/dsv41-hip/prepare_inputs.sh\0x\0"})))  # reading, its folder not there yet
        self.one("Inputs", "todo", "not made yet", lambda b: (unmade(b), b.files.update({"/proc/8/cmdline": "vim\0tools/dsv41-hip/prepare_inputs.sh\0"})))
        for cmd in ("/home/pat/venvs/tfp-inputs/bin/python\0/home/pat/venvs/tfp-inputs/bin/hf\0download\0", "rsync\0--server\0"):
            self.one("Checkpoint", "todo", "under way", lambda b: (fresh(b), [b.sizes.pop(p) for p in CHECKPOINT], b.files.update({"/proc/8/cmdline": cmd, f"{MODEL}/.model-1.safetensors.Ab12Cd": "x"})), n=2 if "hf" in cmd else 1)
        self.one("Checkpoint", "wait", "rsync", lambda b: (fresh(b), [b.sizes.pop(p) for p in CHECKPOINT], b.files.update({"/proc/8/cmdline": "rsync\0--daemon\0"})), n=1)  # some other rsync
        self.one("Inputs", "bad", "Only with their", lambda b: (unmade(b), b.files.update({f"{BUNDLE}/PREPARING.json": "{}"})))
        self.one("Inputs", "bad", "FAILED.json", lambda b: (unmade(b), b.files.update({f"{BUNDLE}/FAILED.json": "{}"})))
        for n in (1, 2):  # made and not yet checked: no more room is needed
            self.one("Disk space", "ok", "about 0 GB", lambda b: (fresh(b), setattr(b, "space", 2 * G)), n=n)
        self.one("Inputs", "wait", "cache is missing", lambda b: b.files.pop(CACHES[0]), n=1)
        self.one("Inputs", "wait", "wrong size", lambda b: b.sizes.update({CACHES[0]: 5 * G}), n=1)
        self.one("Inputs", "todo", "cache is missing", lambda b: b.files.pop(CACHES[1]))
        self.one("Inputs", "ok", "there", lambda b: b.files.pop(CACHES[0]))  # node1 doesn't run rank 0's
        self.one("Inputs", "ok", f"BUNDLE_SHA256 is {SHA}", lambda b: None)
        self.one("Inputs", "wait", "rsync", lambda b: b.files.pop(f"{BUNDLE}/inputs_manifest.json"), n=1)
        conf = lambda t: (lambda b: b.files.update({CONF: t}))  # noqa: E731
        self.one("Configuration", "todo", f"BUNDLE_SHA256={SHA}", conf("BUNDLE_SHA256=" + "b" * 64 + "\n"))
        self.one("Configuration", "todo", "is empty", conf("BUNDLE_SHA256=\n"))
        self.one("Configuration", "todo", "is empty", conf("PEAK_GIB=101\n"))
        self.one("Configuration", "wait", "from node1", conf("BUNDLE_SHA256=" + "b" * 64 + "\n"), n=1)
        self.one("Configuration", "todo", "serve.conf", lambda b: b.files.pop(CONF))
        self.one("Configuration", "bad", "--conf", lambda b: None, conf="/home/pat/mine.conf")
        self.one("Check of the inputs", "todo", "not done", fresh)
        self.one("Check of the inputs", "wait", "On node1", fresh, n=1)
        rep = lambda **k: (lambda b: b.files.update({f"{BUNDLE}.verify-rank1-pats-box.json": json.dumps(k)}))  # noqa: E731
        self.one("Check of the inputs", "todo", "other inputs", rep(status="pass", manifest_sha256="d" * 64))
        self.one("Check of the inputs", "todo", "won't start", rep(status="pass-with-skips", manifest_sha256=SHA))
        self.one("Check of the inputs", "bad", "says fail", rep(status="fail"))
        self.one("Check of the inputs", "note", "VERIFY_REPORT=0", conf(f"BUNDLE_SHA256={SHA}\nVERIFY_REPORT=0\n"))

    def test_serving(self):
        self.one("Deployed", "todo", "deploy_rank0", lambda b: b.files.pop(f"{DEP}/DEPLOY.json"))
        self.one("Deployed", "wait", "deploy_rank0", lambda b: b.files.pop(f"{DEP}/DEPLOY.json"), n=1)
        self.one("Deployed", "todo", "without the engine", lambda b: b.files.pop(DEPLOYED[0]))
        self.one("Free memory", "bad", "Ask your user", lambda b: b.files.update({"/proc/meminfo": "MemTotal: 130495064 kB\nMemAvailable: 50000000 kB\n"}))
        self.one("GPU lock", "note", "held", lambda b: b.files.update({"/proc/locks": "1: FLOCK  ADVISORY  WRITE 5 00:1c:1000 0 EOF\n"}))
        for comm in ("tf-dsv41-lanes\n", "tf-dsv41-layer\n"):
            self.one("Engine", "ok", "running", lambda b: b.files.update({"/proc/9/comm": comm}))
        other = self.with_(lambda b: (b.files.update({"/proc/9/comm": "ds4-server\n"}), b.files.pop(CONF), b.files.pop(f"{DEP}/DEPLOY.json"), b.files.pop("/usr/bin/git")))
        d, r = look(other)
        self.assertEqual((d.next()["title"], r["Engine"]["status"]), ("Packages", "bad"))  # somebody else's engine is no StrixFold set-up
        elsewhere = self.with_(lambda b: (b.files.update({"/proc/9/comm": "tf-dsv41-lanes\n"}), b.files.pop(CONF), b.sizes.clear(), b.files.pop(f"{BUNDLE}/inputs_manifest.json")))
        d, r = look(elsewhere)
        self.assertIsNone(d.next())  # it serves, from a set-up of its owner's: nothing to download
        self.assertIn("--conf", r["Configuration"]["fix"])
        for peak in ("\u00b2", "", "x"):
            self.one("Free memory", "ok", "109", lambda b: b.files.update({CONF: f"BUNDLE_SHA256={SHA}\nPEAK_GIB={peak}\n"}))
        run = lambda h: (lambda b: (b.files.update({"/proc/9/comm": "tensorfold-nati\n"}), setattr(b, "health", h)))  # noqa: E731
        self.one("Server", "ok", "18080", run(200), n=1)
        self.one("Server", "note", "503", run(503), n=1)
        self.one("Server", "note", "no answer", run(None), n=1)

    def test_an_empty_machine_never_crashes_it(self):
        b = Fake()
        b.files, b.sizes, b.addr, b.alive, b.sb, b.pkgs, b.lib, b.asleep, b.kern, b.space, b.ssh_host = {}, {}, {}, set(), None, "", "", "", "6.1", None, "node0"
        for role in (None, "node0", "node1"):
            d, r = look(b, role=role)
            self.assertFalse([x for x in d.out if "could not be checked" in x["detail"]], d.out)
            self.assertEqual(d.next()["title"], "Chip")
        b.gone = set(doctor.COMMANDS)
        self.assertFalse([x for x in look(b)[0].out if "could not be checked" in x["detail"]])


class RealBox(unittest.TestCase):
    """Box itself, on the machine the tests run on."""
    def test_reading(self):
        b = doctor.Box()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f")
            with open(p, "wb") as f:
                f.write(b"abc")
            os.symlink(p, os.path.join(d, "l"))
            self.assertEqual((b.read(p), b.raw(p, 2), b.size(p), b.size(p + "x"), b.read(p + "x")), ("abc", b"ab", 3, None, None))
            self.assertEqual(b.sha256(p), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertEqual(b.real(os.path.join(d, "l")), os.path.realpath(p))
            self.assertGreater(b.free(os.path.join(d, "not", "there")), 0)
            self.assertEqual([os.path.basename(x) for x in b.glob(os.path.join(d, "*"))], ["f", "l"])
            os.mkfifo(os.path.join(d, "pipe"))
            self.assertIsNone(b.read(os.path.join(d, "pipe")))  # never waits on one
            self.assertGreater(b.mtime(p), 0)
        have, member = b.groups()
        self.assertTrue(have <= member)

    def test_a_command_that_is_not_there_or_hangs(self):
        b = doctor.Box()
        keep = doctor.COMMANDS
        doctor.COMMANDS = keep + ("sleep", "no-such-command")
        try:
            self.assertEqual(b.run(["no-such-command"]), (127, ""))
            self.assertEqual(b.run(["sleep", "5"], timeout=0.2), (127, ""))
        finally:
            doctor.COMMANDS = keep

    def test_the_health_question_never_leaves_the_box(self):
        asked = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                asked.append((self.server.name, self.path))
                self.send_response(302 if self.path == "/away" else 200 if self.server.name == "server" else 418)
                if self.path == "/away":
                    self.send_header("Location", f"http://127.0.0.1:{servers[1].server_port}/")
                self.end_headers()

            def log_message(self, *a):
                pass
        servers = []
        for name in ("server", "proxy"):
            s = http.server.HTTPServer(("127.0.0.1", 0), H)
            s.name = name
            threading.Thread(target=s.serve_forever, daemon=True).start()
            self.addCleanup(s.server_close)
            self.addCleanup(s.shutdown)
            servers.append(s)
        keep = {k: os.environ.get(k) for k in ("http_proxy", "no_proxy")}
        os.environ.update(http_proxy=f"http://127.0.0.1:{servers[1].server_port}", no_proxy="")
        try:
            self.assertEqual(doctor.Box().get(f"http://127.0.0.1:{servers[0].server_port}/health"), 200)
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None) if v is None else os.environ.update({k: v})
        self.assertEqual(asked, [("server", "/health")])
        self.assertEqual(doctor.Box().get(f"http://127.0.0.1:{servers[0].server_port}/away"), 302)
        self.assertEqual(len(asked), 2)  # nor follows one elsewhere


class AsTheDocsSay(unittest.TestCase):
    def text(self, path):
        with open(os.path.join(REPO, path), encoding="utf-8") as f:
            return f.read()

    def test_what_it_pins_is_what_the_guides_and_scripts_pin(self):
        start, m5, lib, prep, conf = (self.text(p) for p in ("docs/GETTING-STARTED.md", "docs/BOSGAME-M5.md", "tools/dsv41-hip/serve_lib.sh", "tools/dsv41-hip/prepare_inputs.sh", "tools/dsv41-hip/serve.conf"))
        for word, where in ((doctor.KERNEL, m5), (doctor.KERNEL_KEY, m5), (doctor.ROCM, m5), (str(doctor.GTT), m5), (doctor.ZIG, start), (doctor.TRITON, start), (doctor.LOCK, start),
                            ("debs.ryai.dev", m5), ("MASTER_IP=10.40.0.1", conf), ("TP_STRIPE=10.40.1.1", conf), ("PEAK_GIB=101", conf), ("SERVE_PORT=18080", conf),
                            ("inputs-tp2-c178a012", conf), ("ds1017-1bae0e10", conf), ("libertai-2.9bpw", start), ("~/tfp/run/conf/my.conf", start), ("~/tfp/tri/kit/gfx1151", start)):
            self.assertIn(word, where, word)
        self.assertEqual(set(re.search(r"pgrep -x -d, '([^']+)'", lib).group(1).split("|")), set(doctor.ENGINES + doctor.OTHERS))
        for x in ("PREPARING.json", "FAILED.json"):
            self.assertIn(x, self.text("tools/dsv41-hip/prepare_inputs.py"))
        self.assertEqual({f"{k}-{v}" for k, v in re.findall(r'"(\w+)": "([\d.]+(?:\+cpu)?)"', re.search(r"want = \{(.*?)\}", prep).group(1))}, set(doctor.VENV))
        self.assertIn('r.get("status") != "pass"', lib)
        self.assertIn("aot.json", self.text("tools/dsv41-hip/tri/aot_build.py"))
        self.assertIn("build_log.json", self.text("tools/dsv41-hip/tri/aot_build.py"))
        inputs = self.text("tools/dsv41-hip/prepare_inputs.py")
        self.assertIn('output / f"rank{rank}" / "rank-cache"', inputs)
        self.assertIn('f"rank{rank}of2-*.bin"', inputs)
        self.assertIn("RANK_CACHE='$BUNDLE/rank$RANK/rank-cache'", conf)
        self.assertIn("tfp/arch/receipts/", self.text("tools/dsv41-hip/run_gpu_tests.sh"))
        self.assertIn('"all_passed"', self.text("tools/dsv41-hip/run_gpu_tests.sh"))
        self.assertIn("sudo apt install git rsync wget curl python3-venv", start)
        pins = [sum(x["bytes"] for x in json.loads(self.text(f"tools/dsv41-hip/assets/{n}"))["files"] if re.search(k, x["path"])) for n, k in
                (("checkpoint_files.json", "."), ("original_files.json", "0004[78]-of"))]
        need = [(sum(pins) + r * doctor.CACHE) * doctor.ROOM / G for r in (2, 1)]
        said = [int(x) for x in re.search(r"about (\d+) GB on node1 and (\d+) GB on node0", start).groups()]
        self.assertTrue(need[0] < said[0] < need[0] + 20 and need[1] < said[1] < need[1] + 20, (need, said))
        for x in ("tf-dsv41-rank-cache", "tf-dsv41-check"):
            self.assertIn(x, prep)

    def test_every_section_it_names_is_there(self):
        with open(doctor.__file__) as f:
            src = f.read()
        named = set(re.findall(r"docs/[A-Z0-9-]+\.md", src))
        for path in named:
            self.assertTrue(os.path.exists(os.path.join(REPO, path)), path)
        heads = {p: re.findall(r"(?m)^#+ (.+)$", self.text(p)) for p in named}
        start = "docs/GETTING-STARTED.md"
        for step in re.findall(r'\{START\}, step (\d+a?)', src):
            self.assertTrue([h for h in heads[start] if h.startswith(step + ".")], step)
        for title in ("The Triton kit", "Build and GPU tests", "Model files", "Inputs", "Deploy", "Serve"):
            self.assertTrue([h for h in heads[start] if h.startswith(title)], title)
        for path, kind, n in re.findall(r"(docs/(?:BOSGAME-M5|FRAMEWORK-DESKTOP|FEDORA-PODMAN)\.md), (step|section)s? (\d+)", src):
            self.assertTrue([h for h in heads[path] if h.startswith(n + ".")], (path, n))
        for words in ("Operating systems", "USB4 ports"):
            self.assertIn(words, heads["docs/HARDWARE.md"])

    def test_ai_setup_names_files_that_are_there(self):
        page = self.text("docs/AI-SETUP.md")
        for link in re.findall(r"\]\(([^)#:]+\.md)", page):
            self.assertTrue(os.path.exists(os.path.join(REPO, "docs", link)), link)
        for status in ("todo", "wait", "bad", "note", "guide"):
            self.assertIn(f"`{status}`", page)


if __name__ == "__main__":
    unittest.main()

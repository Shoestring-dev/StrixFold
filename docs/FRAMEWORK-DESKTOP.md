# Two Framework Desktops from scratch (Ubuntu 26.10)

A complete replication guide for a pair of Framework Desktops (AMD Ryzen AI Max 300 Series, Ryzen AI Max+ 395,
Radeon 8060S / gfx1151, 128 GB): from a fresh Ubuntu 26.10 install to a serving TP2 DeepSeek-V4.1-Flash port with an
OpenAI-compatible API and up to a 1M-token window. Every step below was run on one pair in November 2026; the
verified results are at the end and in [COMMUNITY-RESULTS.md](COMMUNITY-RESULTS.md).

The shared procedure lives in [GETTING-STARTED.md](GETTING-STARTED.md); this page gives the Framework- and
Ubuntu-specific parts in full, and the order to run everything. The stack here is Ubuntu 26.10 with the stock
kernel 7.3.0-10-generic and ROCm 10.1.0 (HIP 7.16) — newer than the Debian-13 rex stack and the 10.0.0-4
(HIP 7.15) the published numbers use, as [FEDORA-PODMAN.md](FEDORA-PODMAN.md) reports for its image. The bench
numbers below are on par with the published ones.

Two deviations from the rex path matter most, and both are covered here: **BIOS 3.06 + `amd_iommu=pt` before the
USB4 link will come up at all** (§1–2), and **an intermittent weight-load wedge on this kernel** that needs a
staggered start with a cooldown (§9).

## 0. What you need

- Two Framework Desktops, Ryzen AI Max+ 395, 128 GB each, and two USB4 cables rated 40 Gb/s or more.
- Ubuntu 26.10 on both (Server or Desktop; the IP-configuration step differs, §4).
- Free disk: about 660 GB on node1 and 550 GB on node0 ([GETTING-STARTED §1](GETTING-STARTED.md#1-what-you-need)).
- Internet on both; about 400 GB of model files download on node1 and copy to node0.
- Roles, picked before you start: **node0** is rank 0 and serves the API; **node1** is rank 1 and is the build
  box where every script runs.
- Time: about an hour per box for setup, plus the downloads; preparing the inputs reads about 1 TB and writes
  212 GB; checking them takes about 15 minutes per box.

## 1. BIOS 3.06 on both boxes (the critical one)

On BIOS 3.05, `thunderbolt_net` loaded but created no interface, at any port, on either box: the USB4 routers'
switches exposed no external ports, the UCSI typec controller reported an empty state, and plugging/unplugging a
cable produced no kernel event at all. Framework BIOS 3.06 (fwupd/LVFS: 0.0.3.5 → 0.0.3.6) carries the UCSI fixes
("delay in sending the 3A profile to prevent incomplete UCSI PPM command resets during S0 boot transitions",
"validation logic to reject invalid USB PD index values"). After the update, `thunderbolt0`/`thunderbolt1`
appeared on both boxes within seconds of the cable being attached, without a manual `modprobe`.

```bash
sudo fwupdmgr refresh --force
sudo fwupdmgr update                     # takes the LVFS firmware updates, including the BIOS
cat /sys/class/dmi/id/bios_version       # 3.06 (fwupd reports 0.0.3.6)
```

The BIOS alone is not enough: the kernel option in §2 must also be set. Note this covers Framework-to-Framework
links. The community report in [GETTING-STARTED §3](GETTING-STARTED.md#3-link-the-two-boxes) (Framework +
Minisforum MS-S1 Max) is about the MS-S1 side: that Framework was already on BIOS 3.06, and its ports linked to
the MS-S1 Max's rear Intel USB4 v2 ports but not to the MS-S1's AMD ports.

## 2. Kernel arguments: GPU memory and IOMMU

On both boxes, set the shipped GPU-memory line from [BOSGAME-M5.md step 5](BOSGAME-M5.md#5-give-the-gpu-124-gib)
plus the IOMMU option, in `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.gttsize=126976 ttm.pages_limit=32505856 amd_iommu=pt"
```

```bash
sudo update-grub && sudo reboot
```

- `amd_iommu=pt`, **not** `amd_iommu=off`: the kernel refuses XDomain (host-to-host) links when the IOMMU is
  disabled, but `pt` keeps the passthrough DMA mapping ROCm wants while still creating the IOMMU groups XDomain
  needs. With `off`, no cable will ever link.
- `ttm.pages_limit=32505856` (124 GiB) is the shipped value; it leaves the OS about 2 GiB more than the 126 GiB
  variant this pair ran earlier with no observed difference.
- This pair's boxes also carried `kho=off`, `amdgpu.noretry=0` and `amdgpu.lockup_timeout=...` from an earlier
  setup; none of them is needed here, so they are dropped.

Verify after the reboot, on both boxes:

```bash
cat /proc/cmdline                                          # the line above
cat /sys/class/drm/card*/device/mem_info_gtt_total         # 133143986176 (124 GiB)
```

Mask sleep so the long input preparation is never interrupted:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

## 3. Cable the two boxes

**Only the rear USB-C ports (at the I/O shield) are USB4.** The front USB-C ports are USB 3.2 only; a cable there
produces exactly the "no event anywhere" symptom of §1. Use two cables, each box's port to the matching port on
the other box. Each linked cable becomes a network interface from the `thunderbolt_net` driver:

```bash
ip -br link | grep thunderbolt          # thunderbolt0 and thunderbolt1, UP
sudo dmesg | grep -i thunderbolt        # if none: did the boxes see each other at all?
cat /sys/bus/thunderbolt/devices/*/rx_speed
```

On BIOS 3.06 + `amd_iommu=pt` the interfaces appear by themselves; `sudo modprobe thunderbolt_net` is the manual
fallback. If a cable still makes no interface, try another pair of rear ports and another cable.

## 4. Addresses, SSH, groups

The link layout is [GETTING-STARTED §3](GETTING-STARTED.md#3-link-the-two-boxes)'s, the addresses `serve.conf`
expects:

| | node0 | node1 |
|---|---|---|
| `thunderbolt0` (control and per-token data) | 10.40.0.1/24 | 10.40.0.2/24 |
| `thunderbolt1` (stripe and bulk copies) | 10.40.1.1/24 | 10.40.1.2/24 |
| MTU | 1500 | 1500 |

**Desktop install (NetworkManager):** the nmcli lines from §3 work as written (one profile per cable, `.2` on
node1).

**Server install (netplan, no nmcli):** write `/etc/netplan/10-usb4.yaml` on node0 (node1: `.2` in place of `.1`
in both addresses). Netplan 1.2.2 rejects `ipv6-method: disabled`; use `dhcp6: false` and `accept-ra: false`
instead:

```yaml
network:
  version: 2
  ethernets:
    thunderbolt0:
      dhcp4: false
      dhcp6: false
      accept-ra: false
      addresses: [10.40.0.1/24]
      mtu: 1500
    thunderbolt1:
      dhcp4: false
      dhcp6: false
      accept-ra: false
      addresses: [10.40.1.1/24]
      mtu: 1500
```

```bash
sudo chmod 600 /etc/netplan/10-usb4.yaml && sudo netplan apply
```

Check from node1:

```bash
ping -c 3 10.40.0.1 && ping -c 3 10.40.1.1     # ~1 ms on both cables
ip link show thunderbolt0 | grep -o 'mtu [0-9]*'   # mtu 1500
```

If only one ping answers, the cables are crossed: each box's `thunderbolt0` must be on the same cable. Swap the
two cables at one box, or swap the two addresses in that box's configuration.

node1 reaches node0 by the SSH name `node0`, no password prompt (the deploy, `first_light.sh` and the checks all
run from node1). On node1:

```bash
[ -f ~/.ssh/strixfold ] || ssh-keygen -t ed25519 -N '' -C strixfold -f ~/.ssh/strixfold   # for node0 only, no passphrase
ssh-copy-id -i ~/.ssh/strixfold.pub -o PreferredAuthentications=keyboard-interactive,password 10.40.0.1   # USER@10.40.0.1 if the names differ
printf '\nHost node0\n    HostName 10.40.0.1\n    IdentityFile ~/.ssh/strixfold\n    IdentitiesOnly yes\n' >> ~/.ssh/config && chmod 600 ~/.ssh/config
ssh node0 true
```

The key is for node0 only: your other keys keep their passphrases. Add a `User` line under `Host node0` if the user names differ. Large copies to node0 then go over USB4.

On both boxes, add your user to the GPU groups and log out/in (or open a new SSH session):

```bash
sudo usermod -aG render,video "$USER"
```

## 5. ROCm 10.1 from AMD's Ubuntu repo, gfx1151 only

The rex packages (debs.ryai.dev) are Debian-13 builds; do not add that repo on Ubuntu. AMD's own stable repo has
an ubuntu2604 suite that works on 26.10. Its key is checked against the one that address gave us on 25 November 2026
("AMD MLSE DevOps", made in January 2026; we found no second source for it). On both boxes:

```bash
sudo install -d /etc/apt/keyrings
curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg -o /tmp/amdrocm.asc
K=$(gpg --show-keys --with-colons /tmp/amdrocm.asc | awk -F: '$1=="pub"{p=1} $1=="fpr"&&p{print $10;p=0}')
[ "$K" = D0F004A0025A1145C7807FCD0701EAC4D5E02107 ] && sudo gpg --yes --dearmor -o /etc/apt/keyrings/amdrocm.gpg /tmp/amdrocm.asc \
  && echo "the key is in" || echo "not the key we saw"
```

It must print `the key is in`. If it doesn't, stop. Otherwise:

```bash
sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources >/dev/null <<'EOF'
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
EOF
sudo apt update
sudo apt install amdrocm10.1-gfx1151 amdrocm-core-sdk10.1-gfx1151
```

Do **not** install the plain `amdrocm10.1` meta: it pulls every architecture's libraries (several GB of
gfx908/gfx942/gfx1201/gfx1250) for nothing. This gives ROCm 10.1.0 (HIP 7.16), which matches the pinned Triton
wheel (rocm10.1.0a).

**Register `/opt/rocm/lib` with ldconfig (the therock equivalent).** The ubuntu2604 repo has no `therock-gfx1151`
package, and without the registration every StrixFold program fails with `error: DriverUnavailable` at 0.0 s
(rocminfo still works — it is a loader-cache issue, not a kernel one):

```bash
echo /opt/rocm/lib | sudo tee /etc/ld.so.conf.d/amdrocm.conf
sudo ldconfig
```

Verify (after the re-login from §4):

```bash
rocminfo | grep -m1 gfx1151
hipconfig --version                          # 7.16...
ldconfig -p | grep libamdhip64.so.7          # => /opt/rocm/lib/libamdhip64.so.7
```

## 6. Python 3.13, zig, source, venvs

Ubuntu 26.10 ships Python 3.14/3.15; the pinned Triton wheel is cp313, and the inputs lock file wants 3.13. Two
ways to get it:

- `uv python install 3.13` (what [FEDORA-PODMAN.md](FEDORA-PODMAN.md)'s Containerfile does) — no PPA, no suite
  mixing. Use that interpreter for the venvs below.
- Or the deadsnakes PPA: it has no 26.10 (stonking) build, but its 26.04 (resolute) pocket installs and works:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo sed -i 's/^Suites: stonking/Suites: resolute/' /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-*.sources
sudo apt update && sudo apt install -y python3.13 python3.13-venv
```

Then follow [GETTING-STARTED §4](GETTING-STARTED.md#4-toolchain-and-source) with one change: where it says
`python3 -m venv`, use **`python3.13 -m venv`** (on Debian, `python3` is already 3.13; here it is not). In order,
on both boxes unless noted:

1. `sudo apt install git rsync python3-venv` (both).
2. Zig 0.17.0 at `~/opt/zig/zig` with the published sha256 checked (node1; the exact commands are in §4).
3. Clone to `~/tfp/src/TensorFold` (node1).
4. The inputs venv `~/venvs/tfp-inputs` from `requirements-inputs.lock.txt`, `--no-deps --require-hashes`,
   `--extra-index-url https://download.pytorch.org/whl/cpu` (both; node0's over ssh, as §4 shows).
5. The Triton venv `~/venvs/triton` with the pinned wheel
   `triton-3.8.0+git675c5987.rocm10.1.0a20260822-cp313-cp313-linux_x86_64.whl`, sha256 checked (node1).

## 7. Build, prepare, deploy — the ordered checklist

Everything from [GETTING-STARTED §5](GETTING-STARTED.md#5-build-prepare-deploy-serve), on node1 from
`~/tfp/src/TensorFold`, in this order. Each step's commands are in that section and
[BUILD_AND_RUN.md](../tools/dsv41-hip/BUILD_AND_RUN.md); the expected outcome on this pair is noted here.

| # | Step | Command (summary) | Verified outcome on this pair |
|---|---|---|---|
| 1 | Triton AOT kit | `tri/aot_build.py … ~/tfp/tri/kit/gfx1151` (§5) | 280 variants built, 0 failed |
| 2 | Build | `tools/dsv41-hip/zig_build.sh install` | `Build Summary: N/N steps succeeded` |
| 3 | Test build | `zig_build.sh test … -Dhip-triton-kit=$HOME/tfp/tri/kit/gfx1151` | green |
| 4 | GPU tests | `tools/dsv41-hip/run_gpu_tests.sh` (no EXTRA_KIT) | smoke, ABI, glue all exit 0 |
| 5 | Model files | the two `hf download` commands (§5), pinned revisions | checkpoint 211 GB + Engram shards 203 GB |
| 6 | Copy models | `rsync -a … node0:models/tf-port/` | both boxes hold both downloads |
| 7 | Inputs | `prepare_inputs.sh … ~/models/tf-port/inputs-tp2-c178a012`; note `$H` | bundle + manifest |
| 8 | Copy bundle | `rsync -a --exclude rank1/ … node0:models/tf-port/` | |
| 9 | Verify rank 1 | `verify_inputs.sh … 1 … $H` | report `pass` |
| 10 | Deploy | `deploy_rank0.sh --native …` then `--host local …` (§5) | one commit's bytes on both boxes |
| 11 | Verify rank 0 | scp `tf-dsv41-check`, run `verify_inputs.sh … 0 … $H` on node0 | report `pass` |

Run `run_gpu_tests.sh` without EXTRA_KIT: the installed dsv41 kit loads, but the extra-kit check wants a
loader-format kit whose entries are all Triton variants, which the raw tri kit is not.

For a 1M-token window you can prepare the inputs with more rows from the start — see §10 before running step 7.

## 8. Serve

Make your configuration from the shipped `serve.conf`, set your bundle's hash, and put the same file on both
boxes ([GETTING-STARTED §5 Serve](GETTING-STARTED.md#serve)):

```bash
mkdir -p ~/tfp/run/conf
install -m 644 ~/tfp/deploy/current/tools/dsv41-hip/serve.conf ~/tfp/run/conf/my.conf
sed -i "s/^BUNDLE_SHA256=[0-9a-f]*/BUNDLE_SHA256=$H/" ~/tfp/run/conf/my.conf
ssh node0 mkdir -p tfp/run/conf && scp ~/tfp/run/conf/my.conf node0:tfp/run/conf/my.conf

cd ~/tfp/deploy/current/tools/dsv41-hip
TF_SERVE_CONF=$HOME/tfp/run/conf/my.conf ./first_light.sh up
```

`up` checks both ranks' identities, takes the GPU locks, starts rank 1 then rank 0, waits for `/health` and a
probe answer ("Paris"), and with the shipped `BENCH=1` times three requests. On this pair at the shipped
`CONTEXT=8192`:

| case | prompt tokens | TTFT | decode tok/s | prefill tok/s |
|---|---:|---:|---:|---:|
| code | 31 | 0.21 s | 70.8 | — |
| prose | 34 | 0.23 s | 41.9 | — |
| prefill | 1,818 | 1.44 s | — | 1,260 |

(`first_light.sh up`, DRAFTS=1, both cables, tcp proxy transport, ROCm 10.1.0 / kernel 7.3.0-10-generic.)

The server listens on node0 at `127.0.0.1:18080` and speaks the OpenAI chat API; the model name is `SERVE_NAME`
in your configuration (`dsv41` as shipped). First request and port forwarding: see
[GETTING-STARTED §5 Serve](GETTING-STARTED.md#serve). `first_light.sh status` and `down` work as documented.

**On this kernel, `up` may fail with a load wedge — read §9 before retrying.**

## 9. The load wedge on this kernel, and the staggered start

Symptom: `first_light.sh up` fails; rank 0's log ends with

```
info: rank cache: 95.1 GiB of expert entries read whole in 15.4 s
tensorfold: the native HIP engine cannot load … (PeerNeverLinked)
```

and rank 1's log is **empty**. Rank 1 is alive but its weight load has wedged about 3 GiB in (layer-0 experts):
its main thread spins at 100% CPU in `libhsa-runtime64`'s `BusyWaitSignal::WaitRelaxed`, the GPU is idle
(`gpu_busy_percent` 0–1), and the kernel log shows `svm_range_restore_work` /
`amdgpu_amdkfd_restore_userptr_worker` hogging CPU with doubling counters. A copy's completion signal never
arrives; the kernel's SVM-restore work is livelocked. This was observed on this pair's stock Ubuntu kernel
7.3.0-10-generic, not on the rex stack the repo is tested on. The initial trigger is unknown; controlled tests
ruled out concurrent boots, the spawn path, the GPU locks and monitoring as causes.

What propagates it is established: **after a wedged rank is SIGKILLed, the kernel's SVM-restore work stays
livelocked for a while.** Retries within seconds wedge again (9/9 on this pair); a load started 150 s or more
after the kill is clean. A healthy load reads all 95 GiB in about 15 s, so a wedge is unmistakable: flat
`read_bytes` in `/proc/<pid>/io` for 30 s.

The workaround is a staggered start with stall detection and a cooldown: rank 1 loads first, and rank 0 boots
only after rank 1's weights are in (rank 1 retries its peer connect for about 2 minutes, so start rank 0
promptly once rank 1 is loaded — a rank 0 that starts later than that gets `ConnectFailed`). Save this as
`~/tfp/staggered-start.sh` (adjust the health URL if you change `SERVE_PORT`):

```bash
#!/bin/bash
# Staggered TP2 start: rank 1 loads first; a wedged load is killed, cooled down 150 s, retried.
set -euo pipefail
cd "$HOME/tfp/deploy/current/tools/dsv41-hip"
export TF_SERVE_CONF=$HOME/tfp/run/conf/my.conf
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ); export RUN_ID
RUN=$HOME/tfp/run/serve/$RUN_ID
R0CMD="RUN_ID=$RUN_ID TF_SERVE_CONF=$TF_SERVE_CONF $PWD/serve-rank0.sh"
ssh node0 "$R0CMD lock"
for attempt in 1 2 3 4; do
  ./serve-rank1.sh start                      # rank 1 takes its own GPU lock (flock)
  P=$(cat "$RUN/rank1.pid"); last=0; flat=0; ok=0
  for i in $(seq 1 120); do
    if grep -q 'read whole' "$RUN/rank1.log" 2>/dev/null; then ok=1; break; fi
    kill -0 "$P" 2>/dev/null || break
    now=$(awk '/read_bytes/{print $2}' /proc/$P/io 2>/dev/null || echo 0)
    if [ "$now" = "$last" ]; then flat=$((flat+1)); else flat=0; fi
    last=$now
    if [ "$flat" -ge 30 ]; then echo "load wedged at $((now/1048576)) MiB — killing"; kill -9 "$P"; break; fi
    sleep 1
  done
  [ "$ok" = 1 ] && break
  ./serve-rank1.sh stop >/dev/null 2>&1 || true
  [ "$attempt" -lt 4 ] || { echo "load failed 4 attempts"; exit 1; }
  echo "cooldown 150 s — the kernel's SVM-restore work needs to settle"
  sleep 150
done
ssh node0 "$R0CMD start --locked"
for i in $(seq 1 300); do
  curl -fsS -m 3 http://10.40.0.1:18080/health >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS http://10.40.0.1:18080/health; echo
```

```bash
bash ~/tfp/staggered-start.sh
```

The run dirs, locks and `resolved.json` are the same ones `first_light.sh` writes, so `first_light.sh down` and
`status` (with the same `TF_SERVE_CONF`) manage a staggered run normally. If four cooldown attempts all wedge,
reboot node1 and try again; the wedge has always cleared well before that on this pair.

## 10. A 1M-token window

`CONTEXT` can be at most the RoPE tables' rows minus 49, and the shipped inputs carry 65,536 rows (max 65,487).
The repository keeps verified reference hashes up to 1,114,112 rows — the creator kit's full tables.

**The documented path** (recommended from scratch): [GETTING-STARTED §5 Longer context](GETTING-STARTED.md#longer-context)
— a second `prepare_inputs.sh` with `R=1048576` into a new bundle, deploy with that bundle's kit, verify with
`$((R - 49))`, and set `BUNDLE`, `BUNDLE_SHA256` and `CONTEXT` in your configuration. A full 1M window read cold
takes about 16 minutes ([RESULTS.md](RESULTS.md)).

**The shortcut this pair runs** (avoids the second full prepare): extend the deployed ds-kit's RoPE tables in
place to the creator's full 1,114,112 rows. The tables are the same verified creator prefixes the documented path
uses; the rank caches and the bundle are unchanged. On node1, with the inputs venv (CPU torch):

```python
# extend_rope.py ROWS OUT — generate the FULL creator tables (hash-verified), keep the first ROWS rows
import hashlib, json, sys, tempfile
from pathlib import Path
TOOLS = Path.home() / "tfp/deploy/current/tools/dsv41-hip"
sys.path.insert(0, str(TOOLS))
from prepare_inputs import generate_rope
rows, out = int(sys.argv[1]), Path(sys.argv[2])
creator = json.loads((TOOLS / "assets/creator_kit_v0.6.1_rope.json").read_text())
config = json.loads((Path.home() / "models/tf-port/inputs-tp2-c178a012/model/config.json").read_text())
kit = Path.home() / "tfp/deploy/current/ds-kit"
old = json.loads((kit / "rope.json").read_text())
out.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
    generate_rope(config, Path(tmp), creator["rows"], creator)   # raises unless all four tables hash-match
    meta = dict(old); meta["rows"] = rows
    for name in ("rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32"):
        data = (Path(tmp) / name).read_bytes()[: rows * 128]
        (out / name).write_bytes(data)
        meta[name] = hashlib.sha256(data).hexdigest()
    (out / "rope.json").write_text(json.dumps(meta, indent=1) + "\n")
```

```bash
~/venvs/tfp-inputs/bin/python extend_rope.py 1114112 ~/tfp/run/ds-kit-1114112
# back up, unlock the read-only deploy, install, re-pin — on BOTH boxes:
K=~/tfp/deploy/current/ds-kit
mkdir -p ~/tfp/run/ds-kit-65536-backup && cp -p $K/rope*.f32 $K/rope.json ~/tfp/run/ds-kit-65536-backup/
chmod u+w $K $K/rope*.f32 $K/rope.json
cp -p ~/tfp/run/ds-kit-1114112/rope*.f32 ~/tfp/run/ds-kit-1114112/rope.json $K/
chmod 400 $K/rope*.f32 $K/rope.json && chmod 555 $K
scp ~/tfp/run/ds-kit-1114112/rope*.f32 ~/tfp/run/ds-kit-1114112/rope.json node0:tfp/deploy/current/ds-kit/
# (on node0: the same backup + chmod dance around the scp)
```

The ds-kit is not covered by the deployment's kit-manifest check (that covers the HIP kit), so the swap breaks
no identity — but both boxes must hold identical files. Then in `my.conf` on both boxes:

```
CONTEXT=1048576      # at most 1,114,112 - 49 = 1,114,063 with the full tables
PEAK_GIB=105         # raised from 101: the pool and split buffers grow with CONTEXT
```

Verified on this pair with `SERVE_NAME=local-model`, `SERVE_HOST=0.0.0.0`: `/health` reports
`context_length: 1048558`, and the bench cases are unchanged by the window size (short prompts use the same
pool mechanics):

| case | TTFT | decode tok/s | prefill tok/s |
|---|---:|---:|---:|
| code | 0.21 s | 71.7 | — |
| prose | 0.23 s | 41.7 | — |
| prefill (1,818 tokens) | 1.45 s | — | 1,258 |

(`serve-rank0.sh bench` at CONTEXT=1048576.) Writing speed against context depth
(`ctx_sweep.py`, drafts on, 400 tokens written per request):

| prompt tokens | write tok/s | prefill tok/s | TTFT |
|---:|---:|---:|---:|
| 47 | 68.9 | — | 0.29 s |
| 2,012 | 71.7 | 1,123 | 1.79 s |
| 8,071 | 72.5 | 1,237 | 6.53 s |
| 31,624 | 68.2 | 1,253 | 25.2 s |

Writing speed is flat to 32k depth; with drafts off it is about 31 tok/s at every depth (drafting roughly doubles
it, as documented).

Vision is not available with this deployment: `MODALITIES=text_image` needs a vision-tower sidecar
(`VISION_DIR`) and `DS_KIT/vision/gate_bias_vl.safetensors`, neither of which the native-hip deploy ships. The
pair serves text only.

## 11. Day-2 operations

- **Stop / status:** `TF_SERVE_CONF=$HOME/tfp/run/conf/my.conf ./first_light.sh down` (SIGTERM rank 0; rank 1
  hears it and leaves; both locks released). `status` shows both ranks and the locks. Logs:
  `~/tfp/run/serve/latest/rank1.log` locally, the same path on node0 for rank 0.
- **Configuration changes** (port, model alias, context): edit `my.conf`, copy it to node0 (`scp …
  node0:tfp/run/conf/my.conf` — both ranks must read the same file), then stop and start. `SERVE_PORT` is the
  API port; `SERVE_NAME` is the model id `/v1/models` and chat requests use; `SERVE_HOST=0.0.0.0` serves the LAN
  (mind your firewall; the shipped `127.0.0.1` needs an SSH forward).
- **Memory gate:** a rank starts only with `MemAvailable >= PEAK_GIB + 8` on its box and no other engine
  running. Stop other GPU work first (another inference server, a container with the GPU, a benchmark). At
  CONTEXT=1048576 this pair needs about 113 GiB free per box; the weights alone are about 99 GiB a rank.
- **After a wedge:** wait at least 150 s from the kill before starting again (§9), or use the staggered script.
- **Bench:** `ssh node0 'cd ~/tfp/deploy/current/tools/dsv41-hip && RUN_ID=<run> TF_SERVE_CONF=… ./serve-rank0.sh bench'`
  against a live run; `ctx_sweep.py` for speed against context depth
  ([COMMUNITY-RESULTS.md](COMMUNITY-RESULTS.md#add-yours)).

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| No `thunderbolt*` interface, no plug events | BIOS 3.05 (§1), front ports (§3), or `amd_iommu=off` (§2). All three must be right. |
| One cable pings, the other doesn't | Crossed cables (§4). |
| `error: DriverUnavailable` at 0.0 s, rocminfo fine | `/opt/rocm/lib` not registered with ldconfig (§5). |
| `PeerNeverLinked`, rank 1's log empty | The load wedge (§9). Cooldown 150 s, staggered start. |
| Rank 1 `error: ConnectFailed` after a clean load | Rank 0 started more than ~2 minutes after rank 1 loaded; its connect retries ran out. Start rank 0 promptly. |
| `MemAvailable … a rank needs` / `is not idle` | Another engine or too little free memory (§11). |
| `inputs_manifest.json is X, the config pins Y` | `BUNDLE_SHA256` isn't your bundle's `$H` (§7–8). |
| netplan: `unknown key ipv6-method` | Use `dhcp6: false` + `accept-ra: false` (§4). |
| deadsnakes 404 on 26.10 | Point the PPA's `Suites:` at `resolute` (§6), or use `uv python install 3.13`. |
| Triton wheel won't install | It is cp313: the venv must be Python 3.13 (§6). |
| Rank 0's log empty after `up` returns | logind ends session processes on SSH disconnect; `SPAWN_UNIT=1` in serve.conf ([GETTING-STARTED §6](GETTING-STARTED.md#6-if-something-goes-wrong)). |

## 13. Verified on this pair (November 2026)

| Item | Result |
|---|---|
| BIOS / kernel | 3.06 both boxes; Ubuntu 26.10, 7.3.0-10-generic |
| Link | both cables up, MTU 1500, ping ~1 ms |
| GPU stack | ROCm 10.1.0, HIP 7.16, gfx1151; GTT 124 GiB |
| Triton AOT kit | 280 variants, 0 failed (pinned cp313 wheel) |
| Build + GPU tests | zig build green; smoke, ABI, glue exit 0 |
| Inputs | prepared, verified `pass` on both boxes |
| Serving, CONTEXT=8192 | `first_light.sh up` + bench: code 70.8, prose 41.9 tok/s; 1,260 tok/s at 1,818 tokens |
| Serving, CONTEXT=1048576 | staggered start; `/health` context_length 1,048,558; bench: code 71.7, prose 41.7 tok/s; 1,258 tok/s at 1,818 tokens |
| Context sweep | `ctx_sweep.py` 0–32k: writing 68–72 tok/s flat, prefill ~1,250 tok/s (§10) |
| Load wedge | characterized (§9); cooldown retry verified |

Numbers in [COMMUNITY-RESULTS.md](COMMUNITY-RESULTS.md); the pair's entry in [HARDWARE.md](HARDWARE.md).
Questions: the [StrixFold Discord](https://discord.gg/D4FgAJnKd9) or a GitHub issue.
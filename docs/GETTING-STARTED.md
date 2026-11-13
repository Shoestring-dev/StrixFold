# Getting started: two new boxes to a first answer

This page gives the order of the steps, from two new boxes to a first answer and a benchmark, on either of the two box
types we use. The long parts already have their own guides. This page links to the right section of each and lists the
commands to run in order.

The two boxes play different parts. **node0** is rank 0 and serves the API. **node1** is rank 1, and it is also where
you build and run the scripts. Pick which box is which before you start.

Our published numbers come from a pair of AMD Ryzen AI Halo boxes (see [Tested on](../README.md#tested-on)). We've
also run pairs of Bosgame M5s: their speed, cooling and USB4 links are in [BOSGAME-M5.md](BOSGAME-M5.md).

## 1. What you need

- **Two boxes of the same type:** two AMD Ryzen AI Halo developer boxes, or two Bosgame M5s. Both types have the Ryzen
  AI Max+ 395 (Radeon 8060S, gfx1151) with 128 GB of memory.
- **Two USB4 cables** between the boxes.
- **For a Bosgame M5:** a USB stick of 1 GB or more, a monitor and a USB keyboard
  ([BOSGAME-M5.md](BOSGAME-M5.md)).
- **Internet on both boxes,** for packages and for about 400 GB of model files (on node1; it copies them to node0).
- **Free disk space:** about 620 GB on node1 and 510 GB on node0. Each box holds the checkpoint (197 GB) and the
  Engram shards (203 GB). node1 also holds both ranks' caches (106 GB each), and node0 holds its own
  ([BUILD_AND_RUN.md §1](../tools/dsv41-hip/BUILD_AND_RUN.md#1-what-you-need)).
- **Time:** about 30 minutes per box for the Bosgame setup, plus the downloads. Preparing the inputs reads about 1 TB
  and writes 212 GB, and checking them takes about 15 minutes per box.

## 2a. AMD Ryzen AI Halo

The Ryzen AI Halo ships with AMD's OS "AMD Ryzen AI Developer Platform 1 (rex)", which is based on Debian 13. Ours came
with the stack StrixFold needs already installed:
- kernel 7.2.7+rex-amd64, from AMD's apt repositories;
- ROCm 10.0.0-4 and therock-gfx1151;
- Secure Boot on;
- the kernel options that give the GPU 124 GiB.

Check each of these on both boxes:

```bash
grep PRETTY_NAME /etc/os-release       # AMD Ryzen AI Developer Platform 1 (rex)
uname -r                               # 7.2.7+rex-amd64
mokutil --sb-state                     # SecureBoot enabled
cat /sys/class/dmi/id/bios_version     # ours: 04.01
dpkg -l linux-image-7.2.7+rex-amd64 amdrocm-core10.0-gfx1151 amdrocm-core-sdk10.0-gfx1151 therock-gfx1151
cat /sys/class/drm/card*/device/mem_info_gtt_total   # 133143986176 (124 GiB)
```

- **Packages:** each package's line in the `dpkg -l` output should start with `ii` (installed) or `hi` (installed and
  held). The ROCm packages should be at 10.0.0-4 and therock-gfx1151 at 10.0+rex1bvdebian13.1. Install any missing package with the command in
  [BOSGAME-M5.md step 7](BOSGAME-M5.md#7-rocm-100), which uses the same repository.
- **GPU memory:** if the GTT total is lower than 133143986176, the kernel options are missing. Add them as in
  [BOSGAME-M5.md step 5](BOSGAME-M5.md#5-give-the-gpu-124-gib), then restart.
- **Sleep:** if the box is set to sleep when idle, mask the sleep targets as in that step too. The input preparation
  runs for a long time.

You don't need the Secure Boot step: the Ryzen AI Halo's firmware already trusts the kernel's signer.

Hold the stack at these versions. This also stops their security updates; see
[BOSGAME-M5.md step 4](BOSGAME-M5.md#4-kernel-727-and-firmware) for how to take one.

```bash
sudo apt-mark hold linux-image-7.2.7+rex-amd64 firmware-amd-graphics amd64-microcode
sudo apt-mark hold $(dpkg-query -W -f '${Package}\n' 'amdrocm*') therock-gfx1151
sudo usermod -aG render,video "$USER"
```

Log out and back in (or open a new SSH session) so the new groups apply, then check the GPU:

```bash
/opt/rocm/core-10.0/bin/rocminfo | grep -m1 gfx1151
/opt/rocm/core-10.0/bin/hipconfig --version   # 7.15...
/sbin/ldconfig -p | grep libamdhip64.so.7     # => /opt/rocm/lib/libamdhip64.so.7
```

Then go to step 3.

## 2b. Bosgame M5

Follow [BOSGAME-M5.md](BOSGAME-M5.md) to its end on both boxes. It sets up the BIOS and installs Debian 13. It
installs the 7.2.7 kernel with Secure Boot on, gives the GPU 124 GiB, and installs ROCm 10.0. That is the same stack
as the Ryzen AI Halo. Then go to step 3.

## 3. Link the two boxes

Why the link is set up this way (two cables with separate jobs, MTU 1500) is in
[TUNING.md](TUNING.md#the-link). The addresses below are the ones `serve.conf` expects.

| | node0 | node1 |
|---|---|---|
| `thunderbolt0` (control and per-token data) | 10.40.0.1/24 | 10.40.0.2/24 |
| `thunderbolt1` (stripe and bulk copies) | 10.40.1.1/24 | 10.40.1.2/24 |
| MTU | 1500 on both | 1500 on both |

Connect each USB4 port to the matching port on the other box. Each cable shows up as a network interface from the
`thunderbolt_net` driver:

```bash
ip -br link | grep thunderbolt     # thunderbolt0 and thunderbolt1; if none, run: sudo modprobe thunderbolt_net
```

Give the interfaces fixed addresses that survive a restart, with IPv6 off on the link. A Bosgame M5 set up with
BOSGAME-M5.md uses ifupdown; write this file on node0, and on node1 the same file with `.2` in place of `.1` in both
addresses:

```bash
sudo tee /etc/network/interfaces.d/usb4 >/dev/null <<'EOF'
allow-hotplug thunderbolt0
iface thunderbolt0 inet static
    address 10.40.0.1/24
    mtu 1500
    post-up sysctl -q -w net.ipv6.conf.thunderbolt0.disable_ipv6=1

allow-hotplug thunderbolt1
iface thunderbolt1 inet static
    address 10.40.1.1/24
    mtu 1500
    post-up sysctl -q -w net.ipv6.conf.thunderbolt1.disable_ipv6=1
EOF
sudo ifup thunderbolt0 thunderbolt1
```

The Ryzen AI Halo's OS uses NetworkManager instead. Add one profile per cable; these are node0's, use `.2` on node1
(ours are set up exactly like this):

```bash
sudo nmcli con add type ethernet ifname thunderbolt0 con-name usb4-a ipv4.method manual ipv4.addresses 10.40.0.1/24 ethernet.mtu 1500 ipv6.method disabled
sudo nmcli con add type ethernet ifname thunderbolt1 con-name usb4-b ipv4.method manual ipv4.addresses 10.40.1.1/24 ethernet.mtu 1500 ipv6.method disabled
```

Check from node1:

```bash
ping -c 3 10.40.0.1 && ping -c 3 10.40.1.1
ip link show thunderbolt0 | grep -o 'mtu [0-9]*'   # mtu 1500
```

If only one ping answers, the cables are crossed: each box's `thunderbolt0` must be on the same cable. Swap the two
cables at one box, or swap the two addresses in that box's configuration.

node1 reaches node0 over SSH by the name `node0`, with no password prompt. The build deploy, `first_light.sh` and the
checks all run from node1. On node1:

```bash
ssh-keygen -t ed25519               # only if you have no key yet
ssh-copy-id 10.40.0.1
cat >> ~/.ssh/config <<'EOF'
Host node0
    HostName 10.40.0.1
EOF
ssh node0 true
```

Add a `User` line under `Host node0` if your user name on node0 is different. Large copies to node0 then go over the
USB4 link.

## 4. Toolchain and source

On both boxes:

```bash
sudo apt install git rsync python3-venv
```

On node1, install Zig 0.17.0 so that `~/opt/zig/zig` is the zig binary (`zig_build.sh` looks there; set `ZIG` to
use another path). The hash is the one [ziglang.org](https://ziglang.org/download/) publishes for this file:

```bash
cd /tmp && wget https://ziglang.org/download/0.17.0/zig-x86_64-linux-0.17.0.tar.xz
echo "1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026  zig-x86_64-linux-0.17.0.tar.xz" | sha256sum -c
mkdir -p ~/opt && tar -xf zig-x86_64-linux-0.17.0.tar.xz -C ~/opt && ln -sfn zig-x86_64-linux-0.17.0 ~/opt/zig
~/opt/zig/zig version                 # 0.17.0
cd ~
```

Clone this repository on node1:

```bash
mkdir -p ~/tfp/src && git clone https://github.com/Shoestring-dev/StrixFold.git ~/tfp/src/TensorFold
cd ~/tfp/src/TensorFold
```

Both boxes need the Python venv for the input tools. node1 uses it to prepare the inputs, and both use it to check
them. It is pinned by hash in `tools/dsv41-hip/requirements-inputs.lock.txt`; Debian 13's `python3` is 3.13, which the
lock file needs.

```bash
python3 -m venv ~/venvs/tfp-inputs
~/venvs/tfp-inputs/bin/pip install --no-deps --require-hashes -r tools/dsv41-hip/requirements-inputs.lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
scp tools/dsv41-hip/requirements-inputs.lock.txt node0:/tmp/
ssh node0 'python3 -m venv ~/venvs/tfp-inputs && ~/venvs/tfp-inputs/bin/pip install --no-deps --require-hashes \
    -r /tmp/requirements-inputs.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu'
```

On node1, make a second venv for the pinned AMD Triton build, which compiles the auxiliary kernels. The wheel is
named in [tri/README.md](../tools/dsv41-hip/tri/README.md) under "Toolchain":
`triton==3.8.0+git675c5987.rocm10.1.0a20260822` (cp313), from AMD's nightly index. Download it, check it and install
it:

```bash
mkdir -p ~/tfp/tri/wheels && cd ~/tfp/tri/wheels
W=triton-3.8.0+git675c5987.rocm10.1.0a20260822-cp313-cp313-linux_x86_64.whl
wget -O $W "https://rocm.nightlies.amd.com/whl-multi-arch/${W/+/%2B}"
echo "1bab2eeb8f4beccad389a497a45b3018fa6291ab93c577f04b1376cf12422499  $W" | sha256sum -c
python3 -m venv ~/venvs/triton && ~/venvs/triton/bin/pip install --no-deps ./$W
cd ~/tfp/src/TensorFold
```

## 5. Build, prepare, deploy, serve

Each part below follows [BUILD_AND_RUN.md](../tools/dsv41-hip/BUILD_AND_RUN.md). Run everything on node1, from
`~/tfp/src/TensorFold`, unless it says otherwise.

### The Triton kit (once)

Compile the recorded and derived Triton launches for gfx1151 into `~/tfp/tri/kit/gfx1151`. This is step 3 of
[tri/README.md](../tools/dsv41-hip/tri/README.md).

```bash
cd tools/dsv41-hip/tri
systemd-run --user --scope -p CPUQuota=800% -p MemoryHigh=12G -p MemoryMax=16G taskset -c 8-15,24-31 nice -n 10 \
    ~/venvs/triton/bin/python aot_build.py ../../../zig/src/families/dsv41/fixtures/launches.json ~/tfp/tri/kit/gfx1151 \
    --extra ../../../zig/src/families/dsv41/fixtures/derived_launches.json --jobs 8
cd ~/tfp/src/TensorFold
```

You can skip steps 1 and 2 of that README. They regenerate files that are already committed: `tri_kernels.py`,
`derived_launches.json`, and the recorded launches, which are byte-identical to the committed `launches.json`.

### Build and GPU tests

[BUILD_AND_RUN.md §2](../tools/dsv41-hip/BUILD_AND_RUN.md#2-build-node1-from-a-clean-committed-tree):

```bash
tools/dsv41-hip/zig_build.sh install
tools/dsv41-hip/zig_build.sh test test-dsv41-hip-host native-hip dsv41-hip-gpu-test -Dhip-triton-kit=$HOME/tfp/tri/kit/gfx1151
tools/dsv41-hip/run_gpu_tests.sh
```

A green build ends with `Build Summary: N/N steps succeeded`. On a Bosgame M5, one failing `clock` check in the glue
tests is expected ([BOSGAME-M5.md step 7](BOSGAME-M5.md#7-rocm-100)).

### Model files

The two downloads are pinned in `tools/dsv41-hip/source_lock.json`:
- the LibertAI EXL3 2.9 bpw checkpoint;
- shards 47 and 48 of DeepSeek-V4.1-Flash-1017, which hold the Engram tables.

The input venv includes `huggingface_hub`, whose `hf` command downloads them:

```bash
~/venvs/tfp-inputs/bin/hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
    --revision c178a01284fc5c0e859d8668cffaac81ca8f4042 --local-dir ~/models/tf-port/libertai-2.9bpw
~/venvs/tfp-inputs/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ~/models/tf-port/ds1017-1bae0e10
```

node0 needs both too, because it checks its own copy of the inputs against them. Copy them over the USB4 link:

```bash
ssh node0 mkdir -p models/tf-port
rsync -a ~/models/tf-port/libertai-2.9bpw ~/models/tf-port/ds1017-1bae0e10 node0:models/tf-port/
```

### Inputs

[BUILD_AND_RUN.md §3](../tools/dsv41-hip/BUILD_AND_RUN.md#3-inputs-cpu-only-once-per-checkpoint-revision): this builds
the token map, the Engram constants, the RoPE tables and both ranks' caches. It authenticates every checkpoint and
Engram file first, so a wrong download stops it there.

```bash
tools/dsv41-hip/prepare_inputs.sh ~/models/tf-port/libertai-2.9bpw ~/models/tf-port/ds1017-1bae0e10 ~/models/tf-port/inputs-tp2-c178a012
H=$(sha256sum ~/models/tf-port/inputs-tp2-c178a012/inputs_manifest.json | cut -d' ' -f1); echo $H
```

`$H` is your bundle's identity: keep this shell, or note the value, because the steps below use it. The manifest
records where the files were read from, among other things, so `$H` will differ from the value in the shipped
`serve.conf`. Use yours below.

Copy node0's part of the bundle (everything except node1's cache), then check each box's copy:

```bash
rsync -a --exclude rank1/ ~/models/tf-port/inputs-tp2-c178a012 node0:models/tf-port/
tools/dsv41-hip/verify_inputs.sh ~/models/tf-port/inputs-tp2-c178a012 1 \
    ~/models/tf-port/libertai-2.9bpw ~/models/tf-port/ds1017-1bae0e10 $H
```

Rank 0's check runs on node0 after the deploy below, because it uses the deployed tools.

### Deploy

[BUILD_AND_RUN.md §4](../tools/dsv41-hip/BUILD_AND_RUN.md#4-deploy-both-boxes-run-one-commits-bytes): both boxes run
one commit's bytes. The checkout must be clean and on `main`.

```bash
C=$(git rev-parse HEAD)
tools/dsv41-hip/deploy_rank0.sh --native zig-out/native-hip --bin-commit $C --ds-kit ~/models/tf-port/inputs-tp2-c178a012/kit
tools/dsv41-hip/deploy_rank0.sh --host local --native zig-out/native-hip --bin-commit $C --ds-kit ~/models/tf-port/inputs-tp2-c178a012/kit
```

Now check node0's copy of the inputs. The check needs `tf-dsv41-check`, which is in node1's build:

```bash
ssh node0 mkdir -p tfp/bin && scp zig-out/bin/tf-dsv41-check node0:tfp/bin/
ssh node0 "TFP_BIN=\$HOME/tfp/bin ~/tfp/deploy/current/tools/dsv41-hip/verify_inputs.sh ~/models/tf-port/inputs-tp2-c178a012 0 \
    ~/models/tf-port/libertai-2.9bpw ~/models/tf-port/ds1017-1bae0e10 $H"
```

Each check writes a report beside the bundle (`inputs-tp2-c178a012.verify-rank<R>-<host>.json`). A rank starts only
if its box's report passed.

### Serve

[BUILD_AND_RUN.md §5](../tools/dsv41-hip/BUILD_AND_RUN.md#5-serve). Make your configuration from the shipped
`serve.conf`, set your bundle's hash in it, and put the same file on both boxes:

```bash
mkdir -p ~/tfp/run/conf
install -m 644 ~/tfp/deploy/current/tools/dsv41-hip/serve.conf ~/tfp/run/conf/my.conf
sed -i "s/^BUNDLE_SHA256=[0-9a-f]*/BUNDLE_SHA256=$H/" ~/tfp/run/conf/my.conf
ssh node0 mkdir -p tfp/run/conf && scp ~/tfp/run/conf/my.conf node0:tfp/run/conf/my.conf

cd ~/tfp/deploy/current/tools/dsv41-hip
TF_SERVE_CONF=$HOME/tfp/run/conf/my.conf ./first_light.sh up
```

`up` does the following, and takes everything down again if any step fails:
- checks that both ranks use the same deployment, kit and configuration;
- takes each box's GPU lock;
- starts rank 1, then rank 0;
- waits until the weights load and `/health` answers;
- sends a probe question. The run is ready only when the reply contains "Paris".

With `BENCH=1`, the shipped setting, `up` then times three requests and prints their speed: a code prompt, a prose
prompt and a 1,818-token prompt. On our Ryzen AI Halo pair, the code prompt writes about 74.5 tok/s with drafting on
([Settings that matter](../tools/dsv41-hip/BUILD_AND_RUN.md#settings-that-matter)).

The server listens on node0 at `127.0.0.1:18080` and speaks the OpenAI chat API. Your first request, on node0 (or
from node1 inside `ssh node0 '...'`):

```bash
curl -s http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "dsv41",
  "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 64, "temperature": 0}'
```

A good reply is a JSON chat completion whose `choices[0].message.content` names Paris. To reach the server from
another machine, forward the port: `ssh -L 18080:127.0.0.1:18080 node0`. For agents and tool calls, see
[AGENTS.md](AGENTS.md).

`./first_light.sh status` shows both ranks. `./first_light.sh down` stops them and releases the GPU locks; run it
with the same `TF_SERVE_CONF` as `up`.

### Benchmark

The published numbers come from `qual/qual_bench.sh`, which runs the prompt set of the published two-DGX-Spark
benchmark ([RESULTS.md](RESULTS.md#how-we-measure)). It starts its own run, so stop the server first.

It needs the benchmark's prompt file, from the repository and commit that `source_lock.json` names under `recipe`:

```bash
TF_SERVE_CONF=$HOME/tfp/run/conf/my.conf ./first_light.sh down
git clone https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark ~/tfp/qual/recipe
git -C ~/tfp/qual/recipe checkout 9fa230d3eb119f588e690bf9b87ce41ed902db60
mkdir -p ~/tfp/qual/bench/creator && cp ~/tfp/qual/recipe/bench/prompts-v1.json ~/tfp/qual/bench/creator/

cd ~/tfp/src/TensorFold/tools/dsv41-hip
qual/qual_bench.sh bench '$HOME/tfp/run/conf/my.conf' first --reps 5 --prefill 2000,8000
```

Results land in `~/tfp/qual/out/bench-first/`.

Ours, from [Results](../README.md#results):
- One chat writes 52.8 tok/s: the mean of the eight category medians, drafting on.
- Prompt reading is 1,414 tok/s at 2,952 tokens and 1,543 tok/s at 11,594 tokens.

Prompts longer than about 8,000 tokens need a larger `CONTEXT` in your configuration
([Settings that matter](../tools/dsv41-hip/BUILD_AND_RUN.md#settings-that-matter)).

Our numbers were measured inside quiet windows on both boxes, with nothing else running. Other load on either box
lowers the speed (`QUIET_BIN` in [BUILD_AND_RUN.md §6](../tools/dsv41-hip/BUILD_AND_RUN.md#6-check-and-measure)).
That section also has the served functional suite (`qual/qual_serve.sh`) and the fault checks. If you measure a pair,
please add your numbers to [COMMUNITY-RESULTS.md](COMMUNITY-RESULTS.md).

## 6. If something goes wrong

- **Box setup** (USB stick, boot menu, GPU memory, Secure Boot):
  [BOSGAME-M5.md, If something goes wrong](BOSGAME-M5.md#if-something-goes-wrong).
- **The link:** [TUNING.md](TUNING.md). A missing ping usually means crossed cables (step 3).
- **Serving, the ranks and the link at run time:**
  [BUILD_AND_RUN.md §7 Debugging](../tools/dsv41-hip/BUILD_AND_RUN.md#7-debugging) and
  [§8 Known issues](../tools/dsv41-hip/BUILD_AND_RUN.md#8-known-issues).
- **`inputs_manifest.json is X, the config pins Y`:** `BUNDLE_SHA256` in your configuration isn't your bundle's hash.
  Set it to `$H` (above) on both boxes.
- **`no verify report`:** run `verify_inputs.sh` on that box (the Inputs and Deploy steps above).
- **`is not idle` or `MemAvailable ... a rank needs`:** another engine is running on that box, or there's too little
  free memory. A rank needs `PEAK_GIB` + 8 GiB free.
- **`the GPU lock ... was not free`:** another GPU job holds `/var/tmp/halo-gpu.lock` on that box. For example, a
  run you didn't stop: use `./first_light.sh down`.
- **Rank 0's log is empty after `up` returns:** your system ends a session's processes when SSH disconnects. See
  `SPAWN_UNIT` in `serve.conf`.
- **Questions:** the [StrixFold Discord](https://discord.gg/D4FgAJnKd9) or a GitHub issue.

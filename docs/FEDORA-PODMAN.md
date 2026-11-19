# StrixFold in Podman on Fedora

This guide runs StrixFold on Fedora with its userspace in a rootless Podman container: ROCm, Python 3.13 and the build
tools live in the image, and the host keeps only the kernel, its GPU driver, the USB4 links and Podman. The files are in
[packaging/podman/](../packaging/podman/). It is a different route to the same steps as
[GETTING-STARTED.md](GETTING-STARTED.md): sections 1 to 3 below replace its steps 2 and 3, and from step 4 on you follow
GETTING-STARTED inside the container, with the differences in [section 4](#4-getting-started-inside-the-container).

A community pair was set up this way in November 2026:

| | node0 | node1 |
|---|---|---|
| Box | Framework Desktop (BIOS 3.06) | Minisforum MS-S1 Max |
| OS | Fedora 44 Server, kernel 7.3.3 | Fedora 45 Server, kernel 7.3.3 |
| Podman | 6.1.3 (crun) | 6.1.3 (crun) |

Both have the Ryzen AI Max+ 395 with 128 GB. The image has ROCm 10.1.0 (HIP 7.16, AMD clang 24), newer than the
10.0.0-4 (HIP 7.15) the published numbers use; StrixFold's build, kit and GPU tests pass on it
([Results](#results)).

## 1. The host, on both boxes

Install Podman if it isn't there, put yourself in the GPU's groups, then log out and back in:

```bash
sudo dnf install -y podman
sudo usermod -aG render,video "$USER"
```

Give the GPU 124 GiB with the same kernel options as the other boxes, then restart:

```bash
sudo grubby --update-kernel=ALL --args="amdgpu.gttsize=126976 ttm.pages_limit=32505856"
cat /sys/class/drm/card*/device/mem_info_gtt_total   # 133143986176 after the restart
```

Ours also had `iommu=pt`. A pair of Framework Desktops needed BIOS 3.06 and `amd_iommu=pt` both before their USB4
link came up ([HARDWARE.md](HARDWARE.md#usb4-ports)). Stop the boxes from sleeping: the input preparation runs for a
long time.

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

The container runs as a systemd user service. Let your user manager run without a login, so the container starts at
boot and stays up when the last SSH session closes:

```bash
sudo loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger   # Linger=yes
```

Fedora updates its kernel often, and the GPU driver comes with it. Keep a kernel that works until you have tested the
next one.

## 2. The USB4 links

Load `thunderbolt_net` at every start, and keep the USB4 controllers awake (why: [BOSGAME-M5.md, Keep the USB4
controllers awake](BOSGAME-M5.md#keep-the-usb4-controllers-awake)). Fedora needs no `bolt` setup for a host-to-host
link.

```bash
echo thunderbolt_net | sudo tee /etc/modules-load.d/thunderbolt_net.conf
sudo modprobe thunderbolt_net
sudo tee /etc/systemd/system/usb4-awake.service >/dev/null <<'EOF'
[Unit]
Description=Keep the USB4 controllers awake
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for f in /sys/bus/pci/drivers/thunderbolt/0000:*/power/control /sys/bus/thunderbolt/devices/[0-9]-0/power/control; do [ -e "$f" ] && echo on > "$f"; done'

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now usb4-awake.service
```

**Ports.** On the MS-S1 Max, the front USB4 ports (AMD) found no link to the Framework Desktop's USB4 ports, while a
USB4 drive worked in them; the rear USB4 v2 ports (an Intel controller) linked at once. Cable both of the Framework's
ports to the MS-S1 Max's rear ports. `sudo dmesg | grep -i thunderbolt` shows whether the boxes saw each other.

**Addresses.** NetworkManager makes a "Wired connection" profile for each new interface; delete those, then add one
profile per cable with the addresses `serve.conf` expects
([GETTING-STARTED step 3](GETTING-STARTED.md#3-link-the-two-boxes)). Match each profile by the interface's MAC address,
not its name: `thunderbolt0` and `thunderbolt1` are numbered in the order the links come up, while the MAC stays with
the port (on the MS-S1 Max its first byte differs between the two rear ports). Read the MACs:

```bash
ip -br link | grep thunderbolt
nmcli -t -f NAME,DEVICE con show | grep thunderbolt   # generated profiles: sudo nmcli con delete NAME
```

Then on node0, with your own MACs (use `.2` on node1):

```bash
sudo nmcli con add type ethernet con-name usb4-a 802-3-ethernet.mac-address MAC_OF_CABLE_A \
    ipv4.method manual ipv4.addresses 10.40.0.1/24 ipv4.never-default yes ethernet.mtu 1500 ipv6.method disabled \
    connection.zone trusted
sudo nmcli con add type ethernet con-name usb4-b 802-3-ethernet.mac-address MAC_OF_CABLE_B \
    ipv4.method manual ipv4.addresses 10.40.1.1/24 ipv4.never-default yes ethernet.mtu 1500 ipv6.method disabled \
    connection.zone trusted
```

`connection.zone trusted` opens the ranks' ports on the link: Fedora's firewalld blocks them otherwise.
`ipv4.never-default` keeps the boxes' internet traffic off the cables. Check the link as in GETTING-STARTED step 3.

**Speed.** One TCP stream (iperf3) per cable, the pair above:

| Direction | One cable | Both at once |
|---|---:|---:|
| MS-S1 Max (Intel) to Framework (AMD) | 20.9 Gb/s | 22.7 + 22.7 Gb/s |
| Framework (AMD) to MS-S1 Max (Intel) | 9.4 Gb/s | 10 + 10 Gb/s |

The ranks send each other equal amounts, so the slower direction sets the pace; the speed in [Results](#results) is
still close to the published pairs'.

## 3. The container

**Build** the image on node1 and copy it to node0, so both boxes run the same bytes:

```bash
cd TensorFold    # your checkout
packaging/podman/strixfold-ct.sh build
podman save localhost/strixfold-rocm:latest | ssh 10.40.0.1 podman load
podman image inspect --format '{{.Id}}' localhost/strixfold-rocm:latest
ssh 10.40.0.1 "podman image inspect --format '{{.Id}}' localhost/strixfold-rocm:latest"   # the same
```

The [Containerfile](../packaging/podman/Containerfile) pins AMD's ROCm image and uv by digest and Python by version.
node0 needs `packaging/podman/` too (for `strixfold-ct.sh` and the units); copy the directory anywhere on it.

**Install the service** on both boxes:

```bash
mkdir -p ~/.config/containers/systemd
cp packaging/podman/strixfold.container ~/.config/containers/systemd/
systemctl --user daemon-reload
systemctl --user start strixfold
packaging/podman/strixfold-ct.sh check   # gfx1151, HIP 7.16..., clang, libamdhip64.so.7, Python 3.13.18
```

The container gets your home directory, `/var/tmp` (the GPU lock) and the host's time zone. If `~/tfp` or `~/models`
point into another drive, mount it too with a drop-in (here `/data`):

```bash
mkdir -p ~/.config/containers/systemd/strixfold.container.d
printf '[Unit]\nRequiresMountsFor=/data\n\n[Container]\nVolume=/data:/data\n' \
    > ~/.config/containers/systemd/strixfold.container.d/data.conf
systemctl --user daemon-reload && systemctl --user restart strixfold
```

The container is made new at every start, and anything outside the mounted directories goes with it: keep work in
your home directory. `strixfold-ct.sh create` makes the same container by hand if you'd rather not use the service
(`CT_DATA=/data` for the extra mount).

**node0's SSH.** node1 drives node0 over SSH, and the commands it sends must run inside node0's container. On node0,
write an sshd configuration for its USB4 address and install the sshd service, which listens on port 2222 and retries
until the link is up:

```bash
packaging/podman/strixfold-ct.sh sshd-config 10.40.0.1
cp packaging/podman/strixfold-sshd.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now strixfold-sshd
```

It accepts the keys in node0's `~/.ssh/authorized_keys`, so the key you copied in GETTING-STARTED step 3 works. On
node1, point the name `node0` at that port:

```bash
cat >> ~/.ssh/config <<'EOF'
Host node0
    HostName 10.40.0.1
    Port 2222
EOF
packaging/podman/strixfold-ct.sh shell   # then, inside: ssh node0 hostname
```

## 4. GETTING-STARTED inside the container

Open a shell in the container with `strixfold-ct.sh shell` and follow GETTING-STARTED from step 4. The differences:

- **Python.** The image's `python3` is 3.14, which ROCm's own tools use. Make StrixFold's venvs with `python3.13`
  (`python3.13 -m venv ~/venvs/tfp-inputs`, on node0 too); inside a venv, `python3` is 3.13.
- **Builds and input jobs.** StrixFold's scripts cap their builds and input jobs with `systemd-run --scope`, which a
  container doesn't have. Its stand-in keeps the scope's CPU set (`taskset`) and prints which caps it could not apply.
  To get the CPU and memory limits too, run those jobs from the host with `strixfold-ct.sh job` (next section).
- **`SPAWN_UNIT=0`**, as shipped in `serve.conf`. The ranks start inside the container's shell; the stand-in refuses
  user units.
- **The GPU lock** is shared with the host through `/var/tmp`, but the ranks' checks see only the container's
  processes. Don't run other GPU work on the host while StrixFold runs.
- **Restarting a container ends its rank.** Run `first_light.sh down` before `systemctl --user restart strixfold` or a
  restart of either box.

## 5. Limits for builds and input jobs

`strixfold-ct.sh job CMD` runs CMD from the host in a throwaway container with the same image, devices and mounts as
the service, limited to 8 CPUs and 16 GiB (throttled from 12 GiB), with no swap. Run it from the directory CMD needs:

```bash
cd ~/tfp/src/TensorFold
CT_DATA=/data packaging/podman/strixfold-ct.sh job tools/dsv41-hip/zig_build.sh install
```

`CT_JOB_CPUS`, `CT_JOB_MEM` and `CT_JOB_MEM_HIGH` change the limits; `CT_JOB_MEM_HIGH=max` drops the throttle, so a
job that runs away is stopped at `CT_JOB_MEM` at once instead of slowing down first. A job over its limit is killed
(exit 137) and leaves the host's memory alone. The ranks are never limited: they run in the service's container.

## Results

The pair above, `serve.conf` as shipped (`DRAFTS=1`), the bundle's own `BUNDLE_SHA256`:

| | This pair | Published Ryzen AI Halo pair |
|---|---:|---:|
| `first_light.sh` code / prose | 72.2 / 41.9 tok/s | 74.5 / 42.4 tok/s |
| `first_light.sh` prompt reading, 1,818 tokens | 1,363 tok/s | 1,381 tok/s |
| `qual_bench.sh`, mean of the 8 category medians | 58.6 tok/s | 52.8 tok/s |
| `qual_bench.sh` prompt reading, 2,952 tokens | 1,349 tok/s | 1,414 tok/s |

Commit 93d2845 (the `qual_bench.sh` rows at fc1399c). The published pair's `qual_bench.sh` numbers come from the October
release build, so those two rows compare builds as well as boxes. Host tests passed 371 of 379 (8 skipped); the Triton
kit built 280 of 280 kernels; the GPU tests passed (smoke 14, ABI 6, glue 11; the f16x2 lane bug is present with clang
24, as with clang 23). From a cold boot the container started 15 s after the box, and `first_light.sh up` was ready in
41-42 s. Under `qual_bench.sh` the Framework Desktop's GPU reached 91 °C at about 2,880 MHz without throttling.

## If something goes wrong

- **The container stops when you log out:** linger is off ([section 1](#1-the-host-on-both-boxes)).
- **`ssh node0` is refused:** `systemctl --user status strixfold-sshd` on node0. It retries until node0's USB4 address
  exists, so check the link first.
- **`scp` to node0 fails with "Connection closed":** an sshd configuration without the sftp line; run
  `strixfold-ct.sh sshd-config` again.
- **`systemd-run stand-in: no user units in the container`:** set `SPAWN_UNIT=0` in your `serve.conf`.
- **`DriverUnavailable` from StrixFold's programs:** an image built without the Containerfile's `ldconfig` line;
  rebuild it.
- **`first_light.sh down` takes minutes:** the container has no init; the service and `create` both start one.
- **A box restarted while serving:** the other box's rank may still run; run `first_light.sh down`, then `up`.

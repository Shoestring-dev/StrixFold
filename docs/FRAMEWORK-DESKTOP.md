# Running StrixFold on two Framework Desktops (Ubuntu 26.10)

A pair of Framework Desktops (AMD Ryzen AI Max 300 Series, Ryzen AI Max+ 395, Radeon 8060S / gfx1151, 128 GB) on
Ubuntu 26.10 with the stock Ubuntu kernel 7.3.0-10-generic, instead of the Debian-13 rex stack in
[GETTING-STARTED.md](GETTING-STARTED.md). This page records the deviations that were needed and what was verified.
The stack here is ROCm 10.1.0 (HIP 7.16), newer than the 10.0.0-4 (HIP 7.15) the published numbers use, as
[FEDORA-PODMAN.md](FEDORA-PODMAN.md) reports for its image. Status at time of writing: link up, Triton kit, zig
build and the GPU tests (smoke, ABI, glue) green; serving and the bench cases pending.

## What was different, and what fixed it

### 1. BIOS 3.06 is required for the USB4 link (the critical one)

On BIOS 3.05, `thunderbolt_net` loaded but created no interface, at any port, on either box: the USB4 routers'
switches exposed no external ports, the UCSI typec controller reported an empty state, and plugging/unplugging a
cable produced no kernel event at all. `amd_iommu=off` alone did not explain it (the XDomain refusal), and neither
did `amd_iommu=pt` alone.

Two changes together fixed it:

- `amd_iommu=pt` instead of `amd_iommu=off`: the kernel refuses XDomain (host-to-host) links when IOMMU is disabled,
  but `pt` keeps the passthrough DMA mapping ROCm wants while still creating the IOMMU groups XDomain needs.
- Framework BIOS 3.06 (fwupd/LVFS: 0.0.3.5 → 0.0.3.6), whose changelog carries the UCSI fixes ("delay in sending
  the 3A profile to prevent incomplete UCSI PPM command resets during S0 boot transitions", "validation logic to
  reject invalid USB PD index values"). After this update, `thunderbolt0`/`thunderbolt1` appeared on both boxes
  within seconds of the cable being attached, without a manual `modprobe`.

After both boxes were on 3.06 + `amd_iommu=pt`, both cables linked (ping ~1 ms between the boxes).

Note: this covers Framework-to-Framework. The community report in [GETTING-STARTED.md](GETTING-STARTED.md) §3
(Framework + Minisforum MS-S1 Max) is about the MS-S1 side: that Framework was already on BIOS 3.06, and its ports
linked to the MS-S1 Max's rear Intel USB4 v2 ports but not to the MS-S1's AMD ports.

### 2. Only the rear USB-C ports are USB4

The front USB-C ports are USB 3.2 only. A cable there produces exactly the "no event anywhere" symptom above. Use
the rear ports at the I/O shield.

### 3. GRUB line (GPU memory)

The shipped line from [BOSGAME-M5.md](BOSGAME-M5.md) step 5 plus the IOMMU option:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.gttsize=126976 ttm.pages_limit=32505856 amd_iommu=pt"
```

`ttm.pages_limit=32505856` (124 GiB) is the shipped value; this pair previously ran 33030144 (126 GiB) with no
observed difference, and the shipped value leaves the OS about 2 GiB more. The pair's boxes also carried `kho=off`,
`amdgpu.noretry=0` and `amdgpu.lockup_timeout=...` from an earlier setup; none of them is needed here, so they are
dropped.

### 4. ROCm from AMD's Ubuntu repo, gfx1151-only

The rex packages (debs.ryai.dev) are Debian-13 builds; on Ubuntu use AMD's own stable repo for ubuntu2604
(`https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/`, key at
`https://stable.repo.amd.com/rocm/gpg/packages.gpg`). Install the arch-specific metas:

```
sudo apt install amdrocm10.1-gfx1151 amdrocm-core-sdk10.1-gfx1151
```

Do not install the plain `amdrocm10.1` meta: it pulls every architecture's libraries (several GB of
gfx908/gfx942/gfx1201/gfx1250) for nothing. This gives ROCm 10.1.0 (HIP 7.16), newer than the 10.0.0-4 (HIP 7.15)
the published numbers use; the build, kit and GPU tests pass on it.

### 5. Register /opt/rocm/lib with ldconfig (the therock equivalent)

The ubuntu2604 repo has no `therock-gfx1151` package, and without its registration the StrixFold programs fail with
`error: DriverUnavailable` (rocminfo still works). One file replaces it:

```
echo /opt/rocm/lib | sudo tee /etc/ld.so.conf.d/amdrocm.conf
sudo ldconfig
```

### 6. Python 3.13 on Ubuntu 26.10

Ubuntu 26.10 ships 3.14/3.15; the pinned Triton wheel is cp313. Two ways to get 3.13:

- `uv python install 3.13` (what [FEDORA-PODMAN.md](FEDORA-PODMAN.md)'s Containerfile does) — no PPA, no suite mixing.
- Or the deadsnakes PPA: it has no 26.10 (stonking) build, but its 26.04 (resolute) pocket installs and works:

```
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo sed -i 's/^Suites: stonking/Suites: resolute/' /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-*.sources
sudo apt update && sudo apt install -y python3.13 python3.13-venv
```

### 7. IP configuration per install flavor

The link itself follows [GETTING-STARTED.md](GETTING-STARTED.md) §3 (10.40.0.1/.2 + 10.40.1.1/.2, MTU 1500, IPv6
off). Two notes:

- A Server install has no `nmcli`; use a netplan file. This netplan (1.2.2) rejects `ipv6-method: disabled`; use
  `dhcp4: false`, `dhcp6: false`, `accept-ra: false` instead, and `chmod 600` the file.
- A Desktop install uses NetworkManager; the nmcli lines from §3 work as written.

## Verified on this pair (as of this draft)

- BIOS 3.06 on both boxes; `thunderbolt0`/`thunderbolt1` up on both, MTU 1500, ping ~1 ms on both cables.
- Triton AOT kit: 280 variants built, 0 failed (pinned triton 3.8.0+rocm10.1.0a wheel, cp313).
- `zig_build.sh install` and the test build (`native-hip`, kit attached): green.
- GPU tests (`run_gpu_tests.sh`): smoke, ABI and glue all exit 0.
- `rocminfo` sees gfx1151; HIP 7.16.
- Not yet run at draft time: `first_light.sh` up with the bench cases; those numbers go here before this leaves draft.

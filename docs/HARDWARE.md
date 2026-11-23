# Which boxes StrixFold runs on

StrixFold needs two boxes with the AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151) and 128 GB of memory each, linked
by two USB4 cables. Every box built on that chip has the same GPU and memory, so the engine and its speed carry over
from one brand to another. What changes between brands is the firmware, which USB4 ports will link to another
computer, the power limits and the cooling. This page lists the boxes StrixFold is known to run on and what to watch
for on each.

## Boxes

| Boxes | Status | Guide | What to watch |
|---|---|---|---|
| 2 × AMD Ryzen AI Halo | Measured: the published numbers | [GETTING-STARTED 2a](GETTING-STARTED.md#2a-amd-ryzen-ai-halo) | Ships with the tested stack |
| 2 × Bosgame M5 | Measured by us | [BOSGAME-M5.md](BOSGAME-M5.md) | Links train at Gen 2 until replugged; 85 W sustained in the stock power mode |
| Framework Desktop + Minisforum MS-S1 Max | Measured by a user, in a container on Fedora | [FEDORA-PODMAN.md](FEDORA-PODMAN.md) | Cable the Framework to the MS-S1 Max's rear USB4 v2 ports |
| 2 × Framework Desktop | Measured by a user, with 1M-token context | [FRAMEWORK-DESKTOP.md](FRAMEWORK-DESKTOP.md) | BIOS 3.06 and `amd_iommu=pt` before the link comes up; rear USB-C ports only; a load-wedge workaround on the stock Ubuntu kernel (guide §9) |
| 2 × Minisforum MS-S1 Max | Reported working, with 1M-token context | | |
| 2 × GMKtec EVO-X2 | Measured by a user, with the links at Gen 2 | | One USB4 port on the front and one on the back: cable front to front and back to back |
| 2 × Beelink GTR9 Pro | Reported working, two USB4 cables | | |
| Minisforum MS-S1 Max + GMKtec EVO-X2 | Reported working | | 40 Gb/s through the MS-S1 Max's USB4 v2 ports, 20 Gb/s through its other USB4 ports |

**Measured** means a run with the bench cases on [COMMUNITY-RESULTS.md](COMMUNITY-RESULTS.md), where the numbers
are. **Reported working** means an owner told us on the Discord in November 2026 that it serves, without numbers in
that format yet. If you have one of these, a measured run moves it up the table: see [Add your boxes](#add-your-boxes).

A pair doesn't have to be one brand: both mixed pairs above serve normally.

## USB4 ports

Not every USB4 port will link to another computer. The interface only appears once both ends agree on a host-to-host
link, and a port that works for a USB4 drive may still refuse one. What owners have found:

- **Minisforum MS-S1 Max:** its rear USB4 v2 ports (an Intel controller) link to other boxes at once. Its front USB4
  ports (AMD) found no link to a Framework Desktop, and linked at 20 Gb/s to a GMKtec EVO-X2.
- **Framework Desktop:** only the rear USB-C ports are USB4; the front ones are USB 3.2. On BIOS 3.05 no port made an
  interface; BIOS 3.06 together with the kernel option `amd_iommu=pt` fixed it (the kernel refuses host-to-host links
  with the IOMMU off).
- **GMKtec EVO-X2:** one USB4 port on the front and one on the back. An owner cabled front to front and back to back.
- **Bosgame M5:** both of its USB4 ports link, but at Gen 2 until both ends are unplugged and plugged back in
  ([The USB4 links](BOSGAME-M5.md#the-usb4-links)).
- **Cables:** use cables rated for 40 Gb/s or more. A Framework Desktop owner found the cable made the difference.

To see where a link stands:

```bash
sudo dmesg | grep -i thunderbolt                    # did the boxes see each other at all
ip -br link | grep thunderbolt                      # one interface per linked cable
cat /sys/bus/thunderbolt/devices/*/rx_speed          # the speed each link trained at
```

**Link speed matters less than it looks.** Each decode step sends small messages between the boxes, so latency
counts, not bandwidth: on a pair of M5s, 2.7 times the bandwidth gave 1–3% more decode and 6% more prompt reading.
Keeping two cables matters more: one cable cost the same pair 6–7% decode and 14–16% prompt reading
([BOSGAME-M5.md](BOSGAME-M5.md#the-usb4-links)). A cable between an AMD and an Intel controller can also run slower in
one direction than the other (9.4 and 20.9 Gb/s, [FEDORA-PODMAN.md](FEDORA-PODMAN.md#2-the-usb4-links)); the pair
still measured close to ours.

## Firmware

- **AMD's USB4 host-to-host setting** lifts one TCP stream per cable from about 9 to about 30 Gb/s. It is a BIOS
  setting, so it depends on your box's BIOS: Bosgame's BIOS 1.10 carries it, and Framework has announced it for its
  Desktops. On older BIOSes, M5 owners can turn it on from Linux
  ([BOSGAME-M5.md](BOSGAME-M5.md#optional-the-host-to-host-setting)). It changes StrixFold's speed by a few percent at
  most (above).
- **Memory for the GPU** is set by kernel options, not the BIOS: keep the BIOS's own GPU memory (UMA frame buffer) at
  its smallest value and give the GPU 124 GiB as in [BOSGAME-M5.md step 5](BOSGAME-M5.md#5-give-the-gpu-124-gib).

## Power and cooling

Each brand ships its own power limits and fan curves, and they decide the speed more than the link does. An M5 in its
stock mode holds 85 W and runs about 5–10% slower than the same pair at 105–120 W, with the GPU far hotter
([BOSGAME-M5.md](BOSGAME-M5.md#speed-on-a-pair-of-m5s)). Under `qual_bench.sh` a Framework Desktop's GPU reached
91 °C at about 2,880 MHz without throttling. If your numbers come out low, check the GPU clock and temperature under
load before anything else.

## Operating systems

The published numbers use Debian 13 with AMD's kernel 7.2.7 and ROCm 10.0.0-4 (HIP 7.15). Owners also run:

- **Fedora 44 and 45** with ROCm 10.1 inside a rootless Podman container: [FEDORA-PODMAN.md](FEDORA-PODMAN.md).
- **Ubuntu 26.10** with AMD's ROCm 10.1 packages: [FRAMEWORK-DESKTOP.md](FRAMEWORK-DESKTOP.md).

## What doesn't work yet

- **Boxes with less than 128 GB.** Each box holds half of the 197 GB model plus its caches, with the GPU given 124 GiB.
- **Three or more boxes.** StrixFold splits the model across exactly two.
- **Other Ryzen AI Max chips and laptops** with the 395 haven't been reported.

## Add your boxes

Run the bench cases on [COMMUNITY-RESULTS.md](COMMUNITY-RESULTS.md) and post them with the **Results from your
hardware** issue form, or in `#benchmarks` on the [Discord](https://discord.gg/D4FgAJnKd9). Along with the numbers,
tell us the box and its BIOS version, which ports the cables are in and the speed they trained at (`rx_speed`), and
what you had to change to get there. Each of those goes on this page.

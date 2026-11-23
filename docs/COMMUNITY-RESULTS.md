# Results from your hardware

StrixFold's published numbers come from one pair of boxes. This page collects runs on other people's: other Strix Halo
brands, other cables, network cards instead of USB4, other ROCm and kernel versions. Numbers from hardware we don't own
are the best check on ours.

## Add yours

Open an issue with the **Results from your hardware** form (New issue on GitHub), or post the same details in the
`#benchmarks` channel on the [Discord](https://discord.gg/D4FgAJnKd9). We add each run here with your name or handle,
unless you ask us not to.

What to run, on the build you serve:

1. `tools/dsv41-hip/serve-rank0.sh bench`: the code, prose and prompt-reading cases (`RUN/bench.json`).
2. `python3 tools/dsv41-hip/ctx_sweep.py http://127.0.0.1:18080 dsv41 zig/src 0,2000,8000,32000 2 sweep.json RUN/rank0.log`:
   writing speed against context length.
3. Optional: the full prompt set with `tools/dsv41-hip/qual/qual_bench.sh` (BUILD_AND_RUN.md section 6), the method
   behind the published table.

Say which settings you served with. `DRAFTS` matters most: drafting on (the default) roughly doubles writing speed and
gives exactly the same text.

## Runs

| Who | Boxes | Link | GPU stack | Settings | Code, prose (tok/s) | Prompt reading | Notes |
|---|---|---|---|---|---:|---:|---|
| StrixFold (6 Nov 2026) | 2 × AMD Ryzen AI Halo, 128 GB | 2 × USB4, MTU 1500 | ROCm 10.0.0-4, kernel 7.2.7 | `serve.conf` defaults, `DRAFTS=1` | 74.5, 42.4 | 1,381 at 1,818 tokens | balanced power; the bench cases above |
| StrixFold (12 Nov 2026) | 2 × Bosgame M5, 128 GB | 2 × USB4 at Gen 3 with the host-to-host setting, MTU 1500 | ROCm 10.0.0-4, kernel 7.2.7 | `serve.conf` defaults, `DRAFTS=1` | 70.7, 41.1 | 1,284 at 1,818 tokens | stock power (85 W); `first_light.sh`; [BOSGAME-M5.md](BOSGAME-M5.md#speed-on-a-pair-of-m5s) |
| derekwheel (18 Nov 2026, [#36](https://github.com/Shoestring-dev/StrixFold/pull/36)) | Framework Desktop + Minisforum MS-S1 Max, 128 GB each | 2 × USB4 into the MS-S1 Max's USB4 v2 ports, MTU 1500 | ROCm 10.1.0 (HIP 7.16) in Podman, kernel 7.3.3 | `serve.conf` as shipped, `DRAFTS=1` | 72.2, 41.9 | 1,363 at 1,818 tokens | Fedora 44/45; `first_light.sh` at 93d2845; [FEDORA-PODMAN.md](FEDORA-PODMAN.md) |
| dineshr93 (19 Nov 2026, [#50](https://github.com/Shoestring-dev/StrixFold/pull/50)) | 2 × Framework Desktop, 128 GB | 2 × USB4, MTU 1500 | ROCm 10.1.0 (HIP 7.16), kernel 7.3.0-10 | `serve.conf` + `CONTEXT=1048576`, `DRAFTS=1` | 71.7, 41.7 | 1,258 at 1,818 tokens | Ubuntu 26.10; BIOS 3.06 + `amd_iommu=pt`; at 8K: 70.8, 41.9, 1,260; staggered start (load wedge, guide §9); [FRAMEWORK-DESKTOP.md](FRAMEWORK-DESKTOP.md) |
| willaien (19 Nov 2026, Discord `#benchmarks`) | 2 × GMKtec EVO-X2, 128 GB | 2 × USB4, front to front and back to back, both at Gen 2 (10 Gb/s per lane) | not given | `serve.conf` with 256K context; other settings not given | 72.2, 40.5 | 1,222 at 1,818 tokens | balanced power on both boxes; the bench cases above |

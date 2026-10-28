# tools/dsv41-hip: the Strix Halo port's tools and notes

Start with [BUILD_AND_RUN.md](BUILD_AND_RUN.md): it is the one guide to building, deploying, serving and measuring.

| What | Where |
|---|---|
| Build, deploy, serve | `zig_build.sh`, `run_gpu_tests.sh`, `deploy_rank0.sh`, `serve.conf`, `first_light.sh`, `serve-rank0.sh`, `serve-rank1.sh`, `serve_lib.sh` |
| Inputs | `prepare_inputs.sh`, `verify_inputs.sh` / `.py`, `requirements-inputs.lock.txt`, `rope_reference.py` |
| Checks and benchmarks | `qual/` (served suite, faults, benchmarks, soak, numerics traces), `qualify.sh` / `qualify.py`, `gates.json` / `gates.py` |
| Kernels | `tri/` (the Triton kernels and how the AOT kit is built), `exl3/` (the EXL3 GEMM work), `kernels.json`, `kernel_registry.py` |
| The two-box link | `link/`, `coll_fault_matrix.sh`, `COLLECTIVES-DESIGN.md` |
| Engineering notes | `ADAPTER.md`, `KERNELS.md`, `COLLECTIVES-DESIGN.md`, `RESULTS.md`, `exl3/KDEC.md` |

## Reading the engineering notes

The notes were written while the port was built, by several work streams at once, and keep their working names:

| Name | What it was |
|---|---|
| `w-tri` | the Triton kernels: porting them to AMD and the ahead-of-time kit (`tri/`) |
| `w-exl3`, `k-dec` | the EXL3 weight kernels and their decode-speed work (`exl3/`) |
| `w-arch` | the engine glue: model plan, rounds, memory (`zig/src/families/dsv41/`) |
| `w-link` | the USB4 transport and collectives between the boxes (`link/`) |
| `w-ref` | the reference outputs and fixtures the numerics are checked against |
| `q-qual`, `p-par`, `p-pre2`, `a-arch2` | qualification, parity, prefill and review passes recorded in `gates.json` |
| `node0`, `node1` | rank 0 (serves) and rank 1 (builds), as in BUILD_AND_RUN.md |
| `~/tfp` | the working folder on both boxes: sources, deployments, runs and outputs |
| `strix-port` | the branch the port was developed on; this repository's `main` |

Paths such as `node1:~/tfp/...` in `gates.json` and the notes are where evidence was kept on the boxes the port was
measured on. They are a record, not something to recreate.

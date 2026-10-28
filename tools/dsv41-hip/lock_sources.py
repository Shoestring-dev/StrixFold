#!/usr/bin/env python3
"""lock_sources.py OUT_JSON: the port's lock manifest, assembled from what is on node1 (and, over ssh, node0), never
typed in.

Every SHA here is read back from git, every file hash computed or taken from the verify_hf.py result files, every
toolchain version from the preflight JSONs of both boxes. Pinned values from the handoff are listed beside what was
found, with a match flag. The manifest names its own generator (path, SHA-256, git commit: provenance.py). Re-run
after any change; a frozen copy goes with each benchmark. Read-only except OUT_JSON (git reads never fetch: blobs
absent from a blobless clone are recorded by blob id only).

Sources (handoff section 3 plus PLAN.md): A base, B TP4 reference, D CUDA/Triton source, the 9f820999 fix, the B2
recipe, C ashhart HIP infrastructure and its Strix anchor, E ds4, Kyojin, ExLlamaV3 at both revisions in use (with the
EXL3 format-file comparison), upstream PR #824, the user's component kit (tensorfold-strix) and the AMD Triton wheel.
"""
import glob, hashlib, json, os, re, subprocess, sys, time, zipfile

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from provenance import record, sha256_file as sha256  # noqa: E402

H = os.path.expanduser("~")
SRC = f"{H}/tfp/src"
TF = f"{SRC}/TensorFold"
M = f"{H}/tfp/manifest"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
out = sys.argv[1]
NOFETCH = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}

def git(repo, *a):
    p = subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, env=NOFETCH)
    return p.stdout.strip() if p.returncode == 0 else None

def lic(repo, rev, paths):
    """Licence/notice files at rev: git blob id always (trees are local even in a blobless clone), SHA-256 of the
    content when the blob is local."""
    r = {}
    for path in paths:
        blob = git(repo, "rev-parse", f"{rev}:{path}")
        if not blob:
            continue
        p = subprocess.run(["git", "-C", repo, "cat-file", "blob", blob], capture_output=True, env=NOFETCH)
        r[path] = {"git_blob": blob, "sha256": hashlib.sha256(p.stdout).hexdigest() if p.returncode == 0 else None}
    return r

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

L = {"schema": "tf-port-lock/2", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
     "generator": record(__file__), "command": " ".join([os.path.realpath(__file__)] + sys.argv[1:]),
     "container_images": "none used (no Docker in this port's preparation path)"}

# ---- sources
PIN = {
    "base": "3ff3b2ac1d71a789cb930068c46d9abed025eefa", "tp4": "59402086381a42c74d53053443224eec2790b4c5",
    "kernels": "222a9c0b6ecaa9fd3c5dbded65a668524c9510ff", "later_fix": "9f820999f2a7ee0ffee3b6e4389ba22338bcf4a3",
    "recipe": "9fa230d3eb119f588e690bf9b87ce41ed902db60", "ashhart": "d46d17a784758b8a8b66badde1f728f36bdc9541",
    "ashhart_strix_anchor": "833bbfcba9109bb38c7af7303f80da0ca12b213c", "ds4": "f46631e483e19d13f23d3c4bced7ade0d086a3ad",
    "baseline_tree": "1beea23831603b98242311979137e91da4e92f25",
}
def src(name, url, repo, commit, role, license_paths, branch=None, licence=None):
    found = git(repo, "rev-parse", f"{commit}^{{commit}}")
    return {"name": name, "url": url, "branch": branch, "commit": commit, "found": found == commit, "tree": git(repo, "rev-parse", f"{commit}^{{tree}}"),
            "committed": git(repo, "show", "-s", "--format=%cI", commit), "role": role, "licence": licence, "local_repo": repo,
            "license_files": lic(repo, commit, license_paths)}

tf_lic = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "LICENSES/Apache-2.0.txt", "LICENSES/MIT.txt", "LICENSES/CUTLASS-BSD-3-Clause.txt",
          "LICENSES/PyTorch-BSD-3-Clause.txt", "LICENSES/Pillow-MIT-CMU.txt", "LICENSES/MiaAI-Lab-MIT.txt", "tools/dsv41-zig/ATTRIBUTION.md"]
ash_lic = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]
PIN.update(kyojin="30b0e3ab91cf25b4ccb5b6d044ef4946907c6259", exl3_wexl3="a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a",
           exl3_wref="4c84273155359ebb305accc39f74ab683eaaadf5", pr824="98072f763b86eccbb6fd7e527a525232d5b74b5e",
           kit="def87879ce82f7acaf8d8b441c7429776c87a221")
EXL3 = f"{H}/tfp/exl3/src/exllamav3"
L["sources"] = [
    src("A base", "https://github.com/bertholomus/TensorFold", TF, PIN["base"], "exact creator two-node 1017 engine base", tf_lic, "deepseek-v41-zig", "Apache-2.0"),
    src("B four-node reference", "https://github.com/bertholomus/TensorFold", TF, PIN["tp4"], "TP4 comparison only", tf_lic, "deepseek-v41-zig-tp4", "Apache-2.0"),
    src("D CUDA/Triton kernel source", "https://github.com/bertholomus/TensorFold", TF, PIN["kernels"], "historical Python/CUDA family: kernels.py, markov.py, cuda/exl3", ["LICENSE", "NOTICE"], "deepseek-v41-tp2", "Apache-2.0"),
    src("later fix 9f820999", "https://github.com/bertholomus/TensorFold", TF, PIN["later_fix"], "child of A (2026-10-19): kept-prompt extent fix; on strix-port as 500814a (cherry-pick -x)", tf_lic, "deepseek-v41-zig", "Apache-2.0"),
    src("B2 recipe", "https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark", f"{SRC}/refs/recipe", PIN["recipe"], "TP2 patches, chat template, creator results (patches Apache-2.0 per CREDITS.md)", ["LICENSE", "CREDITS.md"], "main", "MIT"),
    src("C HIP infrastructure", "https://github.com/ashhart/TensorFold", f"{SRC}/refs/ashhart-TensorFold", PIN["ashhart"], "zig/src/hip runtime, graph_split, gfx1151 receipt", ash_lic, "main", "Apache-2.0"),
    src("C Strix qualification anchor", "https://github.com/ashhart/TensorFold", f"{SRC}/refs/ashhart-TensorFold", PIN["ashhart_strix_anchor"], "tested inference source of receipt gfx1151-qwen-833bbfcb (blobless clone: licence blobs by id)", ash_lic, "tag v1.0.12", "Apache-2.0"),
    src("E AMD transport reference", "https://github.com/kyuz0/ds4", f"{SRC}/refs/kyuz0-ds4", PIN["ds4"], "host-staged ROCm TP reference; reference only", ash_lic, "main-gfx1151", "MIT"),
    src("Kyojin", "https://github.com/Yamz-Labs/kyojin", f"{H}/tfp/exl3/src/kyojin", PIN["kyojin"],
        "gfx1151 EXL3 HIP kernels (FP16 WMMA), an ExLlamaV3 fork: starting point of the port's EXL3 kernels (W-exl3 hip/vendor/exl3_dec_dev.cuh); clone owned by w-exl3", ["LICENSE"], "main", "MIT"),
    src("ExLlamaV3 v1.6.3 (W-exl3)", "https://github.com/turboderp-org/exllamav3", EXL3, PIN["exl3_wexl3"],
        "second CPU decoder of W-exl3's G-a oracle cross-check (oracle/crosscheck.py)", ["LICENSE"], "master (tag v1.6.3)", "MIT"),
    src("ExLlamaV3 v1.6.2 (w-ref)", "https://github.com/turboderp-org/exllamav3", EXL3, PIN["exl3_wref"],
        "EXL3 format reference of w-ref's independent CPU decoder (node0 ~/tfp/ref/exl3ref/spec.py); AUTHORITATIVE format reference", ["LICENSE"], "tag v1.6.2", "MIT"),
    src("upstream TensorFold PR #824", "https://github.com/ashhart/TensorFold/pull/824 (head: akol1/TensorFold)", f"{SRC}/build-pr824", PIN["pr824"],
        "Zig Engram token-map builder: the fifth, code-independent token-map cross-check (assets-xcheck/pr824-xcheck.json); not wired into the build", ash_lic, "refs/pull/824/head", "Apache-2.0"),
    src("component kit tensorfold-strix", "none: supplied by the user as a git bundle (~/tfp/kit/pkg/strix_handoff/tensorfold-strix.bundle)", f"{H}/tfp/kit/tensorfold-strix", PIN["kit"],
        "the component kit: input tools used by prepare_inputs.sh/verify_inputs.sh (pinned by kit_pin.sh), HIP components, node qualification", tf_lic, "port/strix-halo", "Apache-2.0"),
]
# The two ExLlamaV3 revisions (review M3): the files that define the EXL3 format, compared by git blob id
EXL3_FORMAT_FILES = ["exllamav3/exllamav3_ext/quant/codebook.cuh", "exllamav3/exllamav3_ext/quant/pack.cu", "exllamav3/exllamav3_ext/quant/exl3_dq.cuh",
                     "exllamav3/exllamav3_ext/quant/reconstruct.cu", "exllamav3/exllamav3_ext/hadamard.cpp", "exllamav3/exllamav3_ext/hadamard.h",
                     "exllamav3/modules/quant/exl3_lib/quantize.py", "exllamav3/modules/quant/exl3.py", "exllamav3/util/hadamard.py"]
fmt = {f: {"v1.6.2": git(EXL3, "rev-parse", f"{PIN['exl3_wref']}:{f}"), "v1.6.3": git(EXL3, "rev-parse", f"{PIN['exl3_wexl3']}:{f}")} for f in EXL3_FORMAT_FILES}
changed = (git(EXL3, "diff", "--name-only", PIN["exl3_wref"], PIN["exl3_wexl3"]) or "").split()
scope = [p for p in changed if p.startswith(("exllamav3/exllamav3_ext/quant/", "exllamav3/modules/quant/", "exllamav3/util/hadamard", "exllamav3/exllamav3_ext/hadamard"))]
def ssh_out(host, cmd):
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, cmd], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None
L["exllamav3_revisions"] = {
    "authoritative": {"revision": "v1.6.2", "commit": PIN["exl3_wref"], "why": "w-ref's independent decoder spec cites it file by file"},
    "v1.6.3_descends_from_v1.6.2": subprocess.run(["git", "-C", EXL3, "merge-base", "--is-ancestor", PIN["exl3_wref"], PIN["exl3_wexl3"]], env=NOFETCH).returncode == 0,
    "commits_between": int(git(EXL3, "rev-list", "--count", f"{PIN['exl3_wref']}..{PIN['exl3_wexl3']}") or -1),
    "format_files": fmt, "format_files_identical": all(v["v1.6.2"] and v["v1.6.2"] == v["v1.6.3"] for v in fmt.values()),
    "changed_files_in_quant_or_hadamard": scope,
    "changed_files_note": "only the quantizer (quantize.cu, quantize_tiles_*.cuh: the encoder's tile-search launch rules and LDS layout on ROCm) changed; decoding, packing, codebook, tile placement and Hadamard are blob-identical",
    "w_ref_checkout_node0": {"path": "node0:~/tfp/ref/src/exllamav3-v1.6.2", "head": ssh_out("node0", "git -C ~/tfp/ref/src/exllamav3-v1.6.2 rev-parse HEAD"),
                             "uncommitted_changes": ssh_out("node0", "git -C ~/tfp/ref/src/exllamav3-v1.6.2 status --porcelain | wc -l")},
    "w_exl3_checkout_node1": {"path": EXL3, "head": git(EXL3, "rev-parse", "HEAD"), "uncommitted_changes": len((git(EXL3, "status", "--porcelain") or "").splitlines())},
}
# The user's component kit (review M3): bundle, package id, checkouts on both boxes
KIT = f"{H}/tfp/kit/tensorfold-strix"
PKG = f"{H}/tfp/kit/pkg/strix_handoff"
pkg = load(f"{PKG}/PACKAGE_ID.json") or {}
L["component_kit"] = {
    "package_id": {k: pkg.get(k) for k in ("schema", "port_commit", "port_tree", "patched_tree", "base_commit", "branch", "created_at_utc", "source_status", "hardware_qualification")},
    "bundle": {"path": f"{PKG}/tensorfold-strix.bundle", "sha256": sha256(f"{PKG}/tensorfold-strix.bundle"), "expected_sha256": (pkg.get("bundle") or {}).get("sha256"),
               "heads": (git(KIT, "bundle", "list-heads", f"{PKG}/tensorfold-strix.bundle") or "").splitlines()},
    "package_id_sha256": sha256(f"{PKG}/PACKAGE_ID.json"),
    "checkouts": {"node1": {"path": KIT, "head": git(KIT, "rev-parse", "HEAD"), "clean": git(KIT, "status", "--porcelain") == ""},
                  "node0": {"path": "node0:~/tfp/kit/tensorfold-strix", "head": ssh_out("node0", "git -C ~/tfp/kit/tensorfold-strix rev-parse HEAD"),
                            "clean": ssh_out("node0", "git -C ~/tfp/kit/tensorfold-strix status --porcelain | wc -l") == "0"}},
    "licence": "Apache-2.0 (LICENSE); third-party notices in NOTICE, THIRD_PARTY_NOTICES.md and LICENSES/ (license_files in its sources entry)",
}
L["component_kit"]["bundle"]["match"] = L["component_kit"]["bundle"]["sha256"] == L["component_kit"]["bundle"]["expected_sha256"]
# The AMD Triton build w-tri compiled the 126 code objects with (review M3): wheel and licence by content
def wheel(path, expected):
    r = {"file": os.path.basename(path), "path": path, "sha256": sha256(path) if os.path.exists(path) else None, "expected_sha256": expected}
    r["match"] = r["sha256"] == expected
    if r["sha256"]:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if ".dist-info/" in n and os.path.basename(n).upper().startswith(("LICENSE", "COPYING")) and not n.endswith("/"):
                    r.setdefault("license_files", {})[n] = hashlib.sha256(z.read(n)).hexdigest()
                if n.endswith(".dist-info/METADATA"):
                    r["license_metadata"] = [l for l in z.read(n).decode().splitlines() if l.startswith(("License:", "License-Expression:"))]
    return r
tri_readme = subprocess.run(["git", "-C", REPO, "show", "strix-port:tools/dsv41-hip/tri/README.md"], capture_output=True, text=True).stdout
want_tri = dict(re.findall(r"`(triton|ml_dtypes)==[^`]+`.*?sha256 `([0-9a-f]{64})`", tri_readme, re.S))
L["triton_amd"] = {
    "role": "Triton for gfx1151: compiled w-tri's 204 recorded variants to 126 code objects (tools/dsv41-hip/tri, kit/gfx1151/aot.json)",
    "index": "https://rocm.nightlies.amd.com/whl-multi-arch/", "expected_from": "strix-port:tools/dsv41-hip/tri/README.md",
    "triton": wheel(f"{H}/tfp/tri/wheels/triton-3.8.0+git675c5987.rocm10.1.0a20260822-cp313-cp313-linux_x86_64.whl", want_tri.get("triton")),
    "ml_dtypes": wheel(f"{H}/tfp/tri/wheels/ml_dtypes-0.5.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", want_tri.get("ml_dtypes")),
}
L["sources_not_fetched"] = [
    {"name": "tp2-2xgb10 kit/recipe", "url": "https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10", "tag": "v0.6.1", "commit": "1b5ebbee3cc8b63b8b875e46a314135eb6d669db",
     "note": "kit used by the TP2 run (rope.json and engram.json hashes live here); tag dereferenced with gh api 2026-10-19; main is ahead (abe07c7d at that time)"},
]

# ---- patches and the reconstructed baseline
pdir = f"{SRC}/refs/recipe/runs/2026-10-18-1017-tensorfold-tp2/patches"
want = {"0001-feat-dsv41-each-node-s-weight-file-and-token-map-fro.patch": "e51f24bca9b3e164b44cbce54ef22fbd0143af465c3059acba12edb62beadef8",
        "0002-server-opt-in-random-sampling-seed-per-request-and-t.patch": "4824db1d9db2d2b7cbd464a1cf42b6475bbb7c75d162136af6b853800fc4aa83",
        "0003-dsv41-tf-dsv41-rank-cache-and-tf-dsv41-check-on-this.patch": "0451ff82163d7098a30526a339e158479593cb0e27f4adc4c83b7296b668c762"}
commits = (git(TF, "rev-list", "--reverse", f"{PIN['base']}..strix-port-base") or "").split()
L["patches"] = []
for i, (name, h) in enumerate(sorted(want.items())):
    got = sha256(f"{pdir}/{name}")
    c = commits[i] if i < len(commits) else None
    L["patches"].append({"order": i + 1, "file": f"runs/2026-10-18-1017-tensorfold-tp2/patches/{name}", "sha256": got, "expected_sha256": h, "match": got == h,
                         "applied_commit": c, "tree_after": git(TF, "rev-parse", f"{c}^{{tree}}") if c else None,
                         "author": git(TF, "show", "-s", "--format=%an <%ae>", c) if c else None})
head_tree = git(TF, "rev-parse", "strix-port-base^{tree}")
L["baseline"] = {"repo": TF, "branch": "strix-port-base", "tag": "baseline-1beea238", "head": git(TF, "rev-parse", "strix-port-base"), "tree": head_tree,
                 "expected_tree": PIN["baseline_tree"], "tree_match": head_tree == PIN["baseline_tree"],
                 "method": "git am --keep-cr --committer-date-is-author-date of 0001, 0002, 0003 on 3ff3b2ac, then a message-only amend adding provenance trailers (tree unchanged)",
                 "worktree_clean": git(TF, "status", "--porcelain") == ""}
cand = git(TF, "rev-parse", "candidate/base+9f82099")
L["candidates"] = [{"branch": "candidate/base+9f82099", "head": cand, "tree": git(TF, "rev-parse", "candidate/base+9f82099^{tree}"),
                    "content": "cherry-pick -x of 9f820999 onto the baseline (applies cleanly; git merge-tree gave the same tree)",
                    "decision": "taken by the integration owner: on strix-port as 500814a",
                    "on_strix_port": git(TF, "merge-base", "--is-ancestor", "500814a", "strix-port") == ""}]
L["chat_template"] = {"path": f"{SRC}/refs/recipe/runs/2026-10-17-1017-tensorfold/launch/chat_template.jinja", "recipe_commit": PIN["recipe"],
                      "sha256": sha256(f"{SRC}/refs/recipe/runs/2026-10-17-1017-tensorfold/launch/chat_template.jinja"),
                      "expected_sha256": "c282975e81949382a878e06060d1ef3e6e7b22628f9b51ee2188b2dfbcc7e217"}
L["chat_template"]["match"] = L["chat_template"]["sha256"] == L["chat_template"]["expected_sha256"]

# ---- toolchain (from both boxes' preflight JSONs, plus Zig and the built CPU tools)
def ssh_json_sha(host, path):
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, "cat", path], capture_output=True)
    try:
        return (json.loads(p.stdout), hashlib.sha256(p.stdout).hexdigest()) if p.returncode == 0 else (None, None)
    except json.JSONDecodeError:
        return None, None
pf = load(f"{M}/preflight/preflight-node1.json") or {}
pf1, pf1_sha = ssh_json_sha("node0", "~/tfp/manifest/preflight/preflight-node0.json")
def pf_summary(r, path, digest):
    if not r:
        return {"path": path, "status": "missing"}
    g = r.get("gpu", {})
    return {"path": path, "sha256": digest, "schema": r.get("schema"), "generated_utc": r.get("generated_utc"), "provenance": r.get("provenance"),
            "isa": g.get("isa"), "rocminfo": (g.get("rocminfo") or {}).get("status"), "hip_probe": (g.get("hip_probe") or {}).get("status"),
            "hip_mem_get_info": g.get("hip_mem_get_info"), "hsa_override_gfx_version_set": g.get("hsa_override_gfx_version_set"),
            "odirect": [o.get("status") for o in (r.get("disk", {}).get("odirect_probe") or []) if isinstance(o, dict)],
            "zig": [z.get("version") for z in (r.get("toolchain", {}).get("zig_all") or [])], "unavailable": r.get("unavailable"),
            "drift_vs_baseline_changed": (r.get("drift_vs_baseline") or {}).get("changed"),
            "packages_installed_set_sha256": (r.get("packages") or {}).get("installed_set_sha256")}
L["preflight"] = {"node1": pf_summary(pf, f"node1:{M}/preflight/preflight-node1.json", sha256(f"{M}/preflight/preflight-node1.json") if pf else None),
                  "node0": pf_summary(pf1, "node0:~/tfp/manifest/preflight/preflight-node0.json", pf1_sha),
                  "same_gpu_stack": bool(pf and pf1) and (pf.get("packages") or {}).get("gpu_kernel_firmware") == (pf1.get("packages") or {}).get("gpu_kernel_firmware")}
tc = pf.get("toolchain", {})
zig_dl = f"{H}/opt/dl/zig"
L["toolchain"] = {
    "zig": {"version": subprocess.run([f"{H}/opt/zig/zig", "version"], capture_output=True, text=True).stdout.strip(), "path": os.path.realpath(f"{H}/opt/zig/zig"),
            "sha256": sha256(os.path.realpath(f"{H}/opt/zig/zig")), "copies_found_by_preflight": {"node1": tc.get("zig_all"), "node0": ((pf1 or {}).get("toolchain") or {}).get("zig_all")},
            "tarball": "https://ziglang.org/download/0.17.0/zig-x86_64-linux-0.17.0.tar.xz", "tarball_sha256": sha256(f"{zig_dl}/zig-x86_64-linux-0.17.0.tar.xz"),
            "index_json_shasum": (load(f"{zig_dl}/index.json") or {}).get("0.17.0", {}).get("x86_64-linux", {}).get("shasum"),
            "minisign_public_key": "RWSGOq2NVecA2UPNdBUZykf1CCb147pkmdtYxgb3Ti+JO/wCYvhbAb/U", "signature_verified": "minisign ED (prehashed) signature and trusted comment checked with minisign_verify.py (python cryptography Ed25519)"},
    "hip": (tc.get("hipconfig_version") or {}).get("out"), "rocm": tc.get("rocm_version_files"), "hipcc": (tc.get("hipcc_version") or {}).get("out"),
    "device_lib": tc.get("device_lib_bitcode_dirs"), "hip_include": tc.get("hip_include_dirs"), "binaries_sha256": tc.get("binaries_sha256"),
    "hip_stacks": tc.get("hip_stacks"),
    "libraries": {k: v[:3] for k, v in (tc.get("libraries") or {}).items()},
    "packages_of_interest": {k: v for k, v in ((pf.get("packages") or {}).get("gpu_kernel_firmware") or tc.get("packages") or {}).items()
                             if any(s in k for s in ("blas", "ccl", "core", "runtime", "llvm", "hip", "amdgpu", "firmware", "linux-image"))},
    "apt_holds": (pf.get("packages") or {}).get("apt_holds"),
    "python3": (tc.get("python3") or {}).get("out"), "python_modules": (tc.get("python_modules") or {}).get("out"),
    "gcc": (tc.get("gcc") or {}).get("out"), "git": (tc.get("git") or {}).get("out"),
    "kernel": pf.get("kernel", {}).get("release"), "distro": pf.get("distro", {}).get("PRETTY_NAME"), "gpu_isa": pf.get("gpu", {}).get("isa"),
}
L["built_cpu_tools"] = {"command": "cd ~/tfp/src/TensorFold && ~/opt/zig/zig build install -j8 --summary all   (no -Dnvcc: host-only runtime, empty kernel images)",
                        "log": f"{SRC}/logs/zig-build-install-cpu.log", "target": "x86_64-linux native CPU (znver5 features), -Doptimize default fast",
                        "binaries": {os.path.basename(p): sha256(p) for p in sorted(glob.glob(f"{TF}/zig-out/bin/*"))}}

# ---- checkpoints
def ckpt(name, repo, rev, tree_json, verify_json, local):
    v = load(verify_json) or {}
    files = {f["path"]: {k: f.get(k) for k in ("size", "sha256", "expected_sha256", "git_blob_sha1", "expected_git_oid", "status")} for f in v.get("files", [])}
    return {"name": name, "repo": f"https://huggingface.co/{repo}", "revision": rev, "local_dir": local, "tree_metadata": tree_json, "verify_result": verify_json,
            "files_verified": sum(1 for f in files.values() if f["status"] == "ok"), "files_listed": len(files), "files": files}
L["checkpoints"] = [
    ckpt("LibertAI DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw (MIT)", "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw", "c178a01284fc5c0e859d8668cffaac81ca8f4042",
         f"{M}/hf/libertai-c178a012.tree.json", f"{M}/hashes/libertai-c178a012.verify.json", f"{H}/models/tf-port/libertai-2.9bpw"),
    ckpt("DeepSeek-V4.1-Flash-1017 official, selected files (MIT)", "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170",
         f"{M}/hf/ds1017-1bae0e10.tree.json", f"{M}/hashes/ds1017-1bae0e10.verify.json", f"{H}/models/tf-port/ds1017-1bae0e10"),
]
L["engram"] = {"source": "official 1017 @1bae0e10 shards model-00047-of-00048 (layer 1) and model-00048-of-00048 (layer 14): engram.embed.weight F8_E4M3, embed.scale F8_E8M0, q/k_weight, wkv",
               "expected_sha256": {x["path"]: x["lfs"]["oid"] for x in (load(f"{M}/hf/ds1017-1bae0e10.tree.json") or [])
                                   if x["path"] in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")},
               "headers": [f"{M}/hf/ds1017-shard47.header.json", f"{M}/hf/ds1017-shard48.header.json"]}
for c in L["checkpoints"]:
    for k in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json", "exl3_k_map.json", "quantization_config.json"):
        f = c["files"].get(k)
        if f:
            c.setdefault("key_file_sha256", {})[k] = f["sha256"]

# ---- generated inputs
rc = {}
for p in sorted(glob.glob(f"{H}/models/tf-port/rank-cache-tp2/rank*of2-*.bin")):
    rc[os.path.basename(p)] = {"bytes": os.path.getsize(p)}
L["generated"] = {
    "rank_caches": {"dir": f"{H}/models/tf-port/rank-cache-tp2", "files": rc, "format": "rank_cache.zig magic TFDSRK01: data, JSON index, u64 index offset, magic; name carries the index sha256 (20 hex)",
                    "generator": "tf-dsv41-rank-cache MODEL_DIR OUT_DIR R 2 (dspark included)", "index_only_outputs": [f"{M}/ckpt/rank0of2.index.json", f"{M}/ckpt/rank1of2.index.json"],
                    "full_checks": [f"{M}/logs/check-r0of2-bytes1.log", f"{M}/logs/check-r1of2-bytes1.log"]},
    "token_map": "in the prepared bundle (dsv41_token_map.json, de40bcdd...); see cpu_assets_crosscheck",
    "rope_tables": "in the prepared bundle (kit/rope-*.f32, 65,536 rows, reference_hashes_verified via rope_reference.py); full 1,114,112-row tables equal the creator kit's",
    "engram_json": "in the prepared bundle (kit/engram.json; multipliers/primes/offsets equal the creator kit's; 6/6 recorded cases reproduce)",
}
L["config"] = {"tp_world": 2, "streams": 1, "context": 8192, "text_only": True, "rdma": False, "note": "bring-up target from PHASE0/handoff §5; not yet a runnable configuration"}

# ---- CPU input preparation (P0/P2 prep): kit tools, pinned venv, cross-checks, the prepared bundle, node0 copies
KIT = f"{H}/tfp/kit/tensorfold-strix"
X = f"{M}/assets-xcheck"
L["input_tools"] = {
    "kit_repo": KIT, "kit_commit": git(KIT, "rev-parse", "HEAD"), "kit_tree": git(KIT, "rev-parse", "HEAD^{tree}"),
    "kit_source_lock_sha256": sha256(f"{KIT}/tools/dsv41-hip/source_lock.json"),
    "kit_file_manifests_vs_hf_trees": "assets/checkpoint_files.json 66/66 and original_files.json 83/83 equal to the HF trees fetched by W-src",
    "kit_pin": "prepare_inputs.sh and verify_inputs.sh refuse input tools other than a clean checkout at the kit commit (kit_pin.sh)",
    "wrappers": {"repo": REPO, "commit": git(REPO, "rev-parse", "HEAD"),
                 "files": {f: sha256(f"{REPO}/{f}") for f in ("tools/dsv41-hip/prepare_inputs.sh", "tools/dsv41-hip/verify_inputs.sh", "tools/dsv41-hip/verify_inputs.py",
                                                              "tools/dsv41-hip/rope_reference.py", "tools/dsv41-hip/kit_pin.sh") if os.path.exists(f"{REPO}/{f}")}},
    "repo_assets": {os.path.relpath(f, REPO): sha256(f) for f in sorted(glob.glob(f"{REPO}/tools/dsv41-hip/assets/*"))},
    "venv": {"path": f"{H}/venvs/tfp-inputs (same on node0: ~/venvs/tfp-inputs)", "python": "3.13.5",
             "pinned": {"numpy": "2.3.5", "tokenizers": "0.22.1", "torch": "2.9.0+cpu"},
             "freeze": f"{M}/venv-tfp-inputs.freeze.txt", "freeze_sha256": sha256(f"{M}/venv-tfp-inputs.freeze.txt"),
             "require_hashes_lock": ({"path": "tools/dsv41-hip/requirements-inputs.lock.txt", "sha256": sha256(f"{REPO}/tools/dsv41-hip/requirements-inputs.lock.txt")}
                                     if os.path.exists(f"{REPO}/tools/dsv41-hip/requirements-inputs.lock.txt") else None)},
    "kit_input_tests": "19/19 pass on node1 (python -m unittest discover -s tools/dsv41-hip/tests -p 'test_inputs*.py')",
}
L["creator_kit_v0.6.1"] = {"repo": "bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10", "commit": "1b5ebbee3cc8b63b8b875e46a314135eb6d669db",
                           "files": {n: sha256(f"{X}/creator-kit-v0.6.1/{n}") for n in ("rope.json", "engram.json", "MANIFEST", "README.md")}}
L["cpu_assets_crosscheck"] = {
    "token_map": {"sha256": "de40bcdd71f24430ba82752ef385093be962adbf845a1c97db39161597d5dc28",
                  "agree": ["creator make_token_map.py (tree 1beea238) in the pinned venv", "kit prepare_inputs.py", "kit pkg cpu-assets-8192",
                            "creator kit v0.6.1 engram.json token_map_sha256", "upstream PR #824 Zig builder (akol1/TensorFold@98072f76): lookup identical; its .bin with the official config = b9a7cf9d..., the PR's golden"]},
    "engram_constants": load(f"{X}/engram-xcheck.json"), "engram_zig_pr824": load(f"{X}/pr824-xcheck.json"),
    "rope": load(f"{X}/rope-xcheck.json"),
    "kit_assets_8192_regenerated_byte_identical": True,
}
B = f"{H}/models/tf-port/inputs-tp2-c178a012"
bm = load(f"{B}/inputs_manifest.json")
L["prepared_bundle"] = {"path": B, "manifest_sha256": sha256(f"{B}/inputs_manifest.json") if bm else None,
                        "status": (bm or {}).get("status"), "scope": (bm or {}).get("scope"),
                        "rank_caches": [{k: r.get(k) for k in ("rank", "file", "bytes")} for r in (bm or {}).get("rank_caches", [])],
                        "rope_reference": f"{B}.rope-reference.json"}
L["node0_copies"] = {
    "libertai_pack_by_w_ref": {"path": "node0:~/tfp/ref/data/libertai-2.9bpw (43 shards)", "verification": "w-ref copy_pack.log: 43 shards OK against libertai-2.9bpw.sha256.json"},
    "libertai_model_dir": {"path": "node0:~/models/tf-port/libertai-2.9bpw", "content": "reflink (CoW) clones of w-ref's 43 shards + the 23 small files rsynced from node1; 210,819,043,909 B total = HF total"},
    "rank0_bundle_and_engram": {"path": "node0:~/models/tf-port/{inputs-tp2-c178a012 (rank1/ excluded), ds1017-1bae0e10}",
                                "verification": "verify_inputs.sh rank 0 on node0 (report beside the bundle)"},
}
L["rank_cache_checks"] = {"rank0_full": f"{M}/logs/check-r0of2-bytes1.log", "rank1_full_and_negative": f"{M}/logs/check-caches.log",
                          "negative_test": "byte of layers.14.ffn.experts.97.w2 svh (entry 50001) flipped in a reflinked copy: --bytes 1000 passed (2,216 compared), --bytes 1 failed with byte_mismatches 1"}
def ssh_json(host, path):
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, "cat", path], capture_output=True, text=True)
    try:
        return json.loads(p.stdout) if p.returncode == 0 else None
    except json.JSONDecodeError:
        return None
def verdict(r):
    """A verify_inputs report: schema 1 had pass flags only (a skipped check read pass: true); schema 2 has a status
    per check (pass/fail/skipped) and an overall pass/pass-with-skips/fail."""
    if not r:
        return None
    ch = r.get("checks", {})
    return {"schema": r.get("schema"), "status": r.get("status"), "started_utc": r.get("started_utc"), "finished_utc": r.get("finished_utc"),
            "provenance": r.get("provenance"), "manifest_sha256": r.get("manifest_sha256"),
            "checks": {k: v.get("status") or ("skipped" if v.get("skipped") else "pass" if v.get("pass") else "fail") for k, v in ch.items()},
            "skipped": [k for k, v in ch.items() if v.get("skipped") or v.get("status") == "skipped"],
            "rank_check": (ch.get("rank_cache_full_bytes") or {}).get("check")}
L["per_box_verification"] = {
    "node1_rank1": {"report": f"{B}.verify-rank1-node1.json", **(verdict(load(f"{B}.verify-rank1-node1.json")) or {})},
    "node0_rank0": {"report": "node0:~/models/tf-port/inputs-tp2-c178a012.verify-rank0-node0.json",
                    **(verdict(ssh_json("node0", "~/models/tf-port/inputs-tp2-c178a012.verify-rank0-node0.json")) or {})},
}
L["input_tools"]["venv"]["wheels"] = {"node1": load(f"{M}/wheels-node1.json"), "node0": ssh_json("node0", "~/tfp/manifest/wheels-node0.json"),
                                      "method": "pip download of the three pins, SHA-256 vs the index's published digest, every installed file vs the wheel's RECORD hash (wheel_lock.py)"}
def host_tests(name):
    try:
        t = open(f"{M}/logs/host-tests-{name}.log").read()
    except OSError:
        return None
    import re as _re
    m = _re.search(r"Build Summary: (\d+)/(\d+) steps succeeded; (\d+)/(\d+) tests passed", t)
    rc = _re.search(r"rc=(\d+)", t)
    return {"log": f"{M}/logs/host-tests-{name}.log", "summary": m.group(0) if m else None, "rc": int(rc.group(1)) if rc else None,
            "command": "zig build test -j8 --summary all (host-side unit tests, no GPU), confined: taskset 8-15,24-31, CPUQuota 800%, nice 19, quiet lock shared"}
L["host_tests"] = {"baseline_1beea238": host_tests("base"), "candidate_base+9f82099": host_tests("cand-9f82099")}
# Which committed script made each M0 output (review M1): outputs made before the scripts were committed are tied to
# them by content (the copy that ran = the file committed verbatim in 5235adb); later outputs carry their own record.
VERBATIM = "5235adb"
def made_by(output, script, ran_sha256, note=None, commit=VERBATIM):
    blob = subprocess.run(["git", "-C", REPO, "show", f"{commit}:tools/dsv41-hip/{script}"], capture_output=True)
    if ran_sha256 is None and blob.returncode == 0:  # ran straight from the worktree file that commit records
        ran_sha256 = hashlib.sha256(blob.stdout).hexdigest()
    return {"output": output, "script": f"tools/dsv41-hip/{script}", "ran_sha256": ran_sha256,
            "committed_verbatim_in": commit, "same_bytes": blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == ran_sha256,
            "current_sha256": sha256(f"{REPO}/tools/dsv41-hip/{script}"), "note": note}
L["evidence_generators"] = [
    made_by(f"{M}/preflight/m0-r1/preflight-halo{{1,2}}.json (review round 1)", "preflight.sh", "73dbb91ee4f083d351b3f52fea4fc22be847938b31f9c782991433d9b69034d5", "superseded by the schema-2 preflights, which record their own provenance"),
    made_by(f"{M}/lock.json (review round 1)", "lock_sources.py", "73096b2a3e7ea2c12c8c8ac5a07b9d8fa9cd3462dcbecf9fb848ff4d85f19100", "superseded by this file"),
    made_by(f"{M}/ckpt/checkpoint_{{tensors,linears}}.jsonl, rank{{0,1}}of2 ledger", "inventory_checkpoint.py", "3f937fb36646777484d1ba303361f926a86daf050fd40330b12c271219121326", "rerun with the committed script: the jsonl files are byte-identical, checkpoint_inventory.json records its generator"),
    made_by(f"{M}/hashes/{{libertai-c178a012,ds1017-1bae0e10}}.verify.json and node0 small-files checks", "m0/verify_hf.py", "268eb1fb2e5a534552b864902174af382c2109fb393d6a364e99927417906901", "the surviving copy is node0 ~/tfp/manifest/verify_hf.py"),
    made_by(f"{M}/inventory_cuda.json", "m0/inventory_cuda_src.py", "49bba16c25ab5e5c44ea2767832b2814104bfe9149f1bc9ba22370a7cba8fe4e", "W-src's inventory_cuda.py, renamed in the repository"),
    made_by(f"{M}/wheels-halo{{1,2}}.json (the three hash-pinned wheels)", "m0/wheel_lock.py", "038a20d9dbad6ffdea7228c2779c71153d7f41004688c2aaabdc2d77fb741d50", "its RECORD-selection bug (eff1d3e) cannot affect these three wheels: none ships a vendored dist-info"),
    made_by("Zig 0.17.0 tarball signature check (toolchain.zig.signature_verified)", "m0/minisign_verify.py", "e42079e70cc603a7e568f2df2457082e42b16314cf0de70d8647b07c5e008e17"),
    made_by(f"{M}/logs/check-caches.log (rank-1 full check, negative test)", "m0/check_caches_when_idle.sh", "6005e9ad0541b9909e115823f9360a774a6e1614a58fcd1aec31f630ef16dc77", "with m0/check_caches.sh and m0/check_caches_locked.sh"),
    made_by(f"{M}/logs/host-tests-{{base,cand-9f82099}}.log", "m0/host_tests.sh", "75141ff9c650dd7260a9c4eca3967bbaa00b2fda52436ac42a589ddadfc84912"),
    made_by(f"{M}/logs/dedupe-copy.log (node0 copies)", "m0/copy_to_node0.sh", "2da13fda2515c18b312280ca2f39b809ef0daf1b65c5249fb285b2b4e3912803", "with m0/dedupe_and_copy.sh"),
    made_by(f"{M}/wheels-all-node1.json, node0 {M.replace(H, '~')}/wheels-all-node0.json (all 24 wheels)", "m0/wheel_lock.py", None, "run from the worktree file committed in eff1d3e (copied to node0 for its run)", commit="eff1d3e"),
    made_by("tools/dsv41-hip/requirements-inputs.lock.txt and ~/tfp/manifest/venv-lock-node1.json", "m0/venv_lock.py", None, "run from the worktree file committed in eff1d3e", commit="eff1d3e"),
]
small = {k: ssh_json("node0", f"~/tfp/manifest/hashes/{k}.small-files.node0.verify.json") for k in ("ds1017-1bae0e10", "libertai-c178a012")}
L["node0_copies"]["small_files_sha"] = {k: v and {"n": v["n"], "bad": v["bad"]} for k, v in small.items()}
L["deployments"] = {"node0": ssh_json("node0", "~/tfp/deploy/current/DEPLOY.json") and
                    {k: ssh_json("node0", "~/tfp/deploy/current/DEPLOY.json")[k] for k in ("commit", "tree", "clean", "host", "created_utc", "target")}}
json.dump(L, open(out, "w"), indent=1)
bad = [s["name"] for s in L["sources"] if not s["found"]] + [p["file"] for p in L["patches"] if not p["match"]]
if not L["exllamav3_revisions"]["format_files_identical"]: bad.append("exllamav3 format files differ")
if not L["component_kit"]["bundle"]["match"]: bad.append("kit bundle hash")
if not (L["triton_amd"]["triton"]["match"] and L["triton_amd"]["ml_dtypes"]["match"]): bad.append("triton wheels")
print(out, "sha256", sha256(out), "sources", len(L["sources"]), "patch_match", all(p["match"] for p in L["patches"]), "tree_match", L["baseline"]["tree_match"],
      "template_match", L["chat_template"]["match"], "ckpt_verified", [(c["files_verified"], c["files_listed"]) for c in L["checkpoints"]],
      "preflight", {h: (v.get("hip_probe"), v.get("odirect")) for h, v in L["preflight"].items() if isinstance(v, dict)}, "problems", bad)

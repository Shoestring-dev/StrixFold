#!/usr/bin/env python3
"""run_manifest.py OUT.json --deploy DIR --conf CONF [--resolved FILE ...] [--rank-cmd R=FILE ...] [--prompt-ids FILE]
                   [--box NAME=STATE ...] [--note TEXT]

The complete identity of one qualification or benchmark run (the source audit's frozen deployment manifest), written
beside its results:
  deploy      DEPLOY.json (commit, tree, clean, source host) and the SHA-256 of its SHA256SUMS; every binary under
              native-hip/bin and native-hip/test re-hashed now (and checked against DEPLOY.json)
  kit         the AMD kit's manifest.json SHA-256, its schema/kit/ABI/arch/toolchain and source inputs, and every code
              object (HIP and AOT Triton) by SHA-256, size and origin
  model       the input bundle's inputs_manifest.json SHA-256 (must equal the configuration's BUNDLE_SHA256), the declared
              checkpoint and its revision, config/tokenizer/chat-template hashes, each rank's cache file (its content key),
              the Engram shards' hashes
  config      the configuration file's path, SHA-256 and text
  ranks       both ranks' resolved.json (argv, the whole environment, binary/kit/config hashes, per box)
  toolchain   per box: hipcc, the ROCm runtime/core package versions, kernel, Python; this box's Zig if present
  prompts     the request token ids file (SHA-256, count) when given
  box_state   per box: load, GPU clock, DPM level, power profile, temperatures at the start of the measured block
Standard library only; run on node1 (reaches node0 by ssh for its toolchain)."""
import argparse
import hashlib
import json
import os
import subprocess
import time


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, remote=None):
    if remote:
        cmd = ["ssh", "-n", remote, " ".join(cmd) if isinstance(cmd, list) else cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str))
    return r.stdout.strip()


TOOLCHAIN = ("hipcc --version 2>/dev/null | head -1; dpkg-query -W -f='${Package} ${Version}\\n' amdrocm-core10.0-gfx1151 "
             "amdrocm-runtime10.0 amdrocm-llvm10.0 2>/dev/null; uname -r; python3 --version; "
             "(ls ~/opt/zig/zig >/dev/null 2>&1 && ~/opt/zig/zig version) || true")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--deploy", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--resolved", action="append", default=[])
    ap.add_argument("--prompt-ids", default=None)
    ap.add_argument("--rank-cmd", action="append", default=[], help="RANK=FILE: a rank's command line (harness runs without resolved.json)")
    ap.add_argument("--box", action="append", default=[])
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    dep = json.load(open(os.path.join(a.deploy, "DEPLOY.json")))
    files = dep.get("files", {})
    bins = {}
    for sub in ("native-hip/bin", "native-hip/test"):
        d = os.path.join(a.deploy, sub)
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                h = sha(p)
                bins[f"{sub}/{n}"] = {"sha256": h, "matches_deploy_json": files.get(f"{sub}/{n}") == h}
    conf_text = open(a.conf).read()
    cv = {}
    for line in conf_text.splitlines():
        if "=" in line and line[:1].isupper():
            k, v = line.split("=", 1)
            cv[k] = v.split("#")[0].strip().strip("'")
    kit_dir = os.path.join(a.deploy, cv.get("KIT_SUBDIR", "native-hip/share/tensorfold/hip/gfx1151/dsv41"))
    km = json.load(open(os.path.join(kit_dir, "manifest.json")))
    kit = {"manifest_sha256": sha(os.path.join(kit_dir, "manifest.json")),
           **{k: km.get(k) for k in ("schema", "kit", "abi_version", "arch", "wave", "toolchain", "inputs")},
           "kernels": len(km.get("kernels", [])),
           "objects": [{k: o.get(k) for k in ("file", "sha256", "bytes", "origin", "owner")} for o in km.get("objects", [])]}
    kit["objects_verified"] = all(sha(os.path.join(kit_dir, o["file"])) == o["sha256"] for o in kit["objects"])
    bundle = cv.get("BUNDLE", "").replace("$HOME", os.path.expanduser("~"))
    im_path = os.path.join(bundle, "inputs_manifest.json")
    model = {}
    if os.path.exists(im_path):
        im = json.load(open(im_path))
        model = {"inputs_manifest_sha256": sha(im_path), "pinned": cv.get("BUNDLE_SHA256"),
                 "declared_checkpoint": {k: (im.get("declared_checkpoint") or {}).get(k) for k in ("repository", "revision", "codebook", "expected_tensors")},
                 "config_sha256": im.get("config_sha256"), "tokenizer_sha256": im.get("tokenizer_sha256"),
                 "files": {f["path"]: f["sha256"] for f in im.get("files", []) if f["path"].endswith((".json", ".jinja"))},
                 "rank_caches": [((rc.get("build") or {}).get("report") or {}).get("file") for rc in im.get("rank_caches", [])],
                 "engram_files": {f["path"]: f["sha256"] for f in im.get("engram_files", [])}}
        model["pinned_matches"] = model["inputs_manifest_sha256"] == model["pinned"]
    ranks = {}
    for p in a.resolved:
        try:
            d = json.load(open(p))
            ranks[f"rank{d.get('rank')}"] = d
        except (OSError, ValueError) as e:
            ranks[os.path.basename(p)] = {"error": str(e)}
    for rc in a.rank_cmd:
        r, p = rc.split("=", 1)
        ranks[f"rank{r}"] = {"command_file": p, "command": open(p).read().strip() if os.path.exists(p) else None}
    tool = {"node1": run(TOOLCHAIN).splitlines(), "node0": run(TOOLCHAIN, remote="node0").splitlines()}
    prompts = None
    if a.prompt_ids and os.path.exists(a.prompt_ids):
        ids = json.load(open(a.prompt_ids))
        prompts = {"file": a.prompt_ids, "sha256": sha(a.prompt_ids), "requests": len(ids)}
    doc = {"schema": "tf-port-run-manifest/1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": a.note,
           "deploy": {"dir": os.path.realpath(a.deploy), **{k: v for k, v in dep.items() if k != "files"},
                      "sha256sums_sha256": sha(os.path.join(a.deploy, "SHA256SUMS")), "binaries": bins},
           "kit": kit, "model": model, "config": {"path": a.conf, "sha256": sha(a.conf), "text": conf_text}, "ranks": ranks,
           "toolchain": tool, "prompts": prompts, "box_state": dict(b.split("=", 1) for b in a.box)}
    doc["complete"] = bool(bins) and all(b["matches_deploy_json"] for b in bins.values()) and kit["objects_verified"] \
        and model.get("pinned_matches", False) and len(ranks) == 2
    json.dump(doc, open(a.out, "w"), indent=1)
    print(json.dumps({"manifest": a.out, "complete": doc["complete"], "commit": dep.get("commit"), "kit": kit["manifest_sha256"][:16],
                      "objects": len(kit["objects"]), "binaries": len(bins), "ranks": sorted(ranks)}))


if __name__ == "__main__":
    main()

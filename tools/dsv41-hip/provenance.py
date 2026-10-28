#!/usr/bin/env python3
"""provenance.py [SCRIPT ...]: which script made an output, and from which commit.

record(path) returns the script's real path and SHA-256 and where it came from:
- in a git checkout: `git_head` (HEAD), `script_committed` (the file is byte-identical to HEAD's copy) and
  `git_tree_clean` (no change to tracked files);
- in a copy deployed by deploy_rank0.sh (no checkout, e.g. node0): the DEPLOY.json found in a parent directory, its
  source commit as `git_head`, and `script_committed` = the script's SHA-256 equals the one DEPLOY.json lists for it
  and the deployed tree was clean.
Every M0 generator (preflight.sh, lock_sources.py, inventory_checkpoint.py, ...) embeds this record in its output, so
an output names the exact script version that produced it. Run directly, it prints the records of its arguments.
"""
import hashlib
import json
import os
import subprocess
import sys


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git(cwd, *args):
    p = subprocess.run(["git", "-C", cwd, *args], capture_output=True)
    return p.stdout if p.returncode == 0 else None


def record(path):
    path = os.path.realpath(path)
    here = os.path.dirname(path)
    r = {"script": path, "script_sha256": sha256_file(path), "git_head": None, "script_committed": None,
         "git_tree_clean": None, "deploy": None}
    out = _git(here, "rev-parse", "HEAD", "--show-toplevel")
    if out:
        head, top = out.decode().split()
        blob = _git(top, "show", f"HEAD:{os.path.relpath(path, top)}")
        status = _git(top, "status", "--porcelain", "--untracked-files=no")
        r.update(git_head=head, script_committed=blob is not None and hashlib.sha256(blob).hexdigest() == r["script_sha256"],
                 git_tree_clean=status == b"")
        return r
    up = here
    for _ in range(5):
        manifest = os.path.join(up, "DEPLOY.json")
        if os.path.exists(manifest):
            dep = json.load(open(manifest))
            listed = dep.get("files", {}).get(os.path.relpath(path, up))
            r["deploy"] = {"manifest": manifest, "source_commit": dep.get("commit"), "source_tree_clean": dep.get("clean"),
                           "built_on": dep.get("host"), "listed_sha256_matches": listed == r["script_sha256"]}
            r.update(git_head=dep.get("commit"), git_tree_clean=dep.get("clean"),
                     script_committed=listed == r["script_sha256"] and dep.get("clean") is True)
            break
        up = os.path.dirname(up)
    return r


if __name__ == "__main__":
    print(json.dumps([record(p) for p in sys.argv[1:]], indent=1))

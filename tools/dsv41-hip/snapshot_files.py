"""Authenticate selected local files against immutable public HF blob metadata."""
from __future__ import annotations

import hashlib
from pathlib import Path

from input_common import InputError, checked_relative, read_json, require_hash

HERE = Path(__file__).resolve().parent


def load_snapshot(lock: dict, name: str) -> dict:
    spec = lock[name]
    entry = spec["file_manifest"]
    path = HERE / entry["path"]
    require_hash(path, entry["sha256"])
    manifest = read_json(path)
    if manifest.get("repository") != spec["repository"] or manifest.get("revision") != spec["revision"]:
        raise InputError(f"wrong locked file-manifest identity: {name}")
    return manifest


def verify_files(root: Path, manifest: dict, names: list[str]) -> list[dict]:
    files = {item["path"]: item for item in manifest["files"]}
    if not names or len(names) != len(set(names)):
        raise InputError("file authentication needs a nonempty unique list")
    results = []
    for name in names:
        if name not in files:
            raise InputError(f"file is not in pinned snapshot: {name}")
        expected = files[name]
        path = checked_relative(root, name)
        size = path.stat().st_size
        if size != expected["bytes"]:
            raise InputError(f"pinned snapshot size mismatch: {name}")
        sha256, blob = hashlib.sha256(), hashlib.sha1()
        blob.update(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as src:
            while block := src.read(4 << 20):
                sha256.update(block)
                blob.update(block)
        if "sha256" in expected:
            if sha256.hexdigest() != expected["sha256"]:
                raise InputError(f"pinned snapshot SHA-256 mismatch: {name}")
        elif expected.get("git_blob_sha1") != blob.hexdigest():
            raise InputError(f"pinned snapshot Git blob mismatch: {name}")
        results.append({"path": name, "bytes": size, "sha256": sha256.hexdigest(),
                        "verified_against": "hf_lfs_sha256" if "sha256" in expected else "hf_git_blob_sha1"})
    return results

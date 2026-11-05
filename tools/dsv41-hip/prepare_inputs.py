#!/usr/bin/env python3
"""Prepare DeepSeek CPU assets; optionally build and verify both native rank caches.

The default full-input mode requires the complete locked checkpoint, two real CPU
executables and enough disk space. --assets-only is explicitly a smaller scope.
Neither mode compiles GPU kernels or claims model/HIP numerical qualification.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from input_common import (InputError, atomic_bytes, atomic_json, checked_int,
                          cli_error, read_json, require_hash, sha256_file)
from inventory_checkpoint import inventory
from snapshot_files import load_snapshot, verify_files

HERE = Path(__file__).resolve().parent
ROPE_COMPAT = 1 << 19
ROPE_BLOCK = 1 << 16


def compressed_token_map(path: Path) -> tuple[list[int], int]:
    # Adapted from the pinned Apache-2.0 CPU helper; no CUDA model import.
    try:
        from tokenizers import Regex, Tokenizer, normalizers
    except ImportError as exc:
        raise InputError("CPU preparation requires tokenizers; see requirements-inputs.txt") from exc
    tok = Tokenizer.from_file(str(path))
    sentinel = "\ue000"
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    keys, lookup = {}, []
    for tid in range(tok.get_vocab_size(with_added_tokens=True)):
        text = tok.decode([tid], skip_special_tokens=False)
        key = tok.id_to_token(tid) if "\ufffd" in text else norm.normalize_str(text) or text
        if key is None:
            raise InputError(f"token id {tid} is not represented by tokenizer.json")
        if key not in keys:
            keys[key] = len(keys)
        lookup.append(keys[key])
    if not lookup:
        raise InputError("empty tokenizer")
    return lookup, len(keys)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    d, rounds = n - 1, 0
    while d % 2 == 0:
        d //= 2
        rounds += 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(rounds - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def engram_constants(raw: dict, lookup: list[int]) -> dict:
    try:
        import numpy as np
    except ImportError as exc:
        raise InputError("CPU Engram preparation requires numpy") from exc
    c = raw.get("text_config", raw)
    layers = c.get("engram_layer_ids")
    if not isinstance(layers, list) or not layers:
        raise InputError("missing Engram layer ids")
    layers = [checked_int(v, "engram_layer_ids") for v in layers]
    if len(set(layers)) != len(layers):
        raise InputError("duplicate Engram layer id")
    ngram = checked_int(c.get("engram_max_ngram_size"), "Engram ngram", 2)
    heads = checked_int(c.get("engram_n_heads"), "Engram heads", 1)
    vocab = checked_int(c.get("engram_vocab_size"), "Engram vocabulary", 2)
    cvocab = checked_int(c.get("engram_compressed_vocab_size"), "compressed vocabulary", 1)
    pad = checked_int(c.get("engram_pad_token_id"), "Engram pad id")
    if pad >= len(lookup) or any(v < 0 or v >= cvocab for v in lookup):
        raise InputError("invalid compressed token map or pad id")
    if set(lookup) != set(range(cvocab)):
        raise InputError("compressed token ids must cover the declared contiguous vocabulary")
    if ngram > 16 or heads > 64 or len(layers) > 48 or vocab > (1 << 50):
        raise InputError("Engram constants exceed bounded preparation limits")
    seen, primes, offsets, multipliers = set(), [], [], []
    bound = max(1, (((1 << 63) - 1) // cvocab) // 2)
    for layer in layers:
        row = []
        for _ in range(ngram - 1):
            cur = vocab - 1
            for _ in range(heads):
                cur += 1
                while cur in seen or not is_prime(cur):
                    cur += 1
                seen.add(cur)
                row.append(cur)
        off, acc = [], 0
        for p in row:
            off.append(acc)
            acc += p
        primes.append(row)
        offsets.append(off)
        rng = np.random.default_rng(10007 * layer)
        multipliers.append((rng.integers(0, bound, size=ngram, dtype=np.int64) * 2 + 1).tolist())
    expected_rows = c.get("engram_num_embeddings")
    if expected_rows is not None and [sum(p) for p in primes] != expected_rows:
        raise InputError("generated Engram table rows disagree with config")
    return {"layers": layers, "ngram": ngram, "heads": heads, "pad": lookup[pad],
            "primes": primes, "offsets": offsets, "multipliers": multipliers,
            "token_map_len": len(lookup), "compressed_vocab_size": cvocab,
            "cases": [], "generator": "standalone_cpu_pinned_engram_algorithm",
            "numpy_version": np.__version__}


def engram_hashes(constants: dict, lookup: list[int], tokens: list[int], start: int, n: int) -> list:
    """Integer reference used to create/check portable Engram fixtures."""
    if start < 0 or n < 0 or start + n > len(tokens):
        raise InputError("Engram request range is outside the supplied sequence")
    if any(t >= len(lookup) for t in tokens):
        raise InputError("token outside compressed token map")
    result = []
    for pos in range(start, start + n):
        values, blocked = [], False
        for shift in range(constants["ngram"]):
            token = tokens[pos - shift] if pos >= shift else tokens[0]
            blocked |= pos < shift or token < 0
            values.append(constants["pad"] if blocked else lookup[token])
        rows = []
        for l, mult in enumerate(constants["multipliers"]):
            rolling, row = values[0] * mult[0], []
            for order in range(1, constants["ngram"]):
                rolling ^= values[order] * mult[order]
                for head in range(constants["heads"]):
                    col = (order - 1) * constants["heads"] + head
                    row.append(rolling % constants["primes"][l][col] + constants["offsets"][l][col])
            rows.append(row)
        result.append(rows)
    return result


def rope_contract(raw: dict) -> dict:
    c = raw.get("text_config", raw)
    dim = checked_int(c.get("qk_rope_head_dim"), "rotary dimension", 2)
    rs = c.get("rope_scaling", {})
    result = {"dimension": dim, "plain_base": c.get("rope_theta"),
              "compressed_base": c.get("compress_rope_theta"),
              "original_length": rs.get("original_max_position_embeddings", 0),
              "factor": rs.get("factor", 1.0), "beta_fast": rs.get("beta_fast", 32.0),
              "beta_slow": rs.get("beta_slow", 1.0)}
    if dim % 2 or dim > 512:
        raise InputError("unsupported rotary dimension")
    checked_int(result["original_length"], "original rotary length")
    for key in ("plain_base", "compressed_base", "factor", "beta_fast", "beta_slow"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise InputError(f"invalid rotary parameter: {key}")
    if result["plain_base"] == 1 or result["compressed_base"] == 1:
        raise InputError("rotary base cannot be 1")
    return result


def rope_block(contract: dict, start: int, count: int, compressed: bool):
    import torch
    dim = contract["dimension"]
    base = contract["compressed_base"] if compressed else contract["plain_base"]
    orig = contract["original_length"] if compressed else 0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
    if orig > 0:
        corr = lambda rot: dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(corr(contract["beta_fast"])), 0), min(math.ceil(corr(contract["beta_slow"])), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / contract["factor"] * (1 - smooth) + freqs * smooth
    positions = torch.arange(start, start + count, dtype=torch.int64, device="cpu").to(torch.float32)
    angle = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angle), angle)


def generate_rope(raw: dict, out: Path, rows: int, reference: dict | None = None) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise InputError("RoPE generation requires CPU PyTorch; see requirements-inputs.txt") from exc
    contract = rope_contract(raw)
    checked_int(rows, "rotary rows", 1)
    if rows > (1 << 24):
        raise InputError("rotary rows exceed exact FP32 integer-position range")
    if reference and (reference.get("rows") != rows or reference.get("half") != contract["dimension"] // 2):
        raise InputError("RoPE reference shape differs from requested rows/dimension")
    out.mkdir(parents=True, exist_ok=True)
    result = {"rows": rows, "half": contract["dimension"] // 2, "contract": contract,
              "generator": "pinned_algorithm_cpu_torch", "torch_version": torch.__version__,
              "reference_hashes_verified": reference is not None}
    for kind in ("plain", "compressed"):
        names = [f"rope-{kind}-{part}.f32" for part in ("cos", "sin")]
        handles, temporary, digests = [], [], [hashlib.sha256(), hashlib.sha256()]
        try:
            for name in names:
                fd, path = tempfile.mkstemp(prefix=f".{name}.", dir=out)
                temporary.append(Path(path))
                handles.append(os.fdopen(fd, "wb"))
            # The source preserves one 2^19-position prefix call before 2^16 blocks.
            # The allocation is bounded independently of requested context length.
            pos = 0
            while pos < rows:
                compute = ROPE_COMPAT if pos == 0 else min(ROPE_BLOCK, rows - pos)
                values = rope_block(contract, pos, compute, kind == "compressed")
                used = min(compute, rows - pos)
                for i, tab in enumerate((values.real, values.imag)):
                    for chunk in range(0, used, ROPE_BLOCK):
                        data = tab[chunk:min(used, chunk + ROPE_BLOCK)].contiguous().numpy().astype("<f4", copy=False).tobytes()
                        handles[i].write(data)
                        digests[i].update(data)
                pos += used
                del values
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            for name, tmp, digest in zip(names, temporary, digests):
                got = digest.hexdigest()
                if reference and reference.get(name) != got:
                    raise InputError(f"RoPE reference hash mismatch for {name}; CPU/version parity remains unverified")
                os.replace(tmp, out / name)
                result[name] = got
        finally:
            for handle in handles:
                if not handle.closed:
                    handle.close()
            for tmp in temporary:
                tmp.unlink(missing_ok=True)
    atomic_json(out / "rope.json", result)
    return result


def run_cpu_tool(executable: Path, args: list[str], log: Path) -> dict:
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise InputError(f"missing executable CPU input tool: {executable}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    with log.open("wb") as dst:
        proc = subprocess.run([str(executable.resolve()), *args], stdout=dst,
                              stderr=subprocess.STDOUT, env=env)
    if proc.returncode:
        raise InputError(f"CPU input tool failed ({proc.returncode}); see {log}")
    report = None
    with log.open() as src:
        for line in src:
            if line.startswith("{"):
                try:
                    report = json.loads(line)
                except json.JSONDecodeError:
                    pass
    if not isinstance(report, dict):
        raise InputError(f"CPU input tool emitted no machine-readable report: {log}")
    return {"executable_sha256": sha256_file(executable), "returncode": proc.returncode,
            "log": log.name, "log_sha256": sha256_file(log), "report": report}


def prepare(model: Path, output: Path, rows: int, lock: dict, assets_only=False,
            rank_cache_bin: Path | None = None, check_bin: Path | None = None,
            rope_reference: Path | None = None, engram_dir: Path | None = None,
            allow_unlocked_fixture=False) -> dict:
    model, output = model.resolve(), output.resolve()
    if output == model or output.is_relative_to(model):
        raise InputError("output must be outside the original model snapshot")
    if output.exists() and any(output.iterdir()):
        raise InputError("output must be new or empty; existing bundles are never overwritten")
    raw = read_json(model / "config.json", 4 << 20)
    tokenizer = model / "tokenizer.json"
    if not tokenizer.is_file():
        raise InputError("missing tokenizer.json")
    q = raw.get("quantization_config", {})
    if raw.get("model_type") != "deepseek_v41" or q.get("quant_method") != "exl3" or q.get("codebook") != "mul1":
        raise InputError("input config must declare deepseek_v41 and EXL3 MUL1")
    c = raw.get("text_config", raw)
    if rows > c.get("max_position_embeddings", 0):
        raise InputError("requested rows exceed checkpoint context limit")
    if allow_unlocked_fixture and not assets_only:
        raise InputError("unlocked fixtures are only allowed in the explicit assets-only mode")
    authenticated_assets = [] if allow_unlocked_fixture else verify_files(
        model, load_snapshot(lock, "checkpoint"), ["config.json", "tokenizer.json"])
    required = inventory(model, lock, include_plans=True, verify_content=True) if not assets_only else None
    engram_files = []
    if not assets_only:
        if rank_cache_bin is None or check_bin is None:
            raise InputError("full preparation requires --rank-cache-bin and --check-bin")
        for executable in (rank_cache_bin, check_bin):
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise InputError(f"missing CPU executable: {executable}")
        if engram_dir is None:
            raise InputError("full preparation requires --engram-dir with the pinned original shards")
        original = load_snapshot(lock, "original")
        names = [f["path"] for f in original["files"] if (m := re.fullmatch(r"model-(\d+)-of-\d+\.safetensors", f["path"]))
                 and int(m[1]) in lock["original"]["engram_shard_numbers"]]
        if len(names) != 2:
            raise InputError("locked original metadata does not identify both Engram shards")
        engram_files = verify_files(engram_dir, original, names)
    lookup, compressed = compressed_token_map(tokenizer)
    if len(lookup) != c.get("vocab_size") or compressed != c.get("engram_compressed_vocab_size"):
        raise InputError("tokenizer vocabulary or compressed vocabulary disagrees with config")
    constants = engram_constants(raw, lookup)
    template = HERE / lock["chat_template"]["bundled_path"]
    require_hash(template, lock["chat_template"]["sha256"])
    contract = rope_contract(raw)
    needed = rows * (contract["dimension"] // 2) * 4 * 4 + tokenizer.stat().st_size + (16 << 20)
    if required:
        needed += sum(r["rank_file_bytes"] for r in required["ranks"])
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if needed > free:
        raise InputError(f"insufficient disk: need at least {needed} bytes, available {free}")
    output.mkdir(exist_ok=True)
    atomic_json(output / "PREPARING.json", {"scope": "assets_only" if assets_only else "cpu_rank_inputs"})
    config_hash = sha256_file(model / "config.json")
    try:
        bundle = output / "model"
        bundle.mkdir()
        shutil.copyfile(model / "config.json", bundle / "config.json")
        shutil.copyfile(tokenizer, bundle / "tokenizer.json")
        shutil.copyfile(template, bundle / "chat_template.jinja")
        map_bytes = json.dumps(lookup).encode()
        atomic_bytes(output / "dsv41_token_map.json", map_bytes)
        constants["token_map_sha256"] = hashlib.sha256(map_bytes).hexdigest()
        fixture_ids = [0, min(1, len(lookup) - 1), -1, min(2, len(lookup) - 1)]
        constants["cases"] = [{"name": "start_and_blocked_span", "ids": fixture_ids, "start": 0,
                               "n": len(fixture_ids), "rows": [v for row in engram_hashes(constants, lookup, fixture_ids, 0, len(fixture_ids)) for layer in row for v in layer]}]
        kit = output / "kit"
        atomic_json(kit / "engram.json", constants)
        ref = read_json(rope_reference) if rope_reference else None
        rope = generate_rope(raw, kit, rows, ref)
        rank_reports = []
        if required:
            atomic_json(output / "inventory.json", required)
            for rank in (0, 1):
                rank_dir = output / f"rank{rank}" / "rank-cache"
                rank_dir.mkdir(parents=True)
                build = run_cpu_tool(rank_cache_bin, [str(model), str(rank_dir), str(rank), "2"], output / f"rank{rank}-build.log")
                check = run_cpu_tool(check_bin, [str(model), str(rank_dir), str(rank), "2", "--bytes", "1"], output / f"rank{rank}-check.log")
                report, plan = check["report"], required["ranks"][rank]
                if (report.get("ok") is not True or any(report.get(k) != plan["entry_count"] for k in ("entries", "planned", "compared"))
                        or report.get("compared_bytes") != plan["data_bytes"] or plan["data_bytes"] <= 0
                        or any(report.get(k) != 0 for k in ("mismatches", "layout_mismatches", "byte_mismatches"))):
                    raise InputError(f"rank {rank} full byte validation did not cover its exact schema")
                files = list(rank_dir.glob(f"rank{rank}of2-*.bin"))
                if len(files) != 1 or files[0].stat().st_size != required["ranks"][rank]["rank_file_bytes"]:
                    raise InputError(f"rank {rank} produced an unexpected file count/size")
                rank_reports.append({"rank": rank, "file": str(files[0].relative_to(output)),
                                     "bytes": files[0].stat().st_size, "build": build, "byte_check": check})
        if sha256_file(model / "config.json") != config_hash:
            raise InputError("original config changed during preparation")
        files = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "PREPARING.json":
                # Hashing all rank files is intentional, streaming, and part of the output identity.
                files.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)})
        result = {"schema_version": 1, "status": "pass", "scope": "cpu_assets_only" if assets_only else "cpu_rank_inputs",
                  "config_sha256": config_hash, "tokenizer_sha256": sha256_file(tokenizer),
                  "declared_checkpoint": None if allow_unlocked_fixture else lock["checkpoint"],
                  "synthetic_fixture": allow_unlocked_fixture, "authenticated_assets": authenticated_assets,
                  "input_snapshot_content_authenticated": required is not None,
                  "engram_files": engram_files, "engram_directory": str(engram_dir.resolve()) if engram_dir else None,
                  "text_only": True, "vision_initialized": False, "rows": rows,
                  "compressed_vocab": compressed, "rope_reference_hashes_verified": rope["reference_hashes_verified"],
                  "rank_caches": rank_reports, "files": files,
                  "dependencies": {n: version(n) for n in ("numpy", "tokenizers", "torch")},
                  "python": sys.version, "hip_kernel_kit": "not_provided_by_cpu_preparation",
                  "full_model_numerical_qualification": "unrun", "hardware_qualification": "unrun"}
        atomic_json(output / "inputs_manifest.json", result)
        (output / "PREPARING.json").unlink()
        return result
    except Exception as exc:
        atomic_json(output / "FAILED.json", {"status": "fail", "error": str(exc),
                                           "retained_partial_output": str(output)})
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--rows", required=True, type=int)
    ap.add_argument("--assets-only", action="store_true")
    ap.add_argument("--allow-unlocked-fixture", action="store_true", help="assets-only test data; not the pinned checkpoint")
    ap.add_argument("--lock", type=Path, default=HERE / "source_lock.json")
    ap.add_argument("--rank-cache-bin", type=Path)
    ap.add_argument("--check-bin", type=Path)
    ap.add_argument("--engram-dir", type=Path)
    ap.add_argument("--rope-reference", type=Path, help="optional canonical rope.json with matching rows and hashes")
    args = ap.parse_args()
    try:
        result = prepare(args.model, args.output, args.rows, read_json(args.lock), args.assets_only,
                         args.rank_cache_bin, args.check_bin, args.rope_reference, args.engram_dir,
                         args.allow_unlocked_fixture)
        print(json.dumps({k: result[k] for k in ("status", "scope", "config_sha256", "rows", "hardware_qualification")}, sort_keys=True))
        return 0
    except (InputError, OSError, KeyError, TypeError, ValueError) as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())

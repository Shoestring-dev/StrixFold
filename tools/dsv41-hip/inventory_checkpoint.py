#!/usr/bin/env python3
"""Validate local safetensors headers and emit the exact TP1/TP2 rank-file schema.

No tensor weight payload is materialized. MUL1 marker words alone are read to
verify the codebook. File/content provenance is distinct from header validation.
The default locked mode rejects partial or different checkpoint inventories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from input_common import (InputError, atomic_json, checked_int, checked_relative,
                          cli_error, decode_json, read_json, sha256_file)
from snapshot_files import load_snapshot, verify_files

HERE = Path(__file__).resolve().parent
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "F8_E8M0": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
RANK_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32",
               "I16": "int16", "I32": "int32", "I64": "int64", "U8": "uint8"}
VALID_K2 = frozenset((2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16))
MUL1 = 0x83DCD12D


def header(path: Path, max_header: int = 64 << 20) -> tuple[dict, dict]:
    size = path.stat().st_size
    with path.open("rb") as src:
        lead = src.read(8)
        if len(lead) != 8:
            raise InputError(f"truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", lead)[0]
        if not 2 <= length <= max_header or length + 8 > size:
            raise InputError(f"invalid safetensors header length {length}: {path}")
        data = src.read(length)
        raw = decode_json(data)
        if not isinstance(raw, dict):
            raise InputError(f"safetensors header is not an object: {path}")
        tensors = {}
        spans = []
        for name, desc in raw.items():
            if name == "__metadata__":
                continue
            if not isinstance(desc, dict) or not isinstance(name, str):
                raise InputError(f"invalid tensor descriptor: {name}")
            dtype = desc.get("dtype")
            if dtype not in DTYPE_BYTES:
                raise InputError(f"unsupported safetensors dtype {dtype}: {name}")
            shape = desc.get("shape")
            if not isinstance(shape, list) or len(shape) > 8:
                raise InputError(f"invalid shape: {name}")
            shape = [checked_int(d, f"{name}.shape") for d in shape]
            offsets = desc.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise InputError(f"invalid offsets: {name}")
            start, end = [checked_int(v, f"{name}.offset") for v in offsets]
            expected = math.prod(shape) * DTYPE_BYTES[dtype]
            if end < start or end - start != expected or end > size - length - 8:
                raise InputError(f"invalid size or out-of-bounds tensor: {name}")
            item = {"name": name, "shard": path.name, "dtype": dtype, "shape": shape,
                    "data_offset": start, "file_offset": start + length + 8, "bytes": expected}
            if name.endswith(".mul1"):
                if dtype != "I32" or expected != 4:
                    raise InputError(f"invalid MUL1 marker shape/dtype: {name}")
                src.seek(item["file_offset"])
                marker = src.read(4)
                if len(marker) != 4 or struct.unpack("<I", marker)[0] != MUL1:
                    raise InputError(f"invalid MUL1 codebook marker: {name}")
                item["marker_verified"] = True
            tensors[name] = item
            spans.append((start, end, name))
        cursor = 0
        for start, end, name in sorted(spans):
            if start != cursor:
                raise InputError(f"overlapping or unaccounted tensor data at {name}: {path}")
            cursor = end
        if cursor != size - length - 8:
            raise InputError(f"unaccounted data after last tensor: {path}")
    return tensors, {"file": path.name, "bytes": size, "header_bytes": length,
                     "header_sha256": hashlib.sha256(data).hexdigest(), "tensors": len(tensors)}


def scan(model: Path, max_header: int = 64 << 20) -> tuple[dict, list]:
    index_path = model / "model.safetensors.index.json"
    index = None
    if index_path.exists():
        index = read_json(index_path).get("weight_map")
        if not isinstance(index, dict) or not index:
            raise InputError("index weight_map is missing or empty")
        paths = [checked_relative(model, name) for name in sorted(set(index.values()))]
    else:
        paths = sorted(model.glob("*.safetensors"))
    if not paths:
        raise InputError("no safetensors shards; metadata/index alone is not a complete checkpoint")
    all_tensors, files = {}, []
    for path in paths:
        if not path.is_file():
            raise InputError(f"missing checkpoint shard: {path}")
        tensors, info = header(path, max_header)
        relative = str(path.relative_to(model))
        info["file"] = relative
        for name, desc in tensors.items():
            if name in all_tensors:
                raise InputError(f"tensor occurs in more than one shard: {name}")
            if index is not None and index.get(name) != relative:
                raise InputError(f"index/shard mismatch for {name}")
            desc["shard"] = relative
        all_tensors.update(tensors)
        files.append(info)
    if index is not None and set(index) != set(all_tensors):
        missing = sorted(set(index) - set(all_tensors))
        raise InputError(f"index names absent from shards: {missing[:5]}")
    return all_tensors, files


def quant_inventory(tensors: dict) -> tuple[list, dict]:
    groups = []
    counts = Counter()
    for name, tensor in tensors.items():
        if not name.endswith(".trellis"):
            continue
        prefix = name[:-len(".trellis")]
        shape = tensor["shape"]
        if tensor["dtype"] != "I16" or len(shape) != 3 or min(shape) <= 0 or shape[2] % 8:
            raise InputError(f"invalid EXL3 trellis shape/dtype: {name}")
        k2 = shape[2] // 8
        if k2 not in VALID_K2:
            raise InputError(f"unsupported EXL3 K2={k2}: {name}")
        if prefix + ".mcg" in tensors or prefix + ".3inst" in tensors:
            raise InputError(f"this target requires MUL1, but other codebook marker exists: {prefix}")
        marker = tensors.get(prefix + ".mul1")
        if not marker or not marker.get("marker_verified"):
            raise InputError(f"missing verified MUL1 marker: {prefix}")
        k, n = shape[0] * 16, shape[1] * 16
        for suffix, width in (("suh", k), ("svh", n)):
            scale = tensors.get(f"{prefix}.{suffix}")
            if not scale or scale["dtype"] != "F16" or scale["shape"] != [width]:
                raise InputError(f"missing/invalid EXL3 {suffix}: {prefix}")
        if k % 128 or n % 128:
            raise InputError(f"EXL3 Hadamard dimensions must be multiples of 128: {prefix}")
        counts[k2] += 1
        role = ("drafter" if prefix.startswith("mtp.") else
                "vision" if prefix.startswith(("vision.", "aligner.")) else "target")
        groups.append({"prefix": prefix, "k": k, "n": n, "k2": k2, "bpw": k2 / 2,
                       "codebook": "mul1", "role": role, "expert": ".ffn.experts." in prefix,
                       "trellis_bytes": tensor["bytes"], "source_shard": tensor["shard"]})
    if not groups:
        raise InputError("zero EXL3 tensor groups; a vacuous format check is not a pass")
    return groups, {str(k): counts[k] for k in sorted(counts)}


def rank_plan(raw: dict, tensors: dict, rank: int, world: int = 2, dspark=True) -> dict:
    """Port of pinned plan.zig TP1/TP2 wants() and checkpoint.describe()."""
    if world not in (1, 2) or rank not in range(world):
        raise InputError("only TP1 or TP2 rank schemas are supported")
    c = raw.get("text_config", raw)
    need = lambda key: checked_int(c.get(key), key, 1)
    d, heads, hd = need("hidden_size"), need("num_attention_heads"), need("head_dim")
    groups, olora = need("o_groups"), need("o_lora_rank")
    qlora, width, vocab = need("q_lora_rank"), need("moe_intermediate_size"), need("vocab_size")
    for key, value in (("heads", heads), ("groups", groups), ("width", width), ("vocab", vocab)):
        if value % world:
            raise InputError(f"{key} is not divisible by TP world")
    if heads % groups:
        raise InputError("attention heads not divisible by o_groups")
    kv, ix, eng = [set(c.get(k, [])) for k in
                   ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids")]
    ratios = c.get("compress_ratios", [])
    entries, offset = [], 0

    def plain(name, dtype=None):
        nonlocal offset
        t = tensors.get(name)
        if t is None:
            raise InputError(f"rank {rank} needs missing tensor {name}")
        out_dtype = dtype or t["dtype"]
        if out_dtype not in RANK_DTYPES:
            raise InputError(f"unsupported rank dtype {out_dtype}: {name}")
        if out_dtype != t["dtype"] and (t["dtype"], out_dtype) != ("F16", "F32"):
            raise InputError(f"pinned loader does not support {t['dtype']} -> {out_dtype}: {name}")
        key = name + "|" + ("torch." + RANK_DTYPES[dtype] if dtype else "None")
        size = math.prod(t["shape"]) * DTYPE_BYTES[out_dtype]
        entries.append([key, RANK_DTYPES[out_dtype], list(t["shape"]), offset, size])
        offset += size

    def exl3(prefix, cols=None, rows=None, k=None, n=None):
        nonlocal offset
        source = tensors.get(prefix + ".trellis")
        if source is None:
            raise InputError(f"rank {rank} needs missing trellis {prefix}")
        shape = source["shape"]
        if len(shape) != 3:
            raise InputError(f"invalid trellis rank: {prefix}")
        original_k, original_n = shape[0] * 16, shape[1] * 16
        for span, limit, label in ((rows, original_k, "rows"), (cols, original_n, "cols")):
            if span is not None and (span[0] < 0 or span[1] > limit or span[1] <= span[0]
                                     or span[0] % 16 or span[1] % 16):
                raise InputError(f"invalid {label} slice {span} for {prefix}")
        out_k = rows[1] - rows[0] if rows else original_k
        out_n = cols[1] - cols[0] if cols else original_n
        if (k is not None and k != out_k) or (n is not None and n != out_n):
            raise InputError(f"rank projection dimensions disagree with config: {prefix}")
        for part, suffix, out_shape, dtype in (
                ("tr", "trellis", [out_k // 16, out_n // 16, shape[2]], "I16"),
                ("suh", "suh", [out_k], "F16"), ("svh", "svh", [out_n], "F16")):
            t = tensors.get(f"{prefix}.{suffix}")
            if t is None or t["dtype"] != dtype:
                raise InputError(f"missing/invalid rank source: {prefix}.{suffix}")
            key = f"{prefix}|{str(cols)}|{str(rows)}|{part}"
            size = math.prod(out_shape) * DTYPE_BYTES[dtype]
            entries.append([key, RANK_DTYPES[dtype], out_shape, offset, size])
            offset += size

    span = lambda per: (rank * per, (rank + 1) * per)
    def block(prefix, layer, expert_count):
        for hc in ("hc_attn", "hc_ffn"):
            for part in ("fn", "scale", "base"):
                plain(f"{prefix}.{hc}_{part}", "F32")
        plain(prefix + ".attn_norm.weight")
        plain(prefix + ".ffn_norm.weight")
        exl3(prefix + ".attn.wq_a", k=d, n=qlora)
        plain(prefix + ".attn.q_norm.weight")
        exl3(prefix + ".attn.wq_b", cols=span(heads // world * hd), k=qlora)
        exl3(prefix + ".attn.wkv", k=d, n=hd)
        plain(prefix + ".attn.kv_norm.weight")
        plain(prefix + ".attn.attn_sink", "F32")
        for g in range(rank * groups // world, (rank + 1) * groups // world):
            exl3(f"{prefix}.attn.wo_a.slice.{g}", k=heads // groups * hd, n=olora)
        exl3(prefix + ".attn.wo_b", rows=span(groups // world * olora), n=d)
        if layer in kv:
            exl3(prefix + ".attn.compressor.wkv", k=d)
            plain(prefix + ".attn.compressor.norm.weight")
            if layer >= len(ratios):
                raise InputError("missing compression ratio")
            if ratios[layer] > 1:
                exl3(prefix + ".attn.compressor.wgate", k=d)
        if layer in ix:
            exl3(prefix + ".attn.indexer.wq_b", k=qlora)
            plain(prefix + ".attn.indexer.weights_proj.weight", "F32")
            if layer in kv:
                exl3(prefix + ".attn.indexer.wk", k=hd, n=need("index_head_dim"))
                plain(prefix + ".attn.indexer.k_norm.weight")
        plain(prefix + ".ffn.gate.weight", "F16")
        plain(prefix + ".ffn.gate.bias", "F32")
        for e in range(expert_count + 1):
            ep = f"{prefix}.ffn.experts.{e}" if e < expert_count else prefix + ".ffn.shared_experts"
            exl3(ep + ".w1", cols=span(width // world), k=d)
            exl3(ep + ".w3", cols=span(width // world), k=d)
            exl3(ep + ".w2", rows=span(width // world), n=d)
        if layer in eng:
            rows = (need("engram_max_ngram_size") - 1) * need("engram_n_heads")
            if rows % world:
                raise InputError("Engram hash columns not divisible by world")
            exl3(prefix + ".engram.wkv", rows=span(rows // world * need("engram_head_dim")))
            plain(prefix + ".engram.q_weight", "F32")
            plain(prefix + ".engram.k_weight", "F32")

    plain("embed.weight")
    plain("norm.weight")
    exl3("head", cols=span(vocab // world), k=d)
    layers = need("num_hidden_layers")
    for layer in range(layers):
        block(f"layers.{layer}", layer, need("n_routed_experts"))
    if dspark and c.get("dspark_block_size", 0) > 0:
        drafts = need("num_nextn_predict_layers")
        for stage in range(drafts):
            block(f"mtp.{stage}", layers + stage, need("dspark_n_routed_experts"))
        exl3("mtp.0.main_proj", k=len(c.get("dspark_target_layer_ids", [])) * d, n=d)
        plain("mtp.0.main_norm.weight")
        last = f"mtp.{drafts - 1}"
        for suffix in ("norm.weight", "markov_head.embed.weight", "markov_head.head.weight", "confidence_head.proj.weight"):
            plain(last + "." + suffix)
    encoded = json.dumps(entries).encode()
    return {"rank": rank, "world": world, "dspark": dspark, "entries": entries,
            "entry_count": len(entries), "data_bytes": offset,
            "index_sha256": hashlib.sha256(encoded).hexdigest(), "index_bytes": len(encoded),
            "rank_file_bytes": offset + len(encoded) + 16,
            "schema": "TFDSRK01", "schema_source": "pinned plan.zig TP1/TP2; no TP4 parity"}


def inventory(model: Path, lock: dict | None, include_plans=True, world=2, verify_content=False) -> dict:
    raw = read_json(model / "config.json", 4 << 20)
    if raw.get("model_type") != "deepseek_v41":
        raise InputError("model_type must be deepseek_v41")
    quant = raw.get("quantization_config", {})
    if not isinstance(quant, dict) or quant.get("quant_method") != "exl3" or quant.get("codebook") != "mul1":
        raise InputError("config must declare EXL3 MUL1")
    tensors, files = scan(model)
    groups, counts = quant_inventory(tensors)
    if lock:
        expected = lock["checkpoint"]
        if counts != expected["expected_k2_counts"] or len(tensors) != expected["expected_tensors"]:
            raise InputError(f"checkpoint inventory does not match pinned 1017 pack: {len(tensors)} tensors, K2 {counts}")
    plans = [rank_plan(raw, tensors, r, world) for r in range(world)] if include_plans else []
    authenticated = []
    if verify_content:
        if lock is None:
            raise InputError("unlocked fixtures cannot authenticate the pinned model")
        names = [f["file"] for f in files] + ["config.json", "tokenizer.json", "model.safetensors.index.json"]
        authenticated = verify_files(model, load_snapshot(lock, "checkpoint"), names)
    return {"schema_version": 1, "status": "pass", "scope": "local_header_and_rank_schema",
            "locked_inventory": lock is not None, "weight_content_hashes_verified": bool(authenticated),
            "checkpoint_revision_verified": bool(authenticated), "authenticated_files": authenticated,
            "config_sha256": sha256_file(model / "config.json"),
            "tensor_count": len(tensors), "payload_bytes": sum(t["bytes"] for t in tensors.values()),
            "file_bytes": sum(f["bytes"] for f in files), "files": files,
            "k2_counts": counts, "groups": groups, "tensors": list(tensors.values()), "ranks": plans,
            "limitations": ["Headers and marker words do not authenticate the remaining weight payload.",
                            "Rank schemas are not allocated GPU weights or verified rank-cache files."]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--lock", type=Path, default=HERE / "source_lock.json")
    ap.add_argument("--allow-unlocked-fixture", action="store_true",
                    help="explicitly allow small/other local fixtures; never proves pinned checkpoint coverage")
    ap.add_argument("--no-rank-plans", action="store_true")
    ap.add_argument("--verify-content", action="store_true", help="stream every required file and compare pinned HF blob hashes")
    ap.add_argument("--world", type=int, choices=(1, 2), default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        result = inventory(args.model, None if args.allow_unlocked_fixture else read_json(args.lock),
                           not args.no_rank_plans, args.world, args.verify_content)
        atomic_json(args.output, result)
        print(json.dumps({k: result[k] for k in ("status", "scope", "locked_inventory", "tensor_count", "k2_counts")}, sort_keys=True))
        return 0
    except (InputError, OSError, KeyError, TypeError) as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())

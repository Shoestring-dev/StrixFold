#!/usr/bin/env python3
"""rope_reference.py TOOLS CONFIG ROWS OUT_JSON: a RoPE reference for ROWS rows whose provenance is the creator's kit.

Generates the full 1,114,112-row tables with the input tools' CPU generator (prepare_inputs.generate_rope), which
refuses to finish unless all four tables hash exactly to the creator kit v0.6.1 rope.json (the tables the 2x GB10
TP2 run served; assets/creator_kit_v0.6.1_rope.json). The first ROWS rows of those verified tables are then hashed
into OUT_JSON, the shape prepare_inputs.py --rope-reference takes. Verified 2026-10-19 on node1 (torch 2.9.0+cpu):
all four full tables equal the creator's, and the 8,192-row tables are their exact prefix.
"""
import hashlib, json, sys, tempfile
from pathlib import Path

tools, config, rows, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, tools)
from prepare_inputs import generate_rope  # noqa: E402

here = Path(__file__).resolve().parent
creator = json.loads((here / "assets" / "creator_kit_v0.6.1_rope.json").read_text())
if not 0 < rows <= creator["rows"]:
    raise SystemExit(f"rows must be in 1..{creator['rows']}")
with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
    generate_rope(json.loads(Path(config).read_text()), Path(tmp), creator["rows"], creator)  # raises on mismatch
    ref = {"rows": rows, "half": creator["half"], "derived_from": {
        "creator_kit": "bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10@1b5ebbee3cc8b63b8b875e46a314135eb6d669db kit/rope.json",
        "creator_rows": creator["rows"], "creator_hashes_matched": True}}
    for name in ("rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32"):
        with open(Path(tmp) / name, "rb") as f:
            ref[name] = hashlib.sha256(f.read(rows * creator["half"] * 4)).hexdigest()
out.write_text(json.dumps(ref, indent=1) + "\n")
print(json.dumps({"rows": rows, "out": str(out)}))

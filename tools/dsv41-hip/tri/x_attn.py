"""Decode / verify attention tuning: _sparse_attn_part + merge over key-block width BN, splits and warps, against the
float64 reference (the t_attn contract) with device times from rocprofv3 (run under it; prof_table-style alignment by
kernel order is not needed: each configuration's launches are tagged by the run order in the log)."""

import copy
import json
import sys

import numpy as np

import common as C
import hiprt as H
import refs as R
import t_attn as TA
from hiprt import Buf, P

L = json.loads((C.ROOT / "upstream" / "launches.json").read_text())
C.log_open("x_attn")
rng = np.random.default_rng(77)
HD = 512
# the decode case: ring window + 512 packed FP4 picks, per-row bases (round:r16 recording)
base = next(c for c in L["_sparse_attn_part"] if c["consts"]["NBLK"] == 20 and not c["consts"]["FINAL"])
mbase = L["_sparse_attn_merge_rot2"][1]


def variant(fn, case, warps, scalars=None, **consts):
    c = copy.deepcopy(case)
    c["num_warps"] = warps
    c["scalars"].update(scalars or {})
    return C.extra_variant(fn, c, **consts)


def run_cfg(d, bn, splits, warps, stages, reps):
    rows, Hh, n_idx = d["rows"], d["Hh"], d["n_idx"]
    nblk = 128 // bn + (n_idx + bn - 1) // bn
    import aot_build
    old = dict(aot_build.EXTRA_OPTIONS.get("_sparse_attn_part", {}))
    aot_build.EXTRA_OPTIONS["_sparse_attn_part"] = {"num_stages": stages}
    try:
        kp = variant("_sparse_attn_part", base, warps, BN=bn, SPLITS=splits, NBLK=nblk)
    finally:
        aot_build.EXTRA_OPTIONS["_sparse_attn_part"] = old
    km = variant("_sparse_attn_merge_rot2", mbase, 4, scalars={"rows": 3}, SPLITS=splits, DISCARD=False)
    pm, pl, po = Buf(rows * Hh * splits * 4), Buf(rows * Hh * splits * 4), Buf(rows * Hh * splits * HD * 4)
    out = Buf(rows * Hh * HD * 2)
    pos_b = Buf.of(d["pos"])
    a = dict(Q=P(Buf.of(d["q"]), "*bf16"), WSRC=P(Buf.of(d["wsrc"]), "*bf16"), WLO=P(Buf.of(d["wlo"]), "*i64"),
             COMP=P(Buf.of(d["codes"]), "*u8"), CSC=P(Buf.of(d["scales"]), "*u8"), IDX=P(Buf.of(d["idx"]), "*i64"),
             POS=P(pos_b, "*i64"), PM=P(pm, "*fp32"), PL=P(pl, "*fp32"), PO=P(po, "*fp32"),
             SINK=P(Buf.of(d["sink"]), "*fp32"), OUT=P(out, "*bf16"), scale=d["scale"],
             RBASE=P(Buf.of(d["rbase"]), "*i64"), CBASE=P(Buf.of(d["cbase"]), "*i64"), ring_size=d["ring_size"],
             n_idx=n_idx)
    cos, sin = Buf.of(np.ones((4000, 32), np.float32)), Buf.of(np.zeros((4000, 32), np.float32))   # identity rope
    am = dict(PM=a["PM"], PL=a["PL"], PO=a["PO"], SINK=a["SINK"], OUT=P(out, "*bf16"), COS=P(cos, "*fp32"),
              SIN=P(sin, "*fp32"), POS=P(pos_b, "*i64"), SUH=P(Buf.of(np.ones(Hh * HD, np.float16)), "*fp16"),
              XH=P(Buf(rows * Hh * HD * 2), "*fp16"), rows=rows)
    for _ in range(reps):
        C.run_variant(kp, (rows, Hh // 16, splits), a)
        C.run_variant(km, (rows, Hh, 1), am)
    H.sync()
    # merge_rot2 also applies the inverse rope (cos 1 sin 0 here: identity) and stores o: compare o
    got = C.from_bf16_bits(out.download((rows, Hh, HD), np.uint16))
    ref, bnd = TA.attn_ref(d["q"], TA.keys_of(d), d["sink"].astype(np.float64), d["scale"])
    m = C.cmp_rounded(got, ref, R.bf16(ref), C.ulp_bf16, bound=bnd, max_frac=1.0)
    m["ok"] = m["ok"] and m["nl2"] <= 4e-3
    C.record(kernel="_sparse_attn_part+merge_rot2", case=f"rows={rows} BN={bn} SPLITS={splits} warps={warps} stages={stages}",
             vgpr=kp["isa_stats"].get("vgpr_count"), spill=kp["isa_stats"].get("vgpr_spill_count"),
             scratch=kp["isa_stats"].get("private_segment_fixed_size"), **m)


if __name__ == "__main__":
    for rows in (1, 6):
        d = TA.build(base, rows)
        # COS/SIN identity for the merge's rope check: positions index a zero-angle table
        d["pos"] = np.minimum(d["pos"], 3999)
        for bn, splits, warps, stages in ((32, 8, 4, 1), (32, 16, 4, 1), (32, 32, 4, 1), (16, 8, 4, 1), (16, 16, 4, 1),
                                          (16, 32, 4, 1), (32, 16, 8, 2), (16, 16, 8, 2), (16, 32, 8, 2)):
            C.record(kernel="x_attn", case=f"cfg rows={rows} BN={bn} SPLITS={splits} warps={warps} stages={stages}")
            run_cfg(d, bn, splits, warps, stages, reps=20)
    C.done()

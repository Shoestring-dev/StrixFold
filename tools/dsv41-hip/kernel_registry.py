#!/usr/bin/env python3
"""The kernels the native DeepSeek-V4.1 family launches in the advertised Strix configuration, with owners.

Read-only. kernel_registry.py ROOT [--json OUT.json] [--md OUT.md]
ROOT is the TensorFold checkout at tree 1beea238. Sources of truth, all read from ROOT:
  - Triton: zig/src/families/dsv41/fixtures/launches.json (the recorded launches) and the t.run("...") call sites
  - native CUDA images: the function("...") lookups in ops.zig and exact.zig and their call sites
  - EXL3 kit cubins: the fixed template sets the loaders select (exl3_linear.zig, exl3_prefill.zig, exl3_experts.zig),
    expanded over the checkpoint's K set from the TP2 recipe's k-coverage (dense K {4,5,6,8}, routed K {2,3}, MTP K 4)
Reach: always | drafts (DSpark on) | engram (Engram layers) | profile (--profile only) | cuda-only (refused on HIP)
"""
import json
import re
import sys
from pathlib import Path

FAM = "zig/src/families/dsv41"

# Triton entry kernels -> (owner, group, reach). _gemm is EXL3's prefill matmul (exl3/prefill.py) hosted in tri_markov.zig.
TRITON = {
    "_embed_init": ("w-tri", "mhc", "always"), "_hc_partial": ("w-tri", "mhc", "always"),
    "_hc_finish": ("w-tri", "mhc", "always"), "_hc_finish_s": ("w-tri", "mhc", "always"),
    "_hc_sinkhorn": ("w-tri", "mhc", "always"), "_hc_mix_part": ("w-tri", "mhc", "always"),
    "_hc_mix_pf": ("w-tri", "mhc", "always"), "_hc_post": ("w-tri", "mhc", "always"),
    "_hc_post_only": ("w-tri", "mhc", "always"), "_collapse": ("w-tri", "mhc", "always"),
    "_collapse_norm": ("w-tri", "mhc", "always"), "_tap": ("w-tri", "draft", "drafts"),
    "_rmsnorm": ("w-tri", "norm", "always"), "_rmsnorm_rot": ("w-tri", "norm", "always"),
    "_q_kv_norm": ("w-tri", "norm/rope", "always"), "_kv_norm_rope": ("w-tri", "norm/rope", "always"),
    "_rope_heads": ("w-tri", "rope", "always"), "_engram_gate": ("w-tri", "engram", "engram"),
    "_rowmm2": ("w-tri", "router/small-gemm", "always"), "_rowmm2r": ("w-tri", "router/small-gemm", "always"),
    "_rowmm_parts": ("w-tri", "router/small-gemm", "always"), "_route": ("w-tri", "router", "always"),
    "_route_parts": ("w-tri", "router", "always"),
    "_index_score": ("w-tri", "indexer", "always"), "_index_score_rows": ("w-tri", "indexer", "always"),
    "_index_score_cand": ("w-tri", "indexer", "always"), "_score_keys": ("w-tri", "indexer", "always"),
    "_gather_tiles": ("w-tri", "indexer", "always"), "_topk_sel": ("w-tri", "indexer", "always"),
    "_topk_finish": ("w-tri", "indexer", "always"),
    "_sparse_attn_part": ("w-tri", "attention", "always"), "_sparse_attn_merge": ("w-tri", "attention", "always"),
    "_sparse_attn_merge_rot2": ("w-tri", "attention", "always"),
    "_fp4_store": ("w-tri", "cache", "always"), "_fp4qd_p2": ("w-tri", "cache", "always"),
    "_bias": ("w-tri", "markov", "drafts"), "_score": ("w-tri", "markov", "drafts"),
    "_finish": ("w-tri", "markov", "drafts"), "_pick": ("w-tri", "markov", "drafts"),
    "_gemm": ("w-exl3", "exl3-prefill", "always"),
}
# referenced by Zig but only for shapes this checkpoint never has (hidden % 1024 == 0, head_dim a power of two, rowmm2 fits)
TRITON_UNREACHED = {"_hc_finish_u": "hidden 5120 is a multiple of 1024 (tri_hc.zig:137)",
                    "_sparse_attn_merge_rot": "head_dim 512 is a power of two (tri_attn.zig:122)",
                    "_rowmm": "rowmm2's fallback, never recorded (tri_norm.zig:97)"}

# native CUDA images: symbol -> (owner, group, reach, image)
NATIVE = {
    "tf_bf16_to_f32_kernel": ("w-arch", "glue", "always", "torch_ops/pointwise.cu"),
    "tf_f32_to_bf16_kernel": (None, "unused", "unused", "torch_ops/pointwise.cu"),
    "tf_strided_copy_kernel": ("w-arch", "glue", "always", "torch_ops/movement.cu"),
    "tf_gather_rows_kernel": ("w-arch", "glue", "always", "torch_ops/movement.cu"),
    "tf_ds_f16_to_f32_kernel": ("w-arch", "glue", "always", "dsv41_ops.cu"),
    "tf_ds_add2_bf16_kernel": ("w-arch", "glue (rank-ordered add)", "always", "dsv41_ops.cu"),
    "tf_ds_scatter_rows_kernel": ("w-arch", "glue", "always", "dsv41_ops.cu"),
    "tf_ds_copy_rows16_kernel": ("w-arch", "glue", "always", "dsv41_ops.cu"),
    "tf_ds_argmax_rows_kernel": ("w-arch", "glue", "always", "dsv41_ops.cu"),
    "tf_ds_clock_kernel": ("w-arch", "glue", "profile", "dsv41_ops.cu"),
    "tf_ds_l2_prefetch_kernel": (None, "nvidia-l2", "cuda-only", "dsv41_ops.cu"),
    "tf_ds_topk_i64_kernel": ("w-tri", "indexer", "always", "dsv41_ops.cu"),
    "tf_ds_topk_indices_kernel": ("w-tri", "indexer", "always", "dsv41_ops.cu"),
    "tf_ds_cand_fast_kernel": ("w-tri", "indexer", "always", "dsv41_ops.cu"),
    "tf_ds_iota_vis_kernel": ("w-tri", "indexer", "always", "dsv41_ops.cu"),
    "tf_ds_block_max_kernel": ("w-tri", "indexer", "always", "dsv41_ops.cu"),
    "tf_ds_pool_pick_kernel": ("w-tri", "indexer", "always", "dsv41_ops.cu"),
    "tf_ds_apply_pool_kernel": ("w-tri", "indexer", "always", "dsv41_ops.cu"),
    "tf_ds_rms_norm_kernel": ("w-tri", "norm", "always", "dsv41_torch.cu"),
    "tf_ds_compress2_kernel": ("w-tri", "cache", "always", "dsv41_torch.cu"),
    "tf_ds_rope_kernel": ("w-tri", "rope", "always", "dsv41_torch.cu"),
    "tf_ds_bf16_scale_kernel": ("w-tri", "indexer", "always", "dsv41_torch.cu"),
    "tf_ds_hc_mean4_kernel": ("w-tri", "draft", "drafts", "dsv41_torch.cu"),
}

DENSE_K2 = [8, 10, 12, 16]  # dense K {4,5,6,8}
# EXL3 kit kernels: id -> (owner, group, reach, CUDA object, specialization list)
EXL3 = {
    "exl3.rot_many": ("w-exl3", "dense-decode", "always", "linear_grouped.cubin rot_many_kernel", [{}]),
    "exl3.glinear": ("w-exl3", "dense-decode", "always", "linear_grouped.cubin glinear_kernel<K2,CB,W>",
                     [{"k2": k, "cb": 2, "warps": w} for k in DENSE_K2 for w in (4, 8)]),
    "exl3.rot_in": ("w-exl3", "dense-prefill", "always", "linear.cubin rot_in_kernel", [{}]),
    "exl3.unpack": ("w-exl3", "dense-prefill", "always", "linear.cubin unpack_kernel<K2,CB>", [{"k2": k, "cb": 2} for k in DENSE_K2]),
    "exl3.experts.group_count": ("w-exl3", "experts", "always", "experts.cubin", [{}]),
    "exl3.experts.group_place": ("w-exl3", "experts", "always", "experts.cubin", [{}]),
    "exl3.experts.work_list": ("w-exl3", "experts", "always", "experts.cubin", [{}]),
    "exl3.experts.rot_in": ("w-exl3", "experts", "always", "experts.cubin rot_in_kernel<bf16>", [{}]),
    "exl3.experts.gateup_epilogue": ("w-exl3", "experts", "always", "experts.cubin", [{}]),
    "exl3.experts.down_combine": ("w-exl3", "experts", "always", "experts.cubin", [{}]),
    "exl3.experts.decode_prep": ("w-exl3", "experts", "always", "experts.cubin decode_prep_kernel<bf16>", [{}]),
    # K2 ranges as range() picks them: routed K {2,3} -> (2,10); MTP experts K 4 -> (8,8)
    "exl3.experts.grouped_mma": ("w-exl3", "experts-prompt", "always", "experts_cb.cubin grouped_mma_kernel<CB,LO,HI,1,4>",
                                 [{"cb": 2, "lo": 2, "hi": 10}, {"cb": 2, "lo": 8, "hi": 8}]),
    "exl3.experts.grouped_rows": ("w-exl3", "experts-prompt", "always", "experts_cb.cubin grouped_rows_kernel<CB,8,4,1,LO,HI,2,0>",
                                  [{"cb": 2, "lo": 2, "hi": 10}, {"cb": 2, "lo": 8, "hi": 8}]),
    "exl3.experts.grouped_cp": ("w-exl3", "experts-decode", "always", "experts_cb.cubin grouped_cp_kernel<CB,8,4,3,LO,HI,EPI>",
                                [{"cb": 2, "lo": lo, "hi": hi, "epi": epi} for lo, hi in ((2, 10), (8, 8)) for epi in (1, 2)]),
}

# library calls and collectives (not kernels we write, but each needs an AMD path and an owner)
LIBRARY = {
    "blas.sgemm_xwT": ("w-arch (binding) / w-tri (routing+indexer numerics)", "fp32 GEMM", "always",
                       "cublasSgemm_v2 T,N", ["prompt.zig:946 indexer weights", "prompt.zig:1041 MoE gate logits", "draft.zig:631 draft gate"]),
    "blas.sgemm_xv": ("w-arch (binding) / w-tri (numerics)", "fp32 GEMV", "drafts", "cublasSgemm_v2 N,N m=1", ["draft.zig:541 confidence head"]),
    "coll.bootstrap": ("w-link", "collective", "always", "ncclGetUniqueId + ncclCommInitRank over the TCP link",
                       ["model.zig:206 Fan.open", "model.zig:208 fan.comm -> prompt2d.zig:66-84", "model.zig:378 comm.deinit"]),
    "coll.allgather_prompt": ("w-link", "collective", "always", "ncclAllGather fp32 [world, n, D]",
                              ["prompt.zig:1012 attention/MoE partials (model.zig:632,638)", "prompt.zig:1232 Engram wkv", "prompt.zig:1348 head"]),
    "coll.allgather_decode": ("w-link", "collective", "always", "Engine.gatherF32 -> ncclAllGather (or verbs ring)",
                              ["round.zig:978 Engram", "round.zig:998 attention", "round.zig:1008 MoE", "round.zig:1027 head",
                               "draft.zig:504 block attention", "draft.zig:511 block MoE", "draft.zig:108 Markov gather"]),
    "coll.device_proxy": ("w-link", "collective (graphs)", "graphs", "rdma_gather.cu dsv41_rdma_stage/collect/collect_into",
                          ["rdma.zig:448,487,512 (verbs ring; replaced by a TCP proxy with the same stage/collect kernel pair)"]),
}


def call_sites(root: Path, pattern: str, files):
    out = []
    rx = re.compile(pattern)
    for f in files:
        for i, line in enumerate((root / f).read_text().splitlines(), 1):
            if rx.search(line) and not line.strip().startswith("//"):
                out.append(f"{Path(f).name}:{i}")
    return out


def main():
    root = Path(sys.argv[1])
    args = sys.argv[2:]
    rec = json.loads((root / FAM / "fixtures/launches.json").read_text())
    tri_files = [f"{FAM}/{n}.zig" for n in ("tri_basic", "tri_hc", "tri_norm", "tri_index", "tri_attn", "tri_markov")]
    served = [f"{FAM}/{n}.zig" for n in ("prompt", "round", "draft", "model", "lanes")]
    rows = []
    for fn, (owner, group, reach) in TRITON.items():
        sites = call_sites(root, r'(t\.run|runRows)\(t?,? ?"' + re.escape(fn) + r'"', tri_files)
        if fn == "_sparse_attn_merge_rot2":
            sites = call_sites(root, r'_sparse_attn_merge_rot2', tri_files)
        rows.append({"id": f"tri.{fn}", "kind": "triton", "owner": owner, "group": group, "reach": reach,
                     "cuda": "captured Triton cubin (kit aot/)", "recorded_variants": len(rec.get(fn, [])), "sites": sites, "specs": None})
    for fn, why in TRITON_UNREACHED.items():
        rows.append({"id": f"tri.{fn}", "kind": "triton", "owner": "w-tri", "group": "fallback", "reach": "unreached",
                     "cuda": why, "recorded_variants": 0, "sites": call_sites(root, r'"' + re.escape(fn) + r'"', tri_files), "specs": None})
    for sym, (owner, group, reach, image) in NATIVE.items():
        field = None
        for f in ("ops.zig", "exact.zig"):
            m = re.search(r"\.(\w+) = try \w+\.function\(\"" + re.escape(sym) + r"\"\)", (root / FAM / f).read_text())
            if m:
                field = m.group(1)
        users = call_sites(root, r"\b(o|e|x)\." + field + r"\b", [f"{FAM}/ops.zig", f"{FAM}/exact.zig"]) if field else []
        rows.append({"id": f"native.{sym.replace('_kernel', '')}", "kind": "native", "owner": owner, "group": group, "reach": reach,
                     "cuda": f"zig/kernels/cuda/{image}", "recorded_variants": None, "sites": users, "specs": None})
    # where each EXL3 kernel is launched (tree 1beea238)
    exl3_sites = {
        "exl3.rot_many": ["exl3_linear.zig:172"], "exl3.glinear": ["exl3_linear.zig:219"],
        "exl3.rot_in": ["exl3_prefill.zig:74"], "exl3.unpack": ["exl3_prefill.zig:87"],
        "exl3.experts.group_count": ["exl3_experts.zig:219"], "exl3.experts.group_place": ["exl3_experts.zig:224"],
        "exl3.experts.work_list": ["exl3_experts.zig:236"], "exl3.experts.rot_in": ["exl3_experts.zig:243"],
        "exl3.experts.gateup_epilogue": ["exl3_experts.zig:257"], "exl3.experts.down_combine": ["exl3_experts.zig:269"],
        "exl3.experts.decode_prep": ["exl3_experts.zig:359,363"], "exl3.experts.grouped_mma": ["exl3_experts.zig:212"],
        "exl3.experts.grouped_rows": ["exl3_experts.zig:213"], "exl3.experts.grouped_cp": ["exl3_experts.zig:360-363"],
    }
    for kid, (owner, group, reach, obj, specs) in EXL3.items():
        rows.append({"id": kid, "kind": "exl3", "owner": owner, "group": group, "reach": reach, "cuda": obj,
                     "recorded_variants": None, "sites": exl3_sites[kid], "specs": specs})
    for kid, (owner, group, reach, cuda, sites) in LIBRARY.items():
        rows.append({"id": kid, "kind": "library", "owner": owner, "group": group, "reach": reach, "cuda": cuda,
                     "recorded_variants": None, "sites": sites, "specs": None})
    if "--json" in args:
        Path(args[args.index("--json") + 1]).write_text(json.dumps(rows, indent=1))
    md = ["| id | kind | owner | group | reach | CUDA origin | recorded variants / specs | Zig launch sites |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        spec = r["recorded_variants"] if r["recorded_variants"] is not None else (len(r["specs"]) if r["specs"] else "")
        md.append(f"| `{r['id']}` | {r['kind']} | {r['owner']} | {r['group']} | {r['reach']} | {r['cuda']} | {spec} | {', '.join(r['sites'][:6])} |")
    text = "\n".join(md) + "\n"
    if "--md" in args:
        Path(args[args.index("--md") + 1]).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

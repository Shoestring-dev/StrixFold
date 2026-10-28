| id | kind | owner | group | reach | CUDA origin | recorded variants / specs | Zig launch sites |
|---|---|---|---|---|---|---|---|
| `tri._embed_init` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 1 | tri_basic.zig:40 |
| `tri._hc_partial` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 1 | tri_basic.zig:20 |
| `tri._hc_finish` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 1 | tri_basic.zig:21, tri_hc.zig:171 |
| `tri._hc_finish_s` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 5 | tri_hc.zig:105 |
| `tri._hc_sinkhorn` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 1 | tri_hc.zig:113 |
| `tri._hc_mix_part` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 15 | tri_hc.zig:91, tri_hc.zig:95 |
| `tri._hc_mix_pf` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 2 | tri_hc.zig:154, tri_hc.zig:161 |
| `tri._hc_post` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 3 | tri_basic.zig:30 |
| `tri._hc_post_only` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 1 | tri_hc.zig:125 |
| `tri._collapse` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 4 | tri_basic.zig:55 |
| `tri._collapse_norm` | triton | w-tri | mhc | always | captured Triton cubin (kit aot/) | 2 | tri_basic.zig:50 |
| `tri._tap` | triton | w-tri | draft | drafts | captured Triton cubin (kit aot/) | 1 | tri_basic.zig:45 |
| `tri._rmsnorm` | triton | w-tri | norm | always | captured Triton cubin (kit aot/) | 8 | tri_basic.zig:35 |
| `tri._rmsnorm_rot` | triton | w-tri | norm | always | captured Triton cubin (kit aot/) | 4 | tri_norm.zig:61 |
| `tri._q_kv_norm` | triton | w-tri | norm/rope | always | captured Triton cubin (kit aot/) | 2 | tri_norm.zig:50 |
| `tri._kv_norm_rope` | triton | w-tri | norm/rope | always | captured Triton cubin (kit aot/) | 6 | tri_norm.zig:69 |
| `tri._rope_heads` | triton | w-tri | rope | always | captured Triton cubin (kit aot/) | 10 | tri_norm.zig:78 |
| `tri._engram_gate` | triton | w-tri | engram | engram | captured Triton cubin (kit aot/) | 2 | tri_basic.zig:60 |
| `tri._rowmm2` | triton | w-tri | router/small-gemm | always | captured Triton cubin (kit aot/) | 2 | tri_norm.zig:92, tri_norm.zig:105 |
| `tri._rowmm2r` | triton | w-tri | router/small-gemm | always | captured Triton cubin (kit aot/) | 5 | tri_norm.zig:87 |
| `tri._rowmm_parts` | triton | w-tri | router/small-gemm | always | captured Triton cubin (kit aot/) | 16 | tri_norm.zig:129 |
| `tri._route` | triton | w-tri | router | always | captured Triton cubin (kit aot/) | 2 | tri_norm.zig:139 |
| `tri._route_parts` | triton | w-tri | router | always | captured Triton cubin (kit aot/) | 5 | tri_norm.zig:135 |
| `tri._index_score` | triton | w-tri | indexer | always | captured Triton cubin (kit aot/) | 16 | tri_index.zig:76, tri_index.zig:105 |
| `tri._index_score_rows` | triton | w-tri | indexer | always | captured Triton cubin (kit aot/) | 16 | tri_index.zig:86 |
| `tri._index_score_cand` | triton | w-tri | indexer | always | captured Triton cubin (kit aot/) | 1 | tri_index.zig:119 |
| `tri._score_keys` | triton | w-tri | indexer | always | captured Triton cubin (kit aot/) | 6 | tri_index.zig:166 |
| `tri._gather_tiles` | triton | w-tri | indexer | always | captured Triton cubin (kit aot/) | 2 | tri_index.zig:158 |
| `tri._topk_sel` | triton | w-tri | indexer | always | captured Triton cubin (kit aot/) | 10 | tri_index.zig:137 |
| `tri._topk_finish` | triton | w-tri | indexer | always | captured Triton cubin (kit aot/) | 2 | tri_index.zig:140 |
| `tri._sparse_attn_part` | triton | w-tri | attention | always | captured Triton cubin (kit aot/) | 15 | tri_attn.zig:107 |
| `tri._sparse_attn_merge` | triton | w-tri | attention | always | captured Triton cubin (kit aot/) | 4 | tri_attn.zig:130 |
| `tri._sparse_attn_merge_rot2` | triton | w-tri | attention | always | captured Triton cubin (kit aot/) | 2 | tri_attn.zig:122, tri_attn.zig:220 |
| `tri._fp4_store` | triton | w-tri | cache | always | captured Triton cubin (kit aot/) | 4 | tri_attn.zig:136 |
| `tri._fp4qd_p2` | triton | w-tri | cache | always | captured Triton cubin (kit aot/) | 2 | tri_attn.zig:143 |
| `tri._bias` | triton | w-tri | markov | drafts | captured Triton cubin (kit aot/) | 5 | tri_markov.zig:93 |
| `tri._score` | triton | w-tri | markov | drafts | captured Triton cubin (kit aot/) | 4 | tri_markov.zig:116 |
| `tri._finish` | triton | w-tri | markov | drafts | captured Triton cubin (kit aot/) | 4 | tri_markov.zig:122 |
| `tri._pick` | triton | w-tri | markov | drafts | captured Triton cubin (kit aot/) | 4 | tri_markov.zig:131 |
| `tri._gemm` | triton | w-exl3 | exl3-prefill | always | captured Triton cubin (kit aot/) | 8 | tri_markov.zig:161 |
| `tri._hc_finish_u` | triton | w-tri | fallback | unreached | hidden 5120 is a multiple of 1024 (tri_hc.zig:137) | 0 | tri_hc.zig:142, tri_hc.zig:314 |
| `tri._sparse_attn_merge_rot` | triton | w-tri | fallback | unreached | head_dim 512 is a power of two (tri_attn.zig:122) | 0 | tri_attn.zig:122 |
| `tri._rowmm` | triton | w-tri | fallback | unreached | rowmm2's fallback, never recorded (tri_norm.zig:97) | 0 | tri_norm.zig:99 |
| `native.tf_bf16_to_f32` | native | w-arch | glue | always | zig/kernels/cuda/torch_ops/pointwise.cu |  | ops.zig:92 |
| `native.tf_f32_to_bf16` | native | None | unused | unused | zig/kernels/cuda/torch_ops/pointwise.cu |  | ops.zig:257 |
| `native.tf_strided_copy` | native | w-arch | glue | always | zig/kernels/cuda/torch_ops/movement.cu |  | ops.zig:280 |
| `native.tf_gather_rows` | native | w-arch | glue | always | zig/kernels/cuda/torch_ops/movement.cu |  | ops.zig:317 |
| `native.tf_ds_f16_to_f32` | native | w-arch | glue | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:102 |
| `native.tf_ds_add2_bf16` | native | w-arch | glue (rank-ordered add) | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:113 |
| `native.tf_ds_scatter_rows` | native | w-arch | glue | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:158 |
| `native.tf_ds_copy_rows16` | native | w-arch | glue | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:272 |
| `native.tf_ds_argmax_rows` | native | w-arch | glue | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:292 |
| `native.tf_ds_clock` | native | w-arch | glue | profile | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:302 |
| `native.tf_ds_l2_prefetch` | native | None | nvidia-l2 | cuda-only | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:187 |
| `native.tf_ds_topk_i64` | native | w-tri | indexer | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:127 |
| `native.tf_ds_topk_indices` | native | w-tri | indexer | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:143 |
| `native.tf_ds_cand_fast` | native | w-tri | indexer | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:173 |
| `native.tf_ds_iota_vis` | native | w-tri | indexer | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:198 |
| `native.tf_ds_block_max` | native | w-tri | indexer | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:216 |
| `native.tf_ds_pool_pick` | native | w-tri | indexer | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:233 |
| `native.tf_ds_apply_pool` | native | w-tri | indexer | always | zig/kernels/cuda/dsv41_ops.cu |  | ops.zig:247 |
| `native.tf_ds_rms_norm` | native | w-tri | norm | always | zig/kernels/cuda/dsv41_torch.cu |  | exact.zig:75 |
| `native.tf_ds_compress2` | native | w-tri | cache | always | zig/kernels/cuda/dsv41_torch.cu |  | exact.zig:87 |
| `native.tf_ds_rope` | native | w-tri | rope | always | zig/kernels/cuda/dsv41_torch.cu |  | exact.zig:104 |
| `native.tf_ds_bf16_scale` | native | w-tri | indexer | always | zig/kernels/cuda/dsv41_torch.cu |  | exact.zig:115 |
| `native.tf_ds_hc_mean4` | native | w-tri | draft | drafts | zig/kernels/cuda/dsv41_torch.cu |  | exact.zig:127 |
| `exl3.rot_many` | exl3 | w-exl3 | dense-decode | always | linear_grouped.cubin rot_many_kernel | 1 | exl3_linear.zig:172 |
| `exl3.glinear` | exl3 | w-exl3 | dense-decode | always | linear_grouped.cubin glinear_kernel<K2,CB,W> | 8 | exl3_linear.zig:219 |
| `exl3.rot_in` | exl3 | w-exl3 | dense-prefill | always | linear.cubin rot_in_kernel | 1 | exl3_prefill.zig:74 |
| `exl3.unpack` | exl3 | w-exl3 | dense-prefill | always | linear.cubin unpack_kernel<K2,CB> | 4 | exl3_prefill.zig:87 |
| `exl3.experts.group_count` | exl3 | w-exl3 | experts | always | experts.cubin | 1 | exl3_experts.zig:219 |
| `exl3.experts.group_place` | exl3 | w-exl3 | experts | always | experts.cubin | 1 | exl3_experts.zig:224 |
| `exl3.experts.work_list` | exl3 | w-exl3 | experts | always | experts.cubin | 1 | exl3_experts.zig:236 |
| `exl3.experts.rot_in` | exl3 | w-exl3 | experts | always | experts.cubin rot_in_kernel<bf16> | 1 | exl3_experts.zig:243 |
| `exl3.experts.gateup_epilogue` | exl3 | w-exl3 | experts | always | experts.cubin | 1 | exl3_experts.zig:257 |
| `exl3.experts.down_combine` | exl3 | w-exl3 | experts | always | experts.cubin | 1 | exl3_experts.zig:269 |
| `exl3.experts.decode_prep` | exl3 | w-exl3 | experts | always | experts.cubin decode_prep_kernel<bf16> | 1 | exl3_experts.zig:359,363 |
| `exl3.experts.grouped_mma` | exl3 | w-exl3 | experts-prompt | always | experts_cb.cubin grouped_mma_kernel<CB,LO,HI,1,4> | 2 | exl3_experts.zig:212 |
| `exl3.experts.grouped_rows` | exl3 | w-exl3 | experts-prompt | always | experts_cb.cubin grouped_rows_kernel<CB,8,4,1,LO,HI,2,0> | 2 | exl3_experts.zig:213 |
| `exl3.experts.grouped_cp` | exl3 | w-exl3 | experts-decode | always | experts_cb.cubin grouped_cp_kernel<CB,8,4,3,LO,HI,EPI> | 4 | exl3_experts.zig:360-363 |
| `blas.sgemm_xwT` | library | w-arch (binding) / w-tri (routing+indexer numerics) | fp32 GEMM | always | cublasSgemm_v2 T,N |  | prompt.zig:946 indexer weights, prompt.zig:1041 MoE gate logits, draft.zig:631 draft gate |
| `blas.sgemm_xv` | library | w-arch (binding) / w-tri (numerics) | fp32 GEMV | drafts | cublasSgemm_v2 N,N m=1 |  | draft.zig:541 confidence head |
| `coll.bootstrap` | library | w-link | collective | always | ncclGetUniqueId + ncclCommInitRank over the TCP link |  | model.zig:206 Fan.open, model.zig:208 fan.comm -> prompt2d.zig:66-84, model.zig:378 comm.deinit |
| `coll.allgather_prompt` | library | w-link | collective | always | ncclAllGather fp32 [world, n, D] |  | prompt.zig:1012 attention/MoE partials (model.zig:632,638), prompt.zig:1232 Engram wkv, prompt.zig:1348 head |
| `coll.allgather_decode` | library | w-link | collective | always | Engine.gatherF32 -> ncclAllGather (or verbs ring) |  | round.zig:978 Engram, round.zig:998 attention, round.zig:1008 MoE, round.zig:1027 head, draft.zig:504 block attention, draft.zig:511 block MoE |
| `coll.device_proxy` | library | w-link | collective (graphs) | graphs | rdma_gather.cu dsv41_rdma_stage/collect/collect_into |  | rdma.zig:448,487,512 (verbs ring; replaced by a TCP proxy with the same stage/collect kernel pair) |

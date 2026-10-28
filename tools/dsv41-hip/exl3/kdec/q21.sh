#!/bin/bash
# q21 (k-dec2, after q20): q-qual's M1.exl3-numeric receipt from the DEPLOYED kit (deploy/b010ee4ae317 = current):
# the kit's exl3 object (EXL3_HSACO) and exl3_pf object (PF_HSACO) in the k-dec harness (hip/kdec_r), w-ref's
# contract per (kernel id, K2, row class) the engine reaches: dense glinear groups (K2 8 / 10 / 12 / 16, warps 4 / 8) at
# 1-128 rows, prompt GEMMs (exl3.gemm) at 129-2048 rows, decode experts (grouped_cp lo2_hi10 at the four (routed,
# shared) K2 classes, lo2_hi16 for the MTP blocks) at 1-63 rows, prompt experts (pf_experts, and grouped_mma / rows on
# the same routes) at 64-2048 rows. One GPU lock a run. Timing iterations cut (ITERS=5): contract only.
cd ~/tfp/kdec
while systemctl --user is-active -q run-kdec2-q20 || systemctl --user is-active -q run-kdec2-q20s; do sleep 30; done
D=$(readlink -f ~/tfp/deploy/b010ee4ae317); K=$D/native-hip/share/tensorfold/hip/gfx1151/dsv41
X=$K/objects/d639fcc6e3acb98d78873958562e7bbf9b8c036c56532c1158da5688e9a7335b.hsaco
P=$K/objects/b7f94ce9196255330ee49e31273308a8bed63388fe6ab07878135d0370ab3f62.hsaco
E="EXL3_HSACO=$X PF_HSACO=$P ITERS=5"
echo "deploy $D exl3 $X pf $P" > results/r2.meta
H=hip/kdec_r
tools/runq.sh r2_gl_a env $E $H glinear 1,2,5,6,8,9 receipt
tools/runq.sh r2_gl_b env $E $H glinear 16,17,33 receipt
tools/runq.sh r2_gl_c env $E $H glinear 63,64 receipt
tools/runq.sh r2_gl_d env $E $H glinear 100,128 receipt
for L in 20 39 11 29; do tools/runq.sh r2_x$L env $E $H dexperts 1,2,5,6,8,16,17,33,63 $L; done
tools/runq.sh r2_xm0 env $E MTP=1 ROUTED=128 $H dexperts 1,2,5,6,8,16,17,33,63 0
for L in 20 39 11 29; do tools/runq.sh r2_pf$L env $E $H pfexperts 64,100,128,512,1000,2048 $L; done
for M in 129 512 1000 2048; do tools/runq.sh r2_pg$M env $E $H pgemm $M; done
echo q21 done

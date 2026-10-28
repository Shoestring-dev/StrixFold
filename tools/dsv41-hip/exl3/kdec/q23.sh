#!/bin/bash
# q23 (k-dec2, lead-approved prefetch check, node1 only): (1) mallg: does a forked branch of a captured HIP graph run beside
# the main branch (touch S MB beside a T-us spin, then the consumer's NT read); (2) glinear's launch pair right after its
# weights were touched into the memory-side cache vs cold (harness MALL=1), the deployed exl3 object, R = 1 and 6.
cd ~/tfp/kdec
X=$(readlink -f ~/tfp/deploy/b010ee4ae317)/native-hip/share/tensorfold/hip/gfx1151/dsv41/objects/d639fcc6e3acb98d78873958562e7bbf9b8c036c56532c1158da5688e9a7335b.hsaco
tools/runq.sh mallg_80 env TB=80 hip/mallg
tools/runq.sh mallg_40 env TB=40 hip/mallg
tools/runq.sh glmall_1 env EXL3_HSACO=$X MALL=1 ITERS=20 GRAPH_REPS=5 hip/kdec_r glinear 1,6 all
echo q23 done

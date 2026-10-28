#!/bin/bash
# q22 (k-dec2, after q21): task 4 estimates. (1) wmall (every glinear decode pass on WMMA tiles: one arithmetic for all
# row counts) against new5 (the row-invariant dot2 rows, = the deployed decode kernels): graph timing ABBA at 1-63 rows,
# row invariance (rowinv_gl synthetic and real) and the contract under wmall. (2) k-dec's fused gate/up + down
# experiment (q12) and the interleaved-k glinear (q11). One GPU lock a run.
cd ~/tfp/kdec
while systemctl --user is-active -q run-kdec-q21; do sleep 30; done
N=$HOME/tfp/kdec/tmp/new5.hsaco; W=$HOME/tfp/kdec/tmp/wmall.hsaco
RS=1,2,4,6,8,9,12,16,17,24,33,48,63
tools/runq.sh wm_gl_new5_1 env EXL3_HSACO=$N hip/kdec glinear $RS all
tools/runq.sh wm_gl_wmall_1 env EXL3_HSACO=$W hip/kdec glinear $RS all
tools/runq.sh wm_gl_wmall_2 env EXL3_HSACO=$W hip/kdec glinear $RS all
tools/runq.sh wm_gl_new5_2 env EXL3_HSACO=$N hip/kdec glinear $RS all
tools/runq.sh wm_ri_synth env EXL3_HSACO=$W SYNTH=1 hip/kdec rowinv_gl 63 all
tools/runq.sh wm_ri_real env EXL3_HSACO=$W hip/kdec rowinv_gl 63 all
tools/runq.sh wm_ct env EXL3_HSACO=$W ITERS=5 hip/kdec_r glinear 1,6,16,33,63 receipt
F=$HOME/tfp/kdec/tmp/fused.hsaco
x() { local t=$1 o=$2; shift 2
  tools/runq.sh t_x39_$t env EXL3_HSACO=$o "$@" hip/kdec dexperts 1,2,3,4,5,6 39
  tools/runq.sh t_x20_$t env EXL3_HSACO=$o "$@" hip/kdec dexperts 1,2,3,4,5,6 20
  tools/runq.sh t_xm0_$t env EXL3_HSACO=$o "$@" MTP=1 ROUTED=128 hip/kdec dexperts 1,2,3,4,5,6 0; }
x new5_f1 $N; x fused_f1 $F FUSED=1; x fused_f2 $F FUSED=1; x new5_f2 $N
I=$HOME/tfp/kdec/tmp/kil.hsaco
tools/runq.sh kil_gl_new5_1 env EXL3_HSACO=$N hip/kdec glinear 1 all
tools/runq.sh kil_gl_kil_1 env EXL3_HSACO=$I hip/kdec glinear 1 all
tools/runq.sh kil_gl_kil_2 env EXL3_HSACO=$I hip/kdec glinear 1 all
tools/runq.sh kil_gl_new5_2 env EXL3_HSACO=$N hip/kdec glinear 1 all
echo q22 done

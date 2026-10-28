#!/bin/bash
# s2: served A/B for the gfx1151 plan table: A = strix-port b010ee4, B = kdec 77df5c7 (A + the table), both in
# ~/tfp/deploy-kdec; default conf at GPU_STRICT=1, code and prose 400 tokens; drafts off once each, drafts on twice each
# with FROZEN_TIMING=1 (draft widths from the tables only, so A and B run the same rounds); order A B A B B A.
# My kernel queue holds (PAUSE) while this runs. Not under bg.sh.
K=$HOME/tfp/kdec
touch $K/PAUSE
A=deploy-kdec/b010ee4ae317; B=deploy-kdec/77df5c70fce3
S="GPU_STRICT=1 BENCH_CASES=code,prose"
$K/tools/serve_run.sh pt-A-off $A $S DRAFTS=0
$K/tools/serve_run.sh pt-B-off $B $S DRAFTS=0
$K/tools/serve_run.sh pt-A-on1 $A $S DRAFTS=1 FROZEN_TIMING=1
$K/tools/serve_run.sh pt-B-on1 $B $S DRAFTS=1 FROZEN_TIMING=1
$K/tools/serve_run.sh pt-B-on2 $B $S DRAFTS=1 FROZEN_TIMING=1
$K/tools/serve_run.sh pt-A-on2 $A $S DRAFTS=1 FROZEN_TIMING=1
rm -f $K/PAUSE
echo s2 done

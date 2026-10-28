#!/bin/bash
# q25 (k-dec2): the MALL prefetch kernel alone (dev object of kdec-pf exl3.hip): harness l2pf at 8 / 16 / 24 MB
cd ~/tfp/kdec
tools/runq.sh l2pf env EXL3_HSACO=$HOME/tfp/kdec/tmp/pf.hsaco hip/kdec_r l2pf 8,16,24
echo q25 done

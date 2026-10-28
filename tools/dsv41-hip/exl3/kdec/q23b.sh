#!/bin/bash
# q23b: mallg with the HIP runtime graph knobs that may run forked branches on their own queues
cd ~/tfp/kdec
while systemctl --user is-active -q run-kdec-q23; do sleep 10; done
tools/runq.sh mallg_seg env TB=80 DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 hip/mallg
tools/runq.sh mallg_seg_q2 env TB=80 DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1 DEBUG_HIP_FORCE_GRAPH_QUEUES=2 hip/mallg
tools/runq.sh mallg_classic env TB=80 DEBUG_HIP_GRAPH_CLASSIC_PATH=1 hip/mallg
echo q23b done

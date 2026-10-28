#!/bin/bash
# q35 (k-dec2): _hc_fused counters over long sessions: drafts off (1-row rounds) then drafts on (verify widths 2-6), each
# >= 1,000 rounds with an early stop, an abandoned stream and a stop sequence; code / prose before and after.
cd ~/tfp/kdec
hc/kd_session.sh hc-sess-off deploy-cand/8526b22db6ad 0 399
hc/kd_session.sh hc-sess-on deploy-cand/8526b22db6ad 1 150
echo "$(date -Is) q35 done"

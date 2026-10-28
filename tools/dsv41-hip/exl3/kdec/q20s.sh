#!/bin/bash
# q20s (k-dec2): after q20: the planHip served A/B (s2.sh). Plain user unit (the ranks start here).
cd ~/tfp/kdec
while systemctl --user is-active -q run-kdec2-q20; do sleep 30; done
echo "$(date -Is) q20s s2"
tools/s2.sh > logs/s2.out 2>&1
echo "$(date -Is) q20s s2 rc=$? done"

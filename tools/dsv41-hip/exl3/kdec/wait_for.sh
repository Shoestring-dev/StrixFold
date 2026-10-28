#!/bin/bash
# wait_for.sh PATTERN: block until logs/runq.log has a line matching PATTERN (prints it)
cd ~/tfp/kdec
until grep -q "$1" logs/runq.log; do sleep 10; done
grep "$1" logs/runq.log | tail -1

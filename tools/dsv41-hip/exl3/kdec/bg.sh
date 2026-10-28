#!/bin/bash
# bg.sh NAME CMD...: CPU work as a transient systemd user service (survives a dropped ssh; app.slice/run-kdec-NAME, so
# the quiet-window guard can freeze it), upper cores, low priority; output and the exit code in logs/NAME.log.
# Wait: systemctl --user is-active run-kdec-NAME
cd ~/tfp/kdec
name=$1; shift
exec systemd-run --user --unit=run-kdec-$name --collect --quiet -p CPUQuota=800% -p MemoryHigh=12G -p MemoryMax=16G \
    -p WorkingDirectory=$HOME/tfp/kdec -p StandardOutput=file:$HOME/tfp/kdec/logs/$name.log -p StandardError=inherit \
    taskset -c 8-15,24-31 nice -n 19 ionice -c3 bash -c "\"\$@\"; echo \"bg.sh exit \$?\"" bg.sh "$@"

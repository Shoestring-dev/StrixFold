#!/bin/sh
# lock_watch.sh PIDFILE [MAX_S]: what holds a run's GPU lock (serve_lib.sh serve_lock_hold, as flock's command). It waits
# for this box's rank (PIDFILE, written when the rank starts) and ends when the rank does, so a rank the kernel killed
# (the OOM killer) takes its box's lock with it; before, a sleeping holder kept the lock a day and the next first_light
# waited on it. Until the rank starts it just waits; MAX_S (default a day) bounds it either way.
f=$1 max=${2:-86400} t=0
while [ "$t" -lt "$max" ]; do
  if [ -s "$f" ] && ! kill -0 "$(cat "$f")" 2>/dev/null; then exit 0; fi
  sleep 2
  t=$((t + 2))
done

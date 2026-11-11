#!/bin/sh
# pidexec.sh PIDFILE CMD...: write this process's pid to PIDFILE, then become CMD (serve_lib.sh: a rank or a lock holder
# started as a transient user unit, where a shell's $$ cannot be written on the unit's command line: systemd reads $
# itself).
echo $$ > "$1"
shift
exec "$@"

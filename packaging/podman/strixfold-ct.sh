#!/bin/bash
# StrixFold's container on a Podman host; run it without arguments for its commands.
set -euo pipefail
NAME=${CT_NAME:-strixfold}
IMAGE=${CT_IMAGE:-localhost/strixfold-rocm:latest}
DATA=${CT_DATA:-}
SSHD_DIR=${CT_SSHD_DIR:-$HOME/.config/strixfold-sshd}
HERE=$(cd "$(dirname "$0")" && pwd)

usage() {
  cat >&2 <<EOF
usage: $0 COMMAND   (on the host of each box)
  build              build the image
  create             create and start the serving container; CT_DATA=DIR also mounts DIR (a drive ~/tfp points into)
  start              start it again after a reboot
  job CMD...         CMD in a throwaway container with CPU and memory limits (builds, input preparation and checks)
                     run from the directory CMD needs; CT_JOB_MEM (16g), CT_JOB_MEM_HIGH (12g; max: no throttle,
                     a runaway dies at CT_JOB_MEM), CT_JOB_CPUS (8)
  sshd-config ADDR   write node0's sshd configuration, listening on ADDR:2222 (node0: 10.40.0.1)
  sshd ADDR          write it and start that sshd inside the container
  shell              a shell inside the container, in the current directory
  check              the GPU checks from GETTING-STARTED step 2a
EOF
  exit 2
}

# The container spec that create and job share.
spec() {
  id -nG | grep -qw render || { echo "first: sudo usermod -aG render,video $USER, then log in again" >&2; exit 1; }
  # Host network/UTS/IPC for the USB4 links and ranks; keep-id and keep-groups for the GPU; label=disable, never :Z.
  SPEC=(--init --network host --uts host --ipc host --device /dev/kfd --device /dev/dri --group-add keep-groups
    --userns keep-id --security-opt label=disable -e HOME="$HOME")
  # /var/tmp holds the GPU lock that StrixFold's scripts take, shared with the host.
  SPEC+=(-v "$HOME:$HOME" -v /var/tmp:/var/tmp -v /etc/localtime:/etc/localtime:ro)
  if [ -n "$DATA" ]; then
    [ -d "$DATA" ] || { echo "CT_DATA=$DATA is not a directory" >&2; exit 1; }
    SPEC+=(-v "$DATA:$DATA")
  fi
}

sshd_config() {
  mkdir -p "$SSHD_DIR"
  [ -f "$SSHD_DIR/hostkey" ] || ssh-keygen -q -t ed25519 -N '' -f "$SSHD_DIR/hostkey"
  cat > "$SSHD_DIR/sshd_config" <<CFG
Port 2222
ListenAddress $1
HostKey $SSHD_DIR/hostkey
PidFile $SSHD_DIR/sshd.pid
UsePAM no
PasswordAuthentication no
KbdInteractiveAuthentication no
AuthorizedKeysFile .ssh/authorized_keys
Subsystem sftp /usr/lib/openssh/sftp-server
CFG
  podman exec "$NAME" /usr/sbin/sshd -t -f "$SSHD_DIR/sshd_config"
}

case "${1:-}" in
  build) podman build -t "$IMAGE" "$HERE" ;;
  create) spec; podman run -d --name "$NAME" "${SPEC[@]}" -w "$HOME" "$IMAGE" sleep infinity ;;
  start) podman start "$NAME" ;;
  job)
    shift; [ $# -ge 1 ] || usage
    spec
    tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
    high=${CT_JOB_MEM_HIGH:-12g}; [ "$high" = max ] || high=$(numfmt --from=iec "$(echo "$high" | tr a-z A-Z)")
    # --memory is the scopes' MemoryMax and memory.high (bytes) their MemoryHigh; no swap, so the limit stops a runaway.
    exec podman run --rm "${tty[@]}" "${SPEC[@]}" -w "$PWD" --cpus "${CT_JOB_CPUS:-8}" --memory "${CT_JOB_MEM:-16g}" \
      --memory-swap "${CT_JOB_MEM:-16g}" --cgroup-conf memory.high="$high" "$IMAGE" "$@" ;;
  sshd-config) [ $# -ge 2 ] || usage; sshd_config "$2" ;;
  sshd)
    [ $# -ge 2 ] || usage
    sshd_config "$2"
    podman exec "$NAME" /usr/sbin/sshd -f "$SSHD_DIR/sshd_config" -E "$SSHD_DIR/sshd.log"
    echo "sshd in $NAME on $2:2222 (log: $SSHD_DIR/sshd.log)" ;;
  shell) exec podman exec -it -w "$PWD" "$NAME" bash -l ;;
  check)
    podman exec "$NAME" bash -c '
      rocminfo | grep -m1 gfx1151
      hipconfig --version; echo
      hipcc --version | grep -m1 clang
      /sbin/ldconfig -p | grep libamdhip64.so.7
      python3.13 --version
      ls -d /opt/rocm/lib/llvm/amdgcn/bitcode' ;;
  *) usage ;;
esac

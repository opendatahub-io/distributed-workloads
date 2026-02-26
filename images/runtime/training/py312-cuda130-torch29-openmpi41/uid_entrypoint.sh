#!/usr/bin/env bash
set -euo pipefail

# Registers the OpenShift random UID in /etc/passwd at container startup.
# Without this, Python getpass.getuser() and PyTorch cache dir setup fail
# with: KeyError: 'getpwuid(): uid not found: <random-uid>'
if ! whoami > /dev/null 2>&1; then
  if [ -w /etc/passwd ]; then
    uid="$(id -u)"
    home_dir="${HOME:-/home/mpiuser}"
    printf 'mpiuser:x:%s:0:mpiuser:%s:/bin/sh\n' "$uid" "$home_dir" >> /etc/passwd
  else
    echo "ERROR: /etc/passwd is not writable; cannot register runtime UID" >&2
    exit 1
  fi
fi
exec "$@"


#!/usr/bin/env bash
set -euo pipefail

uid="$(id -u)"
if ! getent passwd "$uid" >/dev/null 2>&1; then
  if [ ! -w /etc/passwd ]; then
    echo "ERROR: /etc/passwd is not writable; cannot register runtime UID" >&2
    exit 1
  fi

  home_dir="${HOME:-/home/mpiuser}"
  if [ -d /home/mpiuser/.ssh ]; then
    home_dir=/home/mpiuser
  fi

  if [[ "$home_dir" == *:* || "$home_dir" == *$'\n'* || "$home_dir" == *$'\r'* ]]; then
    echo "ERROR: HOME contains invalid characters for /etc/passwd entry" >&2
    exit 1
  fi

  printf 'mpiuser:x:%s:0:mpiuser:%s:/bin/sh\n' "$uid" "$home_dir" >> /etc/passwd
fi

launcher="$(basename "$0")"
if [[ "$launcher" == "mpirun" || "$launcher" == "mpiexec" ]]; then
  # Wait for the TrainJob hostfile names before OpenMPI starts SSH daemons.
  # Set RHOAI_TRAINER_ENABLE_MPI_HOST_WAIT=false to skip this check.
  hostfile="${OMPI_MCA_orte_default_hostfile:-}"
  host_wait_enabled="${RHOAI_TRAINER_ENABLE_MPI_HOST_WAIT:-true}"
  if [[ "$host_wait_enabled" != "false" && -n "$hostfile" && -f "$hostfile" ]]; then
    dns_deadline=$((SECONDS + 60))
    while read -r host _ || [[ -n "$host" ]]; do
      [[ -z "$host" || "$host" == \#* ]] && continue
      wait_logged=false
      while :; do
        remaining=$((dns_deadline - SECONDS))
        if (( remaining <= 0 )); then
          echo "ERROR: MPI host '$host' did not resolve within 60 seconds" >&2
          exit 1
        fi
        if timeout "${remaining}s" getent hosts "$host" >/dev/null 2>&1; then
          break
        fi
        if (( SECONDS >= dns_deadline )); then
          echo "ERROR: MPI host '$host' did not resolve within 60 seconds" >&2
          exit 1
        fi
        if [[ "$wait_logged" == false ]]; then
          echo "Waiting for MPI host '$host' to resolve" >&2
          wait_logged=true
        fi
        sleep 1
      done
    done < "$hostfile"
  fi

  # Bound the inherited open-file limit for MPI startup, as in
  # https://github.com/kubeflow/trainer/pull/3783.
  max_nofile="${MPIRUN_MAX_NOFILE:-65536}"
  if [ "$(ulimit -n)" -gt "$max_nofile" ]; then
    ulimit -n "$max_nofile"
  fi

  # Scope the MPI SSH settings to OpenMPI worker launches. This image is also
  # used as a notebook image, so a global ssh_config drop-in would break users
  # connecting to normal SSH services such as GitHub.
  export OMPI_MCA_plm_rsh_args="${OMPI_MCA_plm_rsh_args:+${OMPI_MCA_plm_rsh_args} }-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentityFile=/home/mpiuser/.ssh/id_rsa -o SendEnv=PATH -o SendEnv=LD_LIBRARY_PATH"
  exec "/usr/lib64/openmpi/bin/$launcher" "$@"
fi

exec "$@"

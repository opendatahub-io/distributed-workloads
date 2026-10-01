#!/usr/bin/env bash
set -euo pipefail

if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  if [ ! -w /etc/passwd ]; then
    echo "ERROR: /etc/passwd is not writable; cannot register runtime UID" >&2
    exit 1
  fi

  uid="$(id -u)"
  # Prefer /home/mpiuser when SSH keys are mounted there (MPI TrainJob), even if
  # the Jupyter base image sets HOME=/opt/app-root/src.
  if [ -d /home/mpiuser/.ssh ]; then
    home_dir=/home/mpiuser
  else
    home_dir="${HOME:-/home/mpiuser}"
  fi

  if [[ "$home_dir" == *:* || "$home_dir" == *$'\n'* || "$home_dir" == *$'\r'* ]]; then
    echo "ERROR: HOME contains invalid characters for /etc/passwd entry" >&2
    exit 1
  fi

  printf 'mpiuser:x:%s:0:mpiuser:%s:/bin/sh\n' "$uid" "$home_dir" >> /etc/passwd
fi

launcher="$(basename "$0")"
if [[ "$launcher" == "mpirun" || "$launcher" == "mpiexec" ]]; then
  # Scope the MPI SSH settings to OpenMPI worker launches. This image is also
  # used as a notebook image, so a global ssh_config drop-in would break users
  # connecting to normal SSH services such as GitHub.
  export OMPI_MCA_plm_rsh_args="${OMPI_MCA_plm_rsh_args:+${OMPI_MCA_plm_rsh_args} }-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentityFile=/home/mpiuser/.ssh/id_rsa -o SendEnv=PATH -o SendEnv=LD_LIBRARY_PATH"
  exec "/usr/lib64/openmpi/bin/$launcher" "$@"
fi

exec "$@"

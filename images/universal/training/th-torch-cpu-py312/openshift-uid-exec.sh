#!/usr/bin/env bash
set -euo pipefail

if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  echo "ERROR: runtime UID $(id -u) has no passwd entry; OpenShift/CRI-O must provide UID mapping" >&2
  exit 1
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

#!/usr/bin/env bash
set -euo pipefail
# Universal entrypoint for workbench mode
#
# Workbench (OpenShift): NOTEBOOK_ARGS env var is set → starts Jupyter notebook
# Training jobs: Controller overrides entrypoint entirely, this script is not used
#
# Fallback: If no NOTEBOOK_ARGS and a command is provided, run that command

if [ -n "${NOTEBOOK_ARGS:-}" ]; then
    # Workbench mode: NOTEBOOK_ARGS is set (OpenShift injects this)
    # Note: NOTEBOOK_ARGS is trusted platform input (set by OpenShift workbench controller)
    # and requires word splitting for multiple arguments
    exec sh -lc 'exec start-notebook.sh ${NOTEBOOK_ARGS}'
fi

# Fallback: run provided command (e.g., from CMD or manual override)
exec "${@:-start-notebook.sh}"

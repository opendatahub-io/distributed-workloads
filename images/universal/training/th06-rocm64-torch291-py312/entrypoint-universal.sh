#!/usr/bin/env sh
set -e
# Universal entrypoint for workbench mode
#
# Workbench (OpenShift): NOTEBOOK_ARGS env var is set → starts Jupyter notebook
# Training jobs: Controller overrides entrypoint entirely, this script is not used
#
# Fallback: If no NOTEBOOK_ARGS and a command is provided, run that command

if [ -n "${NOTEBOOK_ARGS:-}" ]; then
    # Workbench mode: NOTEBOOK_ARGS is set (OpenShift injects this)
    exec sh -lc "exec start-notebook.sh ${NOTEBOOK_ARGS}"
fi

# Fallback: run provided command (e.g., from CMD or manual override)
exec "${@:-start-notebook.sh}"

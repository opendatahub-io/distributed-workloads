#!/usr/bin/env sh
set -e
# Universal entrypoint
# Behavior:
# - If NOTEBOOK_ARGS is set, start the workbench notebook with those args (if any).
# - Otherwise, run the provided command. If no command provided, exit with help.

# If NOTEBOOK_ARGS is present (even if empty), run start-notebook.sh
if [ -n "${NOTEBOOK_ARGS+x}" ]; then
	if [ -n "${NOTEBOOK_ARGS}" ]; then
		# Use a login shell to correctly parse NOTEBOOK_ARGS word splitting and quotes
		exec sh -lc "exec start-notebook.sh ${NOTEBOOK_ARGS}"
	else
		exec start-notebook.sh
	fi
fi

# Otherwise, run provided command, or error if none
if [ "$#" -gt 0 ]; then
	exec "$@"
else
	echo "No NOTEBOOK_ARGS set and no command provided. Either set NOTEBOOK_ARGS to run start-notebook.sh, or provide a command, e.g.: python -m your.module" >&2
	exit 2
fi


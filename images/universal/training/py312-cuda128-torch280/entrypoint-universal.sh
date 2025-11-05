#!/usr/bin/env sh
set -e
# Universal entrypoint
# Behavior:
# - If RUNTIME_MODE is set to a truthy value, run provided command (headless). If no command provided, exit with help.
# - Otherwise, start the workbench notebook exactly like the base image.

is_truthy() {
	case "$(printf %s "$1" | tr '[:upper:]' '[:lower:]')" in
		y|yes|true|1) return 0 ;;
		*) return 1 ;;
	esac
}

if is_truthy "${RUNTIME_MODE:-}"; then
	if [ "$#" -gt 0 ]; then
		exec "$@"
	else
		echo "RUNTIME_MODE=true but no command provided. Provide a command, e.g.: python -m your.module" >&2
		exit 2
	fi
fi

exec start-notebook.sh

#!/usr/bin/env bash
set -euo pipefail

uv run ./hack/sync_agents_config.py > /dev/null

bad_files=$(git status --porcelain -- .claude .cursor)

if [[ -n ${bad_files} ]]; then
    echo "!!! AI agent config is out of sync with ai/:"
    echo "${bad_files}"
    echo "Try running 'make sync-agents-config'"
    exit 1
fi

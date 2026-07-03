#!/usr/bin/env bash
set -euo pipefail

SKILLS_DIR=".claude/skills"
CURSOR_RULES_DIR=".cursor/rules"

get_description() {
  case "$1" in
    add-e2e-test)       echo "Guide for adding E2E tests to the distributed-workloads repo" ;;
    add-benchmark)      echo "Guide for adding benchmarks to benchmarks/" ;;
    update-support-lib) echo "Guide for modifying the shared test support library" ;;
    *)                  echo "$1" ;;
  esac
}

get_globs() {
  case "$1" in
    add-e2e-test)       echo "tests/**/*.go" ;;
    add-benchmark)      echo "benchmarks/**/*" ;;
    update-support-lib) echo "tests/common/support/**/*.go" ;;
    *)                  echo "" ;;
  esac
}

sync_cursor() {
  mkdir -p "$CURSOR_RULES_DIR"

  for skill_dir in "$SKILLS_DIR"/*/; do
    name=$(basename "$skill_dir")
    skill_file="$skill_dir/SKILL.md"
    [ -f "$skill_file" ] || continue

    desc=$(get_description "$name")
    glob=$(get_globs "$name")
    out="$CURSOR_RULES_DIR/$name.mdc"

    {
      echo "---"
      echo "description: \"$desc\""
      [ -n "$glob" ] && echo "globs: \"$glob\""
      echo "alwaysApply: false"
      echo "---"
      echo ""
      cat "$skill_file"
    } > "$out"

    echo "  cursor: $out"
  done
}

echo "Syncing skills from $SKILLS_DIR ..."
sync_cursor
echo "Done."

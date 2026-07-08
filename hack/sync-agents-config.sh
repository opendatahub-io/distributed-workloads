#!/usr/bin/env bash
set -euo pipefail

AGENTS_DIR="ai"
AGENTS_SKILLS_DIR="$AGENTS_DIR/skills"

CLAUDE_CONFIG_DIR=".claude"
CLAUDE_SKILLS_DIR="$CLAUDE_CONFIG_DIR/skills"

CURSOR_CONFIG_DIR=".cursor"
CURSOR_SKILLS_DIR="$CURSOR_CONFIG_DIR/skills"

sync_skills() {
  local found=0
  for skill_dir in "$AGENTS_SKILLS_DIR"/*/; do
    [ -d "$skill_dir" ] || continue
    local name
    name=$(basename "$skill_dir")
    local skill_file="$skill_dir/SKILL.md"
    [ -f "$skill_file" ] || continue
    found=1

    mkdir -p "$CLAUDE_SKILLS_DIR/$name"
    cp "$skill_file" "$CLAUDE_SKILLS_DIR/$name/SKILL.md"
    echo "  claude: $CLAUDE_SKILLS_DIR/$name/SKILL.md"

    mkdir -p "$CURSOR_SKILLS_DIR/$name"
    cp "$skill_file" "$CURSOR_SKILLS_DIR/$name/SKILL.md"
    echo "  cursor: $CURSOR_SKILLS_DIR/$name/SKILL.md"
  done

  if [ "$found" -eq 0 ]; then
    echo "  (no skills found in $AGENTS_SKILLS_DIR)"
  fi
}

echo "Syncing AI config from $AGENTS_DIR/ ..."
echo ""
echo "Skills:"
sync_skills
echo ""
echo "Done."

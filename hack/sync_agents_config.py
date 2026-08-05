#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""
Sync AI agent config from ai/ to ai agents like cursor, claude etc.

ai/ is the single source of truth:

    ai/skills/<name>/SKILL.md + metadata.json
    ai/rules/<name>/RULE.md + metadata.json

The markdown body is stored once and every agent file is rendered from
body + metadata, because each agent expects its own layout and frontmatter:

    .claude/skills/<name>/SKILL.md   frontmatter: name, description
    .cursor/skills/<name>/SKILL.md   frontmatter: name, description
    .claude/rules/<name>.md          frontmatter: globs
    .cursor/rules/<name>.mdc         frontmatter: description, globs, alwaysApply

Every supported agent is a subclass of the Agent base class implementing its
own rendering, so onboarding a new agent only requires defining a subclass
and adding one instance to AGENTS.

All generated directories are deleted and rebuilt on every run, so outputs
whose source was removed from ai/ disappear as well.

Dependencies are declared inline (PEP 723).
Run via `make sync-agents-config`.
"""

import json
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, NoReturn

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

AI_DIR = REPO_ROOT / "ai"
AI_SKILLS_DIR = AI_DIR / "skills"
AI_RULES_DIR = AI_DIR / "rules"


def abort(message: str) -> NoReturn:
    """Print an error message to stderr and exit with status 1."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


@dataclass(frozen=True)
class SourceDocument:
    """Base for agent documents loaded from ai/ source directories."""

    REQUIRED_METADATA_KEYS: ClassVar[list[str]] = []

    @classmethod
    def _load_metadata(cls, source_dir: Path) -> dict[str, Any]:
        """Read and validate metadata.json from source_dir.

        Aborts the script when the file is missing, is not valid JSON, or
        lacks one of the class's required metadata keys.
        """
        metadata_file = source_dir / "metadata.json"
        source_path = source_dir.relative_to(REPO_ROOT)
        metadata_path = metadata_file.relative_to(REPO_ROOT)
        if not metadata_file.is_file():
            abort(f"{source_path} is missing metadata.json")
        try:
            metadata = json.loads(metadata_file.read_text())
        except json.JSONDecodeError as e:
            abort(f"{metadata_path} is not valid JSON: {e}")
        for key in cls.REQUIRED_METADATA_KEYS:
            if key not in metadata:
                abort(f"{metadata_path} is missing key '{key}'")
        return metadata


@dataclass(frozen=True)
class Skill(SourceDocument):
    """One skill loaded from ai/skills/<name>/."""

    REQUIRED_METADATA_KEYS: ClassVar[list[str]] = ["description"]

    name: str
    description: str
    body: str

    @classmethod
    def load(cls, source_dir: Path) -> "Skill":
        """Load a skill from its source directory."""
        metadata = cls._load_metadata(source_dir)
        return cls(
            name=source_dir.name,
            description=metadata["description"],
            body=(source_dir / "SKILL.md").read_text(),
        )


@dataclass(frozen=True)
class Rule(SourceDocument):
    """One rule loaded from ai/rules/<name>/."""

    REQUIRED_METADATA_KEYS: ClassVar[list[str]] = ["description", "globs"]

    name: str
    description: str
    globs: list[str]
    body: str

    @classmethod
    def load(cls, source_dir: Path) -> "Rule":
        """Load a rule from its source directory."""
        metadata = cls._load_metadata(source_dir)
        return cls(
            name=source_dir.name,
            description=metadata["description"],
            globs=metadata["globs"],
            body=(source_dir / "RULE.md").read_text(),
        )


class Agent(ABC):
    """One supported AI agent and how its config files are generated."""

    name: str
    config_dir: Path
    rule_suffix: str

    @property
    def skills_dir(self) -> Path:
        """Directory holding this agent's generated skills."""
        return self.config_dir / "skills"

    @property
    def rules_dir(self) -> Path:
        """Directory holding this agent's generated rules."""
        return self.config_dir / "rules"

    @staticmethod
    def _render_frontmatter(fields: dict) -> str:
        """Render a YAML frontmatter block from a mapping of field values."""
        rendered_fields = yaml.dump(fields, sort_keys=False, allow_unicode=True)
        return f"---\n{rendered_fields}---"

    @staticmethod
    def _write_generated_file(path: Path, content: str) -> None:
        """Write content to path, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        generated_path = path.relative_to(REPO_ROOT)
        print(f"  {generated_path}")

    @abstractmethod
    def render_skill(self, skill: Skill) -> str:
        """Render a SKILL.md following the Agent Skills standard.

        The standard is shared by all supported agents: SKILL.md with
        name + description frontmatter, name matching the folder name.
        Subclasses can return this default via super() or render their
        own format.
        """
        frontmatter = self._render_frontmatter(
            {"name": skill.name, "description": skill.description}
        )
        return f"{frontmatter}\n\n{skill.body}"

    @abstractmethod
    def render_rule(self, rule: Rule) -> str:
        """Render the content of one rule file for this agent."""

    def write_skill(self, skill: Skill) -> None:
        """Render and write this agent's file for one skill."""
        skill_path = self.skills_dir / skill.name / "SKILL.md"
        self._write_generated_file(skill_path, self.render_skill(skill))

    def write_rule(self, rule: Rule) -> None:
        """Render and write this agent's file for one rule."""
        rule_path = self.rules_dir / f"{rule.name}{self.rule_suffix}"
        self._write_generated_file(rule_path, self.render_rule(rule))

    def remove_generated_dirs(self) -> None:
        """Delete this agent's generated config directories."""
        for generated_dir in (self.skills_dir, self.rules_dir):
            shutil.rmtree(generated_dir, ignore_errors=True)
            removed_path = generated_dir.relative_to(REPO_ROOT)
            print(f"  removed {removed_path}/")


class ClaudeAgent(Agent):
    """Claude Code: generates .claude/skills/ and .claude/rules/."""

    name = "claude"
    config_dir = REPO_ROOT / ".claude"
    rule_suffix = ".md"

    def render_skill(self, skill: Skill) -> str:
        """Claude Code follows the shared Agent Skills standard."""
        return super().render_skill(skill)

    def render_rule(self, rule: Rule) -> str:
        """Render a Claude Code rule: *.md with globs frontmatter."""
        frontmatter = self._render_frontmatter({"globs": rule.globs})
        return f"{frontmatter}\n\n{rule.body}"


class CursorAgent(Agent):
    """Cursor: generates .cursor/skills/ and .cursor/rules/."""

    name = "cursor"
    config_dir = REPO_ROOT / ".cursor"
    rule_suffix = ".mdc"

    def render_skill(self, skill: Skill) -> str:
        """Cursor follows the shared Agent Skills standard."""
        return super().render_skill(skill)

    def render_rule(self, rule: Rule) -> str:
        """Render a Cursor rule: *.mdc with description, globs, and
        alwaysApply frontmatter."""
        frontmatter = self._render_frontmatter(
            {
                "description": rule.description,
                "globs": ",".join(rule.globs),
                "alwaysApply": False,
            }
        )
        return f"{frontmatter}\n\n{rule.body}"


AGENTS = [ClaudeAgent(), CursorAgent()]


def remove_generated_dirs() -> None:
    """Delete every agent's generated config directories."""
    for agent in AGENTS:
        agent.remove_generated_dirs()


def sync_skills() -> None:
    """Render every skill from ai/skills/ into each agent's skills directory."""
    found = False
    for skill_dir in sorted(AI_SKILLS_DIR.iterdir()):
        if not (skill_dir / "SKILL.md").is_file():
            continue
        found = True
        skill = Skill.load(skill_dir)
        for agent in AGENTS:
            agent.write_skill(skill)
    if not found:
        skills_path = AI_SKILLS_DIR.relative_to(REPO_ROOT)
        print(f"  (no skills found in {skills_path})")


def sync_rules() -> None:
    """Render every rule from ai/rules/ into each agent's rules directory."""
    found = False
    for rule_dir in sorted(AI_RULES_DIR.iterdir()):
        if not (rule_dir / "RULE.md").is_file():
            continue
        found = True
        rule = Rule.load(rule_dir)
        for agent in AGENTS:
            agent.write_rule(rule)
    if not found:
        rules_path = AI_RULES_DIR.relative_to(REPO_ROOT)
        print(f"  (no rules found in {rules_path})")


def main() -> None:
    """Regenerate all agent config from the ai/ source of truth."""
    ai_path = AI_DIR.relative_to(REPO_ROOT)
    print(f"Syncing AI config from {ai_path}/ ...")
    print()
    print("Cleanup:")
    remove_generated_dirs()
    print()
    print("Skills:")
    sync_skills()
    print()
    print("Rules:")
    sync_rules()
    print()
    print("Done.")


if __name__ == "__main__":
    main()

"""Patch training_hub/__init__.py to make SFT/OSFT/LoRA imports optional.

Wraps imports that depend on instructlab-training, rhai-innovation-mini-trainer,
or unsloth in try/except so the package loads when only GRPO modules are available.
"""
import pathlib
import sys

init_path = pathlib.Path(sys.argv[1])
src = init_path.read_text()

guarded_prefixes = (
    "from .algorithms.sft ",
    "from .algorithms.osft ",
    "from .algorithms.lora ",
    "from .profiling",
)

lines = src.splitlines(keepends=True)
new_lines = []
for line in lines:
    stripped = line.lstrip()
    if any(stripped.startswith(p) for p in guarded_prefixes):
        indent = line[: len(line) - len(stripped)]
        new_lines.append(f"{indent}try:\n")
        new_lines.append(f"{indent}    {stripped}")
        new_lines.append(f"{indent}except (ImportError, ModuleNotFoundError):\n")
        new_lines.append(f"{indent}    pass\n")
    else:
        new_lines.append(line)

init_path.write_text("".join(new_lines))
print(f"Patched {init_path}")

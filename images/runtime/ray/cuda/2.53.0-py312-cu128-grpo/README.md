# Ray GRPO Runtime — CUDA 12.8, Python 3.12

Extends `quay.io/modh/ray:2.53.0-py312-cu128` with the ML stack needed for **LoRA-GRPO** (Group Relative Policy Optimization with LoRA) training using the [training-hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) library, orchestrated by [verl](https://github.com/volcengine/verl) on KubeRay.

## Key packages

| Package | Version | Role |
|---------|---------|------|
| vLLM | 0.12.0 | LLM rollout engine |
| PyTorch | 2.9.0 | Deep learning framework |
| verl | 0.7.1 | GRPO training orchestrator (FSDP + vLLM) |
| flash-attn | 2.8.3 | Memory-efficient attention |
| training-hub | latest | High-level GRPO training API |
| peft | ≥0.15 | LoRA adapters |
| transformers | ≥4.57.6 | HuggingFace model loading |

## Build

```bash
podman build -t quay.io/<org>/ray-grpo:2.53.0-py312-cu128 .
```

## Usage

Use this image as the `rayImage` for both head and worker pods in a KubeRay `RayCluster` CR. verl launches training via `python -m verl.trainer.main_ppo` on the head node and distributes FSDP training + vLLM rollouts across workers.

## Notes

- **Git branch installs**: `instructlab-training` and `rhai-innovation-mini-trainer` are currently installed from their `main` branches to pick up the relaxed `numba>=0.61.2` constraint (merged but not yet released to PyPI). `training-hub[grpo]` is installed from the `lora-grpo` branch (GRPO code not yet merged to `main`). Once all packages are released, Step 4 simplifies to `pip install "training-hub[grpo]"`.

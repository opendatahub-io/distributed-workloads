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

## Known workarounds

- **`--no-deps` install of training-hub**: training-hub's base dependencies (`instructlab-training`, `rhai-innovation-mini-trainer`) require `numba>=0.62`, which conflicts with vllm's `numba==0.61.2`. The GRPO code paths don't use those packages, so training-hub is installed without its declared dependencies and runtime deps are supplied explicitly.
- **`patch_init.py`**: training-hub's `__init__.py` eagerly imports modules that depend on the excluded packages. This build-time patch wraps those imports in `try/except` blocks.

Both workarounds can be removed once training-hub makes those dependencies optional extras.

# Ray + GRPO runtime image for KubeRay on OpenShift AI
#
# Base: quay.io/modh/ray:2.53.0-py312-cu128
#   Ray 2.53.0, Python 3.12, CUDA 12.8, CUDA dev headers, numpy<2
#
# Adds: vLLM 0.12.0 (rollout engine), verl 0.7.1 (FSDP training orchestrator),
#       training-hub, flash-attn, and supporting packages for LoRA-GRPO training.
#
# Used as the container image for both head and worker pods in a KubeRay
# RayCluster. verl launches via `python -m verl.trainer.main_ppo` on the
# head node and distributes FSDP training + vLLM rollouts across workers.
#
# Build:
#   podman build -t quay.io/<org>/ray-grpo:2.53.0-py312-cu128 .
#
# Install order matters:
#   1. vllm first — pins torch 2.9.0, numba 0.61.2, flashinfer, triton
#   2. flash-attn — pre-built wheel matched to torch 2.9 ABI
#   3. verl + ML libs
#   4. training-hub with --no-deps (base deps conflict with vllm's numba)
#   5. Remaining runtime deps explicitly

FROM quay.io/modh/ray:2.53.0-py312-cu128

LABEL name="ray-grpo-cu128" \
      summary="Ray + vLLM + verl GRPO runtime for KubeRay on OpenShift AI" \
      description="Extends the ODH Ray 2.53.0 CUDA 12.8 image with vLLM 0.12.0, \
verl 0.7.1, flash-attn 2.8.3, and training-hub for LoRA-GRPO training." \
      io.k8s.display-name="Ray GRPO Runtime (CUDA 12.8)" \
      io.k8s.description="Ray 2.53.0 + vLLM 0.12.0 + verl 0.7.1 GRPO training image" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

USER 0
WORKDIR /opt/app-root/bin

# ── 1. vLLM + pinned companions ─────────────────────────────────────
# vllm 0.12.0 requires torch==2.9.0 and numba==0.61.2.
# verl 0.7.1 requires vllm>=0.8.5 and its code needs run_headless
# (first available in vllm ~0.10). vllm 0.12.0 is the max verl allows.
RUN pip install --no-cache-dir \
    "vllm==0.12.0" \
    "torch==2.9.0" \
    "numpy<2.0.0" \
    "numba==0.61.2" \
    "llvmlite==0.44.*"

# ── 2. flash-attn (pre-built wheel) ──────────────────────────────────
# torch 2.9.0 uses cxx11_abi=True (the cu126+ ABI switch from PyTorch PR #142064).
# Use the pre-built wheel from flash-attention GitHub releases matched
# to torch 2.9, CUDA 12, cp312, cxx11abiTRUE.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# ── 3. verl + ML framework libs ─────────────────────────────────────
RUN pip install --no-cache-dir \
    "verl>=0.7" \
    "peft>=0.15" \
    "transformers>=4.57.6,<5.0" \
    "datasets>=4.0.0"

# ── 4. training-hub (code-only) ──────────────────────────────────────
# --no-deps is intentional: training-hub's base [project].dependencies
# pull instructlab-training and rhai-innovation-mini-trainer, which pin
# numba>=0.62 and llvmlite>=0.42. That conflicts with vllm's numba==0.61.2.
# The GRPO/verl code paths don't call into those packages, so skipping
# them is safe. All actually-needed deps are installed in steps 1-3 and 5.
RUN pip install --no-cache-dir --no-deps \
    "training-hub[grpo]"

# ── 5. Remaining runtime deps + grpo extras ──────────────────────────
# These are training-hub runtime deps that are safe to install (no
# numba/llvmlite conflict) and not already satisfied by vllm/verl.
# litellm is capped below 1.82.7 (supply-chain compromise in
# 1.82.7-1.82.8, 2026-03-24).
RUN pip install --no-cache-dir \
    "openpipe-art" \
    "litellm>=1.82.0,<1.82.7" \
    "trl" \
    "packaging>=24.2" \
    "rich>=14.1.0" \
    "pydantic>=2.7.0" \
    "aiofiles>=23.2.1" \
    "accelerate>=0.34.2" \
    "sympy>=1.0" \
    "networkx>=3.0" \
    "jinja2>=3.1.0" \
    "fsspec>=2025.0" \
    "pandas>=2.2" \
    "matplotlib>=3.9" \
    "multiprocess>=0.70.16" \
    "aiohttp>=3.12" \
    "pyparsing>=3.0" \
    "regex>=2025.0" \
    "psutil>=6.0" \
    "urllib3>=2.4" \
    "frozenlist>=1.7" \
    "xxhash>=3.0" \
    "requests>=2.32.5" \
    "filelock>=3.19.1" \
    "mpmath>=1.3.0" \
    "timm>=1.0"

# ── 6. Ray / NCCL / vLLM environment defaults ───────────────────────
ENV NCCL_DEBUG=WARN \
    TOKENIZERS_PARALLELISM=true \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    VLLM_LOGGING_LEVEL=WARN \
    VLLM_USE_V1=1 \
    RAY_DEDUP_LOGS=0

# ── 7. Patch training_hub __init__.py for partial-install ─────────────
# training_hub's __init__ eagerly imports sft/osft/lora which need
# instructlab-training and rhai-innovation-mini-trainer (excluded via
# --no-deps to avoid numba conflict). Wrap those imports in try/except
# so the package loads cleanly with only the GRPO modules available.
COPY patch_init.py /tmp/patch_init.py
RUN python /tmp/patch_init.py /opt/app-root/lib64/python3.12/site-packages/training_hub/__init__.py && \
    rm /tmp/patch_init.py

# ── 8. Smoke test ───────────────────────────────────────────────────
RUN python -c "\
import ray; print(f'ray {ray.__version__}'); \
import vllm; print(f'vllm {vllm.__version__}'); \
import torch; print(f'torch {torch.__version__}'); \
import numpy; print(f'numpy {numpy.__version__}'); \
import verl; print('verl OK'); \
import peft; print(f'peft {peft.__version__}'); \
import transformers; print(f'transformers {transformers.__version__}'); \
import training_hub; print('training_hub OK'); \
from training_hub import lora_grpo; print('lora_grpo OK'); \
from training_hub import VeRLLoRAGRPOBackend; print('verl backend OK'); \
"

USER 1001
WORKDIR /opt/app-root/src

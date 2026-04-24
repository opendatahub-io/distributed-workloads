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
#   4. training-hub[grpo] (numba constraint relaxed to >=0.61.2 upstream)

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

# ── 4. training-hub[grpo] ────────────────────────────────────────────
# GRPO code lives on the lora-grpo branch (not yet merged to main).
# instructlab-training & mini_trainer from main to pick up the relaxed
# numba>=0.61.2 constraint (merged but not yet released to PyPI).
# Once all released, replace with: pip install "training-hub[grpo]"
RUN pip install --no-cache-dir \
    "instructlab-training @ git+https://github.com/instructlab/training.git@main" \
    "rhai-innovation-mini-trainer @ git+https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer.git@main" \
    "training-hub[grpo] @ git+https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git@lora-grpo"

# ── 5. Ray / NCCL / vLLM environment defaults ───────────────────────
ENV NCCL_DEBUG=WARN \
    TOKENIZERS_PARALLELISM=true \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
    VLLM_LOGGING_LEVEL=WARN \
    VLLM_USE_V1=1 \
    RAY_DEDUP_LOGS=0

USER 1001
WORKDIR /opt/app-root/src

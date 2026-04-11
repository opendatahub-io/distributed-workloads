#!/usr/bin/env python3
"""
MPI GRPO Benchmark — Qwen 2.5 7B + GSM8K + DeepSpeed ZeRO-3

Production benchmark for validating multi-node MPI network performance.
Target hardware: IBMCloud gx3d-160x1792x8h100 (8x H100 80GB per node).
Scaling strategy: 1/2/4 nodes (8-32 GPUs), extrapolated to 16 nodes (128 GPUs).

Launch via mpirun (handled by Kubeflow TrainJob MPI runtime):
    mpirun -x MASTER_ADDR=... -x MASTER_PORT=29500 python train_grpo.py

Environment variables:
    GRPO_MODEL           HuggingFace model ID       (default: Qwen/Qwen2.5-7B-Instruct)
    GRPO_MAX_STEPS       Training steps              (default: 50)
    GRPO_BATCH_SIZE      Per-device batch size       (default: 1)
    GRPO_NUM_GENERATIONS Completions per prompt      (default: 4)
    GRPO_DATASET_SIZE    GSM8K examples to use       (default: all ~7.5K)
    GRPO_MAX_COMPLETION  Max completion tokens       (default: 512)
    GRPO_MAX_PROMPT      Max prompt tokens           (default: 256)
    GRPO_DS_CONFIG       DeepSpeed config path       (default: disabled)
    GRPO_LEARNING_RATE   Learning rate               (default: 5e-7)
    GRPO_LOG_FREQ        Log every N steps           (default: 1)
"""

import os
import re
import socket
import time

# ---------------------------------------------------------------------------
# MPI -> PyTorch environment bridging
# OpenMPI sets OMPI_COMM_WORLD_* variables; map them to the RANK/LOCAL_RANK/
# WORLD_SIZE that PyTorch distributed and HuggingFace Accelerate expect.
# ---------------------------------------------------------------------------
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))

os.environ.setdefault("RANK", str(rank))
os.environ.setdefault("LOCAL_RANK", str(local_rank))
os.environ.setdefault("WORLD_SIZE", str(world_size))
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

# ---------------------------------------------------------------------------
# nvcc stub workaround
# The runtime image (quay.io/ksuta/odh-mpi-cuda:0.0.14) ships CUDA runtime
# libraries but not the full CUDA toolkit. DeepSpeed's module-level import
# runs `nvcc -V` and fails without it. Create a minimal stub.
# ---------------------------------------------------------------------------
_nvcc_dir = "/tmp/cuda_stub/bin"
os.makedirs(_nvcc_dir, exist_ok=True)
_nvcc_path = os.path.join(_nvcc_dir, "nvcc")
if not os.path.exists(_nvcc_path):
    with open(_nvcc_path, "w") as f:
        f.write(
            '#!/bin/bash\n'
            'echo "Cuda compilation tools, release 13.0, V13.0.76"\n'
        )
    os.chmod(_nvcc_path, 0o755)
os.environ["CUDA_HOME"] = "/tmp/cuda_stub"

hostname = socket.gethostname()
if rank == 0:
    print(f"{'=' * 60}", flush=True)
    print("MPI GRPO Benchmark", flush=True)
    print(f"{'=' * 60}", flush=True)
print(
    f"[Rank {rank}/{world_size}] host={hostname} local_rank={local_rank}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Imports (must come after nvcc stub so DeepSpeed import doesn't crash)
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (all tuneable via env vars)
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_STEPS = int(os.environ.get("GRPO_MAX_STEPS", "50"))
BATCH_SIZE = int(os.environ.get("GRPO_BATCH_SIZE", "1"))
NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", "4"))
DATASET_SIZE = os.environ.get("GRPO_DATASET_SIZE", "")
MAX_COMPLETION = int(os.environ.get("GRPO_MAX_COMPLETION", "512"))
MAX_PROMPT = int(os.environ.get("GRPO_MAX_PROMPT", "256"))
DS_CONFIG = os.environ.get("GRPO_DS_CONFIG", "")
LEARNING_RATE = float(os.environ.get("GRPO_LEARNING_RATE", "5e-7"))
LOG_FREQ = int(os.environ.get("GRPO_LOG_FREQ", "1"))


# ---------------------------------------------------------------------------
# Benchmark metrics callback
# ---------------------------------------------------------------------------
class BenchmarkMetricsCallback(TrainerCallback):
    """Tracks per-step timing and GPU memory for benchmark analysis.

    Prints a per-step log line and a final summary table. The first two
    steps are treated as warmup and excluded from the steady-state averages.
    """

    WARMUP_STEPS = 2

    def __init__(self):
        self.step_start = None
        self.step_times = []
        self.train_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        if rank == 0:
            alloc = torch.cuda.memory_allocated(local_rank) / 1e9
            reserved = torch.cuda.memory_reserved(local_rank) / 1e9
            print(
                f"[Benchmark] GPU memory at train start: "
                f"allocated={alloc:.1f}GB reserved={reserved:.1f}GB",
                flush=True,
            )

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start is None:
            return
        elapsed = time.time() - self.step_start
        self.step_times.append(elapsed)

        if rank == 0 and state.global_step % LOG_FREQ == 0:
            alloc = torch.cuda.memory_allocated(local_rank) / 1e9
            reserved = torch.cuda.memory_reserved(local_rank) / 1e9
            samples_per_sec = (BATCH_SIZE * world_size) / elapsed
            print(
                f"[Benchmark] step={state.global_step} "
                f"step_time={elapsed:.2f}s "
                f"samples/s={samples_per_sec:.2f} "
                f"gpu_mem_alloc={alloc:.1f}GB "
                f"gpu_mem_reserved={reserved:.1f}GB",
                flush=True,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if rank != 0 or not self.step_times:
            return

        total_time = time.time() - self.train_start
        warmup = min(self.WARMUP_STEPS, len(self.step_times))
        steady = self.step_times[warmup:]
        avg_step = sum(steady) / len(steady) if steady else 0
        min_step = min(steady) if steady else 0
        max_step = max(steady) if steady else 0
        throughput = (BATCH_SIZE * world_size) / avg_step if avg_step > 0 else 0
        peak_mem = torch.cuda.max_memory_allocated(local_rank) / 1e9

        print(f"\n{'=' * 60}", flush=True)
        print("BENCHMARK RESULTS", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"  Model:              {MODEL_NAME}", flush=True)
        print(f"  World size:         {world_size} GPUs", flush=True)
        print(f"  Batch size/device:  {BATCH_SIZE}", flush=True)
        print(f"  Global batch size:  {BATCH_SIZE * world_size}", flush=True)
        print(f"  Num generations:    {NUM_GENERATIONS}", flush=True)
        print(f"  DeepSpeed config:   {DS_CONFIG or 'disabled'}", flush=True)
        print(f"  Total steps:        {state.global_step}", flush=True)
        print(
            f"  Total time:         {total_time:.1f}s ({total_time / 60:.1f} min)",
            flush=True,
        )
        print(f"  Warmup steps:       {warmup}", flush=True)
        print(f"  Avg step time:      {avg_step:.2f}s (post-warmup)", flush=True)
        print(f"  Min step time:      {min_step:.2f}s", flush=True)
        print(f"  Max step time:      {max_step:.2f}s", flush=True)
        print(f"  Avg throughput:     {throughput:.2f} samples/s", flush=True)
        print(f"  Peak GPU memory:    {peak_mem:.1f}GB", flush=True)
        print(f"{'=' * 60}", flush=True)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def reward_correctness(completions, answer, **kwargs):
    """Binary reward: 1.0 if the model's final number matches GSM8K ground truth."""
    rewards = []
    for completion, gt in zip(completions, answer):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        gt_match = re.search(r"####\s*([\d,]+)", gt)
        gt_num = gt_match.group(1).replace(",", "") if gt_match else ""
        pred_nums = re.findall(r"[\d,]+", content)
        if gt_num and pred_nums and pred_nums[-1].replace(",", "") == gt_num:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_format(completions, **kwargs):
    """Small reward for showing step-by-step reasoning."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        has_steps = bool(
            re.search(r"step|therefore|so |thus |=", content, re.IGNORECASE)
        )
        rewards.append(0.5 if has_steps else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
if rank == 0:
    print("Loading GSM8K dataset...", flush=True)

split = f"train[:{DATASET_SIZE}]" if DATASET_SIZE else "train"
dataset = load_dataset("openai/gsm8k", "main", split=split)

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "then give the final numerical answer after '####'."
)


def format_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
    }


dataset = dataset.map(format_prompt)

if rank == 0:
    print(f"Dataset loaded: {len(dataset)} examples", flush=True)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
if rank == 0:
    print(f"\nTraining configuration:", flush=True)
    print(f"  Model:            {MODEL_NAME}", flush=True)
    print(f"  World size:       {world_size}", flush=True)
    print(f"  Batch/device:     {BATCH_SIZE}", flush=True)
    print(f"  Num generations:  {NUM_GENERATIONS}", flush=True)
    print(f"  Max steps:        {MAX_STEPS}", flush=True)
    print(f"  Max completion:   {MAX_COMPLETION}", flush=True)
    print(f"  Max prompt:       {MAX_PROMPT}", flush=True)
    print(f"  Learning rate:    {LEARNING_RATE}", flush=True)
    print(f"  DeepSpeed:        {DS_CONFIG or 'disabled'}", flush=True)
    print(f"  Dataset size:     {len(dataset)}", flush=True)

training_args = GRPOConfig(
    output_dir="/tmp/grpo_output",
    per_device_train_batch_size=BATCH_SIZE,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION,
    max_prompt_length=MAX_PROMPT,
    max_steps=MAX_STEPS,
    logging_steps=LOG_FREQ,
    learning_rate=LEARNING_RATE,
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="no",
    report_to="none",
    deepspeed=DS_CONFIG if DS_CONFIG else None,
)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    reward_funcs=[reward_correctness, reward_format],
    args=training_args,
    train_dataset=dataset,
    callbacks=[BenchmarkMetricsCallback()],
)

trainer.train()

if rank == 0:
    print("\n=== GRPO TRAINING COMPLETE ===", flush=True)

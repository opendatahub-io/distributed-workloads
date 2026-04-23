#!/usr/bin/env python3
"""
MPI DDP SFT Benchmark — Qwen 2.5 1.5B + GSM8K

Distributed Supervised Fine-Tuning using PyTorch DistributedDataParallel (DDP)
with MPI as the communications backend.

Key constraints for MPI + DDP:
  - Model must use float32 (MPI has no datatype mapping for bfloat16/float16)
  - Model parameter count must be under ~2.1B (MPI int32 count limit)

Launch via mpirun (handled by Kubeflow TrainJob MPI runtime):
    mpirun python train_sft_ddp.py

Environment variables:
    BENCH_MODEL           HuggingFace model ID       (default: Qwen/Qwen2.5-1.5B-Instruct)
    BENCH_DATASET         HuggingFace dataset ID     (default: openai/gsm8k)
    BENCH_DATASET_CONFIG  Dataset configuration       (default: main)
    BENCH_BATCH_SIZE      Per-device batch size       (default: 2)
    BENCH_MAX_SEQ_LENGTH  Max token sequence length   (default: 512)
    BENCH_MAX_STEPS       Training steps              (default: 200)
    BENCH_LEARNING_RATE   AdamW learning rate         (default: 2e-5)
    BENCH_WARMUP_STEPS    Steps excluded from timing  (default: 5)
    BENCH_GRAD_ACCUM      Gradient accumulation steps (default: 1)
    BENCH_LOG_FREQ        Log every N steps           (default: 1)
"""

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATASET_NAME = os.environ.get("BENCH_DATASET", "openai/gsm8k")
DATASET_CONFIG = os.environ.get("BENCH_DATASET_CONFIG", "main")
BATCH_SIZE = int(os.environ.get("BENCH_BATCH_SIZE", "2"))
MAX_SEQ_LENGTH = int(os.environ.get("BENCH_MAX_SEQ_LENGTH", "512"))
MAX_STEPS = int(os.environ.get("BENCH_MAX_STEPS", "200"))
LEARNING_RATE = float(os.environ.get("BENCH_LEARNING_RATE", "2e-5"))
LOG_FREQ = int(os.environ.get("BENCH_LOG_FREQ", "1"))
WARMUP_STEPS = int(os.environ.get("BENCH_WARMUP_STEPS", "5"))
GRADIENT_ACCUMULATION = int(os.environ.get("BENCH_GRAD_ACCUM", "1"))


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
def setup_distributed():
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))

    if not dist.is_mpi_available():
        raise RuntimeError(
            "PyTorch was not built with MPI support. "
            "Verify your image includes a torch build linked against MPI."
        )

    dist.init_process_group(backend="mpi")

    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def format_sft_example(example, tokenizer, max_length):
    """Tokenise a GSM8K example with prompt-masked labels for SFT loss."""
    user_only = [{"role": "user", "content": example["question"]}]
    prompt_text = tokenizer.apply_chat_template(
        user_only, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ],
        tokenize=False,
    )

    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)
    prompt_len = len(prompt_ids["input_ids"])

    labels = encoding["input_ids"].clone()
    labels[0, :prompt_len] = -100
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0),
    }


class SFTCollator:
    """Batch collator that tokenises raw GSM8K rows on-the-fly."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        formatted = [
            format_sft_example(ex, self.tokenizer, self.max_length) for ex in batch
        ]
        return {
            "input_ids": torch.stack([f["input_ids"] for f in formatted]),
            "attention_mask": torch.stack([f["attention_mask"] for f in formatted]),
            "labels": torch.stack([f["labels"] for f in formatted]),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"{'=' * 60}", flush=True)
        print("MPI DDP SFT Benchmark", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(
            f"world_size={world_size}, model={MODEL_NAME}, "
            f"backend={dist.get_backend()}, dtype=float32",
            flush=True,
        )

    # ---- Tokenizer & Model (rank 0 downloads first to avoid cache races) ----
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _ = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        del _
        print("Rank 0: model downloaded and cached", flush=True)
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model loaded: {param_count / 1e9:.2f}B params (float32)", flush=True)

    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print("DDP wrapper applied successfully with MPI backend", flush=True)

    # ---- Dataset ----
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    if rank == 0:
        print(f"Dataset: {len(dataset)} examples", flush=True)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=SFTCollator(tokenizer, MAX_SEQ_LENGTH),
        num_workers=2,
        pin_memory=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ---- Metrics tracking ----
    step_times = []
    losses = []

    if rank == 0:
        print(
            f"\nStarting training: max_steps={MAX_STEPS}, "
            f"batch_size={BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION}",
            flush=True,
        )

    # ---- Training loop ----
    global_step = 0
    start_time = time.time()
    model.train()
    epoch = 0

    while global_step < MAX_STEPS:
        sampler.set_epoch(epoch)
        for batch in dataloader:
            if global_step >= MAX_STEPS:
                break

            step_start = time.time()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRADIENT_ACCUMULATION
            loss.backward()

            if (global_step + 1) % GRADIENT_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            step_time = time.time() - step_start
            step_loss = loss.item() * GRADIENT_ACCUMULATION
            step_times.append(step_time)
            losses.append(step_loss)
            global_step += 1

            if rank == 0 and global_step % LOG_FREQ == 0:
                gpu_alloc = torch.cuda.memory_allocated(local_rank) / 1e9
                gpu_reserved = torch.cuda.memory_reserved(local_rank) / 1e9
                print(
                    f"[DDP-MPI] step={global_step} "
                    f"step_time={step_time:.2f}s "
                    f"loss={step_loss:.4f} "
                    f"samples/s={BATCH_SIZE / step_time:.2f} "
                    f"gpu_mem_alloc={gpu_alloc:.1f}GB "
                    f"gpu_mem_reserved={gpu_reserved:.1f}GB",
                    flush=True,
                )

        epoch += 1

    total_time = time.time() - start_time

    # ---- Summary ----
    if rank == 0:
        post_warmup = step_times[WARMUP_STEPS:] or step_times
        avg_step = sum(post_warmup) / len(post_warmup)
        min_step = min(post_warmup)
        max_step = max(post_warmup)
        throughput = BATCH_SIZE * world_size / avg_step
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        avg_loss = sum(losses[-len(post_warmup):]) / len(post_warmup)

        print(f"\n{'=' * 60}", flush=True)
        print("BENCHMARK RESULTS (DDP + MPI)", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"  Backend:            MPI (via DDP)", flush=True)
        print(f"  Model:              {MODEL_NAME}", flush=True)
        print(f"  Dtype:              float32", flush=True)
        print(f"  World size:         {world_size} GPUs", flush=True)
        print(f"  Batch size/GPU:     {BATCH_SIZE}", flush=True)
        print(f"  Global batch:       {BATCH_SIZE * world_size}", flush=True)
        print(f"  Max seq length:     {MAX_SEQ_LENGTH}", flush=True)
        print(f"  Grad accumulation:  {GRADIENT_ACCUMULATION}", flush=True)
        print(f"  Total steps:        {global_step}", flush=True)
        print(
            f"  Total time:         {total_time:.1f}s ({total_time / 60:.1f} min)",
            flush=True,
        )
        print(f"  Warmup steps:       {WARMUP_STEPS}", flush=True)
        print(f"  Avg step time:      {avg_step:.2f}s (post-warmup)", flush=True)
        print(f"  Min step time:      {min_step:.2f}s", flush=True)
        print(f"  Max step time:      {max_step:.2f}s", flush=True)
        print(f"  Avg throughput:     {throughput:.2f} samples/s", flush=True)
        print(f"  Avg loss:           {avg_loss:.4f} (post-warmup)", flush=True)
        print(f"  Peak GPU memory:    {peak_mem:.1f}GB", flush=True)
        print(f"{'=' * 60}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

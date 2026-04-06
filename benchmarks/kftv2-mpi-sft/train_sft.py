#!/usr/bin/env python3
"""
MPI SFT Benchmark

Production benchmark for validating multi-node MPI network performance.
Target hardware: IBMCloud gx3d-160x1792x8h100 (8x H100 80GB per node).

This script uses Distributed Data Parallel (DDP) with MPI as the backend
rather than the standard NCCL, specifically to benchmark the MPI implementation.
It is adapted from the kfto-sft-llm example.
"""

import os
import socket
import time

def main():
    # ---------------------------------------------------------------------------
    # MPI -> PyTorch environment bridging
    # OpenMPI sets OMPI_COMM_WORLD_* variables; map them to the RANK/LOCAL_RANK/
    # WORLD_SIZE that PyTorch distributed expects.
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
        print("MPI SFT Benchmark (DDP with MPI backend)", flush=True)
        print(f"{'=' * 60}", flush=True)
    print(
        f"[Rank {rank}/{world_size}] host={hostname} local_rank={local_rank}",
        flush=True,
    )

    import torch
    import torch.distributed as dist
    import random
    from datasets import load_dataset
    from transformers import AutoTokenizer, set_seed, TrainerCallback
    from trl import (
        ModelConfig,
        ScriptArguments,
        SFTConfig,
        SFTTrainer,
        TrlParser,
        get_peft_config,
        get_quantization_config,
        get_kbit_device_map,
    )

    # Initialize process group with MPI backend to benchmark MPI
    if not dist.is_initialized():
        dist.init_process_group(backend="mpi")
    torch.cuda.set_device(local_rank)

    # ---------------------------------------------------------------------------
    # Benchmark metrics callback
    # ---------------------------------------------------------------------------
    class BenchmarkMetricsCallback(TrainerCallback):
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

            if rank == 0 and state.global_step % args.logging_steps == 0:
                alloc = torch.cuda.memory_allocated(local_rank) / 1e9
                reserved = torch.cuda.memory_reserved(local_rank) / 1e9
                samples_per_sec = (args.per_device_train_batch_size * world_size) / elapsed
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
            throughput = (args.per_device_train_batch_size * world_size) / avg_step if avg_step > 0 else 0
            peak_mem = torch.cuda.max_memory_allocated(local_rank) / 1e9

            print(f"\n{'=' * 60}", flush=True)
            print("BENCHMARK RESULTS", flush=True)
            print(f"{'=' * 60}", flush=True)
            print(f"  World size:         {world_size} GPUs", flush=True)
            print(f"  Batch size/device:  {args.per_device_train_batch_size}", flush=True)
            print(f"  Global batch size:  {args.per_device_train_batch_size * world_size}", flush=True)
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
    # Parameters
    # ---------------------------------------------------------------------------
    parameters = {
        "model_name_or_path": os.environ.get("SFT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "model_revision": "main",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "use_liger": False,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_modules_to_save": [],
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dataset_name": "gsm8k",
        "dataset_config": "main",
        "dataset_train_split": "train",
        "dataset_test_split": "test",
        "dataset_text_field": "text",
        "dataset_kwargs": {
            "add_special_tokens": False,
            "append_concat_token": False
        },
        "max_seq_length": int(os.environ.get("SFT_MAX_SEQ_LENGTH", "1024")),
        "dataset_batch_size": 1000,
        "packing": False,
        "max_steps": int(os.environ.get("SFT_MAX_STEPS", "50")),
        "per_device_train_batch_size": int(os.environ.get("SFT_BATCH_SIZE", "1")),
        "per_device_eval_batch_size": 1,
        "auto_find_batch_size": False,
        "eval_strategy": "no", # Disable eval for benchmark
        "bf16": True,
        "tf32": False,
        "learning_rate": float(os.environ.get("SFT_LEARNING_RATE", "2.0e-4")),
        "warmup_steps": 10,
        "lr_scheduler_type": "inverse_sqrt",
        "optim": "adamw_torch_fused",
        "max_grad_norm": 1.0,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False
        },
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "activation_checkpointing": True,
            "cpu_ram_efficient_loading": False,
            "sync_module_states": True,
            "use_orig_params": True,
            "limit_all_gathers": False
        },
        "save_strategy": "no",
        "save_total_limit": 1,
        "resume_from_checkpoint": False,
        "log_level": "warning",
        "logging_strategy": "steps",
        "logging_steps": int(os.environ.get("SFT_LOG_FREQ", "1")),
        "report_to": ["none"],
        "output_dir": "/tmp/sft_output",
        "ddp_backend": "mpi", # Explicitly tell Trainer to use MPI backend for DDP
    }

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(parameters)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Model and tokenizer
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing or
                           training_args.fsdp_config.get("activation_checkpointing",
                                                         False) else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        # Models like Llama 3 use a dedicated padding token
        right_pad_id = tokenizer.convert_tokens_to_ids('<|finetune_right_pad_id|>')
        if right_pad_id is not None:
            tokenizer.pad_token = '<|finetune_right_pad_id|>'
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    if rank == 0:
        print("Loading dataset...", flush=True)
    train_dataset = load_dataset(
        path=script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    test_dataset = None
    if training_args.eval_strategy != "no":
        test_dataset = load_dataset(
            path=script_args.dataset_name,
            name=script_args.dataset_config,
            split=script_args.dataset_test_split,
        )

    # Templatize datasets
    def template_dataset(sample):
        messages = [
            {"role": "user", "content": sample['question']},
            {"role": "assistant", "content": sample['answer']},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    train_dataset = train_dataset.map(template_dataset, remove_columns=["question", "answer"])
    if training_args.eval_strategy != "no":
        test_dataset = test_dataset.map(template_dataset, remove_columns=["question", "answer"])

    if rank == 0:
        print(f"Dataset loaded: {len(train_dataset)} examples", flush=True)

    # Training
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        callbacks=[BenchmarkMetricsCallback()],
    )

    if trainer.accelerator.is_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    if rank == 0:
        print("\n=== SFT TRAINING COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()

import os
import logging

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def main():
    model_path = os.environ.get("MODEL_PATH", "/workspace/model")
    dataset_path = os.environ.get("DATASET_PATH", "/workspace/data/train_All_100.jsonl")
    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/output")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info(
        "Starting stock TRL SFTTrainer | WORLD_SIZE: %d, RANK: %d, LOCAL_RANK: %d",
        world_size, global_rank, local_rank,
    )

    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )

    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    logger.info("Dataset loaded: %d samples, columns: %s", len(dataset), dataset.column_names)

    def format_example(ex):
        if "prompt" in ex and "completion" in ex:
            return {"text": f"### Instruction:\n{ex['prompt']}\n\n### Response:\n{ex['completion']}"}
        msgs = ex["messages"]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        asst_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
        return {"text": f"### Instruction:\n{user_msg}\n\n### Response:\n{asst_msg}"}

    dataset = dataset.map(format_example)
    logger.info("Formatted dataset with 'text' field")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_length=512,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        seed=42,
        ddp_timeout=300,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training")
    result = trainer.train()
    logger.info("[STOCK_TRL_SFT] Training complete. Loss=%.4f", result.training_loss)

    if dist.is_initialized():
        dist.barrier()
        logger.info("[GPU%d] Training is finished", global_rank)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

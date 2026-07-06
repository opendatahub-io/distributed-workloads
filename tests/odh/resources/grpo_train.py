"""
GRPO LoRA training script for E2E tests.

Reads all configuration from environment variables with sensible defaults
for the E2E test scenario (small model, minimal iterations).
"""
import os
import json
from training_hub import lora_grpo

model_path = os.environ.get("MODEL_DIR", "/mnt/shared/model")
data_path = os.environ.get("GRPO_DATA_PATH", "/mnt/shared/dataset/train.jsonl")
output_dir = os.environ.get("GRPO_OUTPUT_DIR", "/mnt/shared/grpo-output")
n_gpus = int(os.environ.get("GRPO_N_GPUS", "2"))
num_iterations = int(os.environ.get("GRPO_NUM_ITERATIONS", "1"))
tasks_per_iteration = int(os.environ.get("GRPO_TASKS_PER_ITERATION", "2"))
group_size = int(os.environ.get("GRPO_GROUP_SIZE", "2"))
n_train = int(os.environ.get("GRPO_N_TRAIN", "128"))
n_val = int(os.environ.get("GRPO_N_VAL", "5"))
lora_r = int(os.environ.get("GRPO_LORA_R", "8"))
lora_alpha = int(os.environ.get("GRPO_LORA_ALPHA", "8"))
learning_rate = float(os.environ.get("GRPO_LEARNING_RATE", "1e-5"))
gpu_memory_utilization = float(os.environ.get("GRPO_GPU_MEMORY_UTILIZATION", "0.30"))

print(f"Starting GRPO training: model={model_path}, data={data_path}, gpus={n_gpus}")
print(f"Iterations={num_iterations}, tasks/iter={tasks_per_iteration}, group_size={group_size}")
print(f"n_train={n_train}, n_val={n_val}, lora_r={lora_r}, lora_alpha={lora_alpha}")

result = lora_grpo(
    model_path=model_path,
    data_path=data_path,
    ckpt_output_dir=output_dir,
    backend="verl",
    n_gpus=n_gpus,
    num_iterations=num_iterations,
    tasks_per_iteration=tasks_per_iteration,
    group_size=group_size,
    n_train=n_train,
    n_val=n_val,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    learning_rate=learning_rate,
    gpu_memory_utilization=gpu_memory_utilization,
)

print("GRPO training completed")
print("Training result:", json.dumps(result, default=str))

ckpt_path = result.get("checkpoint_path", os.path.join(output_dir, "checkpoints"))
if not os.path.isdir(ckpt_path):
    raise RuntimeError(f"Checkpoint directory not found: {ckpt_path}")

ckpt_files = []
for root, dirs, files in os.walk(ckpt_path):
    ckpt_files.extend(os.path.join(root, f) for f in files)

if not ckpt_files:
    raise RuntimeError(f"No checkpoint files found in {ckpt_path}")

print(f"Checkpoint validation passed: found {len(ckpt_files)} file(s) in {ckpt_path}")
for f in ckpt_files[:10]:
    print(f"  {f}")

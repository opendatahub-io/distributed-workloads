"""
SFT training script for E2E tests.

Reads the pre-downloaded model from a shared PVC, generates a small synthetic
dataset, runs SFT training, and validates checkpoints.
All configuration is read from environment variables.
"""
import json
import os

from training_hub import sft

MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/shared/model")
DATA_PATH = "/tmp/train_data.jsonl"
CKPT_DIR = os.environ.get("SFT_CKPT_DIR", "/tmp/checkpoints/sft")
N_SAMPLES = int(os.environ.get("SFT_N_SAMPLES", "100"))
MAX_SEQ_LEN = int(os.environ.get("SFT_MAX_SEQ_LEN", "128"))
MAX_TOKENS_PER_GPU = int(os.environ.get("SFT_MAX_TOKENS_PER_GPU", "128"))
EFFECTIVE_BATCH_SIZE = int(os.environ.get("SFT_EFFECTIVE_BATCH_SIZE", "4"))
LEARNING_RATE = float(os.environ.get("SFT_LEARNING_RATE", "1e-5"))
NUM_EPOCHS = int(os.environ.get("SFT_NUM_EPOCHS", "1"))

if not os.path.isdir(MODEL_DIR):
    raise RuntimeError(f"Model directory not found at {MODEL_DIR} - ensure model was downloaded to the shared PVC")
print(f"Using pre-downloaded model from {MODEL_DIR}")

print(f"Generating {N_SAMPLES} synthetic training samples...")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
with open(DATA_PATH, "w") as f:
    for i in range(N_SAMPLES):
        a, b = i % 50, (i * 3 + 7) % 100
        example = {
            "messages": [
                {"role": "user", "content": f"What is {a} + {b}?"},
                {"role": "assistant", "content": f"The answer is {a + b}."},
            ]
        }
        f.write(json.dumps(example) + "\n")
print(f"Dataset written to {DATA_PATH}")

print("Starting SFT training...")
sft(
    model_path=MODEL_DIR,
    data_path=DATA_PATH,
    ckpt_output_dir=CKPT_DIR,
    data_output_dir=f"{CKPT_DIR}/data",
    max_seq_len=MAX_SEQ_LEN,
    max_tokens_per_gpu=MAX_TOKENS_PER_GPU,
    effective_batch_size=EFFECTIVE_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
)
print("SFT training completed")

if not os.path.isdir(CKPT_DIR):
    raise RuntimeError(f"Checkpoint directory not found: {CKPT_DIR}")

ckpt_files = []
for root, dirs, files in os.walk(CKPT_DIR):
    ckpt_files.extend(os.path.join(root, f) for f in files)

if not ckpt_files:
    raise RuntimeError(f"No checkpoint files found in {CKPT_DIR}")

print(f"Checkpoint validation passed: found {len(ckpt_files)} file(s) in {CKPT_DIR}")
for f in ckpt_files[:10]:
    print(f"  {f}")

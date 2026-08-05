"""
LoRA SFT training script for E2E tests.

Reads the pre-downloaded model from a shared PVC, generates a small synthetic
dataset, runs LoRA SFT training, and validates checkpoints.
All configuration is read from environment variables.
"""
import json
import os

from training_hub import lora_sft

MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/shared/model")
DATA_PATH = "/tmp/train_data.jsonl"
CKPT_DIR = os.environ.get("LORA_CKPT_DIR", "/tmp/checkpoints/lora")
N_SAMPLES = int(os.environ.get("LORA_N_SAMPLES", "100"))
LORA_R = int(os.environ.get("LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
MAX_SEQ_LEN = int(os.environ.get("LORA_MAX_SEQ_LEN", "128"))
LEARNING_RATE = float(os.environ.get("LORA_LEARNING_RATE", "1e-4"))
NUM_EPOCHS = int(os.environ.get("LORA_NUM_EPOCHS", "1"))

if not os.path.isdir(MODEL_DIR):
    raise RuntimeError(f"Model directory not found at {MODEL_DIR} - ensure model was downloaded to the shared PVC")
print(f"Using pre-downloaded model from {MODEL_DIR}")

print(f"Generating {N_SAMPLES} synthetic training samples...")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
schemas = [
    "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary DECIMAL)",
    "CREATE TABLE orders (order_id INT, customer_id INT, product VARCHAR, quantity INT)",
    "CREATE TABLE students (id INT, name VARCHAR, grade INT, score DECIMAL)",
]
with open(DATA_PATH, "w") as f:
    for i in range(N_SAMPLES):
        schema = schemas[i % len(schemas)]
        table_name = schema.split("TABLE ")[1].split(" (")[0]
        example = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Given the schema:\n{schema}\nWrite SQL to select all rows from {table_name} where id = {i}.",
                },
                {
                    "role": "assistant",
                    "content": f"SELECT * FROM {table_name} WHERE id = {i};",
                },
            ]
        }
        f.write(json.dumps(example) + "\n")
print(f"Dataset written to {DATA_PATH}")

print("Starting LoRA SFT training...")
lora_sft(
    model_path=MODEL_DIR,
    data_path=DATA_PATH,
    ckpt_output_dir=CKPT_DIR,
    num_epochs=NUM_EPOCHS,
    max_seq_len=MAX_SEQ_LEN,
    learning_rate=LEARNING_RATE,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
)
print("LoRA SFT training completed")

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

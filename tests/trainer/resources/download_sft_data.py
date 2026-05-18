import os
import gzip
import json
import shutil
import random

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
HF_DATASET_ID = "LipengCS/Table-GPT"
HF_DATASET_REVISION = "25754c7a072dffca92e18c56f33832936f53495a"
HF_DATASET_SUBSET = "All"
SUBSET_SIZE = 100


def download_from_s3(dataset_dir, model_dir):
    try:
        import s3fs
    except ImportError:
        print("[download] s3fs not installed, skipping S3")
        return False

    endpoint = os.environ.get("AWS_DEFAULT_ENDPOINT", "")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    bucket = os.environ.get("AWS_STORAGE_BUCKET", "")

    if not all([endpoint, access_key, secret_key, bucket]):
        print("[download] S3 env vars incomplete, skipping S3")
        return False

    allow_insecure = os.environ.get("AWS_ALLOW_INSECURE_ENDPOINT", "").lower() == "true"
    if endpoint.startswith("http://") and not allow_insecure:
        raise RuntimeError("Refusing insecure S3 endpoint over HTTP")

    print(f"[download] S3: endpoint={endpoint}, bucket={bucket}")
    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        endpoint_url=endpoint,
    )

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    s3_prefix = bucket.rstrip("/")
    pulled = 0
    for s3_path in fs.find(s3_prefix):
        rel = s3_path[len(s3_prefix):].lstrip("/")
        if not rel:
            continue

        rel_norm = os.path.normpath(rel).lstrip(os.sep)
        if rel_norm.startswith(".."):
            raise RuntimeError(f"Unsafe object key path: {s3_path}")

        if "table-gpt" in rel.lower() or rel.endswith(".jsonl"):
            dst = os.path.join(dataset_dir, os.path.basename(rel_norm))
        elif "qwen" in rel.lower() or any(rel.endswith(ext) for ext in
                                           [".bin", ".json", ".model", ".safetensors", ".txt"]):
            base = rel_norm.split("Qwen2.5-1.5B-Instruct/")[-1] if "Qwen2.5-1.5B-Instruct" in rel_norm else os.path.basename(rel_norm)
            dst = os.path.join(model_dir, base)
        else:
            print(f"[download] Skipping unrelated file: {rel}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            print(f"[download] S3 get: {s3_path} -> {dst}")
            fs.get(s3_path, dst)
            pulled += 1

            if dst.endswith(".gz"):
                out = os.path.splitext(dst)[0]
                if not os.path.exists(out):
                    with gzip.open(dst, "rb") as fin, open(out, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    os.remove(dst)

    dataset_file = os.path.join(dataset_dir, "train_All_100.jsonl")
    if os.path.exists(dataset_file) and os.listdir(model_dir):
        print(f"[download] S3 download complete: {pulled} files")
        return True

    print("[download] S3 download incomplete, falling back to HuggingFace")
    return False


def download_from_hf(dataset_dir, model_dir):
    dataset_file = os.path.join(dataset_dir, "train_All_100.jsonl")

    if not os.path.exists(dataset_file):
        print(f"[download] Downloading {HF_DATASET_ID} from HuggingFace...")
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET_ID, HF_DATASET_SUBSET, revision=HF_DATASET_REVISION)
        train = ds["train"]
        random.seed(42)
        indices = random.sample(range(len(train)), min(SUBSET_SIZE, len(train)))
        subset = train.select(indices)

        os.makedirs(dataset_dir, exist_ok=True)
        with open(dataset_file, "w") as f:
            for ex in subset:
                f.write(json.dumps(ex) + "\n")
        print(f"[download] Dataset saved: {dataset_file} ({len(subset)} samples)")

    if not os.listdir(model_dir) if os.path.exists(model_dir) else True:
        print(f"[download] Downloading {MODEL_ID} from HuggingFace...")
        from huggingface_hub import snapshot_download

        os.makedirs(model_dir, exist_ok=True)
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=model_dir,
            token=token,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"[download] Model downloaded to {model_dir}")


def main():
    dataset_dir = os.environ.get("DATASET_PATH", "/workspace/data")
    model_dir = os.environ.get("MODEL_PATH", "/workspace/model")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if not download_from_s3(dataset_dir, model_dir):
        download_from_hf(dataset_dir, model_dir)

    dataset_file = os.path.join(dataset_dir, "train_All_100.jsonl")
    if not os.path.exists(dataset_file):
        raise RuntimeError(f"Dataset not found: {dataset_file}")
    if not os.listdir(model_dir):
        raise RuntimeError(f"Model directory empty: {model_dir}")

    print(f"[download] Dataset ready: {dataset_file}")
    print(f"[download] Model ready: {model_dir}")
    print("[download] Initialization complete!")


if __name__ == "__main__":
    main()

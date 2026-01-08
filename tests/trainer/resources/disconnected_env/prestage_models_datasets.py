#!/usr/bin/env python3
"""
Pre-stage HuggingFace models and datasets to MinIO for disconnected trainer tests.

This script downloads models and datasets from HuggingFace Hub and uploads them
to an internal MinIO/S3 storage for use in air-gapped/disconnected environments.

Usage:
    # Set S3/MinIO credentials
    export AWS_DEFAULT_ENDPOINT="https://bastion.example.com:9000"
    export AWS_ACCESS_KEY_ID="minioadmin"
    export AWS_SECRET_ACCESS_KEY="minioadmin"
    export AWS_STORAGE_BUCKET="rhoai-dw"
    
    # Mirror a specific model (auto-generates S3 path: models/distilgpt2/)
    python prestage_models_datasets.py --model distilgpt2
    
    # Mirror with custom S3 path (format: SOURCE:S3_PATH)
    python prestage_models_datasets.py --model distilgpt2:my-models/gpt2
    python prestage_models_datasets.py --dataset yahma/alpaca-cleaned:training-data/alpaca
    
    # Mirror multiple models and datasets
    python prestage_models_datasets.py \
        --model distilgpt2 \
        --model Qwen/Qwen2.5-1.5B-Instruct:models/qwen-1.5b \
        --dataset yahma/alpaca-cleaned
    
    # Use presets (predefined sets for specific tests)
    python prestage_models_datasets.py --preset rhai
    python prestage_models_datasets.py --preset sft
    python prestage_models_datasets.py --preset all

Environment Variables:
    AWS_DEFAULT_ENDPOINT    - MinIO/S3 endpoint URL (required)
    AWS_ACCESS_KEY_ID       - Access key (required)
    AWS_SECRET_ACCESS_KEY   - Secret key (required)
    AWS_STORAGE_BUCKET      - Target bucket name (default: rhoai-dw)
    AWS_DEFAULT_REGION      - S3 region (default: us-east-1)
    VERIFY_SSL              - Verify SSL certificate (default: false)
    DOWNLOAD_DIR            - Local directory for downloads (default: ./downloads)
    SKIP_UPLOAD             - Skip upload, only download (default: false)
    SKIP_DOWNLOAD           - Skip download, only upload existing (default: false)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress SSL warnings for self-signed certs
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# Presets - Predefined sets for specific tests
# Paths match existing MinIO folder structure:
#   - models/<model_name>/
#   - <dataset_name>-datasets/
# =============================================================================

PRESETS: Dict[str, Dict[str, List[str]]] = {
    "rhai": {
        # Format: "source" or "source:custom/path"
        "models": ["distilgpt2:models/distilgpt2"],
        "datasets": ["yahma/alpaca-cleaned:alpaca-cleaned-datasets"],
        "description": "RHAI Features test - small model for quick testing",
    },
    "sft": {
        "models": ["Qwen/Qwen2.5-1.5B-Instruct:models/Qwen2.5-1.5B-Instruct"],
        "datasets": ["LipengCS/Table-GPT:table-gpt-data"],  # Table-GPT dataset for SFT
        "description": "SFT Training test - Qwen model (~3GB) + Table-GPT dataset",
    },
    "osft": {
        "models": ["Qwen/Qwen2.5-1.5B-Instruct:models/Qwen2.5-1.5B-Instruct"],
        "datasets": ["LipengCS/Table-GPT:table-gpt-data"],  # Same dataset as SFT
        "description": "OSFT Training test - Qwen model + Table-GPT dataset",
    },
    "all": {
        "models": [
            "distilgpt2:models/distilgpt2",
            "Qwen/Qwen2.5-1.5B-Instruct:models/Qwen2.5-1.5B-Instruct",
        ],
        "datasets": [
            "yahma/alpaca-cleaned:alpaca-cleaned-datasets",
            "LipengCS/Table-GPT:table-gpt-data",
        ],
        "description": "All models and datasets for trainer tests",
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_env_or_fail(key: str) -> str:
    """Get required environment variable or exit."""
    value = os.environ.get(key)
    if not value:
        print(f"ERROR: Required environment variable '{key}' is not set.")
        sys.exit(1)
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.environ.get(key, str(default)).lower()
    return value in ("true", "1", "yes")


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_s3_client(endpoint: str, access_key: str, secret_key: str, region: str, verify_ssl: bool):
    """Create S3/MinIO client."""
    import boto3
    from botocore.config import Config
    
    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"max_attempts": 3},
    )
    
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=config,
        verify=verify_ssl,
    )


def get_s3_prefix(name: str, resource_type: str) -> str:
    """
    Generate S3 prefix from model/dataset name.
    
    Matches existing MinIO folder structure:
        - Models: models/<model_name>/
        - Datasets: <dataset_name>-datasets/
    """
    # Extract the model/dataset name (last part after /)
    short_name = name.split("/")[-1]
    
    if resource_type == "models":
        return f"models/{short_name}"
    elif resource_type == "datasets":
        # Match existing pattern: fashion-mnist-datasets, mnist-datasets
        return f"{short_name}-datasets"
    else:
        return f"{resource_type}/{short_name}"


def parse_resource_spec(spec: str, resource_type: str) -> Tuple[str, str]:
    """
    Parse resource specification in format 'source' or 'source:s3_path'.
    
    The delimiter is the LAST colon that is followed by a path-like string.
    HuggingFace sources may contain '/' but not ':' in the source name.
    
    Examples:
        'distilgpt2' -> ('distilgpt2', 'models/distilgpt2')
        'distilgpt2:my-models/gpt2' -> ('distilgpt2', 'my-models/gpt2')
        'distilgpt2:models/distilgpt2' -> ('distilgpt2', 'models/distilgpt2')
        'yahma/alpaca-cleaned' -> ('yahma/alpaca-cleaned', 'alpaca-cleaned-datasets')
        'yahma/alpaca-cleaned:alpaca-cleaned-datasets' -> ('yahma/alpaca-cleaned', 'alpaca-cleaned-datasets')
        'yahma/alpaca-cleaned:data/alpaca' -> ('yahma/alpaca-cleaned', 'data/alpaca')
    
    Returns:
        (source_name, s3_prefix)
    """
    # Find the last colon - everything after is the custom path
    last_colon = spec.rfind(":")
    
    if last_colon != -1:
        source = spec[:last_colon]
        s3_path = spec[last_colon + 1:].strip("/")
        
        # Validate: source should not be empty
        if source and s3_path:
            return (source, s3_path)
    
    # No custom path, generate default
    return (spec, get_s3_prefix(spec, resource_type))


def check_s3_prefix_exists(s3_client, bucket: str, s3_prefix: str) -> Tuple[bool, int]:
    """
    Check if an S3 prefix already has objects.
    Returns (exists, file_count).
    """
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        file_count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix + "/", MaxKeys=100):
            contents = page.get("Contents", [])
            file_count += len(contents)
            if file_count > 0:
                # Early exit once we confirm existence
                break
        return (file_count > 0, file_count)
    except Exception:
        return (False, 0)


def list_bucket_tree(s3_client, bucket: str, prefix: str = "") -> Dict[str, any]:
    """
    List all objects in bucket and return as nested dict for tree display.
    """
    tree = {}
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                size = obj["Size"]
                parts = key.split("/")
                current = tree
                for i, part in enumerate(parts):
                    if not part:  # Skip empty parts
                        continue
                    if i == len(parts) - 1:  # Last part is the file
                        current[part] = size
                    else:  # Directory
                        if part not in current:
                            current[part] = {}
                        current = current[part]
    except Exception as e:
        print(f"  Error listing bucket: {e}")
    return tree


def print_tree(tree: Dict, prefix: str = "", is_last: bool = True, indent: str = ""):
    """
    Print tree structure with proper indentation.
    """
    items = sorted(tree.items(), key=lambda x: (isinstance(x[1], dict), x[0]))
    
    for i, (name, value) in enumerate(items):
        is_last_item = i == len(items) - 1
        connector = "└── " if is_last_item else "├── "
        
        if isinstance(value, dict):
            # Directory
            print(f"{indent}{connector}{name}/")
            new_indent = indent + ("    " if is_last_item else "│   ")
            print_tree(value, name, is_last_item, new_indent)
        else:
            # File with size
            size_str = format_size(value)
            print(f"{indent}{connector}{name} ({size_str})")


def upload_directory(s3_client, local_dir: str, bucket: str, s3_prefix: str, skip_existing: bool = True):
    """
    Upload local directory to S3, preserving structure.
    Supports resume by skipping files that already exist in S3 with matching size.
    Shows progress bar with ETA for large uploads.
    """
    from tqdm import tqdm
    from tqdm.utils import CallbackIOWrapper
    
    local_path = Path(local_dir)
    file_count = 0
    skipped_count = 0
    
    # First pass: collect files and calculate total size
    files_to_upload = []
    total_size = 0
    
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_path)
            
            # Skip .git and other hidden files
            if any(part.startswith(".") for part in relative.parts):
                continue
            
            s3_key = f"{s3_prefix}/{relative}"
            file_size = file_path.stat().st_size
            
            # Check if file already exists in S3 with same size (for resume)
            if skip_existing:
                try:
                    response = s3_client.head_object(Bucket=bucket, Key=s3_key)
                    s3_size = response.get("ContentLength", 0)
                    if s3_size == file_size:
                        skipped_count += 1
                        continue  # File already uploaded
                except Exception:
                    pass  # File doesn't exist, proceed with upload
            
            files_to_upload.append((file_path, s3_key, file_size, relative))
            total_size += file_size
    
    if not files_to_upload:
        if skipped_count > 0:
            print(f"    ✅ All {skipped_count} files already uploaded")
        return 0
    
    # Upload with progress bar
    print(f"    Uploading {len(files_to_upload)} files ({format_size(total_size)})")
    
    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="    Progress",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        for file_path, s3_key, file_size, relative in files_to_upload:
            # Upload with callback for progress
            with open(file_path, "rb") as f:
                wrapped_file = CallbackIOWrapper(pbar.update, f, "read")
                s3_client.upload_fileobj(wrapped_file, bucket, s3_key)
            file_count += 1
    
    if skipped_count > 0:
        print(f"    (Skipped {skipped_count} already-uploaded files)")
    
    return file_count


def download_model(model_name: str, local_dir: str):
    """Download model from HuggingFace Hub. Automatically resumes incomplete downloads."""
    from huggingface_hub import snapshot_download, list_repo_files
    
    # Check if already downloaded (has config.json or model files)
    local_path = Path(local_dir)
    if local_path.exists() and any(local_path.glob("*.json")):
        file_count = len(list(local_path.rglob("*")))
        print(f"  Model already downloaded ({file_count} files), verifying...")
    else:
        # Show expected file count
        try:
            repo_files = list_repo_files(model_name)
            print(f"  Downloading model: {model_name} ({len(repo_files)} files)")
        except Exception:
            print(f"  Downloading model: {model_name}")
    
    # HuggingFace's snapshot_download has built-in progress bars with ETA
    snapshot_download(
        model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    
    # Show final size
    total_size = sum(f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file())
    print(f"  ✅ Model ready at: {local_dir} ({format_size(total_size)})")


def download_dataset(dataset_name: str, local_dir: str):
    """Download dataset from HuggingFace Hub. Caches to HF cache dir for resume.
    
    Special handling for specific datasets:
    - LipengCS/Table-GPT: Saved as JSONL files (required by SFT notebook)
    - Others: Saved in Arrow format using save_to_disk()
    """
    from datasets import load_dataset
    import time
    import json
    
    local_path = Path(local_dir)
    
    # Special handling for Table-GPT dataset (SFT tests require JSONL format)
    if "Table-GPT" in dataset_name or "table-gpt" in dataset_name.lower():
        jsonl_file = local_path / "train" / "train_All_100.jsonl"
        if jsonl_file.exists():
            total_size = jsonl_file.stat().st_size
            print(f"  Table-GPT dataset already downloaded ({format_size(total_size)}), skipping...")
            print(f"  ✅ Dataset ready at: {jsonl_file}")
            return
        
        print(f"  Downloading Table-GPT dataset: {dataset_name}")
        print(f"    (Saving as JSONL for SFT notebook compatibility)")
        start_time = time.time()
        
        # Load with "All" config as specified in sft.ipynb
        ds = load_dataset(dataset_name, "All")
        train_data = ds["train"]
        
        # Save first 100 samples as JSONL (matching sft.ipynb behavior)
        subset = train_data.select(range(min(100, len(train_data))))
        
        # Create directory structure
        train_dir = local_path / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        with open(jsonl_file, "w") as f:
            for example in subset:
                f.write(json.dumps(example) + "\n")
        
        elapsed = time.time() - start_time
        total_size = jsonl_file.stat().st_size
        print(f"  ✅ Table-GPT dataset ready at: {jsonl_file} ({format_size(total_size)}, took {elapsed:.1f}s)")
        return
    
    # Standard dataset handling (Arrow format)
    if local_path.exists() and (local_path / "dataset_info.json").exists():
        total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
        print(f"  Dataset already downloaded ({format_size(total_size)}), skipping...")
        print(f"  ✅ Dataset ready at: {local_dir}")
        return
    
    print(f"  Downloading dataset: {dataset_name}")
    print(f"    (HuggingFace datasets library shows download progress)")
    start_time = time.time()
    
    # HuggingFace datasets has its own progress bars
    ds = load_dataset(dataset_name)
    
    print(f"  Saving dataset to disk...")
    ds.save_to_disk(local_dir)
    
    elapsed = time.time() - start_time
    total_size = sum(f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file())
    print(f"  ✅ Dataset ready at: {local_dir} ({format_size(total_size)}, took {elapsed:.1f}s)")


def check_dependencies():
    """Check if required Python packages are installed."""
    missing = []
    
    try:
        import boto3
    except ImportError:
        missing.append("boto3")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        missing.append("huggingface_hub")
    
    try:
        from datasets import load_dataset
    except ImportError:
        missing.append("datasets")
    
    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print("ERROR: Missing required packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def process_model(model_name: str, s3_client, bucket: str, download_dir: str, 
                  skip_download: bool, skip_upload: bool, force: bool = False,
                  custom_s3_prefix: str = None) -> Tuple[str, int, str]:
    """
    Process a single model: download and upload.
    Returns (s3_prefix, file_count, status) where status is 'uploaded', 'skipped', or 'exists'.
    """
    s3_prefix = custom_s3_prefix if custom_s3_prefix else get_s3_prefix(model_name, "models")
    local_dir = os.path.join(download_dir, "models", model_name.replace("/", "_"))
    
    print(f"\n>>> Processing model: {model_name}")
    print(f"    S3 target: s3://{bucket}/{s3_prefix}/")
    
    # Check if already exists in S3
    if s3_client and not skip_upload:
        exists, existing_count = check_s3_prefix_exists(s3_client, bucket, s3_prefix)
        if exists and not force:
            print(f"  ⏭️  SKIPPED: Already exists in S3 ({existing_count}+ files)")
            print(f"     Use --force to overwrite existing data")
            return s3_prefix, existing_count, "exists"
        elif exists and force:
            print(f"  ⚠️  EXISTS: Found {existing_count}+ files, will overwrite (--force)")
    
    # Download
    if not skip_download:
        download_model(model_name, local_dir)
    else:
        print(f"  Skipping download (using existing: {local_dir})")
    
    # Upload
    file_count = 0
    if not skip_upload and s3_client:
        print(f"  Uploading to s3://{bucket}/{s3_prefix}/")
        file_count = upload_directory(s3_client, local_dir, bucket, s3_prefix)
        print(f"  ✅ Uploaded {file_count} files")
    
    return s3_prefix, file_count, "uploaded"


def process_dataset(dataset_name: str, s3_client, bucket: str, download_dir: str,
                    skip_download: bool, skip_upload: bool, force: bool = False,
                    custom_s3_prefix: str = None) -> Tuple[str, int, str]:
    """
    Process a single dataset: download and upload.
    Returns (s3_prefix, file_count, status) where status is 'uploaded', 'skipped', or 'exists'.
    """
    s3_prefix = custom_s3_prefix if custom_s3_prefix else get_s3_prefix(dataset_name, "datasets")
    local_dir = os.path.join(download_dir, "datasets", dataset_name.replace("/", "_"))
    
    print(f"\n>>> Processing dataset: {dataset_name}")
    print(f"    S3 target: s3://{bucket}/{s3_prefix}/")
    
    # Check if already exists in S3
    if s3_client and not skip_upload:
        exists, existing_count = check_s3_prefix_exists(s3_client, bucket, s3_prefix)
        if exists and not force:
            print(f"  ⏭️  SKIPPED: Already exists in S3 ({existing_count}+ files)")
            print(f"     Use --force to overwrite existing data")
            return s3_prefix, existing_count, "exists"
        elif exists and force:
            print(f"  ⚠️  EXISTS: Found {existing_count}+ files, will overwrite (--force)")
    
    # Download
    if not skip_download:
        download_dataset(dataset_name, local_dir)
    else:
        print(f"  Skipping download (using existing: {local_dir})")
    
    # Upload
    file_count = 0
    if not skip_upload and s3_client:
        print(f"  Uploading to s3://{bucket}/{s3_prefix}/")
        file_count = upload_directory(s3_client, local_dir, bucket, s3_prefix)
        print(f"  ✅ Uploaded {file_count} files")
    
    return s3_prefix, file_count, "uploaded"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-stage HuggingFace models and datasets to MinIO for disconnected tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Resource selection
    parser.add_argument(
        "--model", "-m",
        action="append",
        dest="models",
        metavar="MODEL[:PATH]",
        help="HuggingFace model ID to mirror. Format: 'model_id' or 'model_id:custom/s3/path'. Can be specified multiple times."
    )
    parser.add_argument(
        "--dataset", "-d",
        action="append",
        dest="datasets",
        metavar="DATASET[:PATH]",
        help="HuggingFace dataset ID to mirror. Format: 'dataset_id' or 'dataset_id:custom/s3/path'. Can be specified multiple times."
    )
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        help="Use a predefined preset (rhai, sft, osft, all)"
    )
    
    # Actions
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and their contents"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be mirrored without actually doing it"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force upload even if model/dataset already exists in S3"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if models/datasets exist in S3, don't download or upload"
    )
    
    args = parser.parse_args()
    
    # List presets mode
    if args.list_presets:
        print("Available presets:\n")
        for name, preset in PRESETS.items():
            print(f"  {name}:")
            print(f"    Description: {preset['description']}")
            if preset['models']:
                print(f"    Models: {', '.join(preset['models'])}")
            if preset['datasets']:
                print(f"    Datasets: {', '.join(preset['datasets'])}")
            print()
        return
    
    # Collect models and datasets to process as (source, s3_path) tuples
    models_to_process: List[Tuple[str, str]] = []
    datasets_to_process: List[Tuple[str, str]] = []
    
    # From preset (presets use default paths)
    if args.preset:
        preset = PRESETS[args.preset]
        for m in preset["models"]:
            models_to_process.append(parse_resource_spec(m, "models"))
        for d in preset["datasets"]:
            datasets_to_process.append(parse_resource_spec(d, "datasets"))
        print(f"Using preset: {args.preset} - {preset['description']}")
    
    # From CLI arguments (may include custom paths)
    if args.models:
        for m in args.models:
            models_to_process.append(parse_resource_spec(m, "models"))
    if args.datasets:
        for d in args.datasets:
            datasets_to_process.append(parse_resource_spec(d, "datasets"))
    
    # Remove duplicates while preserving order (based on source name)
    seen_models = set()
    unique_models = []
    for source, path in models_to_process:
        if source not in seen_models:
            seen_models.add(source)
            unique_models.append((source, path))
    models_to_process = unique_models
    
    seen_datasets = set()
    unique_datasets = []
    for source, path in datasets_to_process:
        if source not in seen_datasets:
            seen_datasets.add(source)
            unique_datasets.append((source, path))
    datasets_to_process = unique_datasets
    
    # Validate we have something to do
    if not models_to_process and not datasets_to_process:
        print("ERROR: No models or datasets specified.")
        print("\nUsage examples:")
        print("  python prestage_models_datasets.py --model distilgpt2")
        print("  python prestage_models_datasets.py --model distilgpt2:custom/path")
        print("  python prestage_models_datasets.py --dataset yahma/alpaca-cleaned")
        print("  python prestage_models_datasets.py --preset rhai")
        print("\nUse --list-presets to see available presets.")
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Would mirror the following:")
        print("=" * 60)
        if models_to_process:
            print("\nModels:")
            for source, s3_path in models_to_process:
                print(f"  {source} -> {s3_path}/")
        if datasets_to_process:
            print("\nDatasets:")
            for source, s3_path in datasets_to_process:
                print(f"  {source} -> {s3_path}/")
        return
    
    # Check mode - only verify if resources exist in S3
    if args.check:
        check_dependencies()
        endpoint = get_env_or_fail("AWS_DEFAULT_ENDPOINT")
        access_key = get_env_or_fail("AWS_ACCESS_KEY_ID")
        secret_key = get_env_or_fail("AWS_SECRET_ACCESS_KEY")
        bucket = os.environ.get("AWS_STORAGE_BUCKET", "rhoai-dw")
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        verify_ssl = get_env_bool("VERIFY_SSL", False)
        
        print("\n" + "=" * 60)
        print("CHECK MODE - S3 Bucket Contents:")
        print("=" * 60)
        
        s3 = get_s3_client(endpoint, access_key, secret_key, region, verify_ssl)
        
        # Display full bucket tree structure
        print(f"\n📦 s3://{bucket}/")
        tree = list_bucket_tree(s3, bucket)
        if tree:
            print_tree(tree)
        else:
            print("  (empty)")
        
        # Summary of requested resources
        print("\n" + "-" * 60)
        print("Requested Resources Status:")
        print("-" * 60)
        
        all_exist = True
        if models_to_process:
            print("\nModels:")
            for source, s3_path in models_to_process:
                exists, count = check_s3_prefix_exists(s3, bucket, s3_path)
                status = f"✅ EXISTS ({count}+ files)" if exists else "❌ NOT FOUND"
                print(f"  {source} -> {s3_path}/: {status}")
                if not exists:
                    all_exist = False
        
        if datasets_to_process:
            print("\nDatasets:")
            for source, s3_path in datasets_to_process:
                exists, count = check_s3_prefix_exists(s3, bucket, s3_path)
                status = f"✅ EXISTS ({count}+ files)" if exists else "❌ NOT FOUND"
                print(f"  {source} -> {s3_path}/: {status}")
                if not exists:
                    all_exist = False
        
        print("\n" + "-" * 60)
        if all_exist:
            print("✅ All requested models and datasets exist in S3")
        else:
            print("❌ Some models or datasets are missing from S3")
            print("   Run without --check to upload them")
        return
    
    # Check dependencies
    check_dependencies()
    
    # Get configuration from environment (same env vars as v1 tests)
    endpoint = get_env_or_fail("AWS_DEFAULT_ENDPOINT")
    access_key = get_env_or_fail("AWS_ACCESS_KEY_ID")
    secret_key = get_env_or_fail("AWS_SECRET_ACCESS_KEY")
    bucket = os.environ.get("AWS_STORAGE_BUCKET", "rhoai-dw")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    verify_ssl = get_env_bool("VERIFY_SSL", False)
    download_dir = os.environ.get("DOWNLOAD_DIR", "./downloads")
    skip_upload = get_env_bool("SKIP_UPLOAD", False)
    skip_download = get_env_bool("SKIP_DOWNLOAD", False)
    
    print("\n" + "=" * 60)
    print("Pre-staging Models and Datasets for Disconnected Environment")
    print("=" * 60)
    print(f"AWS_DEFAULT_ENDPOINT: {endpoint}")
    print(f"AWS_STORAGE_BUCKET: {bucket}")
    print(f"AWS_DEFAULT_REGION: {region}")
    print(f"Download Dir: {download_dir}")
    print(f"Skip Download: {skip_download}")
    print(f"Skip Upload: {skip_upload}")
    
    if models_to_process:
        print(f"\nModels to mirror: {len(models_to_process)}")
        for source, s3_path in models_to_process:
            print(f"  - {source} -> {s3_path}/")
    
    if datasets_to_process:
        print(f"\nDatasets to mirror: {len(datasets_to_process)}")
        for source, s3_path in datasets_to_process:
            print(f"  - {source} -> {s3_path}/")
    
    print("=" * 60)
    
    # Create S3 client and test connection
    s3 = None
    if not skip_upload:
        print("\nConnecting to S3/MinIO...")
        s3 = get_s3_client(endpoint, access_key, secret_key, region, verify_ssl)
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"Connected! Bucket '{bucket}' exists.")
        except Exception as e:
            print(f"ERROR: Cannot access bucket '{bucket}': {e}")
            sys.exit(1)
    
    # Create download directory
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    # Track results
    results: Dict[str, List[Tuple[str, str, int, str]]] = {"models": [], "datasets": []}
    
    # Process models
    if models_to_process:
        print("\n" + "=" * 60)
        print("MODELS")
        print("=" * 60)
        
        for model_source, custom_s3_path in models_to_process:
            s3_prefix, file_count, status = process_model(
                model_source, s3, bucket, download_dir, skip_download, skip_upload, 
                args.force, custom_s3_prefix=custom_s3_path
            )
            results["models"].append((model_source, s3_prefix, file_count, status))
    
    # Process datasets
    if datasets_to_process:
        print("\n" + "=" * 60)
        print("DATASETS")
        print("=" * 60)
        
        for dataset_source, custom_s3_path in datasets_to_process:
            s3_prefix, file_count, status = process_dataset(
                dataset_source, s3, bucket, download_dir, skip_download, skip_upload, 
                args.force, custom_s3_prefix=custom_s3_path
            )
            results["datasets"].append((dataset_source, s3_prefix, file_count, status))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    uploaded_count = 0
    skipped_count = 0
    
    print(f"\nResults for s3://{bucket}/:")
    if results["models"]:
        print("\n  Models:")
        for name, prefix, count, status in results["models"]:
            if status == "exists":
                icon = "⏭️ "
                skipped_count += 1
            else:
                icon = "✅"
                uploaded_count += 1
            print(f"    {icon} {name} -> {prefix}/ ({count} files) [{status}]")
    
    if results["datasets"]:
        print("\n  Datasets:")
        for name, prefix, count, status in results["datasets"]:
            if status == "exists":
                icon = "⏭️ "
                skipped_count += 1
            else:
                icon = "✅"
                uploaded_count += 1
            print(f"    {icon} {name} -> {prefix}/ ({count} files) [{status}]")
    
    print(f"\n  Uploaded: {uploaded_count}, Skipped (already exists): {skipped_count}")
    
    print("\n" + "-" * 60)
    print("To use in tests, ensure these environment variables are set:")
    print(f"  export AWS_DEFAULT_ENDPOINT=\"{endpoint}\"")
    print(f"  export AWS_ACCESS_KEY_ID=\"<access_key>\"")
    print(f"  export AWS_SECRET_ACCESS_KEY=\"<secret_key>\"")
    print(f"  export AWS_STORAGE_BUCKET=\"{bucket}\"")


if __name__ == "__main__":
    main()

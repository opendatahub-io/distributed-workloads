#!/usr/bin/env python3
"""
Install kubeflow SDK with S3 fallback for disconnected environments.

Usage:
    python install_kubeflow.py

Tries Red Hat PyPI index first (kubeflow is not on public PyPI),
falls back to S3 wheel if PyPI fails.

Environment variables:
    - GPU_TYPE: Accelerator type (cpu, nvidia/cuda, amd/rocm) - determines which Red Hat PyPI index to use
    - KUBEFLOW_REQUIRED_VERSION: Required version (default: 0.2.1+rhai2)

S3 fallback env vars:
    - AWS_DEFAULT_ENDPOINT
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_STORAGE_BUCKET
    - KUBEFLOW_WHEEL_S3_KEY (optional, default: wheels/kubeflow-0.2.1+rhai2-py3-none-any.whl)
"""

import subprocess
import sys
import os
import warnings

# Suppress SSL warnings for self-signed certs in disconnected environments
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*InsecureRequestWarning.*")


def get_required_version():
    """Get required kubeflow version from env or use default."""
    return os.environ.get("KUBEFLOW_REQUIRED_VERSION", "0.2.1+rhai2")


def get_rhai_pypi_index() -> str:
    """
    Get the appropriate Red Hat PyPI index URL based on accelerator type.
    
    kubeflow package is NOT on public PyPI - only on Red Hat indexes:
    - CPU: https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3/cpu-ubi9/simple/
    - CUDA: https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3/cuda12.9-ubi9/simple/
    - ROCm: https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3/rocm6.4-ubi9/simple/
    """
    gpu_type = os.environ.get("GPU_TYPE", "cpu").lower()
    base = "https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3"
    
    if "nvidia" in gpu_type or "cuda" in gpu_type:
        return f"{base}/cuda12.9-ubi9/simple/"
    elif "amd" in gpu_type or "rocm" in gpu_type:
        return f"{base}/rocm6.4-ubi9/simple/"
    else:
        return f"{base}/cpu-ubi9/simple/"


def verify_kubeflow_version():
    """Verify kubeflow version matches required version (handles leading 'v')."""
    required_version = get_required_version()
    try:
        import kubeflow
        installed_version = getattr(kubeflow, "__version__", "unknown")
        # Strip leading 'v' for comparison (v0.2.1 == 0.2.1)
        if installed_version.lstrip("v") == required_version.lstrip("v"):
            print(f"Verified: kubeflow {installed_version} matches required {required_version}")
            return True
        else:
            print(f"Version mismatch: installed '{installed_version}', required '{required_version}'")
            return False
    except ImportError:
        print("kubeflow module not found")
        return False


def install_from_pypi():
    """Install kubeflow from Red Hat PyPI index (not available on public PyPI)."""
    required_version = get_required_version()
    rhai_index = get_rhai_pypi_index()
    
    # Step 1: Install kubeflow dependencies from public PyPI
    # (kubeflow requires: pydantic, kubernetes, kubeflow-trainer-api, kubeflow-katib-api)
    print("Installing kubeflow dependencies from public PyPI...")
    deps_cmd = [
        sys.executable, "-m", "pip", "install", "--quiet",
        "pydantic>=2.10.0", "kubernetes>=27.2.0", 
        "kubeflow-trainer-api>=2.0.0", "kubeflow-katib-api>=0.19.0"
    ]
    deps_result = subprocess.run(deps_cmd, capture_output=True, text=True)
    if deps_result.returncode != 0:
        print(f"Failed to install dependencies: {deps_result.stderr}")
        return False
    
    # Step 2: Install kubeflow SDK from Red Hat index (with --no-deps since deps are installed)
    print(f"Installing kubeflow=={required_version} from {rhai_index}")
    cmd = [
        sys.executable, "-m", "pip", "install", "--quiet",
        "--index-url", rhai_index,
        "--trusted-host", "console.redhat.com",
        "--no-deps",  # Dependencies already installed from public PyPI
        f"kubeflow=={required_version}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        if verify_kubeflow_version():
            print("Successfully installed kubeflow SDK from Red Hat PyPI")
            return True
        else:
            print("Installed kubeflow version doesn't match, will try S3 fallback")
            return False
    print(f"PyPI install failed: {result.stderr}")
    return False


def install_from_s3():
    """Download wheel from S3 and install."""
    print("Falling back to S3 wheel...")
    
    endpoint = os.environ.get("AWS_DEFAULT_ENDPOINT", "")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    bucket = os.environ.get("AWS_STORAGE_BUCKET", "")
    
    if not all([endpoint, access_key, secret_key, bucket]):
        print("S3 credentials not configured, cannot fallback to S3")
        return False
    
    try:
        import boto3
        from botocore.config import Config
        import urllib3
        urllib3.disable_warnings()
        
        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        )
        
        endpoint_url = endpoint if endpoint.startswith("http") else f"https://{endpoint}"
        
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config,
            verify=False,
        )
        
        # Get wheel path from env var or use default
        wheel_key = os.environ.get(
            "KUBEFLOW_WHEEL_S3_KEY",
            "wheels/kubeflow-0.2.1+rhai2-py3-none-any.whl"
        )
        # Preserve original wheel filename (pip requires valid wheel name)
        wheel_filename = wheel_key.split("/")[-1]
        local_path = f"/tmp/{wheel_filename}"
        
        print(f"Downloading s3://{bucket}/{wheel_key} to {local_path}")
        s3.download_file(bucket, wheel_key, local_path)
        
        # Install the wheel
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", local_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("Successfully installed kubeflow from S3 wheel")
            return True
        
        print(f"Wheel install failed: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"S3 fallback failed: {e}")
        return False


def install_from_git():
    """Install kubeflow from git repo (for unreleased versions not yet on Red Hat PyPI)."""
    git_url = os.environ.get(
        "KUBEFLOW_GIT_URL",
        "kubeflow @ git+https://github.com/opendatahub-io/kubeflow-sdk.git"
    )
    print(f"Installing kubeflow from git: {git_url}")
    cmd = [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", git_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Successfully installed kubeflow from git")
        return True
    print(f"Git install failed: {result.stderr}")
    return False


def main():
    # If KUBEFLOW_INSTALL_FROM_GIT is set, install from git (for unreleased versions)
    if os.environ.get("KUBEFLOW_INSTALL_FROM_GIT", "").lower() == "true":
        if install_from_git():
            return 0
        print("WARNING: Git install failed, falling back to PyPI/S3")

    # Try PyPI first
    if install_from_pypi():
        return 0
    
    # Fallback to S3
    if install_from_s3():
        return 0
    
    print("ERROR: Could not install kubeflow from PyPI or S3")
    return 1


if __name__ == "__main__":
    sys.exit(main())


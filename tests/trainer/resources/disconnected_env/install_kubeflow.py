#!/usr/bin/env python3
"""
Install kubeflow SDK with S3 fallback for disconnected environments.

Usage:
    python install_kubeflow.py

Tries PyPI first, falls back to S3 wheel if PyPI fails.
Requires env vars for S3 fallback:
    - AWS_DEFAULT_ENDPOINT
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_STORAGE_BUCKET
    - KUBEFLOW_WHEEL_S3_KEY (optional, default: wheels/kubeflow-0.2.1+rhai0-py3-none-any.whl)
    - KUBEFLOW_REQUIRED_VERSION (optional, default: 0.2.1) - minimum version required
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
    return os.environ.get("KUBEFLOW_REQUIRED_VERSION", "0.2.1+rhai0")


def verify_kubeflow_version():
    """Verify kubeflow version matches required version exactly."""
    required_version = get_required_version()
    try:
        import kubeflow
        installed_version = getattr(kubeflow, "__version__", "unknown")
        # Exact match required (e.g., 0.2.1+rhai0 must match 0.2.1+rhai0)
        if installed_version == required_version:
            print(f"Verified: kubeflow version {installed_version} matches required {required_version}")
            return True
        else:
            print(f"Version mismatch: installed '{installed_version}', required '{required_version}'")
            return False
    except ImportError:
        print("kubeflow module not found")
        return False


def install_from_pypi():
    """Try installing kubeflow from PyPI."""
    required_version = get_required_version()
    print(f"Attempting to install kubeflow=={required_version} from PyPI...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", f"kubeflow=={required_version}"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        # Verify the correct version is installed
        if verify_kubeflow_version():
            print("Successfully installed kubeflow SDK from PyPI")
            return True
        else:
            print("PyPI kubeflow package version doesn't match, will try S3 fallback")
            return False
    print(f"PyPI install failed (version {required_version} not found): {result.stderr}")
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
            "wheels/kubeflow-0.2.1+rhai0-py3-none-any.whl"
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


def main():
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


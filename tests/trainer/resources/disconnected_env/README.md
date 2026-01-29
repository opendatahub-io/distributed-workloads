# Disconnected Environment Setup for Trainer Tests

This guide covers setting up and running trainer v2/SDK tests in a disconnected OpenShift environment (no internet access from cluster nodes).

## Prerequisites

- OpenShift cluster with RHOAI/ODH installed
- Bastion host with internet access for mirroring
- MinIO/S3-compatible storage accessible from cluster
- PyPI mirror (Nexus/DevPI) accessible from cluster (optional - see note below)
- Image registry accessible from cluster (e.g., Quay, internal registry)
- **Go 1.24** (not 1.25+) for running tests
- **Notebook image with Python 3.9+** (required for kubeflow-trainer-api>=2.0.0)

> **Note on PyPI**: The `kubeflow` package is **not on public PyPI** - it's only available on Red Hat indexes:
> - CPU: `https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3/cpu-ubi9/simple/`
> - CUDA: `https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3/cuda12.9-ubi9/simple/`
> - ROCm: `https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3/rocm6.4-ubi9/simple/`
>
> Tests automatically use the correct Red Hat index based on accelerator type. For fully disconnected environments, mirror these indexes or use the S3 wheel fallback.

## Overview

Disconnected environments require:
1. **Container images** mirrored to internal registry
2. **Models/datasets** pre-staged to S3/MinIO
3. **Python packages** available via PyPI mirror or S3 (tests auto-use Red Hat indexes when connected)
4. **Environment variables** configured for tests

---

## Available Test Cases

### RHAI Features Tests

| Test Name | Description | Resources |
|-----------|-------------|-----------|
| `TestRhaiTrainingProgressionCPU` | Progression tracking on CPU | 2 nodes, 2 CPUs each |
| `TestRhaiJitCheckpointingCPU` | JIT checkpoint save/resume on CPU | 2 nodes, 2 CPUs each |
| `TestRhaiFeaturesCPU` | All RHAI features combined (CPU) | 2 nodes, 2 CPUs each |
| `TestRhaiTrainingProgressionCuda` | Progression tracking on NVIDIA GPU | 2 nodes, 1 GPU each |
| `TestRhaiJitCheckpointingCuda` | JIT checkpoint on NVIDIA GPU | 2 nodes, 1 GPU each |
| `TestRhaiFeaturesCuda` | All RHAI features (NVIDIA GPU) | 2 nodes, 1 GPU each |
| `TestRhaiTrainingProgressionRocm` | Progression tracking on AMD GPU | 2 nodes, 1 GPU each |
| `TestRhaiJitCheckpointingRocm` | JIT checkpoint on AMD GPU | 2 nodes, 1 GPU each |
| `TestRhaiFeaturesRocm` | All RHAI features (AMD GPU) | 2 nodes, 1 GPU each |

### Multi-GPU Tests

| Test Name | Description | Resources |
|-----------|-------------|-----------|
| `TestRhaiTrainingProgressionMultiGpuCuda` | Multi-GPU progression (NVIDIA) | 2 nodes, 2 GPUs each |
| `TestRhaiJitCheckpointingMultiGpuCuda` | Multi-GPU checkpoint (NVIDIA) | 2 nodes, 2 GPUs each |
| `TestRhaiFeaturesMultiGpuCuda` | All features multi-GPU (NVIDIA) | 2 nodes, 2 GPUs each |
| `TestRhaiTrainingProgressionMultiGpuRocm` | Multi-GPU progression (AMD) | 2 nodes, 2 GPUs each |
| `TestRhaiJitCheckpointingMultiGpuRocm` | Multi-GPU checkpoint (AMD) | 2 nodes, 2 GPUs each |
| `TestRhaiFeaturesMultiGpuRocm` | All features multi-GPU (AMD) | 2 nodes, 2 GPUs each |

---

## Step 1: Mirror Container Images

### 1.1 Required Images

Get image digests for the notebook and training images:

```bash
# Notebook image - MUST use version with Python 3.9+ (2024.2 or later)
# Python 3.9+ is required for kubeflow-trainer-api>=2.0.0
skopeo inspect docker://quay.io/modh/odh-generic-data-science-notebook:2024.2 | jq -r '.Digest'

# Training runtime images are built into the cluster
```

### 1.2 Mirror Images

```bash
# Mirror notebook image
skopeo copy --all \
  docker://quay.io/modh/odh-generic-data-science-notebook@sha256:<digest> \
  docker://<your-registry>/modh/odh-generic-data-science-notebook@sha256:<digest>
```

### 1.3 Configure ImageDigestMirrorSet (if needed)

```yaml
apiVersion: config.openshift.io/v1
kind: ImageDigestMirrorSet
metadata:
  name: notebook-mirror
spec:
  imageDigestMirrors:
    - mirrors:
        - <your-registry>/modh
      source: quay.io/modh
```

---

## Step 2: Pre-Stage Models and Datasets to S3/MinIO

### 2.1 Install Required Packages (on bastion with internet)

```bash
pip install boto3 huggingface_hub datasets tqdm urllib3
```

### 2.2 Set Environment Variables

```bash
export AWS_DEFAULT_ENDPOINT="https://<minio-endpoint>:9000"
export AWS_ACCESS_KEY_ID="<access-key>"
export AWS_SECRET_ACCESS_KEY="<secret-key>"
export AWS_STORAGE_BUCKET="<bucket-name>"
export HF_TOKEN="<huggingface-token>"  # Optional, for gated models
```

### 2.3 Run Pre-Stage Script

The `prestage_models_datasets.py` script downloads models/datasets from HuggingFace and uploads to S3.

```bash
cd tests/trainer/resources/disconnected_env

# List available presets
python3 prestage_models_datasets.py --list-presets

# Pre-stage for RHAI tests (distilgpt2 + alpaca-cleaned)
python3 prestage_models_datasets.py --preset rhai

# Pre-stage custom model/dataset
python3 prestage_models_datasets.py \
  --model "distilgpt2" \
  --dataset "yahma/alpaca-cleaned"

# Force re-upload (overwrites existing)
python3 prestage_models_datasets.py --preset rhai --force

# Check what exists in S3
python3 prestage_models_datasets.py --preset rhai --check
```

### 2.4 Verify S3 Contents

After pre-staging, verify files are uploaded:

```bash
# Using aws CLI or mc (MinIO client)
mc ls myminio/<bucket>/models/distilgpt2/

# Expected files for distilgpt2:
# - config.json
# - generation_config.json
# - merges.txt
# - model.safetensors (or pytorch_model.bin)
# - tokenizer.json
# - tokenizer_config.json
# - vocab.json
```

### 2.5 S3 Bucket Structure

The prestage script creates this structure:

```
<bucket>/
├── models/
│   └── distilgpt2/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── vocab.json
│       └── ...
├── alpaca-cleaned-datasets/
│   ├── train/
│   │   └── data-*.arrow
│   ├── dataset_dict.json
│   └── dataset_info.json
└── wheels/
    └── kubeflow-0.2.1+rhai0-py3-none-any.whl  # See Step 3
```

---

## Step 3: Prepare Kubeflow SDK Wheel

The RHAI features require a specific version of the Kubeflow SDK that may not be on public PyPI.

### 3.1 Download SDK Wheel

```bash
# Clone and build the wheel
git clone https://github.com/opendatahub-io/kubeflow-sdk.git
cd kubeflow-sdk
git checkout v0.2.1+rhai0
pip install build
python -m build --wheel

# Or download pre-built wheel
pip download "kubeflow @ git+https://github.com/opendatahub-io/kubeflow-sdk.git@v0.2.1+rhai0" \
  --no-deps -d /tmp/wheels
```

### 3.2 Upload to S3

```bash
# Upload wheel to S3
mc cp kubeflow-0.2.1+rhai0-py3-none-any.whl myminio/<bucket>/wheels/
```

### 3.3 (Alternative) Upload to PyPI Mirror

If you have a PyPI mirror (Nexus, DevPI):

```bash
twine upload --repository-url https://<pypi-mirror>/repository/pypi/ \
  dist/kubeflow-0.2.1+rhai0-py3-none-any.whl
```

---

## Step 4: Configure Test Environment Variables

### 4.1 Required Variables

```bash
# Notebook configuration
export NOTEBOOK_USER_NAME="<openshift-user>"
export NOTEBOOK_USER_PASSWORD="<password>"
export NOTEBOOK_IMAGE="quay.io/modh/odh-generic-data-science-notebook@sha256:<digest>"

# S3/MinIO configuration
export AWS_DEFAULT_ENDPOINT="https://<minio-endpoint>:9000"
export AWS_ACCESS_KEY_ID="<access-key>"
export AWS_SECRET_ACCESS_KEY="<secret-key>"
export AWS_STORAGE_BUCKET="<bucket-name>"

# PyPI mirror (optional but recommended)
export PIP_INDEX_URL="https://<pypi-mirror>/simple/"
export PIP_TRUSTED_HOST="<pypi-mirror-hostname>"

# Test timeout (increase for slow environments)
export TEST_TIMEOUT_LONG="15m"
```

### 4.2 Optional Variables

```bash
# Model/dataset S3 paths (defaults shown)
export MODEL_S3_PREFIX="models/distilgpt2"
export DATASET_S3_PREFIX="alpaca-cleaned-datasets"

# Kubeflow wheel S3 path (default shown)
export KUBEFLOW_WHEEL_S3_KEY="wheels/kubeflow-0.2.1+rhai0-py3-none-any.whl"
export KUBEFLOW_REQUIRED_VERSION="0.2.1"

# SSL verification (set to "false" for self-signed certs on S3)
export VERIFY_SSL="false"
```

---

## Step 5: Run Tests

### 5.1 Verify Go Version

The tests require Go 1.24. If you have multiple Go versions:

```bash
# macOS with Homebrew
export PATH=/opt/homebrew/opt/go@1.24/bin:$PATH
export GOROOT=/opt/homebrew/opt/go@1.24/libexec

# Verify
go version  # Should show go1.24.x
```

### 5.2 RHAI Features Test (CPU)

```bash
NOTEBOOK_USER_NAME="htpasswd-cluster-admin-user" \
NOTEBOOK_USER_PASSWORD="<password>" \
NOTEBOOK_IMAGE="quay.io/modh/odh-generic-data-science-notebook@sha256:<digest>" \
AWS_DEFAULT_ENDPOINT="https://<minio-endpoint>:9000" \
AWS_ACCESS_KEY_ID="<access-key>" \
AWS_SECRET_ACCESS_KEY="<secret-key>" \
AWS_STORAGE_BUCKET="<bucket-name>" \
PIP_INDEX_URL="https://<pypi-mirror>/simple/" \
PIP_TRUSTED_HOST="<pypi-mirror-hostname>" \
go test -v ./tests/trainer -run TestRhaiTrainingProgressionCPU -timeout 30m
```

### 5.3 All Trainer Tests

```bash
go test -v ./tests/trainer/... -timeout 60m
```

### 5.4 Monitor Test Progress

While tests are running, monitor pods in another terminal:

```bash
# Watch pods in test namespace
oc get pods -n <test-ns> -w

# Check training pod logs
oc logs -n <test-ns> <trainjob-name>-node-0-0 -f

# Check notebook pod logs
oc logs -n <test-ns> jupyter-nb-<user> -f
```

---

## Troubleshooting

### Issue: Model files missing in S3

**Symptom:**
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**Cause:** Tokenizer files (vocab.json, etc.) not uploaded to S3.

**Fix:** Re-run prestage with `--force`:
```bash
python3 prestage_models_datasets.py --preset rhai --force
```

### Issue: Kubeflow SDK not found

**Symptom:**
```
ModuleNotFoundError: No module named 'kubeflow.trainer'
```

**Cause:** Wrong kubeflow version installed or wheel not in S3.

**Fix:** 
1. Upload correct wheel to S3:
   ```bash
   mc cp kubeflow-0.2.1+rhai0-py3-none-any.whl myminio/<bucket>/wheels/
   ```
2. Verify `KUBEFLOW_WHEEL_S3_KEY` points to correct path

### Issue: SSL Certificate errors

**Symptom:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Cause:** Self-signed certificates on PyPI mirror or S3.

**Fix:** Set trusted host:
```bash
export PIP_TRUSTED_HOST="<hostname>"
```

### Issue: Training pods stuck in ContainerCreating

**Symptom:** Pods stay in ContainerCreating for extended time.

**Cause:** NFS volume provisioning is slow.

**Fix:** Increase timeout:
```bash
export TEST_TIMEOUT_LONG="15m"
```

### Issue: Image pull errors

**Symptom:**
```
ImagePullBackOff
```

**Cause:** Images not mirrored or IDMS not configured.

**Fix:**
1. Mirror images to internal registry
2. Configure ImageDigestMirrorSet
3. Use digest-based image references

## Verifying Success

A successful `TestRhaiTrainingProgressionCPU` test shows:

```
=== RUN   TestRhaiTrainingProgressionCPU
    rhai_features_tests.go:XXX: S3 mode: endpoint=https://..., bucket=...
    rhai_features_tests.go:XXX: PyPI mirror: https://...
    rhai_features_tests.go:XXX: Notebook created successfully
    rhai_features_tests.go:XXX: Training completed, progression tracked
--- PASS: TestRhaiTrainingProgressionCPU (XXXs)
```

In the training pod logs, you should see:
```
[Kubeflow] Progression tracking enabled
Detected accelerator: CPU
Using backend: gloo
Local mode: loading from shared PVC
Loading model from: /workspace/models/distilgpt2
Loading dataset from: /workspace/datasets/alpaca-cleaned
Training started...
{'loss': X.XXX, 'epoch': X.X, ...}
[Kubeflow] Progression: epoch=X, step=X
Training completed successfully
```

---

## Test Flow Diagram

```
┌─────────────────┐
│  Bastion Host   │
│  (has internet) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ prestage script │────▶│    S3/MinIO     │
│ downloads from  │     │ - models/       │
│ HuggingFace     │     │ - datasets/     │
└─────────────────┘     │ - wheels/       │
                        └────────┬────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Notebook Pod   │   │  Training Pod   │   │  Training Pod   │
│ - kubeflow SDK  │   │    (node-0)     │   │    (node-1)     │
│ - downloads S3  │   │ - loads from    │   │ - loads from    │
│   to shared PVC │   │   shared PVC    │   │   shared PVC    │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   Shared PVC    │
                        │ /workspace/     │
                        │  ├─ models/     │
                        │  └─ datasets/   │
                        └─────────────────┘
```
---

## Quick Reference

| Component | Location | Notes |
|-----------|----------|-------|
| Prestage script | `tests/trainer/resources/disconnected_env/prestage_models_datasets.py` | Run on bastion |
| Kubeflow install | `tests/trainer/resources/disconnected_env/install_kubeflow.py` | Used by notebook |
| RHAI notebook | `tests/trainer/resources/rhai_features.ipynb` | Main test notebook |
| Go test | `tests/trainer/sdk_tests/rhai_features_tests.go` | Test runner |

| Env Variable | Required | Description |
|--------------|----------|-------------|
| `NOTEBOOK_USER_NAME` | Yes | OpenShift username |
| `NOTEBOOK_USER_PASSWORD` | Yes | OpenShift password |
| `NOTEBOOK_IMAGE` | Yes | Notebook image with digest (must have Python 3.9+) |
| `AWS_DEFAULT_ENDPOINT` | Disconnected only | S3/MinIO endpoint (for pre-staged models/datasets) |
| `AWS_ACCESS_KEY_ID` | Disconnected only | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | Disconnected only | S3 secret key |
| `AWS_STORAGE_BUCKET` | Disconnected only | S3 bucket name |
| `PIP_INDEX_URL` | Optional | Override PyPI URL (default: Red Hat index based on accelerator) |
| `PIP_TRUSTED_HOST` | Optional | PyPI mirror hostname for SSL bypass (default: console.redhat.com) |
| `TEST_TIMEOUT_LONG` | Recommended | Test timeout (default 5m, set to 15m) |
| `VERIFY_SSL` | Optional | Set to "false" for self-signed S3 certs |
| `GPU_TYPE` | Auto | Auto-detected from test (cpu/nvidia/amd) - selects correct PyPI index |


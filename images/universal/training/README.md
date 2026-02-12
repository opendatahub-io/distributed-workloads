# Universal Training Images

Universal images that serve dual purposes: **Jupyter Workbench** (default) and **Training Runtime** (when command provided). These images are designed for FIPS compliance (best effort) and hermetic builds in downstream Konflux pipelines.

These images are structured to support **hermetic builds** in downstream Konflux/Tekton pipelines:

### Midstream vs Downstream

The Dockerfiles contain sections marked `# MIDSTREAM ONLY` that should be **removed** for downstream builds:

```dockerfile
################################################################################
# MIDSTREAM ONLY: Environment variables and system packages
# ...
################################################################################
ENV ...
RUN dnf install ...
################################################################################
# END MIDSTREAM ONLY
################################################################################
```

**What to remove for downstream:**
- Environment variables (ENV block) - AIPCC base images have these pre-configured
- DNF package installations - AIPCC base images include all required system packages
- Repository files (`cuda.repo`, `rocm.repo`, `mellanox.repo`) - not needed with AIPCC base images

**Downstream base images** (from Notebook with AIPCC base) already include:
- CUDA/ROCm development tools and libraries
- RDMA/InfiniBand packages
- Triton-specific environment variables (`TRITON_PTXAS_PATH`, etc.)
- Properly configured `PATH`, `LD_LIBRARY_PATH`, `CPATH`

---

## Image Flavors

| Flavor | Directory | Use Case |
|--------|-----------|----------|
| **CUDA** | `th<VERSION>-cuda<VERSION>-torch<VERSION>-py<VERSION>/` | NVIDIA GPU training |
| **ROCm** | `th<VERSION>-rocm<VERSION>-torch<VERSION>-py<VERSION>/` | AMD GPU training |
| **CPU** | `th<VERSION>-cpu-torch<VERSION>-py<VERSION>/` | CPU-only training |

All flavors include:
- **Training Hub** with LoRA support
- **PyTorch <VERSION>**
- **Python <VERSION>**
- **JupyterLab** (workbench mode)

### CUDA Image
- CUDA
- Flash Attention
- Mamba SSM, Causal Conv1d
- DeepSpeed, Liger Kernel
- NCCL for multi-GPU/multi-node training

### ROCm Image
- ROCm
- Flash Attention (ROCm build)
- DeepSpeed
- RCCL for multi-GPU/multi-node training

### CPU Image
- Optimized for CPU-only workloads
- NumPy, Numba optimizations
- No GPU-specific packages

---

## Universality: Workbench + Runtime

These images function in two modes:

### Workbench Mode (Default)
```bash
# Starts JupyterLab server
podman run -p 8888:8888 <image>
```

### Runtime Mode
```bash
# Runs training command directly
podman run <image> torchrun --nproc_per_node=2 train.py
```

The `entrypoint-universal.sh` script handles mode detection:
- No command → starts JupyterLab
- Command provided → executes the command

---

## How to Update

Both scenarios require coordination with:
- **AIPCC team** - to get the new index with required packages
- **Notebooks team** - to get the new base image built on top of AIPCC base
- **Training Hub** - dependencies defined in `training-hub` must be available in the AIPCC index

### Training Hub Alignment

The `training-hub` package has its own `pyproject.toml` with dependencies. These dependencies **must be available** in the AIPCC index used by the images. When updating:

1. Check `training-hub`'s dependencies and their version constraints
2. Ensure all transitive dependencies are available in the target AIPCC index
3. If `training-hub` requires a package not in AIPCC, request it from the AIPCC team before updating the images
4. Version constraints in the image's `pyproject.toml` should be compatible with what `training-hub` expects

### Scenario 1: Minor Dependency Update

Update to dependencies **other than** CUDA, ROCm, PyTorch, or Training Hub.

**Same folder, updated files:**

1. Check `training-hub`'s dependencies - ensure new versions are compatible
2. Coordinate with AIPCC team for the new index containing updated packages
3. Coordinate with Notebooks team for the new base image
4. Update `Dockerfile` - change base image reference
5. Update `pyproject.toml` - adjust version constraints as needed
6. Regenerate `requirements.txt`:
   ```bash
   uv pip compile \
     --generate-hashes \
     --index-url=<NEW_AIPCC_INDEX_URL> \
     --python-platform=linux \
     --python-version=<PYTHON_VERSION> \
     -o requirements.txt \
     pyproject.toml
   ```
7. If a dependency is not available in AIPCC, request it from the AIPCC team
8. Build and test

### Scenario 2: Major Version Bump (CUDA / ROCm / PyTorch / Training Hub)

**New folder required:**

1. Check `training-hub`'s dependencies - ensure all are available in the target AIPCC index
2. Coordinate with AIPCC team for the new index with updated major packages
3. Coordinate with Notebooks team for the new base image
4. Create new directory following naming convention:
   - CUDA: `th<TH_VERSION>-cuda<CUDA_VERSION>-torch<TORCH_VERSION>-py<PYTHON_VERSION>/`
   - ROCm: `th<TH_VERSION>-rocm<ROCM_VERSION>-torch<TORCH_VERSION>-py<PYTHON_VERSION>/`
   - CPU: `th<TH_VERSION>-cpu-torch<TORCH_VERSION>-py<PYTHON_VERSION>/`
5. Copy files from previous version and update:
   - `Dockerfile` - update base image, package versions
   - `pyproject.toml` - update dependencies to match `training-hub` requirements
   - `entrypoint-universal.sh`, `LICENSE.md`
   - Repository files if needed (`cuda.repo`, `rocm.repo`, `mellanox.repo`)
6. Regenerate `requirements.txt` with new AIPCC index
7. Build and test

---

## Build Commands

### Local Build (Midstream)

```bash
cd <image-directory>
podman build -t <tag> .
```

### Regenerate Requirements with Hashes

```bash
cd <image-directory>
uv pip compile --generate-hashes \
  --index-url=<AIPCC_INDEX_URL> \
  --python-platform=linux \
  --python-version=<PYTHON_VERSION> \
  -o requirements.txt \
  pyproject.toml
```

---

## Environment Variables (Midstream)

For CUDA images, the following environment variables are set for Triton JIT compilation:

```dockerfile
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CPATH=/usr/local/cuda/include:${CPATH} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
```

For ROCm images:

```dockerfile
ENV ROCM_HOME=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH} \
    CPATH=/opt/rocm/include:${CPATH}
```

**Note:** These are only needed for midstream. AIPCC base images have these pre-configured.

---

## File Structure

```
<image-directory>/
├── Dockerfile              # Multi-stage build with MIDSTREAM ONLY sections
├── pyproject.toml          # Python dependencies (what we add on top of base)
├── requirements.txt        # Locked dependencies with hashes (from AIPCC)
├── entrypoint-universal.sh # Dual-mode entrypoint (workbench/runtime)
├── LICENSE.md              # License file
├── cuda.repo               # CUDA DNF repo (midstream only, CUDA images)
├── rocm.repo               # ROCm DNF repo (midstream only, ROCm images)
└── mellanox.repo           # RDMA DNF repo (midstream only, CUDA/ROCm images)
```

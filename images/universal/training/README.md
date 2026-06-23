# Universal Training Images

Universal images that serve dual purposes: **Jupyter Workbench** (default) and **Training Runtime** (when command provided). These images are designed for FIPS compliance (best effort) and hermetic builds in downstream Konflux pipelines.

These images are structured to support **hermetic builds** in downstream Konflux/Tekton pipelines:

### Midstream vs Downstream

The same Dockerfile works for both builds using the `DOWNSTREAM` build argument:

```bash
# Midstream build (default) - installs system packages and sets env vars
podman build -t <tag> .

# Downstream build - skips midstream-only sections (base image has them)
podman build --build-arg DOWNSTREAM=true -t <tag> .
```

When `DOWNSTREAM=true`:
- DNF package installations are skipped (AIPCC base images include all required system packages)
- Repo files are copied but not used
- Environment variables are still set (safe, may be overridden by downstream base image)

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

## CVE Fixes — Python dependency updates

The training images install Python packages from a **private AIPCC PyPI index** (not public PyPI). Each image's Dockerfile specifies its index URL via `--index-url` — read it from the Dockerfile of the affected image.

### Determining the fix version

**Do NOT rely on the CVE description text to determine affected versions.** The description often mentions only the version where the vulnerability was discovered (e.g., "version 5.2.0"), but the actual affected range may be much wider.

Check the **Product Status** field in the official CVE record at `https://www.cve.org/CVERecord?id=<CVE-ID>`. This field states the authoritative affected version range (e.g., "affected before 5.5.0"). The fix version must be **at or above** the boundary stated in the product status — not just one patch above the version mentioned in the description.

### Scope discipline

Each CVE ticket targets a **single container image** identified by the image name in the ticket summary (e.g., `rhoai/odh-training-cuda124-torch25-py311-rhel9`). When fixing a CVE:

- **Only modify files in the image directory that matches the ticket's target image.** Do not touch other image directories, even if they have the same vulnerability — those are tracked by separate tickets.
- Map the image name to its directory: `odh-training-*` maps to `images/runtime/training/`, `odh-th*` maps to `images/universal/training/`.
- If the image name does not map to any existing directory, flag it — do not modify unrelated directories as a substitute.

### Updating the dependency

When fixing a CVE that requires bumping a Python dependency version:

1. **Do NOT assume the upstream fix version is available.** The private index may not mirror every version from public PyPI.
2. **Query the index to find available versions.** Read the `--index-url` from the affected image's Dockerfile, then fetch `{index-url}/{package}/` to find which versions are available.
3. **If the package is a direct dependency** listed in `pyproject.toml`, update the version constraint there and **regenerate `requirements.txt`** using `uv pip compile` with the index URL from the Dockerfile (see [Regenerate Requirements](#regenerate-requirements-with-hashes) below). **Important:** `uv pip compile` does not emit the `--index-url` line in its output. After regenerating, verify the first line of `requirements.txt` still contains `--index-url=<AIPCC_INDEX_URL>` — if missing, restore it. Without this line, `pip install -r requirements.txt` resolves against public PyPI instead of the AIPCC index, breaking downstream builds.
4. **If the package is a transitive dependency** (only in `requirements.txt`, not in `pyproject.toml`), update the pinned version directly in `requirements.txt` using the exact pin format (`package==x.y.z`).
5. **Use compatible release constraints** when adding or updating constraints in `pyproject.toml`. Use `~=X.Y.Z` (equivalent to `>=X.Y.Z,<X.(Y+1).0`) to pin to the current minor series (e.g., `~=3.14.0`). Unbounded `>=X.Y.Z` allows silent upgrades on lock refresh that may introduce new vulnerabilities.

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

### Downstream Build (Konflux)

```bash
cd <image-directory>
podman build --build-arg DOWNSTREAM=true --build-arg BASE_IMAGE=<aipcc-base-image> -t <tag> .
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

**Note:** These environment variables are set in both midstream and downstream builds (they're safe defaults). AIPCC base images may override these with optimized values.

---

## File Structure

```
<image-directory>/
├── Dockerfile              # Multi-stage build with DOWNSTREAM arg for conditional sections
├── pyproject.toml          # Python dependencies (what we add on top of base)
├── requirements.txt        # Locked dependencies with hashes (from AIPCC)
├── entrypoint-universal.sh # Dual-mode entrypoint (workbench/runtime)
├── LICENSE.md              # License file
├── cuda.repo               # CUDA DNF repo (used only when DOWNSTREAM!=true, CUDA images)
├── rocm.repo               # ROCm DNF repo (used only when DOWNSTREAM!=true, ROCm images)
└── mellanox.repo           # RDMA DNF repo (used only when DOWNSTREAM!=true, CUDA/ROCm images)
```

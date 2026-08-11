# Training Runtime Images

This guide explains how to add new Python dependencies to the training runtime images, regenerate their requirements files, and fix CVEs.

The images use `pyproject.toml` + `requirements.txt` with [uv](https://docs.astral.sh/uv/) for dependency management.

---

## CVE Fixes — Python dependency updates

The runtime training images install Python packages from **public PyPI** (unlike universal training images which use the private AIPCC index).

### Determining the fix version

**Do NOT rely on the CVE description text to determine affected versions.** The description often mentions only the version where the vulnerability was discovered (e.g., "version 5.2.0"), but the actual affected range may be much wider.

Check the **Product Status** field in the official CVE record at `https://www.cve.org/CVERecord?id=<CVE-ID>`. This field states the authoritative affected version range (e.g., "affected before 5.5.0"). The fix version must be **at or above** the boundary stated in the product status — not just one patch above the version mentioned in the description.

### Scope discipline

Each CVE ticket targets a **single container image** identified by the image name in the ticket summary (e.g., `rhoai/odh-training-cuda130-torch210-py312-rhel9`). When fixing a CVE:

- **Only modify files in the image directory that matches the ticket's target image.** Do not touch other image directories, even if they have the same vulnerability — those are tracked by separate tickets.
- Map the image name to its directory: `odh-training-*` maps to `images/runtime/training/`, `odh-th*` maps to `images/universal/training/`.
- If the image name does not map to any existing directory, flag it — do not modify unrelated directories as a substitute.

### Updating the dependency

When fixing a CVE that requires bumping a Python dependency version:

1. **Update the version constraint in `pyproject.toml`.** Use exact pins (`==X.Y.Z`) to prevent supply-chain drift.
2. **Regenerate `requirements.txt`** using the `uv pip compile` command from the header of each image's `pyproject.toml`. If resolution fails locally (e.g., due to `torch` builds requiring GPU libraries), run it inside a container (see [Regenerating requirements.txt](#regenerating-requirementstxt)).
3. **If the package is a transitive dependency** (only in `requirements.txt`, not in `pyproject.toml`), add it as a direct dependency in `pyproject.toml` with an exact pin, then regenerate.

## Adding Dependencies

### Quick Start

1. **Edit `pyproject.toml`** for the image you want to update (e.g., `py312-cuda130-torch210-openmpi41/pyproject.toml`)
2. **Add your dependency** under `[project] dependencies`:
   ```toml
   dependencies = [
       "your-package==1.0.0",
   ]
   ```
3. **Regenerate `requirements.txt`** using the pre-built image (see below)

## Regenerating requirements.txt

Each image's `pyproject.toml` contains the exact `uv pip compile` command in its header comment. The general pattern:

```bash
# CUDA example (py312-cuda130-torch210-openmpi41)
uv pip compile --python-platform=linux --python-version=3.12 \
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cuda13.0-ubi9/simple/ \
    -o requirements.txt pyproject.toml

# ROCm example (py312-rocm64-torch29-openmpi41)
uv pip compile --python-platform=linux --python-version=3.12 \
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/rocm6.4-ubi9-test/simple/ \
    --extra-index-url=https://pypi.org/simple --index-strategy=unsafe-best-match \
    -o requirements.txt pyproject.toml
```

### Running inside a container

If local resolution fails (e.g., flash-attn needs PyTorch at build time), use the pre-built image:

```bash
# 1. Start container
podman run --rm -d --name uv-compile \
    quay.io/opendatahub/odh-training-cuda130-torch210-py312-rhel9:odh-stable \
    sleep 3600

# 2. Copy pyproject.toml
podman cp py312-cuda130-torch210-openmpi41/pyproject.toml uv-compile:/opt/app-root/src/

# 3. Compile requirements
podman exec uv-compile bash -c \
    "uv pip compile --python-platform=linux --python-version=3.12 \
        --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cuda13.0-ubi9/simple/ \
        -o requirements.txt pyproject.toml"

# 4. Copy requirements.txt back
podman cp uv-compile:/opt/app-root/src/requirements.txt py312-cuda130-torch210-openmpi41/

# 5. Clean up
podman stop uv-compile
```

## Which Images Have Flash-attn?

Both current images include flash-attn:
- `py312-cuda130-torch210-openmpi41` — flash-attn==2.8.3
- `py312-rocm64-torch29-openmpi41` — flash-attn==2.8.3

Check via:
```bash
grep "flash-attn" */pyproject.toml
```

## Pre-built Image URLs

Find the latest pre-built images in the `images/` directory Dockerfiles or from:
- CUDA images: `https://quay.io/search?q=training-cuda`
- ROCm images: `https://quay.io/search?q=training-rocm`

## Tips

- Always check the `pyproject.toml` header for the correct `uv pip compile` command — index URLs differ per image
- Verify your new packages: `grep "your-package" requirements.txt`
- OpenMPI is built separately in the Dockerfile and is not a Python dependency


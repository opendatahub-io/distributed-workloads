# Training Runtime Images

This guide explains how to add new Python dependencies to the training runtime images, regenerate their lock files, and fix CVEs.

---

## CVE Fixes — Python dependency updates

The runtime training images install Python packages from **public PyPI** (unlike universal training images which use the private AIPCC index). Each image uses `Pipfile` + `Pipfile.lock` (pipenv).

### Determining the fix version

**Do NOT rely on the CVE description text to determine affected versions.** The description often mentions only the version where the vulnerability was discovered (e.g., "version 5.2.0"), but the actual affected range may be much wider.

Check the **Product Status** field in the official CVE record at `https://www.cve.org/CVERecord?id=<CVE-ID>`. This field states the authoritative affected version range (e.g., "affected before 5.5.0"). The fix version must be **at or above** the boundary stated in the product status — not just one patch above the version mentioned in the description.

### Scope discipline

Each CVE ticket targets a **single container image** identified by the image name in the ticket summary (e.g., `rhoai/odh-training-rocm62-torch25-py311-rhel9`). When fixing a CVE:

- **Only modify files in the image directory that matches the ticket's target image.** Do not touch other image directories, even if they have the same vulnerability — those are tracked by separate tickets.
- Map the image name to its directory: `odh-training-*` maps to `images/runtime/training/`, `odh-th*` maps to `images/universal/training/`.
- If the image name does not map to any existing directory, flag it — do not modify unrelated directories as a substitute.

### Updating the dependency

When fixing a CVE that requires bumping a Python dependency version:

1. **Update the version constraint in `Pipfile`.** Use compatible release constraints (`~=X.Y.Z`) to pin to the current minor series. Do not use unbounded `>=` — it allows silent upgrades on lock refresh that may introduce new vulnerabilities.
2. **Selectively update `Pipfile.lock`** — do NOT hand-edit hashes or version entries in the lockfile. Run `pipenv upgrade --lock-only <package>` to update only the target package and its dependencies while keeping unrelated packages at their current locked versions. If resolution fails locally (e.g., due to `torch` builds from the PyTorch index), run it inside a container (see [Generating Lock Files](#generating-lock-files)).
3. **If the package is a transitive dependency** (only in `Pipfile.lock`, not in `Pipfile`), add it as a direct dependency in `Pipfile` with a `~=` constraint, then run `pipenv upgrade --lock-only <package>`.

## Adding Dependencies

### Quick Start

1. **Edit the Pipfile** for the image you want to update (e.g., `py312-cuda128-torch280/Pipfile`)
2. **Add your dependencies** under `[packages]` with a compatible release constraint to prevent supply-chain drift:
   ```toml
   [packages]
   your-package = "~=1.0.0"
   ```
   `~=X.Y.Z` is equivalent to `>=X.Y.Z,<X.(Y+1).0`, pinning to the current minor series. Unbounded `>=` allows silent upgrades that may introduce new vulnerabilities on lock refresh.
3. **Generate the lock file** using the pre-built image (see below)

## Generating Lock Files

### For images WITH flash-attn

Python 3.12 images with PyTorch 2.8+ include flash-attn, which needs special handling. You should use an image containing relevant Pytorch and Python version and set `PIP_NO_BUILD_ISOLATION=1`:

```bash
# Example: py312-cuda128-torch280

# 1. Start container
podman run --rm -d --name pipenv-lock \
    quay.io/opendatahub/odh-training-cuda128-torch28-py312-rhel9:odh-stable \
    sleep 3600

# 2. Copy Pipfile
podman cp py312-cuda128-torch280/Pipfile pipenv-lock:/opt/app-root/src/

# 3. Run pipenv lock
podman exec pipenv-lock bash -c \
    "pip install pipenv && PIP_NO_BUILD_ISOLATION=1 pipenv lock"

# 4. Copy lock file back
podman cp pipenv-lock:/opt/app-root/src/Pipfile.lock py312-cuda128-torch280/

# 5. Clean up
podman stop pipenv-lock
```

### For images WITHOUT flash-attn

Most Python 3.11 images don't have flash-attn. For these, standard `pipenv lock` command works without build isolation `PIP_NO_BUILD_ISOLATION=1`. In both cases you can use build isolation, but the cost of taking much longer time to generate the lock file. 

## Why PIP_NO_BUILD_ISOLATION=1?

Flash-attn's setup.py imports PyTorch during the build process. By default, pip uses build isolation (PEP 517), which prevents access to system-installed packages. Setting `PIP_NO_BUILD_ISOLATION=1` disables this isolation, allowing flash-attn to find the pre-installed PyTorch.

Without this flag, you'll get:
```text
ERROR: Failed to build 'flash-attn' when getting requirements to build wheel
```

## Which Images Have Flash-attn?

Check the existing `Pipfile.lock` to see if flash-attn is present:
```bash
grep -l "flash-attn" */Pipfile.lock
```

Currently:
- ✅ Has flash-attn: `py312-cuda128-torch280`, `py312-rocm64-torch280`
- ❌ No flash-attn: Most py311 images, py312 torch 2.9+ images

## Pre-built Image URLs

Find the latest pre-built images in the `images/` directory Dockerfiles or from:
- CUDA images: `https://quay.io/search?q=training-cuda`
- ROCm images: `https://quay.io/search?q=training-rocm`

## Tips

- Always use the pre-built image that matches the PyTorch and Python version in the Pipfile
- Verify the lock file has your new packages: `grep "your-package" Pipfile.lock`

## Troubleshooting

**"Cannot import 'setuptools.build_meta'"**
- Don't use `PIPENV_IGNORE_VIRTUALENVS=1` with flash-attn images
- Let pipenv use the existing virtualenv in the pre-built image


# Adding Dependencies to Training Runtime Images

This guide explains how to add new Python dependencies to the training runtime images and regenerate their lock files.

## Quick Start

1. **Edit the Pipfile** for the image you want to update (e.g., `py312-cuda128-torch280/Pipfile`)
2. **Add your dependencies** under `[packages]`:
   ```toml
   [packages]
   your-package = ">=1.0.0"
   ```
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


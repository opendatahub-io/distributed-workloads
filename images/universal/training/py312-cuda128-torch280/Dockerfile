# Universal image Dockerfile
#
# Base image:
# - Minimal Jupyter CUDA workbench with CUDA 12.8 and Python 3.12
# - Provides JupyterLab, Elyra integration, addons, and default ENTRYPOINT start-notebook.sh
# - Source: quay.io/opendatahub/workbench-images:cuda-jupyter-minimal-ubi9-python-3.12-2025a_20250903
#
# Design intent:
# - Preserve workbench behavior by default (no args → start-notebook.sh)
# - Add runtime capabilities on top (Python ML/training stack, RDMA/IB packages)
# - Avoid duplicating dependencies provided by the base image
# - Allow headless runtime mode when a command is provided (args → exec that command)

FROM quay.io/opendatahub/workbench-images:cuda-jupyter-minimal-ubi9-python-3.12-2025a_20250903

LABEL name="universal:py312-cuda128-torch280" \
      summary="Universal CUDA 12.8 Python 3.12 image with PyTorch 2.8.0" \
      description="Universal image combining minimal Jupyter workbench and runtime ML stack (CUDA 12.8, PyTorch 2.8.0, FlashAttention 2.8.3) on UBI9" \
      io.k8s.display-name="Universal CUDA 12.8 Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="Universal image: Jupyter workbench by default; runtime when command provided. Includes RDMA/IB libs, Torch 2.8.0 cu128, FlashAttention 2.8.3."

COPY ./images/universal/training/py312-cuda128-torch280/LICENSE.md /licenses/cuda-license.md

# For OS installs we need elevated privileges; base may default to 1001
USER 0
WORKDIR /opt/app-root/bin

# Keep NVIDIA driver capability constraints consistent with runtime image behavior
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VERSION=12.8 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_UPGRADE_STRATEGY=only-if-needed

# Follow runtime: enable CUDA and Mellanox OFED repositories for RDMA/IB packages.
# Note: The base image already includes CUDA 12.8 runtime; we only add missing components (e.g., RDMA libs).
RUN dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
 && dnf config-manager \
    --add-repo https://linux.mellanox.com/public/repo/mlnx_ofed/latest/rhel9.5/mellanox_mlnx_ofed.repo \
 && dnf install -y --disablerepo="*" --enablerepo="cuda-rhel9-x86_64,mlnx_ofed_24.10-1.1.4.0_base,ubi-9-appstream-rpms,ubi-9-baseos-rpms" \
    libibverbs-utils \
    infiniband-diags \
    libibumad3 \
    librdmacm \
    librdmacm-utils \
    rdma-core \
    mlnx-tools \
 && dnf clean all \
 && rm -rf /var/cache/dnf/*

# Install CUDA NVCC and build toolchain required to build FlashAttention from source
# NOTE: Use command-line CUDA packages to avoid Nsight GUI deps (X11 libs) not available in UBI
RUN dnf install -y --disablerepo="*" --enablerepo="cuda-rhel9-x86_64,ubi-9-appstream-rpms,ubi-9-baseos-rpms" \
    cuda-command-line-tools-12-8 \
    cuda-cudart-devel-12-8 \
    cuda-libraries-devel-12-8 \
    cuda-compat-12-8 \
    cuda-compiler-12-8 \
    cuda-nvcc-12-8-12.8.93-1 \
    gcc \
    gcc-c++ \
    make \
    python3-devel \
    cmake \
    git \
 && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
 && dnf clean all \
 && rm -rf /var/cache/dnf/*

# Ensure CUDA_HOME points to the toolkit and nvcc is discoverable, then sanity check nvcc
ARG CUDA_ARCH_LIST=9.0
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"

# Extra verbosity and progress for CUDA extension builds (Ninja progress, Torch verbose)
ENV USE_NINJA=1 \
    TORCH_CUDA_VERBOSE_BUILD=1 \
    NINJA_STATUS="[%f/%t %es] "
# NOTE: Optional build-time CUDA checks (remove if not needed for faster builds)
# Verify CUDA toolkit present and nvcc available
RUN /usr/local/cuda/bin/nvcc -V
# Verify key CUDA libs are discoverable
RUN ldconfig -p | grep -E 'libcudart|libcublas|libcudnn' || (echo "[fail-fast] CUDA libs not found in ldconfig" >&2; exit 1)
RUN mkdir -p /opt/app-root/.ccache && chown -R 1001:0 /opt/app-root/.ccache

# Quick preflight: verify torch wheel and flash-attn index are reachable to fail fast before large downloads
ARG TORCH_WHEEL_FILE=https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl
RUN curl -IfsS --connect-timeout 10 --max-time 20 "$TORCH_WHEEL_FILE" > /dev/null || (echo "[fail-fast] Torch cu128 wheel not reachable: $TORCH_WHEEL_FILE" >&2; exit 1)
RUN curl -IfsS --connect-timeout 10 --max-time 20 https://pypi.org/simple/flash-attn/ > /dev/null || (echo "[fail-fast] PyPI flash-attn index not reachable" >&2; exit 1)
#
# Additional diagnostics to help debug build env before Python installs
RUN python -m pip debug --verbose || true
RUN gcc --version | head -n1 || true
RUN g++ --version | head -n1 || true
RUN /usr/local/cuda/bin/nvcc -V | head -n1 || true

# Switch back to the non-root user for Python environment changes
USER 1001

WORKDIR /opt/app-root/src

# Ensure user installs land in app-root and are discoverable by Python
ENV PYTHONUSERBASE=/opt/app-root \
    PYTHONNOUSERSITE=0

# Speed up repeated native builds using ccache (if present)
ENV CCACHE_DIR=/opt/app-root/.ccache \
    CCACHE_MAXSIZE=5G \
    CCACHE_COMPRESS=1

# Add runtime Python dependencies on top of the minimal Jupyter stack.
# We intentionally avoid re-installing minimal-provided packages (e.g., jupyterlab) to prevent downgrades.
# Torch/cu128 must match CUDA 12.8. FlashAttention is mandatory and currently supported on amd64.
ARG TARGETARCH
# Enforce amd64 for FlashAttention wheel availability
RUN if [ "$TARGETARCH" != "amd64" ]; then echo "FlashAttention is mandatory and requires amd64 prebuilt wheels. Build with --platform linux/amd64." >&2; exit 1; fi

# Install torch from the PyTorch CUDA index separately to avoid affecting other packages' index resolution
RUN python -m pip install --retries 5 --timeout 300 --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Diagnostics: show interpreter, sys.path, and user site locations
RUN python -c "import sys,site,os; print('exe:',sys.executable); print('sys.path:',sys.path); print('userbase:',site.getuserbase()); print('usersite:',site.getusersitepackages()); print('PYTHONNOUSERSITE=',os.environ.get('PYTHONNOUSERSITE'))"
RUN python -m pip show torch || true && python -c "import importlib.util; print('torch_spec:', importlib.util.find_spec('torch'))"

# NOTE: Optional build-time check (remove if not needed): verify torch build has CUDA enabled
RUN python - <<'PY'
import torch, sys
print("[check] torch", torch.__version__, "cuda build:", torch.version.cuda)
sys.exit(0 if torch.backends.cuda.is_built() else 1)
PY

# Install numpy ahead of building extensions that expect it
RUN python -m pip install --retries 5 --timeout 300 --no-cache-dir numpy==2.3.4

# Install build backend for VCS package and the SDK itself (no build isolation so backend is visible)
RUN python -m pip install --retries 5 --timeout 300 --no-cache-dir hatchling hatch-vcs
RUN python -m pip install --retries 5 --timeout 300 --no-cache-dir --no-build-isolation "git+https://github.com/briangallagher/sdk@training-hub"
#
# NOTE: kubeflow pulls kubernetes==34.x which requires urllib3<2.4, but
# training-hub requires urllib3>=2.4. There is no kubernetes>=35 on PyPI yet.
# We intentionally keep urllib3>=2.4 for training-hub and accept the mismatch.
# To avoid resolution failure, we do NOT try to force-upgrade kubernetes here.
# RUN python -m pip install --retries 5 --timeout 180 --no-cache-dir \
#     "kubernetes>=35.0.0" "urllib3>=2.4,<3"

# Ensure modern build tooling for extensions
RUN python -m pip install --retries 5 --timeout 300 --no-cache-dir -U pip setuptools wheel ninja cmake
#
# Fail-fast: ensure binary wheels exist for packages that are expensive to build
RUN mkdir -p /tmp/wheels && \
    python -m pip download --retries 5 --timeout 120 --only-binary=:all: --no-deps -d /tmp/wheels \
      numba==0.62.1 bitsandbytes==0.48.1 || \
    (echo "[fail-fast] Missing binary wheel for numba or bitsandbytes on this platform (refusing to build from source)." >&2; exit 1)

# Install remaining runtime packages (resolved from default PyPI), including FlashAttention
# Note: We intentionally do not use a Pipfile/lock here to avoid mixing resolvers with the base (uv lock),
#       to control CUDA/FA install order and indexes, and to reduce lock churn across arches/ABI-specific wheels.
RUN python -m pip install --retries 5 --timeout 300 --no-cache-dir --prefer-binary --only-binary=numba,bitsandbytes --upgrade-strategy only-if-needed \
    flash-attn==2.8.3 --no-build-isolation \
    accelerate==1.10.0 \
    transformers==4.57.1 \
    peft==0.17.0 \
    tqdm==4.67.1 \
    datasets==4.0.0 \
    pydantic>=2.11.7 \
    aiofiles==24.1.0 \
    deprecated==1.2.18 \
    typer==0.19.2 \
    "protobuf>=5.28.0,<6.0.0" \
    "simpleeval>=0.9.13,<1.0" \
    safetensors==0.6.2 \
    packaging==25.0 \
    pyyaml==6.0.2 \
    py-cpuinfo==9.0.0 \
    numba==0.62.1 \
    rich==14.1.0 \
    tensorboard==2.19.0 \
    bitsandbytes==0.48.1 \
    liger-kernel==0.6.2 \
    einops==0.8.1 \
    kernels==0.10.3 \
    "sentencepiece>=0.1.99,<0.3" \
    tokenizers==0.22.1 \
    trl==0.21.0 \
    deepspeed>=0.14.3 \
    async-timeout==4.0.3 \
    aiohttp==3.12.15 \
    hf-xet==1.1.8 \
    huggingface-hub==0.34.4 \
    mlflow==3.4.0 \
    psutil==7.0.0 \
    training-hub==0.3.0 \
    instructlab-training==0.12.1 \
    rhai-innovation-mini-trainer==0.3.0 \
 && chmod -R g+w /opt/app-root/lib/python3.12/site-packages \
 && fix-permissions /opt/app-root -P
#
# WARNING: Skipping `pip check` due to known kubernetes(urllib3<2.4) vs
# training-hub(urllib3>=2.4) requirement mismatch. Re-enable once upstream
# loosens kubernetes urllib3 bounds or kubeflow no longer pins 34.x.
# RUN python -m pip check || (python -m pip freeze; exit 1)
#
# Numba diagnostics (helps catch llvmlite/LLVM/NumPy mismatches quickly)
RUN python -m numba -s || (echo "[diagnostics] numba sysinfo failed" >&2; exit 1)

# Provide CUDA user-space libraries via pip, aligning with runtime for extension builds
RUN python -m pip install --retries 5 --timeout 300 --no-cache-dir \
    nvidia-nccl-cu12==2.27.3 \
    nvidia-cublas-cu12==12.8.4.1 \
    nvidia-cuda-cupti-cu12==12.8.90 \
    nvidia-cuda-nvrtc-cu12==12.8.93 \
    nvidia-cuda-runtime-cu12==12.8.90 \
    nvidia-cudnn-cu12==9.10.2.21 \
    nvidia-cufft-cu12==11.3.3.83 \
    nvidia-cufile-cu12==1.13.1.3 \
    nvidia-curand-cu12==10.3.9.90 \
    nvidia-cusolver-cu12==11.7.3.90 \
    nvidia-cusparse-cu12==12.5.8.93 \
    nvidia-cusparselt-cu12==0.7.1 \
    nvidia-nvjitlink-cu12==12.8.93 \
    nvidia-nvtx-cu12==12.8.90 \
 && fix-permissions /opt/app-root -P

# Ensure cuDNN from pip is discoverable during source builds
ENV LD_LIBRARY_PATH="/opt/app-root/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Deterministic 2-step: sub-dep first, then parent without deps (Python 3.12 env)
RUN set -e; \
    ARCH_NO_DOT="$(echo "${CUDA_ARCH_LIST}" | tr -d '.')" ; \
    export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" PYTORCH_NVCC_FLAGS="-gencode arch=compute_${ARCH_NO_DOT},code=sm_${ARCH_NO_DOT}"; \
    python -m pip install --no-build-isolation --no-cache-dir causal-conv1d==1.5.3.post1 && \
    rm -rf "$HOME/.cache/torch_extensions" /tmp/pip-* /tmp/pip-wheel-* || true && \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" PYTORCH_NVCC_FLAGS="${PYTORCH_NVCC_FLAGS}" python -m pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.6.post3 --no-deps && \
    fix-permissions /opt/app-root -P

# Provide a POSIX entrypoint wrapper to choose behavior based on invocation
COPY --chmod=0755 ./images/universal/training/py312-cuda128-torch280/entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh

# Set ENTRYPOINT to the wrapper so that providing a command runs headless.
# Default CMD maintains workbench behavior (no args → start-notebook.sh)
ENTRYPOINT ["/usr/local/bin/entrypoint-universal.sh"]
CMD ["start-notebook.sh"]

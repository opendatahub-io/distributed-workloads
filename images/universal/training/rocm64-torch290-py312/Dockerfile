# ROCm Image Dockerfile
#
# FIPS-friendly Features:
# - uv is used only in build stage (not shipped in runtime image)
# - Build tools are isolated in intermediate stages
# - Final image contains only runtime dependencies

################################################################################
# Build Arguments
################################################################################
# Align base with notebooks minimal ROCm
ARG BASE_IMAGE=quay.io/opendatahub/odh-base-image-rocm-py312-c9s:v6.3
ARG PYTHON_VERSION=3.12
ARG SRC_DIR=rocm64-torch290-py312

################################################################################
# Builder Stage - Install uv for dependency resolution
################################################################################
FROM ${BASE_IMAGE} AS builder

USER 0
WORKDIR /tmp/builder

# Install latest version of uv in builder stage
RUN pip install --no-cache-dir uv

################################################################################
# Base Stage
################################################################################
FROM ${BASE_IMAGE} AS base

LABEL name="rocm:py312-rocm64-torch290" \
      summary="ROCm 6.4 Python 3.12 image with PyTorch 2.9.0" \
      description="ROCm image combining minimal Jupyter workbench and runtime ML stack (ROCm 6.4, PyTorch 2.9.0) on UBI9" \
      io.k8s.display-name="ROCm 6.4 Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="ROCm image: Jupyter workbench by default; runtime when command provided."

# Copy license file
ARG SRC_DIR
COPY ${SRC_DIR}/LICENSE.md /licenses/rocm-license.md

USER 0
WORKDIR /opt/app-root/bin

# Environment variables for ROCm (full paths for HIP/ROCm toolchain)
ENV ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
    CMAKE_PREFIX_PATH=/opt/rocm

################################################################################
# System Dependencies Stage
################################################################################
FROM base AS system-deps

ARG SRC_DIR
USER 0
WORKDIR /opt/app-root/bin

# Core OS packages (minimal)
RUN dnf install -y --setopt=install_weak_deps=False \
    perl \
    mesa-libGL \
    skopeo && \
    dnf clean all && rm -rf /var/cache/dnf/*

# Install the oc client (matches notebooks)
RUN /bin/bash <<'EOF'
set -Eeuxo pipefail
curl -L https://mirror.openshift.com/pub/openshift-v4/$(uname -m)/clients/ocp/stable/openshift-client-linux.tar.gz \
    -o /tmp/openshift-client-linux.tar.gz
tar -xzvf /tmp/openshift-client-linux.tar.gz oc
rm -f /tmp/openshift-client-linux.tar.gz
EOF

# Notebook utils and PDF deps (moved local)
COPY utils/ /opt/app-root/bin/utils/
RUN chmod -R 0755 /opt/app-root/bin/utils && \
    /opt/app-root/bin/utils/install_pdf_deps.sh

# Copy notebook entry script and entrypoint
COPY utils/start-notebook.sh /opt/app-root/bin/start-notebook.sh
COPY utils/entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh
RUN chmod 0755 /opt/app-root/bin/start-notebook.sh /usr/local/bin/entrypoint-universal.sh

# Copy repository configuration files
COPY ${SRC_DIR}/mellanox.repo ${SRC_DIR}/rocm.repo /etc/yum.repos.d/

# Install ROCm development tools
# Using individual packages instead of metapackages to avoid python3-wheel dependency issue
# - rocm-llvm: LLVM compiler required by hipcc (provides /opt/rocm/llvm/bin/clang++)
# - hipcc: HIP compiler wrapper
# - hip-devel: HIP development headers
# - rocm-device-libs: GPU device library required by clang for ROCm compilation
RUN dnf install -y --setopt=install_weak_deps=False \
    rocm-llvm \
    hipcc \
    hip-devel \
    hip-runtime-amd \
    rocthrust \
    hipsparse-devel \
    hipsparse \
    hipcub-devel \
    rocprim-devel \
    hipblaslt-devel \
    rocrand \
    hipfft \
    rocfft \
    rocm-cmake \
    rocm-device-libs \
    rocblas-devel \
    hipblas-devel \
    rocsolver-devel \
    hipsolver-devel && \
    dnf clean all && rm -rf /var/cache/dnf/*

# Fix /opt/rocm symlink - base image has it pointing to /etc/alternatives/rocm
# which doesn't contain the full ROCm installation. We need it to point to /opt/rocm-6.4.3
RUN echo "=== Fixing ROCm symlink ===" && \
    echo "Current /opt/rocm points to:" && readlink /opt/rocm && \
    rm -f /opt/rocm && \
    ln -sf /opt/rocm-6.4.3 /opt/rocm && \
    echo "Fixed /opt/rocm now points to:" && readlink /opt/rocm && \
    ls -la /opt/rocm/ && \
    echo "=== ROCm symlink fixed ==="

# Verify ROCm/HIP toolchain is properly installed
RUN echo "=== Verifying ROCm/HIP installation ===" && \
    echo "hipcc:" && ls -la /opt/rocm/bin/hipcc && \
    echo "clang++:" && ls -la /opt/rocm/lib/llvm/bin/clang++ && \
    echo "Testing hipcc:" && /opt/rocm/bin/hipcc --version && \
    echo "ROCm device libs:" && ls /opt/rocm/amdgcn/bitcode/ | head -5 && \
    echo "=== ROCm verification complete ==="

# Install system packages (RDMA and build toolchain)
#
# RDMA/InfiniBand packages (from mellanox.repo):
# - libibverbs-utils, infiniband-diags: RDMA diagnostics and utilities
# - libibumad: User-space MAD (Management Datagram) library for InfiniBand
# - librdmacm, librdmacm-utils: RDMA connection management
# - rdma-core: Core RDMA user-space libraries
#
# Build toolchain (from UBI repos):
# - gcc, gcc-c++, make: C/C++ compilation tools
# - python3-devel: Python headers for building native extensions
# - cmake: Build system (required by some Python packages)
# - git: Version control (some pip installs need it)
RUN dnf install -y --setopt=install_weak_deps=False \
    libibverbs-utils \
    infiniband-diags \
    libibumad \
    librdmacm \
    librdmacm-utils \
    rdma-core \
    gcc \
    gcc-c++ \
    make \
    python3-devel \
    cmake \
    git && dnf clean all && rm -rf /var/cache/dnf/*

# Bundle RDMA runtime libs to a staging dir
RUN mkdir -p /opt/rdma-runtime \
 && cp -a /usr/lib64/libibverbs* /opt/rdma-runtime/ || true \
 && cp -a /usr/lib64/librdmacm* /opt/rdma-runtime/ || true \
 && cp -a /usr/lib64/libibumad* /opt/rdma-runtime/ || true \
 && cp -a /usr/lib64/libmlx* /opt/rdma-runtime/ || true \
 && cp -a /usr/lib64/libibnetdisc* /opt/rdma-runtime/ || true

################################################################################
# Python Dependencies Stage
################################################################################
FROM system-deps AS python-deps

ARG SRC_DIR
USER 0
WORKDIR /tmp/deps

# Ensure python version arg available in this stage
ARG PYTHON_VERSION

# Copy uv from builder stage (FIPS: uv only used during build, not in runtime)
COPY --from=builder /opt/app-root/bin/uv /usr/local/bin/uv

# Copy dependency files
# pylock.toml: All dependencies including ROCm PyTorch (compiled with --find-links)
# requirements-special.txt: Packages needing --no-build-isolation (flash-attn)
COPY --chown=1001:0 ${SRC_DIR}/pyproject.toml ${SRC_DIR}/pylock.toml ${SRC_DIR}/requirements-special.txt ./

# Switch to user 1001 for pip installations
USER 1001
WORKDIR /opt/app-root/src

# Install main dependencies from pylock.toml using uv pip sync
# This syncs the environment to match exactly what's in the lockfile
# pylock.toml was compiled with --find-links=https://download.pytorch.org/whl/rocm6.4
# so torch comes from ROCm index
ENV UV_NO_CACHE=1
RUN uv pip sync --python-platform=linux --python-version=3.12 /tmp/deps/pylock.toml
ENV UV_NO_CACHE=

# Install kubeflow-sdk from Git (not in pylock.toml or requirements-special.txt)
# TODO: use aipcc index
RUN pip install --retries 5 --timeout 300 --no-cache-dir \
    "git+https://github.com/opendatahub-io/kubeflow-sdk@main"

# Apply notebook customizations (match notebooks minimal)
RUN /bin/bash <<'EOF'
set -Eeuo pipefail
# disable announcements
jupyter labextension disable "@jupyterlab/apputils-extension:announcements" || true
# rename kernel launcher to current python version
sed -i -e "s/Python.*/$(python --version | cut -d '.' -f-2)\",/" /opt/app-root/share/jupyter/kernels/python3/kernel.json
# copy jupyter config
mkdir -p /opt/app-root/etc/jupyter
cp /opt/app-root/bin/utils/jupyter_server_config.py /opt/app-root/etc/jupyter
# apply addons
/opt/app-root/bin/utils/addons/apply.sh
# usercustomize / protobuf patch
cp /opt/app-root/bin/utils/usercustomize.pth /opt/app-root/lib/python${PYTHON_VERSION}/site-packages/
cp /opt/app-root/bin/utils/monkey_patch_protobuf_6x.py /opt/app-root/lib/python${PYTHON_VERSION}/site-packages/
EOF

# Install flash-attn from requirements-special.txt
# Requires:
# - GPU_ARCHS: tells flash-attn which ROCm architectures to build for (no GPU needed at build time)
# - PYTORCH_ROCM_ARCH: additional hint for PyTorch/ROCm
# - MAX_JOBS/CMAKE_BUILD_PARALLEL_LEVEL: parallel kernel compilation (can be overridden via build-args)
# - --no-build-isolation: use pre-installed torch for the build
# - --no-deps: flash-attn deps already satisfied by pylock.toml

# Accept build args for parallelism (can be overridden by argfile.konflux.conf)
ARG MAX_JOBS=16
ARG CMAKE_BUILD_PARALLEL_LEVEL=8

# Set environment for flash-attn build
ENV GPU_ARCHS="gfx90a;gfx942" \
    PYTORCH_ROCM_ARCH="gfx90a;gfx942" \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}

# Verify ROCm tools are accessible before building flash-attn
# This runs in python-deps stage to ensure symlinks from system-deps are inherited
RUN echo "=== Pre-build verification in python-deps stage ===" && \
    echo "Checking /opt/rocm/bin/hipcc:" && \
    ls -la /opt/rocm/bin/hipcc && \
    echo "Checking symlink target exists:" && \
    readlink -f /opt/rocm/bin/hipcc && \
    ls -la $(readlink -f /opt/rocm/bin/hipcc) && \
    echo "Testing hipcc execution:" && \
    /opt/rocm/bin/hipcc --version && \
    echo "=== Pre-build verification passed ==="

# Build flash-attn with verbose output to capture any errors
RUN echo "=== Starting flash-attn build ===" && \
    echo "MAX_JOBS=${MAX_JOBS}" && \
    echo "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}" && \
    echo "GPU_ARCHS=${GPU_ARCHS}" && \
    echo "ROCM_HOME=${ROCM_HOME}" && \
    echo "HIP_PATH=${HIP_PATH}" && \
    pip install --no-build-isolation --no-cache-dir --no-deps --verbose \
    $(grep "^flash-attn" /tmp/deps/requirements-special.txt) 2>&1 | tee /tmp/flash-attn-build.log && \
    echo "=== flash-attn build complete ==="

# Fix permissions for OpenShift
ARG PYTHON_VERSION
USER 0
RUN chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages \
 && fix-permissions /opt/app-root -P

# Clean up uv and build artifacts
RUN rm -f /usr/local/bin/uv \
 && rm -rf /tmp/deps \
 && dnf remove -y gcc gcc-c++ cmake python3-devel \
 && dnf clean all \
 && rm -rf /var/cache/dnf/*

################################################################################
# Final Stage - FIPS-friendly Runtime
################################################################################
FROM ${BASE_IMAGE} AS final

USER 0
WORKDIR /opt/app-root/src

# Copy Python site-packages and CLI entry points from python-deps stage
ARG PYTHON_VERSION
ARG SRC_DIR
COPY --from=python-deps /opt/app-root/lib/python${PYTHON_VERSION}/site-packages /opt/app-root/lib/python${PYTHON_VERSION}/site-packages
COPY --from=python-deps /opt/app-root/bin /opt/app-root/bin
# Copy Jupyter shared assets (lab static files, etc.)
COPY --from=python-deps /opt/app-root/share/jupyter /opt/app-root/share/jupyter
# Copy Jupyter etc configs (server extensions, settings)
COPY --from=python-deps /opt/app-root/etc/jupyter /opt/app-root/etc/jupyter

# Copy RDMA runtime libraries from system-deps
# These are needed for InfiniBand/RDMA support at runtime
COPY --from=system-deps /opt/rdma-runtime/ /usr/lib64/

# Update dynamic linker cache
RUN ldconfig

# FIPS-friendly: Remove uv from final image
RUN rm -f /opt/app-root/bin/uv

# Environment variables for ROCm (full paths for HIP/ROCm toolchain)
ENV ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH

# Copy license file
COPY ${SRC_DIR}/LICENSE.md /licenses/rocm-license.md

# Copy entrypoint
COPY --chmod=0755 utils/entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh

# Fix permissions for OpenShift (final stage)
RUN fix-permissions /opt/app-root -P \
 && chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages

USER 1001
WORKDIR /opt/app-root/src

ENTRYPOINT ["/usr/local/bin/entrypoint-universal.sh"]
CMD ["start-notebook.sh"]

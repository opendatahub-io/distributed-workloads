# Universal Image Dockerfile
#
# FIPS-friendly Features:
# - uv is used only in build stage (not shipped in runtime image)
# - Build tools are isolated in intermediate stages
# - Final image contains only runtime dependencies
# - OpenSSL FIPS mode supported via base image

################################################################################
# Build Arguments
################################################################################
# Align base with notebooks minimal CUDA
ARG BASE_IMAGE=quay.io/opendatahub/odh-base-image-cuda-py312-c9s:v12.8
# CUDA_VERSION: Used for environment variables and documentation purposes
# - Sets CUDA_VERSION env var (helps with debugging and tooling)
# - Not used for package installation (specific versions hardcoded below)
ARG CUDA_VERSION=12.8
# PYTHON_VERSION: Critical for path resolution in multi-stage build
# - Used to locate site-packages directory (e.g., /opt/app-root/lib/python3.12/site-packages)
# - Must match base image Python version
ARG PYTHON_VERSION=3.12
ARG SRC_DIR=th04-cuda128-torch290-py312

################################################################################
# Builder Stage - Install uv for dependency resolution
################################################################################
FROM ${BASE_IMAGE} AS builder

USER 0
WORKDIR /tmp/builder

# Install latest version of uv in builder stage
# Why: Even if base image has uv, we want the latest version for:
#      - Latest bug fixes and performance improvements
#      - Consistent behavior across builds
#      - Specific version control independent of base image
# Note: This uv is isolated in builder stage and copied selectively to other stages
RUN pip install --no-cache-dir uv

################################################################################
# Base Stage
################################################################################
FROM ${BASE_IMAGE} AS base

LABEL name="universal:py312-cuda128-torch290" \
      summary="Universal CUDA 12.8 Python 3.12 image with PyTorch 2.9.0" \
      description="Universal image combining minimal Jupyter workbench and runtime ML stack (CUDA 12.8, PyTorch 2.9.0, FlashAttention 2.8.2) on UBI9" \
      io.k8s.display-name="Universal CUDA 12.8 Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="Universal image: Jupyter workbench by default; runtime when command provided."

# Copy license file
ARG SRC_DIR
COPY ${SRC_DIR}/LICENSE.md /licenses/cuda-license.md

USER 0
WORKDIR /opt/app-root/bin

# Environment variables for NVIDIA and CUDA
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VERSION=${CUDA_VERSION} \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

################################################################################
# System Dependencies Stage
################################################################################
FROM base AS system-deps

ARG SRC_DIR
USER 0
WORKDIR /opt/app-root/bin

# Core OS packages (minimal, keep from notebooks)
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

# Copy repository configuration files
COPY ${SRC_DIR}/cuda.repo ${SRC_DIR}/mellanox.repo /etc/yum.repos.d/

# Notebook utils and PDF deps (local copy)
COPY utils/ /opt/app-root/bin/utils/
RUN chmod -R 0755 /opt/app-root/bin/utils && \
    /opt/app-root/bin/utils/install_pdf_deps.sh

# Copy notebook entry script and entrypoint
COPY utils/start-notebook.sh /opt/app-root/bin/start-notebook.sh
COPY utils/entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh
RUN chmod 0755 /opt/app-root/bin/start-notebook.sh /usr/local/bin/entrypoint-universal.sh

# Install system packages (RDMA, CUDA tools, build toolchain)
# Package list defined in rpms.in.yaml for documentation
#
# RDMA/InfiniBand packages (from mellanox.repo):
# - libibverbs-utils, infiniband-diags: RDMA diagnostics and utilities
# - libibumad: User-space MAD (Management Datagram) library for InfiniBand
# - librdmacm, librdmacm-utils: RDMA connection management
# - rdma-core: Core RDMA user-space libraries
#
# NOTE: mlnx-tools is intentionally NOT included
# - mlnx-tools provides Mellanox-specific diagnostics (mlxlink, mlxconfig, etc.)
# - For containerized ML workloads, standard InfiniBand tools (infiniband-diags) are sufficient
# - Reduces image size and dependency complexity
# - If needed, can be added: mlnx-tools \
#
# CUDA packages (from cuda.repo):
# - cuda-command-line-tools-12-8: CUDA CLI utilities
# - cuda-cudart-devel-12-8: CUDA runtime development headers
# - cuda-nvcc-12-8-12.8.93-1: CUDA compiler (specific version for reproducibility)
#
# Build toolchain (from UBI repos):
# - gcc, gcc-c++, make: C/C++ compilation tools
# - python3-devel: Python headers for building native extensions
# - cmake: Build system (required by some Python packages)
# - git: Version control (some pip installs need it)
#
# --setopt=install_weak_deps=False: Don't install recommended packages (minimizes image size)
RUN dnf install -y --setopt=install_weak_deps=False \
    libibverbs-utils \
    infiniband-diags \
    libibumad \
    librdmacm \
    librdmacm-utils \
    rdma-core \
    cuda-command-line-tools-12-8 \
    cuda-cudart-devel-12-8 \
    cuda-nvcc-12-8-12.8.93-1 \
    gcc \
    gcc-c++ \
    make \
    python3-devel \
    cmake \
    git && dnf clean all && rm -rf /var/cache/dnf/*

# Verify CUDA toolkit
RUN /usr/local/cuda/bin/nvcc -V
RUN ldconfig -p | grep -E 'libcudart|libcublas|libcudnn' || \
    (echo "[fail-fast] CUDA libs not found" >&2; exit 1)

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

# Copy pyproject.toml, pylock.toml, and requirements-special.txt
# pylock.toml contains most dependencies
# requirements-special.txt contains packages needing --no-build-isolation
COPY --chown=1001:0 ${SRC_DIR}/pyproject.toml ${SRC_DIR}/pylock.toml ${SRC_DIR}/requirements-special.txt ./

# Switch to user 1001 for pip installations
USER 1001
WORKDIR /opt/app-root/src

# Install main dependencies from pylock.toml using uv pip sync
# This syncs the environment to match exactly what's in the lockfile
#
# UV_NO_CACHE explained:
# What: Sets UV_NO_CACHE=1 temporarily, then unsets it (empty string)
# Why: Running as user 1001 causes uv to try writing to cache directory
#      which may have permission issues. Disabling cache avoids this.
# Why unset: ENV changes persist across layers. Unsetting prevents
#            the variable from affecting subsequent operations or runtime.
#            Empty value effectively unsets the environment variable.
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

# Install special packages with proper flags
# These packages require --no-build-isolation to use pre-installed CUDA tools
# and must be installed in a specific order

# To avoid out-of-memory during CUDA extension builds, cap parallel jobs
# These environment variables are respected by PyTorch/ninja/CMake builds
ENV MAX_JOBS=4 \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    NINJA_FLAGS=-j4

# 1. Flash Attention (standalone, needs --no-build-isolation --no-deps)
RUN pip install --no-build-isolation --no-cache-dir --no-deps \
    $(grep "^flash-attn" /tmp/deps/requirements-special.txt)

# 2. Mamba SSM dependencies (order matters!)
# - causal-conv1d first (needs --no-build-isolation)
# - mamba-ssm second (needs --no-build-isolation --no-deps)
RUN pip install --no-build-isolation --no-cache-dir \
    $(grep "^causal-conv1d" /tmp/deps/requirements-special.txt) \
 && pip install --no-build-isolation --no-cache-dir --no-deps \
    $(grep "^mamba-ssm" /tmp/deps/requirements-special.txt)

# Fix permissions for OpenShift
# What: Adjusts file permissions for OpenShift/Kubernetes compatibility
# Why: OpenShift runs containers with arbitrary user IDs but fixed group ID (root group)
#      - chmod g+w: Allows group write access to site-packages (for pip installs at runtime)
#      - fix-permissions: UBI-provided script that ensures group ownership/permissions
# When needed: Required for any container that may run in OpenShift with arbitrary UIDs
ARG PYTHON_VERSION
USER 0
RUN chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages \
 && fix-permissions /opt/app-root -P

# Clean up uv and build artifacts (FIPS: remove build-only tools)
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
# This excludes build tools like gcc, cmake, uv (FIPS friendly)
ARG PYTHON_VERSION
ARG SRC_DIR
COPY --from=python-deps /opt/app-root/lib/python${PYTHON_VERSION}/site-packages /opt/app-root/lib/python${PYTHON_VERSION}/site-packages
COPY --from=python-deps /opt/app-root/bin /opt/app-root/bin
# Copy Jupyter shared assets (lab static files, etc.)
COPY --from=python-deps /opt/app-root/share/jupyter /opt/app-root/share/jupyter
# Copy Jupyter etc configs (server extensions, settings)
COPY --from=python-deps /opt/app-root/etc/jupyter /opt/app-root/etc/jupyter

# Copy CUDA runtime from system-deps (built Python packages need CUDA libs)
# Contains all necessary CUDA libraries - no need to install via dnf
COPY --from=system-deps /usr/local/cuda /usr/local/cuda

# Copy RDMA runtime libraries from system-deps
# These are needed for InfiniBand/RDMA support at runtime
COPY --from=system-deps /opt/rdma-runtime/ /usr/lib64/

# Update dynamic linker cache for CUDA libraries
# What: ldconfig updates the runtime linker's cache of shared libraries
# Why: After copying CUDA libraries to /usr/local/cuda, the system needs to know where to find them
#      - Scans directories like /usr/local/cuda/lib64 (defined in /etc/ld.so.conf.d/)
#      - Updates /etc/ld.so.cache so programs can locate libcudart.so, libcublas.so, etc.
# When needed: Required after installing/copying shared libraries to non-standard locations
# Test: Run "ldconfig -p | grep cuda" to see if CUDA libs are in the cache
RUN ldconfig

# FIPS-friendly: Remove uv from final image (inherited from base image)
# uv is only needed during build, not at runtime
RUN rm -f /opt/app-root/bin/uv

# Environment variables for NVIDIA and CUDA
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VERSION=${CUDA_VERSION} \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Copy license file
COPY ${SRC_DIR}/LICENSE.md /licenses/cuda-license.md

# Copy entrypoint
COPY --chmod=0755 utils/entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh

# Fix permissions for OpenShift (final stage)
# What: Ensures proper permissions for OpenShift/Kubernetes arbitrary UIDs
# Why: After copying site-packages from python-deps stage, permissions need adjustment
#      - OpenShift assigns random UID but fixed GID (usually 0, root group)
#      - Group write permissions allow pip to install packages at runtime
#      - fix-permissions ensures all files have correct group ownership
# When: Required in final stage because COPY operations reset permissions
# Context: This is the second time we do this - once after building packages,
#          and again after copying them to the final stage
RUN fix-permissions /opt/app-root -P \
 && chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages

USER 1001
WORKDIR /opt/app-root/src

ENTRYPOINT ["/usr/local/bin/entrypoint-universal.sh"]
CMD ["start-notebook.sh"]
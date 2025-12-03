# ROCm Image Dockerfile
#
# FIPS-friendly Features:
# - uv is used only in build stage (not shipped in runtime image)
# - Build tools are isolated in intermediate stages
# - Final image contains only runtime dependencies

################################################################################
# Build Arguments
################################################################################
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-rocm-py312-ubi9:2025b-v1.39
ARG PYTHON_VERSION=3.12

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

LABEL name="rocm:py312-rocm64-torch280" \
      summary="ROCm 6.4 Python 3.12 image with PyTorch 2.8.0" \
      description="ROCm image combining minimal Jupyter workbench and runtime ML stack (ROCm 6.4, PyTorch 2.8.0) on UBI9" \
      io.k8s.display-name="ROCm 6.4 Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="ROCm image: Jupyter workbench by default; runtime when command provided."

# Copy license file
COPY LICENSE.md /licenses/rocm-license.md

USER 0
WORKDIR /opt/app-root/bin

# Environment variables for ROCm
ENV ROCM_HOME=/opt/rocm \
    PATH=/opt/rocm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

################################################################################
# System Dependencies Stage
################################################################################
FROM base AS system-deps

USER 0
WORKDIR /opt/app-root/bin

# Copy repository configuration files
COPY mellanox.repo rocm.repo /etc/yum.repos.d/

# Install ROCm development tools
# Using individual packages instead of metapackages to avoid python3-wheel dependency issue
# hipcc is the HIP compiler needed for flash-attention build
# rocm-device-libs provides the GPU device library required by clang for ROCm compilation
RUN dnf install -y --setopt=install_weak_deps=False \
    hipcc \
    hip-devel \
    hip-runtime-amd \
    rocm-cmake \
    rocm-device-libs \
    rocblas-devel \
    hipblas-devel \
    rocsolver-devel \
    hipsolver-devel && \
    dnf clean all && rm -rf /var/cache/dnf/* && \
    # hipcc installs to /opt/rocm-X.Y.Z/bin but we need /opt/rocm/bin/hipcc
    ln -sf /opt/rocm-*/bin/hipcc /opt/rocm/bin/hipcc

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

USER 0
WORKDIR /tmp/deps

# Copy uv from builder stage (FIPS: uv only used during build, not in runtime)
COPY --from=builder /opt/app-root/bin/uv /usr/local/bin/uv

# Copy dependency files
# pylock.toml: All dependencies including ROCm PyTorch (compiled with --find-links)
COPY --chown=1001:0 pyproject.toml pylock.toml ./

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

# TODO: Re-enable Flash Attention after confirming base image works
# Install Flash Attention from original Dao-AILab repo
# --no-build-isolation: Use already-installed torch instead of isolated env
# USER 0
# ENV GPU_ARCHS="gfx90a;gfx942"
# RUN cd /tmp \
#     && git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git \
#     && cd flash-attention \
#     && git submodule update --init \
#     && MAX_JOBS="16" pip install --no-build-isolation --no-cache-dir --no-deps . \
#     && cd / && rm -rf /tmp/flash-attention


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
COPY --from=python-deps /opt/app-root/lib/python${PYTHON_VERSION}/site-packages /opt/app-root/lib/python${PYTHON_VERSION}/site-packages
COPY --from=python-deps /opt/app-root/bin /opt/app-root/bin

# Copy RDMA runtime libraries from system-deps
# These are needed for InfiniBand/RDMA support at runtime
COPY --from=system-deps /opt/rdma-runtime/ /usr/lib64/

# Update dynamic linker cache
RUN ldconfig

# FIPS-friendly: Remove uv from final image
RUN rm -f /opt/app-root/bin/uv

# Environment variables for ROCm
ENV ROCM_HOME=/opt/rocm \
    PATH=/opt/rocm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Copy license file
COPY LICENSE.md /licenses/rocm-license.md

# Copy entrypoint
COPY --chmod=0755 entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh

# Fix permissions for OpenShift (final stage)
RUN fix-permissions /opt/app-root -P \
 && chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages

USER 1001
WORKDIR /opt/app-root/src

ENTRYPOINT ["/usr/local/bin/entrypoint-universal.sh"]
CMD ["start-notebook.sh"]

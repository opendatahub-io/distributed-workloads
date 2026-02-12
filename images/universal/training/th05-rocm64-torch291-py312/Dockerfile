# ROCm Image Dockerfile
#
# FIPS-friendly Features:
# - Build tools are isolated in intermediate stage
# - Final image contains only runtime dependencies
# - Uses pip install (additive) to preserve base image packages

################################################################################
# Build Arguments
################################################################################
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-rocm-py312-ubi9:3.4_ea1-v1.41
ARG PYTHON_VERSION=3.12

################################################################################
# Build Stage - Install Python Dependencies
################################################################################
FROM ${BASE_IMAGE} AS builder

USER 0
WORKDIR /tmp/deps

# Copy requirements files
COPY --chown=1001:0 pyproject.toml requirements.txt ./

# Switch to user 1001 for pip installations
USER 1001
WORKDIR /opt/app-root/src

# Install main dependencies from AIPCC index
# All packages (including flash-attn) are pre-built wheels in AIPCC
# Strict AIPCC-only - no fallback indexes (request missing packages be added to AIPCC)
RUN uv pip install --no-cache-dir --require-hashes \
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4-EA1/rocm6.4-ubi9-test/simple/ \
    -r /tmp/deps/requirements.txt

# Install kubeflow-sdk from Git (not in AIPCC index yet)
# TODO: use aipcc index when available
RUN pip install --retries 5 --timeout 300 --no-cache-dir \
    "git+https://github.com/opendatahub-io/kubeflow-sdk@main"

# Fix permissions for OpenShift
ARG PYTHON_VERSION
USER 0
RUN chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages \
 && fix-permissions /opt/app-root -P

# Clean up
RUN rm -rf /tmp/deps

################################################################################
# Final Stage - FIPS-friendly Runtime
################################################################################
FROM ${BASE_IMAGE} AS final

LABEL name="rocm:py312-rocm64-torch291" \
      summary="ROCm 6.4 Python 3.12 image with PyTorch 2.9.1" \
      description="ROCm image combining minimal Jupyter workbench and runtime ML stack (ROCm 6.4, PyTorch 2.9.1) on UBI9" \
      io.k8s.display-name="ROCm 6.4 Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="ROCm image: Jupyter workbench by default; runtime when command provided."

USER 0
WORKDIR /opt/app-root/src

################################################################################
# MIDSTREAM ONLY: Environment variables and system packages
# These are only needed for midstream builds where the base image doesn't include
# ROCm environment setup. In downstream (Konflux), the AIPCC base image already
# has all ROCm/HIP environment variables set and this section should be removed.
################################################################################

# Environment variables for ROCm/HIP (MIDSTREAM ONLY)
# - ROCM_HOME: Standard ROCm installation path
# - HIP_PATH: HIP installation path (same as ROCM_HOME)
# - PATH: Include ROCm binaries (hipcc, etc.) for Triton JIT compilation
# - LD_LIBRARY_PATH: ROCm libraries
# - CPATH: C include path for gcc to find HIP headers (needed by Triton)
ENV ROCM_HOME=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH} \
    CPATH=/opt/rocm/include:${CPATH}

# System packages (MIDSTREAM ONLY)
# Add repos for ROCm and RDMA packages
COPY rocm.repo mellanox.repo /etc/yum.repos.d/

RUN dnf install -y --setopt=install_weak_deps=False \
    # Core OS packages
    perl \
    mesa-libGL \
    skopeo \
    # RDMA/InfiniBand packages (for multi-node training)
    libibverbs-utils \
    infiniband-diags \
    libibumad \
    librdmacm \
    librdmacm-utils \
    rdma-core \
    # ROCm development tools (needed for Triton JIT compilation at runtime)
    rocm-llvm \
    hipcc \
    hip-devel \
    hip-runtime-amd \
    rocm-device-libs && \
    dnf clean all && rm -rf /var/cache/dnf/*

# Fix /opt/rocm symlink if needed
RUN if [ -L /opt/rocm ] && [ "$(readlink /opt/rocm)" != "/opt/rocm-6.4.3" ]; then \
        rm -f /opt/rocm && ln -sf /opt/rocm-6.4.3 /opt/rocm; \
    fi
################################################################################
# END MIDSTREAM ONLY
################################################################################

# Copy Python site-packages and CLI entry points from builder stage
# This excludes any build artifacts (FIPS friendly)
ARG PYTHON_VERSION
COPY --from=builder /opt/app-root/lib/python${PYTHON_VERSION}/site-packages /opt/app-root/lib/python${PYTHON_VERSION}/site-packages
COPY --from=builder /opt/app-root/bin /opt/app-root/bin

# Remove uv from final image (inherited from base image, not needed at runtime)
RUN rm -f /opt/app-root/bin/uv

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

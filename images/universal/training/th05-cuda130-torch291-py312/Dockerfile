# Universal Image Dockerfile
#
# FIPS-friendly Features:
# - Build tools are isolated in intermediate stage
# - Final image contains only runtime dependencies
# - OpenSSL FIPS mode supported via base image
# - Uses pip install (additive) to preserve base image packages
#
# Build Modes:
# - Midstream (default): DOWNSTREAM=false - installs system packages and sets env vars
# - Downstream: DOWNSTREAM=true - skips midstream-only sections (base image has them)

################################################################################
# Build Arguments
################################################################################
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-cuda-py312-ubi9:3.4_ea1-v1.41
ARG PYTHON_VERSION=3.12
ARG DOWNSTREAM=false

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
# All packages (including flash-attn, mamba-ssm) are pre-built wheels in AIPCC
RUN uv pip install --no-cache-dir --require-hashes \
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4-EA1/cuda13.0-ubi9-test/simple/ \
    -r /tmp/deps/requirements.txt

# Install kubeflow-sdk from Git (not in requirements.txt)
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

LABEL name="universal:py312-cuda130-torch291" \
      summary="Universal CUDA 13.0 Python 3.12 image with PyTorch 2.9.1" \
      description="Universal image combining minimal Jupyter workbench and runtime ML stack (CUDA 13.0, PyTorch 2.9.1, FlashAttention 2.8.3) on UBI9" \
      io.k8s.display-name="Universal CUDA 13.0 Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="Universal image: Jupyter workbench by default; runtime when command provided."

USER 0
WORKDIR /opt/app-root/src

################################################################################
# MIDSTREAM ONLY: Environment variables and system packages
# Controlled by ARG DOWNSTREAM (default: false)
# - DOWNSTREAM=false (midstream): Installs CUDA dev tools, RDMA packages, sets env vars
# - DOWNSTREAM=true: Skips this section (AIPCC base image has everything pre-configured)
################################################################################
ARG DOWNSTREAM

# Environment variables for NVIDIA and CUDA
# These are safe to set in both modes - they define standard paths
# In downstream, the base image may override these appropriately
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CPATH=/usr/local/cuda/include:${CPATH} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# System packages (MIDSTREAM ONLY - skipped when DOWNSTREAM=true)
# Repo files for CUDA and RDMA packages
COPY cuda.repo mellanox.repo /etc/yum.repos.d/

RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    echo "MIDSTREAM BUILD: Installing system packages..." && \
    dnf install -y --setopt=install_weak_deps=False \
        perl \
        mesa-libGL \
        skopeo \
        libibverbs-utils \
        infiniband-diags \
        libibumad \
        librdmacm \
        librdmacm-utils \
        rdma-core \
        cuda-cudart-devel-13-0 \
        cuda-nvcc-13-0 && \
    dnf clean all && rm -rf /var/cache/dnf/*; \
else \
    echo "DOWNSTREAM BUILD: Skipping system packages (provided by base image)"; \
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
COPY LICENSE.md /licenses/cuda-license.md

# Copy entrypoint
COPY --chmod=0755 entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh

# Fix permissions for OpenShift (final stage)
RUN fix-permissions /opt/app-root -P \
 && chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages

USER 1001
WORKDIR /opt/app-root/src

ENTRYPOINT ["/usr/local/bin/entrypoint-universal.sh"]
CMD ["start-notebook.sh"]

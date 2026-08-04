# ROCm Universal Image Dockerfile
#
# FIPS-friendly Features:
# - Build tools are isolated in intermediate stage
# - Final image contains only runtime dependencies
# - Uses pip install (additive) to preserve base image packages
#
# Build Modes:
# - Midstream (default): DOWNSTREAM=false - installs system packages and sets env vars
# - Downstream: DOWNSTREAM=true - skips midstream-only sections (base image has them)

################################################################################
# Build Arguments
################################################################################
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-rocm-py312-ubi9:3.5-v1.46
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

# Install dependencies from AIPCC GA index
RUN uv pip install --no-cache-dir \
    --index-strategy=unsafe-best-match \
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.5/rocm7.14-ubi9/simple/ \
    -r /tmp/deps/requirements.txt

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

LABEL name="th-torch-rocm-py312" \
      summary="ROCm 7.14 Python 3.12 Training Hub 09 PyTorch 2.11.0 image on UBI9" \
      description="Universal ROCm image combining Jupyter workbench and runtime ML stack on UBI9" \
      io.k8s.display-name="ROCm Python 3.12 Universal Training Image" \
      io.k8s.description="Universal ROCm training image: Jupyter workbench by default; runtime when command provided."

USER 0
WORKDIR /opt/app-root/src

################################################################################
# MIDSTREAM ONLY: Environment variables and system packages
# Controlled by ARG DOWNSTREAM (default: false)
# - DOWNSTREAM=false (midstream): Installs ROCm dev tools, RDMA packages, sets env vars
# - DOWNSTREAM=true: Skips this section (AIPCC base image has everything pre-configured)
################################################################################
ARG DOWNSTREAM

# ROCm 7.14 environment variables
# ROCM_HOME=/opt/rocm/core per ROCm 7.14 layout (backward compat symlinks at /opt/rocm/)
ENV ROCM_HOME=/opt/rocm/core \
    HIP_PATH=/opt/rocm/core \
    PATH=/opt/rocm/core/bin:/opt/rocm/core/hip/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib64/glibc-compat:/opt/rocm/core/lib:/opt/rocm/core/lib64:${LD_LIBRARY_PATH}

# PyArrow TLS workaround (RHELAI-2417)
ENV LD_PRELOAD=/usr/lib64/libjemalloc.so.2

# System packages (MIDSTREAM ONLY - skipped when DOWNSTREAM=true)
# ROCm 7.14 uses THEROCK repositories (amdrocm-* packages)
COPY rocm.repo mellanox.repo /etc/yum.repos.d/

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
        amdrocm-llvm \
        amdrocm-runtime \
        amdrocm-runtime-devel \
        amdrocm-core-sdk \
        amdrocm-core-devel \
        protobuf && \
    dnf clean all && rm -rf /var/cache/dnf/* && \
    # Backward compat symlinks for /opt/rocm
    ln -sfn /opt/rocm/core/include /opt/rocm/include && \
    ln -sfn /opt/rocm/core/lib /opt/rocm/lib && \
    ln -sfn /opt/rocm/core/bin /opt/rocm/bin && \
    ln -sfn /opt/rocm/core/share /opt/rocm/share; \
else \
    echo "DOWNSTREAM BUILD: Skipping system packages (provided by base image)"; \
fi
################################################################################
# END MIDSTREAM ONLY
################################################################################

# Copy Python site-packages and CLI entry points from builder stage
ARG PYTHON_VERSION
COPY --from=builder /opt/app-root/lib/python${PYTHON_VERSION}/site-packages /opt/app-root/lib/python${PYTHON_VERSION}/site-packages
COPY --from=builder /opt/app-root/bin /opt/app-root/bin

# Remove uv from final image (not needed at runtime)
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

# CPU Training Image Dockerfile
#
# FIPS-friendly Features:
# - uv is used only in build stage (not shipped in runtime image)
# - Build tools are isolated in intermediate stages
# - Final image contains only runtime dependencies

################################################################################
# Build Arguments
################################################################################
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-cpu-py312-ubi9:2025b-v1.39
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

LABEL name="cpu:py312-torch290" \
      summary="CPU Python 3.12 image with PyTorch 2.9.0" \
      description="CPU image combining minimal Jupyter workbench and runtime ML stack (PyTorch 2.9.0) on UBI9" \
      io.k8s.display-name="CPU Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="CPU image: Jupyter workbench by default; runtime when command provided."

# Copy license file
COPY LICENSE.md /licenses/cpu-license.md

USER 0
WORKDIR /opt/app-root/bin

################################################################################
# System Dependencies Stage
################################################################################
FROM base AS system-deps

USER 0
WORKDIR /opt/app-root/bin

# Install build toolchain (from UBI repos)
# - gcc, gcc-c++, make: C/C++ compilation tools
# - python3-devel: Python headers for building native extensions
# - cmake: Build system (required by some Python packages)
# - git: Version control (some pip installs need it)
RUN dnf install -y --setopt=install_weak_deps=False \
    gcc \
    gcc-c++ \
    make \
    python3-devel \
    cmake \
    git && dnf clean all && rm -rf /var/cache/dnf/*

################################################################################
# Python Dependencies Stage
################################################################################
FROM system-deps AS python-deps

USER 0
WORKDIR /tmp/deps

# Copy uv from builder stage (FIPS: uv only used during build, not in runtime)
COPY --from=builder /opt/app-root/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY --chown=1001:0 pyproject.toml pylock.toml ./

# Switch to user 1001 for pip installations
USER 1001
WORKDIR /opt/app-root/src

# Install main dependencies from pylock.toml using uv pip sync
ENV UV_NO_CACHE=1
RUN uv pip sync --python-platform=linux --python-version=3.12 /tmp/deps/pylock.toml
ENV UV_NO_CACHE=

# Install kubeflow-sdk from Git (not in pylock.toml)
# TODO: use aipcc index
RUN pip install --retries 5 --timeout 300 --no-cache-dir \
    "git+https://github.com/opendatahub-io/kubeflow-sdk@main"

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

# FIPS-friendly: Remove uv from final image
RUN rm -f /opt/app-root/bin/uv

# Copy license file
COPY LICENSE.md /licenses/cpu-license.md

# Copy entrypoint
COPY --chmod=0755 entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh

# Fix permissions for OpenShift (final stage)
RUN fix-permissions /opt/app-root -P \
 && chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages

USER 1001
WORKDIR /opt/app-root/src

ENTRYPOINT ["/usr/local/bin/entrypoint-universal.sh"]
CMD ["start-notebook.sh"]


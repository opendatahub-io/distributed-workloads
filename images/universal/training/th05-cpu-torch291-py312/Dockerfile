# TH05 CPU Training Image Dockerfile
#
# FIPS-friendly Features:
# - Build tools are isolated in intermediate stage
# - Final image contains only runtime dependencies
# - Uses pip install (additive) to preserve base image packages

################################################################################
# Build Arguments
################################################################################
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-cpu-py312-ubi9:3.4_ea1-v1.41
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
# Strict AIPCC-only - no fallback indexes (request missing packages be added to AIPCC)
RUN uv pip install --no-cache --require-hashes \
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4-EA1/cpu-ubi9/simple/ \
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

LABEL name="universal:th05-cpu-py312-torch291" \
      summary="TH05 CPU Python 3.12 image with PyTorch 2.9.1" \
      description="Universal CPU image combining minimal Jupyter workbench and runtime ML stack (PyTorch 2.9.1) on UBI9" \
      io.k8s.display-name="TH05 CPU Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="Universal CPU image: Jupyter workbench by default; runtime when command provided."

USER 0
WORKDIR /opt/app-root/src

################################################################################
# MIDSTREAM ONLY: System packages
# These DNF installs are only needed for midstream builds where the base image
# doesn't include these packages. In downstream (Konflux), these come from the
# certified base image and this section should be removed.
################################################################################
RUN dnf install -y --setopt=install_weak_deps=False \
    # Core OS packages
    perl \
    mesa-libGL \
    skopeo && \
    dnf clean all && rm -rf /var/cache/dnf/*
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

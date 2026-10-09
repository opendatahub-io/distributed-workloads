# TH CPU Universal Image Dockerfile (torch 2.13.0)
#
# FIPS-friendly Features:
# - Build tools are isolated in intermediate stage
# - Final image contains only runtime dependencies
# - Uses pip install (additive) to preserve base image packages
#
# Build Modes:
# - Midstream (default): DOWNSTREAM=false - installs system packages
# - Downstream: DOWNSTREAM=true - skips midstream-only sections (base image has them)

################################################################################
# Build Arguments
################################################################################
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-cpu-py312-ubi9:3.6_ea2-v1.49
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

# Install dependencies from AIPCC prod index
RUN uv pip install --no-cache-dir --no-deps \
    --index-strategy=unsafe-best-match \
    # For RHOAI 3.6GA "prod" suffix to be removed from index-url - https://redhat.atlassian.net/browse/AIPCC-32410
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.6-EA2/cpu-ubi9-prod/simple/ \
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

LABEL name="th-torch-cpu-py312" \
      summary="CPU Python 3.12 Training Hub 10 PyTorch 2.13.0 image on UBI9" \
      description="Universal CPU training image combining Jupyter workbench and runtime ML stack on UBI9" \
      io.k8s.display-name="CPU Python 3.12 Universal Training Image" \
      io.k8s.description="Universal CPU training image: Jupyter workbench by default; runtime when command provided."

USER 0
WORKDIR /opt/app-root/src

################################################################################
# MIDSTREAM ONLY: System packages
# Controlled by ARG DOWNSTREAM (default: false)
# - DOWNSTREAM=false (midstream): Installs system packages
# - DOWNSTREAM=true: Skips this section (AIPCC base image has everything pre-configured)
################################################################################
ARG DOWNSTREAM

RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    echo "MIDSTREAM BUILD: Installing system packages..." && \
    dnf install -y --setopt=install_weak_deps=False \
        perl \
        mesa-libGL \
        protobuf \
        skopeo && \
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

# polyfile-weave registers cv2.typing stubs that trigger import cv2 during
# pkgutil scanning in multiprocessing.spawn, which fails on UBI9
RUN uv pip uninstall polyfile-weave 2>/dev/null || true

# Note: uv is kept in the final image — needed at runtime by openpipe-art

# The SDK replaces the container entrypoint and execs an MPI launcher directly,
# so uid_entrypoint never runs on the launcher. The wrapper is linked at
# /usr/local/bin/mpirun and /usr/local/bin/mpiexec, which PATH lists before
# /usr/lib64/openmpi/bin.
RUN dnf install -y --setopt=install_weak_deps=False openssh-server && dnf clean all \
 && mkdir -p /var/run/sshd /home/mpiuser/.openmpi \
 && test -x /usr/lib64/openmpi/bin/mpirun \
    || { echo "ERROR: base image has no OpenMPI at /usr/lib64/openmpi"; exit 1; }

COPY mpiuser-sshd.conf /home/mpiuser/.sshd_config

# The wrapper handles arbitrary OpenShift UIDs before launching MPI. Keep the
# MPI helper names available in standard locations for OpenMPI's remote sshd.
COPY --chmod=0755 openshift-uid-exec.sh /usr/local/bin/openshift-uid-exec.sh
RUN ln -sfn openshift-uid-exec.sh /usr/local/bin/mpirun \
 && ln -sfn openshift-uid-exec.sh /usr/local/bin/mpiexec \
 && ln -sfn openshift-uid-exec.sh /usr/local/bin/uid_entrypoint.sh \
 && ln -sfn /usr/lib64/openmpi/bin/orted /usr/bin/orted \
 && ln -sfn /usr/lib64/openmpi/bin/mpiexec /usr/bin/mpiexec \
 && if [ ! -e /usr/local/bin/python ]; then \
      printf '#!/bin/sh\nexec /opt/app-root/bin/python "$@"\n' > /usr/local/bin/python \
      && chmod 0755 /usr/local/bin/python; \
    fi \
 && grep -qxF 'LD_LIBRARY_PATH=/usr/lib64/openmpi/lib' /etc/environment \
    || echo 'LD_LIBRARY_PATH=/usr/lib64/openmpi/lib' >> /etc/environment \
 && chgrp -R 0 /home/mpiuser && chmod -R g=u /home/mpiuser \
 && chmod g=u /etc/passwd

ENV MPI_HOME=/usr/lib64/openmpi \
    PATH=/usr/local/bin:/usr/lib64/openmpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH}

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


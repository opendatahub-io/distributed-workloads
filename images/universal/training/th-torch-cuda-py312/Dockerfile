# Universal CUDA Image Dockerfile (CUDA 13.0.2, torch 2.13.0)
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
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-cuda-py312-ubi9:3.6_ea2-v1.49
ARG PYTHON_VERSION=3.12
ARG DOWNSTREAM=false
ARG OPENMPI_VERSION=4.1.6
ARG UCX_VERSION=1.19.1
ARG UCX_SHA256=0fe68f2592b7b378bdac7ccb78331e9e8c5be250f40be118edb1795e94c60209
ARG OPENMPI_SHA256=44da277b8cdc234e71c62473305a09d63f4dcca292ca40335aab7c4bf0e6a566

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
RUN UV_LINK_MODE=copy uv pip install --no-cache-dir --no-deps \
    --index-strategy=unsafe-best-match \
    # For RHOAI 3.6GA "prod" suffix to be removed from index-url - https://redhat.atlassian.net/browse/AIPCC-32410
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.6-EA2/cuda13.0-ubi9-prod/simple/ \
    -r /tmp/deps/requirements.txt

# Fix permissions for OpenShift
ARG PYTHON_VERSION
USER 0
RUN chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages \
 && fix-permissions /opt/app-root -P

# Clean up
RUN rm -rf /tmp/deps

################################################################################
# Build Stage - CUDA-aware OpenMPI from upstream sources
################################################################################
FROM ${BASE_IMAGE} AS openmpi-builder

ARG DOWNSTREAM
ARG OPENMPI_VERSION
ARG UCX_VERSION
ARG UCX_SHA256
ARG OPENMPI_SHA256

USER 0

COPY cuda.repo /etc/yum.repos.d/
# Replace the notebook base's UCX with the CUDA-enabled OpenUCX release.
# The OpenMPI build uses the OS and CUDA repositories, not Mellanox OFED.
RUN mkdir -p /tmp/ucx-rpms \
 && if [ "${DOWNSTREAM}" != "true" ]; then \
    dnf install -y --setopt=install_weak_deps=False --disablerepo='mlnx_ofed*' \
        curl \
        gcc \
        gcc-c++ \
        make \
        perl \
        tar \
        gzip \
        bzip2 \
        cuda-cudart-devel-13-0 \
        rdma-core-devel \
        libibverbs-devel \
        librdmacm-devel \
        hwloc-devel \
        libevent-devel \
        pmix-devel \
        zlib-devel \
 && curl -fsSLo /tmp/ucx.tar.bz2 \
        "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}-rocky9-mofed24.10-cuda13-x86_64.tar.bz2" \
 && echo "${UCX_SHA256}  /tmp/ucx.tar.bz2" | sha256sum -c - \
 && tar -xjf /tmp/ucx.tar.bz2 -C /tmp/ucx-rpms \
 && rpm -Uvh --nodeps --oldpackage --replacepkgs \
        /tmp/ucx-rpms/ucx-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-cuda-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-ib-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-ib-mlx5-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-rdmacm-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-devel-${UCX_VERSION}-1.el9.x86_64.rpm \
 && dnf clean all \
 && rm -f /tmp/ucx.tar.bz2 \
        "/tmp/ucx-rpms/ucx-devel-${UCX_VERSION}-1.el9.x86_64.rpm"; \
 fi

RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    curl -fsSLo /tmp/openmpi.tar.gz \
        "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz" \
 && echo "${OPENMPI_SHA256}  /tmp/openmpi.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/openmpi.tar.gz -C /tmp \
 && cd "/tmp/openmpi-${OPENMPI_VERSION}" \
 && rm -rf /usr/lib64/openmpi \
 && ./configure \
        --prefix=/usr/lib64/openmpi \
        --with-cuda=/usr/local/cuda \
        --with-ucx=/usr \
        --with-verbs \
        --with-hwloc \
        --with-libevent=external \
        --with-pmix=external \
        --enable-mpi-cxx \
        --enable-mpi1-compatibility \
        --disable-silent-rules \
 && make -j"$(nproc)" install \
 && cd / \
 && rm -rf /tmp/openmpi.tar.gz "/tmp/openmpi-${OPENMPI_VERSION}"; \
 fi

RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    /usr/lib64/openmpi/bin/mpirun --version | grep -F "mpirun (Open MPI) ${OPENMPI_VERSION}" \
 && /usr/lib64/openmpi/bin/ompi_info --parsable --all | grep -Eq '(^|:)mpi_built_with_cuda_support:value:true$' \
 && /usr/lib64/openmpi/bin/ompi_info --parsable --all | grep -Eq '(^|:)opal_built_with_cuda_support:value:true$' \
 && /usr/lib64/openmpi/bin/ompi_info --parsable --param pml ucx | grep -q '^mca:pml:ucx:' \
 && ucx_info -v | grep -q -- '--with-cuda'; \
 fi

################################################################################
# Final Stage - FIPS-friendly Runtime
################################################################################
FROM ${BASE_IMAGE} AS final

LABEL name="th-torch-cuda-py312" \
      summary="CUDA 13.0.2 Python 3.12 Training Hub 10 PyTorch 2.13.0 image on UBI9" \
      description="Universal CUDA training image combining Jupyter workbench and runtime ML stack on UBI9" \
      io.k8s.display-name="CUDA Python 3.12 Universal Training Image" \
      io.k8s.description="Universal CUDA training image: Jupyter workbench by default; runtime when command provided."

USER 0
WORKDIR /opt/app-root/src

################################################################################
# MIDSTREAM ONLY: Environment variables and system packages
# Controlled by ARG DOWNSTREAM (default: false)
# - DOWNSTREAM=false (midstream): Installs CUDA dev tools, RDMA packages, sets env vars
# - DOWNSTREAM=true: Skips this section (AIPCC base image has everything pre-configured)
################################################################################
ARG DOWNSTREAM
ARG OPENMPI_VERSION

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
COPY cuda.repo /etc/yum.repos.d/

RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    echo "MIDSTREAM BUILD: Installing system packages..." && \
    dnf install -y --setopt=install_weak_deps=False --disablerepo='mlnx_ofed*' \
        perl \
        mesa-libGL \
        skopeo \
        libibverbs-utils \
        infiniband-diags \
        libibumad \
        libmlx5 \
        librdmacm \
        librdmacm-utils \
        rdma-core \
        protobuf \
        cuda-cudart-devel-13-0 \
        cuda-nvcc-13-0 \
        libnccl-2.30.7-1+cuda13.3 \
        libnccl-devel-2.30.7-1+cuda13.3 && \
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
RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    dnf remove -y --noautoremove --disablerepo='mlnx_ofed*' openmpi openmpi-devel \
    && rm -rf /usr/lib64/openmpi \
    && dnf clean all \
    && ! rpm -q openmpi openmpi-devel; \
 fi
ARG UCX_VERSION
COPY --from=openmpi-builder /tmp/ucx-rpms /tmp/ucx-rpms
RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    rpm -Uvh --nodeps --oldpackage --replacepkgs \
        /tmp/ucx-rpms/ucx-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-cuda-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-ib-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-ib-mlx5-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rpms/ucx-rdmacm-${UCX_VERSION}-1.el9.x86_64.rpm; \
 fi \
 && rm -rf /tmp/ucx-rpms
COPY --from=openmpi-builder /usr/lib64/openmpi /usr/lib64/openmpi

################################################################################
# Workarounds for vLLM 0.21 on UBI9
################################################################################

# opencv-python-headless and polyfile-weave both register cv2.typing stubs
# that trigger import cv2 during pkgutil scanning in vLLM's multiprocessing.spawn
# child processes, which fails with libavcodec.so.60 not found on UBI9
# See: https://github.com/vllm-project/vllm/issues/40741
RUN uv pip uninstall opencv-python-headless polyfile-weave 2>/dev/null || true

# tilelang ships a libcudart_stub.so that shadows the real CUDA runtime
# in vLLM subprocesses, causing CUDA initialization failures.
# However, libtvm.so has a hard DT_NEEDED on libcudart_stub.so, so deleting
# it breaks mamba_ssm -> tilelang -> tvm imports (needed for LoRA training).
# Fix: replace the stub with a symlink to the real CUDA runtime library.
RUN ln -sf /usr/local/cuda/lib64/libcudart.so \
    /opt/app-root/lib/python${PYTHON_VERSION}/site-packages/tilelang/lib/libcudart_stub.so

# Note: uv is kept in the final image — needed at runtime by openpipe-art

# The SDK replaces the container entrypoint and execs an MPI launcher directly,
# so uid_entrypoint never runs on the launcher. The wrapper is linked at
# /usr/local/bin/mpirun and /usr/local/bin/mpiexec, which PATH lists before
# /usr/lib64/openmpi/bin.
RUN dnf install -y --setopt=install_weak_deps=False --disablerepo='mlnx_ofed*' openssh-server && dnf clean all \
 && mkdir -p /var/run/sshd /home/mpiuser/.openmpi \
 && if [ "${DOWNSTREAM}" != "true" ]; then \
      /usr/lib64/openmpi/bin/mpirun --version | grep -F "mpirun (Open MPI) ${OPENMPI_VERSION}"; \
    else \
      test -x /usr/lib64/openmpi/bin/mpirun; \
    fi

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
    LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/usr/lib64/ucx:${LD_LIBRARY_PATH}

RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    /usr/lib64/openmpi/bin/ompi_info --parsable --all | grep -Eq '(^|:)mpi_built_with_cuda_support:value:true$' \
    && /usr/lib64/openmpi/bin/ompi_info --parsable --all | grep -Eq '(^|:)opal_built_with_cuda_support:value:true$' \
    && /usr/lib64/openmpi/bin/ompi_info --parsable --param pml ucx | grep -q '^mca:pml:ucx:' \
    && ucx_info -v | grep -q -- '--with-cuda'; \
 fi

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

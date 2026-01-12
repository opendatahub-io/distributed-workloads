## Global Args ######################################################
ARG IMAGE_TAG=9.7-1767889538
ARG PYTHON_VERSION=312

# use UBI9 latest
FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG} AS base

LABEL name="training:py312-rocm64-torch290" \
      summary="ROCm 6.4 Python 3.12 PyTorch 2.9.0 image based on UBI9 for Training" \
      description="ROCm 6.4 Python 3.12 PyTorch 2.9.0 image based on UBI9 for Training" \
      io.k8s.display-name="ROCm 6.4 Python 3.12 PyTorch 2.9.0 base image for Training" \
      io.k8s.description="ROCm 6.4 Python 3.12 PyTorch 2.9.0 image based on UBI9 for Training" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Copy license
COPY LICENSE.md /licenses/rocm-license.md

# Set the working directory in the container
USER 0

# Install ROCm
WORKDIR /opt/app-root/bin

ARG ROCM_VERSION=6.4.3
ARG AMDGPU_VERSION=6.4.3

ENV GPU_ARCHS=gfx90a;gfx942
# ROCm toolchain env (needed for HIP builds like flash-attn)
ENV ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH \
    CMAKE_PREFIX_PATH=/opt/rocm

RUN <<EOF
cat <<EOD > /etc/yum.repos.d/rocm.repo
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/rhel/9.4/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key

[ROCm-6.4.3]
name=ROCm6.4.3
baseurl=https://repo.radeon.com/rocm/rhel9/$ROCM_VERSION/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOD
EOF

RUN dnf install -y cmake rocm-developer-tools rocm-ml-sdk rocm-openmp-sdk rocm-utils && dnf clean all && rm -rf /var/cache/dnf

# HIP / ROCm dev toolchain (aligns with universal image so flash-attn can build)
RUN dnf install -y --setopt=install_weak_deps=False \
    rocm-llvm \
    hipcc \
    hip-devel \
    hip-runtime-amd \
    rocm-device-libs \
    rocblas-devel \
    hipblas-devel \
    hipblaslt-devel \
    hipsparse-devel \
    hipfft \
    rocfft \
    rocm-cmake \
    git \
    gcc \
    gcc-c++ \
    make \
    python3-devel && \
    dnf clean all && rm -rf /var/cache/dnf/*

# Install Python packages

# Install micropipenv to deploy packages from Pipfile.lock
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Install Python dependencies from Pipfile.lock file
COPY Pipfile.lock ./

RUN micropipenv install -- --no-cache-dir && \
    rm -f ./Pipfile.lock && \
    # Fix permissions to support pip in OpenShift environments \
    chmod -R g+w /opt/app-root/lib/python3.12/site-packages && \
    fix-permissions /opt/app-root -P

# Install BitsAndBytes (ROCm-enabled version from source)
RUN export TMP_DIR=$(mktemp -d) \
    && cd $TMP_DIR \
    && git clone --depth 1 --branch rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes.git \
    && cd bitsandbytes \
    && cmake -S . \
    && make \
    && cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=$GPU_ARCHS -S . \
    && make \
    && pip install --no-deps . \
    && rm -rf $TMP_DIR

# Reapply write permissions on site‑packages
RUN chmod -R g+w /opt/app-root/lib/python3.12/site-packages && \
    fix-permissions /opt/app-root -P

# Install flash-attn (ROCm build) outside Pipfile.lock; requires GPU_ARCHS and ROCm toolchain
RUN pip install --no-build-isolation --no-cache-dir --no-deps flash-attn==2.8.3 && \
    fix-permissions /opt/app-root -P

# Restore user workspace
USER 1001

WORKDIR /opt/app-root/src

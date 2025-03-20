## Global Args ######################################################
ARG IMAGE_TAG=9.5-1741671866
ARG PYTHON_VERSION=311

# use UBI9 latest
FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG} AS base

LABEL name="training:py311-rocm62-torch251" \
      summary="ROCm 6.2 Python 3.11 PyTorch 2.5.1 image based on UBI9 for Training" \
      description="ROCm 6.2 Python 3.11 PyTorch 2.5.1 image based on UBI9 for Training" \
      io.k8s.display-name="ROCm 6.2 Python 3.11 PyTorch 2.5.1 base image for Training" \
      io.k8s.description="ROCm 6.2 Python 3.11 PyTorch 2.5.1 image based on UBI9 for Training" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Copy license
COPY LICENSE.md /licenses/rocm-license.md

# Set the working directory in the container
USER 0
WORKDIR /app

# upgrade requests package
RUN pip install --no-cache-dir --upgrade requests==2.32.3

# Install ROCm
WORKDIR /opt/app-root/bin

ARG ROCM_VERSION=6.2.4
ARG AMDGPU_VERSION=6.2.4

RUN <<EOF
cat <<EOD > /etc/yum.repos.d/rocm.repo
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/rhel/9.4/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key

[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel9/$ROCM_VERSION/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOD
EOF

RUN dnf install -y cmake rocm-developer-tools rocm-ml-sdk rocm-opencl-sdk rocm-openmp-sdk rocm-utils && dnf clean all && rm -rf /var/cache/dnf

# Install Python packages

# Install micropipenv to deploy packages from Pipfile.lock
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Install Python dependencies from Pipfile.lock file
COPY Pipfile.lock ./

RUN micropipenv install && \
    rm -f ./Pipfile.lock && \
    # Fix permissions to support pip in OpenShift environments \
    chmod -R g+w /opt/app-root/lib/python3.11/site-packages && \
    fix-permissions /opt/app-root -P

# Install Flash Attention
ENV GPU_ARCHS=gfx90a;gfx941;gfx942

RUN pip install wheel ninja

RUN export TMP_DIR=$(mktemp -d) \
    && cd $TMP_DIR \
    && git clone --depth 1 --branch v2.7.4 https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && git submodule update --init \
    && MAX_JOBS="16" python3 setup.py install --verbose \
    && rm -rf $TMP_DIR

# Install BitsAndBytes
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
# Install a compatible version of pytorch-triton-rocm
RUN pip install --force-reinstall pytorch-triton-rocm==3.1.0 --index-url https://download.pytorch.org/whl/nightly/rocm6.2

# Restore user workspace
USER 1001

WORKDIR /opt/app-root/src

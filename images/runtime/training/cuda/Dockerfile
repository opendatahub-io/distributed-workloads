## Global Args ######################################################
ARG IMAGE_TAG=9.5-1741671866
ARG PYTHON_VERSION=311

# use UBI9
FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

LABEL name="training:py311-cuda121-torch241" \
      summary="CUDA 12.1 Python 3.11 PyTorch 2.4.1 image based on UBI9 for Training" \
      description="CUDA 12.1 Python 3.11 PyTorch 2.4.1 image based on UBI9 for Training" \
      io.k8s.display-name="CUDA 12.1 Python 3.11 PyTorch 2.4.1 base image for Training" \
      io.k8s.description="CUDA 12.1 Python 3.11 PyTorch 2.4.1 image based on UBI9 for Training" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Copy license
COPY LICENSE.md /licenses/cuda-license.md

# Set the working directory in the container
USER 0
WORKDIR /app

# upgrade requests package
RUN pip install --no-cache-dir --upgrade requests==2.32.3

# Install CUDA
WORKDIR /opt/app-root/bin

# Ref: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/
ENV CUDA_VERSION=12.1.0 \
    NVIDIA_REQUIRE_CUDA="cuda>=12.1 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526" \
    NV_CUDA_LIB_VERSION=12.1.0-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=12.1.55-1 \
    NV_CUDA_COMPAT_VERSION=530.30.02-1 \
    NV_CUDA_NVCC_VERSION=12.1.66-1

# Ref: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.1.1/ubi9/base/Dockerfile
# nvcc is required for Flash Attention
RUN dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
 && dnf install -y \
     cuda-cudart-12-1-${NV_CUDA_CUDART_VERSION} \
     cuda-compat-12-1-${NV_CUDA_COMPAT_VERSION} \
     cuda-nvcc-12-1-${NV_CUDA_NVCC_VERSION} \
 && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
 && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
 PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
 LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

# Install InfiniBand and RDMA packages
RUN dnf config-manager \
        --add-repo https://linux.mellanox.com/public/repo/mlnx_ofed/latest/rhel9.5/mellanox_mlnx_ofed.repo \
    && dnf install -y \
        libibverbs-utils \
        infiniband-diags \
        libibumad3 \
        librdmacm \
        librdmacm-utils \
        rdma-core \
        mlnx-tools \
    && dnf clean all \
    && rm -rf /var/cache/dnf/*

# Install Python packages

# Install micropipenv to deploy packages from Pipfile.lock
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Install Python dependencies from Pipfile.lock file
COPY Pipfile.lock ./

RUN micropipenv install -- --no-cache-dir && \
    rm -f ./Pipfile.lock && \
    # Fix permissions to support pip in OpenShift environments \
    chmod -R g+w /opt/app-root/lib/python3.11/site-packages && \
    fix-permissions /opt/app-root -P

# Install Flash Attention
RUN pip install wheel
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Upgrade NCCL to a more recent version until we upgrade torch
RUN pip install nvidia-nccl-cu12==2.26.2 && \
    fix-permissions /opt/app-root -P

# Restore user workspace
USER 1001

WORKDIR /opt/app-root/src

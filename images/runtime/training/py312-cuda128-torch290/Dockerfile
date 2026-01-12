## Global Args ######################################################
ARG IMAGE_TAG=9.7-1767889538
ARG PYTHON_VERSION=312

# use UBI9
FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

LABEL name="training:py312-cuda128-torch290" \
      summary="CUDA 12.8 Python 3.12 PyTorch 2.9.0 image based on UBI9 for Training" \
      description="CUDA 12.8 Python 3.12 PyTorch 2.9.0 image based on UBI9 for Training" \
      io.k8s.display-name="CUDA 12.8 Python 3.12 PyTorch 2.9.0 base image for Training" \
      io.k8s.description="CUDA 12.8 Python 3.12 PyTorch 2.9.0 image based on UBI9 for Training" \
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

# Ref: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/
ENV CUDA_VERSION=12.8.0 \
    NVIDIA_REQUIRE_CUDA="cuda>=12.8 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526 brand=tesla,driver>=570,driver<571 brand=unknown,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=geforce,driver>=570,driver<571 brand=geforcertx,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=titan,driver>=570,driver<571 brand=titanrtx,driver>=570,driver<571" \
    NV_CUDA_LIB_VERSION=12.8.0-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=12.8.57-1 \
    NV_CUDA_COMPAT_VERSION=3:570.172.08-1.el9 \
    NV_CUDA_NVCC_VERSION=12.8.93-1

# Ref: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.8.0/ubi9/base/Dockerfile
# nvcc is required for Flash Attention
RUN dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
 && dnf install -y \
     --disablerepo=rhel-9-for-x86_64-baseos-rpms \
     --disablerepo=rhel-9-for-x86_64-appstream-rpms \
     cuda-cudart-12-8-${NV_CUDA_CUDART_VERSION} \
     cuda-cudart-devel-12-8-${NV_CUDA_CUDART_VERSION} \
     cuda-libraries-devel-12-8-${NV_CUDA_LIB_VERSION} \
     cuda-compat-12-8-${NV_CUDA_COMPAT_VERSION} \
     cuda-nvcc-12-8-${NV_CUDA_NVCC_VERSION} \
 && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
 && dnf clean all

# Add build tools for compiling PyTorch extensions
RUN dnf install -y --disablerepo="*" --enablerepo="ubi-9-appstream-rpms,ubi-9-baseos-rpms" \
        gcc-c++ \
        make \
        cmake \
        git \
    && dnf clean all \
    && rm -rf /var/cache/dnf/*

ENV CUDA_HOME="/usr/local/cuda" \
 PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
 LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

# Install InfiniBand and RDMA packages
RUN dnf config-manager \
        --add-repo https://linux.mellanox.com/public/repo/mlnx_ofed/latest/rhel9.5/mellanox_mlnx_ofed.repo 

RUN dnf install -y --disablerepo="*" --enablerepo="cuda-rhel9-x86_64,mlnx_ofed_24.10-1.1.4.0_base,ubi-9-appstream-rpms,ubi-9-baseos-rpms" \
        libibverbs-utils \
        infiniband-diags \
        libibumad3 \
        librdmacm \
        librdmacm-utils \
        rdma-core \
        mlnx-tools \
    && dnf clean all \
    && rm -rf /var/cache/dnf/*

# Upgrade NCCL to a more recent version and add Training Hub NVIDIA dependencies
RUN pip install \
    nvidia-nccl-cu12==2.27.3 \
    nvidia-cublas-cu12==12.8.4.1 \
    nvidia-cuda-cupti-cu12==12.8.90 \
    nvidia-cuda-nvrtc-cu12==12.8.93 \
    nvidia-cuda-runtime-cu12==12.8.90 \
    nvidia-cudnn-cu12==9.10.2.21 \
    nvidia-cufft-cu12==11.3.3.83 \
    nvidia-cufile-cu12==1.13.1.3 \
    nvidia-curand-cu12==10.3.9.90 \
    nvidia-cusolver-cu12==11.7.3.90 \
    nvidia-cusparse-cu12==12.5.8.93 \
    nvidia-cusparselt-cu12==0.7.1 \
    nvidia-nvjitlink-cu12==12.8.93 \
    nvidia-nvtx-cu12==12.8.90 \
    && fix-permissions /opt/app-root -P

# Ensure cuDNN from pip is discoverable during source builds
ENV LD_LIBRARY_PATH="/opt/app-root/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
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

# Build helpers to compile PyTorch extensions
RUN pip install -U pip setuptools wheel ninja cmake

# Optional: set GPU archs if you hit arch issues
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Deterministic 2-step: sub-dep first, then parent without deps
RUN pip install --no-build-isolation --no-cache-dir causal-conv1d==1.5.3.post1 && \
    pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.6.post3 --no-deps && \
    fix-permissions /opt/app-root -P

# Install flash-attn separately (keep out of Pipfile.lock to avoid lock-time builds)
RUN pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3 && \
    fix-permissions /opt/app-root -P

# Restore user workspace
USER 1001

WORKDIR /opt/app-root/src

## Global Args ######################################################
ARG IMAGE_TAG=1-77.1729776556
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
    NV_CUDA_COMPAT_VERSION=530.30.02-1

RUN dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
 && dnf install -y \
     cuda-cudart-12-1-${NV_CUDA_CUDART_VERSION} \
     cuda-compat-12-1-${NV_CUDA_COMPAT_VERSION} \
 && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
 && dnf clean all

RUN dnf -y install --allowerasing cudnn9-cuda-12

ENV CUDA_HOME="/usr/local/cuda" \
 PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
 LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH" \
 LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH"

# Ref: https://developer.nvidia.com/nccl/nccl-legacy-downloads
ENV NV_CUDA_CUDART_DEV_VERSION=12.1.55-1 \
    NV_NVML_DEV_VERSION=12.1.55-1 \
    NV_LIBCUBLAS_DEV_VERSION=12.1.0.26-1 \
    NV_LIBNPP_DEV_VERSION=12.0.2.50-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.18.3-1+cuda12.1

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-command-line-tools-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-12-1-${NV_CUDA_CUDART_DEV_VERSION} \
        cuda-nvml-devel-12-1-${NV_NVML_DEV_VERSION} \
        libcublas-devel-12-1-${NV_LIBCUBLAS_DEV_VERSION} \
        libnpp-devel-12-1-${NV_LIBNPP_DEV_VERSION} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

# Install CUDA devel cudnn8 from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.1.1/ubi9/devel/cudnn8/Dockerfile
ENV NV_CUDNN_VERSION=8.9.0.131-1
ENV NV_CUDNN_PACKAGE=libcudnn8-${NV_CUDNN_VERSION}.cuda12.1
ENV NV_CUDNN_PACKAGE_DEV=libcudnn8-devel-${NV_CUDNN_VERSION}.cuda12.1

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN dnf install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && dnf clean all \
    && rm -rf /var/cache/dnf/*

# Install Python packages

# Install micropipenv to deploy packages from Pipfile.lock
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Install Python dependencies from Pipfile.lock file
COPY Pipfile.lock ./

RUN micropipenv install && rm -f ./Pipfile.lock

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src

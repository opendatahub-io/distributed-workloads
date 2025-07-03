ARG PYTHON_VERSION=311
ARG IMAGE_TAG=9.6-1750969934

FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

LABEL name="ray-ubi9-py311-cu121" \
      summary="CUDA 12.1 Python 3.11 image based on UBI9 for Ray" \
      description="CUDA 12.1 Python 3.11 image based on UBI9 for Ray" \
      io.k8s.display-name="CUDA 12.1 Python 3.11 base image for Ray" \
      io.k8s.description="CUDA 12.1 Python 3.11 image based on UBI9 for Ray" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Install CUDA base from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.1.1/ubi9/base/Dockerfile
USER 0
WORKDIR /opt/app-root/bin

ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.1 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526"
ENV NV_CUDA_CUDART_VERSION=12.1.105-1

COPY cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo

RUN NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

ENV CUDA_VERSION=12.1.1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN yum upgrade -y && yum install -y \
    cuda-cudart-12-1-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-12-1 \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# nvidia-docker 1.0
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install CUDA runtime from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.1.1/ubi9/runtime/Dockerfile
ENV NV_CUDA_LIB_VERSION=12.1.1-1
ENV NV_NVTX_VERSION=12.1.105-1
ENV NV_LIBNPP_VERSION=12.1.0.40-1
ENV NV_LIBNPP_PACKAGE=libnpp-12-1-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION=12.1.3.1-1
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION=2.17.1-1
ENV NV_LIBNCCL_VERSION=2.17.1
ENV NCCL_VERSION=2.17.1
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.1

RUN yum install -y \
    cuda-libraries-12-1-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-12-1-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-12-1-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Set this flag so that libraries can find the location of CUDA
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Install CUDA devel from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.1.1/ubi9/devel/Dockerfile
ENV NV_CUDA_LIB_VERSION=12.1.1-1
ENV NV_NVPROF_VERSION=12.1.105-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-1-${NV_NVPROF_VERSION}
ENV NV_CUDA_CUDART_DEV_VERSION=12.1.105-1
ENV NV_NVML_DEV_VERSION=12.1.105-1
ENV NV_LIBCUBLAS_DEV_VERSION=12.1.3.1-1
ENV NV_LIBNPP_DEV_VERSION=12.1.0.40-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-devel-12-1-${NV_LIBNPP_DEV_VERSION}
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-devel
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.17.1-1
ENV NCCL_VERSION=2.17.1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.1
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.1.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-1-${NV_CUDA_NSIGHT_COMPUTE_VERSION}

RUN yum install -y \
    make \
    findutils \
    cuda-command-line-tools-12-1-${NV_CUDA_LIB_VERSION} \
    cuda-libraries-devel-12-1-${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-1-${NV_CUDA_LIB_VERSION} \
    cuda-cudart-devel-12-1-${NV_CUDA_CUDART_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    cuda-nvml-devel-12-1-${NV_NVML_DEV_VERSION} \
    libcublas-devel-12-1-${NV_LIBCUBLAS_DEV_VERSION} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# Install CUDA devel cudnn8 from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.1.1/ubi9/devel/cudnn8/Dockerfile
ENV NV_CUDNN_VERSION=8.9.0.131-1
ENV NV_CUDNN_PACKAGE=libcudnn8-${NV_CUDNN_VERSION}.cuda12.1
ENV NV_CUDNN_PACKAGE_DEV=libcudnn8-devel-${NV_CUDNN_VERSION}.cuda12.1

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Install Python packages

# Install micropipenv to deploy packages from Pipfile.lock
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Install Python dependencies from Pipfile.lock file
COPY Pipfile.lock ./

RUN micropipenv install && rm -f ./Pipfile.lock

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src

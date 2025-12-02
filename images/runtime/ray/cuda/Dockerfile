ARG PYTHON_VERSION=312
ARG IMAGE_TAG=9.7-1764607356

FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

ARG TARGETARCH

LABEL name="ray-ubi9-py312-cu128" \
      summary="CUDA 12.8 Python 3.12 image based on UBI9 for Ray" \
      description="CUDA 12.8 Python 3.12 image based on UBI9 for Ray" \
      io.k8s.display-name="CUDA 12.8 Python 3.12 base image for Ray" \
      io.k8s.description="CUDA 12.8 Python 3.12 image based on UBI9 for Ray" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Install CUDA base from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.8.0/ubi9/base/Dockerfile
USER 0
WORKDIR /opt/app-root/bin

ENV NVIDIA_REQUIRE_CUDA="cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566"
ENV NV_CUDA_CUDART_VERSION=12.8.57-1

RUN NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87 && \
    if [ "${TARGETARCH}" = "arm64" ]; then NVARCH=sbsa; else NVARCH=x86_64; fi && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

ENV CUDA_VERSION=12.8.0

COPY cuda.repo-* ./
COPY NGC-DL-CONTAINER-LICENSE /

RUN if [ "${TARGETARCH}" = "arm64" ]; then \
        cp cuda.repo-arm64 /etc/yum.repos.d/cuda.repo; \
    else \
        cp cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo; \
    fi

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN yum upgrade -y && yum install -y \
    cuda-cudart-12-8-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-12-8 \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# nvidia-docker 1.0
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install CUDA runtime from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.8.0/ubi9/runtime/Dockerfile
ENV NV_CUDA_LIB_VERSION=12.8.0-1
ENV NV_NVTX_VERSION=12.8.55-1
ENV NV_LIBNPP_VERSION=12.3.3.65-1
ENV NV_LIBNPP_PACKAGE=libnpp-12-8-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION=12.8.3.14-1
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION=2.25.1-1
ENV NV_LIBNCCL_VERSION=2.25.1
ENV NCCL_VERSION=2.25.1
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.8

RUN yum install -y \
    cuda-libraries-12-8-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-12-8-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-12-8-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Set this flag so that libraries can find the location of CUDA
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Install CUDA devel from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.8.0/ubi9/devel/Dockerfile
ENV NV_CUDA_LIB_VERSION=12.8.0-1
# ARM64 doesn't have nvprof package - set in runtime
ENV NV_NVPROF_VERSION=12.8.57-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-8-${NV_NVPROF_VERSION}
ENV NV_CUDA_CUDART_DEV_VERSION=12.8.57-1
ENV NV_NVML_DEV_VERSION=12.8.55-1
ENV NV_LIBCUBLAS_DEV_VERSION=12.8.3.14-1
ENV NV_LIBNPP_DEV_VERSION=12.3.3.65-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-devel-12-8-${NV_LIBNPP_DEV_VERSION}
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-devel
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.8
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.8.0-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-8-${NV_CUDA_NSIGHT_COMPUTE_VERSION}

RUN yum install -y \
    make \
    findutils \
    cuda-command-line-tools-12-8-${NV_CUDA_LIB_VERSION} \
    cuda-libraries-devel-12-8-${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-8-${NV_CUDA_LIB_VERSION} \
    cuda-cudart-devel-12-8-${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-nvml-devel-12-8-${NV_NVML_DEV_VERSION} \
    libcublas-devel-12-8-${NV_LIBCUBLAS_DEV_VERSION} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && if [ "${TARGETARCH}" != "arm64" ]; then \
        yum install -y ${NV_NVPROF_DEV_PACKAGE}; \
    fi \
    && yum clean all \
    && rm -rf /var/cache/yum/*

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# Install CUDA devel cudnn from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.8.0/ubi9/devel/cudnn/Dockerfile
ENV NV_CUDNN_VERSION=9.7.0.66-1
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-12-${NV_CUDNN_VERSION}
ENV NV_CUDNN_PACKAGE_DEV=libcudnn9-devel-cuda-12-${NV_CUDNN_VERSION}

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
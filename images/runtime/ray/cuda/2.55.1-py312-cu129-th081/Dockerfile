ARG PYTHON_VERSION=312
ARG IMAGE_TAG=9.7-1778488949

FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

ARG TARGETARCH

LABEL name="ray-ubi9-py312-cu129" \
      summary="CUDA 12.9 Python 3.12 image based on UBI9 for Ray" \
      description="CUDA 12.9 Python 3.12 image based on UBI9 for Ray" \
      io.k8s.display-name="CUDA 12.9 Python 3.12 base image for Ray" \
      io.k8s.description="CUDA 12.9 Python 3.12 image based on UBI9 for Ray" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Install CUDA base from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.9.2/ubi9/base/Dockerfile
USER 0
WORKDIR /opt/app-root/bin

ENV NVIDIA_REQUIRE_CUDA="cuda>=12.9 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=580,driver<581 brand=grid,driver>=580,driver<581 brand=tesla,driver>=580,driver<581 brand=nvidia,driver>=580,driver<581 brand=quadro,driver>=580,driver<581 brand=quadrortx,driver>=580,driver<581 brand=nvidiartx,driver>=580,driver<581 brand=vapps,driver>=580,driver<581 brand=vpc,driver>=580,driver<581 brand=vcs,driver>=580,driver<581 brand=vws,driver>=580,driver<581 brand=cloudgaming,driver>=580,driver<581"
ENV NV_CUDA_CUDART_VERSION=12.9.79-1

RUN NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87 && \
    if [ "${TARGETARCH}" = "arm64" ]; then NVARCH=sbsa; else NVARCH=x86_64; fi && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

ENV CUDA_VERSION=12.9.2

COPY cuda.repo-* ./
COPY NGC-DL-CONTAINER-LICENSE /

RUN if [ "${TARGETARCH}" = "arm64" ]; then \
        cp cuda.repo-arm64 /etc/yum.repos.d/cuda.repo; \
    else \
        cp cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo; \
    fi

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN yum upgrade -y && yum install -y \
    cuda-cudart-12-9-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-12-9 \
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
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.9.2/ubi9/runtime/Dockerfile
ENV NV_CUDA_LIB_VERSION=12.9.2-1
ENV NV_NVTX_VERSION=12.9.79-1
ENV NV_LIBNPP_VERSION=12.4.1.87-1
ENV NV_LIBNPP_PACKAGE=libnpp-12-9-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION=12.9.2.10-1
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION=2.27.3-1
ENV NV_LIBNCCL_VERSION=2.27.3
ENV NCCL_VERSION=2.27.3
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.9

RUN yum install -y \
    cuda-libraries-12-9-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-12-9-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-12-9-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Set this flag so that libraries can find the location of CUDA
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Install CUDA devel from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.9.2/ubi9/devel/Dockerfile
ENV NV_CUDA_LIB_VERSION=12.9.2-1
# ARM64 doesn't have nvprof package - set in runtime
ENV NV_NVPROF_VERSION=12.9.79-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-9-${NV_NVPROF_VERSION}
ENV NV_CUDA_CUDART_DEV_VERSION=12.9.79-1
ENV NV_NVML_DEV_VERSION=12.9.79-1
ENV NV_LIBCUBLAS_DEV_VERSION=12.9.2.10-1
ENV NV_LIBNPP_DEV_VERSION=12.4.1.87-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-devel-12-9-${NV_LIBNPP_DEV_VERSION}
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-devel
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.27.3-1
ENV NCCL_VERSION=2.27.3
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.9
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.9.2-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-9-${NV_CUDA_NSIGHT_COMPUTE_VERSION}

RUN yum install -y \
    make \
    findutils \
    cuda-command-line-tools-12-9-${NV_CUDA_LIB_VERSION} \
    cuda-libraries-devel-12-9-${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-9-${NV_CUDA_LIB_VERSION} \
    cuda-cudart-devel-12-9-${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-nvml-devel-12-9-${NV_NVML_DEV_VERSION} \
    libcublas-devel-12-9-${NV_LIBCUBLAS_DEV_VERSION} \
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
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.9.2/ubi9/devel/cudnn/Dockerfile
ENV NV_CUDNN_VERSION=9.10.2.21-1
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-12-${NV_CUDNN_VERSION}
ENV NV_CUDNN_PACKAGE_DEV=libcudnn9-devel-cuda-12-${NV_CUDNN_VERSION}

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# ---------------------------------------------------------------------------
# Install Python packages
# ---------------------------------------------------------------------------

RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Pipfile.lock provides ray, training-hub, and their transitive Python deps.
# torch is NOT in the Pipfile — it comes exclusively from the AIPCC index below.
COPY Pipfile.lock ./
RUN micropipenv install && rm -f ./Pipfile.lock

# AIPCC index for pre-built CUDA wheels (all compiled against torch 2.9.0 build 13).
ENV AIPCC_INDEX=https://packages.redhat.com/api/pypi/public-rhai/rhoai/3.3/cuda12.9-ubi9/simple/

# CUDA extensions from AIPCC — overwrites PyPI torch with the ABI-matched build.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
    --extra-index-url ${AIPCC_INDEX} \
    "torch==2.9.0" \
    "torchvision==0.24.0" \
    "torchaudio==2.9.0" \
    "triton==3.5.0" \
    "vllm==0.13.0" \
    "flash-attn==2.8.3" \
    "mamba-ssm==2.3.0" \
    "causal-conv1d==1.6.0" \
    "xformers==0.0.33.post2"

# verl: --no-deps because its numpy<2.0.0 pin conflicts with vllm's numpy>=2.
RUN pip install --no-cache-dir --no-deps verl==0.8.0

# vllm 0.13.0 + verl 0.8.0 runtime dependencies.
# Many deps (aiohttp, fastapi, pydantic, numpy, ray, transformers, etc.) are
# already installed via Pipfile.lock and are not repeated here.
RUN pip install --no-cache-dir \
    --extra-index-url ${AIPCC_INDEX} \
    anthropic==0.71.0 \
    blake3 \
    cachetools \
    cbor2 \
    cloudpickle \
    "compressed-tensors==0.13.0" \
    depyf==0.20.0 \
    diskcache==5.6.3 \
    email-validator \
    "gguf>=0.17.0" \
    ijson \
    lark==1.2.2 \
    "llguidance>=1.3.0,<1.4.0" \
    lm-format-enforcer==0.11.3 \
    mcp \
    "mistral-common>=1.8.5" \
    "model-hosting-container-standards>=0.1.9,<1.0.0" \
    msgspec \
    openai \
    "openai-harmony>=0.0.3" \
    outlines-core==0.2.11 \
    partial-json-parser \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    pybase64 \
    python-json-logger \
    python-multipart \
    setproctitle \
    tiktoken \
    watchfiles \
    xgrammar==0.1.27 \
    codetiming \
    hydra-core \
    pybind11 \
    pylatexenc \
    tensorboard \
    "tensordict!=0.9.0,<=0.10.0,>=0.8.0" \
    torchdata \
    wandb \
    "torch==2.9.0"

# pyzmq: must come from PyPI (manylinux wheel bundles libzmq.so.5).
# The AIPCC wheel expects system libzmq which doesn't exist in UBI9.
RUN pip install --no-cache-dir pyzmq

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src
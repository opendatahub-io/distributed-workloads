ARG PYTHON_VERSION=312
ARG IMAGE_TAG=9.8-1788412553

FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

ARG TARGETARCH

LABEL name="ray-ubi9-py312-cu130-th0112" \
      summary="CUDA 13.0 Python 3.12 image based on UBI9 for Ray with Training Hub 0.11.2" \
      description="CUDA 13.0 Python 3.12 image based on UBI9 for Ray with Training Hub 0.11.2" \
      io.k8s.display-name="CUDA 13.0 Python 3.12 base image for Ray with Training Hub 0.11.2" \
      io.k8s.description="CUDA 13.0 Python 3.12 image based on UBI9 for Ray with Training Hub 0.11.2" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Install CUDA 13.0
USER 0
WORKDIR /opt/app-root/bin

ENV NVIDIA_REQUIRE_CUDA="cuda>=13.0"
ENV CUDA_VERSION=13.0.3

RUN NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87 && \
    if [ "${TARGETARCH}" = "arm64" ]; then NVARCH=sbsa; else NVARCH=x86_64; fi && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

COPY cuda.repo-* ./
RUN mkdir -p /licenses
COPY NGC-DL-CONTAINER-LICENSE /licenses/

RUN if [ "${TARGETARCH}" = "arm64" ]; then \
        cp cuda.repo-arm64 /etc/yum.repos.d/cuda.repo; \
    else \
        cp cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo; \
    fi

# Install CUDA runtime and development packages (without exact patch versions)
# Only NCCL requires explicit version for compatibility
RUN yum upgrade -y && yum install -y \
    make \
    findutils \
    cuda-cudart-13-0 \
    cuda-compat-13-0 \
    cuda-libraries-13-0 \
    cuda-nvtx-13-0 \
    cuda-command-line-tools-13-0 \
    cuda-libraries-devel-13-0 \
    cuda-minimal-build-13-0 \
    cuda-cudart-devel-13-0 \
    cuda-nvml-devel-13-0 \
    libcublas-13-0 \
    libcublas-devel-13-0 \
    libnpp-13-0 \
    libnpp-devel-13-0 \
    libnccl-2.30.7-1+cuda13.3 \
    libnccl-devel-2.30.7-1+cuda13.3 \
    libcudnn9-cuda-13 \
    libcudnn9-devel-cuda-13 \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# nvidia-docker environment
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# ---------------------------------------------------------------------------
# Install Python packages
# ---------------------------------------------------------------------------

RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Pipfile.lock provides ray, training-hub, and their transitive Python deps.
# torch is NOT in the Pipfile — it comes exclusively from the AIPCC index below.
COPY Pipfile.lock ./
RUN micropipenv install && rm -f ./Pipfile.lock

# AIPCC index for pre-built CUDA wheels (all compiled against torch 2.13.0).
ENV AIPCC_INDEX=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.6/cuda13.0-ubi9/simple/

# CUDA extensions from AIPCC — overwrites PyPI torch with the ABI-matched build.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
    --extra-index-url ${AIPCC_INDEX} \
    "torch==2.13.0" \
    "torchvision==0.28.0" \
    "torchaudio==2.11.0" \
    "triton==3.7.1" \
    "vllm==0.31.0+rhaiv.2" \
    "flash-attn==2.8.3.post1" \
    "mamba-ssm==2.3.2.post1" \
    "causal-conv1d==1.7.0" \
    "xformers==0.0.35"

# verl: --no-deps because its numpy<2.0.0 pin conflicts with vllm's numpy>=2.
RUN pip install --no-cache-dir --no-deps verl==0.9.0

# unsloth + unsloth-zoo: --no-deps because their torchao>=0.13.0 and
# datasets constraints conflict with this image's torch 2.13.0 stack.
RUN pip install --no-cache-dir --no-deps \
    --extra-index-url ${AIPCC_INDEX} \
    "unsloth==2026.10.19+rhaiv.0" \
    "unsloth-zoo==2026.10.19+rhaiv.0"

# vllm 0.31.0 + verl 0.9.0 runtime dependencies.
# Many deps (aiohttp, fastapi, pydantic, numpy, ray, transformers, etc.) are
# already installed via Pipfile.lock and are not repeated here.
RUN pip install --no-cache-dir \
    --extra-index-url ${AIPCC_INDEX} \
    anthropic==1.0.0 \
    blake3 \
    cachetools \
    cbor2 \
    cloudpickle \
    compressed-tensors==0.17.1 \
    depyf==0.20.0 \
    diskcache==5.6.3 \
    email-validator \
    "gguf>=0.17.0" \
    ijson \
    lark==1.2.2 \
    "llguidance>=1.3.0" \
    lm-format-enforcer==0.11.3 \
    mcp \
    "mistral-common>=1.8.5" \
    model-hosting-container-standards==0.1.16 \
    msgspec \
    openai \
    openai-harmony==0.0.8 \
    outlines-core==0.2.14 \
    partial-json-parser \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    pybase64 \
    python-json-logger \
    python-multipart \
    setproctitle \
    tiktoken \
    watchfiles \
    xgrammar==0.2.3 \
    codetiming \
    hydra-core \
    pybind11 \
    pylatexenc \
    tensorboard \
    "tensordict!=0.9.0,<=0.10.0,>=0.8.0" \
    torchdata \
    wandb==0.28.0 \
    "torch==2.13.0"

# pyzmq: must come from PyPI (manylinux wheel bundles libzmq.so.5).
# The AIPCC wheel expects system libzmq which doesn't exist in UBI9.
RUN pip install --no-cache-dir pyzmq

# Rebuild wandb-core with CGO for FIPS. wandb is installed from wheel above;
# only the Go binary must be rebuilt (arrow-rs-wrapper stays from the wheel);
# wandb-xpu (Rust/aws-lc) is removed — not needed on CUDA images.
RUN dnf install -y golang gcc openssl-devel && \
    WANDB_VERSION=0.28.0 && \
    SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    mkdir -p /tmp/wandb-build && \
    curl -fsSL -o "/tmp/wandb-build/wandb-${WANDB_VERSION}.tar.gz" \
      "https://files.pythonhosted.org/packages/5f/a7/683bfbd6cbade3012bc90d3e9c4cfc72dd62566195bf4c30321946d64b77/wandb-${WANDB_VERSION}.tar.gz" && \
    tar -xzf "/tmp/wandb-build/wandb-${WANDB_VERSION}.tar.gz" -C /tmp/wandb-build && \
    cd "/tmp/wandb-build/wandb-${WANDB_VERSION}/core" && \
    CGO_ENABLED=1 GOEXPERIMENT=strictfipsruntime \
    go build -tags "disable_grpc_modules parquet_read_only" \
      -ldflags="-s -w -X main.commit=unknown" \
      -o "${SITE_PACKAGES}/wandb/bin/wandb-core" \
      -mod=vendor cmd/wandb-core/main.go && \
    rm -rf /tmp/wandb-build && \
    rm -f "${SITE_PACKAGES}/wandb/bin/wandb-xpu" && \
    dnf remove -y golang openssl-devel && \
    dnf clean all

# Writable cache for Hugging Face datasets/models (container runs as non-root)
ENV HF_HOME=/tmp/huggingface

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src

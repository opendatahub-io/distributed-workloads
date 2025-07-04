FROM quay.io/kryanbeane/base-ray:py311-cuda121

LABEL name="ray-ubi9-py311-cu121-app" \
    summary="Ray application image with vLLM based on CUDA 12.1 Python 3.11" \
    description="Ray application image with vLLM based on CUDA 12.1 Python 3.11" \
    io.k8s.display-name="Ray application image with vLLM" \
    io.k8s.description="Ray application image with vLLM based on CUDA 12.1 Python 3.11" \
    authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Switch to root to install packages
USER 0

RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "vllm" \
    "torch>=2.0.0" \
    "triton>=2.0.0" \
    "transformers"

RUN pip install --no-cache-dir "ray[all]==2.47.1"

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src
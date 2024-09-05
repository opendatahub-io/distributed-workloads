FROM quay.io/rhoai/ray:2.35.0-py39-rocm61

ENV PYTORCH_ROCM_ARCH=gfx942

# Install PyTorch
RUN python3 -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1

# Install Flash Attention
RUN python3 -m pip install wheel ninja

RUN export TMP_DIR=$(mktemp -d) \
    && cd $TMP_DIR \
    && git clone --depth 1 --branch v2.6.3 https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && git submodule update --init \
    && GPU_ARCHS=$PYTORCH_ROCM_ARCH MAX_JOBS="16" python3 setup.py install --verbose \
    && rm -rf $TMP_DIR

# Install DeepSpeed
# The installation fails on ROCm with the latest released version of DeepSpeed (0.15.0 as of this writing)
# This can be updated once https://github.com/microsoft/DeepSpeed/pull/6456 is released
RUN DS_ACCELERATOR=cuda DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 python3 -m pip install git+https://github.com/microsoft/DeepSpeed.git@a7ffe540fc98a945b9d3d2a9c05653e160d62dcf --global-option="build_ext" --global-option="-j8"

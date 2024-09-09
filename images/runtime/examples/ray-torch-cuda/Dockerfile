FROM quay.io/modh/ray:2.35.0-py39-cu121

# Install PyTorch
RUN python3 -m pip install torch==2.4.0

# Install Flash Attention
RUN python3 -m pip install wheel ninja
RUN python3 -m pip install flash-attn==2.6.3 --no-build-isolation

# Install DeepSpeed
RUN DS_ACCELERATOR=cuda DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 python3 -m pip install deepspeed==0.15.0 --global-option="build_ext" --global-option="-j8"

FROM rayproject/ray:2.23.0-py39-cu121

# Flash Attention 2 requires PyTorch at installation time
RUN pip install --no-cache-dir -U pip \
    torch==2.3.1

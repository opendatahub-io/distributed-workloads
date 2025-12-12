ARG PYTHON_VERSION=311
ARG IMAGE_TAG=9.7-1764607342

FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

LABEL name="ray-ubi9-py311-rocm61" \
      summary="ROCm Python 3.11 image based on UBI9 for Ray" \
      description="ROCm Python 3.11 image based on UBI9 for Ray" \
      io.k8s.display-name="ROCm Python 3.11 base image for Ray" \
      io.k8s.description="ROCm Python 3.11 image based on UBI9 for Ray" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Install ROCm
USER 0
WORKDIR /opt/app-root/bin

ARG ROCM_VERSION=6.1.2
ARG AMDGPU_VERSION=6.1.2

RUN <<EOF
cat <<EOD > /etc/yum.repos.d/rocm.repo
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/rhel/9.4/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key

[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel9/$ROCM_VERSION/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOD
EOF

# Install ROCm ML SDK only (runtime optimized)
RUN yum install -y rocm-ml-sdk \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Install Python packages

# Install micropipenv to deploy packages from Pipfile.lock
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Install Python dependencies from Pipfile.lock file
COPY Pipfile.lock ./

RUN micropipenv install && rm -f ./Pipfile.lock && \
    # Cleanup Python cache and debug symbols
    find /opt/app-root/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/app-root/lib/python3.11/site-packages -name "*.pyc" -delete 2>/dev/null || true && \
    find /opt/app-root/lib/python3.11/site-packages -name "*.pyo" -delete 2>/dev/null || true && \
    find /opt/app-root/lib/python3.11/site-packages -name "*.so" -exec strip --strip-debug {} \; 2>/dev/null || true

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src

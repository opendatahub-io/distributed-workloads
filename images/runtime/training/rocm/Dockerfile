## Global Args ######################################################
ARG IMAGE_TAG=1-77.1729776556
ARG PYTHON_VERSION=311

# use UBI9 latest
FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG} AS base

LABEL name="training:py311-rocm61-torch241" \
      summary="ROCm 6.1 Python 3.11 PyTorch 2.4.1 image based on UBI9 for Training" \
      description="ROCm 6.1 Python 3.11 PyTorch 2.4.1 image based on UBI9 for Training" \
      io.k8s.display-name="ROCm 6.1 Python 3.11 PyTorch 2.4.1 base image for Training" \
      io.k8s.description="ROCm 6.1 Python 3.11 PyTorch 2.4.1 image based on UBI9 for Training" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

# Copy license
COPY LICENSE.md /licenses/rocm-license.md

# Set the working directory in the container
USER 0
WORKDIR /app

# upgrade requests package
RUN pip install --no-cache-dir --upgrade requests==2.32.3

# Install ROCm
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

RUN dnf -y install rocm && dnf clean all && rm -rf /var/cache/dnf

# Install Python packages

# Install micropipenv to deploy packages from Pipfile.lock
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Install Python dependencies from Pipfile.lock file
COPY Pipfile.lock ./

RUN micropipenv install && rm -f ./Pipfile.lock

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src

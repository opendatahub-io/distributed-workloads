FROM quay.io/opendatahub/odh-midstream-cuda-base-13-0:268a2d4baec5ed3c3ae09a6cce325fb83622d87b

ARG SSH_PORT=2222
ARG OPENMPI_VERSION=4.1.6
ARG UCX_VERSION=1.19.1
# SHA256 checksums for supply-chain integrity verification.
ARG UCX_SHA256=0fe68f2592b7b378bdac7ccb78331e9e8c5be250f40be118edb1795e94c60209
ARG OPENMPI_SHA256=44da277b8cdc234e71c62473305a09d63f4dcca292ca40335aab7c4bf0e6a566

LABEL name="training:py312-cuda130-torch210-openmpi41" \
      summary="CUDA 13.0 Python 3.12 PyTorch 2.10.0 OpenMPI 4.1 image based on C9S for Training" \
      description="CUDA 13.0 Python 3.12 PyTorch 2.10.0 OpenMPI 4.1 image based on C9S for Training" \
      io.k8s.display-name="CUDA 13.0 Python 3.12 PyTorch 2.10.0 OpenMPI 4.1 base image for Training" \
      io.k8s.description="CUDA 13.0 Python 3.12 PyTorch 2.10.0 OpenMPI 4.1 image based on C9S for Training" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

USER 0

ENV NV_CUDA_CUDART_DEVEL_VERSION=13-0 \
    NV_CUDA_NVCC_VERSION=13-0

# libjpeg-turbo: libjpeg.so.62 required by torchvision image I/O extension.
# libpng:        libpng16.so.16 required by torchvision image I/O extension.
# libwebp:       libwebp.so.7 required by torchvision image I/O extension.
# The C9S base image compiles numpy/scipy/pyarrow/pillow against system libraries
# not available in C9S (libopenblasp.so.0, libthrift-0.15.0.so, libre2.so.9);
# those packages are reinstalled from manylinux wheels after micropipenv runs.
# numactl-libs and openblas-openmp are installed separately AFTER the OpenMPI
# build step to avoid dnf clean_requirements_on_remove sweeping them out.
RUN dnf install -y openssh-server libjpeg-turbo libpng libwebp && dnf clean all

# DeepSpeed runs `nvcc -V` at import for CUDA compatibility; the base image has runtime
# libs but not the nvcc compiler. Install the same minimal toolkit subset
# as universal CUDA 13 training images: headers + nvcc.
RUN dnf config-manager \
        --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y --setopt=install_weak_deps=False \
        cuda-cudart-devel-${NV_CUDA_CUDART_DEVEL_VERSION} \
        cuda-nvcc-${NV_CUDA_NVCC_VERSION} \
    && dnf clean all

# Install UCX 1.19.1 prebuilt RPMs (BSD-3-Clause licensed — fully redistributable).
# Built against MOFED 24.10 + CUDA 13 — aligns with the UCX version in the downstream base image.
# Runtime packages (ucx, ucx-cuda, ucx-ib, ucx-ib-mlx5, ucx-rdmacm) are kept in
# the image; ucx-devel (headers) is removed after OpenMPI is built against it.
# UCX transport plugins are dlopen'd at runtime: ucx-ib-mlx5 gracefully skips
# if MOFED is absent; ucx-cuda activates when CUDA libs are present (always true here).
RUN curl -fsSLo /tmp/ucx.tar.bz2 \
        https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}-rocky9-mofed24.10-cuda13-x86_64.tar.bz2 \
    && echo "${UCX_SHA256}  /tmp/ucx.tar.bz2" | sha256sum -c - \
    && tar -xjf /tmp/ucx.tar.bz2 -C /tmp \
    && rpm -ivh --nodeps \
        /tmp/ucx-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-cuda-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-ib-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-ib-mlx5-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-rdmacm-${UCX_VERSION}-1.el9.x86_64.rpm \
        /tmp/ucx-devel-${UCX_VERSION}-1.el9.x86_64.rpm \
    && rm -f /tmp/ucx-*.rpm /tmp/ucx.tar.bz2

# Build OpenMPI from source with CUDA support (BSD licensed — fully redistributable).
# OpenMPI install prefix: /usr/lib64/openmpi  (keeps existing symlink/PATH layout)
#
# --with-cuda:  opal_built_with_cuda_support=true — MPI calls accept GPU pointers.
# --with-ucx:   links against UCX 1.20.0 (CUDA 13 aware) installed at /usr.
#               ucx-cuda provides cuda_copy/cuda_ipc transports for GPU-Direct;
#               ucx-ib-mlx5 provides rc_mlx5/dc_mlx5 for GPU-Direct RDMA over IB
#               (activated when MOFED is present on the host at runtime).
RUN dnf install -y \
        # Runtime IB/RDMA libraries (kept after build)
        rdma-core libibverbs librdmacm libibumad libmlx5 infiniband-diags \
        # OpenMPI runtime dependencies (kept after build)
        hwloc libevent pmix \
        # Build tools (removed after build)
        gcc gcc-c++ make perl \
        # Dev headers (removed after build)
        rdma-core-devel libibverbs-devel librdmacm-devel \
        hwloc-devel libevent-devel pmix-devel zlib-devel \
    && curl -fsSLo /tmp/openmpi.tar.gz \
        https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz \
    && echo "${OPENMPI_SHA256}  /tmp/openmpi.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/openmpi.tar.gz -C /tmp \
    && cd /tmp/openmpi-${OPENMPI_VERSION} \
    && ./configure \
        --prefix=/usr/lib64/openmpi \
        --with-cuda=/usr/local/cuda \
        --with-ucx=/usr \
        --with-verbs \
        --with-hwloc \
        --with-libevent=external \
        --with-pmix=external \
        --enable-mpi-cxx \
        --enable-mpi1-compatibility \
        --disable-silent-rules \
    && make -j$(nproc) install \
    && rpm -e --nodeps ucx-devel-${UCX_VERSION}-1.el9.x86_64 \
    && dnf remove -y \
        gcc gcc-c++ make perl \
        rdma-core-devel libibverbs-devel librdmacm-devel \
        hwloc-devel libevent-devel pmix-devel zlib-devel \
    && dnf clean all \
    && rm -rf /tmp/openmpi-${OPENMPI_VERSION} /tmp/openmpi.tar.gz

# The OpenMPI step removes gcc; cuda-nvcc depends on gcc, so dnf drops the compiler RPMs.
# Reinstall headers + nvcc after the toolchain strip so DeepSpeed can run `nvcc -V`
RUN dnf install -y --setopt=install_weak_deps=False \
        cuda-cudart-devel-${NV_CUDA_CUDART_DEVEL_VERSION} \
        cuda-nvcc-${NV_CUDA_NVCC_VERSION} \
    && dnf clean all

# Re-install PyTorch runtime deps that dnf clean_requirements_on_remove sweeps
# out during the OpenMPI build step above (numactl-libs and openblas-openmp are
# transitive deps of hwloc-devel / the build toolchain and get auto-removed).
# Installing them in a fresh RUN marks them as explicit user installs.
RUN dnf install -y numactl-libs openblas-openmp && dnf clean all

# Create symlinks for OpenMPI binaries in /usr/bin so they're in default SSH PATH
RUN ln -s /usr/lib64/openmpi/bin/orted /usr/bin/orted \
    && ln -s /usr/lib64/openmpi/bin/mpiexec /usr/bin/mpiexec

# mpirun wrapper: registers the OpenShift random UID in /etc/passwd before launching mpirun.
# This is needed because the SDK overrides the container entrypoint, bypassing uid_entrypoint.sh.
# Placed in /usr/local/bin (not /usr/bin) so it takes precedence over the real mpirun in
# /usr/lib64/openmpi/bin, which the base image puts before /usr/bin in PATH.
COPY mpirun_wrapper.sh /usr/local/bin/mpirun
RUN chmod +x /usr/local/bin/mpirun

# Wrapper script so python is reachable in SSH sessions AND the virtualenv is activated.
# A symlink won't work: Python uses argv[0] to locate pyvenv.cfg, and /usr/local/bin/python
# has no venv in its parent chain. The wrapper execs /opt/app-root/bin/python, which does.
RUN printf '#!/bin/sh\nexec /opt/app-root/bin/python "$@"\n' > /usr/local/bin/python \
    && chmod +x /usr/local/bin/python

# Set PATH and LD_LIBRARY_PATH in /etc/environment for SSH sessions (loaded by PAM, not inherited from container).
# Include CUDA toolkit paths so `nvcc` is on PATH for MPI worker ranks
# OpenMPI and UCX plugin dirs need explicit LD_LIBRARY_PATH entries.
RUN echo "PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" >> /etc/environment \
    && echo "LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/usr/lib64/ucx:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64" >> /etc/environment

RUN mkdir -p /var/run/sshd

# SSH client config
RUN sed -i "s/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g" /etc/ssh/ssh_config \
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && echo "    Port ${SSH_PORT}" >> /etc/ssh/ssh_config \
    && echo "    SendEnv PATH LD_LIBRARY_PATH" >> /etc/ssh/ssh_config

# SSH server config
RUN sed -i "s/#\(StrictModes \).*/\1no/g" /etc/ssh/sshd_config \
    && sed -i "s/#\(Port \).*/\1${SSH_PORT}/g" /etc/ssh/sshd_config \
    && echo "StrictModes no" >> /etc/ssh/sshd_config \
    && echo "Port ${SSH_PORT}" >> /etc/ssh/sshd_config

# User-level sshd config for running as non-root
# OpenMPI MCA params file: read by every OpenMPI process regardless of environment variables,
# so this reliably applies to orted and worker processes launched via SSH.
RUN mkdir -p /home/mpiuser /home/mpiuser/.openmpi && \
    echo "PidFile /tmp/sshd.pid" > /home/mpiuser/.sshd_config && \
    echo "HostKey /home/mpiuser/.ssh/id_rsa" >> /home/mpiuser/.sshd_config && \
    echo "StrictModes no" >> /home/mpiuser/.sshd_config && \
    echo "Port ${SSH_PORT}" >> /home/mpiuser/.sshd_config && \
    echo "AcceptEnv PATH LD_LIBRARY_PATH" >> /home/mpiuser/.sshd_config

# Install Python dependencies from requirements.txt
WORKDIR /opt/app-root/bin
COPY requirements.txt ./

RUN UV_CONFIG_FILE=/dev/null uv pip install --no-cache \
        --python /opt/app-root/bin/python3.12 \
        --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cuda13.0-ubi9/simple/ \
        -r requirements.txt && \
    rm -f ./requirements.txt && \
    # The C9S base image ships numpy, scipy, pyarrow, and pillow compiled against
    # system libraries that are not present in the C9S repos (libopenblasp.so.0,
    # libthrift-0.15.0.so, libre2.so.9).  Reinstall from manylinux wheels, which
    # bundle all required native libraries, replacing the C9S-compiled builds.
    pip install --force-reinstall --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.17.0 \
        pyarrow==22.0.0 \
        pillow==12.1.0 && \
    chmod -R g+w /opt/app-root/lib/python3.12/site-packages

# OpenShift GID 0 pattern: give root group same permissions as owner.
# OpenShift random UIDs always have GID 0 as primary group.
RUN chgrp -R 0 /home/mpiuser && chmod -R g=u /home/mpiuser

# Allow uid_entrypoint to add random UID to /etc/passwd at runtime
RUN chmod g=u /etc/passwd

# uid_entrypoint: register the OpenShift random UID in /etc/passwd so that
# getpwuid() calls (used by Python getpass, PyTorch cache dirs, etc.) succeed.
COPY uid_entrypoint.sh /usr/local/bin/uid_entrypoint.sh
RUN chmod +x /usr/local/bin/uid_entrypoint.sh

WORKDIR /home/mpiuser
ENV HOME=/home/mpiuser
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}:${HOME}/.local/bin
ENV LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/usr/lib64/ucx:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
# Override the base image's overly restrictive NVIDIA_REQUIRE_CUDA which only lists
# specific driver minor versions (535, 550, 565, 570, 575). Any driver >= 570 supports
# CUDA 13.0; the constraint caused the nvidia-container-runtime-hook to fail on 580.x+.
ENV NVIDIA_REQUIRE_CUDA="cuda>=13.0 driver>=570"
ENTRYPOINT ["/usr/local/bin/uid_entrypoint.sh"]
USER 1001

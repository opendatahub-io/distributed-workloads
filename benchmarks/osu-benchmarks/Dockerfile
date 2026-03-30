# Stage 1: Build OSU Micro-Benchmarks
FROM quay.io/aipcc/base-images/cpu:3.4.0-ea.2-1772728237 AS builder

ARG OMPI_VERSION=4.1.6
ARG OSU_VERSION=7.3

USER 0

# The amd64 AIPCC image ships without enabled repos.
# Add CentOS Stream 9 repos (ABI-compatible with RHEL 9) for build dependencies.
RUN printf '[centos-stream-baseos]\n\
name=CentOS Stream 9 - BaseOS\n\
baseurl=https://mirror.stream.centos.org/9-stream/BaseOS/x86_64/os/\n\
gpgcheck=0\n\
enabled=1\n\
\n\
[centos-stream-appstream]\n\
name=CentOS Stream 9 - AppStream\n\
baseurl=https://mirror.stream.centos.org/9-stream/AppStream/x86_64/os/\n\
gpgcheck=0\n\
enabled=1\n' > /etc/yum.repos.d/centos-stream.repo

# Single RUN: keeps gcc available for both builds and avoids WORKDIR issues
# by starting with cd /tmp (the AIPCC image sets WORKDIR=/opt/app-root which
# interferes with compilation if we extract sources there).
RUN cd /tmp \
    && dnf install -y gcc gcc-c++ make wget perl \
    && wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.gz \
    && tar -xf openmpi-${OMPI_VERSION}.tar.gz \
    && cd /tmp/openmpi-${OMPI_VERSION} \
    && ./configure --prefix=/tmp/ompi-build \
        --disable-man-pages \
        --disable-mpi-fortran \
        --with-hwloc=internal \
        --with-libevent=internal \
        --with-pmix=internal \
    && make -j$(nproc) \
    && make install \
    && cd /tmp \
    && wget -q https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz \
    && tar -xf osu-micro-benchmarks-${OSU_VERSION}.tar.gz \
    && cd /tmp/osu-micro-benchmarks-${OSU_VERSION} \
    && LD_LIBRARY_PATH=/tmp/ompi-build/lib \
       ./configure CC=/tmp/ompi-build/bin/mpicc CXX=/tmp/ompi-build/bin/mpicxx \
        LDFLAGS="-Wl,-rpath,/usr/lib64/openmpi/lib" \
        --prefix=/usr/local/osu-benchmarks \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/ompi-build /tmp/openmpi-${OMPI_VERSION}* /tmp/osu-micro-benchmarks-${OSU_VERSION}* \
    && dnf remove -y gcc gcc-c++ make wget perl \
    && dnf clean all

# Stage 2: Runtime image
FROM quay.io/aipcc/base-images/cpu:3.4.0-ea.2-1772728237

ARG SSH_PORT=2222

USER 0

# Add CentOS Stream 9 repos and install openssh-server (not in the AIPCC base).
RUN printf '[centos-stream-baseos]\n\
name=CentOS Stream 9 - BaseOS\n\
baseurl=https://mirror.stream.centos.org/9-stream/BaseOS/x86_64/os/\n\
gpgcheck=0\n\
enabled=1\n\
\n\
[centos-stream-appstream]\n\
name=CentOS Stream 9 - AppStream\n\
baseurl=https://mirror.stream.centos.org/9-stream/AppStream/x86_64/os/\n\
gpgcheck=0\n\
enabled=1\n' > /etc/yum.repos.d/centos-stream.repo \
    && dnf install -y openssh-server \
    && dnf clean all \
    && rm -f /etc/yum.repos.d/centos-stream.repo

LABEL maintainer="srpillai" \
      summary="OSU MPI Benchmarks on OpenMPI (OpenShift-compatible)" \
      description="OSU MPI Benchmarks on OpenMPI 4.1 image based on AIPCC CPU base for MPI Benchmarking" \
      io.k8s.display-name="OSU MPI Benchmarks on OpenMPI 4.1 base image" \
      io.k8s.description="OSU MPI Benchmarks on OpenMPI 4.1 image based on AIPCC CPU base for MPI Benchmarking" \
      version="1.0"

COPY --from=builder /usr/local/osu-benchmarks /usr/local/osu-benchmarks

# Create symlinks for OpenMPI binaries in /usr/bin so they're in default SSH PATH
RUN ln -s /usr/lib64/openmpi/bin/orted /usr/bin/orted \
    && ln -s /usr/lib64/openmpi/bin/mpiexec /usr/bin/mpiexec

# Set LD_LIBRARY_PATH in /etc/environment for SSH sessions (loaded by PAM, not inherited from container)
RUN echo "LD_LIBRARY_PATH=/usr/lib64/openmpi/lib" >> /etc/environment

RUN mkdir -p /var/run/sshd

# SSH client config
RUN sed -i "s/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g" /etc/ssh/ssh_config \
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && echo "    Port ${SSH_PORT}" >> /etc/ssh/ssh_config

# Bake SSH keypair into the image at /tmp/ssh/ — the Training Operator does NOT
# auto-inject SSH keys; the launcher and workers both read from this fixed path.
RUN mkdir -p /tmp/ssh \
    && ssh-keygen -q -t rsa -N "" -f /tmp/ssh/id_rsa \
    && cp /tmp/ssh/id_rsa.pub /tmp/ssh/authorized_keys \
    && printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  Port %s\n  IdentityFile /tmp/ssh/id_rsa\n' "${SSH_PORT}" > /tmp/ssh/config \
    && chmod 644 /tmp/ssh/id_rsa /tmp/ssh/authorized_keys /tmp/ssh/config /tmp/ssh/id_rsa.pub \
    && chgrp -R 0 /tmp/ssh \
    && chmod g+rx /tmp/ssh

# sshd config written to /tmp/sshd_config — used by the worker at runtime via
# "/usr/sbin/sshd -D -f /tmp/sshd_config"; host keys are generated at pod start.
RUN printf 'Port %s\nPidFile /tmp/sshd.pid\nStrictModes no\nHostKey /tmp/hostkeys/ssh_host_rsa_key\nHostKey /tmp/hostkeys/ssh_host_ecdsa_key\nHostKey /tmp/hostkeys/ssh_host_ed25519_key\nAuthorizedKeysFile /tmp/ssh/authorized_keys\nAcceptEnv PATH LD_LIBRARY_PATH\n' "${SSH_PORT}" > /tmp/sshd_config \
    && chmod 644 /tmp/sshd_config \
    && chgrp 0 /tmp/sshd_config

# User-level sshd config (legacy path, kept for compatibility)
RUN mkdir -p /home/mpiuser && \
    echo "PidFile /tmp/sshd.pid" > /home/mpiuser/.sshd_config && \
    echo "HostKey /home/mpiuser/.ssh/id_rsa" >> /home/mpiuser/.sshd_config && \
    echo "StrictModes no" >> /home/mpiuser/.sshd_config && \
    echo "Port ${SSH_PORT}" >> /home/mpiuser/.sshd_config && \
    echo "AcceptEnv PATH LD_LIBRARY_PATH" >> /home/mpiuser/.sshd_config

# OpenShift GID 0 pattern: give root group same permissions as owner.
# OpenShift random UIDs always have GID 0 as primary group.
RUN chgrp -R 0 /home/mpiuser && chmod -R g=u /home/mpiuser

# Allow random UID to be added to /etc/passwd at runtime
RUN chmod g=u /etc/passwd

WORKDIR /usr/local/osu-benchmarks/libexec/osu-micro-benchmarks/mpi
ENV HOME=/home/mpiuser
ENV PATH="/usr/local/bin:/usr/lib64/openmpi/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib64/openmpi/lib"

USER 1001

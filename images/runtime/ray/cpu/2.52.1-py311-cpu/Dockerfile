ARG PYTHON_VERSION=311
ARG IMAGE_TAG=9.7-1764607342

FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

ARG CONSTRAINTS_FILE=requirements_compiled.txt

LABEL name="ray-ubi9-py311-cpu" \
      summary="Python 3.11 image based on UBI9 for Ray (CPU only)" \
      description="CPU-only Python 3.11 image based on UBI9 for Ray" \
      io.k8s.display-name="Python 3.11 base image for Ray (CPU)" \
      io.k8s.description="CPU-only Python 3.11 image based on UBI9 for Ray" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

USER 0
WORKDIR /opt/app-root/bin

# Update base image and install basic build tools
RUN yum upgrade -y && \
    yum install -y \
        openblas \
        openblas-devel \
        make \
        findutils \
    && yum clean all && \
    rm -rf /var/cache/yum/*

# Copy constraints file
COPY ${CONSTRAINTS_FILE} .

RUN pip install --no-cache-dir \
    -c ${CONSTRAINTS_FILE} \
    ray[default,data,train,tune,serve]==2.52.1 \
    --extra-index-url https://wheels.developerfirst.ibm.com/ppc64le/linux/+simple/

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src

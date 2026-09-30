ARG BASE_IMAGE=quay.io/rhoai/odh-workbench-jupyter-minimal-cuda-py312-rhel9@sha256:e075cd95379850122ba7efa9be1b385c680fda1751f43d45b87de657b5166e48

FROM ${BASE_IMAGE}

# ARG values declared before FROM are not available to later instructions.
# Redeclare it so the selected base image is persisted in the helper runtime.
ARG BASE_IMAGE

USER 0

# The base image already provides subscription-manager, skopeo, Python, and pip.
# Install the resolver in the helper image; credentials are supplied only when
# the helper container runs.
RUN python3 -m pip install \
      --index-url https://pypi.org/simple \
      --no-cache-dir \
      --disable-pip-version-check \
      --target /usr/local/lib/python3.9/site-packages \
      --platform manylinux2014_x86_64 \
      --python-version 3.9 \
      --implementation cp \
      --abi cp39 \
      --only-binary=:all: \
      'https://github.com/konflux-ci/rpm-lockfile-prototype/archive/refs/tags/v0.21.0.tar.gz' \
 && rm -rf /root/.cache/pip

COPY entrypoint.sh /usr/local/bin/rpm-lockfile-helper
COPY rpm-lockfile-runner.py /usr/local/bin/rpm-lockfile-runner.py
RUN chmod 0755 /usr/local/bin/rpm-lockfile-helper /usr/local/bin/rpm-lockfile-runner.py

ENV BASE_IMAGE=${BASE_IMAGE} \
    RPM_INPUT=/work/rpms.in.yaml \
    RPM_OUTPUT=/work/rpms.lock.yaml \
    RPM_ARCHES=x86_64,aarch64

ENTRYPOINT ["/usr/local/bin/rpm-lockfile-helper"]

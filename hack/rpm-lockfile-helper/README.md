# RPM lockfile helper

This helper generates an RPM lockfile using a temporary RHSM registration.
Registration, lockfile generation, and unregister/cleanup run in one
ephemeral container. The only host-side output is the mounted lockfile.

## Build

```bash
podman build \
  --build-arg BASE_IMAGE=quay.io/rhoai/odh-workbench-jupyter-minimal-cuda-py312-rhel9@sha256:e075cd95379850122ba7efa9be1b385c680fda1751f43d45b87de657b5166e48 \
  -t localhost/rpm-lockfile-helper:latest \
  hack/rpm-lockfile-helper
```

## Run

Run from the repository root. Keep the subscription values in exported
environment variables; do not put them in the Containerfile or image build
arguments.

```bash
export SUBSCRIPTION_ORG='...'
export SUBSCRIPTION_ACTIVATION_KEY='...'

# The base image is private. Log in to quay.io first, then make the
# rootless Podman auth file available to skopeo inside the helper.
export REGISTRY_AUTH_FILE_HOST="${XDG_RUNTIME_DIR}/containers/auth.json"
test -r "${REGISTRY_AUTH_FILE_HOST}"

podman run --rm \
  --env SUBSCRIPTION_ORG \
  --env SUBSCRIPTION_ACTIVATION_KEY \
  --env HOST_UID="$(id -u)" \
  --env HOST_GID="$(id -g)" \
  --env REGISTRY_AUTH_FILE=/run/containers/auth.json \
  -v "${REGISTRY_AUTH_FILE_HOST}:/run/containers/auth.json:ro,Z" \
  -v "$PWD/images/universal/training/th-torch-cuda-py312:/work:Z" \
  localhost/rpm-lockfile-helper:latest
```

If needed, authenticate on the host first with `podman login quay.io`.
The registry auth file is used only for pulling the base image; it is not
copied into the generated image or lockfile.

The helper writes:

```text
images/universal/training/th-torch-cuda-py312/rpms.lock.yaml
```

Override `BASE_IMAGE`, `RPM_ARCHES`, `RPM_INPUT`, or `RPM_OUTPUT` with
runtime environment variables when generating another lockfile.

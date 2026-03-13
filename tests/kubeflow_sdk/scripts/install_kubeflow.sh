#!/usr/bin/env bash
set -euo pipefail

mode="${1:-env}"
DEFAULT_NOTEBOOK_IMAGE_CUDA="quay.io/opendatahub/odh-training-cuda128-torch29-py312@sha256:87539ef75e399efceefc6ecc54ebdc4453f794302f417bd30372112692eee70c"
DEFAULT_NOTEBOOK_IMAGE_ROCM="quay.io/opendatahub/odh-training-rocm64-torch29-py312:odh-stable"
DEFAULT_NOTEBOOK_IMAGE_CPU="quay.io/rhoai/odh-workbench-jupyter-datascience-cpu-py312-rhel9:rhoai-3.4"

if [[ "${mode}" != "env" ]]; then
  echo "Usage: $0 env" >&2
  exit 2
fi

trim() {
  local value="${1:-}"
  # shellcheck disable=SC2001
  echo "$(echo "${value}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
}

resolve_api_server() {
  if [[ -n "${OPENSHIFT_API_URL:-}" ]]; then
    echo "${OPENSHIFT_API_URL}"
    return
  fi
  oc whoami --show-server
}

resolve_token_from_userpass() {
  local user="${1}"
  local password="${2}"
  local api_server
  api_server="$(resolve_api_server)"
  local tmp_kubeconfig
  tmp_kubeconfig="$(mktemp)"
  trap 'rm -f "${tmp_kubeconfig}"' RETURN

  oc login -u "${user}" -p "${password}" "${api_server}" --insecure-skip-tls-verify=true --kubeconfig="${tmp_kubeconfig}" >/dev/null
  oc whoami --show-token --kubeconfig="${tmp_kubeconfig}"
}

resolve_username_from_token() {
  local token="${1}"
  local api_server
  api_server="$(resolve_api_server)"
  oc whoami --token="${token}" --server="${api_server}" --insecure-skip-tls-verify=true
}

resolve_default_notebook_image() {
  local accelerator_input="${1:-CUDA}"
  local accelerator
  accelerator="$(echo "${accelerator_input}" | tr '[:lower:]' '[:upper:]')"
  case "${accelerator}" in
    ROCM)
      echo "${DEFAULT_NOTEBOOK_IMAGE_ROCM}"
      ;;
    CPU)
      echo "${DEFAULT_NOTEBOOK_IMAGE_CPU}"
      ;;
    CUDA|ALL|"")
      echo "${DEFAULT_NOTEBOOK_IMAGE_CUDA}"
      ;;
    *)
      # Unknown selector: keep behavior predictable and default to CUDA.
      echo "${DEFAULT_NOTEBOOK_IMAGE_CUDA}"
      ;;
  esac
}

token="$(trim "${NOTEBOOK_USER_TOKEN:-}")"
username="$(trim "${NOTEBOOK_USER_NAME:-}")"
password="$(trim "${NOTEBOOK_USER_PASSWORD:-}")"

if [[ -z "${token}" ]]; then
  if [[ -z "${username}" || -z "${password}" ]]; then
    echo "Missing auth input. Provide NOTEBOOK_USER_TOKEN or NOTEBOOK_USER_NAME + NOTEBOOK_USER_PASSWORD." >&2
    exit 1
  fi
  token="$(resolve_token_from_userpass "${username}" "${password}")"
fi

if [[ -z "${username}" ]]; then
  username="$(resolve_username_from_token "${token}")"
fi

if [[ -z "${username}" ]]; then
  echo "Unable to resolve NOTEBOOK_USER_NAME." >&2
  exit 1
fi

git_url="$(trim "${KUBEFLOW_GIT_URL:-}")"
version="$(trim "${KUBEFLOW_REQUIRED_VERSION:-}")"
index_url="$(trim "${KUBEFLOW_SDK_INDEX_URL:-}")"
notebook_image="$(trim "${NOTEBOOK_IMAGE:-}")"
accelerator_tests="$(trim "${ACCELERATOR_TESTS:-}")"
notebook_accelerator="$(trim "${NOTEBOOK_ACCELERATOR:-}")"
if [[ -z "${notebook_accelerator}" ]]; then
  notebook_accelerator="${accelerator_tests}"
fi

echo "export NOTEBOOK_USER_TOKEN='${token}'"
echo "export NOTEBOOK_USER_NAME='${username}'"

if [[ -n "${notebook_image}" ]]; then
  echo "export NOTEBOOK_IMAGE='${notebook_image}'"
else
  default_notebook_image="$(resolve_default_notebook_image "${notebook_accelerator}")"
  echo "export NOTEBOOK_IMAGE='${default_notebook_image}'"
fi

if [[ -n "${git_url}" ]]; then
  echo "export KUBEFLOW_INSTALL_FROM_GIT='true'"
  echo "export KUBEFLOW_GIT_URL='${git_url}'"
elif [[ -n "${version}" ]]; then
  echo "export KUBEFLOW_REQUIRED_VERSION='${version}'"
fi

if [[ -n "${index_url}" ]]; then
  echo "export KUBEFLOW_PYPI_INDEX_URL='${index_url}'"
fi

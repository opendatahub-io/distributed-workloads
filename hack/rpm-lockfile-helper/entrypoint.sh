#!/bin/bash
set -Eeuo pipefail

: "${SUBSCRIPTION_ORG:?SUBSCRIPTION_ORG is required}"
: "${SUBSCRIPTION_ACTIVATION_KEY:?SUBSCRIPTION_ACTIVATION_KEY is required}"
: "${BASE_IMAGE:?BASE_IMAGE is required}"
: "${RPM_INPUT:?RPM_INPUT is required}"
: "${RPM_OUTPUT:?RPM_OUTPUT is required}"
: "${RPM_ARCHES:?RPM_ARCHES is required}"

temp_output="/tmp/rpms.lock.yaml.$$"

cleanup() {
    local rc=$?
    rm -f "${temp_output}"
    subscription-manager unregister >/dev/null 2>&1 || true
    subscription-manager clean >/dev/null 2>&1 || true
    exit "$rc"
}
trap cleanup EXIT

subscription-manager register \
    --org "${SUBSCRIPTION_ORG}" \
    --activationkey "${SUBSCRIPTION_ACTIVATION_KEY}"

client_key="$(find /etc/pki/entitlement -maxdepth 1 -type f -name '*-key.pem' -print -quit)"
client_cert="${client_key%-key.pem}.pem"

if [[ -z "${client_key}" || ! -f "${client_cert}" ]]; then
    echo "ERROR: registration did not create an entitlement certificate/key" >&2
    exit 1
fi

# rpms.in.yaml uses $SSL_CLIENT_CERT and $SSL_CLIENT_KEY in redhat.repo.
export DNF_VAR_SSL_CLIENT_CERT="${client_cert}"
export DNF_VAR_SSL_CLIENT_KEY="${client_key}"

mapfile -t arch_args < <(
    tr ',' '\n' <<< "${RPM_ARCHES}" | awk 'NF { print "--arch"; print $0 }'
)

/usr/local/bin/rpm-lockfile-runner.py \
    --image "${BASE_IMAGE}" \
    "${arch_args[@]}" \
    --outfile "${temp_output}" \
    "${RPM_INPUT}"

mv "${temp_output}" "${RPM_OUTPUT}"

if [[ -n "${HOST_UID:-}" ]]; then
    chown "${HOST_UID}:${HOST_GID:-${HOST_UID}}" "${RPM_OUTPUT}"
fi

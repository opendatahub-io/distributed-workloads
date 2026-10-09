#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

cache_dir=$(mktemp -d)
trap 'rm -rf "$cache_dir"' EXIT
export UV_CACHE_DIR="$cache_dir"

extract_pins() {
    sed -nE 's/^([[:alnum:]_.-]+)==([^[:space:]\\;]+).*$/\1==\2/p' "$1" | sort -u
}

verify_image() {
    local image_dir=$1
    local dockerfile="$image_dir/Dockerfile"
    local pyproject="$image_dir/pyproject.toml"
    local requirements="$image_dir/requirements.txt"
    local expected_index
    local temporary_dir
    local python_version

    if [[ ! -f "$requirements" ]]; then
        echo "$requirements: missing generated file" >&2
        return 1
    fi

    if [[ ! -f "$dockerfile" ]]; then
        echo "$dockerfile: missing Dockerfile" >&2
        return 1
    fi

    expected_index=$(
        grep -m1 -o -- '--index-url=[^[:space:]\\]*' "$dockerfile" || true
    )
    if [[ -z "$expected_index" ]]; then
        echo "$dockerfile: expected --index-url" >&2
        return 1
    fi

    if [[ "$(head -n 1 "$requirements")" != "$expected_index" ]]; then
        echo "$requirements: index URL does not match $dockerfile" >&2
        return 1
    fi

    python_version=$(
        sed -nE 's/^requires-python = "==([0-9]+\.[0-9]+)\.\*"$/\1/p' "$pyproject"
    )
    if [[ -z "$python_version" ]]; then
        echo "$pyproject: expected requires-python in ==X.Y.* form" >&2
        return 1
    fi

    temporary_dir=$(mktemp -d)
    cp "$pyproject" "$requirements" "$temporary_dir/"
    cp "$temporary_dir/requirements.txt" "$temporary_dir/resolved-requirements.txt"

    # uv prefers pins from an existing output file unless --upgrade is used.
    # Seed the temporary output so new index releases do not cause upgrades,
    # while uv still rebuilds the complete graph from pyproject.toml.
    if ! (
        cd "$temporary_dir"
        uv pip compile \
            --python-platform=x86_64-manylinux_2_28 \
            "--python-version=$python_version" \
            --no-annotate \
            --no-header \
            --no-emit-index-url \
            -o resolved-requirements.txt \
            pyproject.toml \
            >uv.stdout 2>uv.stderr
    ); then
        echo "$requirements: contains pins incompatible with $pyproject" >&2
        cat "$temporary_dir/uv.stderr" >&2
        rm -rf "$temporary_dir"
        return 1
    fi

    extract_pins "$temporary_dir/requirements.txt" \
        >"$temporary_dir/locked-requirements.txt"
    extract_pins "$temporary_dir/resolved-requirements.txt" \
        >"$temporary_dir/resolved-pins.txt"

    if ! diff -u \
        "$temporary_dir/locked-requirements.txt" \
        "$temporary_dir/resolved-pins.txt"; then
        echo "$requirements: does not match the dependency graph from $pyproject" >&2
        rm -rf "$temporary_dir"
        return 1
    fi

    echo "$image_dir: requirements.txt is compatible with pyproject.toml"
    rm -rf "$temporary_dir"
}

status=0
image_count=0
for pyproject in images/universal/training/*/pyproject.toml; do
    image_dir=${pyproject%/pyproject.toml}
    image_count=$((image_count + 1))
    verify_image "$image_dir" || status=1
done

if [[ "$image_count" -eq 0 ]]; then
    echo "No universal images with pyproject.toml found" >&2
    exit 1
fi

exit "$status"

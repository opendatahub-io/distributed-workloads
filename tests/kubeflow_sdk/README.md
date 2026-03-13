# Kubeflow SDK Test Entry Point

This directory provides a top-level entry point for SDK-focused tests across components.

For now, it orchestrates existing trainer SDK test entrypoints in `tests/trainer/kubeflow_sdk_test.go` to preserve backward compatibility.

## Run commands

From repository root:

- Show all commands:
  - `make -f tests/kubeflow_sdk/Makefile help`
- Run all SDK tests (currently trainer SDK tests):
  - `make -f tests/kubeflow_sdk/Makefile sdk-test`
- Run all trainer SDK tests (all tiers):
  - `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all`
- Run trainer SDK tier shortcuts:
  - `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-cpu`
  - `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-sanity`
  - `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-tier1`
  - `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-cuda`
  - `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-rocm`
- Run trainer SDK with a custom tier:
  - `SDK_TEST_TIER=Tier1 make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-with-tier`
- Print selected trainer SDK tests:
  - `make -f tests/kubeflow_sdk/Makefile sdk-print-selected-trainer-tests`
- Print trainer tier mapping:
  - `make -f tests/kubeflow_sdk/Makefile sdk-print-trainer-tiers`
- Print tiers across all SDK components:
  - `make -f tests/kubeflow_sdk/Makefile sdk-print-tiers`

## Run Examples

### 1) Trainer sanity tier with a specific SDK git ref

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
KUBEFLOW_GIT_URL="kubeflow @ git+https://github.com/kubeflow/sdk.git@main" \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-sanity
```

What this run does:

- Executes the trainer sanity tier (`sdk-test-trainer-sanity`).
- Uses the provided token and auto-resolves `NOTEBOOK_USER_NAME` if it is not set.
- Installs Kubeflow SDK from `KUBEFLOW_GIT_URL` (takes precedence over version-based install vars).
- If `NOTEBOOK_IMAGE` is unset, the SDK test harness chooses a default image (CUDA by default unless overridden via `NOTEBOOK_ACCELERATOR`).

### 2) Tier1 trainer with a specific SDK git ref

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
KUBEFLOW_GIT_URL="kubeflow @ git+https://github.com/kubeflow/sdk.git@main" \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-tier1
```

What this run does:

- `sdk-test-trainer-tier1` delegates to the trainer harness target with `SDK_TEST_TIER=Tier1`.
- The harness evaluates `tests/kubeflow_sdk/scripts/install_kubeflow.sh env`, which exports auth/install variables for this run.
- Since `KUBEFLOW_GIT_URL` is set, SDK install is done from that git reference.
- `go test` is run for `./tests/trainer`, and tier tags skip non-`Tier1` tests.

### 3) Run all default SDK tests (Sanity + Tier1 + accelerator lane (CUDA by default))

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
make -f tests/kubeflow_sdk/Makefile sdk-test
```

What this run does by default:

- Runs `sdk-test-all`, which executes `Sanity`, then `Tier1`, then an accelerator lane.
- The accelerator lane defaults to CUDA because `ACCELERATOR_TESTS` defaults to `CUDA`.
- To change that lane, set `ACCELERATOR_TESTS=CPU`, `ACCELERATOR_TESTS=ROCM`, or `ACCELERATOR_TESTS=ALL`.

### 4) ROCm lane only (instead of default CUDA)

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
ACCELERATOR_TESTS=ROCM \
make -f tests/kubeflow_sdk/Makefile sdk-test
```

What this run does:

- Runs the default SDK flow with the ROCm accelerator lane instead of CUDA.
- Uses ROCm image defaults unless `NOTEBOOK_IMAGE` is explicitly set.

### 5) CPU-only lane (Sanity + Tier1, no CUDA/ROCm lane)

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
ACCELERATOR_TESTS=CPU \
make -f tests/kubeflow_sdk/Makefile sdk-test
```

What this run does:

- Runs `Sanity` + `Tier1` and skips CUDA/ROCm accelerator lanes.
- Useful for clusters without GPU capacity.

### 6) Use username/password instead of token

```bash
NOTEBOOK_USER_NAME="you@example.com" \
NOTEBOOK_USER_PASSWORD="***" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all
```

What this run does:

- Uses login-based auth to generate a token at runtime.
- Keeps test selection broad by running all trainer SDK tiers.

### 7) Pin SDK version from custom index

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
KUBEFLOW_REQUIRED_VERSION="0.2.1+rhai2" \
KUBEFLOW_SDK_INDEX_URL="https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4-EA2/cuda13.0-ubi9/simple/" \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all
```

What this run does:

- Installs a specific SDK version instead of using git source.
- Pulls the package from the specified index URL.

### 8) Choose notebook image (CUDA, ROCm, CPU, or custom)

You have four ways to control the notebook image used by SDK runs:

- **Default behavior (no image vars):**
  - Uses accelerator-based defaults from the harness.
  - With `sdk-test`, this follows `ACCELERATOR_TESTS` (default: `CUDA`).
- **Select a default image by accelerator only:**
  - Set `NOTEBOOK_ACCELERATOR=CUDA|ROCM|CPU`.
  - Useful when you want to pick a default image family without hardcoding a full image URL.
- **Drive lane + default image together:**
  - Set `ACCELERATOR_TESTS=CUDA|ROCM|CPU|ALL` (for `sdk-test`).
  - This changes which lanes run, and also influences default image selection when `NOTEBOOK_IMAGE` is unset.
- **Force a specific image:**
  - Set `NOTEBOOK_IMAGE=<your full image ref>`.
  - This always takes precedence over `NOTEBOOK_ACCELERATOR` and `ACCELERATOR_TESTS`.

Examples:

```bash
# Use ROCm default image family for this run
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
NOTEBOOK_ACCELERATOR=ROCM \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-tier1
```

```bash
# Run default suite with CPU lane/default image behavior
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
ACCELERATOR_TESTS=CPU \
make -f tests/kubeflow_sdk/Makefile sdk-test
```

```bash
# Use an explicit custom notebook image (highest priority)
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
NOTEBOOK_IMAGE="quay.io/your-org/your-image@sha256:<digest>" \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all
```

### 9) Exclude specific tests by regex

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
SDK_TEST_EXCLUDE_REGEX='^(TestA|TestB|TestC|TestD)$' \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all
```

What this run does:

- Runs the usual selected suite but removes tests whose names match the regex.
- Works with any trainer SDK target (`all`, tiered, or accelerator-specific).

### 10) Run one specific trainer SDK test by name

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
SDK_TRAINER_INCLUDE_REGEX='^TestTorchrunTrainingFailure$' \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all
```

What this run does:

- Narrows candidate tests to an exact test name match.
- Helpful for fast validation/reproduction while keeping harness setup/auth behavior.

### 11) Skip all S3/AWS-related trainer SDK tests

```bash
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
SKIP_S3_TESTS=true \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all
```

What this run does:

- Automatically excludes S3/AWS-related trainer tests (currently those matching `^TestRhaiS3`).
- Combines with `SDK_TEST_EXCLUDE_REGEX` if you set both.

## Writing logs with `TEST_OUTPUT_DIR`

Use `TEST_OUTPUT_DIR` when you want test artifacts saved under a stable local directory instead of ephemeral temp paths.

How it works:

- Set `TEST_OUTPUT_DIR` to a parent directory (absolute path recommended).
- The test framework creates a per-test subdirectory under that parent (it does not write all runs into one flat folder).
- During test cleanup, pod container logs and namespace event logs are written into that output directory.
- If `TEST_OUTPUT_DIR` is unset, an ephemeral temp directory is used and its path is only visible in test output logs.

Example:

```bash
mkdir -p /tmp/sdk-test-logs
NOTEBOOK_USER_TOKEN="$(oc whoami -t)" \
OPENSHIFT_API_URL="$(oc whoami --show-server)" \
TEST_OUTPUT_DIR=/tmp/sdk-test-logs \
make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all
```

## Tier-aware runs

You can run predefined trainer tier targets:

- `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-cpu`
- `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-sanity`
- `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-tier1`
- `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-cuda`
- `make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-rocm`

Trainer SDK tiers exposed by this harness:

- `Sanity`
- `Tier1`
- `CPU` (harness lane composed of `Sanity` + `Tier1`)
- `CUDA` (internally maps to trainer `KFTO-CUDA`)
- `ROCm` (internally maps to trainer `KFTO-ROCm`)

Default `sdk-test` behavior:

- runs `Sanity`
- runs `Tier1`
- runs accelerator lane from `ACCELERATOR_TESTS` (default `CUDA`; supported: `CPU`, `CUDA`, `ROCM`, `ALL`)
- if `NOTEBOOK_IMAGE` is unset, the SDK test harness picks a default image by accelerator:
  - `CUDA`/`ALL` -> `quay.io/opendatahub/odh-training-cuda128-torch29-py312@sha256:87539ef75e399efceefc6ecc54ebdc4453f794302f417bd30372112692eee70c`
  - `ROCM` -> `quay.io/opendatahub/odh-training-rocm64-torch29-py312:odh-stable`
  - `CPU` -> `quay.io/rhoai/odh-workbench-jupyter-datascience-cpu-py312-rhel9:rhoai-3.4`
- you can override image selection only (without changing test lane) via `NOTEBOOK_ACCELERATOR=CUDA|ROCM|CPU`

## Excluding tests

Use `SDK_TEST_EXCLUDE_REGEX` to exclude by test name:

- Exclude a single test:
  - `SDK_TEST_EXCLUDE_REGEX='TestTorchrunTrainingFailure' make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all`
- Exclude a group:
  - `SDK_TEST_EXCLUDE_REGEX='TestRhaiS3' make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-all`
- Exclude while using a tier:
  - `SDK_TEST_EXCLUDE_REGEX='TestRhaiS3' make -f tests/kubeflow_sdk/Makefile sdk-test-trainer-tier1`

## Environment variable reference

The SDK test harness accepts the following environment variables.

### Authentication and cluster access

- `NOTEBOOK_USER_TOKEN`
  - Preferred auth input for SDK runs.
  - If set, the harness uses it directly and derives `NOTEBOOK_USER_NAME` via `oc whoami --token=...` when needed.
- `NOTEBOOK_USER_NAME`
  - Username for login-based auth mode.
  - Required with `NOTEBOOK_USER_PASSWORD` when `NOTEBOOK_USER_TOKEN` is not provided.
- `NOTEBOOK_USER_PASSWORD`
  - Password paired with `NOTEBOOK_USER_NAME` for `oc login` token generation.
- `OPENSHIFT_API_URL`
  - OpenShift API endpoint used for `oc login` / `oc whoami --token`.
  - If unset, the harness attempts to resolve it with `oc whoami --show-server`.

Auth precedence:

1. `NOTEBOOK_USER_TOKEN`
2. `NOTEBOOK_USER_NAME` + `NOTEBOOK_USER_PASSWORD`

### SDK installation source

- `KUBEFLOW_GIT_URL`
  - Git install source, for example:
    - `kubeflow @ git+https://github.com/<org>/<repo>.git@<branch-or-tag-or-sha>`
  - If set, the harness forces git-based install (`KUBEFLOW_INSTALL_FROM_GIT=true` internally).
- `KUBEFLOW_REQUIRED_VERSION`
  - Version to install when git URL is not provided.
  - Example: `0.2.1+rhai2`
- `KUBEFLOW_SDK_INDEX_URL`
  - Optional package index URL for version-based installs.
  - Exported to installer as `KUBEFLOW_PYPI_INDEX_URL`.

Install precedence:

1. `KUBEFLOW_GIT_URL`
2. `KUBEFLOW_REQUIRED_VERSION` (+ optional `KUBEFLOW_SDK_INDEX_URL`)
3. Installer default behavior in `install_kubeflow.py`

### Notebook image and accelerator selection

- `NOTEBOOK_IMAGE`
  - Explicit notebook image override.
  - If set, this always wins over image defaults/accelerator-based selection.
- `ACCELERATOR_TESTS`
  - Controls lane selection for `sdk-test`.
  - Supported values: `CUDA` (default), `CPU`, `ROCM`, `ALL`.
  - Also used as default image selector when `NOTEBOOK_IMAGE` is unset.
- `NOTEBOOK_ACCELERATOR`
  - Optional image-only selector (`CUDA|ROCM|CPU`) when `NOTEBOOK_IMAGE` is unset.
  - Does not change tier/lane logic by itself.

Default image mapping (used only when `NOTEBOOK_IMAGE` is unset):

- `CUDA`/`ALL` -> `quay.io/opendatahub/odh-training-cuda128-torch29-py312@sha256:87539ef75e399efceefc6ecc54ebdc4453f794302f417bd30372112692eee70c`
- `ROCM` -> `quay.io/opendatahub/odh-training-rocm64-torch29-py312:odh-stable`
- `CPU` -> `quay.io/rhoai/odh-workbench-jupyter-datascience-cpu-py312-rhel9:rhoai-3.4`

### Test selection and execution controls

- `SDK_TEST_TIMEOUT`
  - `go test` timeout passed to SDK harness targets.
  - Default: `60m`.
- `SDK_TEST_EXCLUDE_REGEX`
  - Regex for excluding test names from selected SDK trainer tests.
  - Applied after include selection and tier filtering.
- `SKIP_S3_TESTS`
  - Convenience switch to exclude all S3/AWS-related trainer SDK tests.
  - Supported values: `true` or `false` (default: `false`).
  - Under the hood this appends an S3 exclusion regex to the active exclude filter.
- `SDK_TEST_TIER`
  - Tier value used by `sdk-test-trainer-with-tier`.
  - Typical values include `Sanity`, `Tier1`, `KFTO-CUDA`, `KFTO-ROCm`.
- `SDK_TRAINER_INCLUDE_REGEX` (advanced)
  - Overrides trainer SDK include pattern used to enumerate candidate tests.
  - Default pattern maps to test names in `tests/trainer/kubeflow_sdk_test.go`.
- `SDK_TRAINER_SANITY_INCLUDE_REGEX` (advanced)
  - Overrides the internal sanity include pattern used by trainer harness internals.
  - Default: `^TestKubeflowSdk`.
- `SDK_TEST_S3_EXCLUDE_REGEX` (advanced)
  - Trainer-level regex used when `SKIP_S3_TESTS=true`.
  - Default: `^TestRhaiS3`.
- `TEST_OUTPUT_DIR`
  - Optional output directory for test artifacts/logs when supported by underlying test helpers.
  - If unset, test framework may use a temporary directory.

### Storage and dataset inputs (trainer SDK tests)

These are needed for trainer SDK scenarios that read/write artifacts from object storage.

Where these are used:

- Trainer SDK test entrypoints: `tests/trainer/kubeflow_sdk_test.go`
- S3-focused trainer tests include:
  - `TestRhaiS3CheckpointingCPU`
  - `TestRhaiS3FsdpFullStateCheckpointingCPU`
  - `TestRhaiS3FsdpSharedStateCheckpointingCuda`
  - `TestRhaiS3DeepspeedStage0CheckpointingCuda`

- `AWS_DEFAULT_ENDPOINT`
  - S3-compatible endpoint URL/host used by notebook jobs.
- `AWS_ACCESS_KEY_ID`
  - Access key for object storage authentication.
- `AWS_SECRET_ACCESS_KEY`
  - Secret key for object storage authentication.
- `AWS_STORAGE_BUCKET`
  - Bucket name used for datasets/checkpoints/artifacts.
- `AWS_DEFAULT_REGION`
  - Optional region used by some provider/client paths.
- `AWS_STORAGE_BUCKET_MNIST_DIR`
  - Prefix/path for MNIST-related data.
- `AWS_STORAGE_BUCKET_OSFT_DIR`
  - Prefix/path for OSFT-related data.
- `AWS_STORAGE_BUCKET_SFT_DIR`
  - Prefix/path for SFT-related data.
- `AWS_STORAGE_BUCKET_LORA_DIR`
  - Prefix/path for LoRA-related data.
- `MODEL_S3_PREFIX`
  - Optional model prefix used by selected RHAI scenarios.
- `DATASET_S3_PREFIX`
  - Optional dataset prefix used by selected RHAI scenarios.

Other optional trainer test controls:

- `TEST_NAMESPACE_NAME`
  - Reuse an existing namespace instead of creating a generated test namespace.

### Notes

- `KUBEFLOW_INSTALL_FROM_GIT` is exported internally by the harness when `KUBEFLOW_GIT_URL` is provided; you do not need to set it for harness-based runs.
- For direct (non-harness) trainer invocations, behavior may differ because those commands bypass `tests/kubeflow_sdk/scripts/install_kubeflow.sh`.

## Scaling to future components

Future SDK components can add folders like:

- `tests/kubeflow_sdk/core`
- `tests/kubeflow_sdk/spark`

and extend the Makefile with component-specific targets while reusing shared auth/install env preparation in `tests/kubeflow_sdk/scripts/install_kubeflow.sh`.

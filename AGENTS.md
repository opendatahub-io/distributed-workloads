# distributed-workloads

E2E test suite for distributed workloads on RHOAI covering KFTO v1, Trainer v2, and KubeRay, plus training examples and runtime/test images. Built with Go, Python, Kubernetes, Ray, PyTorch.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full repository structure including test suites, images, benchmarks, and examples.

## Structure

- `tests/` - E2E test suites (Go)
- `examples/` - Training examples (Ray, KFTO, etc.)
- `images/` - Runtime and test container images

## Test Suites

- `tests/kfto/` - KFTO v1 (PyTorchJob) tests
- `tests/fms/` - fms-hf-tuning GPU fine-tuning tests
- `tests/odh/` - ODH integration tests (Ray, notebooks)
- `tests/trainer/` - Kubeflow Trainer v2 tests
- `tests/common/support/` - Shared test infrastructure (clients, helpers for Ray, PyTorchJob, Kueue, etc.)

## Key Paths

- `images/universal/training/` - Key training runtime images
- `tests/trainer/` - Main test suite location

## Running Tests

```bash
go test -run <TestName> -v -timeout 60m ./tests/<suite>/
```

Example:
```bash
go test -run TestRhaiS3FsdpSharedStateCheckpointingCuda -v -timeout 60m ./tests/trainer/
```

## Prerequisites

- Logged into OpenShift cluster with admin access
- RHOAI installed with required distributed workload components enabled
- Tests require specific env vars (assertion errors will specify missing vars with context)

See the [Common environment variables](README.md#common-environment-variables) section in `README.md` for the full env var reference.

## Lint/format & pre-commit

```bash
make golangci-lint                                # Run golangci-lint project-wide
go vet ./...                                      # Vet all Go code
make verify-imports                               # Verify import ordering
make precommit                                    # Run all pre-commit hooks
```

### Targeted lint/format

```bash
make golangci-lint LINT_PKG=./path/to/package/...    # Lint a single Go package
go vet ./path/to/package/...                         # Vet a single Go package
gofmt -w path/to/file.go                             # Format a single Go file
pre-commit run --files path/to/file.py               # Run all hooks on a single file
```

## Writing Tests

See [`.claude/skills/add-e2e-test/SKILL.md`](.claude/skills/add-e2e-test/SKILL.md) for the full guide on writing E2E tests (namespace isolation, resource naming, cleanup, tags, notebook editing, environment variables).

## Benchmarks

See [`.claude/skills/add-benchmark/SKILL.md`](.claude/skills/add-benchmark/SKILL.md) for the guide on adding new benchmarks (Dockerfile, ClusterTrainingRuntime, TrainJob, CI workflow).

## Support Library

See [`.claude/skills/update-support-lib/SKILL.md`](.claude/skills/update-support-lib/SKILL.md) for the guide on modifying the shared test support library (getters, condition checkers, client abstraction, option pattern).

## Common Workflows

The most frequent tasks in this repo, based on commit history:

- **CVE-driven Python dependency updates** -- updating a single dependency across training image variants (see CVE Fixes below)
- **Adding E2E tests** -- see [Writing Tests](#writing-tests)
- **Adding benchmarks** -- see [Benchmarks](#benchmarks)
- **Updating the support library** -- see [Support Library](#support-library)

Commit message format for JIRA-tracked work: `RHOAIENG-NNNNN: <description> in <image-variant-name>`

## CVE Fixes -- Python dependency updates

Two image families with different dependency management:

- **Runtime training images** (`images/runtime/training/`) use `Pipfile`/`Pipfile.lock` (pipenv) and pull from public PyPI. See [images/runtime/training/README.md](images/runtime/training/README.md).
- **Universal training images** (`images/universal/training/`) use `pyproject.toml`/`requirements.txt` (pip) and pull from a **private AIPCC PyPI index** -- always query the index for available versions before pinning. See [images/universal/training/README.md](images/universal/training/README.md#cve-fixes--python-dependency-updates).

Each image variant is updated independently with its own commit.

## AI Agent Skills and Rules

`ai/` is the canonical source for AI agent skills (`ai/skills/`) and rules (`ai/rules/`). Each skill or rule holds a markdown body plus a `metadata.json` with agent-specific fields. Run `make sync-agents-config` after editing any of them to regenerate `.claude/` and `.cursor/`. The sync script declares its dependencies inline (PEP 723) and runs through [uv](https://docs.astral.sh/uv/), so `uv` must be installed.

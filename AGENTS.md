# distributed-workloads

E2E test suite for distributed workloads on RHOAI covering KFTO v1, Trainer v2, and KubeRay, plus training examples and runtime/test images. Built with Go, Python, Kubernetes, Ray, PyTorch.

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

For quick feedback on specific files instead of running project-wide:

```bash
# Go
make golangci-lint LINT_PKG=./tests/common/support/...    # Lint a single Go package
go vet ./tests/common/support/...                         # Vet a single Go package
gofmt -w path/to/file.go                                  # Format a single Go file

# Python
pre-commit run --files path/to/file.py                    # Run all hooks on a single file

```

## Writing Tests

See [`.claude/skills/add-e2e-test/SKILL.md`](.claude/skills/add-e2e-test/SKILL.md) for the full guide on writing E2E tests (namespace isolation, resource naming, cleanup, tags, notebook editing, environment variables).

## Benchmarks

See [`.claude/skills/add-benchmark/SKILL.md`](.claude/skills/add-benchmark/SKILL.md) for the guide on adding new benchmarks (Dockerfile, ClusterTrainingRuntime, TrainJob, CI workflow).

## Support Library

See [`.claude/skills/update-support-lib/SKILL.md`](.claude/skills/update-support-lib/SKILL.md) for the guide on modifying the shared test support library (getters, condition checkers, client abstraction, option pattern).

## CVE Fixes — Python dependency updates

See [images/universal/training/README.md](images/universal/training/README.md#cve-fixes--python-dependency-updates) for instructions on updating Python dependencies in training images. Key point: dependencies come from a private AIPCC PyPI index, not public PyPI — always query the index for available versions before pinning.

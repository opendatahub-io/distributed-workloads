---
globs:
  - "tests/common/support/**"
---

# Shared test infrastructure conventions

## Purpose

This package (`tests/common/support/`) is the shared foundation for ALL test suites. Changes here affect every suite (kfto, trainer, fms, odh).

## Environment variables

Declare env var constants and getter functions in `environment.go`. Test files must never call `os.Getenv` directly — always use a getter from this package.

Pattern:
```go
const myEnvVar = "MY_ENV_VAR"

func GetMyValue(test Test) string {
    return GetEnvironmentVariable(test, myEnvVar)
}
```

## File organization

Place domain-specific helpers in dedicated files:
- `ray.go` / `ray_api.go` / `ray_cluster_client.go` — Ray cluster helpers
- `trainjob.go` — Trainer v2 TrainJob helpers
- `pytorchjob.go` — PyTorchJob (KFTO v1) helpers
- `kueue.go` / `kueue_operator.go` — Kueue resource management
- `rbac.go` — Roles, RoleBindings, ClusterRoles
- `batch.go` — Batch job helpers
- `core.go` — Generic resource creation/deletion, Eventually patterns

## API contract

Helpers receive the `Test` interface for access to the Kubernetes client, context, and Gomega assertions. Use `test.T().Helper()` in helper functions.

## No suite-specific logic

Code here must be generic and reusable. Suite-specific logic (e.g., kfto-specific PyTorchJob templates, trainer-specific TrainJob configurations) belongs in the suite's own `support.go` or utility files.

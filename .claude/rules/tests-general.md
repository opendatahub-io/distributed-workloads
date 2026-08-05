---
globs:
- tests/**
---

# General test conventions

## Namespace isolation

Every test must use its own namespace via `test.NewTestNamespace()`. Never use fixed namespace names — they cause interference between parallel tests.

## Resource naming

All Kubernetes resources must use `GenerateName` instead of `Name`:
```go
ObjectMeta: metav1.ObjectMeta{GenerateName: "test-trainjob-"}
```

## Cleanup

Namespace-scoped resources are cleaned up automatically. Cluster-scoped resources (`ClusterRole`, `ClusterRoleBinding`) need explicit cleanup via `t.T().Cleanup(...)` if the creating helper doesn't already register one.

## Import boundaries

Test suites (`tests/kfto/`, `tests/trainer/`, `tests/fms/`, `tests/odh/`) must not import from each other. Shared code belongs in `tests/common/support/`. This boundary is enforced by depguard in `.golangci.yml`.

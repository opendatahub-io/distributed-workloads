---
globs:
  - "tests/trainer/**"
---

# Trainer test suite conventions

## Mandatory tag declaration

Every test function must call `Tags(t, ...)` as its **first statement** — before `With(t)`. This ensures tests are skipped early when `TEST_TIER` is set.

Available tags: `Smoke`, `Tier1`–`Tier3`, `Gpu(accelerator)`, `MultiGpu(accelerator, n)`, `MultiNode(n)`, `MultiNodeGpu(n, accelerator)`, `MultiNodeMultiGpu(n, accelerator, gpus)`.

## Test structure

```go
func TestMyFeature(t *testing.T) {
    Tags(t, Tier1)
    test := With(t)
    namespace := test.NewTestNamespace().Name
    // create resources with GenerateName
    // assert with test.Eventually(...)
}
```

## Helpers

- TrainJob creation/status: `tests/common/support/trainjob.go`
- Trainer-specific utilities: `tests/trainer/utils/`
- Shared infrastructure (clients, Kueue, RBAC, etc.): `tests/common/support/`

## Import boundaries

Only import from:
- `tests/common/` and `tests/common/support/` (shared infrastructure)
- `tests/trainer/utils/` (trainer-specific utilities)

Never import from other test suites (`tests/kfto/`, `tests/fms/`, `tests/odh/`). If you need shared functionality, add it to `tests/common/support/`.

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

## Writing Tests

### Namespace isolation

Every test must operate in its own dedicated namespace. Use `test.NewTestNamespace()` — it creates a uniquely named namespace and registers automatic cleanup (log collection + deletion) via `t.Cleanup`:

```go
namespace := test.NewTestNamespace().Name
```

Never use a fixed namespace name unless driven by an env var for a specific scenario (e.g., pre-upgrade/post-upgrade tests). Shared namespaces cause interference between tests.

### Resource naming

All Kubernetes resources must use `GenerateName` instead of a fixed `Name` to avoid collisions:

```go
// Good
ObjectMeta: metav1.ObjectMeta{GenerateName: "test-trainjob-"}

// Bad
ObjectMeta: metav1.ObjectMeta{Name: "my-trainjob"}
```

### Cleanup

Namespace-scoped resources are deleted automatically when the test namespace is cleaned up. Cluster-scoped resources (e.g., `ClusterRole`, `ClusterRoleBinding`) are not namespace-bound and may need to be explicitly cleaned up if the helper creating them does not already register a cleanup hook via `t.T().Cleanup(...)`.

### Test structure

```go
func TestMyFeature(t *testing.T) {
    Tags(t, Tier1)         // 1. tag / skip checks
    test := With(t)        // 2. create test context

    namespace := test.NewTestNamespace().Name  // 3. isolated namespace

    // 4. create resources with GenerateName
    // 5. ensure cleanup of cluster-scoped resources
    // 6. assert with test.Eventually(...)
}
```

### Editing notebooks

Test notebooks (`tests/**/resources/*.ipynb`) use 1-space JSON indentation with no trailing newline. When editing notebook cells, preserve the array-of-lines source format — do not collapse source arrays into single strings:

```json
// Good — array of lines, readable in raw JSON
"source": [
 "import os\n",
 "print('hello')"
]

// Bad — single string, hard to read in raw JSON
"source": "import os\nprint('hello')"
```

If a tool (e.g. `NotebookEdit`) converts the edited cell's source to a single string, convert it back to array-of-lines before committing. You can use a Python script:

```python
import json
with open(path, encoding="utf-8") as f:
    nb = json.load(f)
for cell in nb["cells"]:
    if isinstance(cell["source"], str):
        cell["source"] = cell["source"].splitlines(True)
        # Ensure last line has no trailing newline (notebook convention)
        if cell["source"] and cell["source"][-1].endswith("\n"):
            cell["source"][-1] = cell["source"][-1][:-1]
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
```

### Tags

Tests in `tests/trainer/` **must** declare a tag — this is mandatory. Apply it as the first statement so tests are skipped early when `TEST_TIER` is set:

| Tag | When to use |
|-----|-------------|
| `Smoke` | Minimal deployment verification |
| `Tier1`–`Tier3` | Progressively deeper coverage |
| `Gpu(accelerator)` | Requires at least one GPU node |
| `MultiGpu(accelerator, n)` | Requires n GPUs per node |
| `MultiNode(n)` | Requires n worker nodes |
| `MultiNodeGpu(n, accelerator)` | Requires n nodes each with at least one GPU |
| `MultiNodeMultiGpu(n, accelerator, gpus)` | Requires n nodes each with at least gpus GPUs |

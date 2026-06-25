# Architecture

E2E test suite for distributed workloads on Red Hat OpenShift AI (RHOAI), covering Kubeflow Training Operator v1 (KFTO), Kubeflow Trainer v2, and KubeRay.

## Test suites

```
tests/
├── kfto/           KFTO v1 — PyTorchJob-based distributed training
├── trainer/        Kubeflow Trainer v2 — TrainJob / JobSet-based training
├── odh/            KubeRay — Ray cluster and RayJob-based training
├── fms/            Foundation model fine-tuning (fms-hf-tuning)
│   ├── kfto/         via KFTO PyTorchJob
│   └── trainer/      via Trainer v2 TrainJob
└── common/         Shared test infrastructure
    └── support/      Client abstractions, resource helpers, test lifecycle
```

### kfto — Kubeflow Training Operator v1

Tests PyTorchJob-based distributed training using the legacy Kubeflow Training Operator. Covers MNIST training (single/multi-node, single/multi-GPU), LLM supervised fine-tuning (SFT), Kueue integration, SDK usage, and upgrade scenarios.

### trainer — Kubeflow Trainer v2

Tests TrainJob-based distributed training using the modern Kubeflow Trainer v2. Covers PyTorch DDP (Fashion MNIST, multi-node/multi-GPU), MPI jobs (OpenMPI), Kueue integration, TrainingRuntime/ClusterTrainingRuntime, Kubeflow SDK, and upgrade scenarios. This is the primary and most actively developed test suite.

### odh — KubeRay

Tests Ray-based distributed training via RayCluster and RayJob. Covers MNIST training, Ray Tune hyperparameter optimization, and LLM fine-tuning with DeepSpeed.

### fms — Foundation model fine-tuning

Tests the fms-hf-tuning container image for LLM fine-tuning (SFT, LoRA, QLoRA) through two parallel orchestration paths: KFTO PyTorchJob (`fms/kfto/`) and Trainer v2 TrainJob (`fms/trainer/`). Both paths test the same training workload with different orchestration, validating that fms-hf-tuning works correctly under each framework. Includes S3 data staging via batch jobs.

## Suite relationships

- **kfto** is the legacy operator; **trainer** is its modern replacement. Both test PyTorch distributed training but via different CRDs (PyTorchJob vs TrainJob).
- **fms** tests the same fms-hf-tuning workload via both kfto and trainer, ensuring parity across orchestration frameworks.
- **odh** covers Ray-based parallelism, complementing the PyTorch-based kfto and trainer suites.

## Shared support library

`tests/common/support/` provides the test infrastructure used by all suites (~40 files).

### Test lifecycle

- **`test.go`** — `Test` interface: wraps `*testing.T` with gomega assertions (`Eventually`, `Expect`), context management, and namespace helpers.
- **`namespace.go`** — `NewTestNamespace()`: creates an isolated namespace per test with automatic cleanup (pod log collection, event capture, namespace deletion) via `t.Cleanup`.

### Client abstraction

- **`client.go`** — `Client` interface: lazy-initialized accessor for multiple Kubernetes API clients:
  - Core Kubernetes, Dynamic, Storage
  - Kubeflow Training Operator (`kubeflowclient`)
  - Kubeflow Trainer v2 (`trainerclient`)
  - KubeRay (`rayclient`)
  - Kueue (`kueueclient`), Kueue Operator
  - JobSet (`jobsetclient`)
  - OpenShift Machine API, Routes, ImageStreams
  - OLM (Operator Lifecycle Manager)

### Per-API resource helpers

Each distributed workload API has a dedicated helper file with getters, condition checkers, and builders:

| File | API |
|------|-----|
| `pytorchjob.go` | PyTorchJob (Running, Succeeded, Failed, Suspended) |
| `trainjob.go` | TrainJob (Complete, Failed, Suspended) |
| `ray.go` | RayJob, RayCluster (status, logs) |
| `kueue.go` | ResourceFlavor, ClusterQueue, LocalQueue, workload admission |
| `jobset.go` | JobSet resources |

### Other shared utilities

| File | Purpose |
|------|---------|
| `environment.go` | Environment variable getters (never use `os.Getenv` directly) |
| `core.go` | Pod, ConfigMap, Secret helpers |
| `rbac.go` | Role / RoleBinding creation for test isolation |
| `conditions.go` | Kubernetes condition evaluation |
| `events.go` | Event capture for debugging |
| `accelerator.go` | GPU node detection |

### Per-suite extensions

Each suite has a `support.go` that imports `tests/common/support` and adds suite-specific utilities (e.g., embedded test resource files via `//go:embed`, Prometheus queries for GPU utilization). Per-suite files extend — never wrap — the common `Test` interface.

### Common utilities outside support/

`tests/common/` (outside `support/`) provides cross-suite utilities:

| File | Purpose |
|------|---------|
| `test_tag.go` | Tag functions (`Smoke`, `Tier1`–`Tier3`, `Gpu`, `MultiNode`, etc.) and `Tags()` helper for test filtering |
| `environment.go` | Shared env var getters (test tier, notebook config, HuggingFace token) |
| `notebook.go` | Notebook creation with GPU allocation and Kueue integration |
| `template.go` | Go template parsing for dynamic Kubernetes manifests |
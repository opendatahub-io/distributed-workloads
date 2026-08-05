---
name: add-benchmark
description: Guide for adding a new benchmark to benchmarks/, covering the Dockerfile,
  ClusterTrainingRuntime, TrainJob, and CI workflow. Use when adding or modifying
  benchmarks.
---

# Add Benchmark

Guide for adding a new benchmark to `benchmarks/` in the distributed-workloads repo.

## Directory layout

Each benchmark lives in its own subdirectory under `benchmarks/`:

```
benchmarks/<benchmark-name>/
  Dockerfile              # Multi-stage build for the benchmark image
  Dockerfile.cuda         # (optional) CUDA variant
  mpi-runtime.yaml        # ClusterTrainingRuntime defining the MPI execution environment
  trainjob.yaml           # TrainJob manifest to submit the benchmark
  README.md               # Documentation (what, files, quick start, parameters, output)
  <scripts>               # (optional) Training/benchmark scripts mounted via ConfigMap
```

See `benchmarks/osu-benchmarks/` and `benchmarks/kftv2-mpi-ddp-sft/` as reference implementations.

## Dockerfile conventions

Follow the multi-stage build pattern used in `benchmarks/osu-benchmarks/Dockerfile`:

1. **Stage 1 (builder)** - compile dependencies from source (e.g., OpenMPI, benchmark binaries)
2. **Stage 2 (runtime)** - copy built artifacts, configure SSH for MPI, set up the runtime environment

Key requirements:
- Base image from `quay.io/opendatahub/` or `quay.io/modh/`
- `USER 0` only during build stages; final image must use `USER 1001`
- OpenShift GID 0 pattern: `chgrp -R 0 <dir> && chmod -R g=u <dir>`
- Allow random UID: `chmod g=u /etc/passwd`
- SSH authentication via Training Operator's `sshAuthMountPath` -- keys are auto-injected at the path specified in the ClusterTrainingRuntime, not baked into the image. Workers generate host keys at startup.
- For CUDA variants, create a separate `Dockerfile.cuda` extending the base

## ClusterTrainingRuntime

Define a `ClusterTrainingRuntime` resource with MPI configuration. Key fields:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: <runtime-name>
spec:
  mlPolicy:
    mpi:
      mpiImplementation: OpenMPI
      sshAuthMountPath: /tmp/ssh
  template:
    spec:
      replicatedJobs:
        - name: launcher
          replicas: 1
          template: ...
        - name: worker
          replicas: <N>
          template: ...
```

- Launcher: runs the benchmark command (mpirun/mpiexec)
- Workers: run sshd and wait for MPI connections
- Both need the SSH setup commands in their entrypoints

See `benchmarks/osu-benchmarks/mpi-runtime-cpu.yaml` for a complete example.

## TrainJob

Submit benchmarks using a `TrainJob` with `generateName` (not fixed `name`):

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  generateName: <benchmark-name>-
  namespace: <namespace>
spec:
  runtimeRef:
    apiGroup: trainer.kubeflow.org
    kind: ClusterTrainingRuntime
    name: <runtime-name>
  trainer:
    numNodes: 2
    resourcesPerNode:
      requests:
        nvidia.com/gpu: "2"
    env:
      - name: PARAM_NAME
        value: "value"
```

Use `trainer.env` for benchmark parameters - the controller injects them into all pod containers.

See `benchmarks/kftv2-mpi-ddp-sft/trainjob.yaml` for a complete example.

## Makefile targets

Add build/push targets to the root `Makefile` following the existing pattern:

```makefile
BENCHMARK_VERSION ?= latest

.PHONY: build-<name>-benchmark-image
build-<name>-benchmark-image:
	$(CONTAINER_ENGINE) build -t quay.io/modh/distributed-workloads-benchmark:trainer-mpi-<name>-$(BENCHMARK_VERSION) \
	  -f benchmarks/<name>/Dockerfile benchmarks/<name>/

.PHONY: push-<name>-benchmark-image
push-<name>-benchmark-image:
	$(CONTAINER_ENGINE) push quay.io/modh/distributed-workloads-benchmark:trainer-mpi-<name>-$(BENCHMARK_VERSION)
```

Registry: `quay.io/modh/distributed-workloads-benchmark`
Tag format: `trainer-mpi-<name>-<version>`

## CI workflow

Create `.github/workflows/build-and-push-<name>-benchmark.yml` matching the structure in `build-and-push-osu-benchmark.yml`:

- Trigger on push/PR when files under `benchmarks/<name>/` change
- Build on all branches, push only on `main`
- Use `docker/build-push-action` with appropriate Dockerfile path

## README

Every benchmark must include a `README.md` with these sections (see `benchmarks/kftv2-mpi-ddp-sft/README.md`):

| Section | Content |
|---------|---------|
| Title + summary | One-line description of what the benchmark measures |
| What this benchmark does | Table with algorithm, model, dataset, backend, runtime, image |
| Files | Table mapping each file to its purpose |
| Quick start | Numbered steps: deploy runtime, create namespace/ConfigMap, submit TrainJob, monitor |
| Scaling | Table showing node/GPU configurations |
| Benchmark parameters | Tables for training and infrastructure parameters with defaults and impact |
| Expected output | Example benchmark summary output |
| Known issues | Documented limitations and workarounds |
| Cleanup | Commands to remove all created resources |

## Checklist

- [ ] Dockerfile builds successfully: `make build-<name>-benchmark-image`
- [ ] ClusterTrainingRuntime applies: `oc apply -f benchmarks/<name>/mpi-runtime.yaml`
- [ ] TrainJob submits and runs: `oc create -f benchmarks/<name>/trainjob.yaml`
- [ ] README has all required sections
- [ ] Makefile targets added for build and push
- [ ] CI workflow triggers on path changes to `benchmarks/<name>/`

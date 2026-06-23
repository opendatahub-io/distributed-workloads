# Kubeflow Trainer (KFT) Tests

This directory contains test coverage for Kubeflow Trainer V2 (https://github.com/opendatahub-io/trainer).

## Prerequisites

Before running the tests, ensure the following components are deployed in your cluster.

### Step 1: Deploy JobSet CRDs (Required First!)

JobSet is a required dependency for Kubeflow Trainer v2. Install it first:

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.9.1/manifests.yaml
```

**Verify JobSet installation:**

```bash
# Check JobSet CRD is installed
kubectl get crd jobsets.jobset.x-k8s.io

# Check JobSet controller is running
kubectl get deployments -n jobset-system jobset-controller-manager
```


### Step 2: Deploy Kubeflow Trainer Operator

Now install the Trainer operator

```bash
kubectl apply --server-side -k "https://github.com/opendatahub-io/trainer.git/manifests/rhoai?ref=main"
```

**Note:** If you see errors about ClusterTrainingRuntimes on first run, simply run the command again. This is due to CRD registration timing.

**Verify Trainer installation:**

```bash
# Check Trainer controller is running
kubectl get deployments -n opendatahub kubeflow-trainer-controller-manager


#Verify all four ClusterTrainingRuntimes are created

```bash
kubectl get clustertrainingruntimes
```


## Running the Tests

### Run all Trainer tests

```bash
go test ./tests/trainer/ -v
```

### Run Specific Test

```bash
go test ./tests/trainer -run TestCustomTrainingRuntimesAvailable -v
```

## Upgrade Tests

Upgrade tests validate that Trainer v2 resources survive an RHOAI upgrade. They run in two phases controlled by `TEST_TIER`:

```bash
# Pre-upgrade: create resources and store baselines
TEST_TIER=Pre-Upgrade go test -v -timeout 10m ./tests/trainer/

# ... perform RHOAI upgrade ...

# Post-upgrade: verify resources survived and complete workloads
TEST_TIER=Post-Upgrade go test -v -timeout 10m ./tests/trainer/
```

### Test Coverage

| Test Pair | What it validates |
|-----------|-------------------|
| `TestSetupSleepTrainJob` / `TestVerifySleepTrainJob` | Running TrainJob survives upgrade with zero pod restarts |
| `TestSetupTrainingRuntime` / `TestVerifyTrainingRuntime` | Custom namespace-scoped TrainingRuntime persists, spec unchanged |
| `TestSetupCustomRuntimeUpgradeTrainJob` / `TestRunCustomRuntimeUpgradeTrainJob` | Custom ClusterTrainingRuntime + Kueue suspend/resume lifecycle |

### Spec Integrity Checks

Post-upgrade tests compare resource `metadata.generation` against pre-upgrade baselines stored in ConfigMaps. When generation changes (indicating a spec mutation), before/after specs are logged as JSON for analysis. The assertion is version-aware — an explicit allowlist in [`utils/utils_upgrade.go`](utils/utils_upgrade.go) defines upgrade paths where spec mutations are expected (e.g., API changes across minor versions). The RHOAI version is read from DSCI `status.release.version`.

### Known Limitations

- **RHOAIENG-48867**: 4 Kueue suspend/resume tests are skipped because the Trainer controller fails updating immutable JobSet `spec.replicatedJobs` when built-in ClusterTrainingRuntime specs change during upgrade. Only affects suspended jobs referencing default/versioned runtimes — running jobs and custom runtimes are not impacted.
- Tests are version-agnostic — which upgrade path is tested depends on Jenkins pipeline deployment configuration.

### Maintenance

- When Trainer API changes introduce spec mutations during upgrade, add the version pair to `specMutationExpectedPaths` in [`utils/utils_upgrade.go`](utils/utils_upgrade.go).
- When RHOAIENG-48867 is fixed upstream, remove the `t.Skip` calls in `trainer_kueue_upgrade_training_test.go` to enable the default and specific runtime Kueue tests.

## GPU Requirements

> **Note:** The TrainingHub SDK tests (`TestOsftTrainingHubMultiNodeMultiGPU`, `TestLoraTrainingHubMultiNodeMultiGPU`, `TestSftTrainingHubMultiNodeMultiGPU`) require **NVIDIA Ampere or newer GPUs** (e.g. A100, H100). The training runtime image (`odh-training-cuda128-torch29-py312-rhel9`, referenced as `DefaultTrainingHubRuntimeCUDA` in [`tests/trainer/utils/utils_runtimes.go`](utils/utils_runtimes.go)) ships with `flash_attn==2.8.3`, which requires compute capability >= 8.0. These tests will not work on pre-Ampere GPUs such as T4 or V100.


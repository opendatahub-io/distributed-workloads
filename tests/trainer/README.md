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


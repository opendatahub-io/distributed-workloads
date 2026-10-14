# MPI SDK VQE training test

`TestKubeflowSdkOpenMPICudaKueueIntegration` submits a two-rank OpenMPI TrainJob
through the Kubeflow Trainer SDK. The SDK notebook is [`mpi.ipynb`](mpi.ipynb),
and its training workload is [`mpi_vqe.py`](mpi_vqe.py).

The workload is a small variational quantum eigensolver (VQE). It optimizes one
circuit angle to lower the energy of a two-qubit H₂ Hamiltonian. The five Pauli
terms and coefficients come from the [Qiskit Algorithms H₂ VQE example](https://qiskit-community.github.io/qiskit-algorithms/tutorials/01_algorithms_introduction.html).
The circuit calculation uses PyTorch already available in the MPI runtime, so
the test needs no dataset download or additional quantum packages.

## What the two ranks do

1. The SDK submits a `CustomTrainer` job using `openmpi-cuda`, with two pods on
   different nodes and one CUDA GPU per rank. The test also requests Kueue
   admission and mounts the training script into both pods.
2. Both ranks initialize the PyTorch MPI process group and start from the same
   circuit angle. Rank 0 evaluates three Hamiltonian terms; rank 1 evaluates
   the other two.
3. Each rank computes its local energy and gradient. MPI `all_reduce` sums the
   local energies for reporting and sums the gradients before both ranks take
   the same optimizer step on their local copies of the circuit parameter.
   The gradients are summed because the Hamiltonian energy is the **sum** of
   its terms.
4. After 25 steps, each rank reports its initial and final energy and the
   optimized angle. The workload fails if either rank lacks a local gradient,
   if energy does not fall by more than 0.01, or if final energy is not below
   -1.85. It prints progress every five steps.

The [Go test](../sdk_tests/mpi_tests.go) requires a completed TrainJob, checks
the runtime image and GPU requests, verifies that the pods are on different
nodes, and checks both ranks' reported energy improvement. It also verifies
the Kueue queue and admission. Notebook and launcher logs are saved in the test
output directory before the notebook deployment is removed.

## Why this validates MPI

Each rank owns different Hamiltonian terms. Neither rank can calculate the
correct global energy or update its circuit parameter for that objective using
only its local result. Successful optimization therefore exercises cross-node MPI
communication during training, in addition to SDK submission, runtime setup,
GPU placement, and Kueue admission. The separate direct MPI tests continue to
check individual collectives such as broadcast, gather, and reduce-scatter.

This is a bounded integration workload, not a quantum simulator benchmark. It
does not test Qiskit or PennyLane, distributed state-vector simulation,
performance scaling, or the scientific accuracy of a larger chemistry model.

## Run locally against a cluster

Use a cluster with the `openmpi-cuda` ClusterTrainingRuntime, Kueue, and two
GPU nodes. From the repository root:

```bash
export NOTEBOOK_USER_NAME="$(oc whoami)"
export NOTEBOOK_USER_TOKEN="$(oc whoami -t)"
go test -run '^TestKubeflowSdkOpenMPICudaKueueIntegration$' -v -timeout 60m ./tests/trainer/
```

The test creates a temporary namespace and cleans up its cluster resources.

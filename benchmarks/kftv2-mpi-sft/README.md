# kftv2-mpi-sft — MPI SFT Training Benchmark

Multi-node MPI performance benchmark using **SFT** (Supervised Fine-Tuning) to train **Qwen 2.5 7B Instruct** on the **GSM8K** math dataset.

## Overview

| Property | Value |
|----------|-------|
| Algorithm | SFT via TRL `SFTTrainer` |
| Model | `Qwen/Qwen2.5-7B-Instruct` (7 billion parameters) |
| Dataset | `openai/gsm8k` (~7.5K grade-school math problems) |
| Distributed | PyTorch DDP with MPI backend via Kubeflow TrainJob |
| Default scale | 1 node x 8 GPUs = **8 H100 80GB GPUs** |
| Runtime image | `odh-mpi-cuda-openmpi-aipcc` (ClusterTrainingRuntime) |

### Why SFT with MPI backend?

This benchmark is specifically designed to test the performance of the **MPI implementation** itself. By explicitly configuring PyTorch Distributed Data Parallel (DDP) to use `backend="mpi"` instead of the standard `nccl`, all GPU-to-GPU gradient synchronization (e.g., `allreduce`) is forced through the MPI layer. This allows the performance team to benchmark the MPI implementation and compare it directly against the standard NCCL implementation.

## Prerequisites

- A Kubernetes cluster with GPU nodes (H100 80GB recommended)
- The `odh-mpi-cuda-openmpi-aipcc` ClusterTrainingRuntime deployed
- `kubectl` access to the cluster
- Outbound internet access from pods (to download the model and dataset from HuggingFace on first run)

## Quick Start

```bash
# 1. Create namespace
kubectl create namespace kftv2-mpi-sft

# 2. Create ConfigMap containing the training script
kubectl create configmap mpi-sft-scripts \
  --from-file=train_sft.py \
  -n kftv2-mpi-sft

# 3. Deploy the TrainJob (default: 1 node x 8 GPUs = 8 GPUs)
kubectl apply -f trainjob.yaml

# 4. Watch training logs
kubectl logs -n kftv2-mpi-sft \
  -l batch.kubernetes.io/job-name=mpi-sft-benchmark-launcher-0 \
  --tail=-1 -f
```

## Scaling

Edit `trainjob.yaml` to adjust the number of nodes and GPUs per node:

| Configuration | `numNodes` | `nvidia.com/gpu` | `memory` | `cpu` | Total GPUs |
|---------------|-----------|-----------------|---------|------|-----------|
| Default | 1 | 8 | 1200Gi | 120 | 8 |
| Inter-node | 2 | 8 | 1200Gi | 120 | 16 |
| Full scale | 5 | 8 | 1200Gi | 120 | 40 |

## Configuration

All training parameters are tuneable via environment variables passed in the TrainJob command. Edit the `export` block in `trainjob.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SFT_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `SFT_MAX_STEPS` | `50` | Number of training steps |
| `SFT_BATCH_SIZE` | `1` | Per-device batch size |
| `SFT_DATASET_SIZE` | (all ~7.5K) | Number of GSM8K examples; empty = full dataset |
| `SFT_MAX_SEQ_LENGTH` | `1024` | Maximum sequence length |
| `SFT_LEARNING_RATE` | `5e-7` | Learning rate |
| `SFT_LOG_FREQ` | `1` | Log benchmark metrics every N steps |

## Benchmark Output

The training script automatically logs per-step metrics and prints a summary at the end:

```text
[Benchmark] step=3 step_time=12.42s samples/s=3.22 gpu_mem_alloc=27.3GB gpu_mem_reserved=31.2GB
...
============================================================
BENCHMARK RESULTS
============================================================
  Model:              Qwen/Qwen2.5-7B-Instruct
  World size:         8 GPUs
  Batch size/device:  1
  Global batch size:  8
  Total steps:        50
  Total time:         842.1s (14.0 min)
  Warmup steps:       2
  Avg step time:      16.84s (post-warmup)
  Min step time:      15.21s
  Max step time:      19.47s
  Avg throughput:     2.38 samples/s
  Peak GPU memory:    34.2GB
============================================================
```

Key metrics for perf analysis:
- **Avg step time** — end-to-end per-step duration (compute + communication)
- **Throughput** — samples/sec across all GPUs
- **Peak GPU memory** — maximum allocated GPU memory per device
- **Step time variance** (min/max) — indicates communication jitter

## Cleanup

```bash
kubectl delete trainjob mpi-sft-benchmark -n kftv2-mpi-sft
kubectl delete configmap mpi-sft-scripts -n kftv2-mpi-sft
kubectl delete namespace kftv2-mpi-sft
```

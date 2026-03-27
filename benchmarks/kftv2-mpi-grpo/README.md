# kftv2-mpi-grpo — MPI GRPO Training Benchmark

Multi-node MPI performance benchmark using **GRPO** (Group Relative Policy
Optimization) to train **Qwen 2.5 7B Instruct** on the **GSM8K** math dataset
with **DeepSpeed ZeRO-3**.

## Overview

| Property | Value |
|----------|-------|
| Algorithm | GRPO via TRL `GRPOTrainer` |
| Model | `Qwen/Qwen2.5-7B-Instruct` (7 billion parameters) |
| Dataset | `openai/gsm8k` (~7.5K grade-school math problems) |
| Reward | Rule-based (answer correctness + format quality) — no reward model needed |
| Distributed | DeepSpeed ZeRO-3 + OpenMPI via Kubeflow TrainJob |
| Default scale | 5 nodes x 8 GPUs = **40 A100 80GB GPUs** |
| Runtime image | `quay.io/ksuta/odh-mpi-cuda:0.0.14` |
| Training runtime | `odh-mpi-cuda-openmpi-aipcc` (ClusterTrainingRuntime) |

### Why GRPO + GSM8K?

GRPO is a single-model RL algorithm (no separate reward/value/reference models)
that generates rich MPI communication patterns representative of real
production training:

| MPI Pattern | Source in GRPO |
|-------------|---------------|
| `allreduce` | Gradient synchronisation every training step |
| `allgather` | ZeRO-3 parameter gathering for forward/backward passes |
| `broadcast` | Distribution of generation outputs across ranks |
| `reduce_scatter` | ZeRO-3 gradient partitioning |

This covers 3 of 4 PSAP metric categories (latency, bandwidth, collectives).
Point-to-point at the API level is covered by the separate OSU micro-benchmarks.

## Prerequisites

- A Kubernetes cluster with GPU nodes (A100 80GB recommended)
- The `odh-mpi-cuda-openmpi-aipcc` ClusterTrainingRuntime deployed
- `kubectl` access to the cluster
- Outbound internet access from pods (to download the model and dataset from
  HuggingFace on first run)

## Quick Start

```bash
# 1. Create namespace
kubectl create namespace kftv2-mpi-grpo

# 2. Create ConfigMap containing the training script and DeepSpeed config
kubectl create configmap mpi-grpo-scripts \
  --from-file=train_grpo.py \
  --from-file=ds_config_zero3.json \
  -n kftv2-mpi-grpo

# 3. Deploy the TrainJob (default: 5 nodes x 8 GPUs = 40 GPUs)
kubectl apply -f trainjob.yaml

# 4. Watch training logs
kubectl logs -n kftv2-mpi-grpo \
  -l batch.kubernetes.io/job-name=mpi-grpo-benchmark-launcher-0 \
  --tail=-1 -f
```

## Scaling

Edit `trainjob.yaml` to adjust the number of nodes and GPUs per node:

| Configuration | `numNodes` | `nvidia.com/gpu` | `memory` | `cpu` | Total GPUs |
|---------------|-----------|-----------------|---------|------|-----------|
| Full benchmark | 5 | 8 | 200Gi | 32 | 40 |
| Medium | 2 | 8 | 200Gi | 32 | 16 |
| Smoke test | 2 | 1 | 40Gi | 8 | 2 |

For 13B+ models, increase `memory` to 300Gi and consider reducing
`GRPO_BATCH_SIZE` to avoid OOM.

## Configuration

All training parameters are tuneable via environment variables passed through
`mpirun -x` in the TrainJob command. Edit the `mpirun` block in
`trainjob.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPO_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `GRPO_MAX_STEPS` | `50` | Number of training steps |
| `GRPO_BATCH_SIZE` | `1` | Per-device batch size |
| `GRPO_NUM_GENERATIONS` | `4` | Completions generated per prompt |
| `GRPO_DATASET_SIZE` | (all ~7.5K) | Number of GSM8K examples; empty = full dataset |
| `GRPO_MAX_COMPLETION` | `512` | Maximum tokens per completion |
| `GRPO_MAX_PROMPT` | `256` | Maximum prompt tokens |
| `GRPO_DS_CONFIG` | `/mnt/scripts/ds_config_zero3.json` | DeepSpeed config path |
| `GRPO_LEARNING_RATE` | `5e-7` | Learning rate |
| `GRPO_LOG_FREQ` | `1` | Log benchmark metrics every N steps |

### Estimated Run Times

Based on spike validation (Qwen 2.5 7B, DeepSpeed ZeRO-3):

| Scale | Step Time (approx) | 50 Steps |
|-------|-------------------|---------|
| 2 x 1 GPU | ~41 s/step | ~35 min |
| 2 x 8 GPUs | ~15-25 s/step | ~15-20 min |
| 5 x 8 GPUs | ~10-20 s/step | ~10-15 min |

Step time varies with batch size, generation length, and inter-node network
bandwidth. Increase `GRPO_MAX_STEPS` for longer runs if needed.

## Benchmark Output

The training script automatically logs per-step metrics and prints a summary
at the end:

```
[Benchmark] step=3 step_time=38.42s samples/s=1.04 gpu_mem_alloc=27.3GB gpu_mem_reserved=31.2GB
...
============================================================
BENCHMARK RESULTS
============================================================
  Model:              Qwen/Qwen2.5-7B-Instruct
  World size:         40 GPUs
  Batch size/device:  1
  Global batch size:  40
  Num generations:    4
  DeepSpeed config:   /mnt/scripts/ds_config_zero3.json
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

## Files

| File | Description |
|------|-------------|
| `train_grpo.py` | GRPO training script with benchmark instrumentation |
| `ds_config_zero3.json` | DeepSpeed ZeRO Stage 3 configuration |
| `trainjob.yaml` | Kubeflow TrainJob manifest |
| `README.md` | This file |

## Known Issues and Workarounds

These issues were discovered during the spike validation and are already
handled in the training script and TrainJob manifest:

| Issue | Workaround |
|-------|-----------|
| DeepSpeed `nvcc` not found | `train_grpo.py` creates a stub at `/tmp/cuda_stub/bin/nvcc` |
| SSH key permissions (0644 instead of required 0600) | Launcher copies key to `/tmp/id_rsa` with `chmod 600` |
| NCCL `/dev/shm` too small (K8s default is 64MB) | `emptyDir` with `medium: Memory` mounted at `/dev/shm` (16Gi) |
| Pods landing on same node | `podAntiAffinity` on `jobset.sigs.k8s.io/jobset-name` label |

## Cleanup

```bash
kubectl delete trainjob mpi-grpo-benchmark -n kftv2-mpi-grpo
kubectl delete configmap mpi-grpo-scripts -n kftv2-mpi-grpo
kubectl delete namespace kftv2-mpi-grpo
```

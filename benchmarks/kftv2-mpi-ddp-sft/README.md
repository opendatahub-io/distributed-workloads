# MPI DDP SFT Benchmark — Qwen 2.5-1.5B-Instruct

Distributed Supervised Fine-Tuning benchmark using **PyTorch DDP** with **MPI** as the communications backend, submitted via Kubeflow Trainer v2.

## What this benchmark does

| Aspect | Details |
|--------|---------|
| Algorithm | SFT with PyTorch DistributedDataParallel (DDP) |
| Model | Qwen/Qwen2.5-1.5B-Instruct (1.5B params, float32) |
| Dataset | openai/gsm8k (~7.5 K grade-school math) |
| Communication backend | MPI |
| Gradient sync | DDP automatic allreduce via MPI |
| Runtime | `openmpi-cuda-benchmark` |
| Image | `quay.io/opendatahub/odh-training-cuda130-torch210-py312-openmpi41:odh-stable` |

### MPI communication patterns exercised

- **allreduce** — DDP gradient synchronisation every backward pass
- **broadcast** — DDP parameter broadcast during initialisation
- **barrier** — rank-0-first model download coordination

### Why float32

The MPI backend in this image only supports `float32` for CUDA collective operations. `float16` and `bfloat16` fail with `IndexError: map::at` due to missing MPI datatype mappings. See the [Known issues](#known-issues) section for details.

### Why 1.5B model

PyTorch DDP groups all gradients into a single flat buffer for the first training step. For models with more than ~2.1B `float32` parameters, the total element count exceeds MPI's `int32` count limit, causing `MPI_ERR_COUNT: invalid count argument`. The 1.5B model (~1.54B parameters) fits within this limit.

## Files

| File | Description |
|------|-------------|
| `train_sft_ddp.py` | PyTorch training script performing Supervised Fine-Tuning with DDP and MPI-based gradient synchronization |
| `trainjob.yaml` | Kubeflow Trainer v2 `TrainJob` manifest defining the distributed training workload and parameters |
| `mpi-runtime.yaml` | `ClusterTrainingRuntime` resource providing the OpenMPI + CUDA execution environment |

## Quick Start

### 1. Deploy the ClusterTrainingRuntime

```bash
oc apply -f mpi-runtime.yaml
```

### 2. Create namespace and ConfigMap

```bash
oc create namespace kftv2-mpi-ddp-sft

oc create configmap mpi-ddp-sft-scripts \
  --from-file=train_sft_ddp.py \
  -n kftv2-mpi-ddp-sft
```

### 3. Submit the TrainJob

```bash
oc create -f trainjob.yaml
```

### 4. Monitor

```bash
# Watch pod status
oc get pods -n kftv2-mpi-ddp-sft -w

# Stream launcher logs (training output)
oc logs -n kftv2-mpi-ddp-sft \
  -l batch.kubernetes.io/job-name=<job-name>-launcher-0 \
  --tail=-1 -f
```

## Scaling

Adjust `numNodes` and `resourcesPerNode` in `trainjob.yaml` to scale:

| Nodes | GPUs per node | Total GPUs | Use case |
|------:|:-------------:|-----------:|----------|
| 2 | 2 | 4 | Default (multi-node verification) |
| 2 | 4 | 8 | Higher per-node density |
| 4 | 2 | 8 | More nodes, fewer GPUs each |

## Benchmark parameters

All parameters below are configurable in `trainjob.yaml`. Update them to match your benchmarking requirements.

### Training parameters

Set via `trainer.env` in `trainjob.yaml`. These are injected into all pod containers (launcher and workers) by the Kubeflow Trainer controller.

| Variable | Default | Description | Impact of change |
|----------|---------|-------------|-----------------|
| `BENCH_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID | Larger model → more GPU memory, slower steps. Must stay under ~2.1B float32 params (MPI int32 limit) |
| `BENCH_DATASET` | `openai/gsm8k` | HuggingFace dataset ID | Different dataset changes training content but not per-step performance |
| `BENCH_DATASET_CONFIG` | `main` | Dataset configuration | Selects a subset/split of the dataset |
| `BENCH_BATCH_SIZE` | `2` | Per-device batch size | Increase → higher GPU utilisation and throughput, more memory. Decrease → less memory, lower throughput |
| `BENCH_MAX_SEQ_LENGTH` | `512` | Max token sequence length | Increase → more context per example, significantly more memory (attention is O(n²)), slower steps |
| `BENCH_MAX_STEPS` | `200` | Training steps | Controls total benchmark duration. Does not affect per-step performance |
| `BENCH_LEARNING_RATE` | `2e-5` | AdamW learning rate | Affects training quality (loss convergence), not throughput |
| `BENCH_WARMUP_STEPS` | `5` | First N steps excluded from final timing averages | Increase for more accurate steady-state metrics. These steps still execute and log |
| `BENCH_GRAD_ACCUM` | `1` | Gradient accumulation steps | Increase → simulates larger batch without extra memory, less frequent allreduce |
| `BENCH_LOG_FREQ` | `1` | Log every N steps | Increase for less verbose output |

### Infrastructure parameters

Set in `trainer` and `podTemplateOverrides` sections of `trainjob.yaml`.

| Parameter | Default | Description | Impact of change |
|-----------|---------|-------------|-----------------|
| `numNodes` | `2` | Number of nodes (each runs training + sshd) | More nodes → more GPUs, tests multi-node network scaling |
| `nvidia.com/gpu` | `"2"` | GPUs per node (also sets MPI processes per node) | More GPUs per node → higher per-node density, more intra-node communication |
| `memory` | `40Gi` | Memory per node | Increase for larger models or batch sizes |
| `cpu` | `"8"` | CPU cores per node | Increase if data loading becomes a bottleneck |
| `dshm sizeLimit` | `16Gi` | Shared memory (`/dev/shm`) for inter-GPU communication | Increase if you see `/dev/shm` out-of-memory errors |

## Expected output

The benchmark prints per-step metrics and a final summary:

```
============================================================
BENCHMARK RESULTS (DDP + MPI)
============================================================
  Backend:            MPI (via DDP)
  Model:              Qwen/Qwen2.5-1.5B-Instruct
  Dtype:              float32
  World size:         4 GPUs
  Batch size/GPU:     2
  Global batch:       8
  Total steps:        200
  Avg step time:      X.XXs (post-warmup)
  Avg throughput:     X.XX samples/s
  Peak GPU memory:    X.XGB
============================================================
```

## Known issues

### MPI backend dtype limitation on CUDA

**Symptom:** `dist.all_reduce` on CUDA tensors fails with `IndexError: map::at` for `bfloat16` / `float16`.

**Root cause:** OpenMPI's MPI datatype table does not include mappings for half-precision formats.

**Workaround:** Load the model with `dtype=torch.float32`.

### MPI int32 count limit (models > ~2B parameters)

**Symptom:** `MPI_ERR_COUNT: invalid count argument` during DDP gradient allreduce on the first training step.

**Root cause:** PyTorch DDP groups all gradients into a single flat buffer. For models with more than ~2.1 billion `float32` parameters, the element count exceeds MPI's `int32` count limit. This is why the benchmark uses the 1.5B model.

## Cleanup

```bash
oc delete trainjobs -n kftv2-mpi-ddp-sft --all
oc delete configmap mpi-ddp-sft-scripts -n kftv2-mpi-ddp-sft
oc delete namespace kftv2-mpi-ddp-sft
```

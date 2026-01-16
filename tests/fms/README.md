# FMS (Foundation Model Suite) Tests

End-to-end tests for fine-tuning Large Language Models (LLMs) using the [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) library on Kubernetes/OpenShift.

## Directory Structure

```
tests/fms/
├── kfto/                           # Training Operator v1 (PyTorchJob) tests
│   ├── kfto_kueue_sft_test.go      # Single-GPU SFT tests
│   └── kfto_kueue_sft_GPU_test.go  # Multi-GPU SFT tests
├── trainer/                        # Trainer Operator v2 (TrainJob) tests
│   ├── sft_trainjob_test.go        # Single-GPU SFT tests
│   └── sft_trainjob_gpu_test.go    # Multi-GPU SFT tests
├── resources/                      # Training configuration files
│   ├── config.json                 # Full fine-tuning config
│   ├── config_lora.json            # LoRA config
│   ├── config_qlora.json           # QLoRA config
│   └── config_*.json               # Model-specific configs
├── environment.go                  # Environment variables
├── support.go                      # Shared utilities (S3, file reading)
└── README.md                       # This file
```

## Test Categories

### Training Operator v1 (KFTO) Tests

Located in `kfto/`, these tests use **PyTorchJob** custom resources:

| Test | Description | GPUs |
|------|-------------|------|
| `TestPytorchjobWithSFTtrainerFinetuning` | Full SFT fine-tuning | 1 |
| `TestPytorchjobWithSFTtrainerLoRa` | LoRA fine-tuning | 1 |
| `TestPytorchjobWithSFTtrainerQLoRa` | QLoRA fine-tuning | 1 |
| `TestPytorchjobUsingKueueQuota` | Kueue quota management | 1 |
| `TestMultiGpu*` (GPU tests) | Multi-GPU training | 2-8 |

### Trainer Operator v2 Tests

Located in `trainer/`, these tests use **TrainJob** custom resources:

| Test | Description | GPUs |
|------|-------------|------|
| `TestTrainJobWithSFTtrainerFinetuning` | Full SFT fine-tuning | 1 |
| `TestTrainJobWithSFTtrainerLoRa` | LoRA fine-tuning | 1 |
| `TestTrainJobWithSFTtrainerQLoRa` | QLoRA fine-tuning | 1 |
| `TestMultiGpuTrainJob*` (GPU tests) | Multi-GPU training | 2-8 |

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `FMS_HF_TUNING_IMAGE` | FMS HF Tuning container image | `quay.io/modh/fms-hf-tuning:v3.1.1-rc3` |

### Optional (S3 Integration)

| Variable | Description |
|----------|-------------|
| `AWS_STORAGE_BUCKET_DOWNLOAD` | S3 bucket for downloading base models |
| `AWS_STORAGE_BUCKET_UPLOAD` | S3 bucket for uploading trained models |
| `AWS_STORAGE_BUCKET_DOWNLOAD_MODEL_PATH` | Path within download bucket |
| `AWS_STORAGE_BUCKET_UPLOAD_MODEL_PATH` | Path within upload bucket |
| `AWS_DEFAULT_ENDPOINT` | S3 endpoint URL |
| `AWS_ACCESS_KEY_ID` | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key |

### Optional (GPTQ Models)

| Variable | Description |
|----------|-------------|
| `GPTQ_MODEL_PVC_NAME` | PVC containing pre-downloaded GPTQ models |
| `MINIO_CLI_IMAGE` | Minio CLI image for S3 operations |

## Running Tests

### Prerequisites

1. Access to a Kubernetes/OpenShift cluster with GPU nodes
2. Training Operator v1 (KFTO) or Trainer Operator v2 installed
3. Kueue installed (for quota tests)

### Run Single-GPU Tests (Training Operator v1)

```bash
export FMS_HF_TUNING_IMAGE=quay.io/modh/fms-hf-tuning:v3.1.1-rc3

# Run all lightweight tests
go test -v -timeout 30m ./tests/fms/kfto/... -run "TestPytorchjobWithSFTtrainerFinetuning"

# Run LoRA test
go test -v -timeout 30m ./tests/fms/kfto/... -run "TestPytorchjobWithSFTtrainerLoRa"

# Run QLoRA test
go test -v -timeout 30m ./tests/fms/kfto/... -run "TestPytorchjobWithSFTtrainerQLoRa"
```

### Run Single-GPU Tests (Trainer Operator v2)

```bash
export FMS_HF_TUNING_IMAGE=quay.io/modh/fms-hf-tuning:v3.1.1-rc3

# Run all lightweight tests
go test -v -timeout 30m ./tests/fms/trainer/... -run "TestTrainJobWithSFTtrainerFinetuning"

# Run LoRA test
go test -v -timeout 30m ./tests/fms/trainer/... -run "TestTrainJobWithSFTtrainerLoRa"
```

### Run Multi-GPU Tests

```bash
export FMS_HF_TUNING_IMAGE=quay.io/modh/fms-hf-tuning:v3.1.1-rc3

# Run specific multi-GPU test (Training Operator v1)
go test -v -timeout 60m ./tests/fms/kfto/... -run "TestMultiGpuPytorchjobMetaLlama318b"

# Run specific multi-GPU test (Trainer Operator v2)
go test -v -timeout 60m ./tests/fms/trainer/... -run "TestMultiGpuTrainJobMetaLlama318b"
```

### Run with S3 Integration

```bash
export FMS_HF_TUNING_IMAGE=quay.io/modh/fms-hf-tuning:v3.1.1-rc3
export AWS_DEFAULT_ENDPOINT=https://s3.example.com
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_STORAGE_BUCKET_DOWNLOAD=models-bucket
export AWS_STORAGE_BUCKET_DOWNLOAD_MODEL_PATH=granite-3b-code-base-2k
export AWS_STORAGE_BUCKET_UPLOAD=output-bucket
export AWS_STORAGE_BUCKET_UPLOAD_MODEL_PATH=fine-tuned-models

go test -v -timeout 60m ./tests/fms/kfto/... -run "TestPytorchjobWithSFTtrainerFinetuning"
```

## Training Configuration

### Default Model

By default, tests fine-tune `ibm-granite/granite-3b-code-base-2k` from HuggingFace on the Alpaca dataset.

### Configuration Files

| Config | Description | Technique |
|--------|-------------|-----------|
| `config.json` | Full fine-tuning | Standard SFT |
| `config_lora.json` | LoRA adapters | Parameter-efficient |
| `config_qlora.json` | Quantized LoRA | 4-bit quantization |
| `config_*_gptq.json` | GPTQ models | Pre-quantized models |
| `config_*_lora.json` | Model-specific LoRA | Various models |

### Key Configuration Parameters

```json
{
    "model_name_or_path": "<MODEL_PATH_PLACEHOLDER>",
    "training_data_path": "/tmp/dataset/alpaca_data.json",
    "output_dir": "/mnt/output/model",
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine"
}
```

## Architecture

### Training Operator v1 (KFTO)

```
┌────────────────────────────────────────────────────────────┐
│                     PyTorchJob                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Master Pod                                         │   │
│  │  ┌─────────────────┐  ┌────────────────────────┐    │   │
│  │  │ Init: copy-data │  │ Container: pytorch     │    │   │
│  │  │ (Alpaca dataset)│  │ (fms-hf-tuning image)  │    │   │
│  │  └─────────────────┘  └────────────────────────┘    │   │
│  │                                                     │   │
│  │  Volumes: config, tmp, base-model, output           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  Kueue: ResourceFlavor → ClusterQueue → LocalQueue         │
└────────────────────────────────────────────────────────────┘
```

### Trainer Operator v2

```
┌────────────────────────────────────────────────────────────┐
│                    TrainingRuntime                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Template: JobSet with trainer ReplicatedJob        │   │
│  │  - Init container: copy-dataset                     │   │
│  │  - Main container: node (fms-hf-tuning)             │   │
│  │  - Volumes: tmp-volume                              │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│                       TrainJob                             │
│  - RuntimeRef: TrainingRuntime                             │
│  - Trainer: image, env, resources                          │
│  - PodTemplateOverrides: volumes, volume mounts            │
└────────────────────────────────────────────────────────────┘
```

## Supported Models

### Multi-GPU Tests

| Model | GPUs | Config |
|-------|------|--------|
| Meta Llama 3.1 8B | 2 | `config_meta_llama3_1_8b.json` |
| Meta Llama 3.1 70B LoRA | 4 | `config_meta_llama3_1_70b_lora.json` |
| Meta Llama 3.1 405B GPTQ | 8 | `config_meta_llama3_1_405b_gptq.json` |
| Granite 8B Code Instruct GPTQ | 2 | `config_granite_8b_code_instruct_gptq.json` |
| Granite 20B Code Instruct | 4 | `config_granite_20b_code_instruct.json` |
| Granite 34B Code Base GPTQ | 2 | `config_granite_34b_code_base_gptq.json` |
| Mistral 7B v0.3 | 2 | `config_mistral_7b_v03.json` |
| Mixtral 8x7B v0.1 | 8 | `config_mixtral_8x7b_v01.json` |
| Merlinite 7B | 2 | `config_merlinite_7b.json` |

## Troubleshooting

### Common Issues

1. **Missing FMS_HF_TUNING_IMAGE**
   ```
   Expected environment variable FMS_HF_TUNING_IMAGE not found
   ```
   Solution: Export the environment variable before running tests.

2. **Image pull timeout**
   The FMS image is ~5GB. First runs may take time to pull.

3. **OOM errors**
   Ensure sufficient GPU memory for the model being tested.

4. **Kueue workload not admitted**
   Check ClusterQueue has sufficient resources and LocalQueue is configured correctly.

### Debugging

```bash
# Check training pod logs
oc logs <pod-name> -n <namespace>

# Check TrainJob/PyTorchJob status
oc get trainjob -n <namespace> -o yaml
oc get pytorchjob -n <namespace> -o yaml

# Check Kueue workloads
oc get workloads -n <namespace>
```

## Contributing

1. Add new model configs to `resources/`
2. Add test functions following existing patterns
3. Update this README with new test documentation


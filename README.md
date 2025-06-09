# Distributed Workloads

## Examples

* Fine-Tune LLMs with Ray and DeepSpeed on OpenShift AI
* Fine-Tune Stable Diffusion with DreamBooth and Ray Train
* Hyperparameters Optimization with Ray Tune on OpenShift AI

## Integration Tests

### Prerequisites

* Admin access to an OpenShift cluster ([CRC](https://developers.redhat.com/products/openshift-local/overview) is fine)

* Installed OpenDataHub or RHOAI, enabled all Distributed Workload components

* Installed Go 1.21

### Common environment variables

* `TEST_OUTPUT_DIR` - Output directory for test logs
* `TEST_TIMEOUT_SHORT` - Timeout duration for short tasks
* `TEST_TIMEOUT_MEDIUM` - Timeout duration for medium tasks
* `TEST_TIMEOUT_LONG` - Timeout duration for long tasks
* `TEST_RAY_IMAGE` (Optional) - Ray image used for raycluster configuration
* `MINIO_CLI_IMAGE` (Optional) - Minio CLI image used for uploading/downloading data from/into s3 bucket
* `TEST_TIER` (Optional) - Specifies test tier to run, skipping tests which don't belong to specified test tier. Supported test tiers: Smoke, Sanity, Tier1, Tier2, Tier3, Pre-Upgrade and Post-Upgrade. Test tier can also be provided using test parameter `testTier`.

    NOTE: `quay.io/modh/ray:2.35.0-py311-cu121` is the default image used for creating a RayCluster resource. If you have your own custom ray image which suits your purposes, specify it in `TEST_RAY_IMAGE` environment variable.

### Environment variables for fms-hf-tuning test suite

* `FMS_HF_TUNING_IMAGE` - Image tag used in PyTorchJob CR for model training

### Environment variables for fms-hf-tuning GPU test suite

* `TEST_NAMESPACE_NAME` (Optional) - Existing namespace where will the Training operator GPU tests be executed
* `HF_TOKEN` - HuggingFace token used to pull models which has limited access
* `GPTQ_MODEL_PVC_NAME` - Name of PersistenceVolumeClaim containing downloaded GPTQ models

To upload trained model into S3 compatible storage, use the environment variables mentioned below :
* `AWS_DEFAULT_ENDPOINT` - Storage bucket endpoint to upload trained dataset to, if set then test will upload model into s3 bucket
* `AWS_ACCESS_KEY_ID` - Storage bucket access key
* `AWS_SECRET_ACCESS_KEY` - Storage bucket secret key
* `AWS_STORAGE_BUCKET` - Storage bucket name
* `AWS_STORAGE_BUCKET_MODEL_PATH` (Optional) - Path in the storage bucket where trained model will be stored to

### Environment variables for ODH integration test suite

* `ODH_NAMESPACE` - Namespace where ODH components are installed to
* `NOTEBOOK_USER_NAME` - Username of user used for running Workbench
* `NOTEBOOK_USER_PASSWORD` - Password of user used for running Workbench
* `NOTEBOOK_USER_TOKEN` - Login token of user used for running Workbench
* `NOTEBOOK_IMAGE` - Image used for running Workbench

To download MNIST training script datasets from S3 compatible storage, use the environment variables mentioned below : 
* `AWS_DEFAULT_ENDPOINT` - Storage bucket endpoint from which to download MNIST datasets
* `AWS_ACCESS_KEY_ID` - Storage bucket access key
* `AWS_SECRET_ACCESS_KEY` - Storage bucket secret key
* `AWS_STORAGE_BUCKET` - Storage bucket name
* `AWS_STORAGE_BUCKET_MNIST_DIR` - Storage bucket directory from which to download MNIST datasets.

### Running Tests

Execute tests like standard Go unit tests.

```bash
go test -timeout 60m ./tests/kfto/
```

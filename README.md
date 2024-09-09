# Distributed Workloads

## Examples

* Fine-Tune LLMs with Ray and DeepSpeed on OpenShift AI

## Integration Tests

### Prerequisites

* Admin access to an OpenShift cluster ([CRC](https://developers.redhat.com/products/openshift-local/overview) is fine)

* Installed OpenDataHub or RHOAI, enabled all Distributed Workload components

* Installed Go 1.21

### Common environment variables

* `CODEFLARE_TEST_OUTPUT_DIR` - Output directory for test logs
* `CODEFLARE_TEST_TIMEOUT_SHORT` - Timeout duration for short tasks
* `CODEFLARE_TEST_TIMEOUT_MEDIUM` - Timeout duration for medium tasks
* `CODEFLARE_TEST_TIMEOUT_LONG` - Timeout duration for long tasks
* `CODEFLARE_TEST_RAY_IMAGE` (Optional) - Ray image used for raycluster configuration 

    NOTE: `quay.io/modh/ray:2.35.0-py39-cu121` is the default image used for creating a RayCluster resource. If you have your own custom ray image which suits your purposes, specify it in `CODEFLARE_TEST_RAY_IMAGE` environment variable.

### Environment variables for Training operator test suite

* `FMS_HF_TUNING_IMAGE` - Image tag used in PyTorchJob CR for model training

### Environment variables for ODH integration test suite

* `ODH_NAMESPACE` - Namespace where ODH components are installed to
* `NOTEBOOK_USER_NAME` - Username of user used for running Workbench
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

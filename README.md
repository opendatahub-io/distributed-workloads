# Distributed Workloads

## Examples

* Fine-Tune Llama 2 Models with Ray and DeepSpeed on OpenShift AI

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

### Environment variables for Training operator test suite

* `FMS_HF_TUNING_IMAGE` - Image tag used in PyTorchJob CR for model training

### Environment variables for ODH integration test suite

* `ODH_NAMESPACE` - Namespace where ODH components are installed to
* `NOTEBOOK_USER_NAME` - Username of user used for running Workbench
* `NOTEBOOK_USER_TOKEN` - Login token of user used for running Workbench
* `NOTEBOOK_IMAGE` - Image used for running Workbench
* `MNIST_DATASET_URL` - External source from which to download MNIST datasets (Example : http://yann.lecun.com/exdb/mnist)

To download MNIST training script datasets from S3 compatible storage, use the environment variables mentioned below : 
* `AWS_DEFAULT_ENDPOINT` - Storage bucket endpoint from which to download MNIST datasets
* `AWS_ACCESS_KEY_ID` - Storage bucket access key
* `AWS_SECRET_ACCESS_KEY` - Storage bucket secret key
* `AWS_STORAGE_BUCKET` - Storage bucket name
* `AWS_STORAGE_BUCKET_MNIST_DIR` - Storage bucket directory from which to download MNIST datasets.

Note : Either use 

### Running Tests

Execute tests like standard Go unit tests.

```bash
go test -timeout 60m ./tests/kfto/
```

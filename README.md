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
* `NOTEBOOK_IMAGE_STREAM_NAME` - Name of the image stream used for running Workbench
* `NOTEBOOK_USER_NAME` - Username of user used for running Workbench
* `NOTEBOOK_USER_TOKEN` - Login token of user used for running Workbench

### Running Tests

Execute tests like standard Go unit tests.

```bash
go test -timeout 60m ./tests/kfto/
```

# InstructLab Standalone Tool Integration Tests

### Prerequisites

* Admin access to an OpenShift cluster

* Installed OpenDataHub or RHOAI, enabled all Distributed Workload and Serving components

* OpenShift contains StorageClass with ReadWriteMany access mode

* OpenAI compliant Judge model deployed and served on an endpoint reachable from OpenShift

* Installed Go 1.21

### Sample Judge Model Deployment

* The sample manifest for deploying judge-model can be found here - `tests/ilab/resources/judge_model_deployment.yaml`

## Required environment variables

### Environment variables to download SDG and upload trained model

* `AWS_DEFAULT_ENDPOINT` - Storage bucket default endpoint
* `AWS_ACCESS_KEY_ID` - Storage bucket access key
* `AWS_SECRET_ACCESS_KEY` - Storage bucket secret key
* `AWS_STORAGE_BUCKET` - Storage bucket name
* `SDG_OBJECT_STORE_DATA_KEY` - Path in the storage bucket where SDG bundle is located

### Environment variables for connection to Judge model

* `JUDGE_ENDPOINT` - Endpoint where the Judge model is deployed to (it should end with `/v1`)
* `JUDGE_NAME` - Name of the Judge model
* `JUDGE_API_KEY` - API key needed to access the Judge model
* `JUDGE_CA_CERT_FROM_OPENSHIFT` (Optional) - If Judge model is deployed in the same OpenShift instance and the OpenShift certificate is insecure then set this env variable to `true`. It will indicate to the test to set OpenShift CA certificate as trusted certificate.

### Misc environment variables

* `TEST_NAMESPACE` (Optional) - Specify test namespace which should be used to run the tests
* `TEST_ILAB_STORAGE_CLASS_NAME` (Optional) - Specify name of StorageClass which supports ReadWriteMany access mode. If not specified then test assumes StorageClass `nfs-csi` to exist.
* `RHELAI_WORKBENCH_IMAGE` (Optional) - Specify Workbench image to be used to run Standalone tool. If not specified then test uses Workbench image `quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.11-20241004-609ffb8`.
Provided image should contain `click==8.1.7` and `kubernetes==26.1.0` packages.

## Running Tests

Execute tests like standard Go unit tests.

```bash
go test -run TestInstructlabTrainingOnRhoai -v -timeout 180m ./tests/ilab/
```

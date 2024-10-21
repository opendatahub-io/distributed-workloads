# Standalone Tool Documentation

## Overview

The `standalone.py` script simulates the [InstructLab](https://instructlab.ai/) workflow within a [Kubernetes](https://kubernetes.io/) environment,
replicating the functionality of a [KubeFlow Pipeline](https://github.com/kubeflow/pipelines). This allows for distributed training and evaluation
of models without relying on centralized orchestration tools like KubeFlow.

The `standalone.py` tool provides support for fetching generated SDG (Synthetic Data Generation) data from an AWS S3 compatible object store.
While AWS S3 is supported, alternative object storage solutions such as Ceph, Nooba, and MinIO are also compatible.

## Overall end-to-end workflow

```text
+-------------------------------+
|       Kubernetes Job          |
|         "data-download"       |
+-------------------------------+
|      Init Container           |
| "download-data-object-store"  |
|  (Fetches data from object    |
|        storage)               |
+-------------------------------+
|        Main Container         |
|  "sdg-data-preprocess"        |
| (Processes the downloaded     |
|         data)                 |
+-------------------------------+
              |
              v
+-------------------------------+
|   "watch for completion"      |
+-------------------------------+
              |
              v
+-----------------------------------+
|   PytorchJob CR training phase 1  |
|                                   |
|       +---------------------+     |
|       |    Master Pod       |     |
|       | (Trains and         |     |
|       |  Coordinates the    |     |
|       |   distributed       |     |
|       |   training)         |     |
|       +---------------------+     |
|                |                  |
|                v                  |
|       +---------------------+     |
|       |    Worker Pod 1     |     |
|       |  (Handles part of   |     |
|       |   the training)     |     |
|       +---------------------+     |
|                |                  |
|                v                  |
|       +---------------------+     |
|       |    Worker Pod 2     |     |
|       |  (Handles part of   |     |
|       |   the training)     |     |
|       +---------------------+     |
+-----------------------------------+
              |
              v
+-------------------------------+
|   "wait for completion"       |
+-------------------------------+
              |
              v
+-----------------------------------+
|   PytorchJob CR training phase 2  |
|                                   |
|       +---------------------+     |
|       |    Master Pod       |     |
|       | (Trains and         |     |
|       |  Coordinates the    |     |
|       |   distributed       |     |
|       |   training)         |     |
|       +---------------------+     |
|                |                  |
|                v                  |
|       +---------------------+     |
|       |    Worker Pod 1     |     |
|       |  (Handles part of   |     |
|       |   the training)     |     |
|       +---------------------+     |
|                |                  |
|                v                  |
|       +---------------------+     |
|       |    Worker Pod 2     |     |
|       |  (Handles part of   |     |
|       |   the training)     |     |
|       +---------------------+     |
+-----------------------------------+
              |
              v
+-------------------------------+
|   "wait for completion"       |
+-------------------------------+
              |
              v
+-------------------------------+
|       Kubernetes Job          |
|         "eval-mt-bench"       |
+-------------------------------+
|      Init Container           |
|     "run-eval-mt-bench"       |
|  (Runs evaluation on MT Bench)|
+-------------------------------+
|        Main Container         |
|  "output-eval-mt-bench-scores"|
| (Outputs evaluation scores)   |
+-------------------------------+
              |
              v
+-------------------------------+
|   "wait for completion"       |
+-------------------------------+
              |
              v
+-------------------------------+
|       Kubernetes Job          |
|          "eval-final"         |
+-------------------------------+
|      Init Container           |
|       "run-eval-final"        |
|  (Runs final evaluation)      |
+-------------------------------+
|        Main Container         |
|  "output-eval-final-scores"   |
|  (Outputs final evaluation    |
|          scores)              |
+-------------------------------+
              |
              v
+-------------------------------+
|   "wait for completion"       |
+-------------------------------+
              |
              v
+-------------------------------+
|       Kubernetes Job          |
|      "trained-model-upload"   |
+-------------------------------+
|        Main Container         |
|  "upload-data-object-store"   |
|  (Uploads the trained model to|
|     the object storage)       |
+-------------------------------+
```

## Requirements

The `standalone.py` script is designed to run within a Kubernetes environment. The following requirements must be met:
* A Kubernetes cluster with the necessary resources to run the InstructLab workflow.
  * Nodes with GPUs available for training.
  * [KubeFlow training operator](https://github.com/kubeflow/training-operator) must be running and `PyTorchJob` CRD must be installed.
  * A [StorageClass](https://kubernetes.io/docs/concepts/storage/storage-classes/) that supports dynamic provisioning with [ReadWriteMany](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes) access mode.
* A Kubernetes configuration that allows access to the Kubernetes cluster.
  * Both cluster and in-cluster configurations are supported.
* SDG data generated and uploaded to an object store.

> [!NOTE]
> The script can be run outside of the cluster from the command-line, but it requires that the user is currently logged into a Kubernetes cluster.

> [!TIP]
> Check the `show` command to display an example of a Kubernetes Job that runs the script. Run `./standalone.py show`.

### RBAC Requirements when running in a Kubernetes Job

The script manipulates a number of Kubernetes resources, and therefore requires the following RBAC
permissions on the [ServiceAccount](https://kubernetes.io/docs/concepts/security/service-accounts/)
running the script:

```yaml
# logs
- verbs:
    - get
    - list
  apiGroups:
    - ""
  resources:
    - pods/log
# Jobs
- verbs:
    - create
    - get
    - list
    - watch
  apiGroups:
    - batch
  resources:
    - jobs
# Pods
- verbs:
    - create
    - get
    - list
    - watch
  apiGroups:
    - ""
  resources:
    - pods
# Secrets
- verbs:
    - create
    - get
  apiGroups:
    - ""
  resources:
    - secrets
# ConfigMaps
- verbs:
    - create
    - get
  apiGroups:
    - ""
  resources:
    - configmaps
# PVCs
- verbs:
    - create
  apiGroups:
    - ""
  resources:
    - persistentvolumeclaims
# PyTorchJob
- verbs:
    - create
    - get
    - list
    - watch
  apiGroups:
    - kubeflow.org
  resources:
    - pytorchjobs
# Watchers
- verbs:
    - get
    - list
    - watch
  apiGroups:
    - ""
  resources:
    - events
```

### Run in a Kubernetes Job

The script can be run in a Kubernetes Job by creating a Job resource that runs the script. The
`show` subcommand displays an example of a Kubernetes Job that runs the script:

```bash
./standalone/standalone.py show \
  --image quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.11-20241004-609ffb8 \
  --script-configmap standalone \
  --script-name script \
  --namespace leseb \
  --args "--storage-class=nfs-csi" \
  --args "--namespace=leseb" \
  --args "--sdg-object-store-secret=sdg-object-store-credentials" \
  --args "--judge-serving-model-secret=judge-serving-details"

apiVersion: batch/v1
kind: Job
metadata:
  name: distributed-ilab
  namespace: leseb
spec:
  template:
    spec:
      containers:
      - args:
        - --storage-class=nfs-csi
        - --namespace=leseb
        - --sdg-object-store-secret=sdg-object-store-credentials
        - --judge-serving-model-secret=judge-serving-details
        command:
        - python3
        - /config/script
        - run
        image: quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.11-20241004-609ffb8
        name: distributed-ilab
        volumeMounts:
        - mountPath: /config
          name: script-config
      restartPolicy: Never
      serviceAccountName: default
      volumes:
      - configMap:
          name: standalone
        name: script-config
```

Optional arguments can be added to the `args` list to customize the script's behavior. They
represent the script options that would be passed to the script if run from the command line.

List of available options of the `show` subcommand:

* `--namespace`: Kubernetes namespace to run the job
* `--name`: Name of the job
* `--image`: The image to use for the job
* `--script-configmap`: The name of the ConfigMap that holds the script
* `--script-name`: The name of the script in the ConfigMap
* `--args`: Additional arguments to pass to the script - can be passed multiple times

## Features

* Run any part of the InstructLab workflow in a standalone environment independently or a full end-to-end workflow:
  * Fetch SDG data, model, and taxonomy from an object store with `sdg-data-fetch` subcommand.
      * Support for AWS S3 and compatible object storage solutions.
      * Configure S3 details via CLI options, environment variables, or Kubernetes secret.
  * Train model with `train` subcommand.
  * Evaluate model by running MT_Bench with `evaluation` subcommand along with `--eval-type mt-bench` option.
  * Final model evaluation with `evaluation` subcommand along with `--eval-type final` option.
      * Final evaluation runs both MT Bench_Branch and MMLU_Branch
  * Push the final model back to the object store -  same location as the SDG data with
    `upload-trained-model` subcommand.
* Dry-run mode to print the generated Kubernetes resources without executing - `--dry-run` option.

> [!NOTE]
> Read about InstructLab model evaluation in the [instructlab/eval repository](https://github.com/instructlab/eval/blob/main/README.md).

## Usage

The `standalone.py` script includes a main command to execute the full workflow, along with
subcommands to run individual parts of the workflow separately. The full workflow includes fetching SDG data from S3, training,
and evaluating a model. To view all available commands, use `./standalone.py --help`.

The script requires information regarding the location and method for accessing the SDG
data/model/taxonomy tree and the evaluation Judge model serving endpoint. This information can be
provided in two main ways:

1. CLI Options or/and Environment Variables: Supply all necessary information via CLI options or environment variables.
    * See [CLI Options](#cli-options) for full list. In particular `--sdg-object-store-*` and `--judge-serving-model-*` options.
2. Kubernetes Secret: Provide the name of a Kubernetes secret that contains all relevant details
   using the `--sdg-object-store-secret` option and `--judge-serving-model-secret` option.
    *  See [Creating the Kubernetes Secret for S3 Details](#creating-the-kubernetes-secret-for-s3-details) for information on how to create the secret.

The examples below assume there is a secret in `my-namespace` named `sdg-data` that holds
information about the S3 bucket and `judge-serving-details` secret that includes information about
the judge server model. A judge server model is assumed to run external to the script. See [Judge
Model Details](#judge-model-details) for required information.


### Usage Examples

```bash
./standalone.py run \
  --namespace my-namespace \
  --judge-serving-model-secret judge-serving-details \
  --sdg-object-store-secret sdg-data
```

Now let's say you only want to fetch the SDG data, you can use the `sdg-data-fetch` subcommand:

```bash
./standalone.py run sdg-data-fetch \
  --namespace my-namespace \
  --judge-serving-model-secret judge-serving-details \
  --sdg-object-store-secret sdg-data
```

Other subcommands are available to run the training and evaluation steps:

```bash
train
evaluation
```

### CLI Options

> [!CAUTION]
> CLI options MUST be positioned AFTER the `run` command and BEFORE any subcommands.

* `--namespace`: The namespace in which the Kubernetes resources are located - **Required**
* `--storage-class`: The storage class to use for the PVCs - **Optional** - Default: cluster default storage class.
* `--nproc-per-node`: Number of GPU to use per node - for training only - **Optional** - Default: 1.
* `--sdg-object-store-secret`: The name of the Kubernetes secret containing the SDG object store
  credentials. **Optional** - If not provided, the script will expect the provided CLI options to fetch the SDG data.
* `--sdg-object-store-endpoint`: The endpoint of the object store. `SDG_OBJECT_STORE_ENDPOINT`
  environment variable can be used as well. **Optional**
* `--sdg-object-store-bucket`: The bucket name in the object store. `SDG_OBJECT_STORE_BUCKET`
  environment variable can be used as well. **Required** - If `--sdg-object-store-secret` is not provided.
* `--sdg-object-store-access-key`: The access key for the object store.
  `SDG_OBJECT_STORE_ACCESS_KEY` environment variable can be used as well. **Required** - If `--sdg-object-store-secret` is not provided.
* `--sdg-object-store-secret-key`: The secret key for the object store.
  `SDG_OBJECT_STORE_SECRET_KEY` environment variable can be used as well. **Required** - If `--sdg-object-store-secret` is not provided.
* `--sdg-object-store-data-key`: The key for the SDG data in the object store. e.g.,
  `data.tar.gz`.`SDG_OBJECT_STORE_DATA_KEY` environment variable can be used as well. **Required** - If `--sdg-object-store-secret` is not provided.
* `--sdg-object-store-verify-tls`: Whether to verify TLS for the object store endpoint (default:
  true). `SDG_OBJECT_STORE_VERIFY_TLS` environment variable can be used as well. **Optional**
* `--sdg-object-store-region`: The region of the object store. `SDG_OBJECT_STORE_REGION` environment
  variable can be used as well. **Optional**
* `--judge-serving-model-endpoint`: Serving endpoint for evaluation. e.g:
  http://serving.kubeflow.svc.cluster.local:8080/v1 - **Optional**
* `--judge-serving-model-name`: The name of the model to use for evaluation. **Optional**
* `--judge-serving-model-api-key`: The API key for the model to evaluate. `JUDGE_SERVING_MODEL_API_KEY`
  environment variable can be used as well. **Optional**
* `--judge-serving-model-secret`: The name of the Kubernetes secret containing the judge serving model
  API key. **Optional** - If not provided, the script will expect the provided CLI options to evaluate the model.
* `--force-pull`: Force pull the data (sdg data, model and taxonomy) from the object store even if it already
  exists in the PVC. **Optional** - Default: false.
* `--training-1-epoch-num`: The number of epochs to train the model for phase 1. **Optional** - Default: 7.
* `--training-2-epoch-num`: The number of epochs to train the model for phase 2. **Optional** -
  Default: 10.
* `--eval-type`: The evaluation type to use. **Optional** - Default: `mt-bench`. Available options:
  `mt-bench`, `final`.
* `--dry-run`: Print the generated Kubernetes resources without executing them. **Optional** - Default: false.


## Example Workflow with Synthetic Data Generation (SDG)

### Generating and Uploading SDG Data

The following example demonstrates how to generate SDG data, package it as a tarball, and upload it
to an object store. This assumes that AWS CLI is installed and configured with the necessary
credentials.
In this scenario the name of the bucket is `sdg-data` and the tarball file is `data.tar.gz`.

```bash
ilab data generate
mv generated data
tar -czvf data.tar.gz data model taxonomy
aws cp data.tar.gz s3://sdg-data/data.tar.gz
```

> [!CAUTION]
> SDG data must exist in a directory called "data".
> The model to train must exist in a directory called "model".
> The taxonomy tree used to generate the SDG data must exist in a directory called "taxonomy".
> The tarball must contain three top-level directories: `data`, `model` and `taxonomy`.

> [!CAUTION]
> The tarball format must be `.tar.gz`.

#### Alternative Method to AWS CLI

Alternatively, you can use the [standalone/sdg-data-on-s3.py](standalone/sdg-data-on-s3.py) script
to upload the SDG data to the object store.

```bash
./sdg-data-on-s3.py upload \
  --object-store-bucket sdg-data \
  --object-store-access-key $ACCESS_KEY \
  --object-store-secret-key $SECRET_KEY \
  --sdg-data-archive-file-path data.tar.gz
```

Run `./sdg-data-on-s3.py upload --help` to see all available options.

### Creating the Kubernetes Secret for S3 Details

The simplest method to supply the script with the required information for retrieving SDG data is by
creating a Kubernetes secret. In the example below, we create a secret called `sdg-data` within the
`my-namespace` namespace, containing the necessary credentials. Ensure that you update the access
key and secret key as needed. The `data_key` field refers to the name of the tarball file in the
object store that holds the SDG data. In this case, it's named `data.tar.gz`, as we previously
uploaded the tarball to the object store using this name.

```bash
cat <<EOF | kubectl create -f -
apiVersion: v1
kind: Secret
metadata:
  name: sdg-data
  namespace: my-namespace
type: Opaque
stringData:
  bucket: sdg-data
  access_key: *****
  secret_key: *****
  data_key: data.tar.gz
EOF
```

> [!WARNING]
> The secret must be part of the same namespace as the resources that the script interacts with.
> It's inherented from the `--namespace` option.

The list of all supported keys:

* `bucket`: The bucket name in the object store - **Required**
* `access_key`: The access key for the object store - **Required**
* `secret_key`: The secret key for the object store - **Required**
* `data_key`: The key for the SDG data in the object store - **Required**
* `verify_tls`: Whether to verify TLS for the object store endpoint (default: true) - **Optional**
* `endpoint`: The endpoint of the object store, e.g: https://s3.openshift-storage.svc:443 - **Optional**
* `region`: The region of the object store - **Optional**

A similar operation can be performed for the evaluation judge model serving service. Currently, the script expects the Judge serving service to be running and accessible from within the cluster. If it is not present, the script will not create this resource.

```bash
cat <<EOF | kubectl create -f -
apiVersion: v1
kind: Secret
metadata:
  name: judge-serving-details
  namespace: my-namespace
type: Opaque
stringData:
  JUDGE_API_KEY: ********
  JUDGE_ENDPOINT: https://mistral-sallyom.apps.ocp-beta-test.nerc.mghpcc.org/v1
  JUDGE_NAME: mistral
EOF
```

The list of all mandatory keys:

* `JUDGE_API_KEY`: The API key for the model to evaluate - **Required**
* `JUDGE_ENDPOINT`: Serving endpoint for evaluation - **Required**
* `JUDGE_NAME`: The name of the model to use for evaluation - **Required**

> [!WARNING]
> Mind the upper case of the keys, as the script expects them to be in upper case.

#### Running the Script Without Kubernetes Secret

Alternatively, you can provide the necessary information directly via CLI options or environment,
the script will use the provided information to fetch the SDG data and create its own Kubernetes
Secret named `sdg-object-store-credentials` in the same namespace as the resources it interacts with (in this case, `my-namespace`).

```bash
export JUDGE_SERVING_MODEL_API_KEY=********

./standalone.py run \
  --namespace my-namespace \
  --judge-serving-model-endpoint http://serving.kubeflow.svc.cluster.local:8080/v1 \
  --judge-serving-model-name my-model \
  --sdg-object-store-access-key key \
  --sdg-object-store-secret-key key \
  --sdg-object-store-bucket sdg-data \
  --sdg-object-store-data-key data.tar.gz
```

### Judge Model Details

A judge model is assumed to be running external to the script. This is used for model evaluation.

* The `--judge-serving-model-endpoint` and `--judge-serving-model-name` values will be stored in a ConfigMap named `judge-serving-details` in the same namespace as the resources that the script interacts with.
* `--judge-serving-model-api-key` or environment variable `JUDGE_SERVING_MODEL_API_KEY` value will be stored in a secret named `judge-serving-details` in the same namespace as the resources that the script interacts with.
* In all examples, the `JUDGE_SERVING_MODEL_API_KEY` environment variable is exported rather than setting the CLI option.

#### Advanced Configuration Using an S3-Compatible Object Store

If you don't use the official AWS S3 endpoint, you can provide additional information about the object store:

```bash
export JUDGE_SERVING_MODEL_API_KEY=********

./standalone.py run \
  --namespace my-namespace \
  --judge-serving-model-endpoint http://serving.kubeflow.svc.cluster.local:8080/v1 \
  --judge-serving-model-name my-model \
  --sdg-object-store-access-key key \
  --sdg-object-store-secret-key key \
  --sdg-object-store-bucket sdg-data \
  --sdg-object-store-data-key data.tar.gz \
  --sdg-object-store-verify-tls false \
  --sdg-object-store-endpoint https://s3.openshift-storage.svc:443
```

> [!IMPORTANT]
> The `--sdg-object-store-endpoint` option must be provided in the format
> `scheme://host:<port>`, the port can be omitted if it's the default port.

> [!TIP]
> If you don't want to run the entire workflow and only want to fetch the SDG data, you can use the `run sdg-data-fetch ` command

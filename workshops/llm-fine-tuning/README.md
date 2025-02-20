# LLM Fine-Tuning Workshop

## Requirements

* An OpenShift cluster with admin permissions (for the setup steps)
* The `oc`, `curl`, and `git` (or equivalent) binaries installed locally
* Enough worker nodes with NVIDIA GPUs (Ampere-based or newer recommended) or AMD GPUs (AMD Instinct MI300X)
* The NFD operator and the NVIDIA GPU operator or AMD GPU operator installed and configured 
* A dynamic storage provisioner supporting RWX PVC provisioning (or see the NFS provisioner section)

## Setup

### Install OpenShift AI

* Log into your OpenShift Web console
* Go to "Operators" > "OperatorHub" > "AI/Machine Learning"
* Select the "Red Hat OpenShift AI" operator
* Install the latest version and Create a default DataScienceCluster resource

### Checkout the workshop

* Clone the following repository:
    ```console
    git clone https://github.com/opendatahub-io/distributed-workloads.git
    ```
* Change directory:
    ```console
    cd workshops/llm-fine-tuning
    ```

### NFS Provisioner (optional)

> [!NOTE]
> This is optional if your cluster already has a PVC dynamic provisioner with RWX support.

* Install the NFS CSI driver:
    ```console
    curl -skSL https://raw.githubusercontent.com/kubernetes-csi/csi-driver-nfs/v4.9.0/deploy/install-driver.sh | bash -s v4.9.0 --
    ```
* Create a new project:
    ```console
    oc new-project nfs
    ```
* Deploy the in-cluster NFS server:
    ```console
    oc apply -f nfs/nfs_deployment.yaml
    ```
* Create the NFS StorageClass:
    ```console
    oc apply -f nfs/nfs_storage_class.yaml
    ```

### Configure RHOAI

* Go the OpenShift AI Dashboard (accessible from the applications menu in the top navigation bar)
* Go to "Settings" > "Storage classes"
* Check the storage class supporting RWX PVC provisioning you plan to use, or the `nfs-csi` one created previously, is enabled

## Manage Quotas with Kueue

* Update the `nodeLabels` in the `kueue/resource_flavor.yaml` file to match those of your AI worker nodes
* Create the ResourceFlavor:
    ```console
    oc apply -f kueue/resource_flavor.yaml
    ```
* Update the `team1` and `team2` ClusterQueues according to your cluster compute resources and the ResourceFlavor you've just created
* Create the ClusterQueues:
    ```console
    oc apply -f "kueue/team*_cq.yaml"
    ```

## Fine-Tune LLama 3.1 with Ray

### Create a new project

* Go the OpenShift AI Dashboard (accessible from the applications menu in the top navigation bar)
* Go to "Data Science Projects"
* Click "Create project"
* Choose a name and click "Create"

### Create a local queue

* From a terminal, create a LocalQueue pointing to your team ClusterQueue:
    ```console
    oc apply -f "kueue/local_queue.yaml"
    ```

### Create a workbench

* In the project you've just created, click "Create workbench"
* Enter a name
* Select the "Standard Data Science" notebook image
* In "Cluster storage", click "Create storage"
* Enter `training-storage` as a name and select the storage class with RWX capability, or the `nfs-csi` one if you created it previously
* Enter a mount directory under `/opt/app-root/src/`
* Click "Create workbench"
* Back to the project page, wait for the workbench to become ready and then open it

### Create a Ray cluster

* In the workbench you've just created, clone the https://github.com/opendatahub-io/distributed-workloads.git repository (you can click on the "Git clone" under the top menu)
* Navigate to "distributed-workloads" / "examples" / "ray-finetune-llm-deepspeed"
* Open the "ray_finetune_llm_deepspeed.ipynb" notebook
* In the "Authenticate the CodeFlare SDK" cell, enter your cluster API server URL and your authorization token
** This can either be retrieved by running `oc whoami -t`,
** Or from the OpenShift Web console, click on the user name at the right-hand side of the top navigation bar, and then select "Copy login command"
* In the "Configure the Ray cluster" cell:
    * Add the following fields to the `ClusterConfiguration`:
        ```python
        volumes=[
            V1Volume(
                name="training-storage",
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name="training-storage"),
            ),
        ],
        volume_mounts=[
            V1VolumeMount(name="training-storage", mount_path="/opt/app-root/src/training/"),
        ],
        ```
    * Review the compute resources so they match that of your cluster

### Open the Ray cluster dashboard

* Wait until the Ray cluster becomes ready
* Once you've executed the `cluster.details()` cell, you can click on the Ray cluster dashboard URL printed in the output.

### Submit the fine-tuning job

* In the "Storage configuration" cell, set the `storage_path` variable to `/opt/app-root/src/training`
* In the "job submission" cell:
    * Add the `HF_HOME` environment variable and set it to `f'{storage_path}/.cache'`
    * Review the compute resources so they match that of the Ray cluster you've created

### Monitor training with TensorBoard

* Install TensorBoard in the Ray head node:
    ```console
    oc exec `oc get pod -l ray.io/node-type=head -o name` -- pip install tensorboard
    ```
* Start TensorBoard server:
    ```console
    oc exec `oc get pod -l ray.io/node-type=head -o name` -- tensorboard --logdir /tmp/ray --bind_all --port 6006
    ```
* Port-foward the TensorBoard UI endpoint:
    ```console
    oc port-forward `oc get pod -l ray.io/node-type=head -o name` 6006:6006
    ```
* Access TensorBoard at http://localhost:6006

## Fine-Tune LLama 3.1 with Kubeflow Training

### Enable the training operator

* In the OpenShift Web console, navigate to
* DataScienceCluster resource

### Configure the fine-tuning job

* Review / edit the `kfto/config.yaml` configuration file
* Create the fine-tuning job ConfigMap by running:
    ```console
    oc create configmap llm-training --from-file=config.yaml=kfto/config.yaml --from-file=sft.py=kfto/sft.py
    ```

### Create the fine-tuning job

* Review / edit the `kfto/job.yaml` file
* Set the value of the `HF_TOKEN` environment variable if needed
* Create the fine-tuning PyTorchJob by running:
    ```
    oc apply -f kfto/job.yaml
    ```

### Monitor training with TensorBoard

* Start TensorBoard server:
    ```console
    oc exec `oc get pod -l training.kubeflow.org/job-role=master -o name` -- tensorboard --logdir /mnt/runs --bind_all --port 6006
    ```
* Port-foward the TensorBoard UI endpoint:
    ```console
    oc port-forward `oc get pod -l training.kubeflow.org/job-role=master -o name` 6006:6006
    ```
* Access TensorBoard at http://localhost:6006

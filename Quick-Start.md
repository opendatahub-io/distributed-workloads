# Quick Start Guide for Distributed Workflows with the CodeFlare Stack

This quick start guide is intended to walk existing Open Data Hub users through installation of the CodeFlare stack and an initial demo using the CodeFlare-SDK from within a Jupyter notebook environment. This will enable users to run and submit distributed workloads.  

The CodeFlare-SDK was built to make managing distributed compute infrastructure in the cloud easy and intuitive for Data Scientists. However, that means there needs to be some cloud infrastructure on the backend for users to get the benefit of using the SDK. Currently, we support the CodeFlare stack, which consists of the Open Source projects, [MCAD](https://github.com/project-codeflare/multi-cluster-app-dispatcher), [Instascale](https://github.com/project-codeflare/instascale), [Ray](https://www.ray.io/), and [Pytorch](https://pytorch.org/).

This stack integrates well with [Open Data Hub](https://opendatahub.io/), and helps to bring batch workloads, jobs, and queuing to the Data Science platform.

## Prerequisites

### Resources

In addition to the resources required by the odh-core deployment, you will need the following to deploy the Distributed Workloads stack infrastructure pods:

```text
Total:
    CPU: 4100m
    Memory: 4608Mi

# By component
Ray:
    CPU: 100m
    Memory: 512Mi
MCAD
    cpu: 2000m
    memory: 2Gi
InstaScale:
    cpu: 2000m
    memory: 2Gi
```

NOTE: The above resources are just for the infrastructure pods. To be able to run actual workloads on your cluster you will need additional resources based on the size and type of workload.

### OpenShift and Open Data Hub

This Quick Start guide assumes that you have administrator access to an OpenShift cluster and an existing Open Data Hub installation on your cluster. If you do not currently have the Open Data Hub operator installed on your cluster, you can find instructions for installing it [here](https://opendatahub.io/docs/quick-installation/). The default settings for the Open Data Hub Operator will suffice.

### CodeFlare Operator

The CodeFlare operator must be installed from the OperatorHub on your OpenShift cluster. The default settings will
suffice.

### NFD and GPU Operators

If you want to run GPU enabled workloads, you will need to install the [Node Feature Discovery Operator](https://github.com/openshift/cluster-nfd-operator) and the [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) from the OperatorHub. For instructions on how to install and configure these operators, we recommend [this guide](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/openshift/steps-overview.html#high-level-steps).


## Creating K8s resources

1. Create the opendatahub namespace with the following command:

    ```bash
    oc create ns opendatahub
    ```

1. Apply the odh-core kfdef with this command:

    ```bash
    oc apply -f https://raw.githubusercontent.com/opendatahub-io/odh-manifests/master/kfdef/odh-core.yaml -n opendatahub
    ```

1. Create the CodeFlare-Stack kfdef with this command:

    ```bash
    oc apply -f https://raw.githubusercontent.com/opendatahub-io/distributed-workloads/main/codeflare-stack-kfdef.yaml -n opendatahub
    ```

Applying the above kfdef will result in the following objects being added to your cluster:

1. MCAD
1. InstaScale
1. KubeRay Operator
1. CodeFlare Notebook Image for the Open Data Hub notebook interface

    This image is managed by project CodeFlare and contains the correct packages of codeflare-sdk, pytorch, torchx, ect required to run distributed workloads.

At this point you should be able to go to your notebook spawner page and select "Codeflare Notebook" from your list of notebook images and start an instance.

You can access the spawner page through the Open Data Hub dashboard. The default route should be `https://odh-dashboard-<your ODH namespace>.apps.<your cluster's uri>`. Once you are on your dashboard, you can select "Launch application" on the Jupyter application. This will take you to your notebook spawner page.

## Submit your first job

We can now go ahead and submit our first distributed model training job to our cluster.

This can be done from any python based environment, including a script or a jupyter notebook. For this guide, we'll assume you've selected the "Codeflare Notebook" from the list of available images on your notebook spawner page.

### Clone the demo code

Once your notebook environment is ready, in order to test our CodeFlare stack we will want to run though some of the demo notebooks provided by the CodeFlare community. So let's start by cloning their repo into our working environment.

```bash
git clone https://github.com/project-codeflare/codeflare-sdk
cd codeflare-sdk
```

We will rely on this demo code to train an mnist model. So feel free to open `codeflare-sdk/demo-notebooks/batch-job/batch_mnist.ipynb` to follow along instead.

### Run the demo notebook

First, we will import what we need from the SDK.

```python
from codeflare_sdk.cluster.cluster import Cluster, ClusterConfiguration
from codeflare_sdk.cluster.auth import TokenAuthentication

```

Then we will go ahead and create an authentication object to access our cluster.

```python
# Create authentication object for oc user permissions
auth = TokenAuthentication(
    token = "XXXX",
    server = "XXXX",
    skip_tls=True
)
auth.login()
```

Next, we will define the configuration we'd like for our Ray cluster. A user can update this as needed for the resource requirements of their job.

_Instascale specific configs:_

The configuration for `machine_types` is only used if you have instascale installed. It defines the machine types for the head node and worker nodes, in that order. You must also have the appropriate `machine_set` templates available on your cluster for instascale to recognize them.

If you are working in an on-prem environment, you can simply set `instascale=False` and ignore the `machine_types` configuration.

```python
cluster_config = ClusterConfiguration(
    name='mnist', 
    namespace="opendatahub", 
    machine_types = ["m4.xlarge", "g4dn.xlarge"]
    min_worker=2, 
    max_worker=2, 
    min_cpus=2, 
    max_cpus=2, 
    min_memory=8, 
    max_memory=8, 
    gpu=1, 
    instascale=True,
)
```

Once the cluster configurations are defined, we can go ahead and create our cluster object.

```python
cluster = Cluster(cluster_config)
```

In addition to instantiating our cluster object, this will also write a file, `mnist.yaml`, to your working directory. This file defines an AppWrapper custom resource; everything MCAD needs to deploy your Ray cluster.

Next, we can apply this YAML file and spin up our Ray cluster.

```python
cluster.up()
```

You can check the status of the Ray cluster and see when its ready to use with:

```Python
cluster.status()
```

Once the cluster is up, you are ready to submit your first job. Here we will rely on torchx with a ray backend as our distributed training engine. We've created a file `demo-notebook/batch-job/mnist.py` with the required pytorch training code that we'll be submitting.  

```python
! torchx run -s ray -cfg dashboard_address=mnist-head-svc.<Your Namespace>.svc:8265,requirements=requirements.txt dist.ddp -j 2x1 --gpu 1 --script mnist.py
```

Once the job is submitted you can follow it on the Ray dashboard using `cluster.cluster_dash board_uri()` to get the link or `cluster.list_jobs()` and `cluster.job_status(job_id)` to output the job status directly into you're notebook.

Finally, once the job is done you can shutdown your Ray nodes, logout and free up the resources on your cluster.

```python
cluster.down()
auth.logout()
```

Great! You have now submitted your first distributed training job with CodeFlare!

## Next Steps

And with that you have gotten started using the CodeFlare stack alongside your Open Data Hub Deployment to add distributed workflows and batch computing to your machine learning platform.

You are now ready to try out the stack with your own machine learning workloads. If you'd like some more examples, you can also run through the existing demo code provided by the Codeflare-SDK community.

* [Submit batch jobs](https://github.com/project-codeflare/codeflare-sdk/tree/main/demo-notebooks/batch-job)
* [Run an interactive session](https://github.com/project-codeflare/codeflare-sdk/tree/main/demo-notebooks/interactive)

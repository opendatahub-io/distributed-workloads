# Deploying Ray on Open Data Hub via the KubeRay Operator

The code found here is a subset of <https://github.com/ray-project/kuberay>. Specifically, the manifests needed to enable KubeRay with an Open Data Hub deployment.

## Components of the KubeRay deployment

1. Custom Resource Definitions
2. Operator
3. Roles and Service Accounts

## Installation

There are two ways to install KubeRay. You can either **install it directly into your cluster alongside an existing Open Data Hub deployment using Kustomize** OR you can **include it as an additional component in your Open Data Hub KfDef**. Below we will walk through both approaches.

## Install via Kustomize

This method requires that you have [kustomize](https://kustomize.io/) and the [oc client](https://docs.openshift.com/container-platform/4.12/cli_reference/openshift_cli/getting-started-cli.html) installed in your environment. Then, from within the `ray/` directory of this repo you can run the following:

```bash
cd operator/base
oc kustomize > deploy_kuberay.yaml
oc create -f deploy_kuberay.yaml
```

## Install via KfDef

Alternatively, you can include the KubeRay components in your ODH KfDef. This will ensure that all of the components above get installed as part of your ODH deployment.

```yaml
 # KubeRay
  - kustomizeConfig:
      repoRef:
        name: manifests
        path: ray/operator
    name: ray-operator
```

There is a sample kfdef at the root of this repository that you can use to deploy the recommended CodeFlare stack, which includes KubeRay.

## Confirm the operator is running

Once installed, you can confirm the KubeRay operator has been deployed correctly with `oc get pods`.

```bash
$ oc get pods
NAME                               READY   STATUS    RESTARTS        AGE
kuberay-operator-867bc855b7-2tzxs      1/1     Running   0               4d19h
```

## Create a test cluster

Now that the operator is running, let's create a small Ray cluster and make sure the operator can handle the request correctly. From whatever namespace you want to use you can run the following command:

```bash
oc apply -f tests/resources/ray/ray-test-cluster-test.yaml
```

## Confirm the cluster is running

```bash
$ oc get RayCluster
NAME                   DESIRED WORKERS   AVAILABLE WORKERS   STATUS   AGE
kuberay-cluster-test   1                 2                   ready    13s

```

Once the cluster is running you should be able to connect to it to use ray in a python script or jupyter notebook by using `ray.init('ray://kuberay-cluster-test-head-svc:10001')`.

Make sure that the version of ray in your environment matches the version of ray that is running in your cluster. If you used the `ray-test-cluster-test.yaml` above then you should be using `ray==2.1.0`.

```bash
pip install ray==2.1.0
```

Then, from your python environment you can connect to the cluster with the following code:

```python
import ray
ray.init('ray://kuberay-cluster-test-head-svc:10001')
```

## Delete your test cluster

Now that you've confirmed everything is working feel free to delete your Ray test cluster. `oc delete RayCluster kuberay-cluster-test`

That's it! You should now be able to use Ray as part of your Open Data Hub Deployment for distributed and parallel computing.

# Custom Container Image Examples

> [!IMPORTANT]
> These examples are provided for documentation purpose, on how to build custom container images for Distributed Workloads.
> These container examples are not supported, not part of OpenShift AI.

## Prerequisites

You need to have Podman install on your local environment, to build and push these custom container images to your registry, so they can be used to run distributed workloads in OpenShift AI.

## Build

The container images can be built using Podman, e.g., for the `ray-torch-cuda` example:

```
export IMG=ray-torch-cuda
cd $IMG
podman build -t ${IMG} -f Dockerfile
```

You can then push that image to your container registry.
Alternatively, you can use the [integrated OpenShift container registry](https://docs.openshift.com/container-platform/4.16/registry/index.html#registry-integrated-openshift-registry_registry-overview), by following these intructions:

1. Expose the integrated container registry:
    ```
    oc patch configs.imageregistry.operator.openshift.io/cluster --patch '{"spec":{"defaultRoute":true}}' --type=merge
    ```
2. Wait for the route to the container registry to be admitted:
    ```
    oc wait -n openshift-image-registry route/default-route --for=jsonpath='{.status.ingress[0].conditions[0].status}'=True
    ```
3. Login to the container registry, e.g., with Podman:
    ```
    podman login -u $(oc whoami) -p $(oc whoami -t) $(oc registry info)
    ```
4. Push the image to the integrated container registry:
    ```
    podman tag ${IMG} $(oc registry info)/$(oc project -q)/${IMG}
    podman push $(oc registry info)/$(oc project -q)/${IMG}
    ```
5. Retrieve the image repository for the tag you want, e.g.:
    ```
    oc get is ${IMG} -o jsonpath='{.status.tags[?(@.tag=="<TAG>")].items[0].dockerImageReference}'
    ```
6. You can now use that image repository in your notebook.

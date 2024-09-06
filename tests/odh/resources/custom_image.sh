#!/bin/bash

namespace=$1
image=$2

echo "switch to current project . . ."
oc project $namespace

echo "Build custom container image using podman . . ."
cd ./../../images/runtime/examples
cd $image
podman build -t $image -f Dockerfile

echo "Expose the integrated container registry . . ."
oc patch configs.imageregistry.operator.openshift.io/cluster --patch '{"spec":{"defaultRoute":true}}' --type=merge

echo "Wait for the route to the container registry to be admitted . . ."
oc wait -n openshift-image-registry route/default-route --for=jsonpath='{.status.ingress[0].conditions[0].status}'=True

echo "Login to the container registry . . ."
podman login -u $(oc whoami) -p $(oc whoami -t) $(oc registry info)

echo "Push the image to the integrated container registry . . ."
podman tag $image $(oc registry info)/$namespace/$image
podman push $(oc registry info)/$namespace/$image

echo "Custom Ray Image is . . . "
oc get is $image -o jsonpath='{.status.tags[?(@.tag=="latest")].items[0].dockerImageReference}'

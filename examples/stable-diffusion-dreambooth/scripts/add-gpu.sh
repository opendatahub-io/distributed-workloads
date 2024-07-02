#!/bin/sh

WORK_DIR=$1/gpu

echo "Work dir: $WORK_DIR"
mkdir -p $WORK_DIR

echo
echo "Finding the appropriate machineset in the right az:"
oc get machinesets -n openshift-machine-api | grep '1a\|1b\|2a\|2b' || { echo "No suitable machineset found"; exit 1;}

MS_NAME=$(oc get machinesets -n openshift-machine-api | grep '1a\|1b\|2a\|2b' | head -n1 | awk '{print $1}')
MS_NAME_GPU="${MS_NAME}-gpu"
INSTANCE_TYPE=$(yq eval '.spec.template.spec.providerSpec.value.instanceType' $WORK_DIR/ms.yaml)

echo 
echo "Machineset: $MS_NAME"
echo "New Machineset: $MS_NAME_GPU"

oc get machineset $MS_NAME -n openshift-machine-api -o yaml > $WORK_DIR/ms.yaml

sed -i .bak "s/${MS_NAME}/${MS_NAME_GPU}/g" $WORK_DIR/ms.yaml
sed -i .bak "s/${INSTANCE_TYPE}/p3.8xlarge/g" $WORK_DIR/ms.yaml
sed -i .bak 's/volumeSize: 100/volumeSize: 200/g' $WORK_DIR/ms.yaml
sed -i .bak 's/replicas: 3/replicas: 1/g' $WORK_DIR/ms.yaml

oc create -f $WORK_DIR/ms.yaml
rm -rf $WORK_DIR

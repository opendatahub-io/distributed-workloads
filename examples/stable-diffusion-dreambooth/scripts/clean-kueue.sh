#!/bin/sh

NAMESPACE=$1

echo "Deleting all rayclusters in namespace ${NAMESPACE}"
oc delete raycluster --all -n ${NAMESPACE} > /dev/null

echo "Deleting all localqueue in namespace ${NAMESPACE}"
oc delete localqueue --all -n ${NAMESPACE} > /dev/null

echo "Deleting all clusterqueues"
oc delete clusterqueue --all --all-namespaces > /dev/null

echo "Deleting all resourceflavors"
oc delete resourceflavor --all --all-namespaces > /dev/null
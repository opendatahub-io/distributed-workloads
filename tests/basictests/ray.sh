#!/bin/bash

source $TEST_DIR/common

MY_DIR=$(readlink -f `dirname "${BASH_SOURCE[0]}"`)

RESOURCEDIR="${MY_DIR}/../resources"

source ${MY_DIR}/../util

os::test::junit::declare_suite_start "$MY_SCRIPT"

function check_ray_operator() {
    header "Testing Ray Operator"
    os::cmd::expect_success "oc project ${ODHPROJECT}"
    os::cmd::try_until_text "oc get crd rayclusters.ray.io" "rayclusters.ray.io" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get role kuberay-operator-leader-election" "kuberay-operator-leader-election" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get rolebinding kuberay-operator-leader-election" "kuberay-operator-leader-election" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get sa kuberay-operator" "kuberay-operator" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get deployment kuberay-operator" "kuberay-operator" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get pods -l app.kubernetes.io/component=kuberay-operator --field-selector='status.phase=Running' -o jsonpath='{$.items[*].metadata.name}' | wc -w" "1" $odhdefaulttimeout $odhdefaultinterval    
}

function start_test_ray_cluster(){
    header "Starting Ray Cluster"
    os::cmd::expect_success "oc project ${ODHPROJECT}"
    os::cmd::expect_success "oc apply -f ${RESOURCEDIR}/ray/ray-test-cluster-test.yaml"
    os::cmd::try_until_text "oc get RayCluster kuberay-cluster-test" "kuberay-cluster-test" $odhdefaulttimeout $odhdefaultinterval
    sleep 15
}

function check_functionality(){
    header "Testing Ray Functionality"
    os::cmd::expect_success "oc project ${ODHPROJECT}"
    os::cmd::expect_success "oc apply -f ${RESOURCEDIR}/ray/ray-simple-test.yaml"
    sleep 30
    os::cmd::try_until_text "oc get pods -l app=ray-simple-test -o jsonpath='{$.items[*].status.containerStatuses[0].lastState.terminated.exitCode}'" "" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get pods -l app=ray-simple-test -o jsonpath='{$.items[*].status.containerStatuses[0].restartCount}'" "0" $odhdefaulttimeout $odhdefaultinterval
    pod_name=($(oc get pods -l app=ray-simple-test -o jsonpath='{$.items[*].metadata.name}'))
    os::cmd::try_until_text "oc logs ${pod_name} | grep 'Simple tests passed'" "Simple tests passed" $odhdefaulttimeout $odhdefaultinterval
}

function clean_up_ray_cluster(){
    header "Cleaning up Ray cluster"
    os::cmd::expect_success "oc project ${ODHPROJECT}"
    os::cmd::expect_success "oc delete deployment ray-simple-test -n ${ODHPROJECT}"
    os::cmd::expect_success "oc delete RayCluster kuberay-cluster-test -n ${ODHPROJECT}"
}

check_ray_operator
start_test_ray_cluster
check_functionality
clean_up_ray_cluster

os::test::junit::declare_suite_end

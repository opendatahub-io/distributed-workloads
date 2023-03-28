#!/bin/bash

source $TEST_DIR/common

MY_DIR=$(readlink -f `dirname "${BASH_SOURCE[0]}"`)

RESOURCEDIR="${MY_DIR}/../resources"

source ${MY_DIR}/../util

os::test::junit::declare_suite_start "$MY_SCRIPT"

function install_codeflare_operator() {
    header: "Installing Codeflare Operator"
}

function install_distributed_workloads_kfdef(){
    header: "Installing distributed workloads kfdef"
    os::cmd::expect_success "oc apply -f ../distributed-workloads-kfdef.yaml"
    os::cmd::expect_success "oc project opendatahub"
    
    # KubeRay tests
    os::cmd::try_until_text "oc get crd rayclusters.ray.io" "rayclusters.ray.io" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get role kuberay-operator-leader-election" "kuberay-operator-leader-election" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get rolebinding kuberay-operator-leader-election" "kuberay-operator-leader-election" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get sa kuberay-operator" "kuberay-operator" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get deployment kuberay-operator" "kuberay-operator" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get pods -l app.kubernetes.io/component=kuberay-operator --field-selector='status.phase=Running' -o jsonpath='{$.items[*].metadata.name}' | wc -w" "1" $odhdefaulttimeout $odhdefaultinterval   

}

function test_mcad_torchx_functionality() {
    header: "Testing MCAD TorchX Functionality" 
}

function tests_mcad_ray_functionality() {
    header: "Testing MCAD Ray Functionality"
}    

function uninstall_distributed_workloads_kfdef() {
    header: "Uninstalling distributed workloads kfdef"
}

function uninstall_codeflare_operator() {
    header: "Uninstalling Codeflare Operator"
}


install_codeflare_operator
install_distributred_workloads_kfdef
test_mcad_torchx_functionality
tests_mcad_ray_functionality
uninstall_distributed_workloads_kfdef
uninstall_codeflare_operator


os::test::junit::declare_suite_end

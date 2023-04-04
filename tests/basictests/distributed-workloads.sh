#!/bin/bash

source $TEST_DIR/common

MY_DIR=$(readlink -f `dirname "${BASH_SOURCE[0]}"`)

RESOURCEDIR="${MY_DIR}/../resources"

source ${MY_DIR}/../../util

os::test::junit::declare_suite_start "$MY_SCRIPT"

function example_test() {
    header "Running example test"
    os::cmd::expect_success "oc project ${ODHPROJECT}"
    os::cmd::expect_success "oc get pods"
}

function install_codeflare_operator() {
    header "Installing Codeflare Operator"
    os::cmd::expect_success "oc apply -f $RESOURCEDIR/codeflare-subscription.yaml"

    # Wait until both pods are ready
    os::cmd::try_until_text "oc get pods -n openshift-operators | grep "codeflare-operator-controller-manager" | awk '{print \$2}'" "2/2" $odhdefaulttimeout $odhdefaultinterval

    # Ensure that all CRDs are created
    os::cmd::expect_success_and_text "oc get crd instascales.codeflare.codeflare.dev  | wc -l" "2"
    os::cmd::expect_success_and_text "oc get crd mcads.codeflare.codeflare.dev | wc -l" "2"
    os::cmd::expect_success_and_text "oc get crd appwrappers.mcad.ibm.com | wc -l" "2"
    os::cmd::expect_success_and_text "oc get crd queuejobs.mcad.ibm.com | wc -l" "2"
    os::cmd::expect_success_and_text "oc get crd schedulingspecs.mcad.ibm.com | wc -l" "2"
}

function install_distributed_workloads_kfdef(){
    header "Installing distributed workloads kfdef"   
}

function test_mcad_torchx_functionality() {
    header "Testing MCAD TorchX Functionality" 
}

function tests_mcad_ray_functionality() {
    header "Testing MCAD Ray Functionality"
}    

function uninstall_distributed_workloads_kfdef() {
    header "Uninstalling distributed workloads kfdef"
}

function uninstall_codeflare_operator() {
    header "Uninstalling Codeflare Operator"
    # Figure out the csv name of the codeflare operator
    CODEFLARE_CSV_VER=$(oc get csv | awk '{print $1}' |grep codeflare-operator)

    # Uninstall the subscription, csv and crds of the CodeFlare Operator
    os::cmd::expect_success "oc delete sub codeflare-operator -n openshift-operators"
    os::cmd::expect_success "oc delete csv $CODEFLARE_CSV_VER -n openshift-operators"
    os::cmd::expect_success "oc delete crd appwrappers.mcad.ibm.com instascales.codeflare.codeflare.dev mcads.codeflare.codeflare.dev queuejobs.mcad.ibm.com schedulingspecs.mcad.ibm.com"

    # Wait until the CodeFlare Operator pods is gone
    os::cmd::try_until_text "oc get pods -n openshift-operators -l control-plane=controller-manager" "No resources found in openshift-operators namespace." $odhdefaulttimeout $odhdefaultinterval

    # Ensure that the CodeFlare Operator subscription and csv are deleted
    os::cmd::expect_failure "oc get sub codeflare-operator -n openshift-operators"
    os::cmd::expect_failure "oc get csv codeflare-operator.v0.0.1 -n openshift-operators"

    # Ensure that all CRDs are deleted
    os::cmd::expect_failure "oc get crd instascales.codeflare.codeflare.dev"
    os::cmd::expect_failure "oc get crd mcads.codeflare.codeflare.dev"
    os::cmd::expect_failure "oc get crd appwrappers.mcad.ibm.com"
    os::cmd::expect_failure "oc get crd queuejobs.mcad.ibm.com"
    os::cmd::expect_failure "oc get crd schedulingspecs.mcad.ibm.com"
}


example_test
install_codeflare_operator
install_distributed_workloads_kfdef
test_mcad_torchx_functionality
tests_mcad_ray_functionality
uninstall_distributed_workloads_kfdef
uninstall_codeflare_operator


os::test::junit::declare_suite_end

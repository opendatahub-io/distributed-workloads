#!/bin/bash

source $TEST_DIR/common

MY_DIR=$(readlink -f `dirname "${BASH_SOURCE[0]}"`)

RESOURCEDIR="${MY_DIR}/../resources"

source ${MY_DIR}/../util

os::test::junit::declare_suite_start "$MY_SCRIPT"

function example_test() {
    header "Running example test"
    os::cmd::expect_success "oc project ${ODHPROJECT}"
    os::cmd::expect_success "oc get pods"
}

function install_codeflare_operator() {
    header "Installing Codeflare Operator"
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
}


example_test
install_codeflare_operator
install_distributed_workloads_kfdef
test_mcad_torchx_functionality
tests_mcad_ray_functionality
uninstall_distributed_workloads_kfdef
uninstall_codeflare_operator


os::test::junit::declare_suite_end

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
    os::cmd::expect_success "oc apply -f https://raw.githubusercontent.com/opendatahub-io/distributed-workloads/main/codeflare-stack-kfdef.yaml -n ${ODHPROJECT}"

    # Ensure that MCAD, Instascale, KubeRay pods start
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} |grep mcad-controller | awk '{print \$3}'"  "Running"
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} |grep instascale-instascale | awk '{print \$3}'"  "Running"
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} |grep kuberay-operator | awk '{print \$3}'"  "Running"

    # Ensure the codeflare-notebook imagestream is there
    os::cmd::expect_success_and_text "oc get imagestreams -n ${ODHPROJECT} codeflare-notebook --no-headers=true |awk '{print \$1}'"  "codeflare-notebook"
}

function test_mcad_torchx_functionality() {
    header "Testing MCAD TorchX Functionality"
}

function tests_mcad_ray_functionality() {
    header "Testing MCAD Ray Functionality"

    ########### ToDo: Clean Cluser should be free of those resoruces ############
    # Clean up resources
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin|| true"
    os::cmd::expect_success "oc delete cm notebooks || true"
    os::cmd::expect_success "oc delete appwrapper mnisttest -n default || true"
    os::cmd::expect_success "oc delete raycluster mnisttest -n default || true"
    ########################################################################################

    # Wait for the notebook controller ready
    os::cmd::try_until_text "oc get deployment odh-notebook-controller-manager -n ${ODHPROJECT} --no-headers=true | awk '{print \$2}'" "1/1" $odhdefaulttimeout $odhdefaultinterval

    # Create a mcad.ipynb as a configMap
    os::cmd::expect_success "oc create configmap notebooks --from-file=${RESOURCEDIR}/mcad.ipynb"

    # Spawn notebook-server using the codeflare custom nb image
    os::cmd::expect_success "cat ${RESOURCEDIR}/custom-nb-small.yaml | sed s/%INGRESS%/$(oc get ingresses.config/cluster -o jsonpath={.spec.domain})/g |sed s/OCPSERVER/$(oc whoami --show-server=true|cut -f3 -d "/")/g | sed s/OCPTOKEN/$(oc whoami --show-token=true)/g | oc apply -n ${ODHPROJECT} -f -"

    # Wait for the nodebook-server to be ready
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} | grep "jupyter-nb-kube-3aadmin" | awk '{print \$2}'" "2/2" $odhdefaulttimeout $odhdefaultinterval

    # Wait for the mnisttest appwrapper state to become running
    os::cmd::try_until_text "oc get appwrapper mnisttest -n ${ODHPROJECT} -ojsonpath='{.status.state}'" "Running" $odhdefaulttimeout $odhdefaultinterval

    # Wait for Raycluster to be ready
    os::cmd::try_until_text "oc get raycluster -n ${ODHPROJECT} mnisttest -ojsonpath='{.status.state}'" "ready" $odhdefaulttimeout $odhdefaultinterval
}

function uninstall_distributed_workloads_kfdef() {
    header "Uninstalling distributed workloads kfdef"
    os::cmd::expect_success "oc delete kfdef codeflare-stack -n ${ODHPROJECT}"

    # Ensure that MCAD, Instascale, KubeRay pods are gone
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} -l app=mcad-mcad" "No resources found in ${ODHPROJECT} namespace."
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} -l app=instascale-instascale" "No resources found in ${ODHPROJECT} namespace."
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} -l app.kubernetes.io/component=kuberay-operator" "No resources found in ${ODHPROJECT} namespace."

    # Ensure the codeflare-notebook imagestream is removed
    os::cmd::expect_failure "oc get imagestreams -n ${ODHPROJECT} codeflare-notebook"
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

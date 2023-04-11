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
    os::cmd::expect_success "oc apply -f $MY_DIR/../../codeflare-stack-kfdef.yaml -n ${ODHPROJECT}"

    # Ensure that MCAD, Instascale, KubeRay pods start
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} |grep mcad-controller | awk '{print \$2}'"  "1/1" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} |grep instascale-instascale | awk '{print \$2}'"  "1/1" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} |grep kuberay-operator | awk '{print \$2}'"  "1/1" $odhdefaulttimeout $odhdefaultinterval

    # Ensure the codeflare-notebook imagestream is there
    os::cmd::expect_success_and_text "oc get imagestreams -n ${ODHPROJECT} codeflare-notebook --no-headers=true |awk '{print \$1}'"  "codeflare-notebook"
}

function test_mcad_torchx_functionality() {
    header "Testing MCAD TorchX Functionality"

    ########### Clean Cluster should be free of these resources ############
    # Get appwrapper name
    AW=$(oc get appwrapper -n ${ODHPROJECT} | grep mnistjob | cut -d ' ' -f 1) || true
    # Clean up resources
    if [[ -n $AW ]]; then
        os::cmd::expect_success "oc delete appwrapper $AW -n ${ODHPROJECT} || true"
    fi
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete cm notebooks-mcad -n ${ODHPROJECT} || true"
    ##############################################################################

    # Wait for the notebook controller ready
    os::cmd::try_until_text "oc get deployment odh-notebook-controller-manager -n ${ODHPROJECT} --no-headers=true | awk '{print \$2}'" "1/1" $odhdefaulttimeout $odhdefaultinterval

    # Create a mnist_ray_mini.ipynb as a configMap
    os::cmd::expect_success "oc create configmap notebooks-mcad -n ${ODHPROJECT} --from-file=${RESOURCEDIR}/mnist_mcad_mini.ipynb"

    # Spawn notebook-server using the codeflare custom nb image
    os::cmd::expect_success "cat ${RESOURCEDIR}/custom-nb-small-mcad.yaml \
                            | sed s/%INGRESS%/$(oc get ingresses.config/cluster -o jsonpath={.spec.domain})/g \
                            | sed s/OCPSERVER/$(oc whoami --show-server=true|cut -f3 -d "/")/g \
                            | sed s/OCPTOKEN/$(oc whoami --show-token=true)/g | oc apply -n ${ODHPROJECT} -f -"

    # Wait for the notebook-server to be ready
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} | grep "jupyter-nb-kube-3aadmin" | awk '{print \$2}'" "2/2" $odhdefaulttimeout $odhdefaultinterval

    # Wait for appwrapper to exist
    os::cmd::try_until_text "oc get appwrapper -n ${ODHPROJECT} | grep mnistjob" "mnistjob-*" $odhdefaulttimeout $odhdefaultinterval

    # Get appwrapper name
    AW=$(oc get appwrapper -n ${ODHPROJECT} | grep mnistjob | cut -d ' ' -f 1)
    
    # Wait for the mnisttest appwrapper state to become running
    os::cmd::try_until_text "oc get appwrapper $AW -n ${ODHPROJECT} -ojsonpath='{.status.state}'" "Running" $odhdefaulttimeout $odhdefaultinterval

    # Wait for workload to succeed and clean up
    os::cmd::try_until_text "oc get appwrapper $AW -n ${ODHPROJECT}" "*NotFound*" $odhdefaulttimeout $odhdefaultinterval

    # Test clean up resources
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT}"
    os::cmd::expect_failure "oc get notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete cm notebooks-mcad -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get cm notebooks-mcad -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete appwrapper $AW -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get appwrapper $AW -n ${ODHPROJECT}"
}

function test_mcad_ray_functionality() {
    header "Testing MCAD Ray Functionality"

    ########### ToDo: Clean Cluster should be free of those resources ############
    # Clean up resources
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete cm notebooks-ray -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete appwrapper mnisttest -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete raycluster mnisttest -n ${ODHPROJECT} || true"
    ##############################################################################

    # Wait for the notebook controller ready
    os::cmd::try_until_text "oc get deployment odh-notebook-controller-manager -n ${ODHPROJECT} --no-headers=true | awk '{print \$2}'" "1/1" $odhdefaulttimeout $odhdefaultinterval

    # Create a mnist_ray_mini.ipynb as a configMap
    os::cmd::expect_success "oc create configmap notebooks-ray -n ${ODHPROJECT} --from-file=${RESOURCEDIR}/mnist_ray_mini.ipynb --from-file=${RESOURCEDIR}/mnist.py --from-file=${RESOURCEDIR}/requirements.txt"

    # Spawn notebook-server using the codeflare custom nb image
    os::cmd::expect_success "cat ${RESOURCEDIR}/custom-nb-small-ray.yaml \
                            | sed s/%INGRESS%/$(oc get ingresses.config/cluster -o jsonpath={.spec.domain})/g \
                            | sed s/OCPSERVER/$(oc whoami --show-server=true|cut -f3 -d "/")/g \
                            | sed s/OCPTOKEN/$(oc whoami --show-token=true)/g | oc apply -n ${ODHPROJECT} -f -"

    # Wait for the notebook-server to be ready
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} | grep "jupyter-nb-kube-3aadmin" | awk '{print \$2}'" "2/2" $odhdefaulttimeout $odhdefaultinterval

    # Wait for the mnisttest appwrapper state to become running
    os::cmd::try_until_text "oc get appwrapper mnisttest -n ${ODHPROJECT} -ojsonpath='{.status.state}'" "Running" $odhdefaulttimeout $odhdefaultinterval

    # Wait for Raycluster to be ready
    os::cmd::try_until_text "oc get raycluster -n ${ODHPROJECT} mnisttest -ojsonpath='{.status.state}'" "ready" $odhdefaulttimeout $odhdefaultinterval

    # Wait for job to be completed and cleaned up
    os::cmd::try_until_text "oc get appwrapper mnisttest -n ${ODHPROJECT}" "*NotFound*" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::expect_failure "oc get raycluster mnisttest -n ${ODHPROJECT}"

    # Test clean up resources
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT}"
    os::cmd::expect_failure "oc get notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete cm notebooks-ray -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get cm notebooks-ray -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete appwrapper mnisttest -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get appwrapper mnisttest -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete raycluster mnisttest -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get raycluster mnisttest -n ${ODHPROJECT}"

}

function uninstall_distributed_workloads_kfdef() {
    header "Uninstalling distributed workloads kfdef"
    echo "NOTE, kfdef deletion can take up to 5-8 minutes..."
    os::cmd::try_until_success "oc delete kfdef codeflare-stack -n ${ODHPROJECT}" $odhdefaulttimeout $odhdefaultinterval

    # Ensure that MCAD, Instascale, KubeRay pods are gone
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} -l app=mcad-mcad" "No resources found in ${ODHPROJECT} namespace." $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} -l app=instascale-instascale" "No resources found in ${ODHPROJECT} namespace." $odhdefaulttimeout $odhdefaultinterval
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} -l app.kubernetes.io/component=kuberay-operator" "No resources found in ${ODHPROJECT} namespace." $odhdefaulttimeout $odhdefaultinterval

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


example_test
install_codeflare_operator
install_distributed_workloads_kfdef
test_mcad_torchx_functionality
test_mcad_ray_functionality
uninstall_distributed_workloads_kfdef
uninstall_codeflare_operator


os::test::junit::declare_suite_end

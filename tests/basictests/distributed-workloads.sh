#!/bin/bash

source $TEST_DIR/common

MY_DIR=$(readlink -f `dirname "${BASH_SOURCE[0]}"`)

RESOURCEDIR="${MY_DIR}/../resources"

source ${MY_DIR}/../util

TEST_USER=${OPENSHIFT_TESTUSER_NAME:-"admin"}
TEST_PASS=${OPENSHIFT_TESTUSER_PASS:-"admin"}
OPENSHIFT_OAUTH_ENDPOINT="https://$(oc get route -n openshift-authentication   oauth-openshift -o json | jq -r '.spec.host')"

os::test::junit::declare_suite_start "$MY_SCRIPT"

function check_distributed_workloads_kfdef(){
    header "Checking distributed workloads stack"

    # Ensure that KubeRay pods start
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} |grep kuberay-operator | awk '{print \$2}'"  "1/1" $odhdefaulttimeout $odhdefaultinterval

    # Ensure the codeflare-notebook imagestream is there
    os::cmd::expect_success_and_text "oc get imagestreams -n ${ODHPROJECT} codeflare-notebook --no-headers=true |awk '{print \$1}'"  "codeflare-notebook"

    # Add additional role required by notebook sa
    oc adm policy add-role-to-user admin -n ${ODHPROJECT} --rolebinding-name "admin-$TEST_USER" $TEST_USER
    oc adm policy add-role-to-user kuberay-operator -n ${ODHPROJECT} --rolebinding-name "kuberay-operator-$TEST_USER" $TEST_USER
}

function test_mcad_torchx_functionality() {
    header "Testing MCAD TorchX Functionality"

    ########### Clean Cluster should be free of these resources ############
    # Get appwrapper name
    AW=$(oc get appwrapper.workload.codeflare.dev -n ${ODHPROJECT} | grep mnistjob | cut -d ' ' -f 1) || true
    # Clean up resources
    if [[ -n $AW ]]; then
        os::cmd::expect_success "oc delete appwrapper.workload.codeflare.dev $AW -n ${ODHPROJECT} || true"
    fi
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete cm notebooks-mcad -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete pvc jupyterhub-nb-kube-3aadmin-pvc -n ${ODHPROJECT} || true"
    ##############################################################################

    # Wait for the notebook controller ready
    os::cmd::try_until_text "oc get deployment odh-notebook-controller-manager -n ${ODHPROJECT} --no-headers=true | awk '{print \$2}'" "1/1" $odhdefaulttimeout $odhdefaultinterval

    # Create a mnist_ray_mini.ipynb as a configMap
    os::cmd::expect_success "oc create configmap notebooks-mcad -n ${ODHPROJECT} --from-file=${RESOURCEDIR}/mnist_mcad_mini.ipynb"

    # Get Token
    local TESTUSER_BEARER_TOKEN="$(curl -skiL -u $TEST_USER:$TEST_PASS -H 'X-CSRF-Token: xxx' "$OPENSHIFT_OAUTH_ENDPOINT/oauth/authorize?response_type=token&client_id=openshift-challenging-client" | grep -oP 'access_token=\K[^&]*')"

    # Spawn notebook-server using the codeflare custom nb image
    os::cmd::expect_success "cat ${RESOURCEDIR}/custom-nb-small.yaml \
                            | sed s/%INGRESS%/$(oc get ingresses.config/cluster -o jsonpath={.spec.domain})/g \
                            | sed s/%OCPSERVER%/$(oc whoami --show-server=true|cut -f3 -d "/")/g \
                            | sed s/%OCPTOKEN%/${TESTUSER_BEARER_TOKEN}/g \
                            | sed s/%NAMESPACE%/${ODHPROJECT}/g \
                            | sed s/%JOBTYPE%/mcad/g | oc apply -n ${ODHPROJECT} -f -"

    # Wait for the notebook-server to be ready
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} | grep "jupyter-nb-kube-3aadmin" | awk '{print \$2}'" "2/2" $odhdefaulttimeout $odhdefaultinterval

    # Wait for appwrapper to exist
    os::cmd::try_until_text "oc get appwrapper.workload.codeflare.dev -n ${ODHPROJECT} | grep mnistjob" "mnistjob-*" $odhdefaulttimeout $odhdefaultinterval

    # Get appwrapper name
    AW=$(oc get appwrapper.workload.codeflare.dev -n ${ODHPROJECT} | grep mnistjob | cut -d ' ' -f 1)

    # Wait for the mnisttest appwrapper state to become running
    os::cmd::try_until_text "oc get appwrapper.workload.codeflare.dev $AW -n ${ODHPROJECT} -ojsonpath='{.status.state}'" "Running" $odhdefaulttimeout $odhdefaultinterval

    # Wait for workload to succeed and clean up
    os::cmd::try_until_text "oc get appwrapper.workload.codeflare.dev $AW -n ${ODHPROJECT}" "*NotFound*" $odhdefaulttimeout $odhdefaultinterval

    # Test clean up resources
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT}"
    os::cmd::expect_failure "oc get notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete cm notebooks-mcad -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get cm notebooks-mcad -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete appwrapper.workload.codeflare.dev $AW -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get appwrapper.workload.codeflare.dev $AW -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete pvc jupyterhub-nb-kube-3aadmin-pvc -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get pvc jupyterhub-nb-kube-3aadmin-pvc -n ${ODHPROJECT}"
}

function test_mcad_ray_functionality() {
    header "Testing MCAD Ray Functionality"

    ########### ToDo: Clean Cluster should be free of those resources ############
    # Clean up resources
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete cm notebooks-ray -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete appwrapper.workload.codeflare.dev mnisttest -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete raycluster mnisttest -n ${ODHPROJECT} || true"
    os::cmd::expect_success "oc delete pvc jupyterhub-nb-kube-3aadmin-pvc -n ${ODHPROJECT} || true"
    ##############################################################################

    # Wait for the notebook controller ready
    os::cmd::try_until_text "oc get deployment odh-notebook-controller-manager -n ${ODHPROJECT} --no-headers=true | awk '{print \$2}'" "1/1" $odhdefaulttimeout $odhdefaultinterval

    # Create a mnist_ray_mini.ipynb as a configMap
    os::cmd::expect_success "oc create configmap notebooks-ray -n ${ODHPROJECT} --from-file=${RESOURCEDIR}/mnist_ray_mini.ipynb --from-file=${RESOURCEDIR}/mnist.py --from-file=${RESOURCEDIR}/requirements.txt"

    # Get Token
    local TESTUSER_BEARER_TOKEN="$(curl -skiL -u $TEST_USER:$TEST_PASS -H 'X-CSRF-Token: xxx' "$OPENSHIFT_OAUTH_ENDPOINT/oauth/authorize?response_type=token&client_id=openshift-challenging-client" | grep -oP 'access_token=\K[^&]*')"

    # Spawn notebook-server using the codeflare custom nb image
    os::cmd::expect_success "cat ${RESOURCEDIR}/custom-nb-small.yaml \
                            | sed s/%INGRESS%/$(oc get ingresses.config/cluster -o jsonpath={.spec.domain})/g \
                            | sed s/%OCPSERVER%/$(oc whoami --show-server=true|cut -f3 -d "/")/g \
                            | sed s/%OCPTOKEN%/${TESTUSER_BEARER_TOKEN}/g \
                            | sed s/%NAMESPACE%/${ODHPROJECT}/g \
                            | sed s/%JOBTYPE%/ray/g | oc apply -n ${ODHPROJECT} -f -"

    # Wait for the notebook-server to be ready
    os::cmd::try_until_text "oc get pod -n ${ODHPROJECT} | grep "jupyter-nb-kube-3aadmin" | awk '{print \$2}'" "2/2" $odhdefaulttimeout $odhdefaultinterval

    # Wait for the mnisttest appwrapper state to become running
    os::cmd::try_until_text "oc get appwrapper.workload.codeflare.dev mnisttest -n ${ODHPROJECT} -ojsonpath='{.status.state}'" "Running" $odhdefaulttimeout $odhdefaultinterval

    # Wait for Raycluster to be ready
    os::cmd::try_until_text "oc get raycluster -n ${ODHPROJECT} mnisttest -ojsonpath='{.status.state}'" "ready" $odhdefaulttimeout $odhdefaultinterval

    # Wait for job to be completed and cleaned up
    os::cmd::try_until_text "oc get appwrapper.workload.codeflare.dev mnisttest -n ${ODHPROJECT}" "*NotFound*" $odhdefaulttimeout $odhdefaultinterval
    os::cmd::expect_failure "oc get raycluster mnisttest -n ${ODHPROJECT}"

    # Test clean up resources
    os::cmd::expect_success "oc delete notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT}"
    os::cmd::expect_failure "oc get notebook jupyter-nb-kube-3aadmin -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete cm notebooks-ray -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get cm notebooks-ray -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete appwrapper.workload.codeflare.dev mnisttest -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get appwrapper.workload.codeflare.dev mnisttest -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete raycluster mnisttest -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get raycluster mnisttest -n ${ODHPROJECT}"

    os::cmd::expect_success "oc delete pvc jupyterhub-nb-kube-3aadmin-pvc -n ${ODHPROJECT} || true"
    os::cmd::expect_failure "oc get pvc jupyterhub-nb-kube-3aadmin-pvc -n ${ODHPROJECT}"

}

function clean_permissions() {
    header "Cleaning extra admin roles"
    oc adm policy remove-role-from-user admin -n ${ODHPROJECT} $TEST_USER
    oc adm policy remove-role-from-user kuberay-operator -n ${ODHPROJECT} $TEST_USER
}


check_distributed_workloads_kfdef
test_mcad_torchx_functionality
test_mcad_ray_functionality
clean_permissions


os::test::junit::declare_suite_end

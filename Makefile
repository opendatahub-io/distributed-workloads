export PATH:=$(GOBIN):$(PATH)

SHELL := /bin/bash
OCI_RUNTIME ?= $(shell which podman  || which docker)
CMD_DIR=./cmd/
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

## Tool Binaries
KUSTOMIZE ?= $(LOCALBIN)/kustomize

## Tool Versions
KUSTOMIZE_VERSION ?= v4.5.4

.DEFAULT_GOAL := help

FORCE: ;

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ all-in-one
.PHONY: all-in-one # All-In-One
all-in-one: ## Install distributed AI platform  
	@echo -e "\n==> Installing Everything needed for distributed AI platform on OpenShift cluster \n"
	-make delete-codeflare delete-ndf-operator delete-nvidia-operator delete-codeflare-operator delete-opendatahub-operator
	make install-opendatahub-operator install-codeflare-operator install-ndf-operator install-nvidia-operator deploy-codeflare
	make opendatahub-dashboard
	@echo -e "\n==> Done (Deploy everything)\n" 

.PHONY: opendatahub-dashboard
opendatahub-dashboard:
	echo -e "\n\n =====>>>>> Waiting for OpenDataHub dashboard to be ready\n";
	@while [[ -z $$(oc rollout -n opendatahub status deployment odh-dashboard --timeout=600s) ]]; do echo "."; sleep 5; done
	@export URL=`oc get route odh-dashboard -n opendatahub -o jsonpath="http://{.spec.host}"`; \
	echo -e "\n\n =====>>>>> Access OpenDataHub using: $$URL\n";

##@ general
.PHONY: install-opendatahub-operator
install-opendatahub-operator: ## Install OpenDataHub operator
	@echo -e "\n==> Installing OpenDataHub Operator \n"
	-oc create ns opendatahub
	oc create -f contrib/configuration/opendatahub-operator-subscription.yaml
	@echo Waiting for opendatahub-operator Subscription to be ready
	oc wait -n openshift-operators subscription/opendatahub-operator --for=jsonpath='{.status.state}'=AtLatestKnown --timeout=180s

.PHONY: delete-opendatahub-operator
delete-opendatahub-operator: ## Delete OpenDataHub operator
	@echo -e "\n==> Deleting OpenDataHub Operator \n"
	-oc delete subscription opendatahub-operator -n openshift-operators
	-export CLUSTER_SERVICE_VERSION=`oc get clusterserviceversion -n openshift-operators -l operators.coreos.com/opendatahub-operator.openshift-operators -o custom-columns=:metadata.name`; \
	oc delete clusterserviceversion $$CLUSTER_SERVICE_VERSION -n openshift-operators

.PHONY: install-codeflare-operator
install-codeflare-operator: ## Install CodeFlare operator
	@echo -e "\n==> Installing CodeFlare Operator \n"
	oc create -f contrib/configuration/codeflare-operator-subscription.yaml
	@echo Waiting for codeflare-operator Subscription to be ready
	oc wait -n openshift-operators subscription/codeflare-operator --for=jsonpath='{.status.state}'=AtLatestKnown --timeout=180s

.PHONY: delete-codeflare-operator
delete-codeflare-operator: ## Delete CodeFlare operator
	@echo -e "\n==> Deleting CodeFlare Operator \n"
	-oc delete subscription codeflare-operator -n openshift-operators
	-export CLUSTER_SERVICE_VERSION=`oc get clusterserviceversion -n openshift-operators -l operators.coreos.com/codeflare-operator.openshift-operators -o custom-columns=:metadata.name`; \
	oc delete clusterserviceversion $$CLUSTER_SERVICE_VERSION -n openshift-operators

##@ CodeFlare

.PHONY: deploy-codeflare
deploy-codeflare: ## Deploy CodeFlare
	@echo -e "\n==> Deploying CodeFlare \n"
	-oc create ns opendatahub
	@while [[ -z $$(oc get customresourcedefinition kfdefs.kfdef.apps.kubeflow.org) ]]; do echo "."; sleep 10; done
	oc apply -f https://raw.githubusercontent.com/opendatahub-io/odh-manifests/master/kfdef/odh-core.yaml -n opendatahub
	oc apply -f https://raw.githubusercontent.com/opendatahub-io/distributed-workloads/main/codeflare-stack-kfdef.yaml -n opendatahub

.PHONY: delete-codeflare
delete-codeflare: ## Delete CodeFlare
	@echo -e "\n==> Deleting CodeFlare \n"
	-oc delete -f https://raw.githubusercontent.com/opendatahub-io/odh-manifests/master/kfdef/odh-core.yaml -n opendatahub
	-oc delete -f https://raw.githubusercontent.com/opendatahub-io/distributed-workloads/main/codeflare-stack-kfdef.yaml -n opendatahub
	-oc delete ns opendatahub

.PHONY: deploy-codeflare-from-filesystem
deploy-codeflare-from-filesystem: kustomize ## Deploy CodeFlare from local file system
	@echo -e "\n==> Deploying CodeFlare \n"
	-oc create ns opendatahub
	@while [[ -z $$(oc get customresourcedefinition mcads.codeflare.codeflare.dev) ]]; do echo "."; sleep 10; done
	$(KUSTOMIZE) build ray/operator/base | oc apply --server-side=true -n opendatahub -f -
	$(KUSTOMIZE) build codeflare-stack/base | oc apply --server-side=true -n opendatahub -f -

.PHONY: delete-codeflare-from-filesystem
delete-codeflare-from-filesystem: kustomize ## Delete CodeFlare deployed from local file system
	@echo -e "\n==> Deleting CodeFlare \n"
	-$(KUSTOMIZE) build ray/operator/base | oc delete -n opendatahub -f -
	-$(KUSTOMIZE) build codeflare-stack/base | oc delete -n opendatahub -f -
	-oc delete ns opendatahub

##@ GPU Support

.PHONY: install-ndf-operator
install-ndf-operator: ## Install NDF operator ( Node Feature Discovery )
	@echo -e "\n==> Installing NDF Operator \n"
	-oc create ns openshift-nfd
	oc create -f contrib/configuration/ndf-operator-subscription.yaml

.PHONY: delete-ndf-operator
delete-ndf-operator: ## Delete NDF operator
	@echo -e "\n==> Deleting NDF Operator \n"
	-oc delete subscription nfd -n openshift-nfd
	-export CLUSTER_SERVICE_VERSION=`oc get clusterserviceversion -n openshift-nfd -l operators.coreos.com/nfd.openshift-nfd -o custom-columns=:metadata.name`; \
	oc delete clusterserviceversion $$CLUSTER_SERVICE_VERSION -n openshift-nfd
	-oc delete ns openshift-nfd

.PHONY: install-nvidia-operator
install-nvidia-operator: ## Install nvidia operator
	@echo -e "\n==> Installing nvidia Operator \n"
	-oc create ns nvidia-gpu-operator
	oc create -f contrib/configuration/nvidia-operator-subscription.yaml

.PHONY: delete-nvidia-operator
delete-nvidia-operator: ## Delete nvidia operator
	@echo -e "\n==> Deleting nvidia Operator \n"
	-oc delete subscription gpu-operator-certified -n nvidia-gpu-operator
	-export CLUSTER_SERVICE_VERSION=`oc get clusterserviceversion -n nvidia-gpu-operator -l operators.coreos.com/gpu-operator-certified.nvidia-gpu-operator -o custom-columns=:metadata.name`; \
	oc delete clusterserviceversion $$CLUSTER_SERVICE_VERSION -n nvidia-gpu-operator
	-oc delete ns nvidia-gpu-operator

##@ Tool installations

KUSTOMIZE_INSTALL_SCRIPT ?= "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh"
.PHONY: kustomize
kustomize: $(KUSTOMIZE) ## Download kustomize locally if necessary.
$(KUSTOMIZE): $(LOCALBIN)
	test -s $(LOCALBIN)/kustomize || { curl -s $(KUSTOMIZE_INSTALL_SCRIPT) | bash -s -- $(subst v,,$(KUSTOMIZE_VERSION)) $(LOCALBIN); }

include .mk/observability.mk

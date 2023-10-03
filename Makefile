export PATH:=$(GOBIN):$(PATH)

SHELL := /bin/bash
OCI_RUNTIME ?= $(shell which podman  || which docker)
CMD_DIR=./cmd/
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)
TMPDIR ?= /tmp

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
	-make delete-codeflare delete-nfd-operator delete-nvidia-operator delete-codeflare-operator delete-opendatahub-operator
	make install-opendatahub-operator install-codeflare-operator install-nfd-operator install-nvidia-operator deploy-codeflare
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

.PHONY: install-codeflare-operator-from-github
install-codeflare-operator-from-github: ## Install CodeFlare operator from main branch on GitHub
	@echo -e "\n==> Installing CodeFlare Operator from main branch on GitHub \n"
	test -d $(TMPDIR)/codeflare || git clone https://github.com/project-codeflare/codeflare-operator.git $(TMPDIR)/codeflare
	VERSION=dev make deploy -C $(TMPDIR)/codeflare

.PHONY: delete-codeflare-operator-from-github
delete-codeflare-operator-from-github: ## Delete CodeFlare operator from main branch on GitHub
	@echo -e "\n==> Deleting CodeFlare Operator from main branch on GitHub \n"
	test -d $(TMPDIR)/codeflare || git clone https://github.com/project-codeflare/codeflare-operator.git $(TMPDIR)/codeflare
	make undeploy -C $(TMPDIR)/codeflare

##@ CodeFlare

.PHONY: deploy-codeflare
deploy-codeflare: ## Deploy CodeFlare
	@echo -e "\n==> Deploying CodeFlare \n"
	-oc create ns opendatahub
	@while [[ -z $$(oc get customresourcedefinition datascienceclusters.datasciencecluster.opendatahub.io) ]]; do echo "."; sleep 10; done
	oc apply -f https://raw.githubusercontent.com/opendatahub-io/distributed-workloads/main/codeflare-dsc.yaml -n opendatahub

.PHONY: delete-codeflare
delete-codeflare: ## Delete CodeFlare
	@echo -e "\n==> Deleting CodeFlare \n"
	-oc delete -f https://raw.githubusercontent.com/opendatahub-io/distributed-workloads/main/codeflare-dsc.yaml -n opendatahub
	-oc delete ns opendatahub

.PHONY: deploy-codeflare-from-filesystem
deploy-codeflare-from-filesystem: kustomize ## Deploy CodeFlare from local file system
	@echo -e "\n==> Deploying CodeFlare \n"
	-oc create ns opendatahub
	$(KUSTOMIZE) build ray/operator/base | oc apply --server-side=true -n opendatahub -f -
	$(KUSTOMIZE) build codeflare-stack/base | oc apply --server-side=true -n opendatahub -f -

.PHONY: delete-codeflare-from-filesystem
delete-codeflare-from-filesystem: kustomize ## Delete CodeFlare deployed from local file system
	@echo -e "\n==> Deleting CodeFlare \n"
	-$(KUSTOMIZE) build ray/operator/base | oc delete -n opendatahub -f -
	-$(KUSTOMIZE) build codeflare-stack/base | oc delete -n opendatahub -f -
	-oc delete ns opendatahub

##@ GPU Support

.PHONY: install-nfd-operator
install-nfd-operator: ## Install NFD operator ( Node Feature Discovery )
	@echo -e "\n==> Installing NFD Operator \n"
	-oc create ns openshift-nfd
	oc create -f contrib/configuration/nfd-operator-subscription.yaml
	@echo -e "\n==> Creating default NodeFeatureDiscovery CR \n"
	@while [[ -z $$(oc get customresourcedefinition nodefeaturediscoveries.nfd.openshift.io) ]]; do echo "."; sleep 10; done
	@while [[ -z $$(oc get csv -n openshift-nfd --selector operators.coreos.com/nfd.openshift-nfd) ]]; do echo "."; sleep 10; done
	oc get csv -n openshift-nfd --selector operators.coreos.com/nfd.openshift-nfd -ojsonpath={.items[0].metadata.annotations.alm-examples} | jq '.[] | select(.kind=="NodeFeatureDiscovery")' | oc apply -f -

.PHONY: delete-nfd-operator
delete-nfd-operator: ## Delete NFD operator
	@echo -e "\n==> Deleting NodeFeatureDiscovery CR \n"
	oc delete NodeFeatureDiscovery --all -n openshift-nfd
	@while [[ -n $$(oc get NodeFeatureDiscovery -n openshift-nfd) ]]; do echo "."; sleep 10; done
	@echo -e "\n==> Deleting NFD Operator \n"
	-oc delete subscription nfd -n openshift-nfd
	-export CLUSTER_SERVICE_VERSION=`oc get clusterserviceversion -n openshift-nfd -l operators.coreos.com/nfd.openshift-nfd -o custom-columns=:metadata.name`; \
	oc delete clusterserviceversion $$CLUSTER_SERVICE_VERSION -n openshift-nfd
	-oc delete ns openshift-nfd

.PHONY: install-nvidia-operator
install-nvidia-operator: ## Install nvidia operator
	@echo -e "\n==> Installing nvidia Operator \n"
	-oc create ns nvidia-gpu-operator
	oc create -f contrib/configuration/nvidia-operator-subscription.yaml
	@echo -e "\n==> Creating default ClusterPolicy CR \n"
	@while [[ -z $$(oc get customresourcedefinition clusterpolicies.nvidia.com) ]]; do echo "."; sleep 10; done
	@while [[ -z $$(oc get csv -n nvidia-gpu-operator --selector operators.coreos.com/gpu-operator-certified.nvidia-gpu-operator) ]]; do echo "."; sleep 10; done
	oc get csv -n nvidia-gpu-operator --selector operators.coreos.com/gpu-operator-certified.nvidia-gpu-operator -ojsonpath={.items[0].metadata.annotations.alm-examples} | jq .[] | oc apply -f -

.PHONY: delete-nvidia-operator
delete-nvidia-operator: ## Delete nvidia operator
	@echo -e "\n==> Deleting ClusterPolicy CR \n"
	oc delete ClusterPolicy --all -n nvidia-gpu-operator
	@while [[ -n $$(oc get ClusterPolicy -n nvidia-gpu-operator) ]]; do echo "."; sleep 10; done
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

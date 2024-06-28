BASE:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL=/bin/sh
WORK_DIR=/tmp/openshift-ai-ray
JOB_NAME=git-clone-job
NAMESPACE=distributed

.PHONY: install-openshift-ai add-gpu-machineset setup-kueue-premption setup-ray-distributed-training deplou-odf example

install-openshift-ai:
	add-gpu-operator
	deploy-oai

add-gpu-machineset:
	@mkdir -p $(WORK_DIR)
	@$(BASE)/scripts/add-gpu.sh $(WORK_DIR)	

add-gpu-operator:
	oc apply -f $(BASE)/yaml/operators/nfd.yaml

	@until oc get crd nodefeaturediscoveries.nfd.openshift.io >/dev/null 2>&1; do \
    	echo "Wait until CRD nodefeaturediscoveries.nfd.openshift.io is ready..."; \
	done

	oc apply -f $(BASE)/yaml/operators/nfd-cr.yaml
	oc apply -f $(BASE)/yaml/operators/nvidia.yaml

	@until oc get crd clusterpolicies.nvidia.com>/dev/null 2>&1; do \
    	echo "Wait until CRD clusterpolicies.nvidia.com is ready..."; \
	done

	oc apply -f $(BASE)/yaml/operators/nvidia-cluster-policy.yaml

deploy-odf:
	@node_count=$$(oc get nodes -o json | jq '[.items[] | select(.spec.taints | not)] | length'); \
	if [ $$node_count -lt 3 ]; then \
		echo "There are $$node_count worker nodes without Nvidia taint."; \
		exit 1; \
	fi 

	oc apply -f $(BASE)/yaml/operators/odf-storagecluster.yaml
	oc apply -f $(BASE)/yaml/operators/odf-storagesystem.yaml
	
	@until oc get storagesystem ocs-storagecluster-storagesystem -n openshift-storage -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' | grep -q "True"; do \
		echo "Waiting for ODF to be ready..."; \
		sleep 30; \
	done
	@echo "ODF is ready"

deploy-oai:
	oc apply -f $(BASE)/yaml/operators/serverless.yaml
	oc apply -f $(BASE)/yaml/operators/servicemesh.yaml
	oc apply -f $(BASE)/yaml/operators/oai.yaml
	
setup-ray-distributed-training:
	@$(BASE)/scripts/clean-kueue.sh

	-oc delete -f $(BASE)/yaml/distributed/git-clone.yaml
	-oc delete -f $(BASE)/yaml/distributed/setup-s3.yaml
	-oc delete -f $(BASE)/yaml/distributed/workbench.yaml
	-oc delete -f $(BASE)/yaml/distributed/cephfs-pvc.yaml
	-oc delete -f $(BASE)/yaml/distributed/serving-runtime-template.yaml
	-oc delete -f $(BASE)/yaml/distributed/ns.yaml

	oc create -f $(BASE)/yaml/distributed/ns.yaml
	oc create -f $(BASE)/yaml/distributed/rolebinding.yaml
	
	oc create -f $(BASE)/yaml/distributed/minio.yaml

	@until oc get statefulset minio -n $(NAMESPACE) -o jsonpath='{.status.readyReplicas}' | grep -q '1'; do \
		echo "Waiting for StatefulSet minio to have 1 ready replica..."; \
		sleep 10; \
	done
	@echo "StatefulSet minio has 1 ready replica."

	@AWS_ACCESS_KEY_ID=$$(oc extract secret/minio  --to=- --keys=MINIO_ROOT_USER -n $(NAMESPACE) 2>/dev/null | tr -d '\n' | base64 ) \
     AWS_SECRET_ACCESS_KEY=$$(oc extract secret/minio  --to=- --keys=MINIO_ROOT_PASSWORD -n $(NAMESPACE) 2>/dev/null | tr -d '\n' | base64) \
     AWS_S3_ENDPOINT=$$(oc get route minio -n distributed -o jsonpath='{.spec.host}') \
	 envsubst < $(BASE)/yaml/distributed/data-connection.yaml.tmpl | oc create -f -
	@$(BASE)/scripts/run-job.sh $(BASE)/yaml/distributed/setup-s3.yaml $(NAMESPACE) setup-s3-job

	oc create -f $(BASE)/yaml/distributed/default-flavor.yaml -f $(BASE)/yaml/distributed/gpu-flavor.yaml
	oc create -f $(BASE)/yaml/distributed/cluster-queue.yaml
	oc create -f $(BASE)/yaml/distributed/local-queue.yaml
	oc create -f $(BASE)/yaml/distributed/cephfs-pvc.yaml	
	@$(BASE)/scripts/run-job.sh $(BASE)/yaml/distributed/git-clone.yaml $(NAMESPACE) git-clone-job
	oc create -f $(BASE)/yaml/distributed/workbench.yaml
	oc create -f $(BASE)/yaml/distributed/serving-runtime-template.yaml
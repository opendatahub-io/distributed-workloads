# Image tag for image containing e2e tests
E2E_TEST_IMAGE_VERSION ?= latest
E2E_TEST_IMAGE ?= quay.io/opendatahub/distributed-workloads-tests:${E2E_TEST_IMAGE_VERSION}

.PHONY: setup-kueue
setup-kueue: ## Set up Kueue for e2e tests.
	echo "Installing Kueue into the cluster"
	kubectl create namespace opendatahub --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply --server-side -k "github.com/opendatahub-io/kueue/config/rhoai"
	echo "Wait for Kueue deployment"
	kubectl -n opendatahub wait --timeout=300s --for=condition=Available deployments --all

.PHONY: setup-kfto
setup-kfto: ## Set up Training operator for e2e tests.
	echo "Deploying Training operator"
	kubectl create namespace opendatahub --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -k "github.com/opendatahub-io/training-operator/manifests/rhoai"
	echo "Wait for Training operator deployment"
	kubectl -n opendatahub wait --timeout=300s --for=condition=Available deployments --all

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

OPENSHIFT-GOIMPORTS ?= $(LOCALBIN)/openshift-goimports

.PHONY: openshift-goimports
openshift-goimports: $(OPENSHIFT-GOIMPORTS) ## Download openshift-goimports locally if necessary.
$(OPENSHIFT-GOIMPORTS): $(LOCALBIN)
	test -s $(LOCALBIN)/openshift-goimports || GOBIN=$(LOCALBIN) go install github.com/openshift-eng/openshift-goimports@latest

.PHONY: imports
imports: openshift-goimports ## Organize imports in go files using openshift-goimports. Example: make imports
	$(OPENSHIFT-GOIMPORTS)

.PHONY: verify-imports
verify-imports: openshift-goimports ## Run import verifications.
	./hack/verify-imports.sh $(OPENSHIFT-GOIMPORTS)

.PHONY: build-test-image
build-test-image:
	podman build -f images/tests/Dockerfile -t ${E2E_TEST_IMAGE} .

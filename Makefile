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

.PHONY: push-test-image
push-test-image:
	podman push ${E2E_TEST_IMAGE}

# OSU MPI Benchmark images
OSU_BENCHMARK_VERSION ?= latest
OSU_BENCHMARK_IMAGE ?= quay.io/modh/distributed-workloads-benchmark:trainer-mpi-osu-${OSU_BENCHMARK_VERSION}
OSU_BENCHMARK_CUDA_IMAGE ?= quay.io/modh/distributed-workloads-benchmark:trainer-mpi-osu-cuda-${OSU_BENCHMARK_VERSION}

.PHONY: build-osu-benchmark-image
build-osu-benchmark-image:
	podman build -f benchmarks/osu-benchmarks/Dockerfile -t "${OSU_BENCHMARK_IMAGE}" benchmarks/osu-benchmarks/

.PHONY: push-osu-benchmark-image
push-osu-benchmark-image:
	podman push "${OSU_BENCHMARK_IMAGE}"

.PHONY: build-osu-benchmark-cuda-image
build-osu-benchmark-cuda-image:
	podman build -f benchmarks/osu-benchmarks/Dockerfile.cuda -t "${OSU_BENCHMARK_CUDA_IMAGE}" benchmarks/osu-benchmarks/

.PHONY: push-osu-benchmark-cuda-image
push-osu-benchmark-cuda-image:
	podman push "${OSU_BENCHMARK_CUDA_IMAGE}"

.PHONY: unit-test
unit-test: ## Run unit tests for support packages.
	go test ./tests/common/support/...
GOLANGCI_LINT_VERSION ?= v2.12.1
LINT_PKG ?= ./...
GOLANGCI_LINT ?= $(LOCALBIN)/golangci-lint

.PHONY: golangci-lint-install
golangci-lint-install: $(LOCALBIN) ## Download golangci-lint locally.
	GOBIN=$(LOCALBIN) go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

.PHONY: golangci-lint
golangci-lint: golangci-lint-install ## Run golangci-lint on the codebase.
	$(GOLANGCI_LINT) run --timeout 5m $(LINT_PKG)

.PHONY: precommit
precommit:
	pre-commit run --all-files

.PHONY: sync-agents-config
sync-agents-config: ## Sync AI agent skills and rules from ai/ to .claude/ and .cursor/
	@uv run ./hack/sync_agents_config.py

.PHONY: verify-agents-config
verify-agents-config: ## Verify AI agent config is in sync with ai/
	@./hack/verify-agents-config.sh

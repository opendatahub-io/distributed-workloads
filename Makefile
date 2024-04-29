
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
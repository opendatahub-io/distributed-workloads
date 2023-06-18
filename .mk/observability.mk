##@ observability

.PHONY: opendatahub-observability
opendatahub-observability: ## Observability endpoints 
	@export URL=`kubectl get route prometheus-portal -n opendatahub -o jsonpath="http://{.spec.host}"`; \
	echo -e "\n ==>> OpenDataHub Infra. Prometheus: $$URL\n";
	@export URL=`kubectl get route odh-model-monitoring -n opendatahub -o jsonpath="http://{.spec.host}"`; \
	echo -e "\n ==>> OpenDataHub Model Prometheus: $$URL\n";


SHELL := /usr/bin/env bash

SDK_TEST_TIMEOUT ?= 60m
SDK_TEST_EXCLUDE_REGEX ?=
SDK_TEST_TIER ?=
ACCELERATOR_TESTS ?= CUDA
NOTEBOOK_ACCELERATOR ?=
SKIP_S3_TESTS ?= false
SDK_TEST_S3_EXCLUDE_REGEX ?= ^TestRhaiS3

# Source of truth remains tests/trainer/kubeflow_sdk_test.go
SDK_TRAINER_INCLUDE_REGEX ?= ^(TestKubeflowSdk|TestOsftTrainingHub|TestLoraTrainingHub|TestSftTrainingHub|TestRhai|TestTrainingFailureScenarios|TestTorchrunTrainingFailure)
SDK_TRAINER_SANITY_INCLUDE_REGEX ?= ^TestKubeflowSdk

.PHONY: trainer-sdk-test trainer-sdk-test-with-tier trainer-sdk-test-all trainer-sdk-test-cpu trainer-sdk-test-sanity trainer-sdk-test-tier1 trainer-sdk-test-cuda trainer-sdk-test-rocm trainer-sdk-test-name-sanity trainer-sdk-print-selected trainer-sdk-print-tiers

trainer-sdk-test:
	@set -euo pipefail; \
	eval "$$(ACCELERATOR_TESTS='$(ACCELERATOR_TESTS)' NOTEBOOK_ACCELERATOR='$(NOTEBOOK_ACCELERATOR)' ./tests/kubeflow_sdk/scripts/install_kubeflow.sh env)"; \
	EFFECTIVE_EXCLUDE_REGEX="$${SDK_TEST_EXCLUDE_REGEX:-}"; \
	if [[ "$$(echo "$(SKIP_S3_TESTS)" | tr '[:lower:]' '[:upper:]')" == "TRUE" ]]; then \
		if [[ -n "$$EFFECTIVE_EXCLUDE_REGEX" ]]; then \
			EFFECTIVE_EXCLUDE_REGEX="($$EFFECTIVE_EXCLUDE_REGEX)|($(SDK_TEST_S3_EXCLUDE_REGEX))"; \
		else \
			EFFECTIVE_EXCLUDE_REGEX="$(SDK_TEST_S3_EXCLUDE_REGEX)"; \
		fi; \
	fi; \
	TESTS="$$(TEST_TIER="$(SDK_TEST_TIER)" go test ./tests/trainer -list '$(SDK_TRAINER_INCLUDE_REGEX)' | rg '^Test' | (if [[ -n "$$EFFECTIVE_EXCLUDE_REGEX" ]]; then rg -v "$$EFFECTIVE_EXCLUDE_REGEX"; else awk '1'; fi) | paste -sd'|' -)"; \
	if [[ -z "$$TESTS" ]]; then \
		echo "No SDK trainer tests matched include/exclude filters."; \
		exit 1; \
	fi; \
	if [[ "$$(echo "$(SKIP_S3_TESTS)" | tr '[:lower:]' '[:upper:]')" == "TRUE" ]]; then \
		echo "SKIP_S3_TESTS=true active. Excluding tests matching: $(SDK_TEST_S3_EXCLUDE_REGEX)"; \
	fi; \
	echo "Running SDK trainer tests (TEST_TIER='$(SDK_TEST_TIER)'): $$TESTS"; \
	TEST_TIER="$(SDK_TEST_TIER)" go test -timeout "$(SDK_TEST_TIMEOUT)" -count=1 ./tests/trainer -run "^($$TESTS)$$" -v

trainer-sdk-test-with-tier:
	@if [[ -z "$(SDK_TEST_TIER)" ]]; then \
		echo "SDK_TEST_TIER is required (example: SDK_TEST_TIER=Tier1)"; \
		exit 1; \
	fi
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test SDK_TEST_TIER="$(SDK_TEST_TIER)" SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)"

trainer-sdk-test-all:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test SDK_TEST_TIER="" SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)"

trainer-sdk-test-cpu:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-sanity SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)"
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-tier1 SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)"

trainer-sdk-test-sanity:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test SDK_TEST_TIER="Sanity" SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)"

trainer-sdk-test-tier1:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test SDK_TEST_TIER="Tier1" SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)"

trainer-sdk-test-cuda:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test SDK_TEST_TIER="KFTO-CUDA" SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)"

trainer-sdk-test-rocm:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test SDK_TEST_TIER="KFTO-ROCm" SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)"

trainer-sdk-test-name-sanity:
	@set -euo pipefail; \
	eval "$$(ACCELERATOR_TESTS='$(ACCELERATOR_TESTS)' NOTEBOOK_ACCELERATOR='$(NOTEBOOK_ACCELERATOR)' ./tests/kubeflow_sdk/scripts/install_kubeflow.sh env)"; \
	EFFECTIVE_EXCLUDE_REGEX="$${SDK_TEST_EXCLUDE_REGEX:-}"; \
	if [[ "$$(echo "$(SKIP_S3_TESTS)" | tr '[:lower:]' '[:upper:]')" == "TRUE" ]]; then \
		if [[ -n "$$EFFECTIVE_EXCLUDE_REGEX" ]]; then \
			EFFECTIVE_EXCLUDE_REGEX="($$EFFECTIVE_EXCLUDE_REGEX)|($(SDK_TEST_S3_EXCLUDE_REGEX))"; \
		else \
			EFFECTIVE_EXCLUDE_REGEX="$(SDK_TEST_S3_EXCLUDE_REGEX)"; \
		fi; \
	fi; \
	TESTS="$$(TEST_TIER="$(SDK_TEST_TIER)" go test ./tests/trainer -list '$(SDK_TRAINER_SANITY_INCLUDE_REGEX)' | rg '^Test' | (if [[ -n "$$EFFECTIVE_EXCLUDE_REGEX" ]]; then rg -v "$$EFFECTIVE_EXCLUDE_REGEX"; else awk '1'; fi) | paste -sd'|' -)"; \
	if [[ -z "$$TESTS" ]]; then \
		echo "No SDK sanity tests matched include/exclude filters."; \
		exit 1; \
	fi; \
	echo "Running SDK sanity tests (TEST_TIER='$(SDK_TEST_TIER)'): $$TESTS"; \
	TEST_TIER="$(SDK_TEST_TIER)" go test -timeout "$(SDK_TEST_TIMEOUT)" -count=1 ./tests/trainer -run "^($$TESTS)$$" -v

trainer-sdk-print-selected:
	@set -euo pipefail; \
	EFFECTIVE_EXCLUDE_REGEX="$${SDK_TEST_EXCLUDE_REGEX:-}"; \
	if [[ "$$(echo "$(SKIP_S3_TESTS)" | tr '[:lower:]' '[:upper:]')" == "TRUE" ]]; then \
		if [[ -n "$$EFFECTIVE_EXCLUDE_REGEX" ]]; then \
			EFFECTIVE_EXCLUDE_REGEX="($$EFFECTIVE_EXCLUDE_REGEX)|($(SDK_TEST_S3_EXCLUDE_REGEX))"; \
		else \
			EFFECTIVE_EXCLUDE_REGEX="$(SDK_TEST_S3_EXCLUDE_REGEX)"; \
		fi; \
	fi; \
	TEST_TIER="$(SDK_TEST_TIER)" go test ./tests/trainer -list '$(SDK_TRAINER_INCLUDE_REGEX)' | rg '^Test' | (if [[ -n "$$EFFECTIVE_EXCLUDE_REGEX" ]]; then rg -v "$$EFFECTIVE_EXCLUDE_REGEX"; else awk '1'; fi)

trainer-sdk-print-tiers:
	@echo "Trainer SDK tiers exposed via this wrapper:"
	@echo "  Sanity"
	@echo "  Tier1"
	@echo "  CUDA   (mapped to trainer KFTO-CUDA)"
	@echo "  ROCm   (mapped to trainer KFTO-ROCm)"

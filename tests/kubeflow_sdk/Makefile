SHELL := /usr/bin/env bash

SDK_TEST_TIMEOUT ?= 60m
SDK_TEST_EXCLUDE_REGEX ?=
ACCELERATOR_TESTS ?= CUDA
NOTEBOOK_ACCELERATOR ?=
SKIP_S3_TESTS ?= false

.PHONY: help sdk-test sdk-test-all sdk-test-trainer-all sdk-test-trainer-cpu sdk-test-trainer-sanity sdk-test-trainer-tier1 sdk-test-trainer-cuda sdk-test-trainer-rocm sdk-test-trainer-with-tier sdk-print-selected-trainer-tests sdk-print-trainer-tiers sdk-print-tiers

help:
	@echo "Kubeflow SDK test targets"
	@echo ""
	@echo "Core:"
	@echo "  sdk-test                    Run default SDK set (Sanity + Tier1 + accelerator lane)"
	@echo "  sdk-test-all                Alias for sdk-test (all components, currently trainer only)"
	@echo "  sdk-test-trainer-all        Run all trainer SDK tests (all tiers)"
	@echo "  sdk-test-trainer-cpu        Run trainer SDK CPU lane (Sanity + Tier1)"
	@echo "  sdk-test-trainer-sanity     Run trainer SDK sanity tier"
	@echo "  sdk-test-trainer-tier1      Run trainer SDK tier1"
	@echo "  sdk-test-trainer-cuda       Run trainer SDK CUDA lane"
	@echo "  sdk-test-trainer-rocm       Run trainer SDK ROCm lane"
	@echo "  sdk-test-trainer-with-tier  Run trainer SDK tests with SDK_TEST_TIER=<tier>"
	@echo ""
	@echo "Debug:"
	@echo "  sdk-print-selected-trainer-tests"
	@echo "  sdk-print-trainer-tiers"
	@echo "  sdk-print-tiers            Print tiers for all SDK components"
	@echo ""
	@echo "Supported env vars:"
	@echo "  SDK_TEST_TIMEOUT            (default: 60m)"
	@echo "  SDK_TEST_EXCLUDE_REGEX      Exclude tests by name regex"
	@echo "  SKIP_S3_TESTS               true|false (exclude S3/AWS-related trainer SDK tests)"
	@echo "  ACCELERATOR_TESTS           CUDA (default) | CPU | ROCM | ALL"
	@echo "                              (controls lane selection and default NOTEBOOK_IMAGE when unset)"
	@echo "  NOTEBOOK_ACCELERATOR        Optional image-only override: CUDA | CPU | ROCM"
	@echo "  SDK_TEST_TIER               Custom trainer tier for sdk-test-trainer-with-tier"

sdk-test: sdk-test-all

sdk-test-all:
	@set -euo pipefail; \
	$(MAKE) -f tests/kubeflow_sdk/Makefile sdk-test-trainer-sanity SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"; \
	$(MAKE) -f tests/kubeflow_sdk/Makefile sdk-test-trainer-tier1 SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"; \
	case "$$(echo "$(ACCELERATOR_TESTS)" | tr '[:lower:]' '[:upper:]')" in \
		CPU) ;; \
		CUDA) $(MAKE) -f tests/kubeflow_sdk/Makefile sdk-test-trainer-cuda SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)" ;; \
		ROCM) $(MAKE) -f tests/kubeflow_sdk/Makefile sdk-test-trainer-rocm SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)" ;; \
		ALL)  $(MAKE) -f tests/kubeflow_sdk/Makefile sdk-test-trainer-cuda SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"; \
			  $(MAKE) -f tests/kubeflow_sdk/Makefile sdk-test-trainer-rocm SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)" ;; \
		*) echo "Unsupported ACCELERATOR_TESTS='$(ACCELERATOR_TESTS)'. Use CPU, CUDA, ROCM, or ALL."; exit 1 ;; \
	esac

sdk-test-trainer-all:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-all SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" NOTEBOOK_ACCELERATOR="$(NOTEBOOK_ACCELERATOR)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"

sdk-test-trainer-cpu:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-cpu SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" NOTEBOOK_ACCELERATOR="CPU" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"

sdk-test-trainer-sanity:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-sanity SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" NOTEBOOK_ACCELERATOR="$(NOTEBOOK_ACCELERATOR)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"

sdk-test-trainer-tier1:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-tier1 SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" NOTEBOOK_ACCELERATOR="$(NOTEBOOK_ACCELERATOR)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"

sdk-test-trainer-cuda:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-cuda SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" NOTEBOOK_ACCELERATOR="CUDA" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"

sdk-test-trainer-rocm:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-rocm SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" NOTEBOOK_ACCELERATOR="ROCM" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"

sdk-test-trainer-with-tier:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-test-with-tier SDK_TEST_TIER="$(SDK_TEST_TIER)" SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SDK_TEST_TIMEOUT="$(SDK_TEST_TIMEOUT)" ACCELERATOR_TESTS="$(ACCELERATOR_TESTS)" NOTEBOOK_ACCELERATOR="$(NOTEBOOK_ACCELERATOR)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"

sdk-print-selected-trainer-tests:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-print-selected SDK_TEST_EXCLUDE_REGEX="$(SDK_TEST_EXCLUDE_REGEX)" SKIP_S3_TESTS="$(SKIP_S3_TESTS)"

sdk-print-trainer-tiers:
	@$(MAKE) -f tests/kubeflow_sdk/trainer/Makefile trainer-sdk-print-tiers

sdk-print-tiers:
	@$(MAKE) -f tests/kubeflow_sdk/Makefile sdk-print-trainer-tiers

# Distributed Workloads Integration Tests

## Prerequisites

* Admin access to an OpenShift cluster ([CRC](https://developers.redhat.com/products/openshift-local/overview) is fine)

* Installed OpenDataHub or RHOAI, enabled all Distributed Workload components

* Installed Go 1.21

## Environment variables

* `CODEFLARE_TEST_OUTPUT_DIR` - Output directory for test logs
* `CODEFLARE_TEST_TIMEOUT_SHORT` - Timeout duration for short tasks
* `CODEFLARE_TEST_TIMEOUT_MEDIUM` - Timeout duration for medium tasks
* `CODEFLARE_TEST_TIMEOUT_LONG` - Timeout duration for long tasks

## Running Tests

Execute tests like standard Go unit tests.

```bash
go test -timeout 60m ./tests/kfto/
```

# Running Tests Manually

## Prerequisites

* Admin access to an OpenShift cluster ([CRC](https://developers.redhat.com/products/openshift-local/overview) is fine)

* If you run these tests in a local cluster and have not deployed the Open Data Hub on your OpenShift cluster:

```bash
# Go to the root folder of the repository
cd ..

# Install CodeFlare operator
make install-codeflare-operator

# Install ODH operator
make install-opendatahub-operator

# Deploy ODH and CodeFlare stack
make deploy-codeflare
```

## Go tests - Setup

* Install Go 1.20

## Go tests - Environment variables

* `ODH_NAMESPACE` - Namespace where ODH is installed
* `CODEFLARE_TEST_OUTPUT_DIR` - Output directory for test logs
* `CODEFLARE_TEST_TIMEOUT_SHORT` - Timeout duration for short tasks
* `CODEFLARE_TEST_TIMEOUT_MEDIUM` - Timeout duration for medium tasks
* `CODEFLARE_TEST_TIMEOUT_LONG` - Timeout duration for long tasks

## Go tests - Running Tests

Execute tests like standard Go unit tests.

```bash
go test -v ./integration
```

## Troubleshooting

If any of the above is unclear or you run into any problems, please open an issue in the [opendatahub-io/distributed-workloads](https://github.com/opendatahub-io/distributed-workloads/issues) repository.

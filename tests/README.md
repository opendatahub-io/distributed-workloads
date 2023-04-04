# Running Tests Manually

Manually running tests requires use of the the [opendatahub-io/peak](https://github.com/opendatahub-io/peak) project.

## Prerequisites

* Admin access to an OpenShift cluster ([CRC](https://developers.redhat.com/products/openshift-local/overview) is fine)

* Mac users may need to do the following:
```bash
brew install coreutils
ln -s /usr/local/bin/greadlink /usr/local/bin/readlink
```

## Setup

Clone the [opendatahub-io/peak](https://github.com/opendatahub-io/peak) project anywhere you like in your working environment. But, do not clone it into the `distributed-workloads` directory.

```bash
git clone https://github.com/opendatahub-io/peak
cd peak
```

Then we need to update the peak project with its submodule dependencies. Specifically, [opendatahub-io/openshift-test-kit](https://github.com/opendatahub-io/openshift-test-kit/tree/0e469c4bf967b531780eb05d6b96463214288db7) defined in the `.gitmodules` file.

```bash
git submodule update --init
``` 

Now we need to pull our `distributed workloads` project into the peak repo for testing. This is done by creating a file, `my-list`, that contains the repository name you want to use, the channel, the repo's location (this can be a github url or a relative path to a local directory) and branch name.

For example, if you cloned peak into the same directory level as `distributed-workloads`, then you would create a file, `my-list`, in the following way:

```bash
echo distributed-workloads nil ../distributed-workloads main > my-list
```

Now we can setup our tests.

```bash
./setup.sh -t my-list
```

This should create a directory, `distributed-workloads` in the `operator-tests` directory of the peak repo.

## Running Tests

`run.sh` will search through the 'operator-tests' directory for a *.sh file name we provide to it as an argument. In this case, we want to run the `distributed-workloads.sh` script.

```bash
./run.sh distributed-workloads.sh
```
If everything is working correctly you should see an output similar to the below:

```bash
Running example test

Running operator-tests/distributed-workloads/tests/basictests/distributed-workloads.sh:15: executing 'oc project opendatahub' expecting success...


✔ SUCCESS after 0.184s: operator-tests/distributed-workloads/tests/basictests/distributed-workloads.sh:15: executing 'oc project opendatahub' expecting success

Running operator-tests/distributed-workloads/tests/basictests/distributed-workloads.sh:16: executing 'oc get pods' expecting success...


✔ SUCCESS after 0.127s: operator-tests/distributed-workloads/tests/basictests/distributed-workloads.sh:16: executing 'oc get pods' expecting success


Installing Codeflare Operator


Installing distributed workloads kfdef


Testing MCAD TorchX Functionality


Testing MCAD Ray Functionality


Uninstalling distributed workloads kfdef


Uninstalling Codeflare Operator

```


## Troubleshooting

If any of the above is unclear or you run into any problems, please open an issue in the [opendatahub-io/distributed-workloads](https://github.com/opendatahub-io/distributed-workloads/issues) repository. 
# CodeFlare-Stack

Artifacts for installing the CodeFlare stack as part of ODH

## Overview

CodeFlare is a simple, user-friendly abstraction for scaling,
queuing and resource management of distributed AI/ML and Python workloads.
It consists of three components:

* [CodeFlare SDK](https://github.com/project-codeflare/codeflare-sdk) to define and control remote distributed compute jobs and infrastructure with any Python based environment
* [Multi-Cluster Application Dispatcher (MCAD)](https://github.com/project-codeflare/multi-cluster-app-dispatcher) for management of batch jobs
* [Instascale](https://github.com/project-codeflare/instascale) for on-demand scaling of a Kubernetes cluster

We recommend using this stack alongside [KubeRay](https://github.com/ray-project/kuberay) for management of remote Ray clusters on Kubernetes for running distributed compute workloads

Integration of this stack into the Open Data Hub is owned by the Distributed Workloads Working
Group. See [this page](https://github.com/opendatahub-io/opendatahub-community/tree/master/wg-distributed-workloads)
for further details and how to get in touch.

## Quick Start

There is a sample kfdef at the root of this repository that you can use to deploy the recommended CodeFlare stack.
Follow our quick start guide [here](https://github.com/opendatahub-io/distributed-workloads/blob/main/Quick-Start.md) to get up and running with Distributed Workflows on Open Data Hub.

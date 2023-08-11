# distributed-workloads

Artifacts for installing the Distributed Workloads stack as part of ODH

## Overview

Distributed Workloads is a simple, user-friendly abstraction for scaling,
queuing and resource management of distributed AI/ML and Python workloads.
It consists of three components:

* [CodeFlare SDK](https://github.com/project-codeflare/codeflare-sdk) to define and control remote distributed compute jobs and infrastructure with any Python based environment
* [Multi-Cluster Application Dispatcher (MCAD)](https://github.com/project-codeflare/multi-cluster-app-dispatcher) for management of batch jobs
* [Instascale](https://github.com/project-codeflare/instascale) for on-demand scaling of a Kubernetes cluster

* [KubeRay](https://github.com/ray-project/kuberay) for management of remote Ray clusters on Kubernetes for running distributed compute workloads

Integration of this stack into the Open Data Hub is owned by the Distributed Workloads Working Group. See [this page](https://github.com/opendatahub-io/opendatahub-community/tree/master/wg-distributed-workloads) for further details and how to get in touch.

### Compatibilty Matrix

| Component                    | Version |
|------------------------------|---------|
| CodeFlare Operator           | v0.1.0  |
| Multi-Cluster App Dispatcher | v1.33.0 |
| CodeFlare-SDK                | v0.6.1  |
| InstaScale                   | v0.0.6  |
| KubeRay                      | v0.5.0  |

## Quick Start

Follow our quick start guide [here](/Quick-Start.md) to get up and running with Distributed Workflows on Open Data Hub.  

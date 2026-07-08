# Ray Cluster Autoscaling on OpenShift AI

This example demonstrates the full lifecycle of a Ray cluster with **in-tree autoscaling** on OpenShift AI: create the cluster, drive scale-up with a bursty CPU workload, observe scale-down when idle, and tear the cluster down.

It uses the CodeFlare SDK (`enable_autoscaling=True`, `min_workers`, `max_workers`) and the companion script `autoscaling_load.py` submitted through the Ray Jobs API.

For a shorter SDK-focused walkthrough, see the [CodeFlare SDK guided demo](https://github.com/project-codeflare/codeflare-sdk/tree/main/demo-notebooks/guided-demos/6_autoscaling.ipynb).

> [!IMPORTANT]
> Ray cluster autoscaling is **not supported when Kueue manages your namespace**. Use a project without a default Kueue LocalQueue, or disable Kueue for the namespace before running this example. Elastic Ray jobs with Kueue are tracked in [RHAIRFE-909](https://redhat.atlassian.net/browse/RHAIRFE-909).
>
> This example has been tested with the configuration listed in [Validation](#validation). Adapt CPU and memory requests to your cluster capacity.

## Requirements

* An OpenShift cluster with OpenShift AI (RHOAI) 2.25+ installed:
  * The `codeflare`, `dashboard`, `ray`, and `workbenches` components enabled;
* A data science project namespace **without Kueue admission** for RayClusters (see note above);
* Sufficient worker node capacity for at least `max_workers + 1` Ray pods (head + workers);
* CodeFlare SDK with autoscaling support installed in the workbench (RHOAI 3.5+).

## Setup

* Access the OpenShift AI dashboard from the top navigation bar menu.
* Log in, go to _Data Science Projects_, and create a project (or pick an existing **non-Kueue** project).
* Create a workbench with a Python 3.11 or 3.12 image that includes the CodeFlare SDK.
* When the workbench is ready, clone this repository:

  `https://github.com/opendatahub-io/distributed-workloads.git`

* Navigate to `distributed-workloads/examples/ray-cluster-autoscaling` and open `ray_cluster_autoscaling.ipynb`.
* Set the CodeFlare SDK authentication parameters in the notebook (`TokenAuthentication` token and server from `oc whoami -t` and `oc whoami --show-server`).

## Running the Example

Execute the notebook step by step. It will:

1. Create a Ray cluster with `min_workers=1` and `max_workers=2`.
2. Wait until the cluster is ready and confirm one worker pod is running.
3. Submit `autoscaling_load.py`, which queues three single-CPU tasks (more than head + one worker can run concurrently).
4. Wait until a second worker pod appears (scale-up).
5. Wait for the job to finish and for worker count to return to `min_workers` (scale-down).
6. Delete the Ray job and call `cluster.down()`.

While the workload runs, use the printed Ray dashboard link from `cluster.details()` or the `oc get pods` / `oc get raycluster` commands in the notebook to watch worker pods change over time.

## Validation

This example has been validated with the following configuration:

* OpenShift AI 3.5 development environment
* CPU-only Ray cluster (no GPU required for the autoscaling smoke test)
* CodeFlare SDK with `enable_autoscaling` support

```python
ClusterConfiguration(
    name="ray-autoscale",
    enable_autoscaling=True,
    min_workers=1,
    max_workers=2,
    head_cpu_requests=1,
    head_cpu_limits=1,
    head_memory_requests=7,
    head_memory_limits=8,
    worker_cpu_requests=1,
    worker_cpu_limits=1,
    worker_memory_requests=5,
    worker_memory_limits=6,
    head_extended_resource_requests={"nvidia.com/gpu": 0},
    worker_extended_resource_requests={"nvidia.com/gpu": 0},
)
```

Load job entrypoint:

```python
"AUTOSCALING_TASKS=3 AUTOSCALING_TASK_SLEEP_S=180 python autoscaling_load.py"
```

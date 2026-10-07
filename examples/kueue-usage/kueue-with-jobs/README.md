# Kueue Jobs Example

Here are a set of resources that can be used to test `kueue` features like preemption.

## Install
Install the base resources
```
for i in 00-common.yaml 01-cluster-queues.yaml 02-local-queues.yaml; do
  oc create -f $i
done
```

## Run
Run the workloads from jobs

```
for i in $(seq 1 7); do oc create -f 04-llm-job.yaml; done
for i in $(seq 5); do oc create -f 05-cancer-cure-research.yaml; done
```

## Observe

Observe that `workloads` get created=
```
oc get wl
```
Observe that `workloads` get preempted using `oc describe wl <some not admitted workload>`







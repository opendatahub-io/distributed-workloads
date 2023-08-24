# Distributed Workloads Stack Installation

This document outlines the steps to be followed to install the Distributed Workloads stack on with RHODS. The instrcutions are written for RHODS 1.30 or earlier. With RHODS 1.31, the installation process will be changed.

1. Install RHODS via the operatorhub ui. The installation doesn't matter, but stable is the channel we usually recommend for self-managed installs
2. Install CodeFlare Operator via the operatorhub ui. The only CodeFlare Operator available today is the one from the community catalog. Once available via the official operators, these instructions will be updated
3. Once the `redhat-ods-operator` namespace is created, you will need to give the operator service account permissions to edit SCCs. Create the following clusterrole and clusterrolebinding

    ```yaml
    kind: ClusterRole
    apiVersion: rbac.authorization.k8s.io/v1
    metadata:
    name: rhods-operator-scc
    rules:
    - verbs:
        - get
        - watch
        - list
        - create
        - update
        - patch
        - delete
        apiGroups:
        - security.openshift.io
        resources:
        - securitycontextconstraints
    ```

    ```yaml
    kind: ClusterRoleBinding
    apiVersion: rbac.authorization.k8s.io/v1
    metadata:
    name: rhods-operator-scc
    subjects:
    - kind: ServiceAccount
        name: rhods-operator
        namespace: redhat-ods-operator
    roleRef:
    apiGroup: rbac.authorization.k8s.io
    kind: ClusterRole
    name: rhods-operator-scc
    ```

4. Finally, apply the following kfdef

    ```yaml
    apiVersion: kfdef.apps.kubeflow.org/v1
    kind: KfDef
    metadata:
    name: codeflare-stack
    namespace: redhat-ods-applications
    spec:
    applications:
        - kustomizeConfig:
            repoRef:
            name: manifests
            path: codeflare-stack
        name: codeflare-stack
        - kustomizeConfig:
            repoRef:
            name: manifests
            path: ray/operator
        name: ray-operator
    repos:
        - name: manifests
        uri: https://github.com/red-hat-data-services/distributed-workloads/tarball/main
    ```

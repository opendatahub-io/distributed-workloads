# Distributed Workloads Stack Installation

This document outlines the steps to be followed to install the Distributed Workloads stack on RHODS. The instructions are written for RHODS 1.31 or earlier. With RHODS 1.32+, the installation process might be changed.

1. Install RHODS via the operatorhub ui. The installation doesn't matter, but the `stable` channel is usually recommended for self-managed installs
   - From the OpenShift UI, click Operators --> OperatorHub and search for: `Red Hat OpenShift Data Science`

2. Install CodeFlare Operator via the operatorhub ui. The only CodeFlare Operator available today is the one from the community catalog. Once available via the official operators, these instructions will be updated
   - From the OpenShift UI, click Operators --> OperatorHub and search for: `CodeFlare operator`.  Choose the Community catalog source for now
3. Once the `redhat-ods-operator` namespace is created, you will need to give the operator service account permissions to edit SCCs. Create the following clusterrole and clusterrolebinding by issueing the commands below from a terminal where you issued an `oc login` to your cluster.


   ```bash
   oc apply -f - <<EOF
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
   EOF
   ```

   ```bash
   oc apply -f - <<EOF
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
   EOF
   ```

4. Finally, apply the following kfdef

   ```bash
   oc apply -f - <<EOF
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
   EOF
   ```

You can then continue with the normal CodeFlare Quick-Start from this section on: [Submit-Your-First-Job](https://github.com/opendatahub-io/distributed-workloads/blob/main/Quick-Start.md#submit-your-first-job)

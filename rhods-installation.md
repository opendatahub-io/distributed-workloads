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
  Find the dashboard/notebook UI by:
  ```
  oc get route -n redhat-ods-applications |grep dash |awk '{print $2}'
  ```
   Put the URL into a browser and if prompted, login with your OpenShift userid and password

  Once you are on your dashboard, you can select "Launch application" on the Jupyter application. This will take you to your notebook spawner page. 

  You can then continue with the normal CodeFlare Quick-Start from this section on: [Submit-Your-First-Job](https://github.com/opendatahub-io/distributed-workloads/blob/main/Quick-Start.md#submit-your-first-job)

# RHODS Cleanup steps

To completely clean up all the CodeFlare components after an install, follow these steps:

1.  No appwrappers should be left running:
    ```bash
    oc get appwrappers -A
    ```
     If any are left, you'd want to delete them
    
2. Remove the notebook and notebook pvc:
   ```bash
   oc delete notebook jupyter-nb-kube-3aadmin -n rhods-notebooks
   oc delete pvc jupyterhub-nb-kube-3aadmin-pvc -n rhods-notebooks
   ```

3. Remove the clusterrole and clusterrolebindings that were added:
   ```
   oc delete ClusterRoleBinding rhods-operator-scc
   oc delete ClusterRole rhods-operator-scc
   ```

4. Remove the codeflare-stack kfdef
    ``` bash
    oc delete kfdef codeflare-stack -n redhat-ods-applications
    ```

5. Remove the CodeFlare Operator csv and subscription:
   ```bash
   oc delete sub codeflare-operator -n openshift-operators
   oc delete csv `oc get csv -n opendatahub |grep codeflare-operator |awk '{print $1}'` -n openshift-operators
   ```

6. Remove the CodeFlare CRDs
   ```bash
   oc delete crd instascales.codeflare.codeflare.dev mcads.codeflare.codeflare.dev schedulingspecs.mcad.ibm.com queuejobs.mcad.ibm.com
   ```
7. If you're removing the RHODS kfdefs and operator, you'd want to do this:

   7.1 Delete all the kfdefs:  (Note, this can take awhile as it needs to stop all the running pods in redhat-ods-applications, redhat-ods-monitoring and rhods-notebooks)
   ```
   oc delete kfdef rhods-anaconda rhods-dashboard  rhods-data-science-pipelines-operator rhods-model-mesh  rhods-nbc
   oc delete kfdef  modelmesh-monitoring monitoring -n redhat-ods-monitoring
   oc delete kfdef rhods-notebooks -n rhods-notebooks
   ```

   7.2 And then delete the subscription and the csv:
   ```
   oc delete sub rhods-operator -n redhat-ods-operator
   oc delete csv `oc get csv -n redhat-ods-operator |grep rhods-operator |awk '{print $1}'` -n redhat-ods-operator
   ```

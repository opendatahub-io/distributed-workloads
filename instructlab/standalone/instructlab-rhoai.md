# Distributed InstructLab Training on RHOAI
This file documents an experimental install of InstructLab on Red Hat OpenShift AI.

## Pre-requisites

* An OpenShift cluster with 
    * Sufficient GPU nodes available for training. 
        * 2 nodes with at least 2 NVIDIA A100 GPUs each.
    * NFS Provisioner Operator installed (Optional - only required if the cluster being used doesn’t support ReadWriteMany already)
    * Red Hat - Authorino installed
    * Red Hat Openshift Serverless installed
* An OpenShift AI installation, with the Training Operator and kserve components set to `Managed`
    * A data science project/namespace
* A [StorageClass](https://kubernetes.io/docs/concepts/storage/storage-classes/) that supports dynamic provisioning with [ReadWriteMany](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes) access mode (see step 2a below).
* An AWS S3 compatible object store. Alternative object storage solutions such as Ceph, Nooba and MinIO are also compatible.
* SDG data generated that has been uploaded to an object store (see step 1 below).

## Steps

Before running the training and evaluation steps we must complete the following:

* Step 1 - Prepare data and push to object store
* Step 2 - Create judge model server
    * Step 2a - Create nfs server

### Step 1 - Prepare data and push to object store

Create a tarball with the data, [model](https://huggingface.co/ibm-granite/granite-7b-base/tree/main) and [taxonomy](https://github.com/instructlab/taxonomy) and push them to your object store.

```
$ mkdir -p s3-data/{data,model,taxonomy}

#to generate SDG data
$ cd s3-data/data
$ ilab model download --repository ibm-granite/granite-7b-base
$ ilab config init
# Use the downloaded model path
$ ilab model serve
$ ilab data generate --taxonomy-base=empty

# download ilab model repository in s3-data model direct
$ ilab model download --repository ibm-granite/granite-7b-base
$ cp -r ibm-granite/granite-7b-base/* s3-data/model

# to download taxonomy repository
$ cd s3-data && git clone https://github.com/instructlab/taxonomy.git taxonomy 

# Generate tar archive
$ cd s3-data && tar -czvf rhelai.tar.gz *
```

Upload the created tar archive to your object store.

### Step 2 - Create Judge model server

The judge model is used for model evaluation.

* Create a service account to be used for token authentication

```
apiVersion: v1
kind: ServiceAccount
metadata:
  name: <model-server-service-account-name>
  Namespace: <data-science-project-name/namespace> 
```

* Upload [prometheus-eval](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0-GGUF/blob/main/prometheus-7b-v2.0.Q4_K_M.gguf) judge model file to the same object storage/S3 bucket as before. 
    * This [script](https://github.com/redhat-et/ilab-on-ocp/blob/main/kubernetes_yaml/model_downloader/container_file/download_hf_model.py) can be used for uploading the model to object storage.
    * Update the MODEL and MODEL_PATH parameters if using a different model to that in the example.

Example:
```
export MODEL=mistralai/Mistral-7B-Instruct-v0.2 \                                                                               
S3_ENDPOINT=<s3-bucket-endpoint> \
AWS_ACCESS_KEY=<s3-bucket-access-key> \   
AWS_SECRET_KEY=<s3-bucket-secret-key> \       
AWS_BUCKET_NAME=<bucket-name> \   
S3_FOLDER=mistral \           
MODEL_PATH=model \
HF_TOKEN=<hugging-face-auth-token>
```

* Navigate to the OpenShift AI dashboard
    * Choose Data Science Projects from the left hand menu and choose your data science project/namespace.
    * Choose the data connections tab, and click on the Add data connection button. Enter the details of your S3 bucket and click Add data connection.

    Note: Before following the next step - Ensure that the CapabilityServiceMeshAuthorization status is True in DSCinitialization resource.

    * Create a model server instance
        * Navigate to Data Science Projects and then the Models tab
        * On the right hand side select ‘Deploy model’ under Single- model serving platform
        * Under Serving runtime choose the serving runtime we edited above.
        * Check the `Make deployed models available through an external route` box.
        * Under token authentication check the `Require token authentication` box, select the service account that we have created above.
        * Choose the existing data connection created earlier. 
        * Click deploy.

* Create a secret containing the judge model serving details

```
apiVersion: v1
kind: Secret
metadata:
  name: judge-serving-details
type: Opaque
stringData:
  JUDGE_API_KEY: Service Account Token
  JUDGE_ENDPOINT: Model serving endpoint. You must pass https://url/v1
  JUDGE_NAME: name of the judge model or deployment
  JUDGE_CA_CERT: configmap containing CA cert for the judge model (optional - required if using custom CA cert)
  JUDGE_CA_CERT_CM_KEY: Name of key inside configmap (optional - required if using custom CA cert)
```

* If using a custom CA certificate you must provide the relevant data in a ConfigMap. The config map name and key are then provided as a parameter to the standalone.py script in Step 6 below, as well as in the secret above.

### Step 2a - Create NFS Server

Note - there are 2 options here

* Option 1 - Create NFS server using the CSI driver
* Option 2 - Create NFS server using NFS Provisioner Operator

**Option 1 - Create NFS Server using the CSI Driver**

```
$ git clone https://github.com/OCP-on-NERC/nerc-ocp-config.git 

$ kubectl apply -k  csi-driver-nfs/overlays/nerc-ocp-test

$ kubectl apply -k  nfs/overlays/nerc-ocp-test 
```

This will create the required resources in the cluster, including the required StorageClass. 

**Option 2 - Create NFS Server using NFS Provisioner Operator**

* You have installed NFS Provisioner Operator as a pre-requisite above
* Create a StorageClass that supports dynamic provision with ReadWriteMany access mode.

```
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: nfs-storage
provisioner: nfs.csi.k8s.io
reclaimPolicy: Delete
mountOptions:
  - nfsvers=4.1
volumeBindingMode: Immediate
```

Apply the yaml to the cluster.

* Create an NFSProvisioner resource using the following yaml

Note : Replace scForNFSPvc (gp3-csi) below to use the name of your default storage class to assign storage to the NFS to be provisioned

```
apiVersion: cache.jhouse.com/v1alpha1
kind: NFSProvisioner
metadata:
  name: rhelai-nfsprovisioner
spec:
  nfsImageConfiguration:
	image: k8s.gcr.io/sig-storage/nfs-provisioner@sha256:e943bb77c7df05ebdc8c7888b2db289b13bf9f012d6a3a5a74f14d4d5743d439
	imagePullPolicy: IfNotPresent
  storageSize: "300G"
  scForNFSPvc: gp3-csi
  scForNFS: nfs
```

Apply the yaml to the cluster.

### Now we can continue to set up the required resources in our cluster:

The following resources will be created:

* ConfigMap
* Secret
* ClusterRole
* ClusterRoleBinding
* Pod 

### Step 3 - create a configMap that contains the[ standalone.py script](https://github.com/redhat-et/ilab-on-ocp/blob/main/standalone.py)

```
>> curl -OL https://raw.githubusercontent.com/red-hat-data-services/distributed-workloads/refs/heads/rhoai-2.15/instructlab/standalone/standalone.py

>> oc create configmap -n <data-science-project-name/namespace> standalone-script --from-file ./standalone.py
```

### Step 4 - Create a secret resource that contains the credentials for your Object Storage (AWS S3 Bucket)

Note: encode these credentials in Base-64 form and then add it to the secret yaml file below:

```
apiVersion: v1
kind: Secret
metadata:
  name: sdg-object-store-credentials
type: Opaque
data:
  bucket: #The object store bucket containing SDG data. (Name of S3 bucket)
  access_key: #The object store access key for SDG. (AWS Access key ID)
  secret_key: #The object store secret key for SDG. (AWS Secret Access Key)
  data_key: #The name of the tarball that contains SDG data.
  endpoint: # The object store endpoint for SDG.
  region: # The region for the object store.
  verify_tls:  #Verify TLS for the object store.
```

Apply the yaml file to the cluster

### Step 5 - Create a ServiceAccount, ClusterRole and ClusterRoleBinding 

Provide access to the service account running the standalone.py script for accessing and manipulating related resources.

```
apiVersion: v1
kind: ServiceAccount
metadata:
  name: <name-of-workbench-service-account>
  namespace: <data-science-project-name/namespace>
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  namespace: <data-science-project-name/namespace>
  name: secret-access-role
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "configmaps", "persistentvolumeclaims", "secrets","events"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]

  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["get", "list", "create", "watch"]

  - apiGroups: ["kubeflow.org"]
    resources: ["pytorchjobs"]
    verbs: ["get", "list", "create", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: secret-access-binding
subjects:
- kind: ServiceAccount
  name: <workbench-service-account-name> # created above
  namespace: <data-science-project-name/namespace>
roleRef:
  kind: ClusterRole
  name: secret-access-role
  apiGroup: rbac.authorization.k8s.io
```

Apply the yaml to the cluster.

These are the required [RBAC configuration](https://github.com/opendatahub-io/distributed-workloads/tree/main/instructlab/standalone#rbac-requirements-when-running-in-a-kubernetes-job)s which we are applying on the ServiceAccount.

### Step 6 - Create the workbench pod and run the standalone.py script 

* In this step the standalone.py script will be utilised. The script runs a pytorchjob that utilises Fully Sharded Data Parallel (FSDP), sharing the load across available resources (GPUs).
* Prepare the pod yaml like below including this [workbench image](https://quay.io/repository/opendatahub/workbench-images/manifest/sha256:7f26f5f2bec4184af15acd95f29b3450526c5c28c386b6cb694fbe82d71d0b41). This pod will access and run the standalone.py script from the configmap that we created earlier.
* Note that the value for `judge-serving-model-api-key`should match the jwt token generated when setting up the judge serving mode (step 2 above).
* If using a custom CA certificate, these optional parameters are available when running the standalone.py script. These should be added to the command in the pod yaml container:

    `JUDGE_CA_CERT: The name of ConfigMap containing the custom CA Cert - Optional`
    
    `JUDGE_CA_CERT_CM_KEY: The key of the CA Cert in the ConfigMap - Optional`

```
apiVersion: v1 \
kind: Pod \
metadata: \
  name: ilab-pod \
  namespace:  &lt;data-science-project-name/namespace> \
spec: \
  serviceAccountName: &lt;service-account-name> \
  containers: \
  - name: workbench-container \
    image: quay.io/opendatahub/workbench-images@sha256:7f26f5f2bec4184af15acd95f29b3450526c5c28c386b6cb694fbe82d71d0b41 \
    env: \
    - name: SDG_OBJECT_STORE_ENDPOINT \
      valueFrom: \
        secretKeyRef: \
          name: sdg-object-store-credentials \
          key: endpoint \
    - name: SDG_OBJECT_STORE_BUCKET \
      valueFrom: \
        secretKeyRef: \
          name: sdg-object-store-credentials \
          key: bucket \
    - name: SDG_OBJECT_STORE_ACCESS_KEY \
      valueFrom: \
        secretKeyRef: \
          name: sdg-object-store-credentials \
          key: access_key \
    - name: SDG_OBJECT_STORE_SECRET_KEY \
      valueFrom: \
        secretKeyRef: \
          name: sdg-object-store-credentials \
          key: secret_key \
    - name: SDG_OBJECT_STORE_REGION \
      valueFrom: \
        secretKeyRef: \
          name: sdg-object-store-credentials \
          key: region \
    - name: SDG_OBJECT_STORE_DATA_KEY \
      valueFrom: \
        secretKeyRef: \
          name: sdg-object-store-credentials \
          key: data_key \
    - name: SDG_OBJECT_STORE_VERIFY_TLS \
      valueFrom: \
        secretKeyRef: \
          name: sdg-object-store-credentials \
          key: verify_tls \
    volumeMounts: \
    - name: script-volume \
      mountPath: /home/standalone.py \
      subPath: standalone.py \
    command: ["python3", "/home/standalone.py", "run",
              "--namespace", "&lt;data-science-project-name/namespace>",
              "--judge-serving-model-endpoint","&lt;model-inference-endpoint-url>",
              "--judge-serving-model-name","&lt;model-server-name>",
              "--judge-serving-model-secret", "&lt;judge-model-details-k8s-secret>",
              "--storage-class", "&lt;storage-class-name>",
              "--sdg-object-store-secret", "&lt;created-s3-credentials-secret-name>",
              "--nproc-per-node" , 1,
              "--force-pull"] \
  volumes: \
  - name: script-volume \
    configMap: \
      name: standalone-script
```
Apply the yaml to the cluster.

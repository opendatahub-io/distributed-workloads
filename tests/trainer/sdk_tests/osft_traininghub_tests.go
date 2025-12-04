/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package sdk_tests

import (
	"fmt"
	"os"
	"testing"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"

	common "github.com/opendatahub-io/distributed-workloads/tests/common"
	support "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	osftNotebookName = "osft.ipynb"
	osftNotebookPath = "../resources/" + osftNotebookName
)

// Multi-GPU - Distributed Training with OSFT and TrainingHubTrainer
func RunOsftTrainingHubMultiGpuDistributedTraining(t *testing.T) {
	test := support.With(t)

	// Create a new test namespace
	namespace := test.NewTestNamespace()

	// Create custom ClusterTrainingRuntime for 2 nodes with 1 GPU each (to be updated to 2 gpus each later)
	runtimeName := "training-hub-2node-1gpu"
	image := "quay.io/opendatahub/odh-training-th03-cuda128-torch28-py312-rhel9:latest"
	trainerutils.CreateClusterTrainingRuntime(t, test, runtimeName, 2, 1, image) // TODO: this will need to be updated to use 2 GPUs per node (if/when this config is available)

	// Ensure Notebook ServiceAccount exists (no extra RBAC)
	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	// RBACs setup
	userName := common.GetNotebookUserName(test)
	userToken := common.GetNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Create ConfigMap with notebook
	localPath := osftNotebookPath
	nb, err := os.ReadFile(localPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", localPath))
	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{osftNotebookName: nb})

	// Build command with parameters and pinned deps, and print definitive status line to logs
	endpoint, endpointOK := support.GetStorageBucketDefaultEndpoint()
	accessKey, _ := support.GetStorageBucketAccessKeyId()
	secretKey, _ := support.GetStorageBucketSecretKey()
	bucket, bucketOK := support.GetStorageBucketName()
	prefix, _ := support.GetStorageBucketMnistDir()
	if !endpointOK {
		endpoint = ""
	}
	if !bucketOK {
		bucket = ""
	}

	// Create RWX PVC for shared dataset and pass the claim name to the notebook
	storageClass, err := support.GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	rwxPvc := support.CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		"20Gi",
		support.AccessModes(corev1.ReadWriteMany),
		support.StorageClassName(storageClass.Name),
	)

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export OPENSHIFT_API_URL='%s'; export NOTEBOOK_USER_TOKEN='%s'; "+
			"export NOTEBOOK_NAMESPACE='%s'; "+
			"export SHARED_PVC_NAME='%s'; "+
			"export AWS_DEFAULT_ENDPOINT='%s'; export AWS_ACCESS_KEY_ID='%s'; "+
			"export AWS_SECRET_ACCESS_KEY='%s'; export AWS_STORAGE_BUCKET='%s'; "+
			"export AWS_STORAGE_BUCKET_DATA_DIR='%s'; "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill boto3==1.34.162 git+https://github.com/opendatahub-io/kubeflow-sdk.git@main && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		support.GetOpenShiftApiUrl(test), userToken, namespace.Name, rwxPvc.Name,
		endpoint, accessKey, secretKey, bucket, prefix,
		osftNotebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

	// Create Notebook CR using the RWX PVC
	// For GPU testing, we use a larger container size to ensure sufficient resources
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, osftNotebookName, 0, rwxPvc, common.ContainerSizeMedium)

	// Cleanup
	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), support.TestTimeoutLong).Should(HaveLen(0))
	}()

	// Wait for the Notebook Pod and get pod/container names
	podName, containerName := trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	// Poll logs to check if the notebook execution completed successfully
	err = trainerutils.PollNotebookLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")
}

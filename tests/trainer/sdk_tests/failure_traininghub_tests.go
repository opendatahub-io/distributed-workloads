/*
Copyright 2026.

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
	failureNotebookName         = "failure_scenarios.ipynb"
	failureNotebookPath         = "resources/" + failureNotebookName
	torchrunFailureNotebookName = "torchrun_failure.ipynb"
	torchrunFailureNotebookPath = "resources/" + torchrunFailureNotebookName
)

// RunTrainingFailureScenariosTest verifies that training failures are properly
// propagated — when a training pod fails, the TrainJob should report a Failed
// condition, and the SDK client should surface that failure correctly.
func RunTrainingFailureScenariosTest(t *testing.T) {
	test := support.With(t)

	// Create a new test namespace
	namespace := test.NewTestNamespace()

	// Ensure Notebook ServiceAccount exists
	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	// RBACs setup
	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	trainerutils.CreateUserClusterRoleBindingForTrainerRuntimes(test, userName)

	// Create ConfigMap with notebook
	localPath := failureNotebookPath
	nb, err := os.ReadFile(localPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", localPath))
	installScript, err := os.ReadFile(installScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", installScriptPath))
	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{
		failureNotebookName:   nb,
		installKubeflowScript: installScript,
	})

	// Create RWX PVC required by the notebook pod template
	storageClass, err := support.GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	rwxPvc := support.CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		"5Gi",
		support.AccessModes(corev1.ReadWriteMany),
		support.StorageClassName(storageClass.Name),
	)

	sdkInstallExports := buildKubeflowInstallExports()
	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export OPENSHIFT_API_URL=%s; export NOTEBOOK_USER_TOKEN=%s; "+
			"export NOTEBOOK_NAMESPACE=%s; "+
			"export TRAINING_RUNTIME=%s; "+
			"%s"+
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		shellQuote(support.GetOpenShiftApiUrl(test)), shellQuote(userToken), shellQuote(namespace.Name),
		shellQuote(trainerutils.DefaultTrainingHubRuntimeCPU),
		sdkInstallExports,
		installKubeflowScript,
		failureNotebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

	// Create Notebook CR — CPU-only (ContainerSizeSmall), no GPUs needed
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, failureNotebookName, 0, rwxPvc, common.ContainerSizeSmall, common.GetRecommendedNotebookImageFromImageStream(test, common.NotebookImageStreamTrainingHubCPU))

	// Cleanup
	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), support.TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Wait for the Notebook Pod and get pod/container names
	podName, containerName := trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	// Poll logs — each scenario polls until the expected error appears in logs
	// and failure is confirmed via get_job()/get_job_events(), typically ~1 minute each
	err = trainerutils.PollNotebookLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")
}

// RunTorchrunTrainingFailureTest verifies that a torchrun training failure during
// the forward pass (OOM from extreme max_tokens_per_gpu) is properly propagated
// through the SDK. Requires GPU and S3/HuggingFace for model+data download.
func RunTorchrunTrainingFailureTest(t *testing.T) {
	test := support.With(t)

	// Create a new test namespace
	namespace := test.NewTestNamespace()

	// Ensure Notebook ServiceAccount exists
	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	// RBACs setup
	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	trainerutils.CreateUserClusterRoleBindingForTrainerRuntimes(test, userName)

	// Create ConfigMap with notebook
	localPath := torchrunFailureNotebookPath
	nb, err := os.ReadFile(localPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", localPath))
	installScript, err := os.ReadFile(installScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", installScriptPath))
	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{
		torchrunFailureNotebookName: nb,
		installKubeflowScript:       installScript,
	})

	// S3 configuration for model and dataset download
	endpoint, endpointOK := support.GetStorageBucketDefaultEndpoint()
	accessKey, _ := support.GetStorageBucketAccessKeyId()
	secretKey, _ := support.GetStorageBucketSecretKey()
	bucket, bucketOK := support.GetStorageBucketName()
	prefix, _ := support.GetStorageBucketSftDir()
	if !endpointOK {
		endpoint = ""
	}
	if !bucketOK {
		bucket = ""
	}

	// Create RWX PVC for shared dataset and model
	storageClass, err := support.GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	rwxPvc := support.CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		"20Gi",
		support.AccessModes(corev1.ReadWriteMany),
		support.StorageClassName(storageClass.Name),
	)

	sdkInstallExports := buildKubeflowInstallExports()
	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export OPENSHIFT_API_URL=%s; export NOTEBOOK_USER_TOKEN=%s; "+
			"export NOTEBOOK_NAMESPACE=%s; "+
			"export SHARED_PVC_NAME=%s; "+
			"export AWS_DEFAULT_ENDPOINT=%s; export AWS_ACCESS_KEY_ID=%s; "+
			"export AWS_SECRET_ACCESS_KEY=%s; "+
			"export AWS_STORAGE_BUCKET=%s; "+
			"export AWS_STORAGE_BUCKET_SFT_DIR=%s; "+
			"export TRAINING_RUNTIME=%s; "+
			"%s"+
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		shellQuote(support.GetOpenShiftApiUrl(test)), shellQuote(userToken), shellQuote(namespace.Name), shellQuote(rwxPvc.Name),
		shellQuote(endpoint), shellQuote(accessKey), shellQuote(secretKey), shellQuote(bucket), shellQuote(prefix),
		shellQuote(trainerutils.DefaultTrainingHubRuntimeCUDA),
		sdkInstallExports,
		installKubeflowScript,
		torchrunFailureNotebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

	// Create Notebook CR — GPU test, use ContainerSizeMedium
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, torchrunFailureNotebookName, 0, rwxPvc, common.ContainerSizeMedium, common.GetRecommendedNotebookImageFromImageStream(test, common.NotebookImageStreamTrainingHubCUDA))

	// Cleanup — use longer timeout for GPU tests due to large runtime images
	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), support.TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Wait for the Notebook Pod and get pod/container names
	podName, containerName := trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	// Poll logs — use double timeout for model download + training attempt
	err = trainerutils.PollNotebookLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")
}

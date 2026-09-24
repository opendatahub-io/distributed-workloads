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
	grpoNotebookName = "grpo.ipynb"
	grpoNotebookPath = "resources/" + grpoNotebookName
)

// Single-node Training with LORA_GRPO and TrainingHubTrainer
func RunGrpoTrainingHubTraining(t *testing.T, nnodes int) {
	test := support.With(t)

	// Create a new test namespace
	namespace := test.NewTestNamespace()

	// RBACs setup
	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	trainerutils.GrantTrainerUserAccess(test, userName, namespace.Name)

	// Create ConfigMap with notebook and install script
	localPath := grpoNotebookPath
	nb, err := os.ReadFile(localPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", localPath))

	installScript, err := os.ReadFile(installScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", installScriptPath))

	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{
		grpoNotebookName:      nb,
		installKubeflowScript: installScript,
	})

	// Build command with parameters and pinned deps, and print definitive status line to logs
	endpoint, endpointOK := support.GetStorageBucketDefaultEndpoint()
	accessKey, _ := support.GetStorageBucketAccessKeyId()
	secretKey, _ := support.GetStorageBucketSecretKey()
	bucket, bucketOK := support.GetStorageBucketName()
	prefix, _ := support.GetStorageBucketGrpoDir()
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

	env := append([]corev1.EnvVar{
		{Name: "IPYTHONDIR", Value: "/tmp/.ipython"},
		{Name: "OPENSHIFT_API_URL", Value: support.GetOpenShiftApiUrl(test)},
		{Name: "NOTEBOOK_USER_TOKEN", Value: userToken},
		{Name: "NOTEBOOK_NAMESPACE", Value: namespace.Name},
		{Name: "SHARED_PVC_NAME", Value: rwxPvc.Name},
		{Name: "AWS_DEFAULT_ENDPOINT", Value: endpoint},
		{Name: "AWS_ACCESS_KEY_ID", Value: accessKey},
		{Name: "AWS_SECRET_ACCESS_KEY", Value: secretKey},
		{Name: "AWS_STORAGE_BUCKET", Value: bucket},
		{Name: "AWS_STORAGE_BUCKET_GRPO_DIR", Value: prefix},
		{Name: "TRAINING_RUNTIME", Value: trainerutils.DefaultTrainingHubRuntimeCUDA},
		{Name: "NNODES", Value: fmt.Sprintf("%d", nnodes)},
		{Name: "GPU_TYPE", Value: "nvidia"},
	}, buildKubeflowInstallEnv()...)
	// Dump vLLM runtime logs before the status marker so they are retained for every run.
	shellCmd := fmt.Sprintf(
		"set -e; "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then notebook_status='SUCCESS'; else notebook_status='FAILURE'; fi; "+
			"echo '=== BEGIN vLLM runtime logs ==='; "+
			"find /opt/app-root/src/grpo-output -type f -name 'vllm-runtime.log' -print -exec cat {} + 2>&1 || true; "+
			"echo '=== END vLLM runtime logs ==='; "+
			"if [ ${notebook_status} = SUCCESS ]; then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; "+
			"sleep infinity",
		installKubeflowScript,
		grpoNotebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

	// GRPO requires more memory than SFT/LoRA due to vLLM running alongside training
	deployment := trainerutils.CreateNotebookDeployment(
		test,
		namespace,
		command,
		cm.Name,
		rwxPvc,
		support.ContainerSizeMedium,
		common.GetRecommendedNotebookImageFromImageStream(test, common.NotebookImageStreamTrainingHubCUDA),
		env,
	)

	// Cleanup - use longer timeout for GPU tests due to large runtime images
	defer func() {
		support.DeleteDeployment(test, namespace, deployment.Name)
	}()

	// Wait for the Deployment pod and get pod/container names
	podName, containerName := support.WaitForDeploymentPodRunning(test, namespace.Name, deployment.Name)

	// Poll runner logs to check if execution completed successfully
	// GRPO training takes longer than SFT/LoRA due to generation + RL loop
	err = support.PollPodLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	// Persist the complete notebook output before deferred cleanup deletes the pod.
	// This output includes the vLLM runtime log dumped before the status marker.
	logs := support.GetPodLog(test, namespace.Name, podName, corev1.PodLogOptions{Container: containerName})
	support.WriteToOutputDir(test, "grpo-notebook", support.Log, []byte(logs))
	test.Expect(err).ShouldNot(HaveOccurred(), "Deployment runner execution reported FAILURE")
}

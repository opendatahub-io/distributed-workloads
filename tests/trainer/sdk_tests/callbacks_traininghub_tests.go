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
	"strings"
	"testing"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"

	common "github.com/opendatahub-io/distributed-workloads/tests/common"
	support "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	callbackNotebookLogMarker = "[TH-CB]"

	callbacksMetricsLoggerFileName = "kubeflow_metrics_logger.py"
	callbacksMetricsLoggerPath     = "resources/callback/" + callbacksMetricsLoggerFileName
	trainingHubPyPIInstall         = "training-hub>=0.9.8"
	kubeflowSdkGitInstall          = "kubeflow[trainer] @ git+https://github.com/opendatahub-io/kubeflow-sdk.git@main"

	sftCallbacksNotebookName  = "sft_with_callbacks.ipynb"
	sftCallbacksNotebookPath  = "resources/callback/" + sftCallbacksNotebookName
	loraCallbacksNotebookName = "lora_sft_with_callbacks.ipynb"
	loraCallbacksNotebookPath = "resources/callback/" + loraCallbacksNotebookName
	osftCallbacksNotebookName = "osft_with_callbacks.ipynb"
	osftCallbacksNotebookPath = "resources/callback/" + osftCallbacksNotebookName
)

type callbacksTrainingConfig struct {
	notebookName        string
	notebookPath        string
	bucketDirEnvName    string
	getStorageBucketDir func() (string, bool)
}

func RunSftCallbacksTrainingHub(t *testing.T, nnodes int) {
	runCallbacksTrainingHub(t, nnodes, callbacksTrainingConfig{
		notebookName:        sftCallbacksNotebookName,
		notebookPath:        sftCallbacksNotebookPath,
		bucketDirEnvName:    "AWS_STORAGE_BUCKET_SFT_DIR",
		getStorageBucketDir: support.GetStorageBucketSftDir,
	})
}

func RunLoraCallbacksTrainingHub(t *testing.T, nnodes int) {
	runCallbacksTrainingHub(t, nnodes, callbacksTrainingConfig{
		notebookName:        loraCallbacksNotebookName,
		notebookPath:        loraCallbacksNotebookPath,
		bucketDirEnvName:    "AWS_STORAGE_BUCKET_LORA_DIR",
		getStorageBucketDir: support.GetStorageBucketLoraDir,
	})
}

func RunOsftCallbacksTrainingHub(t *testing.T, nnodes int) {
	runCallbacksTrainingHub(t, nnodes, callbacksTrainingConfig{
		notebookName:        osftCallbacksNotebookName,
		notebookPath:        osftCallbacksNotebookPath,
		bucketDirEnvName:    "AWS_STORAGE_BUCKET_OSFT_DIR",
		getStorageBucketDir: support.GetStorageBucketOsftDir,
	})
}

func runCallbacksTrainingHub(t *testing.T, nnodes int, cfg callbacksTrainingConfig) {
	test := support.With(t)

	namespace := test.NewTestNamespace()

	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	trainerutils.GrantTrainerUserAccess(test, userName, namespace.Name)

	nb, err := os.ReadFile(cfg.notebookPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", cfg.notebookPath))

	installScript, err := os.ReadFile(InstallScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", InstallScriptPath))

	metricsLogger, err := os.ReadFile(callbacksMetricsLoggerPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read callback helper: %s", callbacksMetricsLoggerPath))

	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{
		cfg.notebookName:               nb,
		InstallKubeflowScript:          installScript,
		callbacksMetricsLoggerFileName: metricsLogger,
	})

	endpoint, endpointOK := support.GetStorageBucketDefaultEndpoint()
	accessKey, _ := support.GetStorageBucketAccessKeyId()
	secretKey, _ := support.GetStorageBucketSecretKey()
	bucket, bucketOK := support.GetStorageBucketName()
	prefix, _ := cfg.getStorageBucketDir()
	if !endpointOK {
		endpoint = ""
	}
	if !bucketOK {
		bucket = ""
	}

	storageClass, err := support.GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	rwxPvc := support.CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		"20Gi",
		support.AccessModes(corev1.ReadWriteMany),
		support.StorageClassName(storageClass.Name),
	)

	sdkInstallExports := BuildKubeflowInstallExports()
	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export IPYTHONDIR='/tmp/.ipython'; "+
			"export OPENSHIFT_API_URL=%s; export NOTEBOOK_USER_TOKEN=%s; "+
			"export NOTEBOOK_NAMESPACE=%s; "+
			"export SHARED_PVC_NAME=%s; "+
			"export AWS_DEFAULT_ENDPOINT=%s; export AWS_ACCESS_KEY_ID=%s; "+
			"export AWS_SECRET_ACCESS_KEY=%s; "+
			"export AWS_STORAGE_BUCKET=%s; "+
			"export %s=%s; "+
			"export TRAINING_RUNTIME=%s; "+
			"export NNODES='%d'; "+
			"export GPU_TYPE='nvidia'; "+
			"%s"+
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages %s && "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages %s && "+
			"cp /opt/app-root/notebooks/%s /opt/app-root/src/%s && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		ShellQuote(support.GetOpenShiftApiUrl(test)), ShellQuote(userToken), ShellQuote(namespace.Name), ShellQuote(rwxPvc.Name),
		ShellQuote(endpoint), ShellQuote(accessKey), ShellQuote(secretKey), ShellQuote(bucket),
		cfg.bucketDirEnvName, ShellQuote(prefix),
		ShellQuote(trainerutils.DefaultTrainingHubRuntimeCUDA),
		nnodes,
		sdkInstallExports,
		ShellQuote(trainingHubPyPIInstall),
		ShellQuote(kubeflowSdkGitInstall),
		callbacksMetricsLoggerFileName,
		callbacksMetricsLoggerFileName,
		InstallKubeflowScript,
		cfg.notebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

	common.CreateNotebook(test, namespace, userToken, command, cm.Name, cfg.notebookName, 0, rwxPvc, common.ContainerSizeMedium, common.GetRecommendedNotebookImageFromImageStream(test, common.NotebookImageStreamTrainingHubCUDA))

	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), support.TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	podName, containerName := trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	err = support.PollNotebookLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")

	verifyCallbacksFiredInNotebookLogs(test, namespace.Name, podName, containerName)
}

func verifyCallbacksFiredInNotebookLogs(test support.Test, namespace, podName, containerName string) {
	test.T().Helper()

	var tail int64 = 5000
	logs := support.GetPodLog(test, namespace, podName, corev1.PodLogOptions{
		Container: containerName,
		TailLines: &tail,
	})

	test.Expect(logs).To(ContainSubstring(callbackNotebookLogMarker),
		"Expected unified callback log marker %q in notebook output", callbackNotebookLogMarker)
	test.Expect(logs).NotTo(ContainSubstring("WARNING: No [TH-CB] callback log lines found"),
		"Notebook reported callbacks did not fire during training")
	test.Expect(strings.Count(logs, callbackNotebookLogMarker)).To(BeNumerically(">", 0),
		"Expected at least one callback log line in notebook output")
}

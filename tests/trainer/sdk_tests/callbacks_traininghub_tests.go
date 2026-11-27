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
	"strconv"
	"strings"
	"testing"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

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
		{Name: cfg.bucketDirEnvName, Value: prefix},
		{Name: "TRAINING_RUNTIME", Value: trainerutils.DefaultTrainingHubRuntimeCUDA},
		{Name: "NNODES", Value: fmt.Sprintf("%d", nnodes)},
		{Name: "GPU_TYPE", Value: "nvidia"},
	}, buildKubeflowInstallEnv()...)
	shellCmd := fmt.Sprintf(
		"set -e; "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages %s && "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages %s && "+
			"cp /opt/app-root/notebooks/%s /opt/app-root/src/%s && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		ShellQuote(trainingHubPyPIInstall),
		ShellQuote(kubeflowSdkGitInstall),
		callbacksMetricsLoggerFileName,
		callbacksMetricsLoggerFileName,
		InstallKubeflowScript,
		cfg.notebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

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
	defer func() {
		support.DeleteDeployment(test, namespace, deployment.Name)
	}()

	podName, containerName := support.WaitForDeploymentPodRunning(test, namespace.Name, deployment.Name)

	err = support.PollPodLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Deployment runner execution reported FAILURE")

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

// JIT checkpoint tests: same Training Hub callback setup, plus a TrainJob that is
// suspended mid-run and resumed.

const (
	jitNotebookName     = "jit_checkpoint_traininghub.ipynb"
	jitNotebookPath     = "resources/callback/" + jitNotebookName
	jitStepProbeName    = "jit_step_probe.py"
	jitStepProbePath    = "resources/callback/" + jitStepProbeName
	jitResultPassMarker = "JIT_E2E_RESULT: PASS"
	// JIT checkpointing shipped in training-hub 0.11.0.
	jitTrainingHubInstall = "training-hub>=0.11.0"
)

// JIT checkpoint scenarios run by the notebook (see its first cell).
const (
	JitScenarioResume     = "resume"
	JitScenarioIncomplete = "incomplete"
)

// jitGpuCount returns how many GPUs the single training node uses: JIT_NUM_GPUS
// when set, otherwise requested, capped at the most GPUs any worker node has so
// a cluster with only single-GPU nodes still runs the test on 1 GPU.
func jitGpuCount(test support.Test, requested int) int {
	if value := strings.TrimSpace(os.Getenv("JIT_NUM_GPUS")); value != "" {
		parsed, err := strconv.Atoi(value)
		test.Expect(err).NotTo(HaveOccurred(), "JIT_NUM_GPUS must be an integer, got %q", value)
		test.Expect(parsed).To(BeNumerically(">", 0), "JIT_NUM_GPUS must be at least 1")
		requested = parsed
	}

	nodes, err := test.Client().Core().CoreV1().Nodes().List(test.Ctx(), metav1.ListOptions{
		LabelSelector: "!node-role.kubernetes.io/infra,node-role.kubernetes.io/worker",
	})
	test.Expect(err).NotTo(HaveOccurred())
	maxGpus := 0
	for _, node := range nodes.Items {
		gpus := node.Status.Allocatable[corev1.ResourceName(support.NVIDIA.ResourceLabel)]
		if count := int(gpus.Value()); count > maxGpus {
			maxGpus = count
		}
	}
	test.Expect(maxGpus).To(BeNumerically(">", 0), "No worker node has NVIDIA GPUs")

	if requested > maxGpus {
		test.T().Logf("Requested %d GPUs but the largest node has %d; running on %d", requested, maxGpus, maxGpus)
		return maxGpus
	}
	return requested
}

// RunJitCheckpointTrainingHub runs one JIT checkpoint scenario for a Training Hub
// algorithm (lora_sft, sft or osft) through TrainingHubTrainer on a single node.
// It uses requestedGpus GPUs (or JIT_NUM_GPUS), falling back to fewer when no
// node has that many.
func RunJitCheckpointTrainingHub(t *testing.T, algorithm, scenario string, requestedGpus int) {
	test := support.With(t)
	gpus := jitGpuCount(test, requestedGpus)
	test.T().Logf("Running JIT checkpoint %s/%s on 1 node with %d GPU(s)", algorithm, scenario, gpus)

	namespace := test.NewTestNamespace()

	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	trainerutils.GrantTrainerUserAccess(test, userName, namespace.Name)

	nb, err := os.ReadFile(jitNotebookPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", jitNotebookPath))

	installScript, err := os.ReadFile(InstallScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", InstallScriptPath))

	stepProbe, err := os.ReadFile(jitStepProbePath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read callback helper: %s", jitStepProbePath))

	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{
		jitNotebookName:       nb,
		InstallKubeflowScript: installScript,
		jitStepProbeName:      stepProbe,
	})

	endpoint, _ := support.GetStorageBucketDefaultEndpoint()
	accessKey, _ := support.GetStorageBucketAccessKeyId()
	secretKey, _ := support.GetStorageBucketSecretKey()
	bucket, _ := support.GetStorageBucketName()

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
		{Name: "AWS_STORAGE_BUCKET_JIT_DIR", Value: os.Getenv("AWS_STORAGE_BUCKET_JIT_DIR")},
		{Name: "TRAINING_RUNTIME", Value: trainerutils.DefaultTrainingHubRuntimeCUDA},
		{Name: "JIT_ALGORITHM", Value: algorithm},
		{Name: "JIT_SCENARIO", Value: scenario},
		{Name: "JIT_NUM_GPUS", Value: fmt.Sprintf("%d", gpus)},
	}, buildKubeflowInstallEnv()...)
	shellCmd := fmt.Sprintf(
		"set -e; "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages %s && "+
			"python -m pip install --quiet --no-cache-dir --break-system-packages %s && "+
			"cp /opt/app-root/notebooks/%s /opt/app-root/src/%s && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		ShellQuote(jitTrainingHubInstall),
		ShellQuote(kubeflowSdkGitInstall),
		jitStepProbeName,
		jitStepProbeName,
		InstallKubeflowScript,
		jitNotebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

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
	defer func() {
		support.DeleteDeployment(test, namespace, deployment.Name)
	}()

	podName, containerName := support.WaitForDeploymentPodRunning(test, namespace.Name, deployment.Name)

	// A resume scenario trains, suspends and resumes the same TrainJob.
	err = support.PollPodLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Deployment runner execution reported FAILURE")

	var tail int64 = 2000
	logs := support.GetPodLog(test, namespace.Name, podName, corev1.PodLogOptions{
		Container: containerName,
		TailLines: &tail,
	})
	test.Expect(logs).To(ContainSubstring(jitResultPassMarker),
		"Expected %q in notebook output for %s/%s on %d GPU(s)", jitResultPassMarker, algorithm, scenario, gpus)
}

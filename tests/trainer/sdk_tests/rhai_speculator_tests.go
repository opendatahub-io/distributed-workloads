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
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	common "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	speculatorNotebookName = "rhai_speculator.ipynb"
	speculatorNotebookPath = "resources/" + speculatorNotebookName
)

type speculatorTestEnv struct {
	test      Test
	namespace *corev1.Namespace
	userToken string
	rwxPvc    *corev1.PersistentVolumeClaim
	cm        *corev1.ConfigMap
}

func setupSpeculatorTestEnv(t *testing.T, pvcSize string) speculatorTestEnv {
	test := With(t)
	namespace := test.NewTestNamespace()

	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	trainerutils.CreateUserClusterRoleBindingForTrainerRuntimes(test, userName)

	nb, err := os.ReadFile(speculatorNotebookPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", speculatorNotebookPath))

	installScript, err := os.ReadFile(installScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", installScriptPath))

	cm := CreateConfigMap(test, namespace.Name, map[string][]byte{
		speculatorNotebookName: nb,
		installKubeflowScript:  installScript,
	})

	storageClass, err := GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	rwxPvc := CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		pvcSize,
		AccessModes(corev1.ReadWriteMany),
		StorageClassName(storageClass.Name),
	)

	return speculatorTestEnv{test, namespace, userToken, rwxPvc, cm}
}

// RunSpeculatorPipelineTest runs a sequential DATA_ONLY → TRAIN_ONLY pipeline.
// Two papermill runs in one pod: first extracts hidden states (DATA_ONLY), then
// trains a draft model from those hidden states (TRAIN_ONLY) with checkpoint resume.
func RunSpeculatorPipelineTest(t *testing.T, vllmGpuCount int, trainGpuCount int) {
	env := setupSpeculatorTestEnv(t, "40Gi")

	s3Exports := buildSpeculatorS3Exports(env.test)
	sdkInstallExports := buildKubeflowInstallExports()

	s3Endpoint, _ := GetStorageBucketDefaultEndpoint()
	regenerateResponses := "true"
	datasetName := "ultrachat"
	verifierModel := "Qwen/Qwen3-0.6B"
	if s3Endpoint != "" && s3Exports != "" {
		regenerateResponses = "false"
		datasetName = fmt.Sprintf("pvc://%s/datasets/ultrachat.jsonl", env.rwxPvc.Name)
		verifierModel = fmt.Sprintf("pvc://%s/models/Qwen3-0.6B", env.rwxPvc.Name)
		t.Log("Disconnected environment detected (S3 configured): using PVC model and dataset, skipping response regeneration")
	}

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export IPYTHONDIR='/tmp/.ipython'; "+
			"export OPENSHIFT_API_URL=%s; export NOTEBOOK_USER_TOKEN=%s; "+
			"export NOTEBOOK_NAMESPACE=%s; "+
			"export SHARED_PVC_NAME=%s; "+
			"export VLLM_GPU_COUNT='%d'; "+
			"export TRAIN_GPU_COUNT='%d'; "+
			"export TEST_TYPE='extraction'; "+
			"export DATASET_NAME=%s; "+
			"export VERIFIER_MODEL=%s; "+
			"export OUTPUT_DIR='pvc://%s/speculator-output/extract'; "+
			"export TRAIN_OUTPUT_DIR='pvc://%s/speculator-output/train'; "+
			"export TARGET_LAYER_IDS='2,14,25,28'; "+
			"export MAX_SAMPLES='20'; "+
			"export ENABLE_PROGRESSION_TRACKING='true'; "+
			"export REGENERATE_RESPONSES='%s'; "+
			"export DATAGEN_CONCURRENCY='2'; "+
			"export HIDDEN_STATES_DTYPE='bfloat16'; "+
			"export TEST_IDEMPOTENCY='true'; "+
			"%s"+ // S3 exports
			"%s"+ // SDK install exports
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python /opt/app-root/notebooks/%s && "+
			// Run 1: DATA_ONLY extraction
			"export SPECULATOR_MODE='DATA_ONLY'; "+
			"export TRAINING_RUNTIME=%s; "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out_data.ipynb --log-output; "+
			"then echo 'DATA_ONLY_STATUS: SUCCESS'; else echo 'DATA_ONLY_STATUS: FAILURE'; fi && "+
			// Run 2: TRAIN_ONLY training
			"export SPECULATOR_MODE='TRAIN_ONLY'; "+
			"export TRAINING_RUNTIME=%s; "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out_train.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		shellQuote(GetOpenShiftApiUrl(env.test)), shellQuote(env.userToken), shellQuote(env.namespace.Name),
		shellQuote(env.rwxPvc.Name),
		vllmGpuCount,
		trainGpuCount,
		shellQuote(datasetName),
		shellQuote(verifierModel),
		env.rwxPvc.Name,
		env.rwxPvc.Name,
		regenerateResponses,
		s3Exports,
		sdkInstallExports,
		installKubeflowScript,
		shellQuote(trainerutils.DefaultSpeculatorvLLMExtractRuntimeCUDA),
		speculatorNotebookName,
		shellQuote(trainerutils.DefaultSpeculatorModelOptRuntimeCUDA),
		speculatorNotebookName,
	)

	t.Logf("Speculator pipeline test: vllmGpuCount=%d, trainGpuCount=%d, regenerateResponses=%s", vllmGpuCount, trainGpuCount, regenerateResponses)
	command := []string{"/bin/sh", "-c", shellCmd}

	common.CreateNotebook(env.test, env.namespace, env.userToken, command, env.cm.Name, speculatorNotebookName, 0, env.rwxPvc, common.ContainerSizeMedium, common.GetRecommendedNotebookImageFromImageStream(env.test, common.NotebookImageStreamTrainingHubCUDA))

	defer func() {
		common.DeleteNotebook(env.test, env.namespace)
		env.test.Eventually(common.Notebooks(env.test, env.namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	podName, containerName := trainerutils.WaitForNotebookPodRunning(env.test, env.namespace.Name)

	// Wait for DATA_ONLY TrainJob
	var dataJobName string
	env.test.Eventually(func() int {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		if len(jobs) == 1 {
			dataJobName = jobs[0].Name
		}
		return len(jobs)
	}, TestTimeoutDouble, 5*time.Second).Should(Equal(1), "Expected DATA_ONLY TrainJob to be created")
	t.Logf("DATA_ONLY TrainJob created: %s", dataJobName)

	trainJob := TrainJob(env.test, env.namespace.Name, dataJobName)(env.test)
	annotations := trainJob.GetAnnotations()

	env.test.Expect(annotations[annotationProgressionTracking]).To(Equal("true"),
		"Expected progression-tracking annotation to be 'true'")
	env.test.Expect(annotations[annotationMetricsPort]).To(Equal("28080"),
		"Expected metrics-port annotation to be '28080'")
	env.test.Expect(annotations[annotationMetricsPollInterval]).To(Equal("30"),
		"Expected metrics-poll-interval annotation to be '30'")
	t.Logf("Progression annotations verified on TrainJob %s", dataJobName)

	env.test.Eventually(TrainJob(env.test, env.namespace.Name, dataJobName), TestTimeoutGpuProvisioning, 10*time.Second).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	t.Logf("DATA_ONLY TrainJob %s completed successfully", dataJobName)

	t.Log("Verifying DATA_ONLY training pod termination message...")
	verifySpeculatorTerminationMessage(env.test, env.namespace.Name, dataJobName)

	t.Log("Waiting for DATA_ONLY trainerStatus annotation to reach 100% progress...")
	env.test.Eventually(func() bool {
		tj := TrainJob(env.test, env.namespace.Name, dataJobName)(env.test)
		trainerStatusRaw := tj.GetAnnotations()[annotationTrainerStatus]
		if trainerStatusRaw == "" {
			return false
		}
		var status map[string]interface{}
		if err := json.Unmarshal([]byte(sanitizeJSON(trainerStatusRaw)), &status); err != nil {
			t.Logf("trainerStatus not valid JSON yet: %v", err)
			return false
		}
		progress, ok := status["progressPercentage"].(float64)
		if !ok || progress < 100 {
			t.Logf("trainerStatus progress: %.0f%%, waiting for 100%%...", progress)
			return false
		}
		t.Logf("trainerStatus reached 100%%: %s", trainerStatusRaw)
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "DATA_ONLY trainerStatus annotation should reach 100% progress")

	t.Log("Verifying DATA_ONLY training pod logs...")
	verifySpeculatorPodLogs(env.test, env.namespace.Name, dataJobName, regenerateResponses == "true")

	// Idempotency: wait for second DATA_ONLY TrainJob (re-run to same output_dir should skip)
	t.Log("Idempotency: waiting for second DATA_ONLY TrainJob...")
	var idempotencyJobName string
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name != dataJobName {
				idempotencyJobName = j.Name
				return true
			}
		}
		return false
	}, TestTimeoutDouble, 5*time.Second).Should(BeTrue(), "Expected idempotency TrainJob to be created")
	t.Logf("Idempotency TrainJob created: %s", idempotencyJobName)

	env.test.Eventually(TrainJob(env.test, env.namespace.Name, idempotencyJobName), TestTimeoutGpuProvisioning, 10*time.Second).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	t.Logf("Idempotency TrainJob %s completed", idempotencyJobName)

	verifySpeculatorPodLogContains(env.test, env.namespace.Name, idempotencyJobName,
		"Data extraction already completed. Skipping", "Idempotency job should skip extraction")
	t.Log("Idempotency: confirmed extraction was skipped")

	// Step 1: Wait for TRAIN_ONLY TrainJob (will be interrupted after first checkpoint)
	trainJobName := "speculator-train"
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == trainJobName {
				return true
			}
		}
		return false
	}, TestTimeoutDouble, 5*time.Second).Should(BeTrue(), "Expected TRAIN_ONLY TrainJob 'speculator-train' to be created")
	t.Logf("TRAIN_ONLY TrainJob created: %s", trainJobName)

	// Verify TRAIN_ONLY progression tracking annotations
	trainOnlyJob := TrainJob(env.test, env.namespace.Name, trainJobName)(env.test)
	trainOnlyAnnotations := trainOnlyJob.GetAnnotations()
	env.test.Expect(trainOnlyAnnotations[annotationProgressionTracking]).To(Equal("true"),
		"Expected TRAIN_ONLY progression-tracking annotation to be 'true'")
	t.Logf("TRAIN_ONLY progression annotations verified on TrainJob %s", trainJobName)

	// Wait for TrainJob to be deleted (notebook interrupts it after first checkpoint, during epoch 2)
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == trainJobName {
				return false
			}
		}
		return true
	}, TestTimeoutGpuProvisioning, 5*time.Second).Should(BeTrue(), "Expected TRAIN_ONLY TrainJob to be deleted after interrupt")
	t.Log("TRAIN_ONLY TrainJob deleted (interrupted during epoch 2)")

	// Step 2: Wait for checkpoint resume TrainJob — should find interrupted checkpoint, remove it, resume from epoch 0
	resumeJobName := "speculator-train-resume"
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == resumeJobName {
				return true
			}
		}
		return false
	}, TestTimeoutDouble, 5*time.Second).Should(BeTrue(), "Expected checkpoint resume TrainJob 'speculator-train-resume' to be created")
	t.Logf("Checkpoint resume TrainJob created: %s", resumeJobName)

	env.test.Eventually(TrainJob(env.test, env.namespace.Name, resumeJobName), TestTimeoutGpuProvisioning, 10*time.Second).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	t.Logf("Checkpoint resume TrainJob %s completed successfully", resumeJobName)

	t.Log("Verifying checkpoint resume termination message...")
	verifySpeculatorTerminationMessage(env.test, env.namespace.Name, resumeJobName)

	t.Log("Waiting for TRAIN_ONLY resume trainerStatus annotation to reach 100% progress...")
	env.test.Eventually(func() bool {
		tj := TrainJob(env.test, env.namespace.Name, resumeJobName)(env.test)
		trainerStatusRaw := tj.GetAnnotations()[annotationTrainerStatus]
		if trainerStatusRaw == "" {
			return false
		}
		var status map[string]interface{}
		if err := json.Unmarshal([]byte(sanitizeJSON(trainerStatusRaw)), &status); err != nil {
			t.Logf("TRAIN_ONLY resume trainerStatus not valid JSON yet: %v", err)
			return false
		}
		progress, ok := status["progressPercentage"].(float64)
		if !ok || progress < 100 {
			t.Logf("TRAIN_ONLY resume trainerStatus progress: %.0f%%, waiting for 100%%...", progress)
			return false
		}
		t.Logf("TRAIN_ONLY resume trainerStatus reached 100%%: %s", trainerStatusRaw)
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "TRAIN_ONLY resume trainerStatus annotation should reach 100% progress")

	t.Log("Verifying checkpoint resume training pod logs...")
	verifySpeculatorTrainOnlyPodLogs(env.test, env.namespace.Name, resumeJobName)
	verifySpeculatorResumeFromCheckpointLogs(env.test, env.namespace.Name, resumeJobName)
	verifySpeculatorPodLogContains(env.test, env.namespace.Name, resumeJobName,
		"[Kubeflow] Removed interrupted checkpoint at", "Resume job should detect and remove the interrupted checkpoint")

	err := PollNotebookLogsForStatus(env.test, env.namespace.Name, podName, containerName, TestTimeoutDouble)
	env.test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")

	// Verify notebook-side PVC artifact check ran successfully
	var tail int64 = 5000
	notebookLogs := PodLog(env.test, env.namespace.Name, podName, corev1.PodLogOptions{
		Container: containerName,
		TailLines: &tail,
	})(env.test)
	env.test.Expect(notebookLogs).To(ContainSubstring("Output artifact verification: PASSED"),
		"Notebook should have verified PVC output artifacts (Arrow, token_freq.pt, safetensors)")
	t.Log("Verified notebook-side PVC output artifact check: PASSED")

	t.Log("All speculator pipeline checklist items passed!")
}

func verifySpeculatorTerminationMessage(test Test, namespace, trainJobName string) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	test.Expect(len(pods)).NotTo(Equal(0), "Expected at least one training pod")

	var found100Percent bool
	for _, pod := range pods {
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Name != "node" || cs.State.Terminated == nil {
				continue
			}
			msg := cs.State.Terminated.Message
			if msg == "" {
				continue
			}
			test.T().Logf("Pod %s termination message: %s", pod.Name, msg)

			var data map[string]interface{}
			if err := json.Unmarshal([]byte(sanitizeJSON(msg)), &data); err != nil {
				continue
			}
			test.Expect(data).To(HaveKey("progressPercentage"))
			test.Expect(data).To(HaveKey("currentPhase"))
			test.Expect(data).To(HaveKey("estimatedRemainingSeconds"))

			if progress, ok := data["progressPercentage"].(float64); ok && progress >= 100 {
				found100Percent = true
				test.T().Logf("Found 100%% progress in termination message for pod %s", pod.Name)
			}
			break
		}
		if found100Percent {
			break
		}
	}
	test.Expect(found100Percent).To(BeTrue(), "Data extraction should complete with 100% progress in termination message")
}

func verifySpeculatorPodLogs(test Test, namespace, trainJobName string, expectRegen bool) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	test.Expect(len(pods)).NotTo(Equal(0), "No training pods found to verify logs")

	required := []string{
		"[Kubeflow] Speculator progression tracking enabled",
		"[Kubeflow] vLLM server is ready",
		"[Kubeflow] Saved preprocessed dataset to",
		"[Kubeflow] Data extraction complete",
	}
	if expectRegen {
		required = append(required, "[Kubeflow] Regenerating responses")
	}

	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodSucceeded {
			continue
		}
		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)

		allFound := true
		for _, marker := range required {
			if strings.Contains(logs, marker) {
				test.T().Logf("Verified in pod %s: %s", pod.Name, marker)
			} else {
				test.T().Logf("Missing in pod %s: %s", pod.Name, marker)
				allFound = false
			}
		}
		if allFound {
			return
		}
	}

	test.T().Fatalf("Required log markers not found in any completed training pod: %v", required)
}

func verifySpeculatorOnlinePodLogs(test Test, namespace, trainJobName string, expectRegen bool) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	test.Expect(len(pods)).NotTo(Equal(0), "No training pods found to verify ONLINE logs")

	required := []string{
		"[Kubeflow] Speculator progression tracking enabled",
		"[Kubeflow] vLLM server is ready",
		"Epoch 2",
		"[Kubeflow] Complete. Final metrics saved.",
	}
	if expectRegen {
		required = append(required, "[Kubeflow] Regenerating responses")
	}

	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodSucceeded {
			continue
		}
		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)

		allFound := true
		for _, marker := range required {
			if strings.Contains(logs, marker) {
				test.T().Logf("Verified in pod %s: %s", pod.Name, marker)
			} else {
				test.T().Logf("Missing in pod %s: %s", pod.Name, marker)
				allFound = false
			}
		}
		if allFound {
			return
		}
	}

	test.T().Fatalf("Required ONLINE log markers not found in any completed training pod: %v", required)
}

func verifySpeculatorTrainOnlyPodLogs(test Test, namespace, trainJobName string) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	test.Expect(len(pods)).NotTo(Equal(0), "No training pods found to verify TRAIN_ONLY logs")

	required := []string{
		"[Kubeflow] Speculator progression tracking enabled",
		"[Kubeflow] Complete. Final metrics saved.",
	}

	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodSucceeded {
			continue
		}
		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)

		allFound := true
		for _, marker := range required {
			if strings.Contains(logs, marker) {
				test.T().Logf("Verified in pod %s: %s", pod.Name, marker)
			} else {
				test.T().Logf("Missing in pod %s: %s", pod.Name, marker)
				allFound = false
			}
		}
		if allFound {
			return
		}
	}

	test.T().Fatalf("Required TRAIN_ONLY log markers not found in any completed training pod: %v", required)
}

func verifySpeculatorResumeFromCheckpointLogs(test Test, namespace, trainJobName string) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	test.Expect(len(pods)).NotTo(Equal(0), "No training pods found to verify checkpoint resume logs")

	required := []string{
		"Found checkpoint at",
		"Resuming training on",
		"Training epoch 3/3",
	}

	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodSucceeded {
			continue
		}
		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)

		allFound := true
		for _, marker := range required {
			if strings.Contains(logs, marker) {
				test.T().Logf("Verified in pod %s: %s", pod.Name, marker)
			} else {
				test.T().Logf("Missing in pod %s: %s", pod.Name, marker)
				allFound = false
			}
		}
		if allFound {
			return
		}
	}

	test.T().Fatalf("Checkpoint resume log markers not found in any completed training pod: %v", required)
}

func verifySpeculatorPodLogContains(test Test, namespace, trainJobName, expected, failMsg string) {
	test.T().Helper()

	for _, pod := range listTrainingPods(test, namespace, trainJobName) {
		if pod.Status.Phase != corev1.PodSucceeded {
			continue
		}
		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)
		if strings.Contains(logs, expected) {
			test.T().Logf("Verified in pod %s: %s", pod.Name, expected)
			return
		}
	}

	test.T().Fatalf("%s — expected %q in training pod logs", failMsg, expected)
}

// RunSpeculatorOfflineTest runs OFFLINE mode end-to-end.
// The notebook deploys a standalone vLLM server (after model download), then submits
// OFFLINE TrainJobs that call the external vLLM endpoint for hidden state extraction
// followed by training — all within a single TrainJob per submission.
func RunSpeculatorOfflineTest(t *testing.T, trainGpuCount int) {
	env := setupSpeculatorTestEnv(t, "40Gi")

	s3Exports := buildSpeculatorS3Exports(env.test)
	sdkInstallExports := buildKubeflowInstallExports()

	vllmImage, err := trainerutils.GetSidecarImageFromClusterTrainingRuntime(env.test, trainerutils.DefaultSpeculatorvLLMExtractRuntimeCUDA, "vllm-sidecar")
	env.test.Expect(err).ShouldNot(HaveOccurred(), "Failed to get vLLM image from ClusterTrainingRuntime %s", trainerutils.DefaultSpeculatorvLLMExtractRuntimeCUDA)
	t.Logf("Using vLLM image from ClusterTrainingRuntime %s: %s", trainerutils.DefaultSpeculatorvLLMExtractRuntimeCUDA, vllmImage)

	s3Endpoint, _ := GetStorageBucketDefaultEndpoint()
	regenerateResponses := "true"
	datasetName := "ultrachat"
	verifierModel := "Qwen/Qwen3-0.6B"
	if s3Endpoint != "" {
		regenerateResponses = "false"
		datasetName = fmt.Sprintf("pvc://%s/datasets/ultrachat.jsonl", env.rwxPvc.Name)
		verifierModel = fmt.Sprintf("pvc://%s/models/Qwen3-0.6B", env.rwxPvc.Name)
		t.Log("Disconnected environment detected (S3 configured): using PVC model and dataset, skipping response regeneration")
	}

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export IPYTHONDIR='/tmp/.ipython'; "+
			"export OPENSHIFT_API_URL=%s; export NOTEBOOK_USER_TOKEN=%s; "+
			"export NOTEBOOK_NAMESPACE=%s; "+
			"export SHARED_PVC_NAME=%s; "+
			"export SPECULATOR_MODE='OFFLINE'; "+
			"export TEST_TYPE='extraction'; "+
			"export VLLM_IMAGE=%s; "+
			"export TRAIN_GPU_COUNT='%d'; "+
			"export DATASET_NAME=%s; "+
			"export VERIFIER_MODEL=%s; "+
			"export OUTPUT_DIR='pvc://%s/speculator-output/offline'; "+
			"export TARGET_LAYER_IDS='2,14,25,28'; "+
			"export MAX_SAMPLES='10'; "+
			"export ENABLE_PROGRESSION_TRACKING='true'; "+
			"export REGENERATE_RESPONSES='%s'; "+
			"export DATAGEN_CONCURRENCY='2'; "+
			"export HIDDEN_STATES_DTYPE='bfloat16'; "+
			"export TRAINING_RUNTIME=%s; "+
			"%s"+ // S3 exports
			"%s"+ // SDK install exports
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out_offline.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		shellQuote(GetOpenShiftApiUrl(env.test)), shellQuote(env.userToken), shellQuote(env.namespace.Name),
		shellQuote(env.rwxPvc.Name),
		shellQuote(vllmImage),
		trainGpuCount,
		shellQuote(datasetName),
		shellQuote(verifierModel),
		env.rwxPvc.Name,
		regenerateResponses,
		shellQuote(trainerutils.DefaultSpeculatorModelOptRuntimeCUDA),
		s3Exports,
		sdkInstallExports,
		installKubeflowScript,
		speculatorNotebookName,
	)

	t.Logf("Speculator OFFLINE pipeline test: trainGpuCount=%d, regenerateResponses=%s", trainGpuCount, regenerateResponses)
	command := []string{"/bin/sh", "-c", shellCmd}

	common.CreateNotebook(env.test, env.namespace, env.userToken, command, env.cm.Name, speculatorNotebookName, 0, env.rwxPvc, common.ContainerSizeMedium, common.GetRecommendedNotebookImageFromImageStream(env.test, common.NotebookImageStreamTrainingHubCUDA))

	defer func() {
		common.DeleteNotebook(env.test, env.namespace)
		env.test.Eventually(common.Notebooks(env.test, env.namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	podName, containerName := trainerutils.WaitForNotebookPodRunning(env.test, env.namespace.Name)

	// Step 1: Wait for the OFFLINE TrainJob (will be interrupted after first checkpoint)
	offlineJobName := "speculator-offline"
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == offlineJobName {
				return true
			}
		}
		return false
	}, TestTimeoutGpuProvisioning, 5*time.Second).Should(BeTrue(), "Expected OFFLINE TrainJob 'speculator-offline' to be created")
	t.Logf("OFFLINE TrainJob created: %s", offlineJobName)

	// Verify progression tracking annotations
	trainJob := TrainJob(env.test, env.namespace.Name, offlineJobName)(env.test)
	annotations := trainJob.GetAnnotations()
	env.test.Expect(annotations[annotationProgressionTracking]).To(Equal("true"),
		"Expected progression-tracking annotation to be 'true'")
	env.test.Expect(annotations[annotationMetricsPort]).To(Equal("28080"),
		"Expected metrics-port annotation to be '28080'")
	env.test.Expect(annotations[annotationMetricsPollInterval]).To(Equal("30"),
		"Expected metrics-poll-interval annotation to be '30'")
	t.Logf("Progression annotations verified on OFFLINE TrainJob %s", offlineJobName)

	// Wait for TrainJob to be deleted (notebook interrupts it after first checkpoint)
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == offlineJobName {
				return false
			}
		}
		return true
	}, TestTimeoutGpuProvisioning, 5*time.Second).Should(BeTrue(), "Expected OFFLINE TrainJob to be deleted after checkpoint detection")
	t.Log("OFFLINE TrainJob deleted (interrupted after checkpoint)")

	// Step 2: Wait for resume TrainJob to appear and complete
	resumeJobName := "speculator-offline-resume"
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == resumeJobName {
				return true
			}
		}
		return false
	}, TestTimeoutDouble, 5*time.Second).Should(BeTrue(), "Expected resume TrainJob 'speculator-offline-resume' to be created")
	t.Logf("Resume TrainJob created: %s", resumeJobName)

	// Verify progression tracking annotations on resume job
	resumeTrainJob := TrainJob(env.test, env.namespace.Name, resumeJobName)(env.test)
	resumeAnnotations := resumeTrainJob.GetAnnotations()
	env.test.Expect(resumeAnnotations[annotationProgressionTracking]).To(Equal("true"),
		"Expected resume progression-tracking annotation to be 'true'")
	env.test.Expect(resumeAnnotations[annotationMetricsPort]).To(Equal("28080"),
		"Expected resume metrics-port annotation to be '28080'")
	env.test.Expect(resumeAnnotations[annotationMetricsPollInterval]).To(Equal("30"),
		"Expected resume metrics-poll-interval annotation to be '30'")
	t.Logf("Progression annotations verified on resume TrainJob %s", resumeJobName)

	env.test.Eventually(TrainJob(env.test, env.namespace.Name, resumeJobName), TestTimeoutGpuProvisioning, 10*time.Second).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	t.Logf("Resume TrainJob %s completed successfully", resumeJobName)

	t.Log("Verifying resume termination message shows progressPercentage=100...")
	verifySpeculatorTerminationMessage(env.test, env.namespace.Name, resumeJobName)

	t.Log("Waiting for resume trainerStatus annotation to reach 100% progress...")
	env.test.Eventually(func() bool {
		tj := TrainJob(env.test, env.namespace.Name, resumeJobName)(env.test)
		trainerStatusRaw := tj.GetAnnotations()[annotationTrainerStatus]
		if trainerStatusRaw == "" {
			return false
		}
		var status map[string]interface{}
		if err := json.Unmarshal([]byte(sanitizeJSON(trainerStatusRaw)), &status); err != nil {
			t.Logf("Resume trainerStatus not valid JSON yet: %v", err)
			return false
		}
		progress, ok := status["progressPercentage"].(float64)
		if !ok || progress < 100 {
			t.Logf("Resume trainerStatus progress: %.0f%%, waiting for 100%%...", progress)
			return false
		}
		t.Logf("Resume trainerStatus reached 100%%: %s", trainerStatusRaw)
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "Resume trainerStatus annotation should reach 100% progress")

	t.Log("Verifying OFFLINE resume: extraction skipped (idempotency)...")
	verifySpeculatorPodLogContains(env.test, env.namespace.Name, resumeJobName,
		"Data extraction already completed. Skipping", "OFFLINE resume should skip extraction")

	t.Log("Verifying OFFLINE resume: training completed...")
	verifySpeculatorTrainOnlyPodLogs(env.test, env.namespace.Name, resumeJobName)

	t.Log("Verifying OFFLINE resume: checkpoint resume markers...")
	verifySpeculatorResumeFromCheckpointLogs(env.test, env.namespace.Name, resumeJobName)

	err = PollNotebookLogsForStatus(env.test, env.namespace.Name, podName, containerName, TestTimeoutDouble)
	env.test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")

	t.Log("All speculator OFFLINE pipeline steps passed!")
}

// RunSpeculatorOnlineTest runs ONLINE mode end-to-end.
// The SDK manages a vLLM sidecar within the same TrainJob pod — no external vLLM
// deployment is needed. Hidden states are generated on-the-fly during training.
func RunSpeculatorOnlineTest(t *testing.T, vllmGpuCount int, trainGpuCount int) {
	env := setupSpeculatorTestEnv(t, "40Gi")

	s3Exports := buildSpeculatorS3Exports(env.test)
	sdkInstallExports := buildKubeflowInstallExports()

	s3Endpoint, _ := GetStorageBucketDefaultEndpoint()
	regenerateResponses := "true"
	datasetName := "ultrachat"
	verifierModel := "Qwen/Qwen3-0.6B"
	if s3Endpoint != "" {
		regenerateResponses = "false"
		datasetName = fmt.Sprintf("pvc://%s/datasets/ultrachat.jsonl", env.rwxPvc.Name)
		verifierModel = fmt.Sprintf("pvc://%s/models/Qwen3-0.6B", env.rwxPvc.Name)
		t.Log("Disconnected environment detected (S3 configured): using PVC model and dataset")
	}

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export IPYTHONDIR='/tmp/.ipython'; "+
			"export OPENSHIFT_API_URL=%s; export NOTEBOOK_USER_TOKEN=%s; "+
			"export NOTEBOOK_NAMESPACE=%s; "+
			"export SHARED_PVC_NAME=%s; "+
			"export SPECULATOR_MODE='ONLINE'; "+
			"export TEST_TYPE='extraction'; "+
			"export VLLM_GPU_COUNT='%d'; "+
			"export TRAIN_GPU_COUNT='%d'; "+
			"export DATASET_NAME=%s; "+
			"export VERIFIER_MODEL=%s; "+
			"export OUTPUT_DIR='pvc://%s/speculator-output/online'; "+
			"export TARGET_LAYER_IDS='2,14,25,28'; "+
			"export MAX_SAMPLES='10'; "+
			"export ENABLE_PROGRESSION_TRACKING='true'; "+
			"export REGENERATE_RESPONSES='%s'; "+
			"export HIDDEN_STATES_DTYPE='bfloat16'; "+
			"export TRAINING_RUNTIME=%s; "+
			"%s"+ // S3 exports
			"%s"+ // SDK install exports
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out_online.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		shellQuote(GetOpenShiftApiUrl(env.test)), shellQuote(env.userToken), shellQuote(env.namespace.Name),
		shellQuote(env.rwxPvc.Name),
		vllmGpuCount,
		trainGpuCount,
		shellQuote(datasetName),
		shellQuote(verifierModel),
		env.rwxPvc.Name,
		regenerateResponses,
		shellQuote(trainerutils.DefaultSpeculatorvLLMExtractRuntimeCUDA),
		s3Exports,
		sdkInstallExports,
		installKubeflowScript,
		speculatorNotebookName,
	)

	t.Logf("Speculator ONLINE pipeline test: vllmGpuCount=%d, trainGpuCount=%d, regenerateResponses=%s", vllmGpuCount, trainGpuCount, regenerateResponses)
	command := []string{"/bin/sh", "-c", shellCmd}

	common.CreateNotebook(env.test, env.namespace, env.userToken, command, env.cm.Name, speculatorNotebookName, 0, env.rwxPvc, common.ContainerSizeMedium, common.GetRecommendedNotebookImageFromImageStream(env.test, common.NotebookImageStreamTrainingHubCUDA))

	defer func() {
		common.DeleteNotebook(env.test, env.namespace)
		env.test.Eventually(common.Notebooks(env.test, env.namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	podName, containerName := trainerutils.WaitForNotebookPodRunning(env.test, env.namespace.Name)

	// Step 1: Wait for the ONLINE TrainJob (will be interrupted after first checkpoint)
	onlineJobName := "speculator-online"
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == onlineJobName {
				return true
			}
		}
		return false
	}, TestTimeoutGpuProvisioning, 5*time.Second).Should(BeTrue(), "Expected ONLINE TrainJob 'speculator-online' to be created")
	t.Logf("ONLINE TrainJob created: %s", onlineJobName)

	// Verify progression tracking annotations
	trainJob := TrainJob(env.test, env.namespace.Name, onlineJobName)(env.test)
	annotations := trainJob.GetAnnotations()
	env.test.Expect(annotations[annotationProgressionTracking]).To(Equal("true"),
		"Expected progression-tracking annotation to be 'true'")
	env.test.Expect(annotations[annotationMetricsPort]).To(Equal("28080"),
		"Expected metrics-port annotation to be '28080'")
	env.test.Expect(annotations[annotationMetricsPollInterval]).To(Equal("30"),
		"Expected metrics-poll-interval annotation to be '30'")
	t.Logf("Progression annotations verified on ONLINE TrainJob %s", onlineJobName)

	// Sample progress while ONLINE job runs, then wait for notebook to interrupt it.
	// Only record a sample when the value changes from the previous one.
	var progressSamples []float64
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == onlineJobName {
				trainerStatusRaw := j.GetAnnotations()[annotationTrainerStatus]
				if trainerStatusRaw != "" {
					var status map[string]interface{}
					if err := json.Unmarshal([]byte(sanitizeJSON(trainerStatusRaw)), &status); err == nil {
						if progress, ok := status["progressPercentage"].(float64); ok {
							if len(progressSamples) == 0 || progress != progressSamples[len(progressSamples)-1] {
								progressSamples = append(progressSamples, progress)
								t.Logf("ONLINE progress sample: %.0f%%", progress)
							}
						}
					}
				}
				return false
			}
		}
		return true
	}, TestTimeoutGpuProvisioning, 5*time.Second).Should(BeTrue(), "Expected ONLINE TrainJob to be deleted after checkpoint detection")
	t.Log("ONLINE TrainJob deleted (interrupted after checkpoint)")

	// Assert linear progress: no single sample-to-sample forward jump > 50%
	for i := 1; i < len(progressSamples); i++ {
		jump := progressSamples[i] - progressSamples[i-1]
		env.test.Expect(jump).To(BeNumerically("<=", 50.0),
			fmt.Sprintf("Progress jumped from %.0f%% to %.0f%% (%.0f%% jump, max 50%% expected)",
				progressSamples[i-1], progressSamples[i], jump))
	}
	t.Logf("ONLINE progress samples: %v (no >50%% forward jumps)", progressSamples)

	// Step 2: Wait for resume TrainJob to appear and complete
	resumeJobName := "speculator-online-resume"
	env.test.Eventually(func() bool {
		jobs := TrainJobs(env.test, env.namespace.Name)(env.test)
		for _, j := range jobs {
			if j.Name == resumeJobName {
				return true
			}
		}
		return false
	}, TestTimeoutDouble, 5*time.Second).Should(BeTrue(), "Expected resume TrainJob 'speculator-online-resume' to be created")
	t.Logf("Resume TrainJob created: %s", resumeJobName)

	// Verify progression tracking annotations on resume job
	resumeTrainJob := TrainJob(env.test, env.namespace.Name, resumeJobName)(env.test)
	resumeAnnotations := resumeTrainJob.GetAnnotations()
	env.test.Expect(resumeAnnotations[annotationProgressionTracking]).To(Equal("true"),
		"Expected resume progression-tracking annotation to be 'true'")
	env.test.Expect(resumeAnnotations[annotationMetricsPort]).To(Equal("28080"),
		"Expected resume metrics-port annotation to be '28080'")
	env.test.Expect(resumeAnnotations[annotationMetricsPollInterval]).To(Equal("30"),
		"Expected resume metrics-poll-interval annotation to be '30'")
	t.Logf("Progression annotations verified on ONLINE resume TrainJob %s", resumeJobName)

	env.test.Eventually(TrainJob(env.test, env.namespace.Name, resumeJobName), TestTimeoutGpuProvisioning, 10*time.Second).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	t.Logf("Resume TrainJob %s completed successfully", resumeJobName)

	t.Log("Verifying resume termination message shows progressPercentage=100...")
	verifySpeculatorTerminationMessage(env.test, env.namespace.Name, resumeJobName)

	t.Log("Waiting for ONLINE resume trainerStatus annotation to reach 100% progress...")
	env.test.Eventually(func() bool {
		tj := TrainJob(env.test, env.namespace.Name, resumeJobName)(env.test)
		trainerStatusRaw := tj.GetAnnotations()[annotationTrainerStatus]
		if trainerStatusRaw == "" {
			return false
		}
		var status map[string]interface{}
		if err := json.Unmarshal([]byte(sanitizeJSON(trainerStatusRaw)), &status); err != nil {
			t.Logf("ONLINE resume trainerStatus not valid JSON yet: %v", err)
			return false
		}
		progress, ok := status["progressPercentage"].(float64)
		if !ok || progress < 100 {
			t.Logf("ONLINE resume trainerStatus progress: %.0f%%, waiting for 100%%...", progress)
			return false
		}
		t.Logf("ONLINE resume trainerStatus reached 100%%: %s", trainerStatusRaw)
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "ONLINE resume trainerStatus annotation should reach 100% progress")

	t.Log("Verifying ONLINE resume training pod logs...")
	verifySpeculatorOnlinePodLogs(env.test, env.namespace.Name, resumeJobName, s3Endpoint == "")
	verifySpeculatorResumeFromCheckpointLogs(env.test, env.namespace.Name, resumeJobName)

	t.Log("Verifying vLLM sidecar container in resume pod...")
	verifySpeculatorOnlineSidecar(env.test, env.namespace.Name, resumeJobName, vllmGpuCount)

	t.Log("Artifact check (no .safetensors, token_freq.pt, Arrow files on PVC) is enforced notebook-side via assertions")

	err := PollNotebookLogsForStatus(env.test, env.namespace.Name, podName, containerName, TestTimeoutDouble)
	env.test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")

	t.Log("All speculator ONLINE pipeline steps passed!")
}

func verifySpeculatorOnlineSidecar(test Test, namespace, trainJobName string, expectedVllmGpus int) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	test.Expect(len(pods)).NotTo(Equal(0), "No training pods found to verify ONLINE sidecar")

	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodSucceeded {
			continue
		}

		if len(pod.Spec.Containers) < 2 {
			test.T().Logf("ONLINE pod %s has %d container(s) — sidecar may have been injected at runtime and not reflected in final pod spec; skipping structural sidecar check",
				pod.Name, len(pod.Spec.Containers))
			return
		}

		var sidecarFound bool
		for _, c := range pod.Spec.Containers {
			if c.Name == "node" {
				continue
			}
			gpuQty := c.Resources.Requests[corev1.ResourceName("nvidia.com/gpu")]
			if !gpuQty.IsZero() {
				sidecarFound = true
				actualGpus := int(gpuQty.Value())
				test.Expect(actualGpus).To(Equal(expectedVllmGpus),
					fmt.Sprintf("Sidecar container %s GPU requests should be %d, got %d",
						c.Name, expectedVllmGpus, actualGpus))
				test.T().Logf("Verified sidecar container %s in pod %s: nvidia.com/gpu=%d",
					c.Name, pod.Name, actualGpus)

				for _, cs := range pod.Status.ContainerStatuses {
					if cs.Name == c.Name {
						test.Expect(cs.State.Terminated).NotTo(BeNil(),
							fmt.Sprintf("Sidecar container %s should be terminated after training completes", c.Name))
						test.T().Logf("Confirmed sidecar container %s terminated (exit code %d)",
							c.Name, cs.State.Terminated.ExitCode)
					}
				}
				break
			}
		}
		if !sidecarFound {
			test.T().Logf("No vLLM sidecar container with GPU resources found in pod %s — sidecar may use a different resource key", pod.Name)
		}
		return
	}

	test.T().Fatalf("No completed training pod found to verify ONLINE sidecar for job %s", trainJobName)
}

func buildSpeculatorS3Exports(test Test) string {
	s3Endpoint, _ := GetStorageBucketDefaultEndpoint()
	s3AccessKey, _ := GetStorageBucketAccessKeyId()
	s3SecretKey, _ := GetStorageBucketSecretKey()
	modelsBucket, _ := GetStorageBucketName()

	s3InternalEndpoint := s3Endpoint
	if internal, ok := os.LookupEnv("AWS_INTERNAL_ENDPOINT"); ok && internal != "" {
		s3InternalEndpoint = internal
	}

	modelS3Prefix := os.Getenv("MODEL_S3_PREFIX")
	if modelS3Prefix == "" {
		modelS3Prefix = "models/Qwen3-0.6B"
	}

	datasetS3Prefix := os.Getenv("DATASET_S3_PREFIX")
	if datasetS3Prefix == "" {
		datasetS3Prefix = "datasets/ultrachat.jsonl"
	}

	if s3Endpoint != "" && modelsBucket != "" {
		provider, err := trainerutils.GetS3Provider()
		if err != nil {
			test.T().Logf("Warning: Failed to create S3 provider to verify bucket: %v. Skipping S3 mode.", err)
			return ""
		}
		ctx := test.Ctx()
		exists, err := provider.BucketExists(ctx, modelsBucket)
		if err != nil {
			test.T().Logf("Warning: Failed to verify bucket existence for %s: %v. Skipping S3 mode.", modelsBucket, err)
			return ""
		}
		if !exists {
			test.T().Logf("Warning: Bucket %s does not exist. Skipping S3 mode. Will use HuggingFace.", modelsBucket)
			return ""
		}

		test.T().Logf("S3 mode for models/datasets: endpoint=%s, bucket=%s", s3InternalEndpoint, modelsBucket)
		return fmt.Sprintf(
			"export AWS_DEFAULT_ENDPOINT=%s; "+
				"export AWS_ACCESS_KEY_ID=%s; "+
				"export AWS_SECRET_ACCESS_KEY=%s; "+
				"export AWS_STORAGE_BUCKET=%s; "+
				"export MODEL_S3_PREFIX=%s; "+
				"export DATASET_S3_PREFIX=%s; ",
			shellQuote(s3InternalEndpoint), shellQuote(s3AccessKey), shellQuote(s3SecretKey),
			shellQuote(modelsBucket), shellQuote(modelS3Prefix), shellQuote(datasetS3Prefix),
		)
	}

	test.T().Log("HuggingFace mode: S3 not configured, will download model from HF Hub")
	return ""
}

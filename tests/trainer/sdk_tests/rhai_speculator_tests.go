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

	speculatorPullSecretName     = "aipcc-pull-secret"
	speculatorPullSecretSourceNs = "test-hrathina"
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
	userToken := common.GetNotebookUserToken(test)
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

	srcSecret, err := test.Client().Core().CoreV1().Secrets(speculatorPullSecretSourceNs).Get(test.Ctx(), speculatorPullSecretName, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to get %s from namespace %s", speculatorPullSecretName, speculatorPullSecretSourceNs))
	_, err = test.Client().Core().CoreV1().Secrets(namespace.Name).Create(test.Ctx(), &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      speculatorPullSecretName,
			Namespace: namespace.Name,
		},
		Type: srcSecret.Type,
		Data: srcSecret.Data,
	}, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to create %s in namespace %s", speculatorPullSecretName, namespace.Name))
	t.Logf("Copied pull secret %s to namespace %s", speculatorPullSecretName, namespace.Name)

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
		shellQuote(trainerutils.DefaultSpeculatorDataExtractRuntimeCUDA),
		speculatorNotebookName,
		shellQuote(trainerutils.DefaultSpeculatorTrainRuntimeCUDA),
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

	// Wait for TRAIN_ONLY TrainJob by known name
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

	env.test.Eventually(TrainJob(env.test, env.namespace.Name, trainJobName), TestTimeoutGpuProvisioning, 10*time.Second).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	t.Logf("TRAIN_ONLY TrainJob %s completed successfully", trainJobName)

	t.Log("Verifying TRAIN_ONLY training pod termination message...")
	verifySpeculatorTerminationMessage(env.test, env.namespace.Name, trainJobName)

	t.Log("Waiting for TRAIN_ONLY trainerStatus annotation to reach 100% progress...")
	env.test.Eventually(func() bool {
		tj := TrainJob(env.test, env.namespace.Name, trainJobName)(env.test)
		trainerStatusRaw := tj.GetAnnotations()[annotationTrainerStatus]
		if trainerStatusRaw == "" {
			return false
		}
		var status map[string]interface{}
		if err := json.Unmarshal([]byte(sanitizeJSON(trainerStatusRaw)), &status); err != nil {
			t.Logf("TRAIN_ONLY trainerStatus not valid JSON yet: %v", err)
			return false
		}
		progress, ok := status["progressPercentage"].(float64)
		if !ok || progress < 100 {
			t.Logf("TRAIN_ONLY trainerStatus progress: %.0f%%, waiting for 100%%...", progress)
			return false
		}
		t.Logf("TRAIN_ONLY trainerStatus reached 100%%: %s", trainerStatusRaw)
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "TRAIN_ONLY trainerStatus annotation should reach 100% progress")

	t.Log("Verifying TRAIN_ONLY training pod logs...")
	verifySpeculatorTrainOnlyPodLogs(env.test, env.namespace.Name, trainJobName)

	// Wait for checkpoint resume TrainJob by known name
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

	t.Log("Verifying checkpoint resume training pod logs...")
	verifySpeculatorTrainOnlyPodLogs(env.test, env.namespace.Name, resumeJobName)

	err := PollNotebookLogsForStatus(env.test, env.namespace.Name, podName, containerName, TestTimeoutDouble)
	env.test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")

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
		"[Kubeflow] vLLM sidecar is ready",
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

// RunSpeculatorFailureScenariosTest runs all speculator failure scenarios for both
// DATA_ONLY and TRAIN_ONLY modes in a single notebook pod. Two papermill runs:
// first DATA_ONLY failures (bad model path), then TRAIN_ONLY failures (bad paths).
// Scenarios run sequentially to avoid GPU contention.
func RunSpeculatorFailureScenariosTest(t *testing.T) {
	env := setupSpeculatorTestEnv(t, "5Gi")

	sdkInstallExports := buildKubeflowInstallExports()

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export IPYTHONDIR='/tmp/.ipython'; "+
			"export OPENSHIFT_API_URL=%s; export NOTEBOOK_USER_TOKEN=%s; "+
			"export NOTEBOOK_NAMESPACE=%s; "+
			"export SHARED_PVC_NAME=%s; "+
			"export VLLM_GPU_COUNT='1'; "+
			"export TRAIN_GPU_COUNT='1'; "+
			"export TARGET_LAYER_IDS='2,14,25,28'; "+
			"export TEST_TYPE='failure'; "+
			"%s"+ // SDK install exports
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python /opt/app-root/notebooks/%s && "+
			"export SPECULATOR_MODE='TRAIN_ONLY'; "+
			"export TRAINING_RUNTIME=%s; "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out_train_fail.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		shellQuote(GetOpenShiftApiUrl(env.test)), shellQuote(env.userToken), shellQuote(env.namespace.Name),
		shellQuote(env.rwxPvc.Name),
		sdkInstallExports,
		installKubeflowScript,
		shellQuote(trainerutils.DefaultSpeculatorTrainRuntimeCUDA),
		speculatorNotebookName,
	)

	t.Log("Speculator failure scenarios: TRAIN_ONLY incomplete extraction marker")
	command := []string{"/bin/sh", "-c", shellCmd}

	common.CreateNotebook(env.test, env.namespace, env.userToken, command, env.cm.Name, speculatorNotebookName, 0, env.rwxPvc, common.ContainerSizeSmall, common.GetRecommendedNotebookImageFromImageStream(env.test, common.NotebookImageStreamTrainingHubCUDA))

	defer func() {
		common.DeleteNotebook(env.test, env.namespace)
		env.test.Eventually(common.Notebooks(env.test, env.namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	podName, containerName := trainerutils.WaitForNotebookPodRunning(env.test, env.namespace.Name)

	err := PollNotebookLogsForStatus(env.test, env.namespace.Name, podName, containerName, TestTimeoutDouble)
	env.test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")

	// Log scenario results from notebook output
	var tail int64 = 2000
	logs := PodLog(env.test, env.namespace.Name, podName, corev1.PodLogOptions{
		Container: containerName,
		TailLines: &tail,
	})(env.test)
	for _, line := range strings.Split(logs, "\n") {
		if strings.Contains(line, "PASSED:") || strings.Contains(line, "FAILED:") ||
			strings.Contains(line, "Scenario:") || strings.Contains(line, "All scenarios passed") ||
			strings.Contains(line, "SPECULATOR FAILURE SCENARIOS") {
			t.Log(line)
		}
	}
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

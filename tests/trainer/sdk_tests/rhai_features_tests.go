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
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	common "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	rhaiFeaturesNotebookName = "rhai_features.ipynb"
	rhaiFeaturesNotebookPath = "resources/" + rhaiFeaturesNotebookName

	// Annotation keys for progression tracking
	annotationProgressionTracking = "trainer.opendatahub.io/progression-tracking"
	annotationMetricsPort         = "trainer.opendatahub.io/metrics-port"
	annotationMetricsPollInterval = "trainer.opendatahub.io/metrics-poll-interval"
	annotationTrainerStatus       = "trainer.opendatahub.io/trainerStatus"
)

// RhaiFeatureConfig holds configuration for RHAI feature tests
type RhaiFeatureConfig struct {
	EnableProgressionTracking bool
	EnableJitCheckpoint       bool
	CheckpointOutputDir       string
	CheckpointSaveStrategy    string
	CheckpointSaveTotalLimit  string
	Accelerator               Accelerator // CPU, NVIDIA, or AMD
}

// RunRhaiFeaturesTest runs the e2e test for RHAI features with progression tracking only (CPU)
func RunRhaiFeaturesProgressionTest(t *testing.T) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: true,
		EnableJitCheckpoint:       false,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               CPU,
	})
}

// RunRhaiFeaturesProgressionTestGPU runs the e2e test for RHAI features with progression tracking (GPU)
func RunRhaiFeaturesProgressionTestGPU(t *testing.T) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: true,
		EnableJitCheckpoint:       false,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               NVIDIA,
	})
}

// RunRhaiFeaturesCheckpointTest runs the e2e test for RHAI features with checkpointing only (CPU)
func RunRhaiFeaturesCheckpointTest(t *testing.T) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: false,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               CPU,
	})
}

// RunRhaiFeaturesCheckpointTestGPU runs the e2e test for RHAI features with checkpointing (GPU)
func RunRhaiFeaturesCheckpointTestGPU(t *testing.T) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: false,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               NVIDIA,
	})
}

// RunRhaiFeaturesAllTest runs the e2e test for RHAI features with both progression tracking and checkpointing (CPU)
func RunRhaiFeaturesAllTest(t *testing.T) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: true,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               CPU,
	})
}

// RunRhaiFeaturesAllTestGPU runs the e2e test for RHAI features with both features (GPU)
func RunRhaiFeaturesAllTestGPU(t *testing.T) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: true,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               NVIDIA,
	})
}

// runRhaiFeaturesTestWithConfig runs the e2e test with the given feature configuration
func runRhaiFeaturesTestWithConfig(t *testing.T, config RhaiFeatureConfig) {
	test := With(t)

	// Create a new test namespace
	namespace := test.NewTestNamespace()

	// Ensure Notebook ServiceAccount exists
	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	// RBACs setup for user (user token is used by notebook for Trainer API calls)
	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	// ClusterRoleBinding for cluster-scoped resources (ClusterTrainingRuntimes) - minimal get/list/watch access
	trainerutils.CreateUserClusterRoleBindingForTrainerRuntimes(test, userName)

	// Create ConfigMap with notebook
	localPath := rhaiFeaturesNotebookPath
	nb, err := os.ReadFile(localPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", localPath))
	cm := CreateConfigMap(test, namespace.Name, map[string][]byte{rhaiFeaturesNotebookName: nb})

	// Create shared RWX PVC for distributed training (HF cache shared across nodes)
	storageClass, err := GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	sharedPVC := CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		"20Gi",
		AccessModes(corev1.ReadWriteMany),
		StorageClassName(storageClass.Name),
	)

	// Build command with parameters (API URL, token, namespace, shared PVC, feature flags)
	// Install kubeflow SDK from opendatahub-io/kubeflow-sdk main branch
	// Uses torch-distributed runtime, model/dataset downloaded in training function
	enableProgression := "false"
	if config.EnableProgressionTracking {
		enableProgression = "true"
	}
	enableCheckpoint := "false"
	if config.EnableJitCheckpoint {
		enableCheckpoint = "true"
	}

	// Determine GPU resource label (empty for CPU) and training runtime
	gpuResourceLabel := ""
	trainingRuntime := "torch-distributed" // Default for CPU and NVIDIA
	if config.Accelerator.IsGpu() {
		gpuResourceLabel = config.Accelerator.ResourceLabel
		if config.Accelerator == AMD {
			trainingRuntime = "torch-distributed-rocm"
		}
	}

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export OPENSHIFT_API_URL='%s'; "+
			"export NOTEBOOK_TOKEN='%s'; "+
			"export NOTEBOOK_NAMESPACE='%s'; "+
			"export SHARED_PVC_NAME='%s'; "+
			"export ENABLE_PROGRESSION_TRACKING='%s'; "+
			"export ENABLE_JIT_CHECKPOINT='%s'; "+
			"export CHECKPOINT_OUTPUT_DIR='%s'; "+
			"export CHECKPOINT_SAVE_STRATEGY='%s'; "+
			"export CHECKPOINT_SAVE_TOTAL_LIMIT='%s'; "+
			"export GPU_RESOURCE_LABEL='%s'; "+
			"export TRAINING_RUNTIME='%s'; "+
			"python -m pip install --quiet --no-cache-dir papermill && "+
			"python -m pip install --quiet --no-cache-dir git+https://github.com/opendatahub-io/kubeflow-sdk.git@main && "+
			"python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"sleep infinity",
		GetOpenShiftApiUrl(test), userToken, namespace.Name, sharedPVC.Name,
		enableProgression,
		enableCheckpoint,
		config.CheckpointOutputDir,
		config.CheckpointSaveStrategy,
		config.CheckpointSaveTotalLimit,
		gpuResourceLabel,
		trainingRuntime,
		rhaiFeaturesNotebookName,
	)

	test.T().Logf("Feature config: ProgressionTracking=%v, JitCheckpoint=%v, Accelerator=%s", config.EnableProgressionTracking, config.EnableJitCheckpoint, config.Accelerator.Type)
	command := []string{"/bin/sh", "-c", shellCmd}

	// Create Notebook CR using the RWX PVC
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, rhaiFeaturesNotebookName, 0, sharedPVC, common.ContainerSizeSmall)

	// Cleanup
	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), TestTimeoutLong).Should(HaveLen(0))
	}()

	// Wait for the Notebook Pod to be running
	trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	// Wait for TrainJob to be created in the namespace (notebook creates exactly one)
	var trainJobName string
	test.Eventually(func() int {
		jobs := TrainJobs(test, namespace.Name)(test)
		if len(jobs) == 1 {
			trainJobName = jobs[0].Name
		}
		return len(jobs)
	}, TestTimeoutDouble, 5*time.Second).Should(Equal(1), "Expected exactly one TrainJob to be created in namespace")

	test.T().Logf("TrainJob created: %s", trainJobName)

	// For checkpoint tests, run suspend/resume flow instead of waiting for completion
	if config.EnableJitCheckpoint {
		test.T().Log("Running JIT checkpoint suspend/resume test...")
		verifyCheckpoints(test, namespace.Name, trainJobName, config.CheckpointOutputDir)
	} else {
		// Wait for TrainJob to complete normally
		test.Eventually(TrainJob(test, namespace.Name, trainJobName), TestTimeoutLong, 10*time.Second).
			Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
		test.T().Logf("TrainJob %s completed successfully", trainJobName)
	}

	// Get TrainJob and verify RHAI annotations based on config
	trainJob := TrainJob(test, namespace.Name, trainJobName)(test)
	annotations := trainJob.GetAnnotations()

	// Verify progression tracking if enabled
	if config.EnableProgressionTracking {
		test.T().Log("Verifying progression tracking features...")

		// Verify termination message on training pods contains progress data
		verifyTerminationMessage(test, namespace.Name, trainJobName)

		// Verify progression tracking annotations
		test.Expect(annotations[annotationProgressionTracking]).To(Equal("true"),
			"Expected progression-tracking annotation to be 'true'")
		test.T().Log("progression-tracking annotation is 'true'")

		test.Expect(annotations[annotationMetricsPort]).To(Equal("28080"),
			"Expected metrics-port annotation to be '28080'")
		test.T().Log("metrics-port annotation is '28080'")

		test.Expect(annotations[annotationMetricsPollInterval]).To(Equal("8"),
			"Expected metrics-poll-interval annotation to be '8'")
		test.T().Log("metrics-poll-interval annotation is '8'")

		// Verify trainerStatus annotation contains valid JSON with expected fields
		trainerStatusRaw := annotations[annotationTrainerStatus]
		test.Expect(trainerStatusRaw).NotTo(BeEmpty(), "trainerStatus annotation should not be empty")

		var trainerStatus map[string]interface{}
		err = json.Unmarshal([]byte(trainerStatusRaw), &trainerStatus)
		test.Expect(err).NotTo(HaveOccurred(), "trainerStatus should be valid JSON")
		test.T().Logf("trainerStatus: %s", trainerStatusRaw)

		// Verify progress metrics exist and have valid values
		test.Expect(trainerStatus).To(HaveKey("progressPercentage"))
		progress := trainerStatus["progressPercentage"].(float64)
		test.Expect(progress).To(BeNumerically(">=", 0), "Progress should be non-negative")
		test.Expect(progress).To(BeNumerically("<=", 100), "Progress should not exceed 100%")
		test.T().Logf("progressPercentage: %.0f%%", progress)

		test.Expect(trainerStatus).To(HaveKey("currentStep"))
		test.Expect(trainerStatus).To(HaveKey("totalSteps"))
		test.T().Logf("currentStep: %v/%v", trainerStatus["currentStep"], trainerStatus["totalSteps"])

		test.Expect(trainerStatus).To(HaveKey("currentEpoch"))
		test.Expect(trainerStatus).To(HaveKey("totalEpochs"))
		test.T().Logf("currentEpoch: %v/%v", trainerStatus["currentEpoch"], trainerStatus["totalEpochs"])

		test.Expect(trainerStatus).To(HaveKey("estimatedRemainingSeconds"))
		remaining := trainerStatus["estimatedRemainingSeconds"].(float64)
		test.Expect(remaining).To(BeNumerically(">=", 0), "Remaining time should be non-negative")
		test.T().Logf("estimatedRemainingSeconds: %.0f", remaining)

		test.T().Log("Progression tracking verification passed!")
	}

	test.T().Log("All RHAI features tests passed!")
}

// verifyTerminationMessage checks that training pods have termination messages with progress data
func verifyTerminationMessage(test Test, namespace, trainJobName string) {
	test.T().Helper()

	// List all pods in namespace and filter by TrainJob name prefix
	// Pod naming convention: <trainjob-name>-node-<index>-<index>-<suffix>
	allPods, err := test.Client().Core().CoreV1().Pods(namespace).List(
		test.Ctx(),
		metav1.ListOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to list pods")

	// Filter pods that belong to this TrainJob (name starts with trainjob name and contains "node")
	var pods []corev1.Pod
	for _, pod := range allPods.Items {
		if strings.HasPrefix(pod.Name, trainJobName) && strings.Contains(pod.Name, "-node-") {
			pods = append(pods, pod)
		}
	}
	test.Expect(len(pods)).To(Equal(2), "Expected exactly 2 training pods for distributed job")

	test.T().Logf("Found %d training pod(s) for TrainJob %s", len(pods), trainJobName)

	// Check termination message on at least one pod
	var foundTerminationMessage bool
	for _, pod := range pods {
		for _, containerStatus := range pod.Status.ContainerStatuses {
			if containerStatus.Name != "node" {
				continue
			}

			// Check if container terminated
			if containerStatus.State.Terminated == nil {
				continue
			}

			terminationMessage := containerStatus.State.Terminated.Message
			if terminationMessage == "" {
				continue
			}

			test.T().Logf("Pod %s termination message: %s", pod.Name, terminationMessage)

			// Parse termination message as JSON
			var terminationData map[string]interface{}
			err := json.Unmarshal([]byte(terminationMessage), &terminationData)
			test.Expect(err).NotTo(HaveOccurred(), "Termination message should be valid JSON")

			// Verify expected fields exist in termination message
			test.Expect(terminationData).To(HaveKey("progressPercentage"), "Termination message should have progressPercentage")
			test.Expect(terminationData).To(HaveKey("currentStep"), "Termination message should have currentStep")
			test.Expect(terminationData).To(HaveKey("totalSteps"), "Termination message should have totalSteps")
			test.Expect(terminationData).To(HaveKey("currentEpoch"), "Termination message should have currentEpoch")
			test.Expect(terminationData).To(HaveKey("totalEpochs"), "Termination message should have totalEpochs")
			test.Expect(terminationData).To(HaveKey("estimatedRemainingSeconds"), "Termination message should have estimatedRemainingSeconds")

			// Verify trainMetrics if present
			if trainMetrics, ok := terminationData["trainMetrics"].(map[string]interface{}); ok {
				test.T().Logf("trainMetrics: %v", trainMetrics)
				test.Expect(trainMetrics).To(HaveKey("loss"), "trainMetrics should have loss")
			}

			// Verify evalMetrics if present
			if evalMetrics, ok := terminationData["evalMetrics"].(map[string]interface{}); ok {
				test.T().Logf("evalMetrics: %v", evalMetrics)
			}

			foundTerminationMessage = true
			test.T().Logf("Termination message verified for pod %s", pod.Name)
			break
		}
		if foundTerminationMessage {
			break
		}
	}

	test.Expect(foundTerminationMessage).To(BeTrue(), "Expected at least one training pod to have a termination message with progress data")
}

// verifyCheckpointsWithSuspendResume tests JIT checkpoint functionality by:
// 1. Waiting for training to start and make progress
// 2. Suspending the TrainJob to trigger checkpoint save
// 3. Verifying checkpoint was saved
// 4. Resuming the TrainJob to verify checkpoint restore
// 5. Verifying training completes successfully
func verifyCheckpoints(test Test, namespace, trainJobName, checkpointDir string) {
	test.T().Helper()

	test.T().Logf("Starting JIT checkpoint verification for TrainJob %s (checkpoint_dir=%s)", trainJobName, checkpointDir)

	// Step 1: Wait for training pods to be running and make actual progress
	test.T().Log("Step 1: Waiting for training pods to start and make progress...")
	test.Eventually(func() int {
		pods := listTrainingPods(test, namespace, trainJobName)
		runningCount := 0
		for _, pod := range pods {
			if pod.Status.Phase == corev1.PodRunning {
				runningCount++
			}
		}
		return runningCount
	}, TestTimeoutMedium, 5*time.Second).Should(BeNumerically(">", 0), "At least one training pod should be running")
	test.T().Log("Training pods are running")

	// Wait for 1st epoch to complete before suspending
	// But also check if training already completed (fast GPU case)
	test.T().Log("Waiting for 1st epoch to complete (currentEpoch >= 1.0)...")

	trainingAlreadyComplete := false
	test.Eventually(func() bool {
		trainJob := TrainJob(test, namespace, trainJobName)(test)

		// Check if already completed
		if TrainJobConditionComplete(trainJob) == metav1.ConditionTrue {
			trainingAlreadyComplete = true
			return true
		}

		annotations := trainJob.GetAnnotations()
		statusJSON, ok := annotations[annotationTrainerStatus]
		if !ok {
			return false
		}
		var status map[string]interface{}
		if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
			return false
		}
		if currentEpoch, ok := status["currentEpoch"].(float64); ok {
			return currentEpoch >= 1.0
		}
		return false
	}, TestTimeoutMedium, 2*time.Second).Should(BeTrue(), "Training should complete at least 1 epoch or finish")

	// If training already completed, skip suspend/resume test
	if trainingAlreadyComplete {
		test.T().Fatal("Training completed before suspend could be triggered - cannot verify JIT checkpoint functionality. Consider increasing dataset size or epochs.")
	}

	test.T().Log("1st epoch completed - ready to suspend")

	// Capture progress percentage BEFORE suspension
	preSuspendProgress := getProgressPercentage(test, namespace, trainJobName)
	test.T().Logf("Progress BEFORE suspension: %d%%", preSuspendProgress)
	test.Expect(preSuspendProgress).To(BeNumerically(">", 0), "Expected training progress > 0 before suspension")

	// Step 2: Suspend the TrainJob to trigger JIT checkpoint save
	test.T().Log("Step 2: Suspending TrainJob to trigger checkpoint save...")
	suspendTrainJob(test, namespace, trainJobName, true)

	// Step 3: Wait for pods to fully terminate (checkpoint saved before termination)
	test.T().Log("Step 3: Waiting for training pods to fully terminate...")
	test.Eventually(func() int {
		pods := listTrainingPods(test, namespace, trainJobName)
		// Count pods that are still running or pending termination
		activeCount := 0
		for _, pod := range pods {
			if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodPending {
				activeCount++
			}
		}
		return activeCount
	}, TestTimeoutMedium, 5*time.Second).Should(Equal(0), "All training pods should terminate after suspend")
	test.T().Log("All training pods terminated - checkpoint should be saved")

	// Step 4: Resume the TrainJob
	test.T().Log("Step 4: Resuming TrainJob to verify checkpoint restore...")
	suspendTrainJob(test, namespace, trainJobName, false)

	// Wait for new pods to start
	test.T().Log("Waiting for new training pods to start...")
	test.Eventually(func() int {
		pods := listTrainingPods(test, namespace, trainJobName)
		runningCount := 0
		for _, pod := range pods {
			if pod.Status.Phase == corev1.PodRunning {
				runningCount++
			}
		}
		return runningCount
	}, TestTimeoutMedium, 5*time.Second).Should(BeNumerically(">", 0), "Training pods should start after resume")
	test.T().Log("New training pods started")

	// Wait for training to make progress after resume, then capture progress
	test.T().Log("Waiting for training to make progress after resume...")
	test.Eventually(func() int {
		return getProgressPercentage(test, namespace, trainJobName)
	}, TestTimeoutMedium, 5*time.Second).Should(BeNumerically(">", 0), "Training should show progress after resume")

	postResumeProgress := getProgressPercentage(test, namespace, trainJobName)
	test.T().Logf("Progress AFTER resume: %d%%", postResumeProgress)

	// Verify progress after resume is >= progress before suspension (checkpoint worked)
	test.T().Logf("Checkpoint validation: pre-suspend=%d%%, post-resume=%d%%", preSuspendProgress, postResumeProgress)
	test.Expect(postResumeProgress).To(BeNumerically(">=", preSuspendProgress),
		fmt.Sprintf("Progress after resume (%d%%) should be >= before suspend (%d%%) - checkpoint should preserve progress", postResumeProgress, preSuspendProgress))

	// Step 5: Wait for training to complete or fail (to get final state)
	test.T().Log("Step 5: Waiting for training to complete after resume...")
	test.Eventually(func() bool {
		trainJob := TrainJob(test, namespace, trainJobName)(test)
		complete := TrainJobConditionComplete(trainJob) == metav1.ConditionTrue
		failed := TrainJobConditionFailed(trainJob) == metav1.ConditionTrue
		return complete || failed
	}, TestTimeoutLong, 10*time.Second).Should(BeTrue(), "TrainJob should reach terminal state after resume")

	// Check final status
	finalJob := TrainJob(test, namespace, trainJobName)(test)
	if TrainJobConditionComplete(finalJob) == metav1.ConditionTrue {
		test.T().Log("TrainJob completed successfully after checkpoint restore")
	} else if TrainJobConditionFailed(finalJob) == metav1.ConditionTrue {
		// Log failure details
		test.T().Log("WARNING: TrainJob failed after resume")

		// Get trainerStatus to see progress after resume
		annotations := finalJob.GetAnnotations()
		if statusJSON, ok := annotations[annotationTrainerStatus]; ok {
			test.T().Logf("Final trainerStatus: %s", statusJSON)
		}

		// Get failure reason from TrainJob conditions
		for _, cond := range finalJob.Status.Conditions {
			if cond.Type == "Failed" && cond.Status == metav1.ConditionTrue {
				test.T().Logf("Failure reason: %s - %s", cond.Reason, cond.Message)
			}
		}

		// Fail the test since checkpoint resume didn't lead to successful completion
		test.Expect(TrainJobConditionComplete(finalJob)).To(Equal(metav1.ConditionTrue),
			"TrainJob should complete successfully after checkpoint resume")
	}

	// Step 6: Verify logs show checkpoint was loaded
	test.T().Log("Step 6: Verifying checkpoint was loaded from logs...")
	verifyCheckpointLoadedFromLogs(test, namespace, trainJobName)

	test.T().Log("JIT checkpoint verification completed successfully!")
}

// getProgressPercentage extracts progress percentage from trainerStatus annotation
func getProgressPercentage(test Test, namespace, trainJobName string) int {
	test.T().Helper()

	trainJob := TrainJob(test, namespace, trainJobName)(test)
	annotations := trainJob.GetAnnotations()
	statusJSON, ok := annotations[annotationTrainerStatus]
	if !ok {
		return 0
	}

	var status map[string]interface{}
	if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
		return 0
	}

	if progress, ok := status["progressPercentage"].(float64); ok {
		return int(progress)
	}
	return 0
}

// suspendTrainJob toggles the suspend state of a TrainJob using patch
func suspendTrainJob(test Test, namespace, trainJobName string, suspend bool) {
	test.T().Helper()

	// Use JSON merge patch to only modify spec.suspend field
	// This avoids webhook validation errors about modifying podTemplateOverrides
	patchData := fmt.Sprintf(`{"spec":{"suspend":%t}}`, suspend)

	_, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Patch(
		test.Ctx(),
		trainJobName,
		types.MergePatchType,
		[]byte(patchData),
		metav1.PatchOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to %s TrainJob", map[bool]string{true: "suspend", false: "resume"}[suspend]))
	test.T().Logf("TrainJob %s %s", trainJobName, map[bool]string{true: "suspended", false: "resumed"}[suspend])
}

// listTrainingPods returns all training pods for a TrainJob
func listTrainingPods(test Test, namespace, trainJobName string) []corev1.Pod {
	test.T().Helper()

	allPods, err := test.Client().Core().CoreV1().Pods(namespace).List(
		test.Ctx(),
		metav1.ListOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to list pods")

	var pods []corev1.Pod
	for _, pod := range allPods.Items {
		if strings.HasPrefix(pod.Name, trainJobName) && strings.Contains(pod.Name, "-node-") {
			pods = append(pods, pod)
		}
	}
	return pods
}

// verifyCheckpointLoadedFromLogs checks pod logs for checkpoint resume messages
func verifyCheckpointLoadedFromLogs(test Test, namespace, trainJobName string) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	if len(pods) == 0 {
		test.T().Log("No training pods found to verify checkpoint load logs")
		return
	}

	// Check logs of completed pods for checkpoint resume indicators (from Kubeflow SDK)
	checkpointIndicators := []string{
		"[Kubeflow] Found latest checkpoint:",
		"[Kubeflow] Auto-resuming from:",
	}

	foundCheckpointLog := false
	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodSucceeded {
			continue
		}

		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)

		for _, indicator := range checkpointIndicators {
			if strings.Contains(logs, indicator) {
				test.T().Logf("Found checkpoint resume indicator in pod %s: %s", pod.Name, indicator)
				foundCheckpointLog = true
				break
			}
		}

		if foundCheckpointLog {
			break
		}
	}

	test.Expect(foundCheckpointLog).To(BeTrue(), "Expected checkpoint resume log (e.g. '[Kubeflow] Auto-resuming from:') not found in training pod logs")
}

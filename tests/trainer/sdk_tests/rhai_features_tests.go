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
	"regexp"
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

	// Annotation keys for progression tracking (must match SDK/training-operator constants)
	annotationProgressionTracking = "trainer.opendatahub.io/progression-tracking"
	annotationMetricsPort         = "trainer.opendatahub.io/metrics-port"
	annotationMetricsPollInterval = "trainer.opendatahub.io/metrics-poll-interval"
	annotationTrainerStatus       = "trainer.opendatahub.io/trainerStatus"
)

// Compiled regex for epoch detection (compiled once for performance)
var epochPattern = regexp.MustCompile(`'epoch': [1-9]`)

// boolStr converts bool to "true"/"false" string for env vars
func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// RhaiFeatureConfig holds configuration for RHAI feature tests
type RhaiFeatureConfig struct {
	EnableProgressionTracking bool
	EnableJitCheckpoint       bool
	CheckpointOutputDir       string
	CheckpointSaveStrategy    string
	CheckpointSaveTotalLimit  string
	Accelerator               Accelerator // CPU, NVIDIA, or AMD
}

// RunRhaiFeaturesProgressionTest runs the e2e test for RHAI features with progression tracking
func RunRhaiFeaturesProgressionTest(t *testing.T, accelerator Accelerator) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: true,
		EnableJitCheckpoint:       false,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               accelerator,
	})
}

// RunRhaiFeaturesCheckpointTest runs the e2e test for RHAI features with checkpointing
func RunRhaiFeaturesCheckpointTest(t *testing.T, accelerator Accelerator) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: false,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               accelerator,
	})
}

// RunRhaiFeaturesAllTest runs the e2e test for RHAI features with both progression tracking and checkpointing
func RunRhaiFeaturesAllTest(t *testing.T, accelerator Accelerator) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: true,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               accelerator,
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
	enableProgression := boolStr(config.EnableProgressionTracking)
	enableCheckpoint := boolStr(config.EnableJitCheckpoint)

	// Determine GPU resource label (empty for CPU) and training runtime
	gpuResourceLabel := ""
	trainingRuntime := trainerutils.DefaultClusterTrainingRuntime // Default for CPU and NVIDIA
	if config.Accelerator.IsGpu() {
		gpuResourceLabel = config.Accelerator.ResourceLabel
		if config.Accelerator == AMD {
			trainingRuntime = trainerutils.DefaultClusterTrainingRuntimeROCm
		}
	}

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export IPYTHONDIR='/tmp/.ipython'; "+
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

	// Cleanup - use longer timeout due to large runtime images
	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
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

	// Verify TrainJob annotations match expected config BEFORE running tests
	// This catches configuration issues early (e.g., SDK defaults overriding our settings)
	verifyTrainJobAnnotations(test, namespace.Name, trainJobName, config)

	// For checkpoint tests, run suspend/resume flow instead of waiting for completion
	if config.EnableJitCheckpoint {
		test.T().Log("Running JIT checkpoint suspend/resume test...")
		verifyCheckpoints(test, namespace.Name, trainJobName, config.CheckpointOutputDir, config.EnableProgressionTracking, TestTimeoutGpuProvisioning)
	} else {
		// Wait for TrainJob to complete normally - use longer timeout due to large runtime images
		test.Eventually(TrainJob(test, namespace.Name, trainJobName), TestTimeoutGpuProvisioning, 10*time.Second).
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
		test.Expect(progress).To(BeNumerically("==", 100), "Progress should be 100% at completion")
		test.T().Logf("progressPercentage: %.0f%%", progress)

		test.Expect(trainerStatus).To(HaveKey("currentStep"))
		test.Expect(trainerStatus).To(HaveKey("totalSteps"))
		test.T().Logf("currentStep: %v/%v", trainerStatus["currentStep"], trainerStatus["totalSteps"])

		test.Expect(trainerStatus).To(HaveKey("currentEpoch"))
		test.Expect(trainerStatus).To(HaveKey("totalEpochs"))
		test.T().Logf("currentEpoch: %v/%v", trainerStatus["currentEpoch"], trainerStatus["totalEpochs"])

		test.Expect(trainerStatus).To(HaveKey("estimatedRemainingSeconds"))
		remaining := trainerStatus["estimatedRemainingSeconds"].(float64)
		test.Expect(remaining).To(BeNumerically("==", 0), "Remaining time should be 0 at completion")
		test.T().Logf("estimatedRemainingSeconds: %.0f", remaining)

		test.T().Log("Progression tracking verification passed!")
	} else {
		// Verify progression tracking is NOT enabled when disabled
		test.T().Log("Verifying progression tracking is disabled...")

		// trainerStatus annotation should be empty when progression tracking is disabled
		trainerStatusRaw := annotations[annotationTrainerStatus]
		test.Expect(trainerStatusRaw).To(BeEmpty(),
			"trainerStatus annotation should be empty when progression tracking is disabled")
		test.T().Log("trainerStatus annotation is empty as expected")
	}

	test.T().Log("All RHAI features tests passed!")
}

// verifyTrainJobAnnotations verifies TrainJob annotations match expected config
// This catches configuration issues early (e.g., SDK defaults overriding settings)
func verifyTrainJobAnnotations(test Test, namespace, trainJobName string, config RhaiFeatureConfig) {
	test.T().Helper()

	trainJob := TrainJob(test, namespace, trainJobName)(test)
	annotations := trainJob.GetAnnotations()

	test.T().Log("Verifying TrainJob annotations match expected config...")

	// Verify progression-tracking annotation
	// SDK sets this annotation only when progression tracking is enabled
	actualProgression := annotations[annotationProgressionTracking]
	if config.EnableProgressionTracking {
		test.Expect(actualProgression).To(Equal("true"),
			"progression-tracking annotation should be 'true' when enabled")
		test.T().Logf("progression-tracking: expected=true, actual=%s", actualProgression)

		// Also verify metrics annotations are present when progression is enabled
		test.Expect(annotations[annotationMetricsPort]).NotTo(BeEmpty(),
			"metrics-port annotation should be set when progression tracking is enabled")
		test.Expect(annotations[annotationMetricsPollInterval]).NotTo(BeEmpty(),
			"metrics-poll-interval annotation should be set when progression tracking is enabled")
	} else {
		// When disabled, annotation should be absent or explicitly "false"
		test.Expect(actualProgression).To(Or(BeEmpty(), Equal("false")),
			"progression-tracking annotation should be empty or 'false' when disabled")
		test.T().Logf("progression-tracking: expected=empty/false, actual=%s", actualProgression)
	}

	// Note: SDK does not have a separate jit-checkpoint annotation
	// Checkpoint functionality is verified via pod logs (verifyCheckpointLoadedFromLogs)

	test.T().Log("TrainJob annotations match expected config")
}

// verifyTerminationMessage checks that training pods have termination messages with progress data
func verifyTerminationMessage(test Test, namespace, trainJobName string) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
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

			// Log optional fields if present (per training-operator AnnotationStatus)
			if summary, ok := terminationData["estimatedRemainingTimeSummary"].(string); ok {
				test.T().Logf("estimatedRemainingTimeSummary: %s", summary)
			}
			if lastUpdated, ok := terminationData["lastUpdatedTime"].(string); ok {
				test.T().Logf("lastUpdatedTime: %s", lastUpdated)
			}

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

// verifyCheckpoints tests JIT checkpoint functionality by:
// 1. Waiting for training to start and make progress
// 2. Suspending the TrainJob to trigger checkpoint save
// 3. Verifying checkpoint was saved
// 4. Resuming the TrainJob to verify checkpoint restore
// 5. Verifying training completes successfully
// When progressionEnabled=true, also verifies progress before/after suspend using annotations
func verifyCheckpoints(test Test, namespace, trainJobName, checkpointDir string, progressionEnabled bool, timeout time.Duration) {
	test.T().Helper()

	test.T().Logf("Starting JIT checkpoint verification for TrainJob %s (checkpoint_dir=%s)", trainJobName, checkpointDir)

	// Step 1: Wait for training pods to be running
	test.T().Log("Step 1: Waiting for training pods to start...")
	test.Eventually(func() int {
		return countRunningPods(test, namespace, trainJobName)
	}, timeout, 5*time.Second).Should(Equal(2), "Expected exactly 2 training pods to be running")
	test.T().Log("Training pods are running")

	// Wait for training to complete at least 1 epoch (detected from logs)
	// HuggingFace Trainer logs epoch completion with patterns like "'epoch': 1.0" or "Epoch 1"
	test.T().Log("Waiting for training to complete at least 1 epoch (checking logs)...")
	test.Eventually(func() bool {
		return hasCompletedEpochFromLogs(test, namespace, trainJobName)
	}, TestTimeoutMedium, 5*time.Second).Should(BeTrue(), "Training should complete at least 1 epoch before suspension")
	test.T().Log("At least 1 epoch completed - ready to suspend")

	// Step 2: Suspend the TrainJob to trigger JIT checkpoint save
	test.T().Log("Step 2: Suspending TrainJob to trigger checkpoint save...")
	suspendTrainJob(test, namespace, trainJobName, true)

	// Step 3: Wait for pods to fully terminate (checkpoint saved before termination)
	test.T().Log("Step 3: Waiting for training pods to fully terminate...")
	test.Eventually(func() int {
		return countActivePods(test, namespace, trainJobName)
	}, TestTimeoutMedium, 5*time.Second).Should(Equal(0), "All training pods should terminate after suspend")
	test.T().Log("All training pods terminated - checkpoint should be saved")

	// Step 4: Verify job is suspended and NOT already completed (race condition check)
	test.T().Log("Step 4: Verifying job is suspended (not completed)...")
	trainJob := TrainJob(test, namespace, trainJobName)(test)
	if TrainJobConditionComplete(trainJob) == metav1.ConditionTrue {
		test.T().Fatal("Training completed before suspend took effect - cannot verify JIT checkpoint functionality. Consider increasing dataset size or epochs.")
	}
	test.Expect(TrainJobConditionSuspended(trainJob)).To(Equal(metav1.ConditionTrue), "TrainJob should be in suspended state")
	test.T().Log("TrainJob is suspended")

	// Step 5: Store progress before resume (only when progression tracking is enabled)
	var preSuspendProgress int
	var preSuspendEpoch float64
	if progressionEnabled {
		// Wait for operator to poll metrics and update TrainJob annotations
		// This avoids race condition where job is suspended before progress is tracked
		test.T().Log("Step 5: Waiting for progress to be tracked in TrainJob...")
		test.Eventually(func() int {
			return getProgressPercentage(test, namespace, trainJobName)
		}, TestTimeoutMedium, 5*time.Second).Should(BeNumerically(">", 0), "Progress should be tracked before suspension")

		preSuspendProgress = getProgressPercentage(test, namespace, trainJobName)
		preSuspendEpoch = getCurrentEpoch(test, namespace, trainJobName)
		test.T().Logf("Pre-suspend state: epoch=%.2f, progress=%d%%", preSuspendEpoch, preSuspendProgress)
	}

	// Step 6: Resume the TrainJob
	test.T().Log("Step 6: Resuming TrainJob to verify checkpoint restore...")
	suspendTrainJob(test, namespace, trainJobName, false)

	// Wait for new pods to start
	test.T().Log("Waiting for new training pods to start...")
	test.Eventually(func() int {
		return countRunningPods(test, namespace, trainJobName)
	}, TestTimeoutMedium, 5*time.Second).Should(Equal(2), "Expected exactly 2 training pods to start after resume")
	test.T().Log("New training pods started")

	// Step 7: Verify progress after resume (only when progression tracking is enabled)
	if progressionEnabled {
		// Wait for progress annotation to be updated by resumed pods
		test.T().Log("Step 7: Waiting for progress update after resume...")
		var postResumeProgress int
		var postResumeEpoch float64
		test.Eventually(func() bool {
			postResumeProgress = getProgressPercentage(test, namespace, trainJobName)
			postResumeEpoch = getCurrentEpoch(test, namespace, trainJobName)
			// Wait until we have fresh data (epoch should be at least floor of pre-suspend)
			return postResumeEpoch >= float64(int(preSuspendEpoch))
		}, TestTimeoutMedium, 5*time.Second).Should(BeTrue(), "Progress should be updated after resume")

		test.T().Logf("Post-resume state: epoch=%.2f, progress=%d%%", postResumeEpoch, postResumeProgress)

		// Verify checkpoint preserved training state
		expectedMinEpoch := float64(int(preSuspendEpoch)) // floor of pre-suspend epoch
		test.Expect(postResumeEpoch).To(BeNumerically(">=", expectedMinEpoch),
			fmt.Sprintf("Epoch after resume (%.2f) should be >= floor(pre-suspend epoch %.2f)", postResumeEpoch, expectedMinEpoch))
		test.T().Log("Checkpoint verification: Progress preserved after resume")
	}

	// Step 8: Wait for training to complete or fail
	test.T().Log("Step 8: Waiting for training to complete after resume...")
	test.Eventually(func() bool {
		trainJob := TrainJob(test, namespace, trainJobName)(test)
		complete := TrainJobConditionComplete(trainJob) == metav1.ConditionTrue
		failed := TrainJobConditionFailed(trainJob) == metav1.ConditionTrue
		return complete || failed
	}, timeout, 10*time.Second).Should(BeTrue(), "TrainJob should reach terminal state after resume")

	// Check final status
	finalJob := TrainJob(test, namespace, trainJobName)(test)
	if TrainJobConditionComplete(finalJob) == metav1.ConditionTrue {
		test.T().Log("TrainJob completed successfully after checkpoint restore")
	} else if TrainJobConditionFailed(finalJob) == metav1.ConditionTrue {
		// Log failure details
		test.T().Log("WARNING: TrainJob failed after resume")

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

	// Step 9: Verify logs show checkpoint was loaded
	test.T().Log("Step 9: Verifying checkpoint was loaded from logs...")
	verifyCheckpointLoadedFromLogs(test, namespace, trainJobName)

	test.T().Log("JIT checkpoint verification completed successfully!")
}

// getTrainerStatus extracts trainerStatus annotation as parsed map
func getTrainerStatus(test Test, namespace, trainJobName string) map[string]interface{} {
	trainJob := TrainJob(test, namespace, trainJobName)(test)
	statusJSON := trainJob.GetAnnotations()[annotationTrainerStatus]
	if statusJSON == "" {
		return nil
	}
	var status map[string]interface{}
	if err := json.Unmarshal([]byte(statusJSON), &status); err != nil {
		return nil
	}
	return status
}

// getProgressPercentage extracts progressPercentage from trainerStatus annotation
func getProgressPercentage(test Test, namespace, trainJobName string) int {
	if status := getTrainerStatus(test, namespace, trainJobName); status != nil {
		if progress, ok := status["progressPercentage"].(float64); ok {
			return int(progress)
		}
	}
	return 0
}

// getCurrentEpoch extracts currentEpoch from trainerStatus annotation
func getCurrentEpoch(test Test, namespace, trainJobName string) float64 {
	if status := getTrainerStatus(test, namespace, trainJobName); status != nil {
		if epoch, ok := status["currentEpoch"].(float64); ok {
			return epoch
		}
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

// countPodsInPhase counts training pods matching any of the given phases
func countPodsInPhase(test Test, namespace, trainJobName string, phases ...corev1.PodPhase) int {
	count := 0
	for _, pod := range listTrainingPods(test, namespace, trainJobName) {
		for _, phase := range phases {
			if pod.Status.Phase == phase {
				count++
				break
			}
		}
	}
	return count
}

// countRunningPods returns the number of running training pods
func countRunningPods(test Test, namespace, trainJobName string) int {
	return countPodsInPhase(test, namespace, trainJobName, corev1.PodRunning)
}

// countActivePods returns the number of active (running or pending) training pods
func countActivePods(test Test, namespace, trainJobName string) int {
	return countPodsInPhase(test, namespace, trainJobName, corev1.PodRunning, corev1.PodPending)
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

// hasCompletedEpochFromLogs checks if training has completed at least 1 epoch by examining pod logs
// HuggingFace Trainer logs: {'loss': X, ..., 'epoch': 1.0} - epochPattern matches epoch >= 1
func hasCompletedEpochFromLogs(test Test, namespace, trainJobName string) bool {
	for _, pod := range listTrainingPods(test, namespace, trainJobName) {
		if pod.Status.Phase != corev1.PodRunning {
			continue
		}
		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)
		if epochPattern.MatchString(logs) {
			test.T().Logf("Epoch >= 1 detected in pod %s logs", pod.Name)
			return true
		}
	}
	return false
}

// verifyCheckpointLoadedFromLogs checks pod logs for checkpoint resume messages from Kubeflow SDK
func verifyCheckpointLoadedFromLogs(test Test, namespace, trainJobName string) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	test.Expect(len(pods)).NotTo(Equal(0), "No training pods found to verify checkpoint logs")

	// Checkpoint resume indicators from Kubeflow SDK (transformers.py)
	indicators := []string{"[Kubeflow] Found latest checkpoint:", "[Kubeflow] Auto-resuming from:"}

	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodSucceeded {
			continue
		}
		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)
		for _, indicator := range indicators {
			if strings.Contains(logs, indicator) {
				test.T().Logf("Checkpoint resume verified in pod %s: %s", pod.Name, indicator)
				return // Success - found checkpoint log
			}
		}
	}

	test.T().Fatal("Checkpoint resume log not found in any completed pod (expected '[Kubeflow] Auto-resuming from:')")
}

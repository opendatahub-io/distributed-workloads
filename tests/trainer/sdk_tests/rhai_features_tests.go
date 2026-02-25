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
	"strconv"
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
	rhaiFeaturesNotebookName        = "rhai_features.ipynb"
	rhaiFeaturesNotebookPath        = "resources/" + rhaiFeaturesNotebookName
	rhaiFsdpFullStateNotebookName   = "rhai_features_fsdp_full_state.ipynb"
	rhaiFsdpFullStateNotebookPath   = "resources/" + rhaiFsdpFullStateNotebookName
	rhaiFsdpSharedStateNotebookName = "rhai_features_fsdp_shared_state.ipynb"
	rhaiFsdpSharedStateNotebookPath = "resources/" + rhaiFsdpSharedStateNotebookName
	rhaiDeepspeedStage0NotebookName = "rhai_features_deepspeed_stage0.ipynb"
	rhaiDeepspeedStage0NotebookPath = "resources/" + rhaiDeepspeedStage0NotebookName

	// Annotation keys for progression tracking (must match SDK/training-operator constants)
	annotationProgressionTracking = "trainer.opendatahub.io/progression-tracking"
	annotationMetricsPort         = "trainer.opendatahub.io/metrics-port"
	annotationMetricsPollInterval = "trainer.opendatahub.io/metrics-poll-interval"
	annotationTrainerStatus       = "trainer.opendatahub.io/trainerStatus"
)

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
	NumNodes                  int         // Number of training nodes (default: 2)
	NumGpusPerNode            int         // GPUs per node for multi-GPU tests (default: 1)
	NotebookPath              string      // Path to notebook file (default: rhai_features.ipynb)
	NotebookName              string      // Name of notebook file (default: rhai_features.ipynb)
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
		NumNodes:                  2, // Default: 2 nodes
		NumGpusPerNode:            1, // Default: 1 GPU per node
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
		NumNodes:                  2, // Default: 2 nodes
		NumGpusPerNode:            1, // Default: 1 GPU per node
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
		NumNodes:                  2, // Default: 2 nodes
		NumGpusPerNode:            1, // Default: 1 GPU per node
	})
}

// RunRhaiFeaturesProgressionMultiGpuTest runs multi-GPU test with progression tracking only
func RunRhaiFeaturesProgressionMultiGpuTest(t *testing.T, accelerator Accelerator, numNodes, numGpusPerNode int) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: true,
		EnableJitCheckpoint:       false,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               accelerator,
		NumNodes:                  numNodes,
		NumGpusPerNode:            numGpusPerNode,
	})
}

// RunRhaiFeaturesCheckpointMultiGpuTest runs multi-GPU test with JIT checkpointing only
func RunRhaiFeaturesCheckpointMultiGpuTest(t *testing.T, accelerator Accelerator, numNodes, numGpusPerNode int) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: false,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               accelerator,
		NumNodes:                  numNodes,
		NumGpusPerNode:            numGpusPerNode,
	})
}

// RunRhaiFeaturesAllMultiGpuTest runs multi-GPU test with all RHAI features enabled
func RunRhaiFeaturesAllMultiGpuTest(t *testing.T, accelerator Accelerator, numNodes, numGpusPerNode int) {
	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: true,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       "/workspace/checkpoints",
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               accelerator,
		NumNodes:                  numNodes,
		NumGpusPerNode:            numGpusPerNode,
	})
}

// runS3CheckpointTestWithNotebook is a generic helper that sets up S3 storage and runs the checkpoint test with a custom notebook
func runS3CheckpointTestWithNotebook(t *testing.T, accelerator Accelerator, numNodes, numGpusPerNode int, notebookPath, notebookName string) {
	test := With(t)

	// Get S3 provider (validates credentials internally)
	provider, err := trainerutils.GetS3Provider()
	test.Expect(err).NotTo(HaveOccurred(), "S3 configuration required. Please set AWS_DEFAULT_ENDPOINT, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY")

	// Create unique bucket name using Unix timestamp for parallel test execution
	bucketName := fmt.Sprintf("%s-%d", trainerutils.ConstantBucketName, time.Now().Unix())

	// Get region from environment (CreateBucket will default to us-east-1 if empty)
	region, _ := GetStorageBucketDefaultRegion()

	// Create test bucket
	err = provider.CreateBucket(test.Ctx(), bucketName, region)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create test bucket")

	// Schedule cleanup to run when function exits (ensures bucket is deleted even if test fails)
	defer func() {
		if err := provider.DeleteBucket(test.Ctx(), bucketName); err != nil {
			test.T().Logf("Warning: failed to delete test bucket: %v", err)
		} else {
			test.T().Logf("Test bucket deleted: %s (all checkpoints cleaned)", bucketName)
		}
	}()

	if region != "" {
		test.T().Logf("Test bucket ready: %s (region: %s)", bucketName, region)
	} else {
		test.T().Logf("Test bucket ready: %s (region: us-east-1, default)", bucketName)
	}

	runRhaiFeaturesTestWithConfig(t, RhaiFeatureConfig{
		EnableProgressionTracking: false,
		EnableJitCheckpoint:       true,
		CheckpointOutputDir:       fmt.Sprintf("s3://%s/checkpoints", bucketName),
		CheckpointSaveStrategy:    "epoch",
		CheckpointSaveTotalLimit:  "3",
		Accelerator:               accelerator,
		NumNodes:                  numNodes,
		NumGpusPerNode:            numGpusPerNode,
		NotebookPath:              notebookPath,
		NotebookName:              notebookName,
	})
}

// RunRhaiS3CheckpointTest runs the e2e test for S3 checkpoint storage (CPU only, 2 nodes)
func RunRhaiS3CheckpointTest(t *testing.T, accelerator Accelerator) {
	runS3CheckpointTestWithNotebook(t, accelerator, 2, 1, rhaiFeaturesNotebookPath, rhaiFeaturesNotebookName)
}

// RunRhaiS3FsdpFullStateTest runs the e2e test for FSDP full state checkpoint (CPU only, 2 nodes)
func RunRhaiS3FsdpFullStateTest(t *testing.T, accelerator Accelerator) {
	runS3CheckpointTestWithNotebook(t, accelerator, 2, 1, rhaiFsdpFullStateNotebookPath, rhaiFsdpFullStateNotebookName)
}

// RunRhaiS3FsdpFullStateMultiProcessTest runs the e2e test for FSDP full state checkpoint with multi-process per node
func RunRhaiS3FsdpFullStateMultiProcessTest(t *testing.T, accelerator Accelerator, numNodes, numProcessesPerNode int) {
	runS3CheckpointTestWithNotebook(t, accelerator, numNodes, numProcessesPerNode, rhaiFsdpFullStateNotebookPath, rhaiFsdpFullStateNotebookName)
}

// RunRhaiS3FsdpSharedStateGpuTest runs the e2e test for FSDP shared state checkpoint (GPU required, 2 nodes, 1 GPU each)
func RunRhaiS3FsdpSharedStateGpuTest(t *testing.T, accelerator Accelerator) {
	runS3CheckpointTestWithNotebook(t, accelerator, 2, 1, rhaiFsdpSharedStateNotebookPath, rhaiFsdpSharedStateNotebookName)
}

// RunRhaiS3FsdpSharedStateMultiGpuTest runs the e2e test for FSDP shared state checkpoint (GPU required, multi-GPU per node)
func RunRhaiS3FsdpSharedStateMultiGpuTest(t *testing.T, accelerator Accelerator, numNodes, numProcessesPerNode int) {
	runS3CheckpointTestWithNotebook(t, accelerator, numNodes, numProcessesPerNode, rhaiFsdpSharedStateNotebookPath, rhaiFsdpSharedStateNotebookName)
}

// RunRhaiS3DeepspeedStage0GpuTest runs the e2e test for DeepSpeed Stage 0 checkpoint (GPU required, 2 nodes, 1 GPU each)
func RunRhaiS3DeepspeedStage0GpuTest(t *testing.T, accelerator Accelerator) {
	runS3CheckpointTestWithNotebook(t, accelerator, 2, 1, rhaiDeepspeedStage0NotebookPath, rhaiDeepspeedStage0NotebookName)
}

// RunRhaiS3DeepspeedStage0MultiGpuTest runs the e2e test for DeepSpeed Stage 0 checkpoint (GPU required, multi-GPU per node)
func RunRhaiS3DeepspeedStage0MultiGpuTest(t *testing.T, accelerator Accelerator, numNodes, numProcessesPerNode int) {
	runS3CheckpointTestWithNotebook(t, accelerator, numNodes, numProcessesPerNode, rhaiDeepspeedStage0NotebookPath, rhaiDeepspeedStage0NotebookName)
}

// runRhaiFeaturesTestWithConfig runs the e2e test with the given feature configuration
func runRhaiFeaturesTestWithConfig(t *testing.T, config RhaiFeatureConfig) {
	test := With(t)

	// Set defaults for notebook path/name if not specified
	if config.NotebookPath == "" {
		config.NotebookPath = rhaiFeaturesNotebookPath
	}
	if config.NotebookName == "" {
		config.NotebookName = rhaiFeaturesNotebookName
	}

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

	// Create ConfigMap with notebook and install script
	nb, err := os.ReadFile(config.NotebookPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", config.NotebookPath))

	// Read the kubeflow install helper script
	installScriptPath := "resources/disconnected_env/install_kubeflow.py"
	installScript, err := os.ReadFile(installScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", installScriptPath))

	cmData := map[string][]byte{
		config.NotebookName:   nb,
		"install_kubeflow.py": installScript,
	}
	cm := CreateConfigMap(test, namespace.Name, cmData)

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

	// S3/MinIO configuration for disconnected environments (optional)
	// Note: For S3 checkpoint tests, we create a separate bucket for checkpoints.
	// Models/datasets should come from HuggingFace (connected) or from AWS_STORAGE_BUCKET if set (disconnected).
	// We don't use the checkpoint bucket for models/datasets to avoid confusion.
	s3Endpoint, _ := GetStorageBucketDefaultEndpoint()
	s3AccessKey, _ := GetStorageBucketAccessKeyId()
	s3SecretKey, _ := GetStorageBucketSecretKey()

	// Get bucket from env for models/datasets (separate from checkpoint bucket)
	// For S3 checkpoint tests, checkpoint bucket is created dynamically and passed via CHECKPOINT_OUTPUT_DIR
	modelsBucket, _ := GetStorageBucketName()
	modelS3Prefix := os.Getenv("MODEL_S3_PREFIX")
	if modelS3Prefix == "" {
		modelS3Prefix = "models/distilgpt2"
	}
	datasetS3Prefix := os.Getenv("DATASET_S3_PREFIX")
	if datasetS3Prefix == "" {
		datasetS3Prefix = "alpaca-cleaned-datasets"
	}

	// Build S3 export commands for models/datasets (only if configured and bucket exists)
	// This is separate from checkpoint storage which uses its own bucket
	// Verify bucket exists before setting AWS_STORAGE_BUCKET to ensure notebook can access it
	s3Exports := ""
	if s3Endpoint != "" && modelsBucket != "" {
		// Verify bucket exists before using it
		provider, err := trainerutils.GetS3Provider()
		if err == nil {
			ctx := test.Ctx()
			exists, err := provider.BucketExists(ctx, modelsBucket)
			if err != nil {
				test.T().Logf("Warning: Failed to verify bucket existence for %s: %v. Skipping S3 mode for models/datasets.", modelsBucket, err)
			} else if !exists {
				test.T().Logf("Warning: Bucket %s does not exist. Skipping S3 mode for models/datasets. Will use HuggingFace.", modelsBucket)
			} else {
				test.T().Logf("S3 mode for models/datasets: endpoint=%s, bucket=%s", s3Endpoint, modelsBucket)
				s3Exports = fmt.Sprintf(
					"export AWS_DEFAULT_ENDPOINT='%s'; "+
						"export AWS_ACCESS_KEY_ID='%s'; "+
						"export AWS_SECRET_ACCESS_KEY='%s'; "+
						"export AWS_STORAGE_BUCKET='%s'; "+
						"export MODEL_S3_PREFIX='%s'; "+
						"export DATASET_S3_PREFIX='%s'; ",
					s3Endpoint, s3AccessKey, s3SecretKey, modelsBucket, modelS3Prefix, datasetS3Prefix,
				)
			}
		} else {
			test.T().Logf("Warning: Failed to create S3 provider to verify bucket: %v. Skipping S3 mode for models/datasets.", err)
		}
	}
	if s3Exports == "" {
		test.T().Log("HuggingFace mode: S3 not configured for models/datasets, will download from HF Hub")
	}

	// Create Data Connection secret for cloud checkpointing (if configured)
	// Automatically detects cloud storage from URI scheme (s3://, azure://, etc.)
	// Note: Data Connection is ONLY for checkpoints. The checkpoint bucket is extracted from CHECKPOINT_OUTPUT_DIR.
	// AWS_STORAGE_BUCKET (for models/datasets) is separate and handled in s3Exports above.
	var dataConnectionExports string
	checkpointURI := trainerutils.ParseCloudURI(config.CheckpointOutputDir)
	if checkpointURI != nil && checkpointURI.Scheme == "s3" && checkpointURI.Bucket != "" && s3Endpoint != "" && s3AccessKey != "" && s3SecretKey != "" {
		// Create Data Connection secret for S3 checkpoint storage
		secretData := map[string]string{
			"AWS_ACCESS_KEY_ID":     s3AccessKey,
			"AWS_SECRET_ACCESS_KEY": s3SecretKey,
			"AWS_S3_ENDPOINT":       s3Endpoint,
			"AWS_S3_BUCKET":         checkpointURI.Bucket,
		}

		secret := CreateSecret(test, namespace.Name, secretData)
		test.T().Logf("Created Data Connection secret: %s for cloud checkpoint storage", secret.Name)

		dataConnectionExports = fmt.Sprintf(
			"export DATA_CONNECTION_NAME='%s'; "+
				"export KUBEFLOW_INSTALL_FROM_GIT='true'; ",
			secret.Name,
		)
		test.T().Logf("Data Connection configured for cloud checkpointing: %s", config.CheckpointOutputDir)
	} else if checkpointURI != nil {
		test.T().Logf("Warning: Cloud storage URI detected (%s) but Data Connection not created (credentials may be missing or unsupported scheme)", config.CheckpointOutputDir)
	}

	// Determine GPU type from Accelerator.ResourceLabel (e.g., "nvidia.com/gpu" → "nvidia")
	gpuType := config.Accelerator.Type // "cpu" or "gpu"
	if config.Accelerator.ResourceLabel != "" {
		gpuType = strings.Split(config.Accelerator.ResourceLabel, ".")[0] // "nvidia" or "amd"
	}

	// kubeflow SDK is only on Red Hat PyPI indexes (not public PyPI)
	// install_kubeflow.py uses GPU_TYPE to select the correct index (cpu/cuda/rocm)
	test.T().Logf("Using Red Hat PyPI index for %s (kubeflow not on public PyPI)", gpuType)

	// Build pip exports - GPU_TYPE tells install_kubeflow.py which Red Hat index to use
	pipExports := fmt.Sprintf("export GPU_TYPE='%s'; ", gpuType)
	pipInstallFlags := ""

	// Set defaults for num_nodes and num_gpus_per_node if not specified
	numNodes := config.NumNodes
	if numNodes <= 0 {
		numNodes = 2
	}
	numGpusPerNode := config.NumGpusPerNode
	if numGpusPerNode <= 0 {
		numGpusPerNode = 1
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
			"export NUM_NODES='%d'; "+
			"export NUM_GPUS_PER_NODE='%d'; "+
			"%s"+ // S3 exports (if configured)
			"%s"+ // Data Connection exports (if configured)
			"%s"+ // PyPI/GPU_TYPE exports
			"python -m pip install --quiet --no-cache-dir %s papermill ipykernel boto3==1.34.162 && "+
			"python /opt/app-root/notebooks/install_kubeflow.py && "+
			"python -m ipykernel install --user --name=python3 && "+
			"python -m papermill /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"sleep infinity",
		GetOpenShiftApiUrl(test), userToken, namespace.Name, sharedPVC.Name,
		enableProgression,
		enableCheckpoint,
		config.CheckpointOutputDir,
		config.CheckpointSaveStrategy,
		config.CheckpointSaveTotalLimit,
		gpuResourceLabel,
		trainingRuntime,
		numNodes,
		numGpusPerNode,
		s3Exports,
		dataConnectionExports,
		pipExports,
		pipInstallFlags,
		config.NotebookName,
	)

	test.T().Logf("Feature config: ProgressionTracking=%v, JitCheckpoint=%v, Accelerator=%s, NumNodes=%d, NumGpusPerNode=%d, Notebook=%s",
		config.EnableProgressionTracking, config.EnableJitCheckpoint, config.Accelerator.Type, numNodes, numGpusPerNode, config.NotebookName)
	command := []string{"/bin/sh", "-c", shellCmd}

	// Create Notebook CR using the RWX PVC
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, config.NotebookName, 0, sharedPVC, common.ContainerSizeSmall)

	// Cleanup - use longer timeout due to large runtime images
	defer func() {
		// Clean up Kubernetes resources
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

		// Verify training completed successfully by checking pod termination message.
		// The termination message is written by the training process itself and is the authoritative source.
		// The trainerStatus annotation is operator-generated metadata that may lag due to polling intervals.
		test.T().Log("Verifying training completion via pod termination message (authoritative source)...")
		pods := listTrainingPods(test, namespace.Name, trainJobName)
		var found100PercentInTermination bool
		for _, pod := range pods {
			for _, containerStatus := range pod.Status.ContainerStatuses {
				if containerStatus.Name != "node" || containerStatus.State.Terminated == nil {
					continue
				}
				terminationMessage := containerStatus.State.Terminated.Message
				if terminationMessage == "" {
					continue
				}
				var terminationData map[string]interface{}
				if err := json.Unmarshal([]byte(terminationMessage), &terminationData); err != nil {
					continue
				}
				if termProgress, ok := terminationData["progressPercentage"].(float64); ok && termProgress >= 100 {
					found100PercentInTermination = true
					test.T().Logf("Found 100%% progress in termination message for pod %s", pod.Name)
					break
				}
			}
			if found100PercentInTermination {
				break
			}
		}
		test.Expect(found100PercentInTermination).To(BeTrue(), "Training should complete with 100% progress in termination message")

		// Get trainerStatus annotation to verify other metadata fields.
		// The annotation is operator-generated and may not reach exactly 100% due to polling timing,
		// but it provides additional metadata (steps, epochs, etc.) that we verify.
		test.T().Log("Verifying trainerStatus annotation metadata...")
		trainJob := TrainJob(test, namespace.Name, trainJobName)(test)
		annotations = trainJob.GetAnnotations()
		trainerStatusRaw := annotations[annotationTrainerStatus]
		test.Expect(trainerStatusRaw).NotTo(BeEmpty(), "trainerStatus annotation should not be empty")

		var trainerStatus map[string]interface{}
		err := json.Unmarshal([]byte(trainerStatusRaw), &trainerStatus)
		test.Expect(err).NotTo(HaveOccurred(), "trainerStatus should be valid JSON")
		test.T().Logf("trainerStatus: %s", trainerStatusRaw)

		// Verify progress metrics exist in annotation (for metadata verification)
		test.Expect(trainerStatus).To(HaveKey("progressPercentage"))
		progress := trainerStatus["progressPercentage"].(float64)
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

	// Wait for at least 2 epochs before suspending.
	// This ensures there's a complete previous checkpoint to fall back to when the
	// JIT checkpoint gets .incomplete marker (epoch 1 checkpoint is fully saved/uploaded).
	test.T().Log("Waiting for training to complete at least 2 epochs (checking logs)...")
	test.Eventually(func() bool {
		return hasCompletedEpochFromLogs(test, namespace, trainJobName, 2)
	}, TestTimeoutMedium, 5*time.Second).Should(BeTrue(), "Training should complete at least 2 epochs before suspension")
	test.T().Log("At least 2 epochs completed - ready to suspend")

	// Verify cloud checkpoint upload is working (only for cloud storage mode, not PVC)
	// This catches SDK monkey-patch failures early - if save_strategy override didn't apply,
	// no checkpoints are saved and no uploads happen
	checkpointURI := trainerutils.ParseCloudURI(checkpointDir)
	if checkpointURI != nil && checkpointURI.Scheme == "s3" && checkpointURI.Bucket != "" {
		test.T().Log("Step 1b: Verifying cloud checkpoint upload is working...")
		test.Eventually(func() bool {
			for _, pod := range listTrainingPods(test, namespace, trainJobName) {
				if pod.Status.Phase != corev1.PodRunning {
					continue
				}
				logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)
				// Check SDK applied the save_strategy override (monkey-patch working)
				if !strings.Contains(logs, "[Kubeflow] Applied save_strategy:") {
					test.T().Log("Waiting for SDK save_strategy override to appear in logs...")
					return false
				}
				// Check at least one checkpoint was uploaded to cloud storage (from logs)
				if strings.Contains(logs, "Upload complete") {
					test.T().Log("Cloud checkpoint upload confirmed in pod logs")
					return true
				}
			}
			return false
		}, TestTimeoutMedium, 5*time.Second).Should(BeTrue(),
			"Cloud checkpoint upload not detected in training pod logs. "+
				"Expected 'Upload complete' after epoch completion. "+
				"This usually means the SDK's checkpoint config override (save_strategy, output_dir) "+
				"was not applied to the Trainer.")
		test.T().Log("Cloud checkpoint upload verified in logs - verifying checkpoints exist in S3...")

		// Verify checkpoints actually exist in S3 (not just logs)
		provider, err := trainerutils.GetS3Provider()
		test.Expect(err).NotTo(HaveOccurred(), "Failed to get S3 provider for checkpoint verification")
		test.Eventually(func() bool {
			exists := provider.CheckpointExists(test.Ctx(), checkpointDir)
			if exists {
				test.T().Logf("Checkpoints verified in S3: %s", checkpointDir)
			} else {
				test.T().Logf("No checkpoints found in S3: %s", checkpointDir)
			}
			return exists
		}, TestTimeoutMedium, 5*time.Second).Should(BeTrue(),
			"Checkpoints not found in S3 storage. Expected checkpoint objects to exist in %s. "+
				"This verifies that the SDK's checkpoint upload functionality is working correctly.",
			checkpointDir)
		test.T().Log("Cloud checkpoint upload verified - checkpoints confirmed in S3 storage")
	}

	// Step 2: Suspend the TrainJob to trigger JIT checkpoint save
	test.T().Log("Step 2: Suspending TrainJob to trigger checkpoint save...")
	suspendTrainJob(test, namespace, trainJobName, true)

	// Step 3: Wait for pods to fully terminate (checkpoint saved before termination)
	test.T().Log("Step 3: Waiting for training pods to fully terminate...")
	test.Eventually(func() int {
		return countActivePods(test, namespace, trainJobName)
	}, TestTimeoutMedium, 5*time.Second).Should(Equal(0), "All training pods should terminate after suspend")
	test.T().Log("All training pods terminated - checkpoint should be saved")

	// Step 4: Verify job is suspended and not already completed
	// This ensures the suspend operation took effect before training finished
	test.T().Log("Step 4: Verifying job is suspended (not completed or failed)...")
	trainJob := TrainJob(test, namespace, trainJobName)(test)
	if TrainJobConditionComplete(trainJob) == metav1.ConditionTrue {
		test.T().Fatal("Training completed before suspend took effect - cannot verify JIT checkpoint functionality. Consider increasing dataset size or epochs.")
	}
	if TrainJobConditionFailed(trainJob) == metav1.ConditionTrue {
		test.T().Fatal("Training failed before suspend took effect - cannot verify JIT checkpoint functionality.")
	}
	test.Expect(TrainJobConditionSuspended(trainJob)).To(Equal(metav1.ConditionTrue), "TrainJob should be in suspended state")
	test.T().Log("TrainJob is suspended")

	// Step 5: Store progress before resume (only when progression tracking is enabled)
	var preSuspendProgress int
	var preSuspendEpoch float64
	if progressionEnabled {
		// Wait for operator to poll metrics and update TrainJob annotations.
		// This ensures progress is recorded before suspending the job.
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
	verifyCheckpointLoadedFromLogs(test, namespace, trainJobName, checkpointDir)

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
	// This ensures only the suspend field is updated without affecting other spec fields
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

// hasCompletedEpochFromLogs checks if training has completed the required number of epochs by examining pod logs
// HuggingFace Trainer logs: {'loss': X, ..., 'epoch': 1.0} - matches epochs >= minEpoch
func hasCompletedEpochFromLogs(test Test, namespace, trainJobName string, minEpoch int) bool {
	// Match epoch values in HuggingFace Trainer log format: 'epoch': N or 'epoch': N.M
	pattern := regexp.MustCompile(`'epoch':\s*(\d+)(?:\.\d+)?`)

	for _, pod := range listTrainingPods(test, namespace, trainJobName) {
		if pod.Status.Phase != corev1.PodRunning {
			continue
		}
		logs := PodLog(test, namespace, pod.Name, corev1.PodLogOptions{Container: "node"})(test)

		// Find all epoch values in logs and check if any meet the minimum threshold
		matches := pattern.FindAllStringSubmatch(logs, -1)
		for _, match := range matches {
			if len(match) > 1 {
				if epochVal, err := strconv.Atoi(match[1]); err == nil && epochVal >= minEpoch {
					test.T().Logf("Epoch %d (>= %d) detected in pod %s logs", epochVal, minEpoch, pod.Name)
					return true
				}
			}
		}
	}
	return false
}

// verifyCheckpointLoadedFromLogs checks pod logs for checkpoint resume messages from Kubeflow SDK.
// Uses mode-specific indicators: legacy local resume logs for PVC, cloud download logs for cloud storage.
func verifyCheckpointLoadedFromLogs(test Test, namespace, trainJobName, checkpointDir string) {
	test.T().Helper()

	pods := listTrainingPods(test, namespace, trainJobName)
	test.Expect(len(pods)).NotTo(Equal(0), "No training pods found to verify checkpoint logs")

	// Checkpoint resume indicators from Kubeflow SDK (transformers.py)
	indicators := []string{"Found latest checkpoint:", "Auto-resuming from:"}
	if strings.Contains(checkpointDir, "://") {
		indicators = []string{"Download complete"}
	}

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

	test.T().Fatalf("Checkpoint resume log not found in any completed pod (expected one of: %v)", indicators)
}

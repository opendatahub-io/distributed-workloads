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

package trainer

import (
	"testing"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	support "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	sdktests "github.com/opendatahub-io/distributed-workloads/tests/trainer/sdk_tests"
)

func TestKubeflowSdkSanity(t *testing.T) {
	Tags(t, Sanity)
	sdktests.RunFashionMnistCpuDistributedTraining(t)
}

func TestKubeflowSdkKueueIntegration(t *testing.T) {
	Tags(t, Sanity)
	test := support.With(t)
	support.SetupKueue(test, initialKueueState, support.TrainJobFramework)
	sdktests.RunFashionMnistKueueCpuDistributedTraining(t)
}

// TestOsftTrainingHubMultiNodeMultiGPU tests OSFT training using TrainingHubTrainer
func TestOsftTrainingHubMultiNodeMultiGPU(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 1)) // TODO: may need to be updated once https://issues.redhat.com/browse/RHOAIENG-30719 and https://issues.redhat.com/browse/RHOAIENG-24552 are resolved
	sdktests.RunOsftTrainingHubMultiGpuDistributedTraining(t)
}

// TestLoraTrainingHubMultiNodeMultiGPU tests Lora training using TrainingHubTrainer
func TestLoraTrainingHubMultiNodeMultiGPU(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 1)) // TODO: may need to be updated once https://issues.redhat.com/browse/RHOAIENG-30719 and https://issues.redhat.com/browse/RHOAIENG-24552 are resolved
	sdktests.RunLoraTrainingHubMultiGpuDistributedTraining(t)
}

// TestSftTrainingHubMultiNodeMultiGPU tests SFT training using TrainingHubTrainer
func TestSftTrainingHubMultiNodeMultiGPU(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 1))
	sdktests.RunSftTrainingHubMultiGpuDistributedTraining(t)
}

// CPU tests
func TestRhaiTrainingProgressionCPU(t *testing.T) {
	Tags(t, Tier1)
	sdktests.RunRhaiFeaturesProgressionTest(t, support.CPU)
}

func TestRhaiJitCheckpointingCPU(t *testing.T) {
	Tags(t, Tier1)
	sdktests.RunRhaiFeaturesCheckpointTest(t, support.CPU)
}

func TestRhaiFeaturesCPU(t *testing.T) {
	Tags(t, Tier1)
	sdktests.RunRhaiFeaturesAllTest(t, support.CPU)
}

// CUDA (NVIDIA) GPU tests - 2 nodes, 1 GPU each
func TestRhaiTrainingProgressionCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, support.NVIDIA))
	sdktests.RunRhaiFeaturesProgressionTest(t, support.NVIDIA)
}

func TestRhaiJitCheckpointingCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, support.NVIDIA))
	sdktests.RunRhaiFeaturesCheckpointTest(t, support.NVIDIA)
}

func TestRhaiFeaturesCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, support.NVIDIA))
	sdktests.RunRhaiFeaturesAllTest(t, support.NVIDIA)
}

// ROCm (AMD) GPU tests - 2 nodes, 1 GPU each
func TestRhaiTrainingProgressionRocm(t *testing.T) {
	Tags(t, KftoRocm, MultiNodeGpu(2, support.AMD))
	sdktests.RunRhaiFeaturesProgressionTest(t, support.AMD)
}

func TestRhaiJitCheckpointingRocm(t *testing.T) {
	Tags(t, KftoRocm, MultiNodeGpu(2, support.AMD))
	sdktests.RunRhaiFeaturesCheckpointTest(t, support.AMD)
}

func TestRhaiFeaturesRocm(t *testing.T) {
	Tags(t, KftoRocm, MultiNodeGpu(2, support.AMD))
	sdktests.RunRhaiFeaturesAllTest(t, support.AMD)
}

// Multi-GPU CUDA tests - 2 nodes, 2 GPUs each (requires 4 total NVIDIA GPUs)
func TestRhaiTrainingProgressionMultiGpuCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 2))
	sdktests.RunRhaiFeaturesProgressionMultiGpuTest(t, support.NVIDIA, 2, 2)
}

func TestRhaiJitCheckpointingMultiGpuCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 2))
	sdktests.RunRhaiFeaturesCheckpointMultiGpuTest(t, support.NVIDIA, 2, 2)
}

func TestRhaiFeaturesMultiGpuCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 2))
	sdktests.RunRhaiFeaturesAllMultiGpuTest(t, support.NVIDIA, 2, 2)
}

// Multi-GPU ROCm tests - 2 nodes, 2 GPUs each (requires 4 total AMD GPUs)
func TestRhaiTrainingProgressionMultiGpuRocm(t *testing.T) {
	Tags(t, KftoRocm, MultiNodeMultiGpu(2, support.AMD, 2))
	sdktests.RunRhaiFeaturesProgressionMultiGpuTest(t, support.AMD, 2, 2)
}

func TestRhaiJitCheckpointingMultiGpuRocm(t *testing.T) {
	Tags(t, KftoRocm, MultiNodeMultiGpu(2, support.AMD, 2))
	sdktests.RunRhaiFeaturesCheckpointMultiGpuTest(t, support.AMD, 2, 2)
}

func TestRhaiFeaturesMultiGpuRocm(t *testing.T) {
	Tags(t, KftoRocm, MultiNodeMultiGpu(2, support.AMD, 2))
	sdktests.RunRhaiFeaturesAllMultiGpuTest(t, support.AMD, 2, 2)
}

// S3 Checkpoint tests (CPU only, auto-skip if S3 not configured)
func TestRhaiS3CheckpointingCPU(t *testing.T) {
	Tags(t, Tier1)
	sdktests.RunRhaiS3CheckpointTest(t, support.CPU)
}

// FSDP Full State Checkpoint tests (CPU only, auto-skip if S3 not configured)
func TestRhaiS3FsdpFullStateCheckpointingCPU(t *testing.T) {
	Tags(t, Tier1)
	sdktests.RunRhaiS3FsdpFullStateTest(t, support.CPU)
}

// FSDP Full State Checkpoint tests (2 nodes, 2 processes per node)
func TestRhaiS3FsdpFullStateCheckpointingMultiProcess(t *testing.T) {
	Tags(t, Tier1)
	sdktests.RunRhaiS3FsdpFullStateMultiProcessTest(t, support.CPU, 2, 2)
}

// FSDP Shared State Checkpoint tests (GPU required, 2 nodes, 1 GPU each)
func TestRhaiS3FsdpSharedStateCheckpointingCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, support.NVIDIA))
	sdktests.RunRhaiS3FsdpSharedStateGpuTest(t, support.NVIDIA)
}

// FSDP Shared State Checkpoint tests (2 nodes, 2 GPUs per node)
func TestRhaiS3FsdpSharedStateCheckpointingMultiGpuCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 2))
	sdktests.RunRhaiS3FsdpSharedStateMultiGpuTest(t, support.NVIDIA, 2, 2)
}

// DeepSpeed Stage 0 Checkpoint tests (ZeRO Stage 0 - no sharding, GPU required)
func TestRhaiS3DeepspeedStage0CheckpointingCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, support.NVIDIA))
	sdktests.RunRhaiS3DeepspeedStage0GpuTest(t, support.NVIDIA)
}

// DeepSpeed Stage 0 Checkpoint tests (2 nodes, 2 GPUs per node)
func TestRhaiS3DeepspeedStage0CheckpointingMultiGpuCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 2))
	sdktests.RunRhaiS3DeepspeedStage0MultiGpuTest(t, support.NVIDIA, 2, 2)
}

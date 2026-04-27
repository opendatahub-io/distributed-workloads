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
	"errors"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

// ClusterTrainingRuntime represents a ClusterTrainingRuntime with its expected image name
type ClusterTrainingRuntime struct {
	Name  string
	Image string
}

const (
	// DefaultClusterTrainingRuntimeCUDA is the default runtime for CUDA accelerators
	DefaultClusterTrainingRuntimeCUDA = "torch-distributed"

	// DefaultClusterTrainingRuntimeROCm is the default runtime for AMD/ROCm accelerators
	DefaultClusterTrainingRuntimeROCm = "torch-distributed-rocm"

	// DefaultClusterTrainingRuntimeOpenMPICUDA is the default runtime for OpenMPI CUDA workloads
	DefaultClusterTrainingRuntimeOpenMPICUDA = "openmpi-cuda"

	// DefaultClusterTrainingRuntimeOpenMPICUDAImage is the expected image suffix for the OpenMPI CUDA runtime
	DefaultClusterTrainingRuntimeOpenMPICUDAImage = "odh-training-cuda130-torch210-py312-openmpi41"

	// DefaultTrainingHubRuntimeCUDA is the default CUDA runtime for training hub workloads
	DefaultTrainingHubRuntimeCUDA = "training-hub"

	// DefaultTrainingHubRuntimeCPU is the CPU-only runtime for training hub workloads
	DefaultTrainingHubRuntimeCPU = "training-hub-cpu"

	// DefaultTrainingHubRuntimeROCm is the default runtime for ROCm training hub workloads
	DefaultTrainingHubRuntimeROCm = "training-hub-rocm"

	// DefaultClusterTrainingRuntimeCPU is the default runtime for CPU-only torch-distributed workloads
	DefaultClusterTrainingRuntimeCPU = "torch-distributed-cpu"

	// DefaultClusterTrainingRuntimeOpenMPICUDA is the default runtime for OpenMPI CUDA workloads
	DefaultClusterTrainingRuntimeOpenMPICUDA = "openmpi-cuda"
)

var DefaultClusterTrainingRuntimes = []string{
	DefaultClusterTrainingRuntimeCUDA,
	DefaultClusterTrainingRuntimeROCm,
	DefaultClusterTrainingRuntimeCPU,
	DefaultClusterTrainingRuntimeOpenMPICUDA,
	DefaultTrainingHubRuntimeCUDA,
	DefaultTrainingHubRuntimeCPU,
	DefaultTrainingHubRuntimeROCm,
	DefaultClusterTrainingRuntimeOpenMPICUDA,
}

var mpiRuntimes = map[string]bool{
	DefaultClusterTrainingRuntimeOpenMPICUDA: true,
}

func IsMPIRuntime(name string) bool {
	return mpiRuntimes[name]
}

func IsDefaultRuntime(name string) bool {
	for _, defaultRuntime := range DefaultClusterTrainingRuntimes {
		if name == defaultRuntime {
			return true
		}
	}
	return false
}

// TrainingHubToDefaultClusterRuntime maps each Training Hub and pinned torch-distributed
// runtime (th06) to its corresponding DefaultClusterTrainingRuntime. Both runtimes in
// each pair are expected to have identical CTR specs (only metadata differs).
var TrainingHubToDefaultClusterRuntime = map[string]string{
	// Default (floating) runtimes
	DefaultTrainingHubRuntimeCUDA: DefaultClusterTrainingRuntimeCUDA,
	DefaultTrainingHubRuntimeCPU:  DefaultClusterTrainingRuntimeCPU,
	DefaultTrainingHubRuntimeROCm: DefaultClusterTrainingRuntimeROCm,
	// Pinned th06 runtimes (map to default runtimes; same image, identical spec)
	"training-hub-th06-cuda130-torch291-py312": DefaultClusterTrainingRuntimeCUDA,
	"training-hub-th06-cpu-torch291-py312":     DefaultClusterTrainingRuntimeCPU,
	"training-hub-th06-rocm64-torch291-py312":  DefaultClusterTrainingRuntimeROCm,
	"torch-distributed-cuda130-torch291-py312": DefaultClusterTrainingRuntimeCUDA,
	"torch-distributed-cpu-torch291-py312":     DefaultClusterTrainingRuntimeCPU,
	"torch-distributed-rocm64-torch291-py312":  DefaultClusterTrainingRuntimeROCm,
}

// ExpectedRuntimes is the list of expected ClusterTrainingRuntimes on the cluster
var ExpectedRuntimes = []ClusterTrainingRuntime{
	{Name: DefaultClusterTrainingRuntimeCUDA, Image: "odh-th06-cuda130-torch291-py312"},
	{Name: DefaultClusterTrainingRuntimeROCm, Image: "odh-th06-rocm64-torch291-py312"},
	{Name: DefaultClusterTrainingRuntimeCPU, Image: "odh-th06-cpu-torch291-py312"},
	{Name: DefaultClusterTrainingRuntimeOpenMPICUDA, Image: DefaultClusterTrainingRuntimeOpenMPICUDAImage},
	{Name: "torch-distributed-cuda128-torch29-py312", Image: "odh-training-cuda128-torch29-py312"},
	{Name: "torch-distributed-rocm64-torch29-py312", Image: "odh-training-rocm64-torch29-py312"},
	{Name: "torch-distributed-cuda130-torch291-py312", Image: "odh-th06-cuda130-torch291-py312"},
	{Name: "torch-distributed-rocm64-torch291-py312", Image: "odh-th06-rocm64-torch291-py312"},
	{Name: "torch-distributed-cpu-torch291-py312", Image: "odh-th06-cpu-torch291-py312"},
	{Name: DefaultTrainingHubRuntimeCUDA, Image: "odh-th06-cuda130-torch291-py312"},
	{Name: DefaultTrainingHubRuntimeCPU, Image: "odh-th06-cpu-torch291-py312"},
	{Name: DefaultTrainingHubRuntimeROCm, Image: "odh-th06-rocm64-torch291-py312"},
	{Name: "training-hub-th05-cuda128-torch29-py312", Image: "odh-training-cuda128-torch29-py312"},
	{Name: "training-hub-th06-cuda130-torch291-py312", Image: "odh-th06-cuda130-torch291-py312"},
	{Name: "training-hub-th06-cpu-torch291-py312", Image: "odh-th06-cpu-torch291-py312"},
	{Name: "training-hub-th06-rocm64-torch291-py312", Image: "odh-th06-rocm64-torch291-py312"},
	{Name: DefaultClusterTrainingRuntimeOpenMPICUDA, Image: "odh-training-cuda130-torch210-py312-openmpi41"},
}

// GetImageFromClusterTrainingRuntime retrieves the container image from the named ClusterTrainingRuntime
// on the cluster. Fails the test if the runtime is not found or has no image defined.
func GetImageFromClusterTrainingRuntime(test support.Test, runtimeName string) (string, error) {
	runtime, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Get(
		test.Ctx(),
		runtimeName,
		metav1.GetOptions{},
	)
	if err != nil {
		return "", fmt.Errorf("failed to get ClusterTrainingRuntime %q: %w", runtimeName, err)
	}
	for _, replicatedJob := range runtime.Spec.Template.Spec.ReplicatedJobs {
		for _, container := range replicatedJob.Template.Spec.Template.Spec.Containers {
			if container.Image != "" {
				test.T().Logf("Using image from ClusterTrainingRuntime %q: %s", runtimeName, container.Image)
				return container.Image, nil
			}
		}
	}
	return "", errors.New("no container image found in ClusterTrainingRuntime " + runtimeName)
}

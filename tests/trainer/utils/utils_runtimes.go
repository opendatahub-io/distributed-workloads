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

// ClusterTrainingRuntime represents a ClusterTrainingRuntime with its expected RHOAI image
type ClusterTrainingRuntime struct {
	Name       string
	RHOAIImage string
}

const (
	// DefaultClusterTrainingRuntime is the default runtime for CUDA accelerators
	DefaultClusterTrainingRuntime = "torch-distributed"

	// DefaultClusterTrainingRuntimeROCm is the default runtime for AMD/ROCm accelerators
	DefaultClusterTrainingRuntimeROCm = "torch-distributed-rocm"

	// DefaultTrainingHubRuntime is the default runtime for SFT/OSFT workloads
	DefaultTrainingHubRuntime = "training-hub"
)

var DefaultClusterTrainingRuntimes = []string{
	DefaultClusterTrainingRuntime,
	DefaultClusterTrainingRuntimeROCm,
	DefaultTrainingHubRuntime,
}

func IsDefaultRuntime(name string) bool {
	for _, defaultRuntime := range DefaultClusterTrainingRuntimes {
		if name == defaultRuntime {
			return true
		}
	}
	return false
}

// ExpectedRuntimes is the list of expected ClusterTrainingRuntimes on the cluster
var ExpectedRuntimes = []ClusterTrainingRuntime{
	{Name: DefaultClusterTrainingRuntime, RHOAIImage: "odh-training-cuda128-torch29-py312-rhel9"},
	{Name: DefaultClusterTrainingRuntimeROCm, RHOAIImage: "odh-training-rocm64-torch29-py312-rhel9"},
	{Name: "torch-distributed-cuda128-torch29-py312", RHOAIImage: "odh-training-cuda128-torch29-py312-rhel9"},
	{Name: "torch-distributed-rocm64-torch29-py312", RHOAIImage: "odh-training-rocm64-torch29-py312-rhel9"},
	{Name: DefaultTrainingHubRuntime, RHOAIImage: "odh-training-cuda128-torch29-py312-rhel9"},
	{Name: "training-hub-th05-cuda128-torch29-py312", RHOAIImage: "odh-training-cuda128-torch29-py312-rhel9"},
}

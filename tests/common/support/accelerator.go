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

package support

var (
	AMD    = Accelerator{Type: "gpu", ResourceLabel: "amd.com/gpu"}
	CPU    = Accelerator{Type: "cpu"}
	NVIDIA = Accelerator{Type: "gpu", ResourceLabel: "nvidia.com/gpu", PrometheusGpuUtilizationLabel: "DCGM_FI_DEV_GPU_UTIL"}
)

type Accelerator struct {
	Type                          string
	ResourceLabel                 string
	PrometheusGpuUtilizationLabel string
}

// Method to check if the accelerator is a GPU
func (a Accelerator) IsGpu() bool {
	return a != CPU
}

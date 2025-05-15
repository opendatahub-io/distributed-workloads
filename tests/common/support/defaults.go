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

const (
	RayVersion        = "2.35.0"
	RayImage          = "quay.io/modh/ray:2.35.0-py311-cu121"
	RayROCmImage      = "quay.io/modh/ray:2.35.0-py311-rocm62"
	RayTorchCudaImage = "quay.io/rhoai/ray:2.35.0-py311-cu121-torch24-fa26"
	RayTorchROCmImage = "quay.io/rhoai/ray:2.35.0-py311-rocm61-torch24-fa26"
	TrainingCudaImage = "quay.io/modh/training:py311-cuda124-torch251"
	TrainingROCmImage = "quay.io/modh/training:py311-rocm62-torch251"
)

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
	RayVersion                  = "2.35.0"
	RayImage                    = "quay.io/modh/ray:2.35.0-py311-cu121"
	RayROCmImage                = "quay.io/modh/ray:2.35.0-py311-rocm62"
	RayTorchCudaImage           = "quay.io/rhoai/ray:2.35.0-py311-cu121-torch24-fa26"
	RayTorchROCmImage           = "quay.io/rhoai/ray:2.35.0-py311-rocm61-torch24-fa26"
	TrainingCudaPyTorch241Image = "quay.io/rhoai/odh-training-cuda121-torch24-py311-rhel9:rhoai-3.4"
	TrainingCudaPyTorch251Image = "quay.io/rhoai/odh-training-cuda124-torch25-py311-rhel9:rhoai-3.4"
	TrainingCudaPyTorch28Image  = "quay.io/rhoai/odh-training-cuda128-torch28-py312-rhel9:rhoai-3.4"
	TrainingRocmPyTorch241Image = "quay.io/rhoai/odh-training-rocm62-torch24-py311-rhel9:rhoai-3.4"
	TrainingRocmPyTorch251Image = "quay.io/rhoai/odh-training-rocm62-torch25-py311-rhel9:rhoai-3.4"
	TrainingRocmPyTorch28Image  = "quay.io/rhoai/odh-training-rocm64-torch28-py312-rhel9:rhoai-3.4"

	// Operator RELATED_IMAGE env var names for training images
	RelatedImageTrainingCudaPyTorch241 = "RELATED_IMAGE_ODH_TRAINING_CUDA121_TORCH24_PY311_IMAGE"
	RelatedImageTrainingCudaPyTorch251 = "RELATED_IMAGE_ODH_TRAINING_CUDA124_TORCH25_PY311_IMAGE"
	RelatedImageTrainingCudaPyTorch28  = "RELATED_IMAGE_ODH_TRAINING_CUDA128_TORCH28_PY312_IMAGE"
	RelatedImageTrainingRocmPyTorch241 = "RELATED_IMAGE_ODH_TRAINING_ROCM62_TORCH24_PY311_IMAGE"
	RelatedImageTrainingRocmPyTorch251 = "RELATED_IMAGE_ODH_TRAINING_ROCM62_TORCH25_PY311_IMAGE"
	RelatedImageTrainingRocmPyTorch28  = "RELATED_IMAGE_ODH_TRAINING_ROCM64_TORCH28_PY312_IMAGE"
)

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

package support

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

var (
	SmallContainerResources = ContainerResources{
		Limits:   ResourceConfig{CPU: "2", Memory: "3Gi"},
		Requests: ResourceConfig{CPU: "1", Memory: "3Gi"},
	}
	MediumContainerResources = ContainerResources{
		Limits:   ResourceConfig{CPU: "6", Memory: "24Gi"},
		Requests: ResourceConfig{CPU: "3", Memory: "24Gi"},
	}
)

type ResourceConfig struct {
	CPU              string
	Memory           string
	GPUResourceLabel string // e.g., "nvidia.com/gpu", "amd.com/gpu", or ""
}

type ContainerResources struct {
	Limits   ResourceConfig
	Requests ResourceConfig
}

type ContainerSize string

const (
	ContainerSizeSmall  ContainerSize = "small"
	ContainerSizeMedium ContainerSize = "medium"
)

// ResourceRequirementsFor converts a test container size into Kubernetes
// resource requirements.
func ResourceRequirementsFor(test Test, containerSize ContainerSize) corev1.ResourceRequirements {
	var selected ContainerResources
	switch containerSize {
	case ContainerSizeSmall:
		selected = SmallContainerResources
	case ContainerSizeMedium:
		selected = MediumContainerResources
	default:
		test.T().Errorf("Unsupported container size: %s. Must be '%s' or '%s'. Hence using '%s' container size.",
			containerSize, ContainerSizeSmall, ContainerSizeMedium, ContainerSizeSmall)
		selected = SmallContainerResources
	}

	return corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse(selected.Limits.CPU),
			corev1.ResourceMemory: resource.MustParse(selected.Limits.Memory),
		},
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse(selected.Requests.CPU),
			corev1.ResourceMemory: resource.MustParse(selected.Requests.Memory),
		},
	}
}

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

package trainer

import (
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"

	support "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

// CreateNotebookDeployment creates the Notebook-style runner used by the
// Trainer SDK tests on top of the generic Deployment support helper.
func CreateNotebookDeployment(
	test support.Test,
	namespace *corev1.Namespace,
	command []string,
	configMapName string,
	pvc *corev1.PersistentVolumeClaim,
	containerSize support.ContainerSize,
	image string,
	env []corev1.EnvVar,
	options ...support.DeploymentOption,
) *appsv1.Deployment {
	env = append([]corev1.EnvVar{
		{Name: "PIP_INDEX_URL", Value: support.GetPipIndexURL()},
		{Name: "PIP_TRUSTED_HOST", Value: support.GetPipTrustedHost()},
	}, env...)

	return support.CreateDeployment(test, namespace, support.DeploymentConfig{
		Command:       command,
		Image:         image,
		ContainerName: "notebook",
		Env:           env,
		WorkingDir:    "/opt/app-root/src",
		Resources:     support.ResourceRequirementsFor(test, containerSize),
		VolumeMounts: []corev1.VolumeMount{
			{Name: "workspace", MountPath: "/opt/app-root/src"},
			{Name: "notebook-config", MountPath: "/opt/app-root/notebooks"},
		},
		Volumes: []corev1.Volume{
			{
				Name: "workspace",
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: pvc.Name},
				},
			},
			{
				Name: "notebook-config",
				VolumeSource: corev1.VolumeSource{
					ConfigMap: &corev1.ConfigMapVolumeSource{
						LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
					},
				},
			},
		},
	}, options...)
}

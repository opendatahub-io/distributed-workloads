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
	"testing"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreateDeployment(t *testing.T) {
	test := NewTest(t)
	namespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test-ns"}}
	pvc := &corev1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "workspace"}}

	deployment := CreateDeployment(
		test,
		namespace,
		DeploymentConfig{
			Command:       []string{"/bin/sh", "-c", "sleep infinity"},
			Image:         "quay.io/example/runner:latest",
			ContainerName: "runner",
			VolumeMounts: []corev1.VolumeMount{
				{Name: "workspace", MountPath: "/workspace"},
				{Name: "config", MountPath: "/config"},
			},
			Volumes: []corev1.Volume{
				{
					Name: "workspace",
					VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: pvc.Name},
					},
				},
				{
					Name: "config",
					VolumeSource: corev1.VolumeSource{
						ConfigMap: &corev1.ConfigMapVolumeSource{
							LocalObjectReference: corev1.LocalObjectReference{Name: "runner-config"},
						},
					},
				},
			},
		},
		WithDeploymentLabels(map[string]string{"kueue.x-k8s.io/queue-name": "queue"}),
	)

	test.Expect(deployment.GenerateName).To(gomega.Equal("dw-deployment-"))
	test.Expect(deployment.Labels["kueue.x-k8s.io/queue-name"]).To(gomega.Equal("queue"))
	test.Expect(deployment.Spec.Selector.MatchLabels).To(gomega.HaveKeyWithValue(
		DeploymentLabelKey, DeploymentLabelValue))
	test.Expect(deployment.Spec.Selector.MatchLabels[DeploymentInstanceLabelKey]).NotTo(gomega.BeEmpty())
	test.Expect(deployment.Spec.Template.Labels).To(gomega.Equal(deployment.Spec.Selector.MatchLabels))
	test.Expect(deployment.Spec.Template.Spec.ServiceAccountName).To(gomega.Equal("default"))
	test.Expect(deployment.Spec.Template.Spec.Volumes).To(gomega.ContainElements(
		gomega.HaveField("Name", gomega.Equal("workspace")),
		gomega.HaveField("Name", gomega.Equal("config")),
	))

	DeleteDeployment(test, namespace, deployment.Name)
}

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
	gomega "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
)

const (
	// DeploymentLabelKey/Value identify the pod running the test workload so
	// cleanup and pod discovery cannot match unrelated Deployments.
	DeploymentLabelKey   = "opendatahub.io/dw-deployment"
	DeploymentLabelValue = "runner"

	// DeploymentInstanceLabelKey identifies one Deployment and prevents pod
	// selectors from matching another Deployment in the namespace.
	DeploymentInstanceLabelKey = "opendatahub.io/dw-deployment-instance"
)

// DeploymentConfig contains the required workload configuration for a
// Deployment created by CreateDeployment.
type DeploymentConfig struct {
	Command       []string
	Image         string
	ContainerName string
	Env           []corev1.EnvVar
	WorkingDir    string
	Resources     corev1.ResourceRequirements
	VolumeMounts  []corev1.VolumeMount
	Volumes       []corev1.Volume
}

// DeploymentOption customizes the Deployment created by CreateDeployment.
type DeploymentOption func(*appsv1.Deployment)

// WithDeploymentLabels adds labels to the Deployment object. This is useful
// for integrations such as Kueue, which inspect labels on the owning object.
func WithDeploymentLabels(labels map[string]string) DeploymentOption {
	return func(deployment *appsv1.Deployment) {
		for key, value := range labels {
			deployment.Labels[key] = value
		}
	}
}

// CreateDeployment creates a one-replica Deployment for a test workload.
func CreateDeployment(
	test Test,
	namespace *corev1.Namespace,
	config DeploymentConfig,
	options ...DeploymentOption,
) *appsv1.Deployment {
	test.T().Helper()

	labels := map[string]string{
		DeploymentLabelKey:         DeploymentLabelValue,
		DeploymentInstanceLabelKey: string(uuid.NewUUID()),
	}
	deployment := &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: appsv1.SchemeGroupVersion.String(),
			Kind:       "Deployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "dw-deployment-",
			Namespace:    namespace.Name,
			Labels:       copyLabels(labels),
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: Ptr(int32(1)),
			Selector: &metav1.LabelSelector{MatchLabels: copyLabels(labels)},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: copyLabels(labels)},
				Spec: corev1.PodSpec{
					Affinity: &corev1.Affinity{
						NodeAffinity: &corev1.NodeAffinity{
							PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
								{
									Weight: 1,
									Preference: corev1.NodeSelectorTerm{
										MatchExpressions: []corev1.NodeSelectorRequirement{
											{Key: "nvidia.com/gpu.present", Operator: corev1.NodeSelectorOpNotIn, Values: []string{"true"}},
										},
									},
								},
							},
						},
					},
					Containers: []corev1.Container{
						{
							Name:            config.ContainerName,
							Image:           config.Image,
							ImagePullPolicy: corev1.PullAlways,
							Command:         config.Command,
							Env:             config.Env,
							Resources:       config.Resources,
							WorkingDir:      config.WorkingDir,
							VolumeMounts:    config.VolumeMounts,
						},
					},
					RestartPolicy:      corev1.RestartPolicyAlways,
					ServiceAccountName: "default",
					EnableServiceLinks: Ptr(false),
					Volumes:            config.Volumes,
				},
			},
		},
	}

	for _, option := range options {
		option(deployment)
	}

	created, err := test.Client().Core().AppsV1().Deployments(namespace.Name).Create(test.Ctx(), deployment, metav1.CreateOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.T().Logf("Created test Deployment %s/%s", created.Namespace, created.Name)
	return created
}

func copyLabels(labels map[string]string) map[string]string {
	copy := make(map[string]string, len(labels))
	for key, value := range labels {
		copy[key] = value
	}
	return copy
}

// DeleteDeployment deletes the named test Deployment and waits for the object
// to disappear. Deletion is idempotent so it can safely be used from cleanup.
func DeleteDeployment(test Test, namespace *corev1.Namespace, deploymentName string) {
	test.T().Helper()
	err := test.Client().Core().AppsV1().Deployments(namespace.Name).Delete(test.Ctx(), deploymentName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		test.Expect(err).NotTo(gomega.HaveOccurred())
	}
	test.Eventually(func() bool {
		_, err := test.Client().Core().AppsV1().Deployments(namespace.Name).Get(test.Ctx(), deploymentName, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, TestTimeoutLong).Should(gomega.BeTrue(), "test Deployment %s was not deleted", deploymentName)
}

// WaitForDeploymentPodRunning waits for the named Deployment pod
// and returns its name and primary container name.
func WaitForDeploymentPodRunning(test Test, namespace, deploymentName string) (string, string) {
	test.T().Helper()
	deployment, err := test.Client().Core().AppsV1().Deployments(namespace).Get(test.Ctx(), deploymentName, metav1.GetOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
	labelSelector := metav1.FormatLabelSelector(deployment.Spec.Selector)
	test.Eventually(func() []corev1.Pod {
		return GetPods(test, namespace, metav1.ListOptions{
			LabelSelector: labelSelector,
			FieldSelector: "status.phase=Running",
		})
	}, TestTimeoutLong).Should(gomega.HaveLen(1), "Expected exactly one Trainer SDK deployment pod")

	pods := GetPods(test, namespace, metav1.ListOptions{
		LabelSelector: labelSelector,
		FieldSelector: "status.phase=Running",
	})
	return pods[0].Name, pods[0].Spec.Containers[0].Name
}

func Deployments(test Test, namespace *corev1.Namespace) func(g gomega.Gomega) []*appsv1.Deployment {
	return func(g gomega.Gomega) []*appsv1.Deployment {
		deployments, err := test.Client().Core().AppsV1().Deployments(namespace.Name).List(test.Ctx(), metav1.ListOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())

		result := make([]*appsv1.Deployment, 0, len(deployments.Items))
		for index := range deployments.Items {
			result = append(result, &deployments.Items[index])
		}
		return result
	}
}

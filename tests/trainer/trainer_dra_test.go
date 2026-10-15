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

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

func TestDraFieldsOnTrainJobReachPods(t *testing.T) {
	Tags(t, Smoke)
	test := With(t)
	namespace := test.NewTestNamespace().Name
	const claimName = "test-device"

	claimTemplate, err := test.Client().Core().ResourceV1().ResourceClaimTemplates(namespace).Create(
		test.Ctx(),
		&resourcev1.ResourceClaimTemplate{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "test-dra-"},
			Spec: resourcev1.ResourceClaimTemplateSpec{
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{{
							Name: claimName,
							Exactly: &resourcev1.ExactDeviceRequest{
								DeviceClassName: "test-dra-device",
								AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
								Count:           1,
							},
						}},
					},
				},
			},
		}, metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred())

	trainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(), &trainerv1alpha1.TrainJob{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "test-dra-trainjob-"},
			Spec: trainerv1alpha1.TrainJobSpec{
				RuntimeRef: trainerv1alpha1.RuntimeRef{
					Name: trainerutils.DefaultClusterTrainingRuntimeCUDA,
				},
				Trainer: &trainerv1alpha1.Trainer{
					NumNodes: Ptr(int32(2)),
					Command:  []string{"sleep", "60"},
					ResourceClaimsPerNode: []trainerv1alpha1.TrainerResourceClaim{{
						Name:                      claimName,
						ResourceClaimTemplateName: claimTemplate.Name,
					}},
				},
			},
		}, metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred())

	// Pods can remain Pending: this only checks propagation and needs no DRA driver.
	test.Eventually(func(g Gomega) {
		pods := Pods(test, namespace, metav1.ListOptions{
			LabelSelector: "jobset.sigs.k8s.io/jobset-name=" + trainJob.Name,
		})(g)
		g.Expect(pods).To(HaveLen(2))
		for _, pod := range pods {
			g.Expect(pod.Spec.ResourceClaims).To(ContainElement(corev1.PodResourceClaim{
				Name:                      claimName,
				ResourceClaimTemplateName: Ptr(claimTemplate.Name),
			}), "pod %s should reference the resource claim template", pod.Name)
			g.Expect(pod.Spec.Containers).To(ContainElement(And(
				HaveField("Name", "node"),
				HaveField("Resources.Claims", ContainElement(corev1.ResourceClaim{Name: claimName})),
			)), "pod %s node container should reference the resource claim", pod.Name)
		}
	}, TestTimeoutMedium).Should(Succeed())
}

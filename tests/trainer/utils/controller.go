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
	"time"

	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	TrainerControllerDeployment = "kubeflow-trainer-controller-manager"
	TrainerControllerService    = "kubeflow-trainer-controller-manager"
)

// GetTrainerControllerDeployment returns the installed Trainer controller deployment.
func GetTrainerControllerDeployment(test Test) *appsv1.Deployment {
	test.T().Helper()
	namespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())
	deployment, err := test.Client().Core().AppsV1().Deployments(namespace).Get(
		test.Ctx(), TrainerControllerDeployment, metav1.GetOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred())
	return deployment
}

// GetTrainerControllerService returns the Trainer controller's service.
func GetTrainerControllerService(test Test) *corev1.Service {
	test.T().Helper()
	namespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())
	service, err := test.Client().Core().CoreV1().Services(namespace).Get(
		test.Ctx(), TrainerControllerService, metav1.GetOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred())
	return service
}

// TrainerControllerSelector returns the installed Trainer deployment's selector.
func TrainerControllerSelector(test Test) string {
	test.T().Helper()
	return trainerControllerSelector(test, GetTrainerControllerDeployment(test))
}

func trainerControllerSelector(test Test, deployment *appsv1.Deployment) string {
	test.Expect(deployment.Spec.Selector).NotTo(BeNil(), "Trainer deployment has no selector")
	parsed, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(parsed.Empty()).To(BeFalse(), "Trainer deployment has an empty selector")
	return parsed.String()
}

// PodReady reports the pod's Ready condition.
func PodReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

// GetTrainerControllerPod waits for one Running, Ready controller pod with a pod IP.
func GetTrainerControllerPod(test Test, timeout time.Duration) *corev1.Pod {
	test.T().Helper()
	deployment := GetTrainerControllerDeployment(test)
	namespace := deployment.Namespace
	options := metav1.ListOptions{LabelSelector: trainerControllerSelector(test, deployment)}
	var readyPod *corev1.Pod
	test.Eventually(func(g Gomega) {
		readyPod = nil
		for _, pod := range Pods(test, namespace, options)(g) {
			if pod.Status.Phase == corev1.PodRunning && PodReady(&pod) && pod.Status.PodIP != "" && pod.DeletionTimestamp == nil {
				readyPod = pod.DeepCopy()
				return
			}
		}
		g.Expect(readyPod).NotTo(BeNil(), "Trainer controller pod is not ready")
	}, timeout, 5*time.Second).WithContext(test.Ctx()).Should(Succeed())
	test.T().Logf("Trainer controller pod %s is Ready", readyPod.Name)
	return readyPod
}

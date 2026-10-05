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
	"fmt"
	"time"

	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const TrainerControllerDeployment = "kubeflow-trainer-controller-manager"

// GetTrainerControllerDeployment returns the installed Trainer controller deployment.
func GetTrainerControllerDeployment(test Test, namespace string) *unstructured.Unstructured {
	test.T().Helper()
	deployment, err := test.Client().Dynamic().Resource(schema.GroupVersionResource{
		Group: "apps", Version: "v1", Resource: "deployments",
	}).Namespace(namespace).Get(test.Ctx(), TrainerControllerDeployment, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	return deployment
}

// TrainerControllerSelector includes both matchLabels and matchExpressions.
func TrainerControllerSelector(test Test, deployment *unstructured.Unstructured) string {
	test.T().Helper()
	selectorMap, found, err := unstructured.NestedMap(deployment.Object, "spec", "selector")
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(found).To(BeTrue(), "Trainer deployment has no selector")
	var selector metav1.LabelSelector
	test.Expect(runtime.DefaultUnstructuredConverter.FromUnstructured(selectorMap, &selector)).To(Succeed())
	parsed, err := metav1.LabelSelectorAsSelector(&selector)
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(parsed.Empty()).To(BeFalse(), "Trainer deployment has an empty selector")
	return parsed.String()
}

// TrainerControllerPort resolves a declared manager port; missing names are errors.
func TrainerControllerPort(deployment *unstructured.Unstructured, portName string) (int32, error) {
	var typed appsv1.Deployment
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(deployment.Object, &typed); err != nil {
		return 0, err
	}
	for _, container := range typed.Spec.Template.Spec.Containers {
		if container.Name == "manager" {
			for _, port := range container.Ports {
				if port.Name == portName {
					return port.ContainerPort, nil
				}
			}
		}
	}
	return 0, fmt.Errorf("Trainer deployment %s/%s manager container has no port named %q", deployment.GetNamespace(), deployment.GetName(), portName)
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

// TrainerControllerPod waits for one Running, Ready controller pod with a pod IP.
func TrainerControllerPod(test Test, namespace string, deployment *unstructured.Unstructured, timeout time.Duration) *corev1.Pod {
	test.T().Helper()
	options := metav1.ListOptions{LabelSelector: TrainerControllerSelector(test, deployment)}
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

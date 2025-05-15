/*
Copyright 2023.

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
	"fmt"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func createTestNamespace(t Test, options ...Option[*corev1.Namespace]) *corev1.Namespace {
	t.T().Helper()
	namespace := &corev1.Namespace{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "Namespace",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-ns-",
		},
	}

	for _, option := range options {
		t.Expect(option.ApplyTo(namespace)).To(gomega.Succeed())
	}

	namespace, err := t.Client().Core().CoreV1().Namespaces().Create(t.Ctx(), namespace, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())

	return namespace
}

func deleteTestNamespace(t Test, namespace *corev1.Namespace) {
	t.T().Helper()
	propagationPolicy := metav1.DeletePropagationBackground
	err := t.Client().Core().CoreV1().Namespaces().Delete(t.Ctx(), namespace.Name, metav1.DeleteOptions{
		PropagationPolicy: &propagationPolicy,
	})
	t.Expect(err).NotTo(gomega.HaveOccurred())
}

func CreateTestNamespaceWithName(t Test, namespaceName string, options ...Option[*corev1.Namespace]) *corev1.Namespace {
	t.T().Helper()
	namespace := &corev1.Namespace{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "Namespace",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: namespaceName,
		},
	}

	for _, option := range options {
		t.Expect(option.ApplyTo(namespace)).To(gomega.Succeed())
	}

	namespace, err := t.Client().Core().CoreV1().Namespaces().Create(t.Ctx(), namespace, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())

	return namespace
}

func GetNamespaceWithName(t Test, namespaceName string) *corev1.Namespace {
	t.T().Helper()
	namespace, err := t.Client().Core().CoreV1().Namespaces().Get(t.Ctx(), namespaceName, metav1.GetOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred(), fmt.Sprintf("Failed to retrieve namespace with name: %s", namespaceName))
	return namespace
}

func DeleteTestNamespace(t Test, namespace *corev1.Namespace) {
	t.T().Helper()
	propagationPolicy := metav1.DeletePropagationBackground
	StoreNamespaceLogs(t, namespace)
	err := t.Client().Core().CoreV1().Namespaces().Delete(t.Ctx(), namespace.Name, metav1.DeleteOptions{
		PropagationPolicy: &propagationPolicy,
	})
	t.Expect(err).NotTo(gomega.HaveOccurred())
}

func StoreNamespaceLogs(t Test, namespace *corev1.Namespace) {
	storeAllPodLogs(t, namespace)
	storeEvents(t, namespace)
}

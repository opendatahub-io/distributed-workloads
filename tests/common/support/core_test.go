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

import (
	"testing"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetPods(t *testing.T) {
	test := NewTest(t)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
	}

	test.client.Core().CoreV1().Pods("test-namespace").Create(test.ctx, pod, metav1.CreateOptions{})

	// Call the GetPods function with the fake client and namespace
	pods := GetPods(test, "test-namespace", metav1.ListOptions{})

	test.Expect(pods).Should(gomega.HaveLen(1), "Expected 1 pod, but got %d", len(pods))
	test.Expect(pods[0].Name).To(gomega.Equal("test-pod"), "Expected pod name 'test-pod', but got '%s'", pods[0].Name)
}

func TestGetNodes(t *testing.T) {
	test := NewTest(t)
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
	}

	test.client.Core().CoreV1().Nodes().Create(test.ctx, node, metav1.CreateOptions{})
	nodes := GetNodes(test)

	test.Expect(nodes).Should(gomega.HaveLen(1), "Expected 1 node, but got %d", len(nodes))
	test.Expect(nodes[0].Name).To(gomega.Equal("test-node"), "Expected node name 'test-node', but got '%s'", nodes[0].Name)

}

func TestResourceName(t *testing.T) {
	type TestStruct struct {
		Name string
	}
	test := NewTest(t)

	obj := TestStruct{Name: "test-resource"}
	resourceName, err := ResourceName(obj)

	test.Expect(err).To(gomega.BeNil(), "Expected no error, but got '%v'", err)
	test.Expect(resourceName).To(gomega.Equal("test-resource"), "Expected resource name 'test-resource', but got '%s'", resourceName)
}

func TestGetServiceAccount(t *testing.T) {
	test := NewTest(t)

	createdSa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-sa",
			Namespace: "my-namespace",
		},
	}

	test.client.Core().CoreV1().ServiceAccounts("my-namespace").Create(test.ctx, createdSa, metav1.CreateOptions{})
	sa := GetServiceAccount(test, "my-namespace", "my-sa")

	test.Expect(sa.Name).To(gomega.Equal("my-sa"))
	test.Expect(sa.Namespace).To(gomega.Equal("my-namespace"))
}

func TestGetServiceAccounts(t *testing.T) {
	test := NewTest(t)

	createdSa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-sa-1",
			Namespace: "my-namespace",
		},
	}

	test.client.Core().CoreV1().ServiceAccounts("my-namespace").Create(test.ctx, createdSa, metav1.CreateOptions{})
	sas := GetServiceAccounts(test, "my-namespace")

	test.Expect(len(sas)).To(gomega.Equal(1))
	test.Expect(sas[0].Name).To(gomega.Equal("my-sa-1"))
	test.Expect(sas[0].Namespace).To(gomega.Equal("my-namespace"))
}

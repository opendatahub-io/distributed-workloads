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
	"strings"

	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var ingressConfigResource = schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "ingresses"}
var infrastructureConfigResource = schema.GroupVersionResource{Group: "config.openshift.io", Version: "v1", Resource: "infrastructures"}

func GetOpenShiftIngressDomain(test Test) string {
	test.T().Helper()

	cluster, err := test.Client().Dynamic().Resource(ingressConfigResource).Get(test.Ctx(), "cluster", metav1.GetOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())

	ingressDomain, found, err := unstructured.NestedString(cluster.UnstructuredContent(), "spec", "domain")
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.Expect(found).To(gomega.BeTrue())

	test.T().Logf("Ingress domain: %s", ingressDomain)
	return ingressDomain
}

func GetOpenShiftApiUrl(test Test) string {
	test.T().Helper()

	cluster, err := test.Client().Dynamic().Resource(infrastructureConfigResource).Get(test.Ctx(), "cluster", metav1.GetOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())

	openShiftApiUrl, found, err := unstructured.NestedString(cluster.UnstructuredContent(), "status", "apiServerURL")
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.Expect(found).To(gomega.BeTrue())

	test.T().Logf("OpenShift API URL: %s", openShiftApiUrl)
	return openShiftApiUrl
}

// GetExpectedImagePrefix returns the expected container image prefix (registry + org)
// by inspecting the rhods-operator deployment in the redhat-ods-operator namespace.
// Returns "registry.redhat.io/rhoai" for RHOAI builds or "quay.io/opendatahub" for ODH builds.
func GetExpectedImagePrefix(test Test) string {
	test.T().Helper()

	pods := GetPods(test, "redhat-ods-operator", metav1.ListOptions{
		FieldSelector: "status.phase=Running",
	})

	for _, pod := range pods {
		if strings.HasPrefix(pod.Name, "rhods-operator-") {
			test.Expect(pod.Spec.Containers).NotTo(gomega.BeEmpty(),
				"rhods-operator pod %s has no containers", pod.Name)
			image := pod.Spec.Containers[0].Image
			parts := strings.SplitN(image, "/", 3)
			if len(parts) >= 3 {
				prefix := parts[0] + "/" + parts[1]
				test.T().Logf("Detected operator image prefix: %s", prefix)
				return prefix
			}
		}
	}

	test.T().Fatal("No running rhods-operator pod found in redhat-ods-operator namespace")
	return ""
}

// GetExpectedRegistry returns the expected container registry
// ("registry.redhat.io" for RHOAI builds, "quay.io" for ODH builds).
func GetExpectedRegistry(test Test) string {
	test.T().Helper()
	prefix := GetExpectedImagePrefix(test)
	return strings.SplitN(prefix, "/", 2)[0]
}

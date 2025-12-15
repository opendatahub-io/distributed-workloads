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

func GetExpectedRegistry(test Test, odhNamespace string) string {
	test.T().Helper()

	var registryName string
	switch odhNamespace {
	case "redhat-ods-applications":
		registryName = "registry.redhat.io"
	case "opendatahub":
		registryName = "quay.io"
	default:
		test.T().Fatalf("Unknown ODH namespace: %s. Expected 'redhat-ods-applications' for RHOAI operator or 'opendatahub' ODH operator.", odhNamespace)
	}

	test.T().Logf("ODH namespace '%s' is using registry: %s", odhNamespace, registryName)
	return registryName
}

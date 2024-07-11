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

package odh

import (
	"bytes"

	gomega "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"
)

const recommendedTagAnnotation = "opendatahub.io/workbench-image-recommended"

var notebookResource = schema.GroupVersionResource{Group: "kubeflow.org", Version: "v1", Resource: "notebooks"}

type NotebookProps struct {
	IngressDomain             string
	OpenShiftApiUrl           string
	KubernetesUserBearerToken string
	Namespace                 string
	OpenDataHubNamespace      string
	ImageStreamName           string
	ImageStreamTag            string
	RayImage                  string
	NotebookImage             string
	LocalQueue                string
	NotebookConfigMapName     string
	NotebookConfigMapFileName string
	NotebookPVC               string
	NumGpus                   int
}

func createNotebook(test Test, namespace *corev1.Namespace, notebookUserToken, localQueue, jupyterNotebookConfigMapName, jupyterNotebookConfigMapFileName string, numGpus int) {
	// Create PVC for Notebook
	notebookPVC := CreatePersistentVolumeClaim(test, namespace.Name, "10Gi", corev1.ReadWriteOnce)

	// Read the Notebook CR from resources and perform replacements for custom values using go template
	notebookProps := NotebookProps{
		IngressDomain:             GetOpenShiftIngressDomain(test),
		OpenShiftApiUrl:           GetOpenShiftApiUrl(test),
		KubernetesUserBearerToken: notebookUserToken,
		Namespace:                 namespace.Name,
		OpenDataHubNamespace:      GetOpenDataHubNamespace(test),
		RayImage:                  GetRayImage(),
		NotebookImage:             "quay.io/modh/odh-generic-data-science-notebook@sha256:bb33abc67af1328d3b32899f58bcdc0cf1681605e1b5da57f8fe8da81523a9bd",
		LocalQueue:                localQueue,
		NotebookConfigMapName:     jupyterNotebookConfigMapName,
		NotebookConfigMapFileName: jupyterNotebookConfigMapFileName,
		NotebookPVC:               notebookPVC.Name,
		NumGpus:                   numGpus,
	}
	notebookTemplate, err := files.ReadFile("resources/custom-nb-small.yaml")
	test.Expect(err).NotTo(gomega.HaveOccurred())

	parsedNotebookTemplate := ParseTemplate(test, notebookTemplate, notebookProps)

	// Create Notebook CR
	notebookCR := &unstructured.Unstructured{}
	err = yaml.NewYAMLOrJSONDecoder(bytes.NewBuffer(parsedNotebookTemplate), 8192).Decode(notebookCR)
	test.Expect(err).NotTo(gomega.HaveOccurred())
	_, err = test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Create(test.Ctx(), notebookCR, metav1.CreateOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
}

func deleteNotebook(test Test, namespace *corev1.Namespace) {
	err := test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Delete(test.Ctx(), "jupyter-nb-kube-3aadmin", metav1.DeleteOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
}

func listNotebooks(test Test, namespace *corev1.Namespace) []*unstructured.Unstructured {
	ntbs, err := test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())

	ntbsp := []*unstructured.Unstructured{}
	for _, v := range ntbs.Items {
		ntbsp = append(ntbsp, &v)
	}

	return ntbsp
}

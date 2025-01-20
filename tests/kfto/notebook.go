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

package kfto

import (
	"bytes"
	"os"

	"github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"
)

const (
	NOTEBOOK_POD_NAME       = "jupyter-nb-kube-3aadmin-0"
	NOTEBOOK_CONTAINER_NAME = "jupyter-nb-kube-3aadmin"
)

var notebookResource = schema.GroupVersionResource{Group: "kubeflow.org", Version: "v1", Resource: "notebooks"}

type NotebookProps struct {
	IngressDomain             string
	OpenShiftApiUrl           string
	KubernetesUserBearerToken string
	Namespace                 string
	OpenDataHubNamespace      string
	NotebookImage             string
	NotebookConfigMapName     string
	NotebookConfigMapFileName string
	NotebookPVC               string
	NumGpus                   int
	PipIndexUrl               string
	PipTrustedHost            string
}

func createNotebook(test Test, namespace *corev1.Namespace, notebookUserToken, jupyterNotebookConfigMapName, jupyterNotebookConfigMapFileName string, numGpus int) {
	// Create PVC for Notebook
	notebookPVC := CreatePersistentVolumeClaim(test, namespace.Name, "10Gi", corev1.ReadWriteOnce)
	// Read the Notebook CR from resources and perform replacements for custom values using go template
	notebookProps := NotebookProps{
		IngressDomain:             GetOpenShiftIngressDomain(test),
		OpenShiftApiUrl:           GetOpenShiftApiUrl(test),
		KubernetesUserBearerToken: notebookUserToken,
		Namespace:                 namespace.Name,
		OpenDataHubNamespace:      GetOpenDataHubNamespace(test),
		NotebookImage:             GetNotebookImage(test),
		NotebookConfigMapName:     jupyterNotebookConfigMapName,
		NotebookConfigMapFileName: jupyterNotebookConfigMapFileName,
		NotebookPVC:               notebookPVC.Name,
		NumGpus:                   numGpus,
		PipIndexUrl:               GetPipIndexURL(),
		PipTrustedHost:            GetPipTrustedHost(),
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

const (
	// The environment variable for namespace where ODH is installed to.
	odhNamespaceEnvVar = "ODH_NAMESPACE"
	// Name of the authenticated Notebook user
	notebookUserName = "NOTEBOOK_USER_NAME"
	// Token of the authenticated Notebook user
	notebookUserToken = "NOTEBOOK_USER_TOKEN"
	// Image of the Notebook
	notebookImage = "NOTEBOOK_IMAGE"
)

func GetOpenDataHubNamespace(t Test) string {
	ns, ok := os.LookupEnv(odhNamespaceEnvVar)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify namespace where ODH is installed to.", odhNamespaceEnvVar)
	}
	return ns
}

func GetNotebookUserName(t Test) string {
	name, ok := os.LookupEnv(notebookUserName)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify name of the authenticated Notebook user.", notebookUserName)
	}
	return name
}

func GetNotebookUserToken(t Test) string {
	token, ok := os.LookupEnv(notebookUserToken)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify token of the authenticated Notebook user.", notebookUserToken)
	}
	return token
}

func GetNotebookImage(t Test) string {
	notebook_image, ok := os.LookupEnv(notebookImage)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify image of the Notebook.", notebookImage)
	}
	return notebook_image
}

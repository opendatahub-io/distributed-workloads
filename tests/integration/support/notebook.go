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
	"bytes"
	"embed"
	"html/template"

	. "github.com/onsi/gomega"
	cfosupport "github.com/project-codeflare/codeflare-operator/test/support"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"
)

//go:embed resources/*
var files embed.FS

var notebookResource = schema.GroupVersionResource{Group: "kubeflow.org", Version: "v1", Resource: "notebooks"}

type NotebookProps struct {
	IngressDomain             string
	OpenShiftApiUrl           string
	KubernetesBearerToken     string
	Namespace                 string
	OpenDataHubNamespace      string
	CodeFlareImageStreamTag   string
	NotebookConfigMapName     string
	NotebookConfigMapFileName string
	NotebookPVC               string
}

func CreateNotebook(test cfosupport.Test, namespace *corev1.Namespace, notebookToken, jupyterNotebookConfigMapName, jupyterNotebookConfigMapFileName string) {
	// Create PVC for Notebook
	notebookPVC := &corev1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "jupyterhub-nb-kube-3aadmin-pvc",
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{
				corev1.ReadWriteOnce,
			},
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse("10Gi"),
				},
			},
		},
	}
	notebookPVC, err := test.Client().Core().CoreV1().PersistentVolumeClaims(namespace.Name).Create(test.Ctx(), notebookPVC, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PersistentVolumeClaim %s/%s successfully", notebookPVC.Namespace, notebookPVC.Name)

	// Read the Notebook CR from resources and perform replacements for custom values using go template
	notebookProps := NotebookProps{
		IngressDomain:             GetIngressDomain(test),
		OpenShiftApiUrl:           GetOpenShiftApiUrl(test),
		KubernetesBearerToken:     notebookToken,
		Namespace:                 namespace.Name,
		OpenDataHubNamespace:      GetOpenDataHubNamespace(),
		CodeFlareImageStreamTag:   GetODHCodeFlareImageStreamTag(test),
		NotebookConfigMapName:     jupyterNotebookConfigMapName,
		NotebookConfigMapFileName: jupyterNotebookConfigMapFileName,
		NotebookPVC:               notebookPVC.Name,
	}
	notebookTemplate, err := files.ReadFile("resources/custom-nb-small.yaml")
	test.Expect(err).NotTo(HaveOccurred())
	parsedNotebookTemplate, err := template.New("notebook").Parse(string(notebookTemplate))
	test.Expect(err).NotTo(HaveOccurred())

	// Filter template and store results to the buffer
	notebookBuffer := new(bytes.Buffer)
	err = parsedNotebookTemplate.Execute(notebookBuffer, notebookProps)
	test.Expect(err).NotTo(HaveOccurred())

	// Create Notebook CR
	notebookCR := &unstructured.Unstructured{}
	err = yaml.NewYAMLOrJSONDecoder(notebookBuffer, 8192).Decode(notebookCR)
	test.Expect(err).NotTo(HaveOccurred())
	_, err = test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Create(test.Ctx(), notebookCR, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
}

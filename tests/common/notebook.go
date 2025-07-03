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

package common

import (
	"bytes"
	"embed"
	"strings"

	gomega "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	NOTEBOOK_POD_NAME       = "jupyter-nb-kube-3aadmin-0"
	NOTEBOOK_CONTAINER_NAME = "jupyter-nb-kube-3aadmin"
)

//go:embed resources/*
var files embed.FS

func readFile(t Test, fileName string) []byte {
	t.T().Helper()
	file, err := files.ReadFile(fileName)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return file
}

var (
	SmallContainerResources = ContainerResources{
		Limits:   ResourceConfig{CPU: "2", Memory: "3Gi"},
		Requests: ResourceConfig{CPU: "1", Memory: "3Gi"},
	}
	MediumContainerResources = ContainerResources{
		Limits:   ResourceConfig{CPU: "6", Memory: "24Gi"},
		Requests: ResourceConfig{CPU: "3", Memory: "24Gi"},
	}
)

type ResourceConfig struct {
	CPU              string
	Memory           string
	GPUResourceLabel string // e.g., "nvidia.com/gpu", "amd.com/gpu", or ""
}

type ContainerResources struct {
	Limits   ResourceConfig
	Requests ResourceConfig
}

type ContainerSize string

const (
	ContainerSizeSmall  ContainerSize = "small"
	ContainerSizeMedium ContainerSize = "medium"
)

var notebookResource = schema.GroupVersionResource{Group: "kubeflow.org", Version: "v1", Resource: "notebooks"}

type NotebookProps struct {
	IngressDomain             string
	OpenShiftApiUrl           string
	KubernetesUserBearerToken string
	Namespace                 string
	OpenDataHubNamespace      string
	Command                   string
	NotebookImage             string
	NotebookConfigMapName     string
	NotebookConfigMapFileName string
	NotebookPVC               string
	NumGpus                   int
	PipIndexUrl               string
	PipTrustedHost            string
	S3BucketName              string
	S3AccessKeyId             string
	S3SecretAccessKey         string
	S3Endpoint                string
	S3DefaultRegion           string
	NotebookResources         ContainerResources
	SizeSelection             ContainerSize
}

func CreateNotebook(test Test, namespace *corev1.Namespace, notebookUserToken string, command []string, jupyterNotebookConfigMapName, jupyterNotebookConfigMapFileName string, numGpus int, notebookPVC *corev1.PersistentVolumeClaim, containerSize ContainerSize, acceleratorResourceLabel ...string) {
	s3BucketName, s3BucketNameExists := GetStorageBucketName()
	s3AccessKeyId, _ := GetStorageBucketAccessKeyId()
	s3SecretAccessKey, _ := GetStorageBucketSecretKey()
	s3Endpoint, _ := GetStorageBucketDefaultEndpoint()
	s3DefaultRegion, _ := GetStorageBucketDefaultRegion()
	strCommand := "[\"" + strings.Join(command, "\",\"") + "\"]"

	if !s3BucketNameExists {
		s3BucketName = "''"
		s3AccessKeyId = "''"
		s3SecretAccessKey = "''"
		s3Endpoint = "''"
		s3DefaultRegion = "''"
	}

	var selectedContainerResources ContainerResources
	var gpuResourceLabel string
	if len(acceleratorResourceLabel) == 1 {
		gpuResourceLabel = acceleratorResourceLabel[0]
	} else {
		gpuResourceLabel = ""
	}

	if containerSize == ContainerSizeSmall {
		selectedContainerResources = SmallContainerResources
		// For small, ensure no GPU resource is requested
		selectedContainerResources.Limits.GPUResourceLabel = ""
		selectedContainerResources.Requests.GPUResourceLabel = ""
	} else if containerSize == ContainerSizeMedium {
		selectedContainerResources = MediumContainerResources

		if gpuResourceLabel != "" && gpuResourceLabel != NVIDIA.ResourceLabel && gpuResourceLabel != AMD.ResourceLabel {
			test.T().Errorf("Unsupported GPU resource label for medium size: %s. Must be '%s', '%s', or an empty string.", gpuResourceLabel, NVIDIA.ResourceLabel, AMD.ResourceLabel)
			gpuResourceLabel = "" // Fallback to no GPU if label is invalid
		}

		// Apply the determined GPUResourceLabel
		selectedContainerResources.Limits.GPUResourceLabel = gpuResourceLabel
		selectedContainerResources.Requests.GPUResourceLabel = gpuResourceLabel
	} else {
		test.T().Errorf("Unsupported container size: %s. Must be '%s' or '%s'. Hence using '%s' container size.",
			containerSize, ContainerSizeSmall, ContainerSizeMedium, ContainerSizeSmall)
		selectedContainerResources = SmallContainerResources // Fallback to Small container size
	}

	// Read the Notebook CR from resources and perform replacements for custom values using go template
	notebookProps := NotebookProps{
		IngressDomain:             GetOpenShiftIngressDomain(test),
		OpenShiftApiUrl:           GetOpenShiftApiUrl(test),
		KubernetesUserBearerToken: notebookUserToken,
		Namespace:                 namespace.Name,
		OpenDataHubNamespace:      GetOpenDataHubNamespace(test),
		Command:                   strCommand,
		NotebookImage:             GetNotebookImage(test),
		NotebookConfigMapName:     jupyterNotebookConfigMapName,
		NotebookConfigMapFileName: jupyterNotebookConfigMapFileName,
		NotebookPVC:               notebookPVC.Name,
		NumGpus:                   numGpus,
		S3BucketName:              s3BucketName,
		S3AccessKeyId:             s3AccessKeyId,
		S3SecretAccessKey:         s3SecretAccessKey,
		S3Endpoint:                s3Endpoint,
		S3DefaultRegion:           s3DefaultRegion,
		PipIndexUrl:               GetPipIndexURL(),
		PipTrustedHost:            GetPipTrustedHost(),
		NotebookResources:         selectedContainerResources,
		SizeSelection:             containerSize,
	}
	notebookTemplate, err := files.ReadFile("resources/custom-nb-small.yaml")
	test.Expect(err).NotTo(gomega.HaveOccurred())

	notebookTemplate = ParseTemplate(test, notebookTemplate, notebookProps)
	parsedNotebookTemplate := ParseTemplate(test, notebookTemplate, notebookProps)

	// Create Notebook CR
	notebookCR := &unstructured.Unstructured{}
	err = yaml.NewYAMLOrJSONDecoder(bytes.NewBuffer(parsedNotebookTemplate), 8192).Decode(notebookCR)
	test.Expect(err).NotTo(gomega.HaveOccurred())
	_, err = test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Create(test.Ctx(), notebookCR, metav1.CreateOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
}

func DeleteNotebook(test Test, namespace *corev1.Namespace) {
	err := test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Delete(test.Ctx(), "jupyter-nb-kube-3aadmin", metav1.DeleteOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
}

func ListNotebooks(test Test, namespace *corev1.Namespace) []*unstructured.Unstructured {
	ntbs, err := test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())

	ntbsp := []*unstructured.Unstructured{}
	for _, v := range ntbs.Items {
		ntbsp = append(ntbsp, &v)
	}

	return ntbsp
}

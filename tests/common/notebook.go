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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"
	kueuev1beta2 "sigs.k8s.io/kueue/apis/kueue/v1beta2"

	support "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	NOTEBOOK_POD_NAME       = "jupyter-nb-kube-3aadmin-0"
	NOTEBOOK_CONTAINER_NAME = "jupyter-nb-kube-3aadmin"
)

//go:embed resources/*
var files embed.FS

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
	NotebookResources         support.ContainerResources
	SizeSelection             support.ContainerSize
}

func CreateNotebook(test support.Test, namespace *corev1.Namespace, notebookUserToken string, command []string, jupyterNotebookConfigMapName, jupyterNotebookConfigMapFileName string, numGpus int, notebookPVC *corev1.PersistentVolumeClaim, containerSize support.ContainerSize, notebookImage string, acceleratorResourceLabel ...string) {
	s3BucketName, s3BucketNameExists := support.GetStorageBucketName()
	s3AccessKeyId, _ := support.GetStorageBucketAccessKeyId()
	s3SecretAccessKey, _ := support.GetStorageBucketSecretKey()
	s3Endpoint, _ := support.GetStorageBucketDefaultEndpoint()
	s3DefaultRegion, _ := support.GetStorageBucketDefaultRegion()
	strCommand := "[\"" + strings.Join(command, "\",\"") + "\"]"

	if !s3BucketNameExists {
		s3BucketName = "''"
		s3AccessKeyId = "''"
		s3SecretAccessKey = "''"
		s3Endpoint = "''"
		s3DefaultRegion = "''"
	}

	var selectedContainerResources support.ContainerResources
	var gpuResourceLabel string
	if len(acceleratorResourceLabel) == 1 {
		gpuResourceLabel = acceleratorResourceLabel[0]
	} else {
		gpuResourceLabel = ""
	}

	if containerSize == support.ContainerSizeSmall {
		selectedContainerResources = support.SmallContainerResources
		// For small, ensure no GPU resource is requested
		selectedContainerResources.Limits.GPUResourceLabel = ""
		selectedContainerResources.Requests.GPUResourceLabel = ""
	} else if containerSize == support.ContainerSizeMedium {
		selectedContainerResources = support.MediumContainerResources

		if gpuResourceLabel != "" && gpuResourceLabel != support.NVIDIA.ResourceLabel && gpuResourceLabel != support.AMD.ResourceLabel {
			test.T().Errorf("Unsupported GPU resource label for medium size: %s. Must be '%s', '%s', or an empty string.", gpuResourceLabel, support.NVIDIA.ResourceLabel, support.AMD.ResourceLabel)
			gpuResourceLabel = "" // Fallback to no GPU if label is invalid
		}

		// Apply the determined GPUResourceLabel
		selectedContainerResources.Limits.GPUResourceLabel = gpuResourceLabel
		selectedContainerResources.Requests.GPUResourceLabel = gpuResourceLabel
	} else {
		test.T().Errorf("Unsupported container size: %s. Must be '%s' or '%s'. Hence using '%s' container size.",
			containerSize, support.ContainerSizeSmall, support.ContainerSizeMedium, support.ContainerSizeSmall)
		selectedContainerResources = support.SmallContainerResources // Fallback to Small container size
	}

	// Get the ODH namespace from DSCI
	odhNamespace, err := support.GetApplicationsNamespaceFromDSCI(test, support.DefaultDSCIName)
	test.Expect(err).NotTo(gomega.HaveOccurred())

	// Read the Notebook CR from resources and perform replacements for custom values using go template
	notebookProps := NotebookProps{
		IngressDomain:             support.GetOpenShiftIngressDomain(test),
		OpenShiftApiUrl:           support.GetOpenShiftApiUrl(test),
		KubernetesUserBearerToken: notebookUserToken,
		Namespace:                 namespace.Name,
		OpenDataHubNamespace:      odhNamespace,
		Command:                   strCommand,
		NotebookImage:             notebookImage,
		NotebookConfigMapName:     jupyterNotebookConfigMapName,
		NotebookConfigMapFileName: jupyterNotebookConfigMapFileName,
		NotebookPVC:               notebookPVC.Name,
		NumGpus:                   numGpus,
		S3BucketName:              s3BucketName,
		S3AccessKeyId:             s3AccessKeyId,
		S3SecretAccessKey:         s3SecretAccessKey,
		S3Endpoint:                s3Endpoint,
		S3DefaultRegion:           s3DefaultRegion,
		PipIndexUrl:               support.GetPipIndexURL(),
		PipTrustedHost:            support.GetPipTrustedHost(),
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

	if namespace.Labels["kueue.openshift.io/managed"] == "true" {
		cpuQuota := resource.MustParse("3")
		memQuota := resource.MustParse("4Gi")
		if containerSize == support.ContainerSizeMedium {
			cpuQuota = resource.MustParse("7")
			memQuota = resource.MustParse("25Gi")
		}

		rf := support.CreateKueueResourceFlavor(test, kueuev1beta2.ResourceFlavorSpec{})
		test.T().Cleanup(func() {
			test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(test.Ctx(), rf.Name, metav1.DeleteOptions{})
		})

		cq := support.CreateKueueClusterQueue(test, kueuev1beta2.ClusterQueueSpec{
			NamespaceSelector: &metav1.LabelSelector{},
			ResourceGroups: []kueuev1beta2.ResourceGroup{
				{
					CoveredResources: []corev1.ResourceName{corev1.ResourceCPU, corev1.ResourceMemory},
					Flavors: []kueuev1beta2.FlavorQuotas{
						{
							Name: kueuev1beta2.ResourceFlavorReference(rf.Name),
							Resources: []kueuev1beta2.ResourceQuota{
								{Name: corev1.ResourceCPU, NominalQuota: cpuQuota},
								{Name: corev1.ResourceMemory, NominalQuota: memQuota},
							},
						},
					},
				},
			},
		})
		test.T().Cleanup(func() {
			test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(test.Ctx(), cq.Name, metav1.DeleteOptions{})
		})

		lq := support.CreateKueueLocalQueue(test, namespace.Name, cq.Name)

		labels := notebookCR.GetLabels()
		if labels == nil {
			labels = map[string]string{}
		}
		labels["kueue.x-k8s.io/queue-name"] = lq.Name
		notebookCR.SetLabels(labels)
		test.T().Logf("Created Kueue resources for Notebook: LocalQueue %s -> ClusterQueue %s", lq.Name, cq.Name)
	}

	_, err = test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Create(test.Ctx(), notebookCR, metav1.CreateOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
}

func DeleteNotebook(test support.Test, namespace *corev1.Namespace) {
	err := test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).Delete(test.Ctx(), "jupyter-nb-kube-3aadmin", metav1.DeleteOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
}

func Notebooks(test support.Test, namespace *corev1.Namespace) func(g gomega.Gomega) []*unstructured.Unstructured {
	return func(g gomega.Gomega) []*unstructured.Unstructured {
		ntbs, err := test.Client().Dynamic().Resource(notebookResource).Namespace(namespace.Name).List(test.Ctx(), metav1.ListOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())

		ntbsp := []*unstructured.Unstructured{}
		for _, v := range ntbs.Items {
			ntbsp = append(ntbsp, &v)
		}

		return ntbsp
	}
}

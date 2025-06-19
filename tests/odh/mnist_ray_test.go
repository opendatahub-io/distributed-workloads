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
	"fmt"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestMnistRayCpu(t *testing.T) {
	mnistRay(t, 0, "nvidia.com/gpu", GetRayImage(), "resources/requirements.txt")
}

func TestMnistRayCudaGpu(t *testing.T) {
	mnistRay(t, 1, "nvidia.com/gpu", GetRayImage(), "resources/requirements.txt")
}

func TestMnistRayROCmGpu(t *testing.T) {
	mnistRay(t, 1, "amd.com/gpu", GetRayROCmImage(), "resources/requirements-rocm.txt")
}

func TestMnistCustomRayCudaCpu(t *testing.T) {
	mnistRay(t, 0, "nvidia.com/gpu", GetRayTorchCudaImage(), "resources/requirements.txt")
}

func TestMnistCustomRayCudaGpu(t *testing.T) {
	mnistRay(t, 1, "nvidia.com/gpu", GetRayTorchCudaImage(), "resources/requirements.txt")
}

func TestMnistCustomRayRocmCpu(t *testing.T) {
	mnistRay(t, 0, "amd.com/gpu", GetRayTorchROCmImage(), "resources/requirements-rocm.txt")
}

func TestMnistCustomRayRocmGpu(t *testing.T) {
	mnistRay(t, 1, "amd.com/gpu", GetRayTorchROCmImage(), "resources/requirements-rocm.txt")
}

func mnistRay(t *testing.T, numGpus int, gpuResourceName string, rayImage string, requirementsFileName string) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Create Kueue resources
	resourceFlavor := CreateKueueResourceFlavor(test, v1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := v1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []v1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory"), corev1.ResourceName(gpuResourceName)},
				Flavors: []v1beta1.FlavorQuotas{
					{
						Name: v1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []v1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("8"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("12Gi"),
							},
							{
								Name:         corev1.ResourceName(gpuResourceName),
								NominalQuota: resource.MustParse(fmt.Sprint(numGpus)),
							},
						},
					},
				},
			},
		},
	}
	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	CreateKueueLocalQueue(test, namespace.Name, clusterQueue.Name, AsDefaultQueue)

	// Test configuration
	jupyterNotebookConfigMapFileName := "mnist_ray_mini.ipynb"
	mnist := ParseAWSArgs(test, readFile(test, "resources/mnist.py"))
	if numGpus > 0 {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"gpu\""), 1)
	} else {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"cpu\""), 1)
	}
	jupyterNotebook := readFile(test, "resources/mnist_ray_mini.ipynb")
	jupyterNotebook = bytes.ReplaceAll(jupyterNotebook, []byte("nvidia.com/gpu"), []byte(gpuResourceName))
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		// MNIST Ray Notebook
		jupyterNotebookConfigMapFileName: jupyterNotebook,
		"mnist.py":                       mnist,
		"requirements.txt":               readFile(test, requirementsFileName),
	})

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Create PVC for Notebook
	notebookPVC := CreatePersistentVolumeClaim(test, namespace.Name, "10Gi", AccessModes(corev1.ReadWriteOnce))

	notebookCommand := getNotebookCommand(rayImage)
	// Create Notebook CR
	CreateNotebook(test, namespace, userToken, notebookCommand, config.Name, jupyterNotebookConfigMapFileName, numGpus, notebookPVC, ContainerSizeSmall)

	// Gracefully cleanup Notebook
	defer func() {
		DeleteNotebook(test, namespace)
		test.Eventually(ListNotebooks(test, namespace), TestTimeoutMedium).Should(HaveLen(0))
	}()

	// Make sure the RayCluster is created and running
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutLong).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(RayClusterState, Equal(rayv1.Ready))),
			),
		)

	// Make sure the Workload is created and running
	test.Eventually(GetKueueWorkloads(test, namespace.Name), TestTimeoutMedium).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("Workload failed to be admitted"))),
			),
		)

	// Fetch created raycluster
	rayClusterName := "mnisttest"
	rayCluster, err := test.Client().Ray().RayV1().RayClusters(namespace.Name).Get(test.Ctx(), rayClusterName, metav1.GetOptions{})
	test.Expect(err).ToNot(HaveOccurred())

	// Initialise raycluster client to interact with raycluster to get rayjob details using REST-API
	dashboardUrl := GetDashboardUrl(test, namespace, rayCluster)
	rayClient := GetRayClusterClient(test, dashboardUrl, test.Config().BearerToken)

	// wait until rayjob exists
	test.Eventually(func() ([]RayJobDetailsResponse, error) {
		return rayClient.ListJobs()
	}, TestTimeoutMedium, 1*time.Second).Should(HaveLen(1), "Ray job not found")

	// Get test job-id
	jobID := GetTestJobId(test, rayClient)
	test.Expect(jobID).ToNot(BeEmpty())

	// Wait for the job to be succeeded or failed
	var rayJobStatus string
	test.T().Logf("Waiting for job to be Succeeded...\n")
	test.Eventually(func() (string, error) {
		resp, err := rayClient.GetJobDetails(jobID)
		if err != nil {
			return rayJobStatus, err
		}
		rayJobStatusVal := resp.Status
		if rayJobStatusVal == "SUCCEEDED" || rayJobStatusVal == "FAILED" {
			test.T().Logf("JobStatus - %s\n", rayJobStatusVal)
			rayJobStatus = rayJobStatusVal
			return rayJobStatus, nil
		}
		if rayJobStatus != rayJobStatusVal && rayJobStatusVal != "SUCCEEDED" {
			test.T().Logf("JobStatus - %s...\n", rayJobStatusVal)
			rayJobStatus = rayJobStatusVal
		}
		return rayJobStatus, nil
	}, TestTimeoutDouble, 1*time.Second).Should(Or(Equal("SUCCEEDED"), Equal("FAILED")), "Job did not complete within the expected time")

	// Store job logs in output directory
	WriteRayJobAPILogs(test, rayClient, jobID)

	// Assert ray-job status after job execution
	test.Expect(rayJobStatus).To(Equal("SUCCEEDED"), "RayJob failed !")

	// Make sure the RayCluster finishes and is deleted
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutLong).
		Should(BeEmpty())
}

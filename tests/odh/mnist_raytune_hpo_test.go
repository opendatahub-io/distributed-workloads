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

	. "github.com/onsi/gomega"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestMnistRayTuneHpoCpu(t *testing.T) {
	mnistRayTuneHpo(t, 0)
}

func TestMnistRayTuneHpoGpu(t *testing.T) {
	mnistRayTuneHpo(t, 1)
}

func mnistRayTuneHpo(t *testing.T, numGpus int) {
	test := With(t)

	// Creating a namespace
	namespace := test.NewTestNamespace()

	// Ensure Notebook ServiceAccount exists (no extra RBAC)
	ensureNotebookServiceAccount(test, namespace.Name)

	// Create Kueue resources
	resourceFlavor := CreateKueueResourceFlavor(test, v1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := v1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []v1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory"), corev1.ResourceName("nvidia.com/gpu")},
				Flavors: []v1beta1.FlavorQuotas{
					{
						Name: v1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []v1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("12"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("32Gi"),
							},
							{
								Name:         corev1.ResourceName("nvidia.com/gpu"),
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
	jupyterNotebookConfigMapFileName := "mnist_hpo_raytune.ipynb"
	mnist_hpo := ParseAWSArgs(test, readFile(test, "resources/mnist_hpo.py"))

	if numGpus > 0 {
		mnist_hpo = bytes.Replace(mnist_hpo, []byte("int(\"has to be specified\")"), []byte("1"), 1)
	} else {
		mnist_hpo = bytes.Replace(mnist_hpo, []byte("int(\"has to be specified\")"), []byte("0"), 1)
	}

	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		// MNIST Raytune HPO Notebook
		jupyterNotebookConfigMapFileName: readFile(test, "resources/mnist_hpo_raytune.ipynb"),
		"mnist_hpo.py":                   mnist_hpo,
		"hpo_raytune_requirements.txt":   readFile(test, "resources/hpo_raytune_requirements.txt"),
	})

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Get ray image
	rayImage := GetRayImage()
	notebookCommand := getNotebookCommand(rayImage)

	// Create PVC for Notebook
	notebookPVC := CreatePersistentVolumeClaim(test, namespace.Name, "10Gi", AccessModes(corev1.ReadWriteOnce))

	// Create Notebook CR
	CreateNotebook(test, namespace, userToken, notebookCommand, config.Name, jupyterNotebookConfigMapFileName, numGpus, notebookPVC, ContainerSizeSmall)

	// Gracefully cleanup Notebook
	defer func() {
		DeleteNotebook(test, namespace)
		test.Eventually(Notebooks(test, namespace), TestTimeoutMedium).Should(HaveLen(0))
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

	// Try to monitor the Ray job via external dashboard (best-effort)
	// This provides job status logs and API logs when it works
	rayClusterName := "mnisthpotest"
	jobStatus, monitored := TryMonitorRayJob(test, namespace, rayClusterName)
	if monitored {
		test.T().Logf("Successfully monitored Ray job via external dashboard, status: %s", jobStatus)
		test.Expect(jobStatus).To(Equal("SUCCEEDED"), "RayJob failed!")
	} else {
		test.T().Logf("Could not monitor Ray job via external dashboard, falling back to RayCluster deletion check")
	}

	// Wait for the RayCluster to be deleted (primary success indicator from notebook)
	test.T().Logf("Waiting for notebook to complete and delete the RayCluster...")
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutDouble).
		Should(BeEmpty(), "RayCluster was not deleted - notebook may have failed")
}

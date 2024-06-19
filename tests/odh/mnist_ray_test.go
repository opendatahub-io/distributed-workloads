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
	"testing"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	"sigs.k8s.io/kueue/apis/kueue/v1beta1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestMnistRay(t *testing.T) {
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
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory"), corev1.ResourceName("nvidia.com/gpu")},
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
								Name:         corev1.ResourceName("nvidia.com/gpu"),
								NominalQuota: resource.MustParse("0"),
							},
						},
					},
				},
			},
		},
	}
	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, namespace.Name, clusterQueue.Name)

	// Test configuration
	jupyterNotebookConfigMapFileName := "mnist_ray_mini.ipynb"
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		// MNIST Ray Notebook
		jupyterNotebookConfigMapFileName: ReadFile(test, "resources/mnist_ray_mini.ipynb"),
		"mnist.py":                       readMnistPy(test),
		"requirements.txt":               readRequirementsTxt(test),
	})

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Create Notebook CR
	createNotebook(test, namespace, userToken, localQueue.Name, config.Name, jupyterNotebookConfigMapFileName)

	// Gracefully cleanup Notebook
	defer func() {
		deleteNotebook(test, namespace)
		test.Eventually(listNotebooks(test, namespace), TestTimeoutMedium).Should(HaveLen(0))
	}()

	// Make sure the RayCluster is created and running
	test.Eventually(rayClusters(test, namespace), TestTimeoutLong).
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

	// Make sure the RayCluster finishes and is deleted
	test.Eventually(rayClusters(test, namespace), TestTimeoutLong).
		Should(HaveLen(0))
}

func readRequirementsTxt(test Test) []byte {
	// Read the requirements.txt from resources and perform replacements for custom values using go template
	props := struct {
		PipIndexUrl    string
		PipTrustedHost string
	}{
		PipIndexUrl: "--index " + GetPipIndexURL(),
	}

	// Provide trusted host only if defined
	if len(GetPipTrustedHost()) > 0 {
		props.PipTrustedHost = "--trusted-host " + GetPipTrustedHost()
	}

	template, err := files.ReadFile("resources/requirements.txt")
	test.Expect(err).NotTo(HaveOccurred())

	return ParseTemplate(test, template, props)
}

func readMnistPy(test Test) []byte {
	// Read the mnist.py from resources and perform replacements for custom values using go template
	props := struct {
		MnistDatasetURL string
	}{
		MnistDatasetURL: GetMnistDatasetURL(),
	}
	template, err := files.ReadFile("resources/mnist.py")
	test.Expect(err).NotTo(HaveOccurred())

	return ParseTemplate(test, template, props)
}

// TODO: This belongs on codeflare-common/support/ray.go
func rayClusters(t Test, namespace *corev1.Namespace) func(g Gomega) []*rayv1.RayCluster {
	return func(g Gomega) []*rayv1.RayCluster {
		rcs, err := t.Client().Ray().RayV1().RayClusters(namespace.Name).List(t.Ctx(), metav1.ListOptions{})
		g.Expect(err).NotTo(HaveOccurred())

		rcsp := []*rayv1.RayCluster{}
		for _, v := range rcs.Items {
			rcsp = append(rcsp, &v)
		}

		return rcsp
	}
}

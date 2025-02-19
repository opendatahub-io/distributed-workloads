/*
Copyright 2024 The Kubernetes Authors.

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
	"encoding/json"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	awv1beta2 "github.com/project-codeflare/appwrapper/api/v1beta2"
	. "github.com/project-codeflare/codeflare-common/support"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	vapv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"
	testingraycluster "sigs.k8s.io/kueue/pkg/util/testingjobs/raycluster"
)

// Note: This test must run on an OCP v4.17 or later cluster.
// The Validating Admission Policy feature gate is GA and enabled by default from OCP v4.17 (k8s v1.30)

var (
	ns           *corev1.Namespace
	nsNoLabel    *corev1.Namespace
	rf           *kueuev1beta1.ResourceFlavor
	cq           *kueuev1beta1.ClusterQueue
	lq           *kueuev1beta1.LocalQueue
	rc           *rayv1.RayCluster
	aw           *awv1beta2.AppWrapper
	vapb         *vapv1.ValidatingAdmissionPolicyBinding
	vapbCopy     *vapv1.ValidatingAdmissionPolicyBinding
	awWithLQName = "aw-with-lq"
	awNoLQName   = "aw-no-lq"
	rcWithLQName = "rc-with-lq"
	rcNoLQName   = "rc-no-lq"
)

const (
	withLQ = true
	noLQ   = false
)

func TestValidatingAdmissionPolicy(t *testing.T) {
	test := With(t)

	// Register RayCluster types with the scheme
	err := rayv1.AddToScheme(scheme.Scheme)
	test.Expect(err).ToNot(HaveOccurred())

	// Register AppWrapper types with the scheme
	err = awv1beta2.AddToScheme(scheme.Scheme)
	test.Expect(err).ToNot(HaveOccurred())

	// Create a namespace
	ns = CreateTestNamespaceWithName(test, uniqueSuffix("vap"))
	defer test.Client().Core().CoreV1().Namespaces().Delete(test.Ctx(), ns.Name, metav1.DeleteOptions{})

	// Create a namespace that will not receive the `kueue.x-k8s.io/queue-name` label
	nsNoLabel = CreateTestNamespaceWithName(test, uniqueSuffix("vap-nl"))
	defer test.Client().Core().CoreV1().Namespaces().Delete(test.Ctx(), nsNoLabel.Name, metav1.DeleteOptions{})

	// Create a resource flavor
	rf = CreateKueueResourceFlavor(test, kueuev1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), rf.Name, metav1.DeleteOptions{})

	// Create a cluster queue
	cqSpec := kueuev1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []kueuev1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory"), corev1.ResourceName("nvidia.com/gpu")},
				Flavors: []kueuev1beta1.FlavorQuotas{
					{
						Name: kueuev1beta1.ResourceFlavorReference(rf.Name),
						Resources: []kueuev1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("3"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("8Gi"),
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
	// Create a cluster queue
	cq = CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), cq.Name, metav1.DeleteOptions{})

	// Create a local queue
	lq = CreateKueueLocalQueue(test, ns.Name, cq.Name)
	defer test.Client().Kueue().KueueV1beta1().LocalQueues(ns.Name).Delete(test.Ctx(), lq.Name, metav1.DeleteOptions{})

	// Snapshot the original ValidatingAdmissionPolicyBinding state
	vapb, err = test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Get(test.Ctx(), "kueue-validating-admission-policy-binding", metav1.GetOptions{})
	test.Expect(err).ToNot(HaveOccurred())

	vapbCopy = vapb.DeepCopy()
	defer revertVAPB(test, vapbCopy)

	/**************************************************************************
	Testing the default behavior with the ValidatingAdmissionPolicyBinding enforcement enabled.
	**************************************************************************/
	t.Run("Default ValidatingAdmissionPolicyBinding", func(t *testing.T) {
		t.Run("RayCluster Tests", func(t *testing.T) {
			t.Run("RayCluster should be admitted with the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createRayCluster(test, ns.Name, withLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Ray().RayV1().RayClusters(ns.Name).Delete(test.Ctx(), rc.Name, metav1.DeleteOptions{})
			})
			t.Run("RayCluster should not be admitted without the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createRayCluster(test, ns.Name, noLQ)
				test.Expect(err).ToNot(BeNil())
				defer test.Client().Ray().RayV1().RayClusters(ns.Name).Delete(test.Ctx(), rc.Name, metav1.DeleteOptions{})
			})
		})
		t.Run("AppWrapper Tests", func(t *testing.T) {
			t.Run("AppWrapper should be admitted with the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createAppWrapper(test, ns.Name, withLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(ns.Name).Delete(test.Ctx(), aw.Name, metav1.DeleteOptions{})
			})
			t.Run("AppWrapper should be admitted without the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createAppWrapper(test, ns.Name, noLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(ns.Name).Delete(test.Ctx(), aw.Name, metav1.DeleteOptions{})
			})
		})
	})

	/**************************************************************************
	Testing the 1st alternative behavior with the ValidatingAdmissionPolicyBinding enforcement disabled.
	**************************************************************************/
	t.Run("Disable the ValidatingAdmissionPolicy enforcement", func(t *testing.T) {
		vapb, err := test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Get(test.Ctx(), vapb.Name, metav1.GetOptions{})
		test.Expect(err).ToNot(HaveOccurred())

		vapb.Spec.PolicyName = "none"
		_, err = test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Update(test.Ctx(), vapb, metav1.UpdateOptions{})
		test.Expect(err).ToNot(HaveOccurred())
		defer revertVAPB(test, vapbCopy)
		t.Run("RayCluster Tests", func(t *testing.T) {
			t.Run("RayCluster should be admitted without the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				// Eventually is used here to allow time for the ValidatingAdmissionPolicyBinding updates to be propagated.
				test.Eventually(func() error {
					err = createRayCluster(test, ns.Name, noLQ)
					return err
				}).WithTimeout(10 * time.Second).WithPolling(500 * time.Millisecond).Should(Succeed())
				defer test.Client().Ray().RayV1().RayClusters(ns.Name).Delete(test.Ctx(), rc.Name, metav1.DeleteOptions{})
			})
			t.Run("RayCluster should be admitted with the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createRayCluster(test, ns.Name, withLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Ray().RayV1().RayClusters(ns.Name).Delete(test.Ctx(), rc.Name, metav1.DeleteOptions{})
			})
		})
		t.Run("AppWrapper Tests", func(t *testing.T) {
			t.Run("AppWrapper should be admitted with the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createAppWrapper(test, ns.Name, withLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(ns.Name).Delete(test.Ctx(), aw.Name, metav1.DeleteOptions{})
			})
			t.Run("AppWrapper should be admitted without the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createAppWrapper(test, ns.Name, noLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(ns.Name).Delete(test.Ctx(), aw.Name, metav1.DeleteOptions{})
			})
		})
	})

	/**************************************************************************
	Testing the 2nd alternative behavior which targets specific namespaces that have the 'kueue-managed' label
	**************************************************************************/
	t.Run("Custom ValidatingAdmissionPolicyBinding", func(t *testing.T) {
		// Update the test namespace with the new 'kueue-managed' label
		ns, err = test.Client().Core().CoreV1().Namespaces().Get(test.Ctx(), ns.Name, metav1.GetOptions{})
		if ns.Labels == nil {
			ns.Labels = map[string]string{}
		}
		ns.Labels["kueue-managed"] = "true"
		_, err = test.Client().Core().CoreV1().Namespaces().Update(test.Ctx(), ns, metav1.UpdateOptions{})
		test.Eventually(func() bool {
			ns, _ = test.Client().Core().CoreV1().Namespaces().Get(test.Ctx(), ns.Name, metav1.GetOptions{})
			return ns.Labels["kueue-managed"] == "true"
		}).WithTimeout(10 * time.Second).WithPolling(500 * time.Millisecond).Should(BeTrue())
		test.Expect(err).ToNot(HaveOccurred())

		// Apply the ValidatingAdmissionPolicyBinding targetting namespaces with the label 'kueue-managed'
		vapb, err = test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Get(test.Ctx(), vapb.Name, metav1.GetOptions{})
		test.Expect(err).ToNot(HaveOccurred())

		vapb.Spec.MatchResources.NamespaceSelector.MatchLabels = map[string]string{"kueue-managed": "true"}
		_, err = test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Update(test.Ctx(), vapb, metav1.UpdateOptions{})
		test.Expect(err).ToNot(HaveOccurred())
		defer revertVAPB(test, vapbCopy)
		t.Run("RayCluster Tests", func(t *testing.T) {
			t.Run("RayCluster should not be admitted without the 'kueue.x-k8s.io/queue-name' label in a labeled namespace", func(t *testing.T) {
				test.Eventually(func() error {
					err = createRayCluster(test, ns.Name, noLQ)
					return err
				}).WithTimeout(10 * time.Second).WithPolling(500 * time.Millisecond).ShouldNot(Succeed())
				defer test.Client().Ray().RayV1().RayClusters(ns.Name).Delete(test.Ctx(), rc.Name, metav1.DeleteOptions{})
			})
			t.Run("RayCluster should be admitted with the 'kueue.x-k8s.io/queue-name' label in a labeled namespace", func(t *testing.T) {
				err = createRayCluster(test, ns.Name, withLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Ray().RayV1().RayClusters(ns.Name).Delete(test.Ctx(), rc.Name, metav1.DeleteOptions{})
			})
			t.Run("RayCluster should be admitted with the 'kueue.x-k8s.io/queue-name' label in any other namespace", func(t *testing.T) {
				err = createRayCluster(test, nsNoLabel.Name, withLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Ray().RayV1().RayClusters(nsNoLabel.Name).Delete(test.Ctx(), rc.Name, metav1.DeleteOptions{})
			})
			t.Run("RayCluster should be admitted without the 'kueue.x-k8s.io/queue-name' label in any other namespace", func(t *testing.T) {
				test.Eventually(func() error {
					err = createRayCluster(test, nsNoLabel.Name, noLQ)
					return err
				}).WithTimeout(10 * time.Second).WithPolling(500 * time.Millisecond).Should(Succeed())
				defer test.Client().Ray().RayV1().RayClusters(nsNoLabel.Name).Delete(test.Ctx(), rc.Name, metav1.DeleteOptions{})
			})
		})
		t.Run("AppWrapper Tests", func(t *testing.T) {
			t.Run("AppWrapper should be admitted without the 'kueue.x-k8s.io/queue-name' label in a labeled namespace", func(t *testing.T) {
				err = createAppWrapper(test, ns.Name, noLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(ns.Name).Delete(test.Ctx(), aw.Name, metav1.DeleteOptions{})
			})
			t.Run("AppWrapper should be admitted with the 'kueue.x-k8s.io/queue-name' label in a labeled namespace", func(t *testing.T) {
				err = createAppWrapper(test, ns.Name, withLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(ns.Name).Delete(test.Ctx(), aw.Name, metav1.DeleteOptions{})
			})
			t.Run("AppWrapper should be admitted with the 'kueue.x-k8s.io/queue-name' label in any other namespace", func(t *testing.T) {
				err = createAppWrapper(test, nsNoLabel.Name, withLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(nsNoLabel.Name).Delete(test.Ctx(), aw.Name, metav1.DeleteOptions{})
			})
			t.Run("AppWrapper should be admitted without the 'kueue.x-k8s.io/queue-name' label in any other namespace", func(t *testing.T) {
				err = createAppWrapper(test, nsNoLabel.Name, noLQ)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(nsNoLabel.Name).Delete(test.Ctx(), aw.Name, metav1.DeleteOptions{})
			})
		})
	})
}

// Revert validating-admission-policy-binding to its original state
func revertVAPB(test Test, vapbCopy *vapv1.ValidatingAdmissionPolicyBinding) {
	patchBytes, _ := json.Marshal(map[string]interface{}{
		"spec": map[string]interface{}{
			"policyName": vapbCopy.Spec.PolicyName,
			"matchResources": map[string]interface{}{
				"namespaceSelector": map[string]interface{}{
					"matchLabels": vapbCopy.Spec.MatchResources.NamespaceSelector.MatchLabels,
				},
			},
		},
	})
	_, err := test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Patch(test.Ctx(), vapbCopy.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	test.Expect(err).ToNot(HaveOccurred())
}

func createRayCluster(test Test, namespaceName string, localQueue bool) error {
	if localQueue {
		rc = testingraycluster.MakeCluster(uniqueSuffix(rcWithLQName), namespaceName).Queue(lq.Name).Obj()
	} else {
		rc = testingraycluster.MakeCluster(uniqueSuffix(rcNoLQName), namespaceName).Obj()
	}
	_, err := test.Client().Ray().RayV1().RayClusters(namespaceName).Create(test.Ctx(), rc, metav1.CreateOptions{})
	return err
}

func createAppWrapper(test Test, namespaceName string, localQueue bool) error {
	if localQueue {
		aw = newAppWrapperWithRayCluster(uniqueSuffix(awWithLQName), uniqueSuffix(rcNoLQName), namespaceName)
		if aw.Labels == nil {
			aw.Labels = make(map[string]string)
		}
		aw.Labels["kueue.x-k8s.io/queue-name"] = lq.Name
	} else {
		// Make an AppWrapper without the 'kueue.x-k8s.io/queue-name' label set
		aw = newAppWrapperWithRayCluster(uniqueSuffix(awNoLQName), uniqueSuffix(rcNoLQName), namespaceName)
	}
	awMap, _ := runtime.DefaultUnstructuredConverter.ToUnstructured(aw)
	_, err := test.Client().Dynamic().Resource(awv1beta2.GroupVersion.WithResource("appwrappers")).Namespace(namespaceName).Create(test.Ctx(), &unstructured.Unstructured{Object: awMap}, metav1.CreateOptions{})
	return err
}

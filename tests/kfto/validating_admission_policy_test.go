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

package kfto

import (
	"encoding/json"
	"testing"
	"time"

	kftrainingv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"

	vapv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"
	testingpytorchjob "sigs.k8s.io/kueue/pkg/util/testingjobs/pytorchjob"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

// Note: This test must run on an OCP v4.17 or later cluster.
// The Validating Admission Policy feature gate is GA and enabled by default from OCP v4.17 (k8s v1.30)

var (
	err           error
	ns            *corev1.Namespace
	nsNoLabel     *corev1.Namespace
	rf            *kueuev1beta1.ResourceFlavor
	cq            *kueuev1beta1.ClusterQueue
	lq            *kueuev1beta1.LocalQueue
	pyt           *kftrainingv1.PyTorchJob
	vapb          *vapv1.ValidatingAdmissionPolicyBinding
	vapbCopy      *vapv1.ValidatingAdmissionPolicyBinding
	pytWithLQName = "pyt-with-lq"
	pytNoLQName   = "pyt-no-lq"
)

func TestValidatingAdmissionPolicy(t *testing.T) {
	test := With(t)

	Tags(t, Sanity)

	// Create namespace with unique name and required labels
	var AsDefaultQueueNamespace = ErrorOption[*corev1.Namespace](func(ns *corev1.Namespace) error {
		if ns.Labels == nil {
			ns.Labels = make(map[string]string)
		}
		ns.Labels["kueue.openshift.io/managed"] = "true"
		return nil
	})
	ns = CreateTestNamespaceWithName(
		test,
		uniqueSuffix("vap"),
		AsDefaultQueueNamespace,
	)
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
		t.Run("PyTorchJob Tests", func(t *testing.T) {
			t.Run("PyTorchJob should be admitted with the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createPyTorchJobWithLocalQueue(test, ns.Name, lq.Name)
				test.Expect(err).ToNot(HaveOccurred())
				defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(ns.Name).Delete(test.Ctx(), pyt.Name, metav1.DeleteOptions{})
			})
			t.Run("PyTorchJob should not be admitted without the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				err = createPyTorchJob(test, ns.Name)
				test.Expect(err).ToNot(BeNil())
				test.Expect(err.Error()).To(ContainSubstring("The label 'kueue.x-k8s.io/queue-name' is either missing or does not have a value set"))
				defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(ns.Name).Delete(test.Ctx(), pyt.Name, metav1.DeleteOptions{})
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

		// Add verification that the VAP is actually disabled
		test.Eventually(func() (string, error) {
			updatedVapb, err := test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Get(test.Ctx(), vapb.Name, metav1.GetOptions{})
			if err != nil {
				return "", err
			}
			return updatedVapb.Spec.PolicyName, nil
		}).WithTimeout(10*time.Second).WithPolling(500*time.Millisecond).Should(Equal("none"), "VAP should be disabled")

		defer revertVAPB(test, vapbCopy)

		t.Run("PyTorchJob Tests", func(t *testing.T) {
			t.Run("PyTorchJob should be admitted with the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				test.Eventually(func() error {
					return createPyTorchJobWithLocalQueue(test, ns.Name, lq.Name)
				}).WithTimeout(10*time.Second).WithPolling(500*time.Millisecond).Should(Succeed(), "PyTorchJob with queue label should be created")
				defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(ns.Name).Delete(test.Ctx(), pyt.Name, metav1.DeleteOptions{})
			})
			t.Run("PyTorchJob should be admitted without the 'kueue.x-k8s.io/queue-name' label set", func(t *testing.T) {
				test.Eventually(func() error {
					return createPyTorchJob(test, ns.Name)
				}).WithTimeout(10*time.Second).WithPolling(500*time.Millisecond).Should(Succeed(), "PyTorchJob without queue label should be created")
				defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(ns.Name).Delete(test.Ctx(), pyt.Name, metav1.DeleteOptions{})
			})
		})
	})

	/**************************************************************************
	Testing the 2nd alternative behavior which targets specific namespaces that have the 'kueue.openshift.io/managed' label
	**************************************************************************/
	t.Run("Custom ValidatingAdmissionPolicyBinding", func(t *testing.T) {
		// Apply the ValidatingAdmissionPolicyBinding targetting namespaces with the label 'kueue.openshift.io/managed'
		vapb, err = test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Get(test.Ctx(), vapb.Name, metav1.GetOptions{})
		test.Expect(err).ToNot(HaveOccurred())

		vapb.Spec.MatchResources.NamespaceSelector.MatchLabels = map[string]string{"kueue.openshift.io/managed": "true"}
		_, err = test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Update(test.Ctx(), vapb, metav1.UpdateOptions{})
		test.Expect(err).ToNot(HaveOccurred())

		// Add verification that the VAP namespace selector is updated
		test.Eventually(func() (string, error) {
			updatedVapb, err := test.Client().Core().AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Get(test.Ctx(), vapb.Name, metav1.GetOptions{})
			if err != nil {
				return "", err
			}
			return updatedVapb.Spec.MatchResources.NamespaceSelector.MatchLabels["kueue.openshift.io/managed"], nil
		}).WithTimeout(10*time.Second).WithPolling(500*time.Millisecond).Should(Equal("true"), "VAP namespace selector should be updated")

		defer revertVAPB(test, vapbCopy)

		t.Run("PyTorchJob Tests", func(t *testing.T) {
			t.Run("PyTorchJob should be admitted with the 'kueue.x-k8s.io/queue-name' label in a labeled namespace", func(t *testing.T) {
				test.Eventually(func() error {
					return createPyTorchJobWithLocalQueue(test, ns.Name, lq.Name)
				}).WithTimeout(10*time.Second).WithPolling(500*time.Millisecond).Should(Succeed(), "PyTorchJob with queue label should be created in labeled namespace")
				defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(ns.Name).Delete(test.Ctx(), pyt.Name, metav1.DeleteOptions{})
			})
			t.Run("PyTorchJob should not be admitted without the 'kueue.x-k8s.io/queue-name' label in a labeled namespace", func(t *testing.T) {
				test.Eventually(func() error {
					return createPyTorchJob(test, ns.Name)
				}).WithTimeout(10 * time.Second).WithPolling(500 * time.Millisecond).Should(
					And(
						HaveOccurred(),
						MatchError(ContainSubstring("The label 'kueue.x-k8s.io/queue-name' is either missing or does not have a value set")),
					),
				)
				defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(ns.Name).Delete(test.Ctx(), pyt.Name, metav1.DeleteOptions{})
			})
			t.Run("PyTorchJob should be admitted with the 'kueue.x-k8s.io/queue-name' label in any other namespace", func(t *testing.T) {
				test.Eventually(func() error {
					return createPyTorchJobWithLocalQueue(test, nsNoLabel.Name, lq.Name)
				}).WithTimeout(10*time.Second).WithPolling(500*time.Millisecond).Should(Succeed(), "PyTorchJob with queue label should be created in unlabeled namespace")
				defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(ns.Name).Delete(test.Ctx(), pyt.Name, metav1.DeleteOptions{})
			})
			t.Run("PyTorchJob should be admitted without the 'kueue.x-k8s.io/queue-name' label in any other namespace", func(t *testing.T) {
				test.Eventually(func() error {
					return createPyTorchJob(test, nsNoLabel.Name)
				}).WithTimeout(10*time.Second).WithPolling(500*time.Millisecond).Should(Succeed(), "PyTorchJob without queue label should be created in unlabeled namespace")
				defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(ns.Name).Delete(test.Ctx(), pyt.Name, metav1.DeleteOptions{})
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

func createPyTorchJob(test Test, namespaceName string) error {
	pyt = testingpytorchjob.MakePyTorchJob(uniqueSuffix(pytNoLQName), namespaceName).Obj()
	pyt.Spec.PyTorchReplicaSpecs[kftrainingv1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Name = "pytorch"
	pyt.Spec.PyTorchReplicaSpecs[kftrainingv1.PyTorchJobReplicaTypeWorker].Template.Spec.Containers[0].Name = "pytorch"

	_, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespaceName).Create(test.Ctx(), pyt, metav1.CreateOptions{})
	return err
}

func createPyTorchJobWithLocalQueue(test Test, namespaceName, localQueueName string) error {
	pyt = testingpytorchjob.MakePyTorchJob(uniqueSuffix(pytWithLQName), namespaceName).Queue(localQueueName).Obj()
	pyt.Spec.PyTorchReplicaSpecs[kftrainingv1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Name = "pytorch"
	pyt.Spec.PyTorchReplicaSpecs[kftrainingv1.PyTorchJobReplicaTypeWorker].Template.Spec.Containers[0].Name = "pytorch"

	_, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespaceName).Create(test.Ctx(), pyt, metav1.CreateOptions{})
	return err
}

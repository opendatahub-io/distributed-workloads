/*
Copyright 2024.

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
	"testing"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

var (
	sleepNamespaceName  = "test-kfto-upgrade-sleep"
	sleepPyTorchJobName = "pytorch-upgrade-sleep"
)

func TestSetupSleepPytorchjob(t *testing.T) {
	Tags(t, PreUpgrade)
	test := With(t)

	SetupKueue(test, initialKueueState, PyTorchJobFramework)

	// Create a namespace with Kueue labeled
	CreateOrGetTestNamespaceWithName(test, sleepNamespaceName, WithKueueManaged())
	test.T().Logf("Created Kueue-managed namespace: %s", sleepNamespaceName)

	// Create Kueue resources
	resourceFlavor := CreateKueueResourceFlavor(test, v1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := v1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []v1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory")},
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
								NominalQuota: resource.MustParse("18Gi"),
							},
						},
					},
				},
			},
		},
	}

	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, sleepNamespaceName, clusterQueue.Name, AsDefaultQueue)

	// Create training PyTorch job
	createSleepPyTorchJob(test, sleepNamespaceName, localQueue)

	// Make sure the PyTorch job is running, waiting for Training operator upgrade
	test.Eventually(PyTorchJob(test, sleepNamespaceName, sleepPyTorchJobName), TestTimeoutShort).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))
}

func TestVerifySleepPytorchjob(t *testing.T) {
	Tags(t, PostUpgrade)
	test := With(t)
	namespace := GetNamespaceWithName(test, sleepNamespaceName)

	// Cleanup namespace in the end
	defer DeleteTestNamespace(test, namespace)

	// PyTorch job should be still running
	test.Expect(PyTorchJob(test, sleepNamespaceName, sleepPyTorchJobName)(test)).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Pod job should be running without restart
	test.Expect(GetPods(test, sleepNamespaceName, metav1.ListOptions{})).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(sleepPodRestartCount, BeNumerically("==", 0))),
			),
		)
}

func createSleepPyTorchJob(test Test, namespace string, localQueue *v1beta1.LocalQueue) *kftov1.PyTorchJob {
	// Does PyTorchJob already exist?
	_, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Get(test.Ctx(), sleepPyTorchJobName, metav1.GetOptions{})
	if err == nil {
		// If yes then delete it and wait until there are no PyTorchJobs in the namespace
		err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Delete(test.Ctx(), sleepPyTorchJobName, metav1.DeleteOptions{})
		test.Expect(err).NotTo(HaveOccurred())
		test.Eventually(PyTorchJobs(test, namespace), TestTimeoutShort).Should(BeEmpty())
	} else if !errors.IsNotFound(err) {
		test.T().Fatalf("Error retrieving PyTorchJob with name `%s`: %v", sleepPyTorchJobName, err)
	}

	tuningJob := &kftov1.PyTorchJob{
		ObjectMeta: metav1.ObjectMeta{
			Name: sleepPyTorchJobName,
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": localQueue.Name,
			},
		},
		Spec: kftov1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kftov1.ReplicaType]*kftov1.ReplicaSpec{
				"Master": {
					Replicas:      Ptr(int32(1)),
					RestartPolicy: "OnFailure",
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           GetSleepImage(),
									ImagePullPolicy: corev1.PullIfNotPresent,
									Args:            []string{"24h"},
								},
							},
						},
					},
				},
			},
		},
	}

	tuningJob, err = test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

func sleepPodRestartCount(pod corev1.Pod) int {
	return int(pod.Status.ContainerStatuses[0].RestartCount)
}

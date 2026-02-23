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

package support

import (
	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"
)

const (
	KueueDefaultQueueName = "default"
)

func CreateKueueResourceFlavor(t Test, resourceFlavorSpec kueuev1beta1.ResourceFlavorSpec) *kueuev1beta1.ResourceFlavor {
	t.T().Helper()

	resourceFlavor := &kueuev1beta1.ResourceFlavor{
		TypeMeta: metav1.TypeMeta{
			APIVersion: kueuev1beta1.SchemeGroupVersion.String(),
			Kind:       "ResourceFlavor",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "rf-",
		},
		Spec: resourceFlavorSpec,
	}

	resourceFlavor, err := t.Client().Kueue().KueueV1beta1().ResourceFlavors().Create(t.Ctx(), resourceFlavor, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created Kueue ResourceFlavor %s successfully", resourceFlavor.Name)

	return resourceFlavor
}

func CreateKueueClusterQueue(t Test, clusterQueueSpec kueuev1beta1.ClusterQueueSpec) *kueuev1beta1.ClusterQueue {
	t.T().Helper()

	clusterQueue := &kueuev1beta1.ClusterQueue{
		TypeMeta: metav1.TypeMeta{
			APIVersion: kueuev1beta1.SchemeGroupVersion.String(),
			Kind:       "ClusterQueue",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "cq-",
		},
		Spec: clusterQueueSpec,
	}

	clusterQueue, err := t.Client().Kueue().KueueV1beta1().ClusterQueues().Create(t.Ctx(), clusterQueue, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created Kueue ClusterQueue %s successfully", clusterQueue.Name)

	return clusterQueue
}

var AsDefaultQueue = ErrorOption[*kueuev1beta1.LocalQueue](func(to *kueuev1beta1.LocalQueue) error {
	if to.Annotations == nil {
		to.Annotations = make(map[string]string)
	}
	to.Annotations["kueue.x-k8s.io/default-queue"] = "true"
	return nil
})

func CreateKueueLocalQueue(t Test, namespace string, clusterQueueName string, options ...Option[*kueuev1beta1.LocalQueue]) *kueuev1beta1.LocalQueue {
	t.T().Helper()

	localQueue := &kueuev1beta1.LocalQueue{
		TypeMeta: metav1.TypeMeta{
			APIVersion: kueuev1beta1.SchemeGroupVersion.String(),
			Kind:       "LocalQueue",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "lq-",
			Namespace:    namespace,
		},
		Spec: kueuev1beta1.LocalQueueSpec{
			ClusterQueue: kueuev1beta1.ClusterQueueReference(clusterQueueName),
		},
	}

	//Apply options
	for _, opt := range options {
		t.Expect(opt.ApplyTo(localQueue)).To(gomega.Succeed())
	}

	localQueue, err := t.Client().Kueue().KueueV1beta1().LocalQueues(localQueue.Namespace).Create(t.Ctx(), localQueue, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created Kueue LocalQueue %s/%s successfully", localQueue.Namespace, localQueue.Name)

	return localQueue
}

func KueueWorkloads(t Test, namespace string) func(g gomega.Gomega) []*kueuev1beta1.Workload {
	return func(g gomega.Gomega) []*kueuev1beta1.Workload {
		workloads, err := t.Client().Kueue().KueueV1beta1().Workloads(namespace).List(t.Ctx(), metav1.ListOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())

		workloadsp := []*kueuev1beta1.Workload{}
		for _, v := range workloads.Items {
			workloadsp = append(workloadsp, &v)
		}

		return workloadsp
	}
}

func GetKueueWorkloads(t Test, namespace string) []*kueuev1beta1.Workload {
	t.T().Helper()
	return KueueWorkloads(t, namespace)(t)
}

func KueueWorkloadAdmitted(workload *kueuev1beta1.Workload) bool {
	for _, v := range workload.Status.Conditions {
		if v.Type == "Admitted" && v.Status == "True" {
			return true
		}
	}
	return false
}

func KueueWorkloadEvicted(workload *kueuev1beta1.Workload) bool {
	for _, v := range workload.Status.Conditions {
		if v.Type == "Evicted" && v.Status == "True" {
			return true
		}
	}
	return false
}

func KueueWorkloadInadmissible(workload *kueuev1beta1.Workload) (bool, string) {
	for _, v := range workload.Status.Conditions {
		if v.Type == "QuotaReserved" && v.Status == "False" && v.Reason == "Inadmissible" {
			return true, v.Message
		}
	}
	return false, ""
}

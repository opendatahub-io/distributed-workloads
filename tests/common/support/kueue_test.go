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
	"testing"

	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"
)

func TestCreateKueueResourceFlavor(t *testing.T) {
	test := NewTest(t)

	rfSpec := kueuev1beta1.ResourceFlavorSpec{}

	rf := CreateKueueResourceFlavor(test, rfSpec)

	test.Expect(rf).To(gomega.Not(gomega.BeNil()))
	test.Expect(rf.GenerateName).To(gomega.Equal("rf-"))
}

func TestCreateKueueClusterQueue(t *testing.T) {
	test := NewTest(t)

	cqSpec := kueuev1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
	}

	cq := CreateKueueClusterQueue(test, cqSpec)

	test.Expect(cq).To(gomega.Not(gomega.BeNil()))
	test.Expect(cq.GenerateName).To(gomega.Equal("cq-"))
}

func TestCreateKueueLocalQueue(t *testing.T) {
	test := NewTest(t)

	lq := CreateKueueLocalQueue(test, "ns-1", "cq-1")

	test.Expect(lq).To(gomega.Not(gomega.BeNil()))
	test.Expect(lq.GenerateName).To(gomega.Equal("lq-"))
	annotationKey := "kueue.x-k8s.io/default-queue"
	_, exists := lq.Annotations[annotationKey]
	test.Expect(exists).To(gomega.BeFalse(), "Annotation key %s should not exist", annotationKey)
	test.Expect(lq.Namespace).To(gomega.Equal("ns-1"))
	test.Expect(lq.Spec.ClusterQueue).To(gomega.Equal(kueuev1beta1.ClusterQueueReference("cq-1")))

	default_lq := CreateKueueLocalQueue(test, "ns-2", "cq-2", AsDefaultQueue)

	test.Expect(default_lq).To(gomega.Not(gomega.BeNil()))
	test.Expect(default_lq.GenerateName).To(gomega.Equal("lq-"))
	test.Expect(default_lq.Annotations["kueue.x-k8s.io/default-queue"]).To(gomega.Equal("true"))
	test.Expect(default_lq.Namespace).To(gomega.Equal("ns-2"))
	test.Expect(default_lq.Spec.ClusterQueue).To(gomega.Equal(kueuev1beta1.ClusterQueueReference("cq-2")))

}

func TestGetKueueWorkloads(t *testing.T) {
	test := NewTest(t)

	wl := &kueuev1beta1.Workload{
		TypeMeta: metav1.TypeMeta{
			APIVersion: kueuev1beta1.SchemeGroupVersion.String(),
			Kind:       "Workload",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "wl1",
		},
	}

	_, err := test.Client().Kueue().KueueV1beta1().Workloads("ns-1").Create(test.ctx, wl, metav1.CreateOptions{})
	test.Expect(err).To(gomega.BeNil())

	wls := GetKueueWorkloads(test, "ns-1")

	test.Expect(wls).To(gomega.Not(gomega.BeNil()))
	test.Expect(wls).To(gomega.HaveLen(1))
	test.Expect(wls[0].Name).To(gomega.Equal("wl1"))
	test.Expect(wls[0].Namespace).To(gomega.Equal("ns-1"))
}

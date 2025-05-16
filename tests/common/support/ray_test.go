/*
Copyright 2025.

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
	rayv1alpha1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetRayJob(t *testing.T) {

	test := NewTest(t)

	RayJob := &rayv1alpha1.RayJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-job-1",
			Namespace: "my-namespace",
		},
	}

	test.client.Ray().RayV1().RayJobs("my-namespace").Create(test.ctx, RayJob, metav1.CreateOptions{})

	rayJob := GetRayJob(test, "my-namespace", "my-job-1")
	test.Expect(rayJob.Name).To(gomega.Equal("my-job-1"))
	test.Expect(rayJob.Namespace).To(gomega.Equal("my-namespace"))
}

func TestGetRayCluster(t *testing.T) {

	test := NewTest(t)

	RayCluster := &rayv1alpha1.RayCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-cluster-1",
			Namespace: "my-namespace",
		},
	}

	test.client.Ray().RayV1().RayClusters("my-namespace").Create(test.ctx, RayCluster, metav1.CreateOptions{})
	raycluster := GetRayCluster(test, "my-namespace", "my-cluster-1")

	test.Expect(raycluster.Name).To(gomega.Equal("my-cluster-1"))
	test.Expect(raycluster.Namespace).To(gomega.Equal("my-namespace"))
}

func TestGetRayClusters(t *testing.T) {

	test := NewTest(t)

	RayCluster := &rayv1alpha1.RayCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-cluster-1",
			Namespace: "my-namespace",
		},
	}

	test.client.Ray().RayV1().RayClusters("my-namespace").Create(test.ctx, RayCluster, metav1.CreateOptions{})
	rayclusters := GetRayClusters(test, "my-namespace")

	test.Expect(len(rayclusters)).To(gomega.Equal(1))
	test.Expect(rayclusters[0].Name).To(gomega.Equal("my-cluster-1"))
	test.Expect(rayclusters[0].Namespace).To(gomega.Equal("my-namespace"))
}

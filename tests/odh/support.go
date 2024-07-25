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
	"embed"

	"github.com/onsi/gomega"
	. "github.com/onsi/gomega"
	"github.com/project-codeflare/codeflare-common/support"
	. "github.com/project-codeflare/codeflare-common/support"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

//go:embed resources/*
var files embed.FS

func ReadFile(t support.Test, fileName string) []byte {
	t.T().Helper()
	file, err := files.ReadFile(fileName)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return file
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

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

package integration

import (
	"testing"

	. "github.com/onsi/gomega"
	cfosupport "github.com/project-codeflare/codeflare-operator/test/support"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/distributed-workloads/tests/integration/support"
)

func TestKubeRayRunning(t *testing.T) {
	test := cfosupport.With(t)

	kuberay, err := test.Client().Core().AppsV1().Deployments(support.GetOpenDataHubNamespace()).Get(test.Ctx(), "kuberay-operator", metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	// Assert the KubeRay Deployment is running
	test.Expect(kuberay).To(WithTransform(cfosupport.ConditionStatus(appsv1.DeploymentAvailable), Equal(corev1.ConditionTrue)))
}

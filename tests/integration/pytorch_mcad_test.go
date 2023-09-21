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
	mcadv1beta1 "github.com/project-codeflare/multi-cluster-app-dispatcher/pkg/apis/controller/v1beta1"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/distributed-workloads/tests/integration/support"
)

func TestMnistPyTorchMCAD(t *testing.T) {
	test := cfosupport.With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Test configuration
	jupyterNotebookConfigMapFileName := "mnist_mcad_mini.ipynb"
	config := &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "notebooks-mcad",
		},
		BinaryData: map[string][]byte{
			// MNIST MCAD Notebook
			jupyterNotebookConfigMapFileName: ReadFile(test, "resources/mnist_mcad_mini.ipynb"),
		},
		Immutable: cfosupport.Ptr(true),
	}
	config, err := test.Client().Core().CoreV1().ConfigMaps(namespace.Name).Create(test.Ctx(), config, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created ConfigMap %s/%s successfully", config.Namespace, config.Name)

	// Create RBAC, retrieve token for user with limited rights
	policyRules := []rbacv1.PolicyRule{
		{
			Verbs:     []string{"get", "create", "delete", "list", "patch", "update"},
			APIGroups: []string{mcadv1beta1.GroupName},
			Resources: []string{"appwrappers"},
		},
		// Needed for job.logs()
		{
			Verbs:     []string{"get"},
			APIGroups: []string{""},
			Resources: []string{"pods/log"},
		},
	}
	token := support.CreateTestRBAC(test, namespace, policyRules)

	// Create Notebook CR
	support.CreateNotebook(test, namespace, token, config.Name, jupyterNotebookConfigMapFileName)

	// Make sure the AppWrapper is created and running
	test.Eventually(cfosupport.AppWrappers(test, namespace), cfosupport.TestTimeoutLong).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(cfosupport.AppWrapperName, HavePrefix("mnistjob"))),
				ContainElement(WithTransform(cfosupport.AppWrapperState, Equal(mcadv1beta1.AppWrapperStateActive))),
			),
		)

	// Make sure the AppWrapper finishes and is deleted
	test.Eventually(cfosupport.AppWrappers(test, namespace), cfosupport.TestTimeoutLong).
		Should(HaveLen(0))
}

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
	"github.com/opendatahub-io/distributed-workloads/tests/integration/support"
	. "github.com/project-codeflare/codeflare-common/support"
	mcadv1beta1 "github.com/project-codeflare/multi-cluster-app-dispatcher/pkg/apis/controller/v1beta1"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	rbacv1 "k8s.io/api/rbac/v1"
)

func TestMCADRay(t *testing.T) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Test configuration
	jupyterNotebookConfigMapFileName := "mnist_ray_mini.ipynb"
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		// MNIST Ray Notebook
		jupyterNotebookConfigMapFileName: ReadFile(test, "resources/mnist_ray_mini.ipynb"),
		"mnist.py":                       ReadFile(test, "resources/mnist.py"),
		"requirements.txt":               ReadFile(test, "resources/requirements.txt"),
	})

	// Create RBAC, retrieve token for user with limited rights
	policyRules := []rbacv1.PolicyRule{
		{
			Verbs:     []string{"get", "create", "delete", "list", "patch", "update"},
			APIGroups: []string{mcadv1beta1.GroupName},
			Resources: []string{"appwrappers"},
		},
		{
			Verbs:     []string{"get", "list"},
			APIGroups: []string{rayv1.GroupVersion.Group},
			Resources: []string{"rayclusters", "rayclusters/status"},
		},
		{
			Verbs:     []string{"get", "list"},
			APIGroups: []string{"route.openshift.io"},
			Resources: []string{"routes"},
		},
	}

	// Create cluster wide RBAC, required for SDK OpenShift check
	// TODO reevaluate once SDK change OpenShift detection logic
	clusterPolicyRules := []rbacv1.PolicyRule{
		{
			Verbs:         []string{"get", "list"},
			APIGroups:     []string{"config.openshift.io"},
			Resources:     []string{"ingresses"},
			ResourceNames: []string{"cluster"},
		},
	}

	sa := CreateServiceAccount(test, namespace.Name)
	role := CreateRole(test, namespace.Name, policyRules)
	CreateRoleBinding(test, namespace.Name, sa, role)
	clusterRole := CreateClusterRole(test, clusterPolicyRules)
	CreateClusterRoleBinding(test, sa, clusterRole)
	token := CreateToken(test, namespace.Name, sa)

	// Create Notebook CR
	support.CreateNotebook(test, namespace, token, config.Name, jupyterNotebookConfigMapFileName)

	// Make sure the AppWrapper is created and running
	test.Eventually(AppWrappers(test, namespace), TestTimeoutLong).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(AppWrapperName, HavePrefix("mnisttest"))),
				ContainElement(WithTransform(AppWrapperState, Equal(mcadv1beta1.AppWrapperStateActive))),
			),
		)

	// Make sure the AppWrapper finishes and is deleted
	test.Eventually(AppWrappers(test, namespace), TestTimeoutLong).
		Should(HaveLen(0))
}

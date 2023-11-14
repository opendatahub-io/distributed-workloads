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
	rayv1alpha1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1alpha1"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/distributed-workloads/tests/integration/support"
)

func TestMCADRay(t *testing.T) {
	test := cfosupport.With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Test configuration
	jupyterNotebookConfigMapFileName := "mnist_ray_mini.ipynb"
	config := &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "notebooks-ray",
		},
		BinaryData: map[string][]byte{
			// MNIST MCAD Notebook
			jupyterNotebookConfigMapFileName: ReadFile(test, "resources/mnist_ray_mini.ipynb"),
			"mnist.py":                       ReadFile(test, "resources/mnist.py"),
			"requirements.txt":               ReadFile(test, "resources/requirements.txt"),
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
		{
			Verbs:     []string{"get", "list"},
			APIGroups: []string{rayv1alpha1.GroupVersion.Group},
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
	token, sa := support.CreateTestRBAC(test, namespace, policyRules)
	clusterRole := CreateClusterRole(test, clusterPolicyRules)
	CreateClusterRoleBinding(test, sa, clusterRole)

	// Create Notebook CR
	support.CreateNotebook(test, namespace, token, config.Name, jupyterNotebookConfigMapFileName)

	// Make sure the AppWrapper is created and running
	test.Eventually(cfosupport.AppWrappers(test, namespace), cfosupport.TestTimeoutLong).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(cfosupport.AppWrapperName, HavePrefix("mnisttest"))),
				ContainElement(WithTransform(cfosupport.AppWrapperState, Equal(mcadv1beta1.AppWrapperStateActive))),
			),
		)

	// Make sure the AppWrapper finishes and is deleted
	test.Eventually(cfosupport.AppWrappers(test, namespace), cfosupport.TestTimeoutLong).
		Should(HaveLen(0))
}

func CreateClusterRoleBinding(t cfosupport.Test, serviceAccount *corev1.ServiceAccount, role *rbacv1.ClusterRole) *rbacv1.ClusterRoleBinding {
	t.T().Helper()

	roleBinding := &rbacv1.ClusterRoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rbacv1.SchemeGroupVersion.String(),
			Kind:       "ClusterRoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "crb-",
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.SchemeGroupVersion.Group,
			Kind:     "ClusterRole",
			Name:     role.Name,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				APIGroup:  corev1.SchemeGroupVersion.Group,
				Name:      serviceAccount.Name,
				Namespace: serviceAccount.Namespace,
			},
		},
	}
	rb, err := t.Client().Core().RbacV1().ClusterRoleBindings().Create(t.Ctx(), roleBinding, metav1.CreateOptions{})
	t.Expect(err).NotTo(HaveOccurred())
	t.T().Logf("Created ClusterRoleBinding %s/%s successfully", role.Namespace, role.Name)

	t.T().Cleanup(func() {
		t.Client().Core().RbacV1().ClusterRoleBindings().Delete(t.Ctx(), rb.Name, metav1.DeleteOptions{})
	})

	return rb
}

func CreateClusterRole(t cfosupport.Test, policyRules []rbacv1.PolicyRule) *rbacv1.ClusterRole {
	t.T().Helper()

	role := &rbacv1.ClusterRole{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rbacv1.SchemeGroupVersion.String(),
			Kind:       "ClusterRole",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "clusterrole-",
		},
		Rules: policyRules,
	}
	role, err := t.Client().Core().RbacV1().ClusterRoles().Create(t.Ctx(), role, metav1.CreateOptions{})
	t.Expect(err).NotTo(HaveOccurred())
	t.T().Logf("Created ClusterRole %s/%s successfully", role.Namespace, role.Name)

	t.T().Cleanup(func() {
		t.Client().Core().RbacV1().ClusterRoles().Delete(t.Ctx(), role.Name, metav1.DeleteOptions{})
	})

	return role
}

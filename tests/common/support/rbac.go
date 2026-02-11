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

package support

import (
	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func CreateRole(t Test, namespace string, policyRules []rbacv1.PolicyRule) *rbacv1.Role {
	t.T().Helper()

	role := &rbacv1.Role{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rbacv1.SchemeGroupVersion.String(),
			Kind:       "Role",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "role-",
			Namespace:    namespace,
		},
		Rules: policyRules,
	}
	role, err := t.Client().Core().RbacV1().Roles(namespace).Create(t.Ctx(), role, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created Role %s/%s successfully", role.Namespace, role.Name)

	return role
}

func CreateClusterRole(t Test, policyRules []rbacv1.PolicyRule) *rbacv1.ClusterRole {
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
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created ClusterRole %s/%s successfully", role.Namespace, role.Name)

	t.T().Cleanup(func() {
		t.Client().Core().RbacV1().ClusterRoles().Delete(t.Ctx(), role.Name, metav1.DeleteOptions{})
	})

	return role
}

func CreateRoleBinding(t Test, namespace string, serviceAccount *corev1.ServiceAccount, role *rbacv1.Role) *rbacv1.RoleBinding {
	t.T().Helper()

	roleBinding := &rbacv1.RoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rbacv1.SchemeGroupVersion.String(),
			Kind:       "RoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "rb-",
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.SchemeGroupVersion.Group,
			Kind:     "Role",
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
	rb, err := t.Client().Core().RbacV1().RoleBindings(namespace).Create(t.Ctx(), roleBinding, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created RoleBinding %s/%s successfully", rb.Namespace, rb.Name)

	return rb
}

func CreateUserRoleBinding(t Test, namespace string, userName string, roleName string) *rbacv1.RoleBinding {
	t.T().Helper()

	roleBinding := &rbacv1.RoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rbacv1.SchemeGroupVersion.String(),
			Kind:       "RoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "rb-",
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.SchemeGroupVersion.Group,
			Kind:     "Role",
			Name:     roleName,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:     "User",
				APIGroup: rbacv1.SchemeGroupVersion.Group,
				Name:     userName,
			},
		},
	}
	rb, err := t.Client().Core().RbacV1().RoleBindings(namespace).Create(t.Ctx(), roleBinding, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created User RoleBinding %s/%s successfully", rb.Namespace, rb.Name)

	return rb
}

func CreateClusterRoleBinding(t Test, serviceAccount *corev1.ServiceAccount, role *rbacv1.ClusterRole) *rbacv1.ClusterRoleBinding {
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
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created ClusterRoleBinding %s successfully", rb.Name)

	t.T().Cleanup(func() {
		t.Client().Core().RbacV1().ClusterRoleBindings().Delete(t.Ctx(), rb.Name, metav1.DeleteOptions{})
	})

	return rb
}

func CreateUserClusterRoleBinding(t Test, userName string, roleName string) *rbacv1.ClusterRoleBinding {
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
			Name:     roleName,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:     "User",
				APIGroup: rbacv1.SchemeGroupVersion.Group,
				Name:     userName,
			},
		},
	}
	rb, err := t.Client().Core().RbacV1().ClusterRoleBindings().Create(t.Ctx(), roleBinding, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created User ClusterRoleBinding %s successfully", rb.Name)

	t.T().Cleanup(func() {
		t.Client().Core().RbacV1().ClusterRoleBindings().Delete(t.Ctx(), rb.Name, metav1.DeleteOptions{})
	})

	return rb
}

func CreateUserRoleBindingWithClusterRole(t Test, userName string, namespace string, roleName string) *rbacv1.RoleBinding {
	t.T().Helper()

	// Create a RoleBinding to give specified role access to the user for given namespace
	roleBinding := &rbacv1.RoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rbacv1.SchemeGroupVersion.String(),
			Kind:       "RoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "rb-",
			Namespace:    namespace,
		},
		RoleRef: rbacv1.RoleRef{
			Kind:     "ClusterRole",
			Name:     roleName, // grants specified role access
			APIGroup: rbacv1.SchemeGroupVersion.Group,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:     "User",
				Name:     userName,
				APIGroup: rbacv1.SchemeGroupVersion.Group,
			},
		},
	}

	rb, err := t.Client().Core().RbacV1().RoleBindings(namespace).Create(t.Ctx(), roleBinding, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created User RoleBinding %s/%s successfully", rb.Namespace, rb.Name)

	return rb
}

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

package trainer

import (
	"testing"

	"github.com/onsi/gomega"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestGrantTrainerUserAccess(t *testing.T) {
	test := NewTest(t)
	namespace := test.NewTestNamespace()

	GrantTrainerUserAccess(test, "ldap-user", namespace.Name)

	rbs, err := test.Client().Core().RbacV1().RoleBindings(namespace.Name).List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.Expect(rbs.Items).To(gomega.HaveLen(1))
	test.Expect(rbs.Items[0].RoleRef).To(gomega.Equal(rbacv1.RoleRef{
		APIGroup: rbacv1.SchemeGroupVersion.Group,
		Kind:     "ClusterRole",
		Name:     TrainingEditClusterRole,
	}))
	test.Expect(rbs.Items[0].Subjects).To(gomega.ConsistOf(rbacv1.Subject{
		Kind:     "User",
		Name:     "ldap-user",
		APIGroup: rbacv1.SchemeGroupVersion.Group,
	}))

	crbs, err := test.Client().Core().RbacV1().ClusterRoleBindings().List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.Expect(crbs.Items).To(gomega.HaveLen(1))
	test.Expect(crbs.Items[0].Subjects[0].Name).To(gomega.Equal("ldap-user"))
}

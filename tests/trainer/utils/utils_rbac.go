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
	rbacv1 "k8s.io/api/rbac/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

// CreateUserClusterRoleBindingForTrainerRuntimes creates a ClusterRole with get/list/watch access
// to ClusterTrainingRuntimes and binds it to the specified user.
// This is needed because ClusterTrainingRuntimes are cluster-scoped resources.
func CreateUserClusterRoleBindingForTrainerRuntimes(t Test, userName string) *rbacv1.ClusterRoleBinding {
	t.T().Helper()

	// Create minimal ClusterRole for trainer runtime read access
	role := CreateClusterRole(t, []rbacv1.PolicyRule{
		{
			APIGroups: []string{"trainer.kubeflow.org"},
			Resources: []string{"clustertrainingruntimes"},
			Verbs:     []string{"get", "list", "watch"},
		},
	})

	// Bind the role to the user
	return CreateUserClusterRoleBinding(t, userName, role.Name)
}

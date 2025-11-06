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
	"fmt"
	"os/exec"
	"testing"

	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

// EnsureTrainerClusterReady verifies cluster dependencies required by Kubeflow Trainer tests.
func EnsureTrainerClusterReady(t *testing.T, test Test) {
	t.Helper()
	// JobSet CRD present
	// TODO: Remove once trainer is part of installation
	if out, err := exec.Command("kubectl", "get", "crd", "jobsets.jobset.x-k8s.io").CombinedOutput(); err != nil {
		t.Fatalf("JobSet CRD missing: %v\n%s", err, string(out))
	}
	// Trainer controller deployment available
	// TODO: Remove once trainer is part of installation
	if out, err := exec.Command("kubectl", "-n", "opendatahub", "wait", "--for=condition=available", "--timeout=180s", "deploy/kubeflow-trainer-controller-manager").CombinedOutput(); err != nil {
		t.Fatalf("Trainer controller not available: %v\n%s", err, string(out))
	}
	// Required ClusterTrainingRuntimes present
	runtimes, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "Failed to list ClusterTrainingRuntimes")
	found := map[string]bool{}
	for _, rt := range runtimes.Items {
		found[rt.Name] = true
	}
	// TODO: Extend / tweak with universal image runtime once available
	for _, name := range []string{"torch-cuda-241", "torch-cuda-251", "torch-rocm-241", "torch-rocm-251"} {
		test.Expect(found[name]).To(BeTrue(), fmt.Sprintf("Expected ClusterTrainingRuntime '%s' not found", name))
	}
}

// EnsureNotebookRBAC sets up the Notebook ServiceAccount and RBAC so that notebooks can
// read ClusterTrainingRuntimes (cluster-scoped), and create/read TrainJobs and pod logs in the namespace.
func EnsureNotebookRBAC(t *testing.T, test Test, namespace string) {
	t.Helper()

	// Ensure ServiceAccount exists
	saName := "jupyter-nb-kube-3aadmin"
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: namespace}}
	_, _ = test.Client().Core().CoreV1().ServiceAccounts(namespace).Create(test.Ctx(), sa, metav1.CreateOptions{})
	// Get current SA (created or existing)
	saObj, err := test.Client().Core().CoreV1().ServiceAccounts(namespace).Get(test.Ctx(), saName, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	// Cluster-scoped read for ClusterTrainingRuntimes
	ctrRead := CreateClusterRole(test, []rbacv1.PolicyRule{
		{APIGroups: []string{"trainer.kubeflow.org"}, Resources: []string{"clustertrainingruntimes"}, Verbs: []string{"get", "list", "watch"}},
	})
	CreateClusterRoleBinding(test, saObj, ctrRead)

	// Namespace Role for TrainJobs and pods/log access
	role := CreateRole(test, namespace, []rbacv1.PolicyRule{
		{APIGroups: []string{"trainer.kubeflow.org"}, Resources: []string{"trainjobs", "trainjobs/status"}, Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
		{APIGroups: []string{""}, Resources: []string{"pods", "pods/log"}, Verbs: []string{"get", "list", "watch"}},
	})
	CreateRoleBinding(test, namespace, saObj, role)
}

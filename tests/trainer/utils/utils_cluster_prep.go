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

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

// EnsureNotebookServiceAccount ensures the Notebook ServiceAccount exists in the target namespace.
// This avoids webhook/controller failures when creating the Notebook CR.
func EnsureNotebookServiceAccount(t *testing.T, test Test, namespace string) {
	t.Helper()
	saName := "jupyter-nb-kube-3aadmin"
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: namespace}}
	_, err := test.Client().Core().CoreV1().ServiceAccounts(namespace).Create(test.Ctx(), sa, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("Failed to create ServiceAccount %s/%s: %v", namespace, saName, err)
	}
}

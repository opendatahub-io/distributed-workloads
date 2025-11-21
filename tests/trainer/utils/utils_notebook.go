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
	"strings"
	"time"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	common "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

// CreateNotebookFromBytes creates a ConfigMap with the notebook and a Notebook CR to run it.
func CreateNotebookFromBytes(test Test, namespace *corev1.Namespace, userToken string, notebookName string, notebookBytes []byte, command []string, numGpus int, containerSize common.ContainerSize) *corev1.ConfigMap {
	cm := CreateConfigMap(test, namespace.Name, map[string][]byte{notebookName: notebookBytes})
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, notebookName, numGpus, CreatePersistentVolumeClaim(test, namespace.Name, "10Gi", AccessModes(corev1.ReadWriteOnce)), containerSize)
	return cm
}

// WaitForNotebookPodRunning waits for the Notebook pod (identified by the template's label) to be Running and returns pod/container names.
func WaitForNotebookPodRunning(test Test, namespace string) (string, string) {
	labelSelector := fmt.Sprintf("notebook-name=%s", common.NOTEBOOK_CONTAINER_NAME)
	test.Eventually(func() []corev1.Pod {
		return GetPods(test, namespace, metav1.ListOptions{LabelSelector: labelSelector, FieldSelector: "status.phase=Running"})
	}, TestTimeoutLong).Should(HaveLen(1), "Expected exactly one notebook pod")

	pods := GetPods(test, namespace, metav1.ListOptions{LabelSelector: labelSelector, FieldSelector: "status.phase=Running"})
	return pods[0].Name, pods[0].Spec.Containers[0].Name
}

// PollNotebookLogsForStatus polls the notebook container logs until a definitive NOTEBOOK_STATUS line appears or timeout.
func PollNotebookLogsForStatus(test Test, namespace, podName, containerName string, timeout time.Duration) error {
	var finalErr error

	// Tail last N lines similar to the previous kubectl --tail
	var tail int64 = 2000
	getLogs := PodLog(test, namespace, podName, corev1.PodLogOptions{
		Container: containerName,
		TailLines: &tail,
	})

	// Track failure signal while polling
	sawFailure := false
	test.Eventually(func() bool {
		logs := getLogs(test)
		switch {
		case strings.Contains(logs, "NOTEBOOK_STATUS: SUCCESS"):
			return true
		case strings.Contains(logs, "NOTEBOOK_STATUS: FAILURE"):
			sawFailure = true
			return true
		default:
			return false
		}
	}, timeout).Should(BeTrue(), "Notebook did not reach definitive state")

	if sawFailure {
		finalErr = fmt.Errorf("Notebook execution failed")
	}
	return finalErr
}

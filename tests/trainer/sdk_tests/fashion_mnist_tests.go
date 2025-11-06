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

package sdk_tests

import (
	"fmt"
	"os"
	"testing"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"

	common "github.com/opendatahub-io/distributed-workloads/tests/common"
	support "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	notebookName = "mnist.ipynb"
	notebookPath = "resources/" + notebookName
)

// CPU Only - Distributed Training
func RunFashionMnistCpuDistributedTraining(t *testing.T) {
	test := support.With(t)

	// Create a new test namespace
	namespace := test.NewTestNamespace()

	// Ensure pre-requisites to run the test are met
	trainerutils.EnsureTrainerClusterReady(t, test)

	// Ensure Notebook SA and RBACs are set for this namespace
	trainerutils.EnsureNotebookRBAC(t, test, namespace.Name)

	// RBACs setup
	userName := common.GetNotebookUserName(test)
	userToken := common.GetNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Read notebook from directory
	localPath := notebookPath
	nb, err := os.ReadFile(localPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", localPath))

	// Create ConfigMap with notebook
	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{notebookName: nb})

	// Build command
	marker := "/opt/app-root/src/notebook_completion_marker"
	shellCmd := trainerutils.BuildPapermillShellCmd(notebookName, marker, nil)
	command := []string{"/bin/sh", "-c", shellCmd}

	// Create Notebook CR (with default 10Gi PVC)
	pvc := support.CreatePersistentVolumeClaim(test, namespace.Name, "10Gi", support.AccessModes(corev1.ReadWriteOnce))
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, notebookName, 0, pvc, common.ContainerSizeSmall)

	// Cleanup
	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), support.TestTimeoutLong).Should(HaveLen(0))
	}()

	// Wait for the Notebook Pod and get pod/container names
	podName, containerName := trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	// Poll marker file to check if the notebook execution completed successfully
	if err := trainerutils.PollNotebookCompletionMarker(test, namespace.Name, podName, containerName, marker, support.TestTimeoutDouble); err != nil {
		test.Expect(err).To(Succeed(), "Notebook execution reported FAILURE")
	}
}

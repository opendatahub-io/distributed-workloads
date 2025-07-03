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

package odh

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	corev1 "k8s.io/api/core/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestRaytuneOaiMrGrpcCpu(t *testing.T) {
	raytuneHpo(t, 0)
}

func TestRaytuneOaiMrGrpcGpu(t *testing.T) {
	raytuneHpo(t, 1)
}

func raytuneHpo(t *testing.T, numGpus int) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Get current working directory
	workingDirectory, err := os.Getwd()
	test.Expect(err).ToNot(HaveOccurred())

	// Start the Model Registry service with PostgreSQL as the backend database
	model_registry_postgres_deplyment_yamls := []string{
		"https://raw.githubusercontent.com/opendatahub-io/model-registry-operator/refs/heads/release/v0.2.9/config/samples/postgres/kustomization.yaml",
		"https://raw.githubusercontent.com/opendatahub-io/model-registry-operator/refs/heads/release/v0.2.9/config/samples/postgres/modelregistry_v1alpha1_modelregistry.yaml",
		"https://raw.githubusercontent.com/opendatahub-io/model-registry-operator/refs/heads/release/v0.2.9/config/samples/postgres/postgres-db.yaml",
	}

	outputDir, err := os.MkdirTemp("", "github_downloads_*")
	test.Expect(err).ToNot(HaveOccurred(), fmt.Sprintf("Error in creating directory: %s", err))
	test.T().Logf("Temporary directory created : %s", outputDir)

	// Loop through each url and download the file using curl
	for _, url := range model_registry_postgres_deplyment_yamls {
		fileName := filepath.Base(url) // Extract filename from url
		outputPath := filepath.Join(outputDir, fileName)
		cmd := exec.Command("curl", "-L", "-o", outputPath, "--create-dirs", url)
		if err := cmd.Run(); err != nil {
			test.T().Logf(fmt.Sprintf("Failed to download %s: %v\n", url, err.Error()))
			test.Expect(err).ToNot(HaveOccurred())
		}
		test.T().Logf("File '%s' downloaded sucessfully", fileName)
	}
	defer os.RemoveAll(outputDir)

	cmd := exec.Command("kubectl", "apply", "-n", string(namespace.Name), "-k", fmt.Sprintf("%s/", outputDir))
	if err := cmd.Run(); err != nil {
		test.T().Logf("Failed to start the Model Registry service with PostgreSQL: %v\n", err.Error())
		test.Expect(err).ToNot(HaveOccurred())
	} else {
		test.T().Logf(fmt.Sprint("Successfully started the Model Registry service with PostgreSQL"))
	}

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// list changes required in llm-deepspeed-finetune-demo.ipynb file and update those
	requiredChangesInNotebook := map[string]string{
		"token = 'TOKEN'":           fmt.Sprintf("token='%s'", userToken),
		"server = 'SERVER'":         fmt.Sprintf("server='%s'", GetOpenShiftApiUrl(test)),
		"name='terrestial-raytest'": fmt.Sprintf("name='%s',\\n\",\n\t\t\"    namespace='%s'", "terrestial-raytest", namespace.Name),
		"worker_extended_resource_requests={'nvidia.com/gpu':0}": fmt.Sprintf("worker_extended_resource_requests={'nvidia.com/gpu':%d}", numGpus),
		"image='quay.io/modh/ray:2.35.0-py311-cu121'":            fmt.Sprintf("image='%s'", GetRayImage()),
	}

	updatedNotebookContent := string(ReadFileExt(test, workingDirectory+"/../../examples/hpo-raytune/notebook/raytune-oai-MR-gRPC-demo.ipynb"))
	for oldValue, newValue := range requiredChangesInNotebook {
		updatedNotebookContent = strings.Replace(updatedNotebookContent, oldValue, newValue, -1)
	}
	updatedNotebook := []byte(updatedNotebookContent)

	// Test configuration
	jupyterNotebookConfigMapFileName := "raytune-oai-MR-gRPC-demo.ipynb"
	configMap := map[string][]byte{
		jupyterNotebookConfigMapFileName: updatedNotebook,
	}

	config := CreateConfigMap(test, namespace.Name, configMap)

	// Get ray image
	rayImage := GetRayImage()

	notebookCommand := getNotebookCommand(rayImage)

	// Create PVC for Notebook
	notebookPVC := CreatePersistentVolumeClaim(test, namespace.Name, "10Gi", AccessModes(corev1.ReadWriteOnce))

	// Create Notebook CR
	CreateNotebook(test, namespace, userToken, notebookCommand, config.Name, jupyterNotebookConfigMapFileName, numGpus, notebookPVC, ContainerSizeSmall)

	// Gracefully cleanup Notebook
	defer func() {
		DeleteNotebook(test, namespace)
		test.Eventually(ListNotebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Make sure the RayCluster is created and running
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutGpuProvisioning).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(RayClusterState, Equal(rayv1.Ready))),
			),
		)

	// Make sure the RayCluster finishes and is deleted
	test.Eventually(PodLog(test, namespace.Name, NOTEBOOK_POD_NAME, corev1.PodLogOptions{Container: NOTEBOOK_CONTAINER_NAME}), 20*time.Minute).
		Should(ContainSubstring("Model Prediction:"))

}

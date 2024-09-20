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
	"strings"
	"testing"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
)

func TestRaytuneOaiMrGrpcCpu(t *testing.T) {
	raytuneHpo(t, 0)
}

func raytuneHpo(t *testing.T, numGpus int) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Get current working directory
	workingDirectory, err := os.Getwd()
	test.Expect(err).ToNot(HaveOccurred())

	// Start the Model Registry service with PostgreSQL as the backend database
	cmd := exec.Command("kubectl", "apply", "-n", string(namespace.Name), "-k", fmt.Sprintf("%s/resources/model_registry_config_samples", workingDirectory))
	stdout, err := cmd.Output()
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	// Print the cmd output
	fmt.Println(string(stdout))

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// list changes required in llm-deepspeed-finetune-demo.ipynb file and update those
	requiredChangesInNotebook := map[string]string{
		"skip_tls=True":             "skip_tls=False",
		"token = 'TOKEN'":           fmt.Sprintf("token='%s'", userToken),
		"server = 'SERVER'":         fmt.Sprintf("server='%s'", GetOpenShiftApiUrl(test)),
		"name='terrestial-raytest'": fmt.Sprintf("name='%s',\\n\",\n\t\t\"    namespace='%s'", "terrestial-raytest", namespace.Name),
		"worker_extended_resource_requests={'nvidia.com/gpu':0}": fmt.Sprintf("worker_extended_resource_requests={'nvidia.com/gpu':%d}", numGpus),
		"image='quay.io/modh/ray:2.35.0-py39-cu121'":             fmt.Sprintf("image='%s'", GetRayImage()),
		"print('Model Prediction:', prediction)":                 "print('Model Prediction:', prediction)\\n\",\n\t\"time.sleep(10)",
	}

	// updatedNotebookContent := string(ReadFileExt(test, workingDirectory+"/../../examples/hpo-raytune/notebook/raytune-oai-MR-gRPC-demo.ipynb"))
	updatedNotebookContent := string(ReadFile(test, "resources/raytune-oai-MR-gRPC-demo.ipynb"))
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

	// Create Notebook CR
	createNotebook(test, namespace, userToken, rayImage, config.Name, jupyterNotebookConfigMapFileName, numGpus)

	// Gracefully cleanup Notebook
	defer func() {
		deleteNotebook(test, namespace)
		test.Eventually(listNotebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
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
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutLong).
		Should(BeEmpty())
}

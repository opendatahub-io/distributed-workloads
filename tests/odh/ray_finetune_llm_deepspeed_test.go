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
	"testing"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
)

func TestRayFinetuneDemo(t *testing.T) {
	mnistRayLlmFinetune(t, 1)
}

func mnistRayLlmFinetune(t *testing.T, numGpus int) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Test configuration
	jupyterNotebookConfigMapFileName := "ray_finetune_llm_deepspeed.ipynb"

	// Test configuration
	configMap := map[string][]byte{
		// MNIST Ray Notebook
		jupyterNotebookConfigMapFileName: ReadFile(test, "resources/ray_finetune_demo/ray_finetune_llm_deepspeed.ipynb"),
		"ray_finetune_llm_deepspeed.py":  ReadFile(test, "resources/ray_finetune_demo/ray_finetune_llm_deepspeed.py"),
		"ray_finetune_requirements.txt":  ReadRayFinetuneRequirementsTxt(test),
		"create_dataset.py":              ReadFile(test, "resources/ray_finetune_demo/create_dataset.py"),
		"lora.json":                      ReadFile(test, "resources/ray_finetune_demo/lora.json"),
		"zero_3_llama_2_7b.json":         ReadFile(test, "resources/ray_finetune_demo/zero_3_llama_2_7b.json"),
		"utils.py":                       ReadFile(test, "resources/ray_finetune_demo/utils.py"),
	}

	config := CreateConfigMap(test, namespace.Name, configMap)

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Create Notebook CR
	createNotebook(test, namespace, userToken, config.Name, jupyterNotebookConfigMapFileName, numGpus)

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
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutGpuProvisioning).
		Should(HaveLen(0))
}

func ReadRayFinetuneRequirementsTxt(test Test) []byte {
	// Read the requirements.txt from resources and perform replacements for custom values using go template
	props := struct {
		PipIndexUrl    string
		PipTrustedHost string
	}{
		PipIndexUrl: "--index " + string(GetPipIndexURL()),
	}

	// Provide trusted host only if defined
	if len(GetPipTrustedHost()) > 0 {
		props.PipTrustedHost = "--trusted-host " + GetPipTrustedHost()
	}

	template, err := files.ReadFile("resources/ray_finetune_demo/ray_finetune_requirements.txt")
	test.Expect(err).NotTo(HaveOccurred())

	return ParseTemplate(test, template, props)
}

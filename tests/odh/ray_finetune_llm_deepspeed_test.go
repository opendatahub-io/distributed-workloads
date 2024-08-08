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
	"strings"
	"testing"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
)

func TestRayFinetuneLlmDeepspeedDemo(t *testing.T) {
	rayFinetuneLlmDeepspeed(t, 1)
}

func rayFinetuneLlmDeepspeed(t *testing.T, numGpus int) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()
	var workingDirectory, err = os.Getwd()
	test.Expect(err).ToNot(HaveOccurred())

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// list changes required in llm-deepspeed-finetune-demo.ipynb file and update those
	requiredChangesInNotebook := map[string]string{
		"import os":  "import os,time,sys",
		"import sys": "!cp /opt/app-root/notebooks/* ./",
		"from codeflare_sdk.cluster.auth import TokenAuthentication": "from codeflare_sdk.cluster.auth import TokenAuthentication\\n\",\n\t\"from codeflare_sdk.job import RayJobClient",
		"token = ''":                             fmt.Sprintf("token = '%s'", userToken),
		"server = ''":                            fmt.Sprintf("server = '%s'", GetOpenShiftApiUrl(test)),
		"namespace='ray-finetune-llm-deepspeed'": fmt.Sprintf("namespace='%s'", namespace.Name),
		"head_cpus=16":                           "head_cpus=2",
		"head_gpus=1":                            "head_gpus=0",
		"num_workers=7":                          "num_workers=1",
		"min_cpus=16":                            "min_cpus=4",
		"max_cpus=16":                            "max_cpus=4",
		"min_memory=128":                         "min_memory=48",
		"max_memory=256":                         "max_memory=48",
		"head_memory=128":                        "head_memory=48",
		"num_gpus=1":                             fmt.Sprintf("worker_extended_resource_requests={'nvidia.com/gpu': %d},\\n\",\n\t\"    write_to_file=True,\\n\",\n\t\"    verify_tls=False", numGpus),
		"image='quay.io/rhoai/ray:2.23.0-py39-cu121'":            fmt.Sprintf("image='%s'", GetRayImage()),
		"client = cluster.job_client":                            "ray_dashboard = cluster.cluster_dashboard_uri()\\n\",\n\t\"header = {\\\"Authorization\\\": \\\"Bearer " + userToken + "\\\"}\\n\",\n\t\"client = RayJobClient(address=ray_dashboard, headers=header, verify=False)\\n",
		"--num-devices=8":                                        fmt.Sprintf("--num-devices=%d", numGpus),
		"--num-epochs=3":                                         fmt.Sprintf("--num-epochs=%d", 1),
		"--ds-config=./deepspeed_configs/zero_3_llama_2_7b.json": "--ds-config=./zero_3_llama_2_7b.json \\\"\\n\",\n\t\"               \\\"--lora-config=./lora.json \\\"\\n\",\n\t\"               \\\"--as-test",
		"'pip': 'requirements.txt'":                              "'pip': '/opt/app-root/src/requirements.txt'",
		"'working_dir': './'":                                    "'working_dir': '/opt/app-root/src'",
		"client.stop_job(submission_id)":                         "finished = False\\n\",\n\t\"while not finished:\\n\",\n\t\"    time.sleep(1)\\n\",\n\t\"    status = client.get_job_status(submission_id)\\n\",\n\t\"    finished = (status == \\\"SUCCEEDED\\\")\\n\",\n\t\"if finished:\\n\",\n\t\"    print(\\\"Job completed Successfully !\\\")\\n\",\n\t\"else:\\n\",\n\t\"    print(\\\"Job failed !\\\")\\n\",\n\t\"time.sleep(10)\\n",
	}

	updatedNotebookContent := string(ReadFileExt(test, workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/ray_finetune_llm_deepspeed.ipynb"))
	for oldValue, newValue := range requiredChangesInNotebook {
		updatedNotebookContent = strings.Replace(updatedNotebookContent, oldValue, newValue, -1)
	}
	updatedNotebook := []byte(updatedNotebookContent)

	// Test configuration
	jupyterNotebookConfigMapFileName := "ray_finetune_llm_deepspeed.ipynb"
	configMap := map[string][]byte{
		jupyterNotebookConfigMapFileName: updatedNotebook,
		"ray_finetune_llm_deepspeed.py":  ReadFileExt(test, workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/ray_finetune_llm_deepspeed.py"),
		"requirements.txt":               ReadFileExt(test, workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/requirements.txt"),
		"create_dataset.py":              ReadFileExt(test, workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/create_dataset.py"),
		"lora.json":                      ReadFileExt(test, workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/lora_configs/lora.json"),
		"zero_3_llama_2_7b.json":         ReadFileExt(test, workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/deepspeed_configs/zero_3_llama_2_7b.json"),
		"utils.py":                       ReadFileExt(test, workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/utils.py"),
	}

	config := CreateConfigMap(test, namespace.Name, configMap)

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
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutMedium).
		Should(HaveLen(0))
}

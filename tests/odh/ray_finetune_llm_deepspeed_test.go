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
	"time"

	. "github.com/onsi/gomega"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestRayFinetuneLlmDeepspeedDemoLlama_2_7b(t *testing.T) {
	rayFinetuneLlmDeepspeed(t, 1, "meta-llama/Llama-2-7b-chat-hf", "zero_3_llama_2_7b.json")
}
func TestRayFinetuneLlmDeepspeedDemoLlama_31_8b(t *testing.T) {
	rayFinetuneLlmDeepspeed(t, 1, "meta-llama/Meta-Llama-3.1-8B", "zero_3_offload_optim_param.json")
}

func rayFinetuneLlmDeepspeed(t *testing.T, numGpus int, modelName string, modelConfigFile string) {
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
		"import sys": "!cp /opt/app-root/notebooks/* ./\\n\",\n\t\"!ls",
		"from codeflare_sdk.cluster.auth import TokenAuthentication": "from codeflare_sdk.cluster.auth import TokenAuthentication\\n\",\n\t\"from codeflare_sdk.job import RayJobClient",
		"token = ''":                                fmt.Sprintf("token = '%s'", userToken),
		"server = ''":                               fmt.Sprintf("server = '%s'", GetOpenShiftApiUrl(test)),
		"namespace='ray-finetune-llm-deepspeed'":    fmt.Sprintf("namespace='%s'", namespace.Name),
		"head_cpu_requests=16":                      "head_cpu_requests=2",
		"head_cpu_limits=16":                        "head_cpu_limits=2",
		"head_extended_resource_requests=1":         "head_extended_resource_requests=0",
		"num_workers=7":                             "num_workers=1",
		"worker_cpu_requests=16":                    "worker_cpu_requests=4",
		"worker_cpu_limits=16":                      "worker_cpu_limits=4",
		"worker_memory_requests=128":                "worker_memory_requests=64",
		"worker_memory_limits=256":                  "worker_memory_limits=128",
		"head_memory_requests=128":                  "head_memory_requests=48",
		"head_memory_limits=256":                    "head_memory_limits=48",
		"client = cluster.job_client":               "ray_dashboard = cluster.cluster_dashboard_uri()\\n\",\n\t\"header = {\\\"Authorization\\\": \\\"Bearer " + userToken + "\\\"}\\n\",\n\t\"client = RayJobClient(address=ray_dashboard, headers=header, verify=False)\\n",
		"--num-devices=8":                           fmt.Sprintf("--num-devices=%d", numGpus),
		"--num-epochs=3":                            fmt.Sprintf("--num-epochs=%d", 1),
		"--model-name=meta-llama/Meta-Llama-3.1-8B": fmt.Sprintf("--model-name=%s", modelName),
		"--ds-config=./deepspeed_configs/zero_3_offload_optim_param.json": fmt.Sprintf("--ds-config=./%s \\\"\\n\",\n\t\"               \\\"--lora-config=./lora.json \\\"\\n\",\n\t\"               \\\"--as-test", modelConfigFile),
		"--batch-size-per-device=32":                                      "--batch-size-per-device=6",
		"--eval-batch-size-per-device=32":                                 "--eval-batch-size-per-device=6",
		"'pip': 'requirements.txt'":                                       "'pip': '/opt/app-root/src/requirements.txt'",
		"'working_dir': './'":                                             "'working_dir': '/opt/app-root/src'",
		"client.stop_job(submission_id)":                                  "finished = False\\n\",\n\t\"while not finished:\\n\",\n\t\"    time.sleep(1)\\n\",\n\t\"    status = client.get_job_status(submission_id)\\n\",\n\t\"    finished = (status == \\\"SUCCEEDED\\\")\\n\",\n\t\"if finished:\\n\",\n\t\"    print(\\\"Job completed Successfully !\\\")\\n\",\n\t\"else:\\n\",\n\t\"    print(\\\"Job failed !\\\")\\n\",\n\t\"time.sleep(10)\\n",
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
		modelConfigFile:                  ReadFileExt(test, fmt.Sprintf(workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/deepspeed_configs/%s", modelConfigFile)),
		"utils.py":                       ReadFileExt(test, workingDirectory+"/../../examples/ray-finetune-llm-deepspeed/utils.py"),
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

	// Fetch created raycluster
	rayClusterName := "ray"
	rayCluster, err := test.Client().Ray().RayV1().RayClusters(namespace.Name).Get(test.Ctx(), rayClusterName, metav1.GetOptions{})
	test.Expect(err).ToNot(HaveOccurred())

	// Initialise raycluster client to interact with raycluster to get rayjob details using REST-API
	dashboardUrl := GetDashboardUrl(test, namespace, rayCluster)
	rayClient := GetRayClusterClient(test, dashboardUrl, test.Config().BearerToken)

	// wait until rayjob exists
	test.Eventually(func() ([]RayJobDetailsResponse, error) {
		return rayClient.ListJobs()
	}, TestTimeoutMedium, 1*time.Second).Should(HaveLen(1), "Ray job not found")

	// Get test job-id
	jobID := GetTestJobId(test, rayClient)
	test.Expect(jobID).ToNot(BeEmpty())

	// Wait for the job to be succeeded or failed
	var rayJobStatus string
	test.T().Logf("Waiting for job to be Succeeded...\n")
	test.Eventually(func() (string, error) {
		resp, err := rayClient.GetJobDetails(jobID)
		if err != nil {
			return rayJobStatus, err
		}
		rayJobStatusVal := resp.Status
		if rayJobStatusVal == "SUCCEEDED" || rayJobStatusVal == "FAILED" {
			test.T().Logf("JobStatus - %s\n", rayJobStatusVal)
			rayJobStatus = rayJobStatusVal
			return rayJobStatus, nil
		}
		if rayJobStatus != rayJobStatusVal && rayJobStatusVal != "SUCCEEDED" {
			test.T().Logf("JobStatus - %s...\n", rayJobStatusVal)
			rayJobStatus = rayJobStatusVal
		}
		return rayJobStatus, nil
	}, TestTimeoutDouble, 1*time.Second).Should(Or(Equal("SUCCEEDED"), Equal("FAILED")), "Job did not complete within the expected time")

	// Store job logs in output directory
	WriteRayJobAPILogs(test, rayClient, jobID)

	// Assert ray-job status after job execution
	test.Expect(rayJobStatus).To(Equal("SUCCEEDED"), "RayJob failed !")

	// Make sure the RayCluster finishes and is deleted
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutLong).
		Should(BeEmpty())
}

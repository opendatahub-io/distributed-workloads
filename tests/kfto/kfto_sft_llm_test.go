/*
Copyright 2024.

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

package kfto

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	"github.com/opendatahub-io/distributed-workloads/tests/odh"
)

func TestKftoSftLlmLlama3_1_8BInstructWithCudaPyTorch251(t *testing.T) {
	Tags(t, KftoCuda)
	kftoSftLlm(t, GetTrainingCudaPyTorch251Image(), NVIDIA, "meta-llama/Llama-3.1-8B-Instruct")
}

func TestKftoSftLlmLlama3_1_8BInstructWithROCmPyTorch251(t *testing.T) {
	Tags(t, KftoRocm)
	kftoSftLlm(t, GetTrainingROCmPyTorch251Image(), AMD, "meta-llama/Llama-3.1-8B-Instruct")
}

func kftoSftLlm(t *testing.T, image string, gpu Accelerator, modelName string) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()
	var workingDirectory, err = os.Getwd()
	test.Expect(err).ToNot(HaveOccurred())

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Get token for HuggingFace user
	hfToken := GetHuggingFaceToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Get storageclass that supports RWX PVC provisioning
	storageClass, err := GetRWXStorageClass(test)
	test.Expect(err).ToNot(HaveOccurred(), "Failed to find an RWX supporting StorageClass")

	// Create PVC for Notebook
	notebookPVC := CreatePersistentVolumeClaim(test, namespace.Name, "500Gi", AccessModes(corev1.ReadWriteMany), StorageClassName(storageClass.Name))

	// Create Kueue resources
	resourceFlavor := CreateKueueResourceFlavor(test, v1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := v1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []v1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory"), corev1.ResourceName(gpu.ResourceLabel)},
				Flavors: []v1beta1.FlavorQuotas{
					{
						Name: v1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []v1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("32"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("512Gi"),
							},
							{
								Name:         corev1.ResourceName(gpu.ResourceLabel),
								NominalQuota: resource.MustParse("8"),
							},
						},
					},
				},
			},
		},
	}

	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, namespace.Name, clusterQueue.Name, AsDefaultQueue)

	// Read and update the notebook content
	notebookContent := odh.ReadFileExt(test, workingDirectory+"/../../examples/kfto-sft-llm/sft.ipynb")
	updatedNotebookContent := string(notebookContent)

	// Update notebook parameters for testing
	requiredChangesInNotebook := map[string]string{
		"model_name_or_path: Meta-Llama/Meta-Llama-3.1-8B-Instruct": fmt.Sprintf("model_name_or_path: %s", modelName),
		"num_train_epochs: 10": "num_train_epochs: 1",
		"eval_strategy: epoch": "eval_strategy: 'no'",
		"logging_steps: 1":     "logging_steps: 10",
		"output_dir: /mnt/shared/Meta-Llama-3.1-8B-Instruct":              fmt.Sprintf("output_dir: /mnt/shared/%s", modelName),
		"api_server = \\\"<API_SERVER>\\\"":                               fmt.Sprintf("api_server = \\\"%s\\\"", GetOpenShiftApiUrl(test)),
		"token = \\\"<TOKEN>\\\"":                                         fmt.Sprintf("token = \\\"%s\\\"", userToken),
		"#configuration.verify_ssl = False":                               "configuration.verify_ssl = False",
		"name=\\\"sft\\\"":                                                fmt.Sprintf("name=\\\"sft-%s\\\"", namespace.Name),
		"train_func=main,":                                                fmt.Sprintf("labels= {\\n\",\n\t\"            \\\"kueue.x-k8s.io/queue-name\\\": \\\"%s\\\"\\n\",\n\t\"    },\\n\",\n\t\"    train_func=main,", localQueue.Name),
		"        \\\"nvidia.com/gpu\\\"":                                  fmt.Sprintf("        \\\"%s\\\"", gpu.ResourceLabel),
		"base_image=\\\"quay.io/modh/training:py311-cuda124-torch251\\\"": fmt.Sprintf("base_image=\\\"%s\\\"", image),
		"\"HF_TOKEN\\\": \\\"\\\"":                                        fmt.Sprintf("\"HF_TOKEN\\\": \\\"%s\\\"", hfToken),
		"claim_name=\\\"shared\\\"":                                       fmt.Sprintf("claim_name=\\\"%s\\\"", notebookPVC.Name),
		"\"client.get_job_logs(\\n\",":                                    "\"client.wait_for_job_conditions(\\n\",",
		"\"    follow=True,\\n\",":                                        "\"    wait_timeout=1800,\\n\",\n\t\"    polling_interval=60,\\n\",",
		"os.environ[\\\"TENSORBOARD_PROXY_URL\\\"]":                       "#os.environ[\\\"TENSORBOARD_PROXY_URL\\\"]",
		"%load_ext tensorboard":                                           "#%load_ext tensorboard",
		"%tensorboard --logdir /opt/app-root/src/shared":                  "#%tensorboard --logdir /opt/app-root/src/shared",
		"pretrained_path = \\\"/opt/app-root/src/shared/.cache/hub/models--Meta-Llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/\\\"": "pretrained_path = \\\"/opt/app-root/src/.cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/\\\"",
		"# Test the pre-trained model": "# Test the pre-trained model\\n\",\n\"from IPython.display import Markdown, display\\n\",\n\"import os",
		"display(Markdown(output1))":   "display(Markdown(output1))\\n\",\n\"\\n\",\n\"# Save to file\\n\",\n\"output_path = \\\"/opt/app-root/src/pretrained_output.md\\\"\\n\",\n\"os.makedirs(os.path.dirname(output_path), exist_ok=True)\\n\",\n\"with open(output_path, \\\"w\\\") as f:\\n\",\n\t\"    f.write(output1)",
		"finetuned_path = \\\"/opt/app-root/src/shared/Meta-Llama-3.1-8B-Instruct/checkpoint-300/\\\"": fmt.Sprintf("finetuned_path = \\\"/opt/app-root/src/%s/checkpoint-30/\\\"", modelName),
		"# Test the fine-tuned model": "# Test the fine-tuned model\\n\",\n\"from IPython.display import Markdown, display\\n\",\n\"import os",
		"display(Markdown(output2))":  "display(Markdown(output2))\\n\",\n\"\\n\",\n\"# Save to file\\n\",\n\"output_path = \\\"/opt/app-root/src/finetuned_output.md\\\"\\n\",\n\"os.makedirs(os.path.dirname(output_path), exist_ok=True)\\n\",\n\"with open(output_path, \\\"w\\\") as f:\\n\",\n\t\"    f.write(output2)",
		"client.delete_job":           "#client.delete_job",
		"import gc":                   "#import gc",
		"del base_model, model":       "#del base_model, model",
		"gc.collect()":                "#gc.collect()",
		"torch.cuda.empty_cache()":    "#torch.cuda.empty_cache()",
	}

	for oldValue, newValue := range requiredChangesInNotebook {
		t.Logf("Replacing '%s' with '%s'", oldValue, newValue)
		updatedNotebookContent = strings.Replace(updatedNotebookContent, oldValue, newValue, -1)
	}

	updatedNotebook := []byte(updatedNotebookContent)

	// Create ConfigMap with the notebook
	configMap := map[string][]byte{
		"sft.ipynb": updatedNotebook,
	}
	config := CreateConfigMap(test, namespace.Name, configMap)

	notebookCommand := []string{
		"/bin/sh",
		"-c",
		"set -e; " +
			"pip install --quiet --no-cache-dir papermill; " +
			"if papermill /opt/app-root/notebooks/sft.ipynb /opt/app-root/src/sft-out.ipynb --log-output; then " +
			"    echo 'SUCCESS' > /opt/app-root/src/notebook_completion_marker; " +
			"else " +
			"    echo 'FAILURE' > /opt/app-root/src/notebook_completion_marker; " +
			"fi; " +
			"sleep infinity",
	}

	// Create Notebook CR
	CreateNotebook(test, namespace, userToken, notebookCommand, config.Name, "sft.ipynb", 1, notebookPVC, ContainerSizeMedium, gpu.ResourceLabel)

	// Gracefully cleanup Notebook
	defer func() {
		DeleteNotebook(test, namespace)
		test.Eventually(ListNotebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	test.T().Logf("Wait for pytorch job to start running ...")
	// Make sure the PyTorch job is running
	test.Eventually(PyTorchJob(test, namespace.Name, fmt.Sprintf("sft-%s", namespace.Name)), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob sft-%s is running", namespace.Name)

	test.T().Logf("Wait for pytorch job to complete ...")
	// Make sure the PyTorch job succeeded
	test.Eventually(PyTorchJob(test, namespace.Name, fmt.Sprintf("sft-%s", namespace.Name)), TestTimeoutGpuProvisioning).
		Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob sft-%s is successfully completed", namespace.Name)

	test.T().Logf("Wait for notebook execution to complete ...")
	markerPath := "/opt/app-root/src/notebook_completion_marker"

	// Get pod and container names dynamically
	podList := GetPods(test, namespace.Name, metav1.ListOptions{
		LabelSelector: "notebook-name=jupyter-nb-kube-3aadmin",
		FieldSelector: "status.phase=Running",
	})
	test.Expect(podList).To(HaveLen(1), "Expected exactly one notebook pod")
	podName := podList[0].Name
	containerName := podList[0].Spec.Containers[0].Name

	var notebookExecutionFinalError error // This variable will store the definitive outcome of notebook execution
	test.Eventually(func() bool {
		cmd := exec.Command("kubectl", "-n", namespace.Name, "exec", podName,
			"-c", containerName, "--", "cat", markerPath)

		output, err := cmd.CombinedOutput()
		if err != nil {
			test.T().Logf("Attempt to read marker file '%s' failed: %v", markerPath, err)
			test.T().Logf("Command output: %s", string(output))
			return false
		}

		markerContent := strings.TrimSpace(string(output))
		if markerContent == "SUCCESS" {
			test.T().Logf("Notebook execution completed as indicated by marker file: '%s'", markerContent)
			return true
		} else if markerContent == "FAILURE" {
			errMessage := fmt.Sprintf("Notebook execution failed as indicated by marker file '%s': %s", markerPath, markerContent)
			test.T().Errorf("%s", errMessage)
			notebookExecutionFinalError = fmt.Errorf("%s", errMessage)
			// Return true to stop Eventually polling, because a *final* state has been reached
			return true
		}

		test.T().Logf("Marker file '%s' content not yet 'SUCCESS' or 'FAILURE': '%s'. Retrying...", markerPath, markerContent)
		return false
	}, TestTimeoutLong, 5*time.Second).Should(BeTrue(), "Notebook status never became definitive (SUCCESS or FAILURE) within timeout.")

	if notebookExecutionFinalError != nil {
		test.Expect(notebookExecutionFinalError).To(Succeed(), "Notebook execution explicitly reported FAILURE")
	}

	test.T().Logf("Notebook execution completed successfully")

	pretrainedPath := "/opt/app-root/src/pretrained_output.md"
	finetunedPath := "/opt/app-root/src/finetuned_output.md"

	readFileFromPod := func(path string) string {
		cmd := exec.Command("kubectl", "-n", namespace.Name, "exec", podName,
			"-c", containerName, "--", "cat", path)
		output, err := cmd.CombinedOutput()
		test.Expect(err).ToNot(HaveOccurred(), fmt.Sprintf("Failed to read file from notebook pod %s: %s", path, string(output)))
		return string(output)
	}

	// Read both model outputs
	pretrainedOutput := readFileFromPod(pretrainedPath)
	finetunedOutput := readFileFromPod(finetunedPath)

	t.Logf("Pretrained Output:\n%s", pretrainedOutput)
	t.Logf("Finetuned Output:\n%s", finetunedOutput)

	// Basic validation: outputs must differ
	test.Expect(pretrainedOutput).ToNot(Equal(finetunedOutput), "Expected outputs to differ between pretrained and fine-tuned model")

	// Validate final numeric answer == 18
	preFinal, err1 := extractFinalNumber(pretrainedOutput)
	fineFinal, err2 := extractFinalNumber(finetunedOutput)
	test.Expect(err1).ToNot(HaveOccurred(), "Pretrained model final number not found or invalid")
	test.Expect(err2).ToNot(HaveOccurred(), "Finetuned model final number not found or invalid")
	test.Expect(preFinal).To(Equal(18), "Pretrained model should return 18 as final numeric answer")
	test.Expect(fineFinal).To(Equal(18), "Finetuned model should return 18 as final numeric answer")

	t.Logf("Final numeric answers match: %d", preFinal)

	// Check math reasoning style in finetuned output
	if hasMathReasoningFormat(finetunedOutput) {
		t.Log("Finetuned model uses inline math reasoning (<<...>>)")
	} else {
		t.Log("Finetuned model does NOT use inline math reasoning format")
	}
}

func extractFinalNumber(text string) (int, error) {
	re := regexp.MustCompile(`\b\d+\b`)
	matches := re.FindAllString(text, -1)
	if len(matches) == 0 {
		return 0, fmt.Errorf("no numbers found in output")
	}
	return strconv.Atoi(matches[len(matches)-1])
}

func hasMathReasoningFormat(text string) bool {
	re := regexp.MustCompile(`<<.*?>>`)
	return re.MatchString(text)
}

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

package kfto

import (
	"fmt"
	"os"
	"os/exec"
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
)

func TestMnistSDKPyTorch241(t *testing.T) {
	Tags(t, Tier1)
	runMnistSDK(t, GetTrainingCudaPyTorch241Image())
}

func TestMnistSDKPyTorch251(t *testing.T) {
	Tags(t, Tier1)
	runMnistSDK(t, GetTrainingCudaPyTorch251Image())
}

func runMnistSDK(t *testing.T, trainingImage string) {
	test := With(t)
	// Create a namespace
	namespace := test.NewTestNamespace()
	userName := GetNotebookUserName(test)
	userToken := generateNotebookUserToken(test)

	jupyterNotebookConfigMapFileName := "mnist_kfto.ipynb"
	mnist := ParseAWSArgs(test, readFile(test, "resources/kfto_sdk_mnist.py"))

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Create Kueue resources
	resourceFlavor := CreateKueueResourceFlavor(test, v1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := v1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []v1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceCPU, corev1.ResourceMemory, corev1.ResourceName(NVIDIA.ResourceLabel)},
				Flavors: []v1beta1.FlavorQuotas{
					{
						Name: v1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []v1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("1"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("4Gi"),
							},
							{
								Name:         corev1.ResourceName(NVIDIA.ResourceLabel),
								NominalQuota: resource.MustParse(fmt.Sprint(0)),
							},
						},
					},
				},
			},
		},
	}

	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	CreateKueueLocalQueue(test, namespace.Name, clusterQueue.Name, AsDefaultQueue)

	jupyterNotebook := string(readFile(test, "resources/mnist_kfto.ipynb"))
	requirements := readFile(test, "resources/requirements.txt")

	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		jupyterNotebookConfigMapFileName: []byte(jupyterNotebook),
		"kfto_sdk_mnist.py":              mnist,
		"requirements.txt":               requirements,
	})

	notebookCommand := []string{
		"/bin/sh",
		"-c",
		fmt.Sprintf("pip install papermill && papermill /opt/app-root/notebooks/%s"+
			" /opt/app-root/src/mnist-kfto-out.ipynb -p namespace %s -p openshift_api_url %s"+
			" -p token %s -p num_gpus %d -p training_image %s --log-output && sleep infinity",
			jupyterNotebookConfigMapFileName, namespace.Name, GetOpenShiftApiUrl(test), userToken, 0, trainingImage)}

	// Create PVC for Notebook
	notebookPVC := CreatePersistentVolumeClaim(test, namespace.Name, "10Gi", AccessModes(corev1.ReadWriteOnce))

	// Create Notebook CR
	CreateNotebook(test, namespace, userToken, notebookCommand, config.Name, jupyterNotebookConfigMapFileName, 0, notebookPVC, ContainerSizeSmall)

	// Gracefully cleanup Notebook
	defer func() {
		DeleteNotebook(test, namespace)
		test.Eventually(ListNotebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Make sure pytorch job is created
	test.Eventually(PyTorchJob(test, namespace.Name, "pytorch-ddp"), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Make sure that the job eventually succeeds
	test.Eventually(PyTorchJob(test, namespace.Name, "pytorch-ddp"), TestTimeoutLong, 1*time.Second).
		Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))

	// TODO: write torch job logs?
	// time.Sleep(60 * time.Second)
}

func generateNotebookUserToken(test Test) string {
	userName := GetNotebookUserName(test)
	password := GetNotebookUserPassword(test)

	// Use own kobeconfig file to retrieve user token to keep it separated from main test credentials
	tempFile, err := os.CreateTemp("", "custom-kubeconfig-")
	test.Expect(err).NotTo(HaveOccurred())
	defer os.Remove(tempFile.Name())

	// Login by oc CLI using username and password
	cmd := exec.Command("oc", "login", "-u", userName, "-p", password, GetOpenShiftApiUrl(test), "--insecure-skip-tls-verify=true", "--kubeconfig="+tempFile.Name())
	out, err := cmd.Output()
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			test.T().Logf("Error running 'oc login' command: %v\n", exitError)
			test.T().Logf("Output: %s\n", out)
			test.T().Logf("Error output: %s\n", exitError.Stderr)
		} else {
			test.T().Logf("Error running 'oc login' command: %v\n", err)
		}
		test.T().FailNow()
	}

	// Use oc CLI to retrieve user token from kubeconfig
	cmd = exec.Command("oc", "whoami", "--show-token", "--kubeconfig="+tempFile.Name())
	out, err = cmd.Output()
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			test.T().Logf("Error running 'oc whoami' command: %v\n", exitError)
			test.T().Logf("Output: %s\n", out)
			test.T().Logf("Error output: %s\n", exitError.Stderr)
		} else {
			test.T().Logf("Error running 'oc whoami' command: %v\n", err)
		}
		test.T().FailNow()
	}

	return strings.TrimSpace(string(out))
}

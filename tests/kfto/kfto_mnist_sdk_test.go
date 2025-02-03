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
	"strings"
	"testing"

	. "github.com/onsi/gomega"
	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/project-codeflare/codeflare-common/support"

	v1 "k8s.io/api/core/v1"
)

func TestMnistSDK(t *testing.T) {
	test := With(t)
	// Create a namespace
	namespace := test.NewTestNamespace()
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)
	jupyterNotebookConfigMapFileName := "mnist_kfto.ipynb"
	mnist := ParseAWSArgs(test, readFile(test, "resources/kfto_sdk_mnist.py"))

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	requiredChangesInNotebook := map[string]string{
		"${api_url}":   GetOpenShiftApiUrl(test),
		"${password}":  userToken,
		"${num_gpus}":  "0",
		"${namespace}": namespace.Name,
	}

	jupyterNotebook := string(readFile(test, "resources/mnist_kfto.ipynb"))
	requirements := readFile(test, "resources/requirements.txt")
	for oldValue, newValue := range requiredChangesInNotebook {
		jupyterNotebook = strings.Replace(string(jupyterNotebook), oldValue, newValue, -1)
	}

	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		jupyterNotebookConfigMapFileName: []byte(jupyterNotebook),
		"kfto_sdk_mnist.py":              mnist,
		"requirements.txt":               requirements,
	})

	notebookCommand := []string{
		"bin/sh",
		"-c",
		"pip install papermill && papermill /opt/app-root/notebooks/{{.NotebookConfigMapFileName}}" +
			" /opt/app-root/src/mcad-out.ipynb -p namespace {{.Namespace}} -p openshift_api_url {{.OpenShiftApiUrl}}" +
			" -p kubernetes_user_bearer_token {{.KubernetesUserBearerToken}}" +
			" -p num_gpus {{ .NumGpus }} --log-output && sleep infinity",
	}
	// Create Notebook CR
	CreateNotebook(test, namespace, userToken, notebookCommand, config.Name, jupyterNotebookConfigMapFileName, 0)

	// Gracefully cleanup Notebook
	defer func() {
		DeleteNotebook(test, namespace)
		test.Eventually(ListNotebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Make sure pytorch job is created
	test.Eventually(PyTorchJob(test, namespace.Name, "pytorch-ddp")).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(v1.ConditionTrue)))

	// Make sure that the job eventually succeeds
	test.Eventually(PyTorchJob(test, namespace.Name, "pytorch-ddp")).
		Should(WithTransform(PyTorchJobConditionSucceeded, Equal(v1.ConditionTrue)))

	// TODO: write torch job logs?
	// time.Sleep(60 * time.Second)
}

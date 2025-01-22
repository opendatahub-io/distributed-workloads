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
	"time"

	. "github.com/onsi/gomega"
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
	mnist := readMnistScriptTemplate(test, "resources/kfto_sdk_train.py")

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	requiredChangesInNotebook := map[string]string{
		"${api_url}":        GetOpenShiftApiUrl(test),
		"${train_function}": "train_func_2",
		"${password}":       userToken,
		"${num_gpus}":       "2",
		"${namespace}":      namespace.Name,
	}

	jupyterNotebook := string(ReadFile(test, "resources/mnist_kfto.ipynb"))
	for oldValue, newValue := range requiredChangesInNotebook {
		jupyterNotebook = strings.Replace(string(jupyterNotebook), oldValue, newValue, -1)
	}

	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		jupyterNotebookConfigMapFileName: []byte(jupyterNotebook),
		"kfto_sdk_mnist.py":              mnist,
	})

	// Create Notebook CR
	createNotebook(test, namespace, userToken, config.Name, jupyterNotebookConfigMapFileName, 0)

	// Gracefully cleanup Notebook
	defer func() {
		deleteNotebook(test, namespace)
		test.Eventually(listNotebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Make sure pytorch job is created
	Eventually(PyTorchJob(test, namespace.Name, "pytorch-ddp")).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(v1.ConditionTrue)))

	// Make sure that the job eventually succeeds
	Eventually(PyTorchJob(test, namespace.Name, "pytorch-ddp")).
		Should(WithTransform(PyTorchJobConditionSucceeded, Equal(v1.ConditionTrue)))

	// TODO: write torch job logs?
	time.Sleep(60 * time.Second)
}

func readMnistScriptTemplate(test Test, filePath string) []byte {
	template, err := files.ReadFile(filePath)
	test.Expect(err).NotTo(HaveOccurred())

	props := struct{}{}

	return ParseTemplate(test, template, props)
}

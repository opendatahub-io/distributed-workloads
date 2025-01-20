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
	"testing"
	"time"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
)

func TestMnistSDK(t *testing.T) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	jupyterNotebookConfigMapFileName := "mnist_kfto.ipynb"
	mnist := readMnistScriptTemplate(test, "resources/mnist.py")

	jupyterNotebook := ReadFile(test, "resources/mnist_kfto.ipynb")
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		jupyterNotebookConfigMapFileName: jupyterNotebook,
		"mnist.py":                       mnist,
	})

	// Define the regular(non-admin) user
	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	// Create Notebook CR
	createNotebook(test, namespace, userToken, config.Name, jupyterNotebookConfigMapFileName, 0)
	time.Sleep(60 * time.Second)
}

func readMnistScriptTemplate(test Test, filePath string) []byte {
	template, err := files.ReadFile(filePath)
	test.Expect(err).NotTo(HaveOccurred())

	props := struct{}{}

	return ParseTemplate(test, template, props)
}

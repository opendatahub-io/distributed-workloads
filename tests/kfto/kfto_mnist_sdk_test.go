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
	mnist := readMnistScriptTemplate(test, "resources/kfto_sdk_mnist.py")

	// Create role binding with Namespace specific admin cluster role
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	requiredChangesInNotebook := map[string]string{
		"${api_url}":   GetOpenShiftApiUrl(test),
		"${password}":  userToken,
		"${num_gpus}":  "0",
		"${namespace}": namespace.Name,
	}

	jupyterNotebook := string(ReadFile(test, "resources/mnist_kfto.ipynb"))
	requirements := ReadFile(test, "resources/requirements.txt")
	for oldValue, newValue := range requiredChangesInNotebook {
		jupyterNotebook = strings.Replace(string(jupyterNotebook), oldValue, newValue, -1)
	}

	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		jupyterNotebookConfigMapFileName: []byte(jupyterNotebook),
		"kfto_sdk_mnist.py":              mnist,
		"requirements.txt":               requirements,
	})

	// Create Notebook CR
	createNotebook(test, namespace, userToken, config.Name, jupyterNotebookConfigMapFileName, 0)

	// Gracefully cleanup Notebook
	defer func() {
		deleteNotebook(test, namespace)
		test.Eventually(listNotebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
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

func readMnistScriptTemplate(test Test, filePath string) []byte {
	// Read the mnist.py from resources and perform replacements for custom values using go template
	storage_bucket_endpoint, storage_bucket_endpoint_exists := GetStorageBucketDefaultEndpoint()
	storage_bucket_access_key_id, storage_bucket_access_key_id_exists := GetStorageBucketAccessKeyId()
	storage_bucket_secret_key, storage_bucket_secret_key_exists := GetStorageBucketSecretKey()
	storage_bucket_name, storage_bucket_name_exists := GetStorageBucketName()
	storage_bucket_mnist_dir, storage_bucket_mnist_dir_exists := GetStorageBucketMnistDir()

	props := struct {
		StorageBucketDefaultEndpoint       string
		StorageBucketDefaultEndpointExists bool
		StorageBucketAccessKeyId           string
		StorageBucketAccessKeyIdExists     bool
		StorageBucketSecretKey             string
		StorageBucketSecretKeyExists       bool
		StorageBucketName                  string
		StorageBucketNameExists            bool
		StorageBucketMnistDir              string
		StorageBucketMnistDirExists        bool
	}{
		StorageBucketDefaultEndpoint:       storage_bucket_endpoint,
		StorageBucketDefaultEndpointExists: storage_bucket_endpoint_exists,
		StorageBucketAccessKeyId:           storage_bucket_access_key_id,
		StorageBucketAccessKeyIdExists:     storage_bucket_access_key_id_exists,
		StorageBucketSecretKey:             storage_bucket_secret_key,
		StorageBucketSecretKeyExists:       storage_bucket_secret_key_exists,
		StorageBucketName:                  storage_bucket_name,
		StorageBucketNameExists:            storage_bucket_name_exists,
		StorageBucketMnistDir:              storage_bucket_mnist_dir,
		StorageBucketMnistDirExists:        storage_bucket_mnist_dir_exists,
	}
	template, err := files.ReadFile(filePath)
	test.Expect(err).NotTo(HaveOccurred())

	return ParseTemplate(test, template, props)
}

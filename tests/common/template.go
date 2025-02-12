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

package common

import (
	"bytes"
	"text/template"

	"github.com/onsi/gomega"
	"github.com/project-codeflare/codeflare-common/support"
)

func ParseAWSArgs(t support.Test, inputTemplate []byte) []byte {
	storage_bucket_endpoint, storage_bucket_endpoint_exists := support.GetStorageBucketDefaultEndpoint()
	storage_bucket_access_key_id, storage_bucket_access_key_id_exists := support.GetStorageBucketAccessKeyId()
	storage_bucket_secret_key, storage_bucket_secret_key_exists := support.GetStorageBucketSecretKey()
	storage_bucket_name, storage_bucket_name_exists := support.GetStorageBucketName()
	storage_bucket_mnist_dir, storage_bucket_mnist_dir_exists := support.GetStorageBucketMnistDir()

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

	return ParseTemplate(t, inputTemplate, props)
}

func ParseTemplate(t support.Test, inputTemplate []byte, props interface{}) []byte {
	t.T().Helper()

	// Parse input template
	parsedTemplate, err := template.New("template").Parse(string(inputTemplate))
	t.Expect(err).NotTo(gomega.HaveOccurred())

	// Filter template and store results to the buffer
	buffer := new(bytes.Buffer)
	err = parsedTemplate.Execute(buffer, props)
	t.Expect(err).NotTo(gomega.HaveOccurred())

	return buffer.Bytes()
}

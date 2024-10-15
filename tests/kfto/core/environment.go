/*
Copyright 2024

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

package core

import (
	"fmt"
	"os"

	. "github.com/project-codeflare/codeflare-common/support"
)

const (
	// The environment variable for FMS HF Tuning image to be tested
	fmsHfTuningImageEnvVar = "FMS_HF_TUNING_IMAGE"
	// The environment variable referring to image containing bloom-560m model
	bloomModelImageEnvVar = "BLOOM_MODEL_IMAGE"
	// The environment variable referring to image containing Stanford Alpaca dataset
	alpacaDatasetImageEnvVar = "ALPACA_DATASET_IMAGE"
	// The environment variable referring to image containing minio CLI
	minioCliImageEnvVar = "MINIO_CLI_IMAGE"
	// The environment variable for HuggingFace token to download models which require authentication
	huggingfaceTokenEnvVar = "HF_TOKEN"
	// The environment variable specifying existing namespace name to be used for tests
	testNamespaceEnvVar = "TEST_NAMESPACE_NAME"
	// The environment variable specifying name of PersistenceVolumeClaim containing GPTQ models
	gptqModelPvcNameEnvVar = "GPTQ_MODEL_PVC_NAME"
	// The environment variable referring to image simulating sleep condition in container
	sleepImageEnvVar = "SLEEP_IMAGE"
	// The environment variable specifying s3 bucket folder path used to store model
	storageBucketModelPath = "AWS_STORAGE_BUCKET_MODEL_PATH"
)

func GetFmsHfTuningImage(t Test) string {
	image, ok := os.LookupEnv(fmsHfTuningImageEnvVar)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify fms-hf-tuning image to be tested.", fmsHfTuningImageEnvVar)
	}
	return image
}

func GetBloomModelImage() string {
	return lookupEnvOrDefault(bloomModelImageEnvVar, "quay.io/ksuta/bloom-560m@sha256:f6db02bb7b5d09a8d698c04994d747bfb9e581bbb4c07d00290244d207623733")
}

func GetAlpacaDatasetImage() string {
	return lookupEnvOrDefault(alpacaDatasetImageEnvVar, "quay.io/ksuta/alpaca-dataset@sha256:2e90f631180c7b2c916f9569b914b336b612e8ae86efad82546adc5c9fcbbb8d")
}

func GetMinioCliImage() string {
	return lookupEnvOrDefault(minioCliImageEnvVar, "quay.io/ksuta/mc@sha256:e128ce4caee276bcbfe3bd32ebb01c814f6b2eb2fd52d08ef0d4684f68c1e3d6")
}

func GetHuggingFaceToken(t Test) string {
	image, ok := os.LookupEnv(huggingfaceTokenEnvVar)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify HuggingFace token to download models.", huggingfaceTokenEnvVar)
	}
	return image
}

func GetTestNamespaceName() (namespaceName string, exists bool) {
	return os.LookupEnv(testNamespaceEnvVar)
}

func GetGptqModelPvcName() (string, error) {
	image, ok := os.LookupEnv(gptqModelPvcNameEnvVar)
	if !ok {
		return "", fmt.Errorf("expected environment variable %s not found, please use this environment variable to specify name of PersistenceVolumeClaim containing GPTQ models", gptqModelPvcNameEnvVar)
	}
	return image, nil
}

func GetSleepImage() string {
	return lookupEnvOrDefault(sleepImageEnvVar, "gcr.io/k8s-staging-perf-tests/sleep@sha256:8d91ddf9f145b66475efda1a1b52269be542292891b5de2a7fad944052bab6ea")
}

func GetStorageBucketModelPath() string {
	storageBucketModelPath := lookupEnvOrDefault(storageBucketModelPath, "")
	return storageBucketModelPath
}

func lookupEnvOrDefault(key, value string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return value
}

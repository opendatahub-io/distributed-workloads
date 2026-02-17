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

package support

import (
	"os"
)

const (
	// The environment variables hereafter can be used to change the components
	// used for testing.

	TestRayVersion                  = "TEST_RAY_VERSION"
	TestRayImage                    = "TEST_RAY_IMAGE"
	TestPyTorchImage                = "TEST_PYTORCH_IMAGE"
	TestTrainingCudaPyTorch241Image = "TEST_TRAINING_CUDA_PYTORCH_241_IMAGE"
	TestTrainingCudaPyTorch251Image = "TEST_TRAINING_CUDA_PYTORCH_251_IMAGE"
	TestTrainingCudaPyTorch28Image  = "TEST_TRAINING_CUDA_PYTORCH_28_IMAGE"
	TestTrainingRocmPyTorch241Image = "TEST_TRAINING_ROCM_PYTORCH_241_IMAGE"
	TestTrainingRocmPyTorch251Image = "TEST_TRAINING_ROCM_PYTORCH_251_IMAGE"
	TestTrainingRocmPyTorch28Image  = "TEST_TRAINING_ROCM_PYTORCH_28_IMAGE"

	// The testing output directory, to write output files into.
	TestOutputDir = "TEST_OUTPUT_DIR"

	// Type of cluster test is run on
	ClusterTypeEnvVar = "CLUSTER_TYPE"

	// Hostname of the Kubernetes cluster
	ClusterHostname = "CLUSTER_HOSTNAME"

	// URL for downloading MNIST dataset
	mnistDatasetURL = "MNIST_DATASET_URL"

	// URL for PiPI index containing all the required test Python packages
	pipIndexURL    = "PIP_INDEX_URL"
	pipTrustedHost = "PIP_TRUSTED_HOST"

	// Storage bucket credentials
	storageDefaultEndpoint       = "AWS_DEFAULT_ENDPOINT"
	storageDefaultRegion         = "AWS_DEFAULT_REGION"
	storageAccessKeyId           = "AWS_ACCESS_KEY_ID"
	storageSecretKey             = "AWS_SECRET_ACCESS_KEY"
	storageBucketName            = "AWS_STORAGE_BUCKET"
	storageBucketMnistDir        = "AWS_STORAGE_BUCKET_MNIST_DIR"
	storageBucketFashionMnistDir = "AWS_STORAGE_BUCKET_FASHION_MNIST_DIR"
	storageBucketOsftDir         = "AWS_STORAGE_BUCKET_OSFT_DIR"
	storageBucketSftDir          = "AWS_STORAGE_BUCKET_SFT_DIR"
	storageBucketLoraDir         = "AWS_STORAGE_BUCKET_LORA_DIR"

	// Name of existing namespace to be used for test
	testNamespaceNameEnvVar = "TEST_NAMESPACE_NAME"
)

type ClusterType string

const (
	OsdCluster        ClusterType = "OSD"
	OcpCluster        ClusterType = "OCP"
	HypershiftCluster ClusterType = "HYPERSHIFT"
	KindCluster       ClusterType = "KIND"
	UndefinedCluster  ClusterType = "UNDEFINED"
)

func GetRayVersion() string {
	return lookupEnvOrDefault(TestRayVersion, RayVersion)
}

func GetRayImage() string {
	return lookupEnvOrDefault(TestRayImage, RayImage)
}

func GetRayROCmImage() string {
	return lookupEnvOrDefault(TestRayImage, RayROCmImage)
}

func GetRayTorchCudaImage() string {
	return lookupEnvOrDefault(TestRayImage, RayTorchCudaImage)
}

func GetRayTorchROCmImage() string {
	return lookupEnvOrDefault(TestRayImage, RayTorchROCmImage)
}

func GetTrainingCudaPyTorch241Image() string {
	return lookupEnvOrDefault(TestTrainingCudaPyTorch241Image, TrainingCudaPyTorch241Image)
}

func GetTrainingCudaPyTorch251Image() string {
	return lookupEnvOrDefault(TestTrainingCudaPyTorch251Image, TrainingCudaPyTorch251Image)
}

func GetTrainingCudaPyTorch28Image() string {
	return lookupEnvOrDefault(TestTrainingCudaPyTorch28Image, TrainingCudaPyTorch28Image)
}

func GetTrainingROCmPyTorch241Image() string {
	return lookupEnvOrDefault(TestTrainingRocmPyTorch241Image, TrainingRocmPyTorch241Image)
}

func GetTrainingROCmPyTorch251Image() string {
	return lookupEnvOrDefault(TestTrainingRocmPyTorch251Image, TrainingRocmPyTorch251Image)
}

func GetTrainingRocmPyTorch28Image() string {
	return lookupEnvOrDefault(TestTrainingRocmPyTorch28Image, TrainingRocmPyTorch28Image)
}

func GetClusterType(t Test) ClusterType {
	clusterType, ok := os.LookupEnv(ClusterTypeEnvVar)
	if !ok {
		t.T().Logf("Environment variable %s is unset.", ClusterTypeEnvVar)
		return UndefinedCluster
	}
	switch clusterType {
	case "OSD":
		return OsdCluster
	case "OCP":
		return OcpCluster
	case "HYPERSHIFT":
		return HypershiftCluster
	case "KIND":
		return KindCluster
	default:
		t.T().Logf("Environment variable %s is unset or contains an incorrect value: '%s'", ClusterTypeEnvVar, clusterType)
		return UndefinedCluster
	}
}

func GetClusterHostname(t Test) string {
	hostname, ok := os.LookupEnv(ClusterHostname)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please define cluster hostname.", ClusterHostname)
	}
	return hostname
}

func GetMnistDatasetURL() string {
	return lookupEnvOrDefault(mnistDatasetURL, "https://ossci-datasets.s3.amazonaws.com/mnist/")
}

func GetStorageBucketDefaultEndpoint() (string, bool) {
	storage_endpoint, exists := os.LookupEnv(storageDefaultEndpoint)
	return storage_endpoint, exists
}

func GetStorageBucketDefaultRegion() (string, bool) {
	storage_default_region, exists := os.LookupEnv(storageDefaultRegion)
	return storage_default_region, exists
}

func GetStorageBucketAccessKeyId() (string, bool) {
	storage_access_key_id, exists := os.LookupEnv(storageAccessKeyId)
	return storage_access_key_id, exists
}

func GetStorageBucketSecretKey() (string, bool) {
	storage_secret_key, exists := os.LookupEnv(storageSecretKey)
	return storage_secret_key, exists
}

func GetStorageBucketName() (string, bool) {
	storage_bucket_name, exists := os.LookupEnv(storageBucketName)
	return storage_bucket_name, exists
}

func GetStorageBucketMnistDir() (string, bool) {
	storage_bucket_mnist_dir, exists := os.LookupEnv(storageBucketMnistDir)
	return storage_bucket_mnist_dir, exists
}

func GetStorageBucketFashionMnistDir() (string, bool) {
	storage_bucket_fashion_mnist_dir, exists := os.LookupEnv(storageBucketFashionMnistDir)
	return storage_bucket_fashion_mnist_dir, exists
}

func GetStorageBucketOsftDir() (string, bool) {
	storage_bucket_osft_dir, exists := os.LookupEnv(storageBucketOsftDir)
	return storage_bucket_osft_dir, exists
}

func GetStorageBucketLoraDir() (string, bool) {
	storage_bucket_lora_dir, exists := os.LookupEnv(storageBucketLoraDir)
	return storage_bucket_lora_dir, exists
}

func GetStorageBucketSftDir() (string, bool) {
	storage_bucket_sft_dir, exists := os.LookupEnv(storageBucketSftDir)
	return storage_bucket_sft_dir, exists
}

func GetPipIndexURL() string {
	return lookupEnvOrDefault(pipIndexURL, "https://pypi.python.org/simple")
}

func GetPipTrustedHost() string {
	return lookupEnvOrDefault(pipTrustedHost, "")
}

func GetTestNamespaceName() (string, bool) {
	return os.LookupEnv(testNamespaceNameEnvVar)
}

func lookupEnvOrDefault(key, value string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return value
}

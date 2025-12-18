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

package trainer

import (
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

type ClusterTrainingRuntime struct {
	Name       string
	ODHImage   string
	RHOAIImage string
}

var expectedRuntimes = []ClusterTrainingRuntime{
	{Name: "torch-distributed", ODHImage: "training:py312-cuda128-torch280", RHOAIImage: "odh-training-cuda128-torch28-py312-rhel9"},
	{Name: "torch-distributed-rocm", ODHImage: "training:py312-rocm64-torch280", RHOAIImage: "odh-training-rocm64-torch28-py312-rhel9"},
	{Name: "torch-distributed-th03-cuda128-torch28-py312", ODHImage: "training:py312-cuda128-torch280", RHOAIImage: "odh-training-cuda128-torch28-py312-rhel9"},
	{Name: "training-hub", ODHImage: "training:py312-cuda128-torch280", RHOAIImage: "odh-training-cuda128-torch28-py312-rhel9"},
	{Name: "training-hub03-cuda128-torch28-py312", ODHImage: "training:py312-cuda128-torch280", RHOAIImage: "odh-training-cuda128-torch28-py312-rhel9"},
}

// defaultClusterTrainingRuntime is used across integration tests
var defaultClusterTrainingRuntime = expectedRuntimes[0].Name

func TestDefaultClusterTrainingRuntimes(t *testing.T) {
	Tags(t, Smoke)
	test := With(t)

	// Determine registry based on ODH namespace
	namespace, err := GetApplicationsNamespaceFromDSCI(test, DefaultDSCIName)
	test.Expect(err).NotTo(HaveOccurred())
	registryName := GetExpectedRegistry(test, namespace)

	// Build a map of expected runtimes for quick lookup
	expectedRuntimeMap := make(map[string]ClusterTrainingRuntime)
	for _, runtime := range expectedRuntimes {
		expectedRuntimeMap[runtime.Name] = runtime
	}

	// List all ClusterTrainingRuntimes from the cluster
	runtimeList, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().List(
		test.Ctx(),
		metav1.ListOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to list ClusterTrainingRuntimes")

	// Track unexpected runtimes and found expected runtimes
	var unexpectedRuntimes []string
	foundRuntimes := make(map[string]bool)

	// Iterate over runtimes present in the cluster
	for _, runtime := range runtimeList.Items {
		expectedRuntime, found := expectedRuntimeMap[runtime.Name]
		if !found {
			unexpectedRuntimes = append(unexpectedRuntimes, runtime.Name)
			test.T().Logf("WARNING: Unexpected ClusterTrainingRuntime '%s' found", runtime.Name)
			continue
		}

		foundRuntimes[runtime.Name] = true
		test.T().Logf("ClusterTrainingRuntime '%s' is present", runtime.Name)

		// Find container image from the runtime spec
		var foundImage string
		for _, replicatedJob := range runtime.Spec.Template.Spec.ReplicatedJobs {
			for _, container := range replicatedJob.Template.Spec.Template.Spec.Containers {
				if container.Image != "" {
					foundImage = container.Image
					break
				}
			}
			if foundImage != "" {
				break
			}
		}

		test.Expect(foundImage).NotTo(BeEmpty(), "No container image found in ClusterTrainingRuntime %s", runtime.Name)
		test.T().Logf("Image referred in ClusterTrainingRuntime is %s", foundImage)

		// Verify image based on environment
		var expectedImage string
		switch registryName {
		case "registry.redhat.io":
			expectedImage = registryName + "/rhoai/" + expectedRuntime.RHOAIImage
		case "quay.io":
			expectedImage = registryName + "/modh/" + expectedRuntime.ODHImage
		default:
			test.T().Fatalf("Unexpected registry: %s", registryName)
		}

		test.Expect(foundImage).To(ContainSubstring(expectedImage),
			"Image %s should contain %s", foundImage, expectedImage)
		test.T().Logf("ClusterTrainingRuntime '%s' uses expected image: %s", expectedRuntime.Name, expectedImage)
	}

	// Verify all expected runtimes are present
	var missingRuntimes []string
	for _, expected := range expectedRuntimes {
		if !foundRuntimes[expected.Name] {
			missingRuntimes = append(missingRuntimes, expected.Name)
		}
	}

	// Fail if any unexpected runtimes found
	test.Expect(unexpectedRuntimes).To(BeEmpty(),
		"Unexpected ClusterTrainingRuntimes found: %v. Please update expectedRuntimes list.", unexpectedRuntimes)

	// Fail if any expected runtimes missing
	test.Expect(missingRuntimes).To(BeEmpty(),
		"Missing expected ClusterTrainingRuntimes: %v. These runtimes should be present on the cluster.", missingRuntimes)

	test.T().Log("All ClusterTrainingRuntimes verified successfully!")
}

func TestRunTrainJobWithDefaultClusterTrainingRuntimes(t *testing.T) {
	Tags(t, Sanity)
	test := With(t)

	for _, runtime := range expectedRuntimes {
		test.T().Logf("Running TrainJob with ClusterTrainingRuntime: %s", runtime.Name)

		// Create a namespace
		namespace := test.NewTestNamespace().Name

		// Create TrainJob
		trainJob := createTrainJob(test, namespace, runtime.Name)

		// Wait for TrainJob completion
		test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutLong).
			Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))

		test.T().Logf("TrainJob with ClusterTrainingRuntime '%s' completed successfully", runtime.Name)
	}

	test.T().Log("All TrainJobs with expected ClusterTrainingRuntimes completed successfully !!!")
}

func createTrainJob(test Test, namespace, runtimeName string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-trainjob-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: runtimeName,
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command: []string{
					"python",
					"-c",
					"import torch; print(f'PyTorch version: {torch.__version__}'); print('Training completed successfully')",
				},
			},
		},
	}

	createTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create TrainJob")
	test.T().Logf("Created TrainJob %s/%s successfully", createTrainJob.Namespace, createTrainJob.Name)

	return createTrainJob
}

func deleteTrainJob(test Test, namespace, name string) {
	test.T().Helper()

	err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Delete(
		test.Ctx(),
		name,
		metav1.DeleteOptions{},
	)
	if err != nil {
		test.T().Logf("Warning: Failed to delete TrainJob %s/%s: %v", namespace, name, err)
	} else {
		test.T().Logf("Deleted TrainJob %s/%s successfully", namespace, name)
	}
}

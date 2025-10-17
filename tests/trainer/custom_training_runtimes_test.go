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
	"fmt"
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

type ClusterTrainingRuntime struct {
	Name string
}

var expectedRuntimes = []ClusterTrainingRuntime{
	{Name: "torch-cuda-241"},
	{Name: "torch-cuda-251"},
	{Name: "torch-rocm-241"},
	{Name: "torch-rocm-251"},
}

func TestCustomTrainingRuntimesAvailable(t *testing.T) {
	Tags(t, Smoke)
	test := With(t)

	// List all ClusterTrainingRuntimes
	runtimeList, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().List(
		test.Ctx(),
		metav1.ListOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to list ClusterTrainingRuntimes")

	// Extract runtime names
	foundRuntimeNames := make(map[string]bool)
	test.T().Logf("Found %d ClusterTrainingRuntimes:", len(runtimeList.Items))
	for _, runtime := range runtimeList.Items {
		test.T().Logf("  - %s", runtime.Name)
		foundRuntimeNames[runtime.Name] = true
	}

	// Verify that the expected runtimes are present
	for _, expectedName := range expectedRuntimes {
		test.Expect(foundRuntimeNames[expectedName.Name]).To(BeTrue(),
			fmt.Sprintf("Expected ClusterTrainingRuntime '%s' not found", expectedName.Name))
		test.T().Logf("Found expected runtime: %s", expectedName.Name)
	}

	test.T().Log("All expected ClusterTrainingRuntimes are available")
}

func TestTrainJobWithTorchCuda241(t *testing.T) {
	Tags(t, Sanity)
	runTrainJobWithRuntime(t, "torch-cuda-241")
}

func TestTrainJobWithTorchCuda251(t *testing.T) {
	Tags(t, Sanity)
	runTrainJobWithRuntime(t, "torch-cuda-251")
}

func TestTrainJobWithTorchRocm241(t *testing.T) {
	Tags(t, Sanity)
	runTrainJobWithRuntime(t, "torch-rocm-241")
}

func TestTrainJobWithTorchRocm251(t *testing.T) {
	Tags(t, Sanity)
	runTrainJobWithRuntime(t, "torch-rocm-251")
}

func runTrainJobWithRuntime(t *testing.T, runtimeName string) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace().Name

	// Create TrainJob
	trainJob := createTrainJob(test, namespace, runtimeName)
	defer deleteTrainJob(test, namespace, trainJob.Name)

	// Wait for TrainJob completion
	test.T().Log("Waiting for TrainJob to complete...")
	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))

	test.T().Logf(" TrainJob with %s completed successfully", runtimeName)
}

func createTrainJob(test Test, namespace, runtimeName string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("test-trainjob-%s-", runtimeName),
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

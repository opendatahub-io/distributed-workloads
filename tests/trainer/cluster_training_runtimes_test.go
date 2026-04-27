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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

func TestDefaultClusterTrainingRuntimes(t *testing.T) {
	Tags(t, Smoke)
	test := With(t)

	imagePrefix := GetExpectedImagePrefix(test)

	// Build a map of expected runtimes for quick lookup
	expectedRuntimeMap := make(map[string]trainerutils.ClusterTrainingRuntime)
	for _, runtime := range trainerutils.ExpectedRuntimes {
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

		if trainerutils.IsMPIRuntime(runtime.Name) {
			test.Expect(foundImage).To(HavePrefix("quay.io/"),
				"MPI image %s should originate from quay.io", foundImage)
			test.Expect(foundImage).To(ContainSubstring(expectedRuntime.Image),
				"MPI image %s should contain %s", foundImage, expectedRuntime.Image)
			test.T().Logf("MPI ClusterTrainingRuntime '%s' uses expected image: %s", expectedRuntime.Name, foundImage)
		} else {
			expectedImage := imagePrefix + "/" + expectedRuntime.Image
			test.Expect(foundImage).To(ContainSubstring(expectedImage),
				"Image %s should contain %s", foundImage, expectedImage)
			test.T().Logf("ClusterTrainingRuntime '%s' uses expected image: %s", expectedRuntime.Name, expectedImage)
		}
	}

	// Verify all expected runtimes are present
	var missingRuntimes []string
	for _, expected := range trainerutils.ExpectedRuntimes {
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

func TestOpenMPICudaClusterTrainingRuntime(t *testing.T) {
	Tags(t, Smoke)
	test := With(t)

	runtime, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Get(
		test.Ctx(),
		trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA,
		metav1.GetOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to get ClusterTrainingRuntime %s", trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA)

	foundImage, err := trainerutils.GetImageFromClusterTrainingRuntime(test, trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA)
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(foundImage).NotTo(BeEmpty(), "No container image found in ClusterTrainingRuntime %s", runtime.Name)
	test.Expect(foundImage).To(ContainSubstring(trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDAImage),
		"Image %s should contain %s", foundImage, trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDAImage)
	test.Expect(runtime.Labels).To(HaveKeyWithValue("trainer.kubeflow.org/framework", "openmpi"))
	test.T().Logf("ClusterTrainingRuntime '%s' is present, uses image %s, and is labeled for framework openmpi", runtime.Name, foundImage)
}

// TestDefaultTrainingHubRuntimesMatchDefaultClusterRuntimes is a smoke test that verifies
// Training Hub and pinned torch-distributed CTR resources (th06) have exactly the same
// spec as their corresponding DefaultClusterTrainingRuntime resources.
func TestDefaultTrainingHubRuntimesMatchDefaultClusterRuntimes(t *testing.T) {
	Tags(t, Smoke)
	test := With(t)

	runtimeList, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().List(
		test.Ctx(),
		metav1.ListOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to list ClusterTrainingRuntimes")

	runtimesByName := make(map[string]trainerv1alpha1.ClusterTrainingRuntime)
	for _, r := range runtimeList.Items {
		runtimesByName[r.Name] = r
	}

	for runtimeName, defaultClusterName := range trainerutils.TrainingHubToDefaultClusterRuntime {
		runtime, ok := runtimesByName[runtimeName]
		test.Expect(ok).To(BeTrue(), "CTR %q not found on cluster", runtimeName)

		defaultCluster, ok := runtimesByName[defaultClusterName]
		test.Expect(ok).To(BeTrue(), "DefaultClusterTrainingRuntime %q not found on cluster", defaultClusterName)

		// Compare specs - they must be identical (only metadata differs)
		if !equality.Semantic.DeepEqual(runtime.Spec, defaultCluster.Spec) {
			test.T().Errorf("CTR %q spec differs from DefaultClusterTrainingRuntime %q. "+
				"Training Hub and pinned runtimes must have identical CTR specs to their defaults.",
				runtimeName, defaultClusterName)
		}
		test.T().Logf("CTR %q matches DefaultClusterTrainingRuntime %q", runtimeName, defaultClusterName)
	}

	test.T().Log("All CTRs match their DefaultClusterTrainingRuntime counterparts!")
}

func TestRunTrainJobWithDefaultClusterTrainingRuntimes(t *testing.T) {
	Tags(t, Sanity)
	test := With(t)

	// Run one TrainJob per unique image to avoid redundant runs for CTRs that share the same image
	tested := make(map[string]bool)
	for _, runtime := range trainerutils.ExpectedRuntimes {
		if trainerutils.IsMPIRuntime(runtime.Name) {
			test.T().Logf("Skipping MPI runtime '%s' (covered by TestMultiNodeOpenMPITrainJob)", runtime.Name)
			continue
		}
		if tested[runtime.Image] {
			test.T().Logf("Skipping ClusterTrainingRuntime '%s' (image '%s' already tested)", runtime.Name, runtime.Image)
			continue
		}
		tested[runtime.Image] = true

		test.T().Logf("Running TrainJob with ClusterTrainingRuntime: %s", runtime.Name)

		// Create a namespace
		namespace := test.NewTestNamespace().Name

		// Create TrainJob
		trainJob := createTrainJob(test, namespace, runtime.Name)

		// Wait for TrainJob completion
		test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutDouble).
			Should(Satisfy(TrainJobReachedFinalState))

		finalJob := TrainJob(test, namespace, trainJob.Name)(test)
		test.Expect(finalJob).To(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)),
			"TrainJob %s/%s should be complete; TrainJobFailed message: %s",
			namespace, trainJob.Name, TrainJobFailedMessage(finalJob))

		test.T().Logf("TrainJob with ClusterTrainingRuntime '%s' completed successfully", runtime.Name)

		if IsRhoai(test) {
			// Verify container images in the pods created by the TrainJob
			verifyPodContainerImages(test, namespace, trainJob.Name)
		}
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

func verifyPodContainerImages(test Test, namespace, trainJobName string) {
	imagePrefix := GetExpectedImagePrefix(test)

	product, err := GetProduct(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to get product")

	// Get CSV for the product related images for validation
	// For cluster wide CSVs we can use the namespace "openshift-operators"
	csv, err := FindCSVByPrefix(test, "openshift-operators", product.CsvNamePrefix)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find Product CSV")
	test.T().Logf("Found CSV: %s", csv.Name)

	// Get pods for the TrainJob
	pods := GetPods(test, namespace, metav1.ListOptions{LabelSelector: "jobset.sigs.k8s.io/jobset-name=" + trainJobName})
	test.Expect(pods).NotTo(BeEmpty(), "No pods found for TrainJob %s", trainJobName)

	// Verify container images in the pods created by the TrainJob
	for _, pod := range pods {
		images := getPodContainerImages(pod)
		test.Expect(images).NotTo(BeEmpty(), "No container images found for Pod %s", pod.Name)

		for _, image := range images {
			test.Expect(image).To(HavePrefix(imagePrefix), "Image %s should have prefix %s", image, imagePrefix)
			test.Expect(image).To(MatchRegexp(`@sha256:[a-f0-9]{64}$`),
				"Image %s should be SHA-based with valid digest", image)

			// Verify image is listed in CSV related images
			test.Expect(csv.Spec.RelatedImages).To(ContainElement(HaveField("Image", Equal(image))),
				"Image %s is not listed in CSV %s related images", image, csv.Name)
			test.T().Logf("Image %s is verified in CSV related images", image)
		}
	}
}

func getPodContainerImages(pod corev1.Pod) []string {
	var images []string
	for _, container := range pod.Spec.InitContainers {
		images = append(images, container.Image)
	}
	for _, container := range pod.Spec.Containers {
		images = append(images, container.Image)
	}
	return images
}

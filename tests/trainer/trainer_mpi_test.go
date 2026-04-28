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
	"encoding/base64"
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

func TestMultiNodeOpenMPITrainJob(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, NVIDIA))
	test := With(t)

	namespace := test.NewTestNamespace().Name

	mpiTrainingScript := string(readFile(test, "resources/fashion_mnist_mpi.py"))
	trainJob := createMPITrainJob(test, namespace, mpiTrainingScript)

	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutDouble).
		Should(Satisfy(TrainJobReachedFinalState))

	finalJob := TrainJob(test, namespace, trainJob.Name)(test)
	test.Expect(finalJob).To(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)),
		"TrainJob %s/%s should be complete; TrainJobFailed message: %s",
		namespace, trainJob.Name, TrainJobFailedMessage(finalJob))

	test.T().Logf("MPI TrainJob %s/%s completed successfully", namespace, trainJob.Name)

	jobset := SingleJobSet(test, namespace)(test)
	test.Expect(jobset).To(WithTransform(JobSetReplicatedJobsCount, Equal(2)),
		"MPI JobSet should have exactly 2 replicatedJobs (launcher + node)")

	launcherPods := GetPods(test, namespace, metav1.ListOptions{
		LabelSelector: "jobset.sigs.k8s.io/jobset-name=" + trainJob.Name + ",jobset.sigs.k8s.io/replicatedjob-name=launcher",
	})
	test.Expect(launcherPods).To(HaveLen(1), "Expected exactly 1 launcher pod")

	logs := GetPodLog(test, namespace, launcherPods[0].Name, corev1.PodLogOptions{})
	test.Expect(logs).To(ContainSubstring("MPI TrainJob test PASSED"),
		"Launcher logs should contain MPI success marker")
	test.Expect(logs).To(ContainSubstring("[Rank 0/2]"),
		"Launcher logs should show rank 0 of 2 processes")
	test.Expect(logs).To(ContainSubstring("[Rank 1/2]"),
		"Launcher logs should show rank 1 of 2 processes")
	test.T().Logf("Launcher pod logs:\n%s", logs)
}

func createMPITrainJob(test Test, namespace, trainingScript string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	encodedScript := base64.StdEncoding.EncodeToString([]byte(trainingScript))

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-mpi-trainjob-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA,
			},
			Trainer: &trainerv1alpha1.Trainer{
				NumNodes: Ptr(int32(2)),
				Command: []string{
					"mpirun",
					"python", "-c",
					`import base64; exec(base64.b64decode("` + encodedScript + `").decode("utf-8"))`,
				},
				ResourcesPerNode: &corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceName(NVIDIA.ResourceLabel): resource.MustParse("1"),
					},
					Limits: corev1.ResourceList{
						corev1.ResourceName(NVIDIA.ResourceLabel): resource.MustParse("1"),
					},
				},
			},
		},
	}

	created, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create MPI TrainJob")
	test.T().Logf("Created MPI TrainJob %s/%s successfully", created.Namespace, created.Name)

	return created
}

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

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	kfto "github.com/opendatahub-io/distributed-workloads/tests/kfto"
)

func TestJobSetWorkflow(t *testing.T) {
	Tags(t, Sanity)
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace().Name

	// Create PVC for shared storage
	pvc := CreatePersistentVolumeClaim(test, namespace, "1Gi", AccessModes(corev1.ReadWriteOnce))

	// Create TrainingRuntime with initializer jobs
	trainingRuntime := createTrainingRuntimeWithInitializers(test, namespace, pvc.Name)
	defer deleteTrainingRuntime(test, namespace, trainingRuntime.Name)

	// Create TrainJob referring the TrainingRuntime
	trainJob := createTrainJobWithInitializers(test, namespace, trainingRuntime.Name)
	defer deleteTrainJob(test, namespace, trainJob.Name)

	// Verify JobSet creation
	test.Eventually(SingleJobSet(test, namespace), TestTimeoutMedium).Should(
		WithTransform(JobSetReplicatedJobsCount, Equal(3)),
	)
	test.T().Log("JobSet created with 3 replicated jobs (dataset-initializer, model-initializer, node)")

	// Verify sequential job execution
	verifySequentialJobExecution(test, namespace)

	// Make sure the TrainJob completed
	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s completed", namespace, trainJob.Name)
}

func TestFailedJobSetWorkflow(t *testing.T) {
	Tags(t, Sanity)
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace().Name

	// Create PVC for shared storage
	pvc := CreatePersistentVolumeClaim(test, namespace, "1Gi", AccessModes(corev1.ReadWriteOnce))

	// Create TrainingRuntime With Initializers
	trainingRuntime := createTrainingRuntimeWithInitializers(test, namespace, pvc.Name)
	defer deleteTrainingRuntime(test, namespace, trainingRuntime.Name)

	// Create TrainJob
	trainJob := createTrainJobWithFailingInitializer(test, namespace, trainingRuntime.Name)
	defer deleteTrainJob(test, namespace, trainJob.Name)

	// Wait for JobSet failure
	test.Eventually(SingleJobSet(test, namespace), TestTimeoutMedium).Should(
		And(
			WithTransform(JobSetConditionFailed, Equal(metav1.ConditionTrue)),
			WithTransform(JobSetFailureMessage, ContainSubstring("jobset failed due to one or more job failures (first failed job: test-trainjob-fail-")),
		),
	)
	test.T().Logf("JobSet failed as expected")

	// Wait for TrainJob failure
	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionFailed, Equal(metav1.ConditionTrue)))
	test.T().Log("TrainJob failed as expected")
}

func createTrainingRuntimeWithInitializers(test Test, namespace, pvcName string) *trainerv1alpha1.TrainingRuntime {
	test.T().Helper()

	trainingRuntime := &trainerv1alpha1.TrainingRuntime{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-trainingruntime-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainingRuntimeSpec{
			Template: trainerv1alpha1.JobSetTemplateSpec{
				Spec: jobsetv1alpha2.JobSetSpec{
					ReplicatedJobs: []jobsetv1alpha2.ReplicatedJob{
						{
							Name:     "dataset-initializer",
							Replicas: 1,
							Template: batchv1.JobTemplateSpec{
								ObjectMeta: metav1.ObjectMeta{
									Labels: map[string]string{
										"trainer.kubeflow.org/trainjob-ancestor-step": "dataset-initializer",
									},
								},
								Spec: batchv1.JobSpec{
									BackoffLimit: Ptr(int32(0)),
									Template: corev1.PodTemplateSpec{
										Spec: corev1.PodSpec{
											RestartPolicy: corev1.RestartPolicyNever,
											Containers: []corev1.Container{
												{
													Name:            "dataset-initializer",
													Image:           kfto.GetAlpacaDatasetImage(),
													ImagePullPolicy: corev1.PullIfNotPresent,
													Command:         []string{"/bin/sh", "-c"},
													Args: []string{
														`
														echo "=========================================="
														echo "          Dataset Initializer             "
														echo "=========================================="
														
														# Check if dataset-initializer job should fail on purpose (for failure tests)
														if [ "${FAIL_ON_PURPOSE}" = "true" ]; then
															echo "ERROR: Failing on purpose as requested"
															exit 1
														fi
														
													echo "Dataset: ${DATASET_NAME}"
													echo "Target path: ${DATASET_PATH}"
													echo "Copying ${DATASET_NAME} to shared volume..."
													mkdir -p ${DATASET_PATH}
													cp -r /dataset/* ${DATASET_PATH}/
													ls -la ${DATASET_PATH}/
													echo ""
													echo "Dataset is copied successfully ..."
													echo "Sleeping for 5 seconds to allow test assertions ..."
													sleep 5
													`,
													},
													Resources: corev1.ResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("100m"),
															corev1.ResourceMemory: resource.MustParse("128Mi"),
														},
														Limits: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("200m"),
															corev1.ResourceMemory: resource.MustParse("256Mi"),
														},
													},
													VolumeMounts: []corev1.VolumeMount{
														{
															Name:      "workspace",
															MountPath: "/workspace",
														},
													},
												},
											},
											Volumes: []corev1.Volume{
												{
													Name: "workspace",
													VolumeSource: corev1.VolumeSource{
														PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
															ClaimName: pvcName,
														},
													},
												},
											},
										},
									},
								},
							},
						},
						{
							Name:     "model-initializer",
							Replicas: 1,
							DependsOn: []jobsetv1alpha2.DependsOn{
								{
									Name:   "dataset-initializer",
									Status: jobsetv1alpha2.DependencyComplete,
								},
							},
							Template: batchv1.JobTemplateSpec{
								ObjectMeta: metav1.ObjectMeta{
									Labels: map[string]string{
										"trainer.kubeflow.org/trainjob-ancestor-step": "model-initializer",
									},
								},
								Spec: batchv1.JobSpec{
									BackoffLimit: Ptr(int32(0)),
									Template: corev1.PodTemplateSpec{
										Spec: corev1.PodSpec{
											RestartPolicy: corev1.RestartPolicyNever,
											Containers: []corev1.Container{
												{
													Name:            "model-initializer",
													Image:           kfto.GetBloomModelImage(),
													ImagePullPolicy: corev1.PullIfNotPresent,
													Command:         []string{"/bin/sh", "-c"},
													Args: []string{
														`
														echo "=========================================="
														echo "            Model Initializer             "
														echo "=========================================="
													echo "Model: ${MODEL_NAME}"
													echo "Target path: ${MODEL_PATH}"
													echo "Copying ${MODEL_NAME} model to shared volume ..."
													mkdir -p ${MODEL_PATH}
													cp -r /models/${MODEL_NAME} ${MODEL_PATH}/
													ls -la ${MODEL_PATH}/
													echo ""
													echo "Model is copied successfully ..."
													echo "Sleeping for 5 seconds to allow test assertions ..."
													sleep 5
													`,
													},
													Resources: corev1.ResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("100m"),
															corev1.ResourceMemory: resource.MustParse("128Mi"),
														},
														Limits: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("200m"),
															corev1.ResourceMemory: resource.MustParse("256Mi"),
														},
													},
													VolumeMounts: []corev1.VolumeMount{
														{
															Name:      "workspace",
															MountPath: "/workspace",
														},
													},
												},
											},
											Volumes: []corev1.Volume{
												{
													Name: "workspace",
													VolumeSource: corev1.VolumeSource{
														PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
															ClaimName: pvcName,
														},
													},
												},
											},
										},
									},
								},
							},
						},
						{
							Name:     "node",
							Replicas: 1,
							DependsOn: []jobsetv1alpha2.DependsOn{
								{
									Name:   "model-initializer",
									Status: jobsetv1alpha2.DependencyComplete,
								},
							},
							Template: batchv1.JobTemplateSpec{
								ObjectMeta: metav1.ObjectMeta{
									Labels: map[string]string{
										"trainer.kubeflow.org/trainjob-ancestor-step": "trainer",
									},
								},
								Spec: batchv1.JobSpec{
									BackoffLimit: Ptr(int32(0)),
									Template: corev1.PodTemplateSpec{
										Spec: corev1.PodSpec{
											RestartPolicy: corev1.RestartPolicyNever,
											Containers: []corev1.Container{
												{
													Name:            "node",
													Image:           GetTrainingCudaPyTorch28Image(),
													ImagePullPolicy: corev1.PullIfNotPresent,
													Resources: corev1.ResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("100m"),
															corev1.ResourceMemory: resource.MustParse("128Mi"),
														},
														Limits: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("200m"),
															corev1.ResourceMemory: resource.MustParse("256Mi"),
														},
													},
													VolumeMounts: []corev1.VolumeMount{
														{
															Name:      "workspace",
															MountPath: "/workspace",
														},
													},
												},
											},
											Volumes: []corev1.Volume{
												{
													Name: "workspace",
													VolumeSource: corev1.VolumeSource{
														PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
															ClaimName: pvcName,
														},
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	runtime, err := test.Client().Trainer().TrainerV1alpha1().TrainingRuntimes(namespace).Create(
		test.Ctx(),
		trainingRuntime,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create TrainingRuntime")
	test.T().Logf("Created TrainingRuntime %s/%s", runtime.Namespace, runtime.Name)

	return runtime
}

func createTrainJobWithInitializers(test Test, namespace, runtimeName string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-trainjob-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: runtimeName,
				Kind: Ptr("TrainingRuntime"),
			},
			Initializer: &trainerv1alpha1.Initializer{
				Dataset: &trainerv1alpha1.DatasetInitializer{
					Env: []corev1.EnvVar{
						{
							Name:  "DATASET_NAME",
							Value: "alpaca-dataset",
						},
						{
							Name:  "DATASET_PATH",
							Value: "/workspace/datasets",
						},
					},
				},
				Model: &trainerv1alpha1.ModelInitializer{
					Env: []corev1.EnvVar{
						{
							Name:  "MODEL_NAME",
							Value: "bloom-560m",
						},
						{
							Name:  "MODEL_PATH",
							Value: "/workspace/model",
						},
					},
				},
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command: []string{
					"sh",
					"-c",
					`
				echo "============================================================="
				echo "  Check trainer job has access to dataset and model ... "
				echo "============================================================="
				echo "Dataset path: ${DATASET_PATH}"
				echo "Model path: ${MODEL_PATH}"
				
				if [ -d "${DATASET_PATH}" ]; then
					echo "   Dataset is accessible  "
					ls -la ${DATASET_PATH}/ | head -5
				else
					echo "   Dataset NOT found at ${DATASET_PATH}!   "
					exit 1
				fi
				
				echo ""
				if [ -d "${MODEL_PATH}" ]; then
					echo "   Model is accessible   "
					ls -la ${MODEL_PATH}/ | head -5
				else
					echo "   Model NOT found at ${MODEL_PATH}!   "
					exit 1
				fi
				
				echo ""
				echo "Trainer job has access to dataset and model. Verification is successful !!!"
				`,
				},
				Env: []corev1.EnvVar{
					{
						Name:  "DATASET_PATH",
						Value: "/workspace/datasets",
					},
					{
						Name:  "MODEL_PATH",
						Value: "/workspace/model",
					},
				},
				NumNodes: Ptr(int32(1)),
			},
		},
	}

	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create TrainJob")
	test.T().Logf("Created TrainJob %s/%s", createdTrainJob.Namespace, createdTrainJob.Name)

	return createdTrainJob
}

func createTrainJobWithFailingInitializer(test Test, namespace, runtimeName string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-trainjob-fail-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: runtimeName,
				Kind: Ptr("TrainingRuntime"),
			},
			Initializer: &trainerv1alpha1.Initializer{
				Dataset: &trainerv1alpha1.DatasetInitializer{
					Env: []corev1.EnvVar{
						{
							Name:  "FAIL_ON_PURPOSE",
							Value: "true",
						},
						{
							Name:  "DATASET_NAME",
							Value: "alpaca-dataset",
						},
						{
							Name:  "DATASET_PATH",
							Value: "/workspace/datasets",
						},
					},
				},
				Model: &trainerv1alpha1.ModelInitializer{
					Env: []corev1.EnvVar{
						{
							Name:  "MODEL_NAME",
							Value: "bloom-560m",
						},
						{
							Name:  "MODEL_PATH",
							Value: "/workspace/model",
						},
					},
				},
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command: []string{
					"sh",
					"-c",
					"echo 'This should not run if initializer fails'; exit 0",
				},
				NumNodes: Ptr(int32(1)),
			},
		},
	}

	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create TrainJob")
	test.T().Logf("Created TrainJob %s/%s", createdTrainJob.Namespace, createdTrainJob.Name)

	return createdTrainJob
}

func deleteTrainingRuntime(test Test, namespace, name string) {
	test.T().Helper()

	err := test.Client().Trainer().TrainerV1alpha1().TrainingRuntimes(namespace).Delete(
		test.Ctx(),
		name,
		metav1.DeleteOptions{},
	)
	if err != nil {
		test.T().Logf("Warning: Failed to delete TrainingRuntime %s/%s: %v", namespace, name, err)
	} else {
		test.T().Logf("Deleted TrainingRuntime %s/%s successfully", namespace, name)
	}
}

func verifySequentialJobExecution(test Test, namespace string) {
	test.T().Helper()

	// Define expected job sequence with total job count at each stage
	jobSequence := []struct {
		name          string
		expectedCount int
	}{
		{"dataset-initializer", 1}, // First job only
		{"model-initializer", 2},   // dataset-initializer + model-initializer
		{"node", 3},                // All 3 jobs
	}

	test.T().Log("Monitoring job execution ...")

	for _, stage := range jobSequence {
		var job *batchv1.Job
		test.Eventually(func() error {
			jobs, err := GetAllJobs(test, namespace)
			if err != nil {
				return err
			}
			if len(jobs) != stage.expectedCount {
				return fmt.Errorf("expected %d jobs, found %d", stage.expectedCount, len(jobs))
			}
			job, _ = GetJobByNamePattern(test, namespace, stage.name)
			if job == nil {
				return fmt.Errorf("%s job not found", stage.name)
			}
			return nil
		}, TestTimeoutShort).Should(Succeed())
		test.T().Logf("%s job is created: %s", stage.name, job.Name)

		// Wait for job to complete (except for node)
		if stage.name != "node" {
			test.Eventually(func() error {
				job, err := GetJobByNamePattern(test, namespace, stage.name)
				if err != nil {
					return err
				}
				if job == nil || job.Status.Succeeded != 1 {
					return fmt.Errorf("%s job not yet completed", stage.name)
				}
				return nil
			}, TestTimeoutMedium).Should(Succeed())
			test.T().Logf("%s job is completed: %s", stage.name, job.Name)
		}
	}

	test.T().Log("Sequential job execution is verified successfully: dataset-initializer → model-initializer → node")
}

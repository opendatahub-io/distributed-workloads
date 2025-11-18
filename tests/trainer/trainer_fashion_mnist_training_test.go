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
	"k8s.io/apimachinery/pkg/util/intstr"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestPyTorchDDPMultiNodeMultiCPUWithTorchCuda28(t *testing.T) {
	Tags(t, Tier1, MultiNode(2))
	runPyTorchDDPMultiNodeJob(t, CPU, GetTrainingCudaPyTorch28Image(), 2, 2)
}

func TestPyTorchDDPSingleNodeMultiGPUWithTorchCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNode(2))
	runPyTorchDDPMultiNodeJob(t, NVIDIA, GetTrainingCudaPyTorch28Image(), 1, 2)
}

func TestPyTorchDDPMultiNodeMultiGPUWithTorchCuda(t *testing.T) {
	Tags(t, KftoCuda, MultiNode(2))
	runPyTorchDDPMultiNodeJob(t, NVIDIA, GetTrainingCudaPyTorch28Image(), 2, 2)
}

func TestPyTorchDDPSingleNodeMultiGPUWithTorchRocm(t *testing.T) {
	Tags(t, KftoRocm, MultiNode(2))
	runPyTorchDDPMultiNodeJob(t, AMD, GetTrainingRocmPyTorch28Image(), 1, 2)
}

func TestPyTorchDDPMultiNodeMultiGPUWithTorchRocm(t *testing.T) {
	Tags(t, KftoRocm, MultiNode(2))
	runPyTorchDDPMultiNodeJob(t, AMD, GetTrainingRocmPyTorch28Image(), 2, 2)
}

func runPyTorchDDPMultiNodeJob(t *testing.T, accelerator Accelerator, baseImage string, numNodes, numProcPerNode int32) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace().Name

	// Get storageclass that supports RWX PVC provisioning
	storageClass, err := GetRWXStorageClass(test)
	test.Expect(err).ToNot(HaveOccurred(), "Failed to find an RWX supporting StorageClass")

	// Create PVC
	pvc := CreatePersistentVolumeClaim(test, namespace, "2Gi", AccessModes(corev1.ReadWriteMany), StorageClassName(storageClass.Name))

	// Create ConfigMap
	files := map[string][]byte{
		"fashion_mnist.py":          readFile(test, "resources/fashion_mnist.py"),
		"download_fashion_mnist.py": readFile(test, "resources/download_fashion_mnist.py"),
		"requirements.txt":          readFile(test, "resources/requirements.txt"),
	}
	config := CreateConfigMap(test, namespace, files)

	// Create TrainingRuntime with dataset-initializer
	trainingRuntime := createFashionMNISTTrainingRuntime(test, namespace, config.Name, pvc.Name, baseImage, accelerator, numProcPerNode)
	defer deleteTrainingRuntime(test, namespace, trainingRuntime.Name)

	// Create TrainJob
	trainJob := createFashionMNISTTrainJob(test, namespace, trainingRuntime.Name, accelerator, numNodes, numProcPerNode)
	defer deleteTrainJob(test, namespace, trainJob.Name)

	// Verify JobSet creation
	test.T().Logf("Verifying JobSet creation with replicated jobs...")
	test.Eventually(SingleJobSet(test, namespace), TestTimeoutDouble).Should(
		WithTransform(JobSetReplicatedJobsCount, Equal(2)),
	)
	test.T().Logf("JobSet created with 2 replicated jobs (dataset-initializer, node)")

	test.T().Logf("Checking JobSet status...")
	test.Eventually(SingleJobSet(test, namespace), TestTimeoutDouble).Should(
		WithTransform(JobSetConditionCompleted, Equal(metav1.ConditionTrue)),
	)
	test.T().Logf("JobSet marked as completed")

	// Make sure the TrainJob completed
	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutMedium).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s completed", namespace, trainJob.Name)

}

func createFashionMNISTTrainingRuntime(test Test, namespace, configMapName, pvcName, baseImage string, accelerator Accelerator, numProcPerNode int32) *trainerv1alpha1.TrainingRuntime {
	test.T().Helper()

	var backend string
	if accelerator.IsGpu() {
		backend = "nccl"
	} else {
		backend = "gloo"
	}

	trainingRuntime := &trainerv1alpha1.TrainingRuntime{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-fashion-mnist-runtime-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainingRuntimeSpec{
			MLPolicy: &trainerv1alpha1.MLPolicy{
				NumNodes: Ptr(int32(1)),
				MLPolicySource: trainerv1alpha1.MLPolicySource{
					Torch: &trainerv1alpha1.TorchMLPolicySource{
						NumProcPerNode: Ptr(intstr.FromString("auto")),
					},
				},
			},
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
											RestartPolicy: corev1.RestartPolicyOnFailure,
											Containers: []corev1.Container{
												{
													Name:            "dataset-initializer",
													Image:           baseImage,
													ImagePullPolicy: corev1.PullIfNotPresent,
													Command:         []string{"/bin/bash", "-c"},
													Args: []string{`
															set -e
															echo "=========================================="
															echo "          Dataset Initializer             "
															echo "=========================================="

														echo "Installing dependencies from requirements.txt to ${LIB_PATH}..."
														mkdir -p ${LIB_PATH}

														while IFS= read -r line || [[ -n "$line" ]]; do

															[[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
															
															# Check if line has "# no-deps" marker
															if [[ "$line" =~ "# no-deps" ]]; then
																pkg=$(echo "$line" | sed 's/[[:space:]]*#.*//')
																echo "Installing $pkg (--no-deps)..."
																pip install --no-cache-dir --no-deps "$pkg" --target=${LIB_PATH}
															else
																# Install with dependencies
																pkg=$(echo "$line" | sed 's/[[:space:]]*#.*//')
																echo "Installing $pkg (with deps)..."
																pip install --no-cache-dir "$pkg" --target=${LIB_PATH}
															fi
														done < /mnt/scripts/requirements.txt
														
														echo ""
														echo "Dependencies installed successfully ..."
														ls -la ${LIB_PATH}/ | head -20

														# Download dataset to shared volume
														export PYTHONPATH=${LIB_PATH}:$PYTHONPATH
														python3 /mnt/scripts/download_fashion_mnist.py --dataset_path ${DATASET_PATH}

														echo "Dataset downloaded successfully!"
														echo "Dataset contents:"
														ls -la ${DATASET_PATH}/

														echo "Dataset initialization completed!"
													`},
													Resources: corev1.ResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("1"),
															corev1.ResourceMemory: resource.MustParse("2Gi"),
														},
														Limits: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("2"),
															corev1.ResourceMemory: resource.MustParse("4Gi"),
														},
													},
													VolumeMounts: []corev1.VolumeMount{
														{
															Name:      "workspace",
															MountPath: "/workspace",
														},
														{
															Name:      "training-scripts",
															MountPath: "/mnt/scripts",
															ReadOnly:  true,
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
												{
													Name: "training-scripts",
													VolumeSource: corev1.VolumeSource{
														ConfigMap: &corev1.ConfigMapVolumeSource{
															LocalObjectReference: corev1.LocalObjectReference{
																Name: configMapName,
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

						{
							Name:     "node",
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
										"trainer.kubeflow.org/trainjob-ancestor-step": "trainer",
									},
								},
								Spec: batchv1.JobSpec{
									BackoffLimit: Ptr(int32(0)),
									Template: corev1.PodTemplateSpec{
										Spec: corev1.PodSpec{
											RestartPolicy: corev1.RestartPolicyOnFailure,
											Containers: []corev1.Container{
												{
													Name:            "node",
													Image:           baseImage,
													ImagePullPolicy: corev1.PullIfNotPresent,
													Command:         []string{"/bin/bash", "-c"},
													Args: []string{
														fmt.Sprintf(`
															set -e

														echo "==================== Environment Info ===================="
														echo "PyTorch Backend: %s"
														echo "Dataset Path: ${DATASET_PATH}"
														echo "==========================================================="

														# Verify dataset exists
														if [ ! -d "${DATASET_PATH}/FashionMNIST" ]; then
															echo "ERROR: Dataset not found at ${DATASET_PATH}/FashionMNIST"
															echo "Dataset initializer may have failed!"
															exit 1
														fi

														echo ""
														echo " Dataset found, listing contents:"
														ls -la ${DATASET_PATH}/
														echo ""

														# Verify dependencies exist (installed by dataset-initializer)
														if [ ! -d "${LIB_PATH}" ]; then
															echo "ERROR: Dependencies not found at ${LIB_PATH}"
															echo "Dataset initializer may have failed to install dependencies"
															exit 1
														fi

														echo " Using dependencies from shared volume: ${LIB_PATH}"
														echo ""

														# Set Python path for dependencies 
														export PYTHONPATH=${LIB_PATH}:$PYTHONPATH

															echo "==================== Starting Distributed Training ===================="
															torchrun /mnt/scripts/fashion_mnist.py

															echo ""
															echo "==================== Training Complete ===================="
															`, backend),
													},
													Resources: buildResourceRequirements(accelerator, numProcPerNode),
													VolumeMounts: []corev1.VolumeMount{
														{
															Name:      "workspace",
															MountPath: "/workspace",
														},
														{
															Name:      "training-scripts",
															MountPath: "/mnt/scripts",
															ReadOnly:  true,
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
												{
													Name: "training-scripts",
													VolumeSource: corev1.VolumeSource{
														ConfigMap: &corev1.ConfigMapVolumeSource{
															LocalObjectReference: corev1.LocalObjectReference{
																Name: configMapName,
															},
														},
													},
												},
											},
											Affinity: &corev1.Affinity{
												PodAntiAffinity: &corev1.PodAntiAffinity{
													RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
														{
															LabelSelector: &metav1.LabelSelector{
																MatchLabels: map[string]string{
																	"trainer.kubeflow.org/trainjob-ancestor-step": "trainer",
																},
															},
															TopologyKey: "kubernetes.io/hostname",
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
	test.T().Logf("Created TrainingRuntime %s/%s with dataset-initializer ", runtime.Namespace, runtime.Name)

	return runtime
}

func createFashionMNISTTrainJob(test Test, namespace, runtimeName string, accelerator Accelerator, numNodes, numProcPerNode int32) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-fashion-mnist-trainjob-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: runtimeName,
				Kind: Ptr("TrainingRuntime"),
			},
			Initializer: &trainerv1alpha1.Initializer{
				Dataset: &trainerv1alpha1.DatasetInitializer{
					Env: append(
						[]corev1.EnvVar{
							{Name: "DATASET_PATH", Value: "/workspace/data"},
							{Name: "LIB_PATH", Value: "/workspace/lib"},
						},
						storageBucketEnvVars(test)...,
					),
				},
			},
			Trainer: &trainerv1alpha1.Trainer{
				NumNodes:       Ptr(numNodes),
				NumProcPerNode: Ptr(intstr.FromInt32(numProcPerNode)),
				Env: []corev1.EnvVar{
					{Name: "DATASET_PATH", Value: "/workspace/data"},
					{Name: "LIB_PATH", Value: "/workspace/lib"},
				},
			},
		},
	}

	if accelerator.IsGpu() {
		trainJob.Spec.Trainer.Env = append(trainJob.Spec.Trainer.Env,
			corev1.EnvVar{Name: "NUM_EPOCHS", Value: "3"},
			corev1.EnvVar{Name: "BATCH_SIZE", Value: "256"},
			corev1.EnvVar{Name: "NCCL_DEBUG", Value: "INFO"},
			corev1.EnvVar{Name: "TORCH_DISTRIBUTED_DEBUG", Value: "DETAIL"},
		)
	}

	createTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create TrainJob")
	test.T().Logf("Created TrainJob %s/%s", createTrainJob.Namespace, createTrainJob.Name)

	return createTrainJob
}

func buildResourceRequirements(accelerator Accelerator, numProcPerNode int32) corev1.ResourceRequirements {
	resourceRequirements := corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("3"),
			corev1.ResourceMemory: resource.MustParse("16Gi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("5"),
			corev1.ResourceMemory: resource.MustParse("24Gi"),
		},
	}

	// Add GPU resources if using GPU accelerator
	if accelerator.IsGpu() {
		gpuCount := fmt.Sprintf("%d", numProcPerNode)
		resourceRequirements.Requests[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(gpuCount)
		resourceRequirements.Limits[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(gpuCount)
	}

	return resourceRequirements
}

func storageBucketEnvVars(test Test) []corev1.EnvVar {
	test.T().Helper()

	envVars := []corev1.EnvVar{}

	storage_bucket_endpoint, storage_bucket_endpoint_exists := GetStorageBucketDefaultEndpoint()
	storage_bucket_access_key_id, storage_bucket_access_key_id_exists := GetStorageBucketAccessKeyId()
	storage_bucket_secret_key, storage_bucket_secret_key_exists := GetStorageBucketSecretKey()
	storage_bucket_name, storage_bucket_name_exists := GetStorageBucketName()
	storage_bucket_fashion_mnist_dir, storage_bucket_fashion_mnist_dir_exists := GetStorageBucketFashionMnistDir()

	if storage_bucket_endpoint_exists && storage_bucket_access_key_id_exists && storage_bucket_secret_key_exists && storage_bucket_name_exists && storage_bucket_fashion_mnist_dir_exists {
		test.T().Logf("S3/Minio configuration detected, adding storage environment variables to dataset-initializer")
		envVars = append(envVars,
			corev1.EnvVar{Name: "AWS_DEFAULT_ENDPOINT", Value: storage_bucket_endpoint},
			corev1.EnvVar{Name: "AWS_ACCESS_KEY_ID", Value: storage_bucket_access_key_id},
			corev1.EnvVar{Name: "AWS_SECRET_ACCESS_KEY", Value: storage_bucket_secret_key},
			corev1.EnvVar{Name: "AWS_STORAGE_BUCKET", Value: storage_bucket_name},
			corev1.EnvVar{Name: "AWS_STORAGE_BUCKET_FASHION_MNIST_DIR", Value: storage_bucket_fashion_mnist_dir},
		)
	} else {
		test.T().Logf("S3/Minio configuration incomplete or not provided, dataset-initializer will use public mirrors")
	}

	return envVars
}

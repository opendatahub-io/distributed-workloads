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
	"bytes"
	"testing"
	"time"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	"github.com/opendatahub-io/distributed-workloads/tests/fms"
	"github.com/opendatahub-io/distributed-workloads/tests/kfto"
)

func TestTrainJobWithSFTtrainerFinetuning(t *testing.T) {
	runTrainJobWithSFTtrainer(t, "resources/config.json")
}

func TestTrainJobWithSFTtrainerLoRa(t *testing.T) {
	runTrainJobWithSFTtrainer(t, "resources/config_lora.json")
}

func TestTrainJobWithSFTtrainerQLoRa(t *testing.T) {
	runTrainJobWithSFTtrainer(t, "resources/config_qlora.json")
}

func runTrainJobWithSFTtrainer(t *testing.T, modelConfigFile string) {
	test := With(t)

	// Create a namespace
	namespace := test.CreateOrGetTestNamespace().Name

	// Create TrainingRuntime for FMS SFT training (single-GPU, uses /tmp paths)
	runtime := createSingleGpuTrainingRuntime(test, namespace)
	defer test.Client().Trainer().TrainerV1alpha1().TrainingRuntimes(namespace).Delete(test.Ctx(), runtime.Name, metav1.DeleteOptions{})

	// Create PVC for base model
	baseModelPvc := CreatePersistentVolumeClaim(test, namespace, "10Gi", AccessModes(corev1.ReadWriteOnce))
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), baseModelPvc.Name, metav1.DeleteOptions{})

	// Create PVC for trained model
	outputPvc := CreatePersistentVolumeClaim(test, namespace, "10Gi", AccessModes(corev1.ReadWriteOnce))
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), outputPvc.Name, metav1.DeleteOptions{})

	// Load training job config file
	configContent := fms.ReadFile(test, modelConfigFile)

	_, bucketDownloadSet := fms.GetStorageBucketDownloadName()
	if bucketDownloadSet {
		// Download base model to PVC
		fms.DownloadFromS3(test, namespace, baseModelPvc.Name, "model")

		// Replace model placeholder with mounted model folder
		configContent = bytes.Replace(configContent, []byte("<MODEL_PATH_PLACEHOLDER>"), []byte("/mnt/model/model/granite-3b-code-base-2k"), 1)
	} else {
		// Replace model placeholder with model reference
		configContent = bytes.Replace(configContent, []byte("<MODEL_PATH_PLACEHOLDER>"), []byte("ibm-granite/granite-3b-code-base-2k"), 1)
	}

	// Create a ConfigMap with configuration
	configData := map[string][]byte{
		"config.json": configContent,
	}
	config := CreateConfigMap(test, namespace, configData)

	// Create training TrainJob (without Kueue integration)
	trainJob := createSftTrainJob(test, namespace, runtime.Name, "", *config, baseModelPvc.Name, outputPvc.Name)
	defer test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Delete(test.Ctx(), trainJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the TrainJob succeed
	test.Eventually(TrainJob(test, namespace, trainJob.Name), 30*time.Minute).Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s ran successfully", trainJob.Namespace, trainJob.Name)

	_, bucketUploadSet := fms.GetStorageBucketUploadName()
	if bucketUploadSet {
		fms.UploadToS3(test, namespace, outputPvc.Name, "model")
	}
}

func TestTrainJobUsingKueueQuota(t *testing.T) {
	test := With(t)

	// Create a namespace
	namespace := test.CreateOrGetTestNamespace().Name

	// Create TrainingRuntime for FMS SFT training (single-GPU, uses /tmp paths)
	runtime := createSingleGpuTrainingRuntime(test, namespace)
	defer test.Client().Trainer().TrainerV1alpha1().TrainingRuntimes(namespace).Delete(test.Ctx(), runtime.Name, metav1.DeleteOptions{})

	// Create limited Kueue resources to run just one TrainJob at a time
	resourceFlavor := CreateKueueResourceFlavor(test, kueuev1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := kueuev1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []kueuev1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory"), corev1.ResourceName("nvidia.com/gpu")},
				Flavors: []kueuev1beta1.FlavorQuotas{
					{
						Name: kueuev1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []kueuev1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("3"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("8Gi"),
							},
							{
								Name:         corev1.ResourceName("nvidia.com/gpu"),
								NominalQuota: resource.MustParse("1"),
							},
						},
					},
				},
			},
		},
	}
	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, namespace, clusterQueue.Name, AsDefaultQueue)

	// Create PVC for base model
	baseModelPvc := CreatePersistentVolumeClaim(test, namespace, "10Gi", AccessModes(corev1.ReadWriteOnce))
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), baseModelPvc.Name, metav1.DeleteOptions{})

	// Load training job config file
	configContent := fms.ReadFile(test, "resources/config.json")

	_, bucketDownloadSet := fms.GetStorageBucketDownloadName()
	if bucketDownloadSet {
		// Download base model to PVC
		fms.DownloadFromS3(test, namespace, baseModelPvc.Name, "model")

		// Replace model placeholder with mounted model folder
		configContent = bytes.Replace(configContent, []byte("<MODEL_PATH_PLACEHOLDER>"), []byte("/mnt/model/model/granite-3b-code-base-2k"), 1)
	} else {
		// Replace model placeholder with model reference
		configContent = bytes.Replace(configContent, []byte("<MODEL_PATH_PLACEHOLDER>"), []byte("ibm-granite/granite-3b-code-base-2k"), 1)
	}

	// Create a ConfigMap with configuration
	configData := map[string][]byte{
		"config.json": configContent,
	}
	config := CreateConfigMap(test, namespace, configData)

	// Create first PVC for trained model
	outputPvc := CreatePersistentVolumeClaim(test, namespace, "10Gi", AccessModes(corev1.ReadWriteOnce))
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), outputPvc.Name, metav1.DeleteOptions{})

	// Create first training TrainJob
	trainJob := createSftTrainJob(test, namespace, runtime.Name, localQueue.Name, *config, baseModelPvc.Name, outputPvc.Name)
	defer test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Delete(test.Ctx(), trainJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the TrainJob is not suspended (running)
	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionFalse)))

	// Create second PVC for trained model
	secondOutputPvc := CreatePersistentVolumeClaim(test, namespace, "10Gi", AccessModes(corev1.ReadWriteOnce))
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), secondOutputPvc.Name, metav1.DeleteOptions{})

	// Create second training TrainJob
	secondTrainJob := createSftTrainJob(test, namespace, runtime.Name, localQueue.Name, *config, baseModelPvc.Name, secondOutputPvc.Name)
	defer test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Delete(test.Ctx(), secondTrainJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the second TrainJob is suspended, waiting for first job to finish
	test.Eventually(TrainJob(test, namespace, secondTrainJob.Name), TestTimeoutShort).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionTrue)))

	// Make sure the first TrainJob succeed
	test.Eventually(TrainJob(test, namespace, trainJob.Name), 30*time.Minute).Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s ran successfully", trainJob.Namespace, trainJob.Name)

	// Second TrainJob should be started now
	test.Eventually(TrainJob(test, namespace, secondTrainJob.Name), TestTimeoutShort).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionFalse)))

	// Make sure the second TrainJob succeed
	test.Eventually(TrainJob(test, namespace, secondTrainJob.Name), 30*time.Minute).Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s ran successfully", secondTrainJob.Namespace, secondTrainJob.Name)
}

func createSftTrainJob(test Test, namespace, runtimeName, localQueueName string, config corev1.ConfigMap, baseModelPvcName, outputPvcName string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	labels := map[string]string{}
	if localQueueName != "" {
		labels["kueue.x-k8s.io/queue-name"] = localQueueName
	}

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "trainer-sft-",
			Namespace:    namespace,
			Labels:       labels,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: runtimeName,
				Kind: Ptr("TrainingRuntime"),
			},
			Trainer: &trainerv1alpha1.Trainer{
				Image: Ptr(fms.GetFmsHfTuningImage(test)),
			},
			PodTemplateOverrides: []trainerv1alpha1.PodTemplateOverride{
				{
					TargetJobs: []trainerv1alpha1.PodTemplateOverrideTargetJob{
						{Name: "node"},
					},
					Spec: &trainerv1alpha1.PodTemplateSpecOverride{
						Tolerations: []corev1.Toleration{
							{
								Key:      "nvidia.com/gpu",
								Operator: corev1.TolerationOpExists,
							},
						},
						InitContainers: []trainerv1alpha1.ContainerOverride{
							{
								Name: "copy-dataset",
								VolumeMounts: []corev1.VolumeMount{
									{
										Name:      "tmp-volume",
										MountPath: "/tmp",
									},
									{
										Name:      "scratch-volume",
										MountPath: "/mnt/scratch",
									},
								},
							},
						},
						Containers: []trainerv1alpha1.ContainerOverride{
							{
								Name: "node",
								VolumeMounts: []corev1.VolumeMount{
									{
										Name:      "tmp-volume",
										MountPath: "/tmp",
									},
									{
										Name:      "config-volume",
										MountPath: "/etc/config",
									},
									{
										Name:      "scratch-volume",
										MountPath: "/mnt/scratch",
									},
									{
										Name:      "base-model-volume",
										MountPath: "/mnt/model",
									},
									{
										Name:      "output-volume",
										MountPath: "/mnt/output",
									},
								},
							},
						},
						Volumes: []corev1.Volume{
							{
								Name: "tmp-volume",
								VolumeSource: corev1.VolumeSource{
									EmptyDir: &corev1.EmptyDirVolumeSource{},
								},
							},
							{
								Name: "config-volume",
								VolumeSource: corev1.VolumeSource{
									ConfigMap: &corev1.ConfigMapVolumeSource{
										LocalObjectReference: corev1.LocalObjectReference{
											Name: config.Name,
										},
									},
								},
							},
							{
								Name: "scratch-volume",
								VolumeSource: corev1.VolumeSource{
									EmptyDir: &corev1.EmptyDirVolumeSource{},
								},
							},
							{
								Name: "base-model-volume",
								VolumeSource: corev1.VolumeSource{
									PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
										ClaimName: baseModelPvcName,
									},
								},
							},
							{
								Name: "output-volume",
								VolumeSource: corev1.VolumeSource{
									PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
										ClaimName: outputPvcName,
									},
								},
							},
						},
					},
				},
			},
		},
	}

	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(test.Ctx(), trainJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created TrainJob %s/%s successfully", createdTrainJob.Namespace, createdTrainJob.Name)

	return createdTrainJob
}

// createSingleGpuTrainingRuntime creates a TrainingRuntime for single-GPU FMS SFT training
// Uses /tmp paths for dataset and temp files (simpler setup for lightweight tests)
func createSingleGpuTrainingRuntime(test Test, namespace string) *trainerv1alpha1.TrainingRuntime {
	test.T().Helper()

	runtime := &trainerv1alpha1.TrainingRuntime{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "fms-sft-runtime-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainingRuntimeSpec{
			MLPolicy: &trainerv1alpha1.MLPolicy{
				NumNodes: Ptr(int32(1)),
				MLPolicySource: trainerv1alpha1.MLPolicySource{
					Torch: &trainerv1alpha1.TorchMLPolicySource{},
				},
			},
			Template: trainerv1alpha1.JobSetTemplateSpec{
				Spec: jobsetv1alpha2.JobSetSpec{
					ReplicatedJobs: []jobsetv1alpha2.ReplicatedJob{
						{
							Name: "node",
							Template: batchv1.JobTemplateSpec{
								ObjectMeta: metav1.ObjectMeta{
									Labels: map[string]string{
										"trainer.kubeflow.org/trainjob-ancestor-step": "trainer",
									},
								},
								Spec: batchv1.JobSpec{
									Template: corev1.PodTemplateSpec{
										Spec: corev1.PodSpec{
											Tolerations: []corev1.Toleration{
												{
													Key:      "nvidia.com/gpu",
													Operator: corev1.TolerationOpExists,
												},
											},
											InitContainers: []corev1.Container{
												{
													Name:            "copy-dataset",
													Image:           kfto.GetAlpacaDatasetImage(),
													ImagePullPolicy: corev1.PullIfNotPresent,
													VolumeMounts: []corev1.VolumeMount{
														{
															Name:      "tmp-volume",
															MountPath: "/tmp",
														},
													},
													Command: []string{"/bin/sh", "-c"},
													Args:    []string{"mkdir -p /tmp/dataset; cp /dataset/alpaca_data_hundredth.json /tmp/dataset/alpaca_data.json"},
												},
											},
											Containers: []corev1.Container{
												{
													Name:            "node",
													Image:           fms.GetFmsHfTuningImage(test),
													ImagePullPolicy: corev1.PullIfNotPresent,
													Env: []corev1.EnvVar{
														{
															Name:  "SFT_TRAINER_CONFIG_JSON_PATH",
															Value: "/etc/config/config.json",
														},
														{
															Name:  "HF_HOME",
															Value: "/tmp/huggingface",
														},
														{
															Name:  "HOME",
															Value: "/tmp/home",
														},
													},
													VolumeMounts: []corev1.VolumeMount{
														{
															Name:      "tmp-volume",
															MountPath: "/tmp",
														},
													},
													Resources: corev1.ResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("2"),
															corev1.ResourceMemory: resource.MustParse("8Gi"),
															"nvidia.com/gpu":      resource.MustParse("1"),
														},
														Limits: corev1.ResourceList{
															"nvidia.com/gpu": resource.MustParse("1"),
														},
													},
													SecurityContext: &corev1.SecurityContext{
														RunAsNonRoot:           Ptr(true),
														ReadOnlyRootFilesystem: Ptr(true),
													},
												},
											},
											Volumes: []corev1.Volume{
												{
													Name: "tmp-volume",
													VolumeSource: corev1.VolumeSource{
														EmptyDir: &corev1.EmptyDirVolumeSource{},
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

	createdRuntime, err := test.Client().Trainer().TrainerV1alpha1().TrainingRuntimes(namespace).Create(
		test.Ctx(),
		runtime,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created TrainingRuntime %s/%s successfully", createdRuntime.Namespace, createdRuntime.Name)

	return createdRuntime
}

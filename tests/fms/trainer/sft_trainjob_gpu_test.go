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
	"time"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusmodel "github.com/prometheus/common/model"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	"github.com/opendatahub-io/distributed-workloads/tests/fms"
	"github.com/opendatahub-io/distributed-workloads/tests/kfto"
)

func TestMultiGpuTrainJobAllamBeta13bChatGptq(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_allam_beta_13b_chat_gptq.json", 2, mountModelVolumeIntoTrainer)
}

func TestMultiGpuTrainJobGranite8bCodeInstructGptq(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_granite_8b_code_instruct_gptq.json", 2, mountModelVolumeIntoTrainer)
}

func TestMultiGpuTrainJobGranite20bCodeInstruct(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_granite_20b_code_instruct.json", 4)
}

func TestMultiGpuTrainJobGranite34bCodeBaseGptq(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_granite_34b_code_base_gptq.json", 2, mountModelVolumeIntoTrainer)
}

func TestMultiGpuTrainJobGranite34bCodeInstructLoRa(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_granite_34b_code_instruct_lora.json", 4)
}

func TestMultiGpuTrainJobMetaLlama318b(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_meta_llama3_1_8b.json", 2)
}

func TestMultiGpuTrainJobMetaLlama38bInstruct(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_meta_llama3_8b_instruct.json", 2)
}

func TestMultiGpuTrainJobMetaLlama370bInstructGptqBlue(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_meta_llama3_70b_instruct_gptq_blue.json", 2, mountModelVolumeIntoTrainer)
}

func TestMultiGpuTrainJobMetaLlama31405bGptq(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_meta_llama3_1_405b_gptq.json", 8, mountModelVolumeIntoTrainer)
}

func TestMultiGpuTrainJobMetaLlama3170bLoRa(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_meta_llama3_1_70b_lora.json", 4)
}

func TestMultiGpuTrainJobMetaLlama370bInstructLoRa(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_meta_llama3_70b_instruct_lora.json", 4)
}

func TestMultiGpuTrainJobMistral7bv03Gptq(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_mistral_7b_v03_gptq.json", 2, mountModelVolumeIntoTrainer)
}

func TestMultiGpuTrainJobMistral7bv03(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_mistral_7b_v03.json", 2)
}

func TestMultiGpuTrainJobMixtral8x7bv01(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_mixtral_8x7b_v01.json", 8)
}

func TestMultiGpuTrainJobMixtral8x7bInstructv01Gptq(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_mixtral_8x7b_instruct_v01_gptq.json", 2, mountModelVolumeIntoTrainer)
}

func TestMultiGpuTrainJobMixtral8x7bInstructv01LoRa(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_mixtral_8x7b_instruct_v01_lora.json", 4)
}

func TestMultiGpuTrainJobMerlinite7b(t *testing.T) {
	runMultiGpuTrainJob(t, "resources/config_merlinite_7b.json", 2)
}

func runMultiGpuTrainJob(t *testing.T, modelConfigFile string, numberOfGpus int, options ...Option[*trainerv1alpha1.TrainJob]) {
	test := With(t)

	namespace := test.CreateOrGetTestNamespace().Name

	// Create TrainingRuntime for FMS SFT training (multi-GPU, uses /mnt/scratch paths)
	runtime := createMultiGpuTrainingRuntime(test, namespace, numberOfGpus)
	defer test.Client().Trainer().TrainerV1alpha1().TrainingRuntimes(namespace).Delete(test.Ctx(), runtime.Name, metav1.DeleteOptions{})

	// Create a ConfigMap with configuration
	configData := map[string][]byte{
		"config.json": fms.ReadFile(test, modelConfigFile),
	}
	config := CreateConfigMap(test, namespace, configData)
	defer test.Client().Core().CoreV1().ConfigMaps(namespace).Delete(test.Ctx(), config.Name, *metav1.NewDeleteOptions(0))

	// Create PVC for trained model
	outputPvc := CreatePersistentVolumeClaim(test, namespace, "200Gi", AccessModes(corev1.ReadWriteOnce))
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), outputPvc.Name, metav1.DeleteOptions{})

	// Create training TrainJob (without Kueue integration)
	trainJob := createAlpacaTrainJob(test, namespace, runtime.Name, *config, numberOfGpus, outputPvc.Name, "", options...)
	defer test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Delete(test.Ctx(), trainJob.Name, *metav1.NewDeleteOptions(0))

	if IsOpenShift(test) {
		// Check that GPUs were utilized recently
		// That itself doesn't guarantee that TrainJob generated the load in GPU, but is the best we can achieve for now
		test.Eventually(openShiftPrometheusGpuUtil(test, namespace), 60*time.Minute).
			Should(
				And(
					HaveLen(numberOfGpus),
					ContainElement(
						// Check that at lest some GPU was utilized on more than 50%
						HaveField("Value", BeNumerically(">", 50)),
					),
				),
			)
	}

	// Make sure the TrainJob succeed
	test.Eventually(TrainJob(test, namespace, trainJob.Name), 60*time.Minute).Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s ran successfully", trainJob.Namespace, trainJob.Name)

	_, bucketUploadSet := fms.GetStorageBucketUploadName()
	if bucketUploadSet {
		fms.UploadToS3(test, namespace, outputPvc.Name, "model")
	}
}

func createAlpacaTrainJob(test Test, namespace, runtimeName string, config corev1.ConfigMap, numberOfGpus int, outputPvc string, localQueueName string, options ...Option[*trainerv1alpha1.TrainJob]) *trainerv1alpha1.TrainJob {
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
				Env: []corev1.EnvVar{
					{
						Name:  "HF_TOKEN",
						Value: GetHuggingFaceToken(test),
					},
				},
				ResourcesPerNode: &corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("4"),
						corev1.ResourceMemory: resource.MustParse("32Gi"),
						"nvidia.com/gpu":      resource.MustParse(fmt.Sprint(numberOfGpus)),
					},
					Limits: corev1.ResourceList{
						"nvidia.com/gpu": resource.MustParse(fmt.Sprint(numberOfGpus)),
					},
				},
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
									Ephemeral: &corev1.EphemeralVolumeSource{
										VolumeClaimTemplate: &corev1.PersistentVolumeClaimTemplate{
											Spec: corev1.PersistentVolumeClaimSpec{
												AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
												Resources: corev1.VolumeResourceRequirements{
													Requests: corev1.ResourceList{
														corev1.ResourceStorage: resource.MustParse("2000Gi"),
													},
												},
												VolumeMode: Ptr(corev1.PersistentVolumeFilesystem),
											},
										},
									},
								},
							},
							{
								Name: "output-volume",
								VolumeSource: corev1.VolumeSource{
									PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
										ClaimName: outputPvc,
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, option := range options {
		test.Expect(option.ApplyTo(trainJob)).To(Succeed())
	}

	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(test.Ctx(), trainJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created TrainJob %s/%s successfully", createdTrainJob.Namespace, createdTrainJob.Name)

	return createdTrainJob
}

func openShiftPrometheusGpuUtil(test Test, namespace string) func(g Gomega) prometheusmodel.Vector {
	return func(g Gomega) prometheusmodel.Vector {
		prometheusApiClient := GetOpenShiftPrometheusApiClient(test)
		result, warnings, err := prometheusApiClient.Query(test.Ctx(), "DCGM_FI_DEV_GPU_UTIL", time.Now(), prometheusapiv1.WithTimeout(5*time.Second))
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(warnings).Should(HaveLen(0))

		var util prometheusmodel.Vector
		for _, sample := range result.(prometheusmodel.Vector) {
			if string(sample.Metric["exported_namespace"]) == namespace {
				util = append(util, sample)
			}
		}

		return util
	}
}

var mountModelVolumeIntoTrainer = ErrorOption[*trainerv1alpha1.TrainJob](func(to *trainerv1alpha1.TrainJob) error {
	pvcName, err := fms.GetGptqModelPvcName()
	if err != nil {
		return err
	}

	modelVolume := corev1.Volume{
		Name: "model-volume",
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	}

	modelVolumeMount := corev1.VolumeMount{
		Name:      "model-volume",
		MountPath: "/mnt/model",
	}

	// Find the trainer/node pod template override and add the volume and volume mount
	for i := range to.Spec.PodTemplateOverrides {
		for _, target := range to.Spec.PodTemplateOverrides[i].TargetJobs {
			if target.Name == "node" && to.Spec.PodTemplateOverrides[i].Spec != nil {
				to.Spec.PodTemplateOverrides[i].Spec.Volumes = append(to.Spec.PodTemplateOverrides[i].Spec.Volumes, modelVolume)

				// Find the node container and add the volume mount
				for j := range to.Spec.PodTemplateOverrides[i].Spec.Containers {
					if to.Spec.PodTemplateOverrides[i].Spec.Containers[j].Name == "node" {
						to.Spec.PodTemplateOverrides[i].Spec.Containers[j].VolumeMounts = append(
							to.Spec.PodTemplateOverrides[i].Spec.Containers[j].VolumeMounts,
							modelVolumeMount,
						)
						break
					}
				}
				break
			}
		}
	}

	return nil
})

// createMultiGpuTrainingRuntime creates a TrainingRuntime for multi-GPU FMS SFT training
// Uses /mnt/scratch paths for dataset and temp files (matching KFTO GPU tests pattern)
func createMultiGpuTrainingRuntime(test Test, namespace string, numberOfGpus int) *trainerv1alpha1.TrainingRuntime {
	test.T().Helper()

	runtime := &trainerv1alpha1.TrainingRuntime{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "fms-sft-gpu-runtime-",
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
															Name:      "scratch-volume",
															MountPath: "/mnt/scratch",
														},
													},
													Command: []string{"/bin/sh", "-c"},
													Args:    []string{"mkdir -p /mnt/scratch/dataset; cp /dataset/alpaca_data_hundredth.json /mnt/scratch/dataset/alpaca_data.json"},
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
															Value: "/mnt/scratch/huggingface-home",
														},
														{
															Name:  "HOME",
															Value: "/mnt/scratch/home",
														},
														{
															Name:  "TRITON_HOME",
															Value: "/mnt/scratch/triton-home",
														},
														{
															Name:  "TRITON_CACHE_DIR",
															Value: "/mnt/scratch/triton-cache",
														},
														{
															Name:  "TMPDIR",
															Value: "/mnt/scratch",
														},
													},
													VolumeMounts: []corev1.VolumeMount{
														{
															Name:      "scratch-volume",
															MountPath: "/mnt/scratch",
														},
													},
													Resources: corev1.ResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("4"),
															corev1.ResourceMemory: resource.MustParse("32Gi"),
															"nvidia.com/gpu":      resource.MustParse(fmt.Sprintf("%d", numberOfGpus)),
														},
														Limits: corev1.ResourceList{
															"nvidia.com/gpu": resource.MustParse(fmt.Sprintf("%d", numberOfGpus)),
														},
													},
													SecurityContext: &corev1.SecurityContext{
														RunAsNonRoot:           Ptr(true),
														ReadOnlyRootFilesystem: Ptr(true),
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

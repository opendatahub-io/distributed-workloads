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

package kfto

import (
	"fmt"
	"testing"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestPyTorchJobSingleNodeSingleGpuWithCudaPyTorch241(t *testing.T) {
	Tags(t, Tier1, Gpu(NVIDIA))
	runKFTOPyTorchJob(t, GetTrainingCudaPyTorch241Image(), NVIDIA, 1, 0)
}

func TestPyTorchJobSingleNodeSingleGpuWithCudaPyTorch251(t *testing.T) {
	Tags(t, Tier1, Gpu(NVIDIA))
	runKFTOPyTorchJob(t, GetTrainingCudaPyTorch251Image(), NVIDIA, 1, 0)
}

func TestPyTorchJobSingleNodeMultiGpuWithCudaPyTorch241(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchJob(t, GetTrainingCudaPyTorch241Image(), NVIDIA, 2, 0)
}

func TestPyTorchJobSingleNodeMultiGpuWithCudaPyTorch251(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchJob(t, GetTrainingCudaPyTorch251Image(), NVIDIA, 2, 0)
}

func TestPyTorchJobMultiNodeSingleGpuWithCudaPyTorch241(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchJob(t, GetTrainingCudaPyTorch241Image(), NVIDIA, 1, 1)
}

func TestPyTorchJobMultiNodeSingleGpuWithCudaPyTorch251(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchJob(t, GetTrainingCudaPyTorch251Image(), NVIDIA, 1, 1)
}

func TestPyTorchJobMultiNodeMultiGpuWithCudaPyTorch241(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchJob(t, GetTrainingCudaPyTorch241Image(), NVIDIA, 2, 1)
}

func TestPyTorchJobMultiNodeMultiGpuWithCudaPyTorch251(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchJob(t, GetTrainingCudaPyTorch251Image(), NVIDIA, 2, 1)
}

func TestPyTorchJobSingleNodeSingleGpuWithROCmPyTorch241(t *testing.T) {
	Tags(t, Tier1, Gpu(AMD))
	runKFTOPyTorchJob(t, GetTrainingROCmPyTorch241Image(), AMD, 1, 0)
}

func TestPyTorchJobSingleNodeSingleGpuWithROCmPyTorch251(t *testing.T) {
	Tags(t, Tier1, Gpu(AMD))
	runKFTOPyTorchJob(t, GetTrainingROCmPyTorch251Image(), AMD, 1, 0)
}

func TestPyTorchJobSingleNodeMultiGpuWithROCmPyTorch241(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchJob(t, GetTrainingROCmPyTorch241Image(), AMD, 2, 0)
}

func TestPyTorchJobSingleNodeMultiGpuWithROCmPyTorch251(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchJob(t, GetTrainingROCmPyTorch251Image(), AMD, 2, 0)
}

func TestPyTorchJobMultiNodeSingleGpuWithROCmPyTorch241(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchJob(t, GetTrainingROCmPyTorch241Image(), AMD, 1, 1)
}

func TestPyTorchJobMultiNodeSingleGpuWithROCmPyTorch251(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchJob(t, GetTrainingROCmPyTorch251Image(), AMD, 1, 1)
}

func TestPyTorchJobMultiNodeMultiGpuWithROCmPyTorch241(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchJob(t, GetTrainingROCmPyTorch241Image(), AMD, 2, 1)
}

func TestPyTorchJobMultiNodeMultiGpuWithROCmPyTorch251(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchJob(t, GetTrainingROCmPyTorch251Image(), AMD, 2, 1)
}

func runKFTOPyTorchJob(t *testing.T, image string, gpu Accelerator, numGpus, numberOfWorkerNodes int) {
	test := With(t)

	// Create a namespace
	namespace := test.CreateOrGetTestNamespace().Name

	// Create Kueue resources
	resourceFlavor := CreateKueueResourceFlavor(test, v1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := v1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []v1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory")},
				Flavors: []v1beta1.FlavorQuotas{
					{
						Name: v1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []v1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("8"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("32Gi"),
							},
						},
					},
				},
			},
		},
	}

	if gpu.IsGpu() {
		numberOfGpus := (numberOfWorkerNodes + 1) * numGpus
		cqSpec.ResourceGroups[0].CoveredResources = append(
			cqSpec.ResourceGroups[0].CoveredResources,
			corev1.ResourceName(gpu.ResourceLabel),
		)
		cqSpec.ResourceGroups[0].Flavors[0].Resources = append(
			cqSpec.ResourceGroups[0].Flavors[0].Resources,
			v1beta1.ResourceQuota{
				Name:         corev1.ResourceName(gpu.ResourceLabel),
				NominalQuota: resource.MustParse(fmt.Sprint(numberOfGpus)),
			},
		)
	}

	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, namespace, clusterQueue.Name, AsDefaultQueue)

	// Create a ConfigMap with training script
	configData := map[string][]byte{
		"hf_llm_training.py": readFile(test, "resources/hf_llm_training.py"),
	}
	config := CreateConfigMap(test, namespace, configData)

	// Create PVC for trained model
	outputPvc := CreatePersistentVolumeClaim(test, namespace, "10Gi", AccessModes(corev1.ReadWriteOnce))
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), outputPvc.Name, metav1.DeleteOptions{})

	// Create training PyTorch job
	tuningJob := createKFTOPyTorchJob(test, namespace, *config, gpu, numGpus, numberOfWorkerNodes, outputPvc.Name, image, localQueue)
	defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Delete(test.Ctx(), tuningJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the PyTorch job is running
	test.Eventually(PyTorchJob(test, namespace, tuningJob.Name), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Make sure the PyTorch job succeeded
	test.Eventually(PyTorchJob(test, namespace, tuningJob.Name), TestTimeoutLong).Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)
}

func createKFTOPyTorchJob(test Test, namespace string, config corev1.ConfigMap, gpu Accelerator, numGpus, numberOfWorkerNodes int, outputPvcName string, baseImage string, localQueue *v1beta1.LocalQueue) *kftov1.PyTorchJob {
	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kfto-llm-",
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": localQueue.Name,
			},
		},
		Spec: kftov1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kftov1.ReplicaType]*kftov1.ReplicaSpec{
				kftov1.PyTorchJobReplicaTypeMaster: {
					Replicas:      Ptr(int32(1)),
					RestartPolicy: "OnFailure",
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app": "kfto-llm",
							},
						},
						Spec: corev1.PodSpec{
							Affinity: &corev1.Affinity{
								PodAntiAffinity: &corev1.PodAntiAffinity{
									RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
										{
											LabelSelector: &metav1.LabelSelector{
												MatchLabels: map[string]string{
													"app": "kfto-llm",
												},
											},
											TopologyKey: "kubernetes.io/hostname",
										},
									},
								},
							},
							Tolerations: []corev1.Toleration{
								{
									Key:      gpu.ResourceLabel,
									Operator: corev1.TolerationOpExists,
								},
							},
							InitContainers: []corev1.Container{
								{
									Name:            "copy-model",
									Image:           GetBloomModelImage(),
									ImagePullPolicy: corev1.PullIfNotPresent,
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
									},
									Command: []string{"/bin/sh", "-c"},
									Args:    []string{"mkdir /tmp/model; cp -r /models/bloom-560m /tmp/model"},
								},
								{
									Name:            "copy-dataset",
									Image:           GetAlpacaDatasetImage(),
									ImagePullPolicy: corev1.PullIfNotPresent,
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
									},
									Command: []string{"/bin/sh", "-c"},
									Args:    []string{"mkdir /tmp/all_datasets; cp -r /dataset/* /tmp/all_datasets;ls /tmp/all_datasets"},
								},
							},
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           baseImage,
									ImagePullPolicy: corev1.PullIfNotPresent,
									Command: []string{
										"/bin/bash", "-c",
										`torchrun /etc/config/hf_llm_training.py \
										--model_uri /tmp/model/bloom-560m \
										--model_dir /tmp/model/bloom-560m \
										--dataset_file /tmp/all_datasets/alpaca_data_hundredth.json \
										--transformer_type AutoModelForCausalLM \
										--training_parameters '{"output_dir": "/mnt/output", "per_device_train_batch_size": 8, "num_train_epochs": 3, "logging_dir": "/tmp/logs", "eval_strategy": "epoch", "save_strategy": "no"}' \
										--lora_config '{"r": 4, "lora_alpha": 16, "lora_dropout": 0.1, "bias": "none"}'`,
									},
									Env: []corev1.EnvVar{
										{
											Name:  "HF_HOME",
											Value: "/tmp/.cache",
										},
										{
											Name:  "TRITON_CACHE_DIR",
											Value: "/tmp/.triton",
										},
										{
											Name:  "TOKENIZERS_PARALLELISM",
											Value: "false",
										},
										{
											Name:  "NCCL_DEBUG",
											Value: "INFO",
										},
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "config-volume",
											MountPath: "/etc/config",
										},
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
										{
											Name:      "output-volume",
											MountPath: "/mnt/output",
										},
										{
											Name:      "shm-volume",
											MountPath: "/dev/shm",
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:                     resource.MustParse("2"),
											corev1.ResourceMemory:                  resource.MustParse("8Gi"),
											corev1.ResourceName(gpu.ResourceLabel): resource.MustParse(fmt.Sprint(numGpus)),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:                     resource.MustParse("2"),
											corev1.ResourceMemory:                  resource.MustParse("12Gi"),
											corev1.ResourceName(gpu.ResourceLabel): resource.MustParse(fmt.Sprint(numGpus)),
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
									Name: "tmp-volume",
									VolumeSource: corev1.VolumeSource{
										EmptyDir: &corev1.EmptyDirVolumeSource{},
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
								{
									Name: "shm-volume",
									VolumeSource: corev1.VolumeSource{
										EmptyDir: &corev1.EmptyDirVolumeSource{
											Medium: corev1.StorageMediumMemory,
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
	// Declaring worker replicas separately, if worker replica is declared with number of pods 0 then operator keeps creating and deleting worker pods
	if numberOfWorkerNodes > 0 {
		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeWorker] = &kftov1.ReplicaSpec{
			Replicas:      Ptr(int32(numberOfWorkerNodes)),
			RestartPolicy: "OnFailure",
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": "kfto-llm",
					},
				},
				Spec: corev1.PodSpec{
					Affinity: &corev1.Affinity{
						PodAntiAffinity: &corev1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{
											"app": "kfto-llm",
										},
									},
									TopologyKey: "kubernetes.io/hostname",
								},
							},
						},
					},
					Tolerations: []corev1.Toleration{
						{
							Key:      gpu.ResourceLabel,
							Operator: corev1.TolerationOpExists,
						},
					},
					InitContainers: []corev1.Container{
						{
							Name:            "copy-model",
							Image:           GetBloomModelImage(),
							ImagePullPolicy: corev1.PullIfNotPresent,
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "tmp-volume",
									MountPath: "/tmp",
								},
							},
							Command: []string{"/bin/sh", "-c"},
							Args:    []string{"mkdir /tmp/model; cp -r /models/bloom-560m /tmp/model"},
						},
						{
							Name:            "copy-dataset",
							Image:           GetAlpacaDatasetImage(),
							ImagePullPolicy: corev1.PullIfNotPresent,
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "tmp-volume",
									MountPath: "/tmp",
								},
							},
							Command: []string{"/bin/sh", "-c"},
							Args:    []string{"mkdir /tmp/all_datasets; cp -r /dataset/* /tmp/all_datasets;ls /tmp/all_datasets"},
						},
					},
					Containers: []corev1.Container{
						{
							Name:            "pytorch",
							Image:           baseImage,
							ImagePullPolicy: corev1.PullIfNotPresent,
							Command: []string{
								"/bin/bash", "-c",
								`torchrun /etc/config/hf_llm_training.py \
							--model_uri /tmp/model/bloom-560m \
							--model_dir /tmp/model/bloom-560m \
							--dataset_file /tmp/all_datasets/alpaca_data_hundredth.json \
							--transformer_type AutoModelForCausalLM \
							--training_parameters '{"output_dir": "/mnt/output", "per_device_train_batch_size": 8, "num_train_epochs": 3, "logging_dir": "/logs", "eval_strategy": "epoch", "save_strategy": "no"}' \
							--lora_config '{"r": 4, "lora_alpha": 16, "lora_dropout": 0.1, "bias": "none"}'`,
							},
							Env: []corev1.EnvVar{
								{
									Name:  "HF_HOME",
									Value: "/tmp/.cache",
								},
								{
									Name:  "TRITON_CACHE_DIR",
									Value: "/tmp/.triton",
								},
								{
									Name:  "TOKENIZERS_PARALLELISM",
									Value: "false",
								},
								{
									Name:  "NCCL_DEBUG",
									Value: "INFO",
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "config-volume",
									MountPath: "/etc/config",
								},
								{
									Name:      "tmp-volume",
									MountPath: "/tmp",
								},
								{
									Name:      "shm-volume",
									MountPath: "/dev/shm",
								},
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:                     resource.MustParse("2"),
									corev1.ResourceMemory:                  resource.MustParse("8Gi"),
									corev1.ResourceName(gpu.ResourceLabel): resource.MustParse(fmt.Sprint(numGpus)),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:                     resource.MustParse("2"),
									corev1.ResourceMemory:                  resource.MustParse("12Gi"),
									corev1.ResourceName(gpu.ResourceLabel): resource.MustParse(fmt.Sprint(numGpus)),
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
							Name: "tmp-volume",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
						{
							Name: "shm-volume",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{
									Medium: corev1.StorageMediumMemory,
								},
							},
						},
					},
				},
			},
		}
	}

	tuningJob, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

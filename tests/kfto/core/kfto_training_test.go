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

package core

import (
	"fmt"
	"os"
	"testing"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
)

func TestPyTorchJobWithCuda(t *testing.T) {
	test := With(t)
	cudaBaseImage := GetCudaTrainingImage(test)
	gpuLabel := "nvidia.com/gpu"
	runKFTOPyTorchJob(t, cudaBaseImage, gpuLabel, 1)
}

func TestPyTorchJobWithROCm(t *testing.T) {
	test := With(t)
	rocmBaseImage := GetROCmTrainingImage(test)
	gpuLabel := "amd.com/gpu"
	runKFTOPyTorchJob(t, rocmBaseImage, gpuLabel, 1)
}

func runKFTOPyTorchJob(t *testing.T, image string, gpuLabel string, numGpus int) {
	test := With(t)

	// Create a namespace
	namespace := GetOrCreateTestNamespace(test)

	// Parse training script
	trainingScriptPath := "hf_llm_training.py"
	trainingScript, err := os.ReadFile(trainingScriptPath)
	if err != nil {
		test.T().Fatalf("Error reading training script file: %v", err)
	}

	// Create a ConfigMap with training script
	configData := map[string][]byte{
		"hf_llm_training.py": trainingScript,
	}
	config := CreateConfigMap(test, namespace, configData)

	// Create PVC for trained model
	outputPvc := CreatePersistentVolumeClaim(test, namespace, "10Gi", corev1.ReadWriteOnce)
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), outputPvc.Name, metav1.DeleteOptions{})

	// Create training PyTorch job
	tuningJob := createKFTOPyTorchJob(test, namespace, *config, gpuLabel, numGpus, outputPvc.Name, image)
	defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Delete(test.Ctx(), tuningJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the PyTorch job is running
	test.Eventually(PyTorchJob(test, namespace, tuningJob.Name), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Make sure the PyTorch job succeeded
	test.Eventually(PyTorchJob(test, namespace, tuningJob.Name), TestTimeoutDouble).Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)

}

func createKFTOPyTorchJob(test Test, namespace string, config corev1.ConfigMap, gpuLabel string, numGpus int, outputPvcName string, baseImage string) *kftov1.PyTorchJob {
	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kfto-llm-",
		},
		Spec: kftov1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kftov1.ReplicaType]*kftov1.ReplicaSpec{
				"Master": {
					Replicas:      Ptr(int32(1)),
					RestartPolicy: "OnFailure",
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Tolerations: []corev1.Toleration{
								{
									Key:      gpuLabel,
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
									Image:           "registry.access.redhat.com/ubi9/python-311:9.5-1730564330",
									ImagePullPolicy: corev1.PullIfNotPresent,
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "data-volume",
											MountPath: "/tmp/dataset",
										},
									},
									Command: []string{
										"/bin/sh",
										"-c",
										`pip install --target /tmp/.local datasets && \
									HF_HOME=/tmp/.cache PYTHONPATH=/tmp/.local python -c "from datasets import load_dataset; dataset = load_dataset('tatsu-lab/alpaca', split='train[:100]'); dataset.save_to_disk('/tmp/dataset')"`,
									},
									Env: []corev1.EnvVar{
										{
											Name:  "HF_HOME",
											Value: "/tmp/.cache",
										},
									},
								},
							},
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           baseImage,
									ImagePullPolicy: corev1.PullIfNotPresent,
									Command: []string{
										"/bin/bash", "-c",
										`export HF_HOME=/tmp/.cache && \
										export TRITON_CACHE_DIR=/tmp/.triton && \
										export TOKENIZERS_PARALLELISM=false && \
										export RANK=0 && \
										export WORLD_SIZE=1 && \
										python /etc/config/hf_llm_training.py \
										--model_uri /tmp/model/bloom-560m \
										--model_dir /tmp/model/bloom-560m \
										--dataset_dir /tmp/dataset \
										--transformer_type AutoModelForCausalLM \
										--training_parameters '{"output_dir": "/mnt/output", "per_device_train_batch_size": 8, "num_train_epochs": 3, "logging_dir": "/logs", "eval_strategy": "epoch"}' \
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
											Name:  "RANK",
											Value: "0",
										},
										{
											Name:  "WORLD_SIZE",
											Value: "1",
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
											Name:      "data-volume",
											MountPath: "tmp/dataset",
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:            resource.MustParse("2"),
											corev1.ResourceMemory:         resource.MustParse("16Gi"),
											corev1.ResourceName(gpuLabel): resource.MustParse(fmt.Sprint(numGpus)),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:            resource.MustParse("2"),
											corev1.ResourceMemory:         resource.MustParse("16Gi"),
											corev1.ResourceName(gpuLabel): resource.MustParse(fmt.Sprint(numGpus)),
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
									Name: "data-volume",
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
											ClaimName: outputPvcName,
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

	tuningJob, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

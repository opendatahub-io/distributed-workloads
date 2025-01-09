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
	"bytes"
	"fmt"
	"testing"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPyTorchJobMnistMultiNodeSingleCpu(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 0, 2, 1, "", GetCudaTrainingImage(), "resources/requirements.txt")
}
func TestPyTorchJobMnistMultiNodeMultiCpu(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 0, 2, 2, "", GetCudaTrainingImage(), "resources/requirements.txt")
}

func TestPyTorchJobMnistMultiNodeSingleGpuWithCuda(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 1, 1, 1, "nvidia.com/gpu", GetCudaTrainingImage(), "resources/requirements.txt")
}

func TestPyTorchJobMnistMultiNodeMultiGpuWithCuda(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 2, 1, 2, "nvidia.com/gpu", GetCudaTrainingImage(), "resources/requirements.txt")
}

func TestPyTorchJobMnistMultiNodeSingleGpuWithROCm(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 1, 1, 1, "amd.com/gpu", GetROCmTrainingImage(), "resources/requirements-rocm.txt")
}

func TestPyTorchJobMnistMultiNodeMultiGpuWithROCm(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 2, 1, 2, "amd.com/gpu", GetROCmTrainingImage(), "resources/requirements-rocm.txt")
}

func runKFTOPyTorchMnistJob(t *testing.T, totalNumGpus int, workerReplicas int, numProcPerNode int, gpuLabel string, image string, requirementsFile string) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	mnist := ReadFile(test, "resources/mnist.py")
	requirementsFileName := ReadFile(test, requirementsFile)

	if totalNumGpus > 0 {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"gpu\""), 1)
	} else {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"cpu\""), 1)
	}
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		// MNIST Ray Notebook
		"mnist.py":         mnist,
		"requirements.txt": requirementsFileName,
	})

	outputPvc := CreatePersistentVolumeClaim(test, namespace.Name, "50Gi", corev1.ReadWriteOnce)
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace.Name).Delete(test.Ctx(), outputPvc.Name, metav1.DeleteOptions{})

	// Create training PyTorch job
	tuningJob := createKFTOPyTorchMnistJob(test, namespace.Name, *config, gpuLabel, totalNumGpus, workerReplicas, numProcPerNode, outputPvc.Name, image)
	defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace.Name).Delete(test.Ctx(), tuningJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the PyTorch job is running
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Make sure the PyTorch job succeeded
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)

}

func createKFTOPyTorchMnistJob(test Test, namespace string, config corev1.ConfigMap, gpuLabel string, totalNumGpus int, workerReplicas int, numProcPerNode int, outputPvcName string, baseImage string) *kftov1.PyTorchJob {
	var useGPU = false
	var backend string

	if totalNumGpus > 0 {
		useGPU = true
		backend = "nccl"
	} else {
		backend = "gloo"
	}

	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kfto-mnist-",
		},
		Spec: kftov1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kftov1.ReplicaType]*kftov1.ReplicaSpec{
				kftov1.PyTorchJobReplicaTypeMaster: {
					Replicas:      Ptr(int32(1)),
					RestartPolicy: kftov1.RestartPolicyOnFailure,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app":  "kfto-mnist",
								"role": "master",
							},
						},
						Spec: corev1.PodSpec{
							Affinity: &corev1.Affinity{
								PodAntiAffinity: &corev1.PodAntiAffinity{
									RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
										{
											LabelSelector: &metav1.LabelSelector{
												MatchLabels: map[string]string{
													"app": "kfto-mnist",
												},
											},
											TopologyKey: "kubernetes.io/hostname",
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
										fmt.Sprintf(`mkdir -p /tmp/lib && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
										pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib && \
										torchrun --nproc_per_node=%d /mnt/files/mnist.py --epochs 7 --save_every 2 --batch_size 64 --lr 0.0005 --snapshot_path "mnist_snapshot.pt" --backend %s`, numProcPerNode, backend),
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      config.Name,
											MountPath: "/mnt/files",
										},
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
										{
											Name:      "output-volume",
											MountPath: "/mnt/output",
										},
									},
									Resources: corev1.ResourceRequirements{
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("6Gi"),
										},
									},
								},
							},
							Volumes: []corev1.Volume{
								{
									Name: config.Name,
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
							},
							RestartPolicy: corev1.RestartPolicyOnFailure,
						},
					},
				},
				kftov1.PyTorchJobReplicaTypeWorker: {
					Replicas:      Ptr(int32(workerReplicas)),
					RestartPolicy: kftov1.RestartPolicyOnFailure,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app":  "kfto-mnist",
								"role": "worker",
							},
						},
						Spec: corev1.PodSpec{
							Affinity: &corev1.Affinity{
								PodAntiAffinity: &corev1.PodAntiAffinity{
									RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
										{
											LabelSelector: &metav1.LabelSelector{
												MatchLabels: map[string]string{
													"app": "kfto-mnist",
												},
											},
											TopologyKey: "kubernetes.io/hostname",
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
										fmt.Sprintf(`mkdir -p /tmp/lib && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
										pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib && \
										torchrun --nproc_per_node=%d /mnt/files/mnist.py --epochs 7 --save_every 2 --batch_size 64 --lr 0.0005 --snapshot_path "mnist_snapshot.pt" --backend %s`, numProcPerNode, backend),
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      config.Name,
											MountPath: "/mnt/files",
										},
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
									},
									Resources: corev1.ResourceRequirements{
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("6Gi"),
										},
									},
								},
							},
							Volumes: []corev1.Volume{
								{
									Name: config.Name,
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
							},
							RestartPolicy: corev1.RestartPolicyOnFailure,
						},
					},
				},
			},
		},
	}

	if useGPU {
		// Update resource lists for GPU (NVIDIA/ROCm) usecase
		tuningJob.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Containers[0].Resources.Limits[corev1.ResourceName(gpuLabel)] = resource.MustParse(fmt.Sprint(totalNumGpus))
		tuningJob.Spec.PyTorchReplicaSpecs["Worker"].Template.Spec.Containers[0].Resources.Limits[corev1.ResourceName(gpuLabel)] = resource.MustParse(fmt.Sprint(totalNumGpus))

		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Command = []string{
			"/bin/bash", "-c",
			fmt.Sprintf(`mkdir -p /tmp/lib && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
			pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib && \
			torchrun --nproc_per_node=%d /mnt/files/mnist.py --epochs 7 --save_every 2 --batch_size 128 --lr 0.001 --snapshot_path "mnist_snapshot.pt" --backend %s`, numProcPerNode, backend),
		}
		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.MPIJobReplicaTypeWorker].Template.Spec.Containers[0].Command = []string{
			"/bin/bash", "-c",
			fmt.Sprintf(`mkdir -p /tmp/lib && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
			pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib && \
			torchrun --nproc_per_node=%d /mnt/files/mnist.py --epochs 7 --save_every 2 --batch_size 128 --lr 0.001 --snapshot_path "mnist_snapshot.pt" --backend %s`, numProcPerNode, backend),
		}

		tuningJob.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Containers[0].Env = []corev1.EnvVar{
			{
				Name:  "NCCL_DEBUG",
				Value: "INFO",
			},
			{
				Name:  "TORCH_DISTRIBUTED_DEBUG",
				Value: "DETAIL",
			},
		}
		tuningJob.Spec.PyTorchReplicaSpecs["Worker"].Template.Spec.Containers[0].Env = []corev1.EnvVar{
			{
				Name:  "NCCL_DEBUG",
				Value: "INFO",
			},
			{
				Name:  "TORCH_DISTRIBUTED_DEBUG",
				Value: "DETAIL",
			},
		}

		// Update tolerations
		tuningJob.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Tolerations = []corev1.Toleration{
			{
				Key:      gpuLabel,
				Operator: corev1.TolerationOpExists,
			},
		}
		tuningJob.Spec.PyTorchReplicaSpecs["Worker"].Template.Spec.Tolerations = []corev1.Toleration{
			{
				Key:      gpuLabel,
				Operator: corev1.TolerationOpExists,
			},
		}
	}

	tuningJob, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

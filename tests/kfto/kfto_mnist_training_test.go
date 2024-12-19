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
	"os"
	"testing"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPyTorchJobMnistCpu(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 0, 2, "", GetCudaTrainingImage(), "resources/requirements.txt")
}

func TestPyTorchJobMnistWithCuda(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 1, 1, "nvidia.com/gpu", GetCudaTrainingImage(), "resources/requirements.txt")
}

func TestPyTorchJobMnistWithROCm(t *testing.T) {
	runKFTOPyTorchMnistJob(t, 1, 1, "amd.com/gpu", GetROCmTrainingImage(), "resources/requirements-rocm.txt")
}

func runKFTOPyTorchMnistJob(t *testing.T, numGpus int, workerReplicas int, gpuLabel string, image string, requirementsFile string) {
	test := With(t)

	storageClasses, err := test.Client().Core().StorageV1().StorageClasses().List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "Failed to list StorageClasses")

	// Verify at least one StorageClass supports RWX
	foundRWX := false
	var storageClassWithRWX storagev1.StorageClass
	for _, sc := range storageClasses.Items {
		// Check the allowed access modes in the StorageClass annotations
		if checkStorageClassSupportsRWX(sc) {
			foundRWX = true
			storageClassWithRWX = sc
			break
		}
	}
	test.Expect(foundRWX).To(BeTrue(), "No StorageClass found with RWX access mode")

	// Create a namespace
	namespace := test.NewTestNamespace()

	workingDirectory, err := os.Getwd()
	test.Expect(err).ToNot(HaveOccurred())

	mnist, err := os.ReadFile(workingDirectory + "/resources/mnist.py")
	test.Expect(err).ToNot(HaveOccurred())

	requirementsFileName, err := os.ReadFile(workingDirectory + "/" + requirementsFile)
	if numGpus > 0 {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"gpu\""), 1)
	} else {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"cpu\""), 1)
	}
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		// MNIST Ray Notebook
		"mnist.py":         mnist,
		"requirements.txt": requirementsFileName,
	})

	outputPvc := CreatePersistentVolumeClaimWithStorageClass(test, namespace.Name, storageClassWithRWX, "50Gi", corev1.ReadWriteMany)
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace.Name).Delete(test.Ctx(), outputPvc.Name, metav1.DeleteOptions{})

	// Create training PyTorch job
	tuningJob := createKFTOPyTorchMnistJob(test, namespace.Name, *config, gpuLabel, numGpus, workerReplicas, outputPvc.Name, image)
	defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace.Name).Delete(test.Ctx(), tuningJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the PyTorch job is running
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Make sure the PyTorch job succeeded
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)

}

func createKFTOPyTorchMnistJob(test Test, namespace string, config corev1.ConfigMap, gpuLabel string, numGpus int, workerReplicas int, outputPvcName string, baseImage string) *kftov1.PyTorchJob {
	var useGPU = false
	var backend string

	if numGpus > 0 {
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
				"Master": {
					Replicas:      Ptr(int32(1)),
					RestartPolicy: kftov1.RestartPolicyOnFailure,
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           baseImage,
									ImagePullPolicy: corev1.PullIfNotPresent,
									Command: []string{
										"/bin/bash", "-c",
										fmt.Sprintf(`mkdir -p /tmp/lib && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
										pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib && \
										python /mnt/files/mnist.py --epochs 1 --save-model --output-path /mnt/output --backend %s`, backend),
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
				"Worker": {
					Replicas:      Ptr(int32(workerReplicas)),
					RestartPolicy: kftov1.RestartPolicyOnFailure,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app": "kfto-mnist",
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
										python /mnt/files/mnist.py --epochs 1 --save-model --output-path /mnt/output --backend %s`, backend),
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
			},
		},
	}

	if useGPU {
		// Update resource lists
		tuningJob.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Containers[0].Resources = corev1.ResourceRequirements{
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:            resource.MustParse("2"),
				corev1.ResourceMemory:         resource.MustParse("8Gi"),
				corev1.ResourceName(gpuLabel): resource.MustParse(fmt.Sprint(numGpus)),
			},
		}
		tuningJob.Spec.PyTorchReplicaSpecs["Worker"].Template.Spec.Containers[0].Resources = corev1.ResourceRequirements{
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:            resource.MustParse("2"),
				corev1.ResourceMemory:         resource.MustParse("8Gi"),
				corev1.ResourceName(gpuLabel): resource.MustParse(fmt.Sprint(numGpus)),
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

func checkStorageClassSupportsRWX(sc storagev1.StorageClass) bool {
	// Provisioners like nfs.csi.k8s.io or kubernetes.io/nfs usually support RWX by default.
	if sc.Provisioner == "nfs.csi.k8s.io" || sc.Provisioner == "kubernetes.io/nfs" {
		return true
	}
	return false
}

func CreatePersistentVolumeClaimWithStorageClass(t Test, namespace string, storageClass storagev1.StorageClass, storageSize string, accessMode ...corev1.PersistentVolumeAccessMode) *corev1.PersistentVolumeClaim {
	t.T().Helper()

	pvc := &corev1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    namespace,
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: accessMode,
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse(storageSize),
				},
			},
			StorageClassName: &storageClass.Name,
		},
	}
	pvc, err := t.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Create(t.Ctx(), pvc, metav1.CreateOptions{})
	t.Expect(err).NotTo(HaveOccurred())
	t.T().Logf("Created PersistentVolumeClaim %s/%s successfully", pvc.Namespace, pvc.Name)

	return pvc
}

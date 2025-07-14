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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestPyTorchJobMnistMultiNodeSingleCpu(t *testing.T) {
	Tags(t, Sanity, MultiNode(3))
	runKFTOPyTorchMnistJob(t, CPU, GetTrainingCudaPyTorch251Image(), "resources/requirements.txt", 2, 1)
}

func TestPyTorchJobMnistMultiNodeMultiCpu(t *testing.T) {
	Tags(t, Tier1, MultiNode(3))
	runKFTOPyTorchMnistJob(t, CPU, GetTrainingCudaPyTorch251Image(), "resources/requirements.txt", 2, 2)
}

func TestPyTorchJobMnistMultiNodeSingleGpuWithCudaPyTorch241(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchMnistJob(t, NVIDIA, GetTrainingCudaPyTorch241Image(), "resources/requirements.txt", 1, 1)
}

func TestPyTorchJobMnistMultiNodeSingleGpuWithCudaPyTorch251(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchMnistJob(t, NVIDIA, GetTrainingCudaPyTorch251Image(), "resources/requirements.txt", 1, 1)
}

func TestPyTorchJobMnistMultiNodeMultiGpuWithCudaPyTorch241(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchMnistJob(t, NVIDIA, GetTrainingCudaPyTorch241Image(), "resources/requirements.txt", 1, 2)
}

func TestPyTorchJobMnistMultiNodeMultiGpuWithCudaPyTorch251(t *testing.T) {
	Tags(t, KftoCuda)
	runKFTOPyTorchMnistJob(t, NVIDIA, GetTrainingCudaPyTorch251Image(), "resources/requirements.txt", 1, 2)
}

func TestPyTorchJobMnistMultiNodeSingleGpuWithROCmPyTorch241(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchMnistJob(t, AMD, GetTrainingROCmPyTorch241Image(), "resources/requirements-rocm.txt", 1, 1)
}

func TestPyTorchJobMnistMultiNodeSingleGpuWithROCmPyTorch251(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchMnistJob(t, AMD, GetTrainingROCmPyTorch251Image(), "resources/requirements-rocm.txt", 1, 1)
}

func TestPyTorchJobMnistMultiNodeMultiGpuWithROCmPyTorch241(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchMnistJob(t, AMD, GetTrainingROCmPyTorch241Image(), "resources/requirements-rocm.txt", 1, 2)
}

func TestPyTorchJobMnistMultiNodeMultiGpuWithROCmPyTorch251(t *testing.T) {
	Tags(t, KftoRocm)
	runKFTOPyTorchMnistJob(t, AMD, GetTrainingROCmPyTorch251Image(), "resources/requirements-rocm.txt", 1, 2)
}

func runKFTOPyTorchMnistJob(t *testing.T, accelerator Accelerator, image string, requirementsFile string, workerReplicas, numProcPerNode int) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	mnist := readFile(test, "resources/mnist.py")
	download_mnist_dataset := readFile(test, "resources/download_mnist_datasets.py")
	requirementsFileName := readFile(test, requirementsFile)

	if accelerator.IsGpu() {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"gpu\""), 1)
	} else {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"cpu\""), 1)
	}
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		"mnist.py":                   mnist,
		"download_mnist_datasets.py": download_mnist_dataset,
		"requirements.txt":           requirementsFileName,
	})

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
								NominalQuota: resource.MustParse("18Gi"),
							},
						},
					},
				},
			},
		},
	}

	if accelerator.IsGpu() {
		numGpus := (workerReplicas + 1) * numProcPerNode
		cqSpec.ResourceGroups[0].CoveredResources = append(
			cqSpec.ResourceGroups[0].CoveredResources,
			corev1.ResourceName(accelerator.ResourceLabel),
		)
		cqSpec.ResourceGroups[0].Flavors[0].Resources = append(
			cqSpec.ResourceGroups[0].Flavors[0].Resources,
			v1beta1.ResourceQuota{
				Name:         corev1.ResourceName(accelerator.ResourceLabel),
				NominalQuota: resource.MustParse(fmt.Sprint(numGpus)),
			},
		)
	}

	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, namespace.Name, clusterQueue.Name, AsDefaultQueue)

	// Create training PyTorch job
	tuningJob := createKFTOPyTorchMnistJob(test, namespace.Name, *config, accelerator, workerReplicas, numProcPerNode, image, localQueue)
	defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace.Name).Delete(test.Ctx(), tuningJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the Workload is created and running
	test.Eventually(KueueWorkloads(test, namespace.Name), TestTimeoutMedium).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("Workload failed to be admitted"))),
			),
		)

	// Make sure the PyTorch job is running
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Make sure the PyTorch job succeeded
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)

}

func createKFTOPyTorchMnistJob(test Test, namespace string, config corev1.ConfigMap, accelerator Accelerator, workerReplicas int, numProcPerNode int, baseImage string, localQueue *v1beta1.LocalQueue) *kftov1.PyTorchJob {
	var backend string
	if accelerator.IsGpu() {
		backend = "nccl"
	} else {
		backend = "gloo"
	}

	storage_bucket_endpoint, storage_bucket_endpoint_exists := GetStorageBucketDefaultEndpoint()
	storage_bucket_access_key_id, storage_bucket_access_key_id_exists := GetStorageBucketAccessKeyId()
	storage_bucket_secret_key, storage_bucket_secret_key_exists := GetStorageBucketSecretKey()
	storage_bucket_name, storage_bucket_name_exists := GetStorageBucketName()
	storage_bucket_mnist_dir, storage_bucket_mnist_dir_exists := GetStorageBucketMnistDir()

	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kfto-mnist-",
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": localQueue.Name,
			},
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
										fmt.Sprintf(`mkdir -p /tmp/lib /tmp/datasets/mnist && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
										pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib --verbose &&  \
										echo "Downloading MNIST dataset..." && \
										python3 /mnt/files/download_mnist_datasets.py --dataset_path "/tmp/datasets/mnist" && \
										echo -e "\n\n Dataset downloaded to /tmp/datasets/mnist" && ls -R /tmp/datasets/mnist && \
										echo -e "\n\n Starting training..." && \
										torchrun --nproc_per_node=%d /mnt/files/mnist.py --dataset_path "/tmp/datasets/mnist" --epochs 3 --save_every 2 --batch_size 128 --lr 0.001 --snapshot_path "mnist_snapshot.pt" --backend %s`, numProcPerNode, backend),
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
											Name:      "shm-volume",
											MountPath: "/dev/shm",
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("4Gi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("4Gi"),
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
									Name: "shm-volume",
									VolumeSource: corev1.VolumeSource{
										EmptyDir: &corev1.EmptyDirVolumeSource{
											Medium: corev1.StorageMediumMemory,
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
										fmt.Sprintf(`mkdir -p /tmp/lib /tmp/datasets/mnist && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
										pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib --verbose && \
										echo "Downloading MNIST dataset..." && \
										python3 /mnt/files/download_mnist_datasets.py --dataset_path "/tmp/datasets/mnist" && \
										echo -e "\n\n Dataset downloaded to /tmp/datasets/mnist" && ls -R /tmp/datasets/mnist && \
										echo -e "\n\n Starting training..." && \
										torchrun --nproc_per_node=%d /mnt/files/mnist.py --dataset_path "/tmp/datasets/mnist" --epochs 3 --save_every 2 --batch_size 128 --lr 0.001 --snapshot_path "mnist_snapshot.pt" --backend %s`, numProcPerNode, backend),
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
											Name:      "shm-volume",
											MountPath: "/dev/shm",
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("4Gi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("4Gi"),
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
									Name: "shm-volume",
									VolumeSource: corev1.VolumeSource{
										EmptyDir: &corev1.EmptyDirVolumeSource{
											Medium: corev1.StorageMediumMemory,
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

	// Add PIP Index to download python packages, use provided custom PYPI mirror index url in case of disconnected environemnt
	tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Env = []corev1.EnvVar{
		{
			Name:  "PIP_INDEX_URL",
			Value: GetPipIndexURL(),
		},
		{
			Name:  "PIP_TRUSTED_HOST",
			Value: GetPipTrustedHost(),
		},
	}
	tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeWorker].Template.Spec.Containers[0].Env = []corev1.EnvVar{
		{
			Name:  "PIP_INDEX_URL",
			Value: GetPipIndexURL(),
		},
		{
			Name:  "PIP_TRUSTED_HOST",
			Value: GetPipTrustedHost(),
		},
	}

	if accelerator.IsGpu() {
		// Update resource lists for GPU (NVIDIA/ROCm) usecase
		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Resources.Requests[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(fmt.Sprint(numProcPerNode))
		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Resources.Limits[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(fmt.Sprint(numProcPerNode))
		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeWorker].Template.Spec.Containers[0].Resources.Requests[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(fmt.Sprint(numProcPerNode))
		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeWorker].Template.Spec.Containers[0].Resources.Limits[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(fmt.Sprint(numProcPerNode))

		torch_distributed_debug_env_vars := []corev1.EnvVar{
			{
				Name:  "NCCL_DEBUG",
				Value: "INFO",
			},
			{
				Name:  "TORCH_DISTRIBUTED_DEBUG",
				Value: "DETAIL",
			},
		}
		for _, envVar := range torch_distributed_debug_env_vars {
			tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Env = upsert(tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Env, envVar, withEnvVarName(envVar.Name))
			tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeWorker].Template.Spec.Containers[0].Env = upsert(tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Env, envVar, withEnvVarName(envVar.Name))
		}

		// Update tolerations
		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Tolerations = []corev1.Toleration{
			{
				Key:      accelerator.ResourceLabel,
				Operator: corev1.TolerationOpExists,
			},
		}
		tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeWorker].Template.Spec.Tolerations = []corev1.Toleration{
			{
				Key:      accelerator.ResourceLabel,
				Operator: corev1.TolerationOpExists,
			},
		}
	}

	// Use storage bucket to download the MNIST datasets if required environment variables are provided, else use default MNIST mirror references as the fallback
	if storage_bucket_endpoint_exists && storage_bucket_access_key_id_exists && storage_bucket_secret_key_exists && storage_bucket_name_exists && storage_bucket_mnist_dir_exists {
		storage_bucket_env_vars := []corev1.EnvVar{
			{
				Name:  "AWS_DEFAULT_ENDPOINT",
				Value: storage_bucket_endpoint,
			},
			{
				Name:  "AWS_ACCESS_KEY_ID",
				Value: storage_bucket_access_key_id,
			},
			{
				Name:  "AWS_SECRET_ACCESS_KEY",
				Value: storage_bucket_secret_key,
			},
			{
				Name:  "AWS_STORAGE_BUCKET",
				Value: storage_bucket_name,
			},
			{
				Name:  "AWS_STORAGE_BUCKET_MNIST_DIR",
				Value: storage_bucket_mnist_dir,
			},
		}

		// Append the list of environment variables for the worker container
		for _, envVar := range storage_bucket_env_vars {
			tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Env = upsert(tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeMaster].Template.Spec.Containers[0].Env, envVar, withEnvVarName(envVar.Name))
			tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeWorker].Template.Spec.Containers[0].Env = upsert(tuningJob.Spec.PyTorchReplicaSpecs[kftov1.PyTorchJobReplicaTypeWorker].Template.Spec.Containers[0].Env, envVar, withEnvVarName(envVar.Name))
		}

	} else {
		test.T().Logf("Skipped usage of S3 storage bucket, because required environment variables aren't provided!\nRequired environment variables : AWS_DEFAULT_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_STORAGE_BUCKET, AWS_STORAGE_BUCKET_MNIST_DIR")
	}

	tuningJob, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

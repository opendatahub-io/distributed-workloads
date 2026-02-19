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
	"testing"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"
	kueueacv1beta1 "sigs.k8s.io/kueue/client-go/applyconfiguration/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

var (
	namespaceName      = "test-kfto-upgrade"
	resourceFlavorName = "rf-upgrade"
	clusterQueueName   = "cq-upgrade"
	localQueueName     = "lq-upgrade"
	pyTorchJobName     = "pytorch-upgrade"
)

func TestSetupPytorchjob(t *testing.T) {
	Tags(t, PreUpgrade)
	test := With(t)

	SetupKueue(test, initialKueueState, PyTorchJobFramework)

	// Create a namespace with Kueue labeled
	CreateOrGetTestNamespaceWithName(test, namespaceName, WithKueueManaged())
	test.T().Logf("Created Kueue-managed namespace: %s", namespaceName)

	// Create a ConfigMap with training dataset and configuration
	mnist := readFile(test, "resources/mnist.py")
	download_mnist_dataset := readFile(test, "resources/download_mnist_datasets.py")
	requirementsFileName := readFile(test, "resources/requirements.txt")

	configData := map[string][]byte{
		"mnist.py":                   mnist,
		"download_mnist_datasets.py": download_mnist_dataset,
		"requirements.txt":           requirementsFileName,
	}

	config := CreateConfigMap(test, namespaceName, configData)

	// Create Kueue resources
	resourceFlavor := kueueacv1beta1.ResourceFlavor(resourceFlavorName)
	appliedResourceFlavor, err := test.Client().Kueue().KueueV1beta1().ResourceFlavors().Apply(test.Ctx(), resourceFlavor, metav1.ApplyOptions{FieldManager: "setup-PyTorchJob", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ResourceFlavor %s successfully", appliedResourceFlavor.Name)

	clusterQueue := kueueacv1beta1.ClusterQueue(clusterQueueName).WithSpec(
		kueueacv1beta1.ClusterQueueSpec().
			WithNamespaceSelector(metav1.LabelSelector{}).
			WithResourceGroups(
				kueueacv1beta1.ResourceGroup().WithCoveredResources(
					corev1.ResourceName("cpu"), corev1.ResourceName("memory"),
				).WithFlavors(
					kueueacv1beta1.FlavorQuotas().
						WithName(kueuev1beta1.ResourceFlavorReference(resourceFlavorName)).
						WithResources(
							kueueacv1beta1.ResourceQuota().WithName(corev1.ResourceCPU).WithNominalQuota(resource.MustParse("8")),
							kueueacv1beta1.ResourceQuota().WithName(corev1.ResourceMemory).WithNominalQuota(resource.MustParse("18Gi")),
						),
				),
			).
			WithStopPolicy(kueuev1beta1.Hold),
	)
	appliedClusterQueue, err := test.Client().Kueue().KueueV1beta1().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "setup-PyTorchJob", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ClusterQueue %s successfully", appliedClusterQueue.Name)

	localQueue := kueueacv1beta1.LocalQueue(localQueueName, namespaceName).
		WithAnnotations(map[string]string{"kueue.x-k8s.io/default-queue": "true"}).
		WithSpec(
			kueueacv1beta1.LocalQueueSpec().WithClusterQueue(kueuev1beta1.ClusterQueueReference(clusterQueueName)),
		)
	appliedLocalQueue, err := test.Client().Kueue().KueueV1beta1().LocalQueues(namespaceName).Apply(test.Ctx(), localQueue, metav1.ApplyOptions{FieldManager: "setup-PyTorchJob", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue LocalQueue %s/%s successfully", appliedLocalQueue.Namespace, appliedLocalQueue.Name)

	// Create training PyTorch job
	tuningJob := createUpgradePyTorchJob(test, namespaceName, appliedLocalQueue.Name, *config)

	// Make sure the PyTorch job is suspended, waiting for ClusterQueue to be enabled
	test.Eventually(PyTorchJob(test, tuningJob.Namespace, pyTorchJobName), TestTimeoutShort).
		Should(WithTransform(PyTorchJobConditionSuspended, Equal(corev1.ConditionTrue)))
}

func TestRunPytorchjob(t *testing.T) {
	Tags(t, PostUpgrade)
	test := With(t)
	namespace := GetNamespaceWithName(test, namespaceName)

	// Cleanup everything in the end
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavorName, metav1.DeleteOptions{})
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueueName, metav1.DeleteOptions{})
	defer DeleteTestNamespace(test, namespace)

	// Enable ClusterQueue to process waiting PyTorchJob
	clusterQueue := kueueacv1beta1.ClusterQueue(clusterQueueName).WithSpec(kueueacv1beta1.ClusterQueueSpec().WithStopPolicy(kueuev1beta1.None))
	_, err := test.Client().Kueue().KueueV1beta1().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "application/apply-patch", Force: true})
	test.Expect(err).NotTo(HaveOccurred())

	// PyTorch job should be started now
	test.Eventually(PyTorchJob(test, namespaceName, pyTorchJobName), TestTimeoutLong).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Make sure the PyTorch job succeed
	test.Eventually(PyTorchJob(test, namespaceName, pyTorchJobName), TestTimeoutLong).
		Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
}

func createUpgradePyTorchJob(test Test, namespace, localQueueName string, config corev1.ConfigMap) *kftov1.PyTorchJob {
	// Does PyTorchJob already exist?
	_, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Get(test.Ctx(), pyTorchJobName, metav1.GetOptions{})
	if err == nil {
		// If yes then delete it and wait until there are no PyTorchJobs in the namespace
		err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Delete(test.Ctx(), pyTorchJobName, metav1.DeleteOptions{})
		test.Expect(err).NotTo(HaveOccurred())
		test.Eventually(PyTorchJobs(test, namespace), TestTimeoutShort).Should(BeEmpty())
	} else if !errors.IsNotFound(err) {
		test.T().Fatalf("Error retrieving PyTorchJob with name `%s`: %v", pyTorchJobName, err)
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
			Name: pyTorchJobName,
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": localQueueName,
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
									Image:           GetTrainingCudaPyTorch251Image(),
									ImagePullPolicy: corev1.PullIfNotPresent,
									Command: []string{
										"/bin/bash", "-c",
										(`mkdir -p /tmp/lib /tmp/datasets/mnist && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
										pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib --verbose &&  \
										echo "Downloading MNIST dataset..." && \
										python3 /mnt/files/download_mnist_datasets.py --dataset_path "/tmp/datasets/mnist" && \
										echo -e "\n\n Dataset downloaded to /tmp/datasets/mnist" && ls -R /tmp/datasets/mnist && \
										echo -e "\n\n Starting training..." && \
										torchrun --nproc_per_node 2 /mnt/files/mnist.py --dataset_path "/tmp/datasets/mnist" --epochs 7 --save_every 2 --batch_size 128 --lr 0.001 --snapshot_path "mnist_snapshot.pt" --backend "gloo"`),
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
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("2"),
											corev1.ResourceMemory: resource.MustParse("6Gi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("2"),
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
		}
	} else {
		test.T().Logf("Skipped usage of S3 storage bucket, because required environment variables aren't provided!\nRequired environment variables : AWS_DEFAULT_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_STORAGE_BUCKET, AWS_STORAGE_BUCKET_MNIST_DIR")
	}

	tuningJob, err = test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

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
	"strings"
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"
	"sigs.k8s.io/kueue/apis/kueue/v1beta2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

func TestSftStockTrlSingleNodeSingleGPU(t *testing.T) {
	Tags(t, KftoCuda, Gpu(NVIDIA))
	runSftStockTrlTrainJob(t, 1, 1)
}

func TestSftStockTrlMultiNodeSingleGPU(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, NVIDIA))
	runSftStockTrlTrainJob(t, 2, 1)
}

func runSftStockTrlTrainJob(t *testing.T, numNodes, numProcPerNode int32) {
	test := With(t)
	baseImage, err := trainerutils.GetImageFromClusterTrainingRuntime(test, trainerutils.DefaultClusterTrainingRuntimeCUDA)
	test.Expect(err).ToNot(HaveOccurred(), "Failed to get image from ClusterTrainingRuntime: %v", err)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	namespace := test.NewTestNamespace(WithKueueManaged()).Name
	test.T().Logf("Created Kueue-managed namespace: %s", namespace)

	storageClass, err := GetRWXStorageClass(test)
	test.Expect(err).ToNot(HaveOccurred(), "Failed to find an RWX supporting StorageClass")

	pvc := CreatePersistentVolumeClaim(test, namespace, "20Gi", AccessModes(corev1.ReadWriteMany), StorageClassName(storageClass.Name))

	files := map[string][]byte{
		"sft_stock_trl.py":     readFile(test, "resources/sft_stock_trl.py"),
		"download_sft_data.py": readFile(test, "resources/download_sft_data.py"),
		"requirements.txt":     readFile(test, "resources/requirements-sft-stock-trl.txt"),
	}
	config := CreateConfigMap(test, namespace, files)

	resourceFlavor := CreateKueueResourceFlavor(test, v1beta2.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})

	numGpus := numNodes * numProcPerNode
	cqSpec := v1beta2.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []v1beta2.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{"cpu", "memory", corev1.ResourceName(NVIDIA.ResourceLabel)},
				Flavors: []v1beta2.FlavorQuotas{
					{
						Name: v1beta2.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []v1beta2.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("20"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("128Gi"),
							},
							{
								Name:         corev1.ResourceName(NVIDIA.ResourceLabel),
								NominalQuota: resource.MustParse(fmt.Sprint(numGpus)),
							},
						},
					},
				},
			},
		},
	}

	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, namespace, clusterQueue.Name, AsDefaultQueue)

	trainingRuntime := createSftStockTrlTrainingRuntime(test, namespace, config.Name, pvc.Name, baseImage, numProcPerNode)

	trainJob := createSftStockTrlTrainJob(test, namespace, trainingRuntime.Name, numNodes, numProcPerNode, localQueue.Name)

	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutMedium).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("Workload failed to be admitted"))),
			),
		)
	test.T().Log("Kueue Workload admitted")

	test.T().Logf("Verifying JobSet creation with replicated jobs...")
	test.Eventually(SingleJobSet(test, namespace), TestTimeoutDouble).Should(
		WithTransform(JobSetReplicatedJobsCount, Equal(2)),
	)
	test.T().Logf("JobSet created with 2 replicated jobs (dataset-initializer, node)")

	test.T().Logf("Checking JobSet status...")
	test.Eventually(SingleJobSet(test, namespace), TestTimeoutGpuProvisioning).Should(
		WithTransform(JobSetConditionCompleted, Equal(metav1.ConditionTrue)),
	)
	test.T().Logf("JobSet marked as completed")

	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutMedium).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s completed", namespace, trainJob.Name)
}

func createSftStockTrlTrainingRuntime(test Test, namespace, configMapName, pvcName, baseImage string, numProcPerNode int32) *trainerv1alpha1.TrainingRuntime {
	test.T().Helper()

	trainingRuntime := &trainerv1alpha1.TrainingRuntime{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-sft-stock-trl-runtime-",
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
echo "    SFT Stock TRL Dataset Initializer     "
echo "=========================================="

LOCAL_LIB=/tmp/pip_packages
mkdir -p ${LOCAL_LIB}
mkdir -p ${LIB_PATH}

echo "Installing dependencies from requirements.txt ..."
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# || "$line" =~ ^-- ]] && continue
    pkg=$(echo "$line" | sed 's/[[:space:]]*#.*//')
    echo "Installing $pkg ..."
    pip install --no-cache-dir "$pkg" --target=${LOCAL_LIB} --verbose
done < /mnt/scripts/requirements.txt

echo ""
echo "Copying installed packages to ${LIB_PATH}..."
cp -r ${LOCAL_LIB}/* ${LIB_PATH}/
echo "Dependencies installed successfully!"

export PYTHONPATH=${LIB_PATH}:$PYTHONPATH
python3 /mnt/scripts/download_sft_data.py

echo "Dataset initialization completed!"
`},
													Resources: corev1.ResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("2"),
															corev1.ResourceMemory: resource.MustParse("8Gi"),
														},
														Limits: corev1.ResourceList{
															corev1.ResourceCPU:    resource.MustParse("4"),
															corev1.ResourceMemory: resource.MustParse("16Gi"),
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
													Args: []string{`
set -e

echo "==================== Environment Info ===================="
echo "Model Path: ${MODEL_PATH}"
echo "Dataset Path: ${DATASET_PATH}"
echo "==========================================================="

if [ ! -f "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

export PYTHONPATH=${LIB_PATH}:$PYTHONPATH

echo "==================== Starting Stock TRL SFT Training ===================="
torchrun /mnt/scripts/sft_stock_trl.py

echo ""
echo "==================== Training Complete ===================="
`},
													Resources: corev1.ResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceCPU:                        resource.MustParse("4"),
															corev1.ResourceMemory:                     resource.MustParse("32Gi"),
															corev1.ResourceName(NVIDIA.ResourceLabel): resource.MustParse(fmt.Sprint(numProcPerNode)),
														},
														Limits: corev1.ResourceList{
															corev1.ResourceCPU:                        resource.MustParse("8"),
															corev1.ResourceMemory:                     resource.MustParse("48Gi"),
															corev1.ResourceName(NVIDIA.ResourceLabel): resource.MustParse(fmt.Sprint(numProcPerNode)),
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
														{
															Name:      "dshm",
															MountPath: "/dev/shm",
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
												{
													Name: "dshm",
													VolumeSource: corev1.VolumeSource{
														EmptyDir: &corev1.EmptyDirVolumeSource{
															Medium: corev1.StorageMediumMemory,
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
	test.T().Logf("Created TrainingRuntime %s/%s with dataset-initializer", runtime.Namespace, runtime.Name)

	return runtime
}

func createSftStockTrlTrainJob(test Test, namespace, runtimeName string, numNodes, numProcPerNode int32, queueName string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-sft-stock-trl-trainjob-",
			Namespace:    namespace,
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": queueName,
			},
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
							{Name: "MODEL_PATH", Value: "/workspace/model"},
							{Name: "LIB_PATH", Value: "/workspace/lib"},
						},
						sftStorageBucketEnvVars(test)...,
					),
				},
			},
			Trainer: &trainerv1alpha1.Trainer{
				NumNodes:       Ptr(numNodes),
				NumProcPerNode: Ptr(intstr.FromInt32(numProcPerNode)),
				Env: []corev1.EnvVar{
					{Name: "DATASET_PATH", Value: "/workspace/data/train_All_100.jsonl"},
					{Name: "MODEL_PATH", Value: "/workspace/model"},
					{Name: "OUTPUT_DIR", Value: "/workspace/output"},
					{Name: "LIB_PATH", Value: "/workspace/lib"},
					{Name: "NCCL_DEBUG", Value: "INFO"},
					{Name: "TORCH_DISTRIBUTED_DEBUG", Value: "DETAIL"},
				},
			},
		},
	}

	created, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create TrainJob")
	test.T().Logf("Created TrainJob %s/%s", created.Namespace, created.Name)

	return created
}

func sftStorageBucketEnvVars(test Test) []corev1.EnvVar {
	test.T().Helper()

	endpoint, endpointOK := GetStorageBucketDefaultEndpoint()
	accessKey, accessKeyOK := GetStorageBucketAccessKeyId()
	secretKey, secretKeyOK := GetStorageBucketSecretKey()
	bucket, bucketOK := GetStorageBucketName()
	sftDir, sftDirOK := GetStorageBucketSftDir()

	if endpointOK && accessKeyOK && secretKeyOK && bucketOK && sftDirOK {
		test.T().Log("S3/Minio configuration detected, adding storage environment variables to dataset-initializer")
		envVars := []corev1.EnvVar{
			{Name: "AWS_DEFAULT_ENDPOINT", Value: endpoint},
			{Name: "AWS_ACCESS_KEY_ID", Value: accessKey},
			{Name: "AWS_SECRET_ACCESS_KEY", Value: secretKey},
			{Name: "AWS_STORAGE_BUCKET", Value: bucket},
			{Name: "AWS_STORAGE_BUCKET_SFT_DIR", Value: sftDir},
		}
		if strings.HasPrefix(endpoint, "http://") {
			envVars = append(envVars, corev1.EnvVar{Name: "AWS_ALLOW_INSECURE_ENDPOINT", Value: "true"})
		}
		return envVars
	}

	test.T().Log("S3/Minio configuration incomplete, dataset-initializer will download from HuggingFace")
	return nil
}

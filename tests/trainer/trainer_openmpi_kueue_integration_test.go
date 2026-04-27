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
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

func TestOpenMPICudaTrainJobKueueIntegration(t *testing.T) {
	Tags(t, Sanity, KftoCuda, MultiNodeGpu(2, NVIDIA))
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	namespace := test.NewTestNamespace(WithKueueManaged()).Name
	test.T().Logf("Created Kueue-managed namespace: %s", namespace)

	configMap := CreateConfigMap(test, namespace, map[string][]byte{
		"openmpi_cuda_smoke.py": readFile(test, "resources/openmpi_cuda_smoke.py"),
	})

	resourceFlavor := CreateKueueResourceFlavor(test, kueuev1beta1.ResourceFlavorSpec{
		NodeLabels: map[string]string{
			"nvidia.com/gpu.present": "true",
		},
	})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})

	clusterQueue := CreateKueueClusterQueue(test, kueuev1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"kubernetes.io/metadata.name": namespace,
			},
		},
		ResourceGroups: []kueuev1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{
					corev1.ResourceCPU,
					corev1.ResourceMemory,
					corev1.ResourceName(NVIDIA.ResourceLabel),
				},
				Flavors: []kueuev1beta1.FlavorQuotas{
					{
						Name: kueuev1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []kueuev1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("4"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("16Gi"),
							},
							{
								Name:         corev1.ResourceName(NVIDIA.ResourceLabel),
								NominalQuota: resource.MustParse("2"),
							},
						},
					},
				},
			},
		},
	})
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})

	localQueue := CreateKueueLocalQueue(test, namespace, clusterQueue.Name)
	trainJob := createOpenMPICudaKueueTrainJob(test, namespace, localQueue.Name, configMap.Name)

	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutMedium).Should(
		And(
			HaveLen(1),
			ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("OpenMPI workload failed to be admitted"))),
			ContainElement(WithTransform(func(w *kueuev1beta1.Workload) string {
				return w.Spec.QueueName
			}, Equal(localQueue.Name))),
			ContainElement(WithTransform(openMPIPodSetNames, ConsistOf("launcher", "node"))),
		),
	)
	test.T().Log("OpenMPI Kueue Workload admitted with launcher and node pod sets")

	test.Eventually(SingleJobSet(test, namespace), TestTimeoutMedium).Should(
		WithTransform(JobSetReplicatedJobsCount, Equal(2)),
	)
	test.T().Log("JobSet created with launcher and node replicated jobs")

	test.Eventually(func(g Gomega) {
		launcherRunning, nodeRunning := openMPIRunningPodCounts(test, namespace, trainJob.Name)
		g.Expect(launcherRunning).To(Equal(1), "expected exactly one running launcher pod")
		g.Expect(nodeRunning).To(Equal(1), "expected exactly one running worker pod")
	}, TestTimeoutMedium).Should(Succeed())
	test.T().Log("Launcher and worker pods reached Running concurrently")

	var launcherLog string
	test.Eventually(func(g Gomega) string {
		launcherPod := openMPIPodByRole(test, namespace, trainJob.Name, "launcher")
		launcherLog = GetPodLog(test, namespace, launcherPod.Name, corev1.PodLogOptions{
			Container: launcherPod.Spec.Containers[0].Name,
		})
		g.Expect(launcherLog).NotTo(BeEmpty())
		return launcherLog
	}, TestTimeoutLong).Should(ContainSubstring("MPI CUDA allreduce succeeded"))
	test.T().Log("Launcher logs confirm successful MPI CUDA allreduce")

	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("OpenMPI TrainJob %s/%s completed successfully", namespace, trainJob.Name)
}

func createOpenMPICudaKueueTrainJob(test Test, namespace, queueName, configMapName string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-openmpi-kueue-trainjob-",
			Namespace:    namespace,
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": queueName,
			},
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": queueName,
			},
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA,
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command: []string{
					"/usr/local/bin/uid_entrypoint.sh",
					"mpirun",
					"python",
					"/mnt/scripts/openmpi_cuda_smoke.py",
				},
				NumNodes: Ptr(int32(2)),
				Env: []corev1.EnvVar{
					{Name: "MPI_TEST_HOLD_SECONDS", Value: "15"},
					{Name: "PYTHONUNBUFFERED", Value: "1"},
				},
				ResourcesPerNode: Ptr(corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU:                        resource.MustParse("2"),
						corev1.ResourceMemory:                     resource.MustParse("8Gi"),
						corev1.ResourceName(NVIDIA.ResourceLabel): resource.MustParse("1"),
					},
					Limits: corev1.ResourceList{
						corev1.ResourceCPU:                        resource.MustParse("2"),
						corev1.ResourceMemory:                     resource.MustParse("8Gi"),
						corev1.ResourceName(NVIDIA.ResourceLabel): resource.MustParse("1"),
					},
				}),
			},
			PodTemplateOverrides: []trainerv1alpha1.PodTemplateOverride{
				{
					TargetJobs: []trainerv1alpha1.PodTemplateOverrideTargetJob{
						{Name: "launcher"},
						{Name: "node"},
					},
					Spec: &trainerv1alpha1.PodTemplateSpecOverride{
						Containers: []trainerv1alpha1.ContainerOverride{
							{
								Name: "node",
								VolumeMounts: []corev1.VolumeMount{
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
										Medium:    corev1.StorageMediumMemory,
										SizeLimit: Ptr(resource.MustParse("8Gi")),
									},
								},
							},
						},
					},
				},
			},
		},
	}

	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create OpenMPI TrainJob")
	test.T().Logf("Created OpenMPI TrainJob %s/%s", createdTrainJob.Namespace, createdTrainJob.Name)

	return createdTrainJob
}

func openMPIPodSetNames(workload *kueuev1beta1.Workload) []string {
	if workload == nil {
		return nil
	}

	podSetNames := make([]string, 0, len(workload.Spec.PodSets))
	for _, podSet := range workload.Spec.PodSets {
		podSetNames = append(podSetNames, podSet.Name)
	}

	return podSetNames
}

func openMPIRunningPodCounts(test Test, namespace, trainJobName string) (int, int) {
	test.T().Helper()

	pods := GetPods(test, namespace, metav1.ListOptions{
		LabelSelector: "jobset.sigs.k8s.io/jobset-name=" + trainJobName,
	})

	launcherRunning := 0
	nodeRunning := 0
	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodRunning {
			continue
		}

		switch pod.Labels["jobset.sigs.k8s.io/replicatedjob-name"] {
		case "launcher":
			launcherRunning++
		case "node":
			nodeRunning++
		}
	}

	return launcherRunning, nodeRunning
}

func openMPIPodByRole(test Test, namespace, trainJobName, role string) corev1.Pod {
	test.T().Helper()

	pods := GetPods(test, namespace, metav1.ListOptions{
		LabelSelector: "jobset.sigs.k8s.io/jobset-name=" + trainJobName +
			",jobset.sigs.k8s.io/replicatedjob-name=" + role,
	})

	var selected *corev1.Pod
	for i := range pods {
		pod := pods[i]
		if pod.Status.Phase != corev1.PodRunning &&
			pod.Status.Phase != corev1.PodPending &&
			pod.Status.Phase != corev1.PodSucceeded {
			continue
		}
		if selected == nil {
			selected = &pod
			continue
		}
		if selected.Status.Phase != corev1.PodRunning && pod.Status.Phase == corev1.PodRunning {
			selected = &pod
			continue
		}
		if selected.Status.Phase == corev1.PodSucceeded && pod.Status.Phase == corev1.PodPending {
			selected = &pod
			continue
		}
		if selected.CreationTimestamp.Before(&pod.CreationTimestamp) {
			selected = &pod
		}
	}

	if selected == nil {
		test.T().Fatalf("No active pod found for TrainJob %s with role %s", trainJobName, role)
	}
	return *selected
}

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
	kueuev1beta2 "sigs.k8s.io/kueue/apis/kueue/v1beta2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

func TestOpenMPICudaTrainJobKueueIntegration(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, NVIDIA))
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	namespace := test.NewTestNamespace(WithKueueManaged()).Name
	test.T().Logf("Created Kueue-managed namespace: %s", namespace)

	configMap := CreateConfigMap(test, namespace, map[string][]byte{
		mpiCollectivesScript: readFile(test, "resources/"+mpiCollectivesScript),
	})

	resourceFlavor := CreateKueueResourceFlavor(test, kueuev1beta2.ResourceFlavorSpec{
		NodeLabels: map[string]string{
			"nvidia.com/gpu.present": "true",
		},
	})
	defer test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})

	clusterQueue := CreateKueueClusterQueue(test, kueuev1beta2.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"kubernetes.io/metadata.name": namespace,
			},
		},
		ResourceGroups: []kueuev1beta2.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{
					corev1.ResourceCPU,
					corev1.ResourceMemory,
					corev1.ResourceName(NVIDIA.ResourceLabel),
				},
				Flavors: []kueuev1beta2.FlavorQuotas{
					{
						Name: kueuev1beta2.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []kueuev1beta2.ResourceQuota{
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
	defer test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})

	localQueue := CreateKueueLocalQueue(test, namespace, clusterQueue.Name)
	trainJob := createOpenMPICudaKueueTrainJob(test, namespace, localQueue.Name, configMap.Name, false)

	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutMedium).Should(
		And(
			HaveLen(1),
			ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("OpenMPI workload failed to be admitted"))),
			ContainElement(WithTransform(func(w *kueuev1beta2.Workload) string {
				return string(w.Spec.QueueName)
			}, Equal(localQueue.Name))),
			ContainElement(WithTransform(openMPIPodSetNames, ConsistOf("launcher", "node"))),
		),
	)
	test.T().Log("OpenMPI Kueue Workload admitted with launcher and node pod sets")

	test.Eventually(SingleJobSet(test, namespace), TestTimeoutMedium).Should(
		WithTransform(JobSetReplicatedJobsCount, Equal(2)),
	)
	test.T().Log("JobSet created with launcher and node replicated jobs")

	expectedImage, err := trainerutils.GetImageFromClusterTrainingRuntime(test, trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA)
	test.Expect(err).NotTo(HaveOccurred())
	assertMPIPodLayout(test, namespace, trainJob.Name, expectedImage, "cuda")
	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutLong).
		Should(Satisfy(TrainJobReachedFinalState))
	launcherPod := openMPIPodByRole(test, namespace, trainJob.Name, "launcher")
	launcherLog := GetPodLog(test, namespace, launcherPod.Name, corev1.PodLogOptions{
		Container: "node",
	})
	finalJob := TrainJob(test, namespace, trainJob.Name)(test)
	test.Expect(finalJob).To(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)),
		"OpenMPI TrainJob failed: %s; launcher logs:\n%s", TrainJobFailedMessage(finalJob), launcherLog)
	assertMPICollectivesMarkers(test, launcherLog, "cuda")
	test.T().Logf("OpenMPI TrainJob %s/%s completed successfully", namespace, trainJob.Name)
}

func TestOpenMPICudaTrainJobKueueWorkloadDeactivateReactivate(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, NVIDIA))
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	namespace := test.NewTestNamespace(WithKueueManaged()).Name
	test.T().Logf("Created Kueue-managed namespace: %s", namespace)

	configMap := CreateConfigMap(test, namespace, map[string][]byte{
		mpiCollectivesScript: readFile(test, "resources/"+mpiCollectivesScript),
	})

	resourceFlavor := CreateKueueResourceFlavor(test, kueuev1beta2.ResourceFlavorSpec{
		NodeLabels: map[string]string{
			"nvidia.com/gpu.present": "true",
		},
	})
	defer test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})

	clusterQueue := CreateKueueClusterQueue(test, kueuev1beta2.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"kubernetes.io/metadata.name": namespace,
			},
		},
		ResourceGroups: []kueuev1beta2.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{
					corev1.ResourceCPU,
					corev1.ResourceMemory,
					corev1.ResourceName(NVIDIA.ResourceLabel),
				},
				Flavors: []kueuev1beta2.FlavorQuotas{
					{
						Name: kueuev1beta2.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []kueuev1beta2.ResourceQuota{
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
	defer test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})

	localQueue := CreateKueueLocalQueue(test, namespace, clusterQueue.Name)
	trainJob := createOpenMPICudaKueueTrainJob(test, namespace, localQueue.Name, configMap.Name, true)

	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutMedium).Should(
		And(
			HaveLen(1),
			ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("OpenMPI workload failed to be admitted"))),
			ContainElement(WithTransform(func(w *kueuev1beta2.Workload) string {
				return string(w.Spec.QueueName)
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
	expectedImage, err := trainerutils.GetImageFromClusterTrainingRuntime(test, trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA)
	test.Expect(err).NotTo(HaveOccurred())
	oldLauncher, oldWorker := assertMPIPodLayout(test, namespace, trainJob.Name, expectedImage, "cuda")

	workload := singleOpenMPIWorkload(test, namespace)
	workload.Spec.Active = Ptr(false)
	_, err = test.Client().Kueue().KueueV1beta2().Workloads(namespace).Update(
		test.Ctx(),
		workload,
		metav1.UpdateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to deactivate OpenMPI Workload")

	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutMedium).Should(
		WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionTrue)),
	)
	test.T().Logf("OpenMPI TrainJob %s/%s is suspended after workload deactivation", namespace, trainJob.Name)

	test.Eventually(Pods(test, namespace, metav1.ListOptions{}), TestTimeoutMedium).Should(HaveLen(0))
	test.T().Log("OpenMPI launcher and worker pods were removed after workload deactivation")

	workload = singleOpenMPIWorkload(test, namespace)
	workload.Spec.Active = Ptr(true)
	_, err = test.Client().Kueue().KueueV1beta2().Workloads(namespace).Update(
		test.Ctx(),
		workload,
		metav1.UpdateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to reactivate OpenMPI Workload")

	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutMedium).Should(
		WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionFalse)),
	)
	test.T().Logf("OpenMPI TrainJob %s/%s resumed after workload reactivation", namespace, trainJob.Name)

	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutMedium).Should(
		ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrue())),
	)
	test.Eventually(func(g Gomega) {
		launcherRunning, nodeRunning := openMPIRunningPodCounts(test, namespace, trainJob.Name)
		g.Expect(launcherRunning).To(Equal(1), "expected exactly one running launcher pod after resume")
		g.Expect(nodeRunning).To(Equal(1), "expected exactly one running worker pod after resume")
	}, TestTimeoutMedium).Should(Succeed())
	test.T().Log("OpenMPI launcher and worker pods are running again after workload reactivation")
	newLauncher, newWorker := assertMPIPodLayout(test, namespace, trainJob.Name, expectedImage, "cuda")
	test.Expect(newLauncher.UID).NotTo(Equal(oldLauncher.UID), "Launcher pod should be recreated")
	test.Expect(newWorker.UID).NotTo(Equal(oldWorker.UID), "Worker pod should be recreated")
	test.T().Log("OpenMPI launcher and worker pods were recreated after workload reactivation")
}

func createOpenMPICudaKueueTrainJob(test Test, namespace, queueName, configMapName string, holdUntilStopped bool) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	command := []string{
		"/usr/local/bin/uid_entrypoint.sh",
		"mpirun",
		"python",
		"/mnt/scripts/" + mpiCollectivesScript,
		"--device",
		"cuda",
	}
	if holdUntilStopped {
		command = append(command, "--hold-until-stopped")
	}

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-openmpi-kueue-trainjob-",
			Namespace:    namespace,
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": queueName,
			},
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA,
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command:  command,
				NumNodes: Ptr(int32(2)),
				Env:      append(mpiTestEnv(), corev1.EnvVar{Name: "PYTHONUNBUFFERED", Value: "1"}),
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
			RuntimePatches: []trainerv1alpha1.RuntimePatch{
				{
					Manager: "opendatahub.io/mpi-kueue-e2e",
					TrainingRuntimeSpec: &trainerv1alpha1.TrainingRuntimeSpecPatch{
						Template: &trainerv1alpha1.JobSetTemplatePatch{
							Metadata: &metav1.ObjectMeta{
								Labels: map[string]string{
									"kueue.x-k8s.io/queue-name": queueName,
								},
							},
							Spec: &trainerv1alpha1.JobSetSpecPatch{
								ReplicatedJobs: []trainerv1alpha1.ReplicatedJobPatch{
									{
										Name: "launcher",
										Template: &trainerv1alpha1.JobTemplatePatch{
											Spec: &trainerv1alpha1.JobSpecPatch{
												Template: &trainerv1alpha1.PodTemplatePatch{
													Spec: &trainerv1alpha1.PodSpecPatch{
														Affinity: mpiPodAntiAffinity(),
														Containers: []trainerv1alpha1.ContainerPatch{
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
									},
									{
										Name: "node",
										Template: &trainerv1alpha1.JobTemplatePatch{
											Spec: &trainerv1alpha1.JobSpecPatch{
												Template: &trainerv1alpha1.PodTemplatePatch{
													Spec: &trainerv1alpha1.PodSpecPatch{
														Affinity: mpiPodAntiAffinity(),
														Containers: []trainerv1alpha1.ContainerPatch{
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

func openMPIPodSetNames(workload *kueuev1beta2.Workload) []string {
	if workload == nil {
		return nil
	}

	podSetNames := make([]string, 0, len(workload.Spec.PodSets))
	for _, podSet := range workload.Spec.PodSets {
		podSetNames = append(podSetNames, string(podSet.Name))
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

func singleOpenMPIWorkload(test Test, namespace string) *kueuev1beta2.Workload {
	test.T().Helper()

	workloads := GetKueueWorkloads(test, namespace)
	test.Expect(workloads).To(HaveLen(1), "expected exactly one OpenMPI workload in namespace %s", namespace)
	return workloads[0]
}

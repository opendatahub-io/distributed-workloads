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

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const mpiCollectivesScript = "mpi_collectives.py"

func TestMultiNodeOpenMPICpuTrainJob(t *testing.T) {
	Tags(t, Tier1, MultiNode(2))
	runMPITrainJob(t, "cpu")
}

func TestMultiNodeOpenMPITrainJob(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeGpu(2, NVIDIA))
	runMPITrainJob(t, "cuda")
}

func runMPITrainJob(t *testing.T, deviceMode string) {
	test := With(t)
	namespace := test.NewTestNamespace().Name

	configMap := CreateConfigMap(test, namespace, map[string][]byte{
		mpiCollectivesScript: readFile(test, "resources/"+mpiCollectivesScript),
	})

	runtimeRef := trainerv1alpha1.RuntimeRef{Name: trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA}
	expectedImage, err := trainerutils.GetImageFromClusterTrainingRuntime(test, runtimeRef.Name)
	test.Expect(err).NotTo(HaveOccurred())
	if deviceMode == "cpu" {
		runtimeRef.Name, expectedImage = createCPUOpenMPIRuntime(test, namespace)
		runtimeRef.Kind = Ptr("TrainingRuntime")
	}

	trainJob := createMPITrainJob(test, namespace, configMap.Name, runtimeRef, deviceMode)
	test.Eventually(func(g Gomega) {
		pods := GetPods(test, namespace, metav1.ListOptions{
			LabelSelector: "jobset.sigs.k8s.io/jobset-name=" + trainJob.Name,
		})
		g.Expect(pods).To(HaveLen(2), "MPI launcher and worker pods should both exist")
		for _, pod := range pods {
			g.Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "%s pod should be running", pod.Name)
		}
	}, TestTimeoutMedium).Should(Succeed())
	launcherPod, _ := assertMPIPodLayout(test, namespace, trainJob.Name, expectedImage, deviceMode)

	test.Eventually(TrainJob(test, namespace, trainJob.Name), TestTimeoutDouble).
		Should(Satisfy(TrainJobReachedFinalState))

	logs := GetPodLog(test, namespace, launcherPod.Name, corev1.PodLogOptions{
		Container: "node",
	})

	finalJob := TrainJob(test, namespace, trainJob.Name)(test)
	test.Expect(finalJob).To(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)),
		"TrainJob %s/%s should complete; failure: %s; launcher logs:\n%s",
		namespace, trainJob.Name, TrainJobFailedMessage(finalJob), logs)

	jobset := SingleJobSet(test, namespace)(test)
	test.Expect(jobset).To(WithTransform(JobSetReplicatedJobsCount, Equal(2)),
		"MPI JobSet should have launcher and node replicated jobs")
	assertMPICollectivesMarkers(test, logs, deviceMode)
}

func createCPUOpenMPIRuntime(test Test, namespace string) (string, string) {
	test.T().Helper()

	source, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Get(
		test.Ctx(), trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA, metav1.GetOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to get OpenMPI runtime configuration")
	image, err := trainerutils.GetImageFromClusterTrainingRuntime(test, trainerutils.DefaultClusterTrainingRuntimeCPU)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to get universal CPU image")

	runtime := &trainerv1alpha1.TrainingRuntime{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-openmpi-cpu-",
			Namespace:    namespace,
			Labels: map[string]string{
				"trainer.kubeflow.org/framework": "openmpi",
			},
		},
		Spec: *source.Spec.DeepCopy(),
	}
	imageCount := 0
	for i := range runtime.Spec.Template.Spec.ReplicatedJobs {
		containers := runtime.Spec.Template.Spec.ReplicatedJobs[i].Template.Spec.Template.Spec.Containers
		for j := range containers {
			if containers[j].Name == "node" {
				containers[j].Image = image
				imageCount++
			}
		}
	}
	test.Expect(imageCount).To(Equal(2), "OpenMPI launcher and worker images must both be set")

	created, err := test.Client().Trainer().TrainerV1alpha1().TrainingRuntimes(namespace).Create(
		test.Ctx(), runtime, metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create CPU OpenMPI TrainingRuntime")
	return created.Name, image
}

func createMPITrainJob(test Test, namespace, configMapName string, runtimeRef trainerv1alpha1.RuntimeRef, deviceMode string) *trainerv1alpha1.TrainJob {
	test.T().Helper()

	resources := &corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("2"),
			corev1.ResourceMemory: resource.MustParse("8Gi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("2"),
			corev1.ResourceMemory: resource.MustParse("8Gi"),
		},
	}
	if deviceMode == "cuda" {
		gpu := corev1.ResourceName(NVIDIA.ResourceLabel)
		resources.Requests[gpu] = resource.MustParse("1")
		resources.Limits[gpu] = resource.MustParse("1")
	}

	replicatedJobs := make([]trainerv1alpha1.ReplicatedJobPatch, 0, 2)
	for _, role := range []string{"launcher", "node"} {
		replicatedJobs = append(replicatedJobs, trainerv1alpha1.ReplicatedJobPatch{
			Name: role,
			Template: &trainerv1alpha1.JobTemplatePatch{
				Spec: &trainerv1alpha1.JobSpecPatch{
					Template: &trainerv1alpha1.PodTemplatePatch{
						Spec: &trainerv1alpha1.PodSpecPatch{
							Affinity: mpiPodAntiAffinity(),
							Containers: []trainerv1alpha1.ContainerPatch{{
								Name: "node",
								VolumeMounts: []corev1.VolumeMount{{
									Name: "mpi-test-script", MountPath: "/etc/mpi-test", ReadOnly: true,
								}},
							}},
							Volumes: []corev1.Volume{{
								Name: "mpi-test-script",
								VolumeSource: corev1.VolumeSource{
									ConfigMap: &corev1.ConfigMapVolumeSource{
										LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
									},
								},
							}},
						},
					},
				},
			},
		})
	}

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-mpi-trainjob-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: runtimeRef,
			Trainer: &trainerv1alpha1.Trainer{
				NumNodes: Ptr(int32(2)),
				Command: []string{
					"mpirun", "python", "/etc/mpi-test/" + mpiCollectivesScript, "--device", deviceMode,
				},
				Env:              mpiTestEnv(),
				ResourcesPerNode: resources,
			},
			RuntimePatches: []trainerv1alpha1.RuntimePatch{{
				Manager: "opendatahub.io/mpi-e2e",
				TrainingRuntimeSpec: &trainerv1alpha1.TrainingRuntimeSpecPatch{
					Template: &trainerv1alpha1.JobSetTemplatePatch{
						Spec: &trainerv1alpha1.JobSetSpecPatch{ReplicatedJobs: replicatedJobs},
					},
				},
			}},
		},
	}

	created, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(), trainJob, metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create MPI TrainJob")
	return created
}

func mpiPodAntiAffinity() *corev1.Affinity {
	return &corev1.Affinity{
		PodAntiAffinity: &corev1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{{
				LabelSelector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{{
						Key: "jobset.sigs.k8s.io/jobset-name", Operator: metav1.LabelSelectorOpExists,
					}},
				},
				TopologyKey: "kubernetes.io/hostname",
			}},
		},
	}
}

func mpiTestEnv() []corev1.EnvVar {
	return []corev1.EnvVar{
		{Name: "HOME", Value: "/home/mpiuser"},
		{Name: "OMPI_MCA_pml", Value: "ob1"},
		{Name: "OMPI_MCA_btl", Value: "self,tcp"},
	}
}

func assertMPIPodLayout(test Test, namespace, trainJobName, expectedImage, deviceMode string) (corev1.Pod, corev1.Pod) {
	test.T().Helper()
	rolePod := func(role string) corev1.Pod {
		pods := GetPods(test, namespace, metav1.ListOptions{
			LabelSelector: "jobset.sigs.k8s.io/jobset-name=" + trainJobName +
				",jobset.sigs.k8s.io/replicatedjob-name=" + role,
		})
		test.Expect(pods).To(HaveLen(1), "Expected exactly one %s pod", role)
		pod := pods[0]
		test.Expect(pod.Spec.NodeName).NotTo(BeEmpty(), "%s pod is not scheduled", role)
		container := mpiNodeContainer(test, pod)
		test.Expect(container.Image).To(Equal(expectedImage), "%s pod uses the wrong runtime image", role)
		gpu := corev1.ResourceName(NVIDIA.ResourceLabel)
		if deviceMode == "cuda" {
			test.Expect(container.Resources.Requests[gpu]).To(Equal(resource.MustParse("1")))
			test.Expect(container.Resources.Limits[gpu]).To(Equal(resource.MustParse("1")))
		} else {
			test.Expect(container.Resources.Requests).NotTo(HaveKey(gpu))
			test.Expect(container.Resources.Limits).NotTo(HaveKey(gpu))
		}
		return pod
	}
	launcher := rolePod("launcher")
	worker := rolePod("node")
	test.Expect(launcher.Spec.NodeName).NotTo(Equal(worker.Spec.NodeName),
		"MPI launcher and worker must run on different nodes")
	return launcher, worker
}

func mpiNodeContainer(test Test, pod corev1.Pod) corev1.Container {
	test.T().Helper()
	for _, container := range pod.Spec.Containers {
		if container.Name == "node" {
			return container
		}
	}
	test.T().Fatalf("Pod %s has no OpenMPI node container", pod.Name)
	return corev1.Container{}
}

func assertMPICollectivesMarkers(test Test, logs, deviceMode string) {
	test.T().Helper()
	for _, rank := range []int{0, 1} {
		marker := fmt.Sprintf("MPI COLLECTIVES PASSED device=%s rank=%d world_size=2", deviceMode, rank)
		test.Expect(logs).To(ContainSubstring(marker), "Missing MPI success marker for rank %d", rank)
	}
}

/*
Copyright 2026.

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
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	kueuev1beta2 "sigs.k8s.io/kueue/apis/kueue/v1beta2"
	kueueacv1beta2 "sigs.k8s.io/kueue/client-go/applyconfiguration/kueue/v1beta2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

var (
	gpuUpgradeNamespaceName      = "test-kfto-kueue-gpu-upgrade"
	gpuUpgradeResourceFlavorName = "rf-kueue-gpu-upgrade"
	gpuUpgradeClusterQueueName   = "cq-kueue-gpu-upgrade"
	gpuUpgradeLocalQueueName     = "lq-kueue-gpu-upgrade"
	gpuUpgradePyTorchJobName     = "pytorch-kueue-gpu-upgrade"
	gpuUpgradeBaselineConfigMap  = "kueue-gpu-upgrade-baseline"

	gpuUpgradeResourceFlavorGenerationKey = "resourceflavor-generation"
	gpuUpgradeResourceFlavorSpecKey       = "resourceflavor-spec"
	gpuUpgradeClusterQueueGenerationKey   = "clusterqueue-generation"
	gpuUpgradeClusterQueueSpecKey         = "clusterqueue-spec"
)

func TestSetupGpuKueueUpgrade(t *testing.T) {
	Tags(t, PreUpgrade, Gpu(NVIDIA))
	test := With(t)

	SetupKueue(test, initialKueueState, PyTorchJobFramework)

	CreateOrGetTestNamespaceWithName(test, gpuUpgradeNamespaceName, WithKueueManaged())
	test.T().Logf("Created Kueue-managed namespace: %s", gpuUpgradeNamespaceName)

	resourceFlavor := kueueacv1beta2.ResourceFlavor(gpuUpgradeResourceFlavorName).WithSpec(
		kueueacv1beta2.ResourceFlavorSpec().
			WithNodeLabels(map[string]string{
				"nvidia.com/gpu.present": "true",
			}).
			WithTolerations(
				corev1ac.Toleration().
					WithKey("nvidia.com/gpu").
					WithOperator(corev1.TolerationOpExists).
					WithEffect(corev1.TaintEffectNoSchedule),
			),
	)
	appliedResourceFlavor, err := test.Client().Kueue().KueueV1beta2().ResourceFlavors().Apply(test.Ctx(), resourceFlavor, metav1.ApplyOptions{FieldManager: "setup-gpu-kueue-upgrade", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ResourceFlavor %s successfully", appliedResourceFlavor.Name)

	clusterQueue := kueueacv1beta2.ClusterQueue(gpuUpgradeClusterQueueName).WithSpec(
		kueueacv1beta2.ClusterQueueSpec().
			WithNamespaceSelector(&metav1ac.LabelSelectorApplyConfiguration{}).
			WithResourceGroups(
				kueueacv1beta2.ResourceGroup().WithCoveredResources(
					corev1.ResourceCPU, corev1.ResourceMemory, corev1.ResourceName(NVIDIA.ResourceLabel),
				).WithFlavors(
					kueueacv1beta2.FlavorQuotas().
						WithName(kueuev1beta2.ResourceFlavorReference(gpuUpgradeResourceFlavorName)).
						WithResources(
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceCPU).WithNominalQuota(resource.MustParse("4")),
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceMemory).WithNominalQuota(resource.MustParse("16Gi")),
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceName(NVIDIA.ResourceLabel)).WithNominalQuota(resource.MustParse("1")),
						),
				),
			).
			WithStopPolicy(kueuev1beta2.Hold),
	)
	appliedClusterQueue, err := test.Client().Kueue().KueueV1beta2().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "setup-gpu-kueue-upgrade", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ClusterQueue %s with StopPolicy=Hold successfully", appliedClusterQueue.Name)

	localQueue := kueueacv1beta2.LocalQueue(gpuUpgradeLocalQueueName, gpuUpgradeNamespaceName).
		WithAnnotations(map[string]string{"kueue.x-k8s.io/default-queue": "true"}).
		WithSpec(
			kueueacv1beta2.LocalQueueSpec().WithClusterQueue(kueuev1beta2.ClusterQueueReference(gpuUpgradeClusterQueueName)),
		)
	appliedLocalQueue, err := test.Client().Kueue().KueueV1beta2().LocalQueues(gpuUpgradeNamespaceName).Apply(test.Ctx(), localQueue, metav1.ApplyOptions{FieldManager: "setup-gpu-kueue-upgrade", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue LocalQueue %s/%s successfully", appliedLocalQueue.Namespace, appliedLocalQueue.Name)

	createGpuUpgradePyTorchJob(test, gpuUpgradeNamespaceName, appliedLocalQueue.Name)

	var workloadName string
	test.Eventually(KueueWorkloads(test, gpuUpgradeNamespaceName), TestTimeoutShort).Should(
		ContainElement(WithTransform(func(w *kueuev1beta2.Workload) bool {
			inadmissible, _ := KueueWorkloadInadmissible(w)
			if inadmissible {
				workloadName = w.Name
			}
			return inadmissible
		}, BeTrue())),
	)
	test.T().Logf("Kueue Workload '%s' is Inadmissible", workloadName)

	test.Eventually(PyTorchJob(test, gpuUpgradeNamespaceName, gpuUpgradePyTorchJobName), TestTimeoutShort).
		Should(WithTransform(PyTorchJobConditionSuspended, Equal(corev1.ConditionTrue)))
	test.T().Logf("PyTorchJob %s/%s is suspended, waiting for ClusterQueue to be enabled after upgrade", gpuUpgradeNamespaceName, gpuUpgradePyTorchJobName)

	appliedResourceFlavor, err = test.Client().Kueue().KueueV1beta2().ResourceFlavors().Get(test.Ctx(), gpuUpgradeResourceFlavorName, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	appliedClusterQueue, err = test.Client().Kueue().KueueV1beta2().ClusterQueues().Get(test.Ctx(), gpuUpgradeClusterQueueName, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	data := map[string]string{}
	test.Expect(AddUpgradeResourceBaseline(data, gpuUpgradeResourceFlavorGenerationKey, gpuUpgradeResourceFlavorSpecKey, appliedResourceFlavor.Generation, appliedResourceFlavor.Spec)).To(Succeed())
	test.Expect(AddUpgradeResourceBaseline(data, gpuUpgradeClusterQueueGenerationKey, gpuUpgradeClusterQueueSpecKey, appliedClusterQueue.Generation, appliedClusterQueue.Spec)).To(Succeed())
	StoreUpgradeBaseline(test, gpuUpgradeNamespaceName, gpuUpgradeBaselineConfigMap, data)
}

func TestRunGpuKueueUpgrade(t *testing.T) {
	Tags(t, PostUpgrade, Gpu(NVIDIA))
	test := With(t)
	SetupKueue(test, initialKueueState, PyTorchJobFramework)

	namespace := GetNamespaceWithName(test, gpuUpgradeNamespaceName)

	defer test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(test.Ctx(), gpuUpgradeResourceFlavorName, metav1.DeleteOptions{})
	defer test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(test.Ctx(), gpuUpgradeClusterQueueName, metav1.DeleteOptions{})
	defer test.Client().Core().CoreV1().ConfigMaps(gpuUpgradeNamespaceName).Delete(test.Ctx(), gpuUpgradeBaselineConfigMap, metav1.DeleteOptions{})
	defer DeleteTestNamespace(test, namespace)

	configMap, err := test.Client().Core().CoreV1().ConfigMaps(gpuUpgradeNamespaceName).Get(
		test.Ctx(), gpuUpgradeBaselineConfigMap, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "Baseline ConfigMap should exist")

	resourceFlavor, err := test.Client().Kueue().KueueV1beta2().ResourceFlavors().Get(test.Ctx(), gpuUpgradeResourceFlavorName, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "GPU ResourceFlavor should exist after upgrade")
	VerifyUpgradeResourceSpecIntegrity(test, "ResourceFlavor", resourceFlavor.Generation, resourceFlavor.Spec,
		configMap, gpuUpgradeResourceFlavorGenerationKey, gpuUpgradeResourceFlavorSpecKey)
	test.Expect(resourceFlavor.Spec.NodeLabels["nvidia.com/gpu.present"]).To(Equal("true"))

	clusterQueue, err := test.Client().Kueue().KueueV1beta2().ClusterQueues().Get(test.Ctx(), gpuUpgradeClusterQueueName, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "GPU ClusterQueue should exist after upgrade")
	VerifyUpgradeResourceSpecIntegrity(test, "ClusterQueue", clusterQueue.Generation, clusterQueue.Spec,
		configMap, gpuUpgradeClusterQueueGenerationKey, gpuUpgradeClusterQueueSpecKey)
	test.Expect(ClusterQueueNominalGPUQuota(clusterQueue, NVIDIA.ResourceLabel)).To(Equal(resource.MustParse("1")))

	clusterQueuePatch := kueueacv1beta2.ClusterQueue(gpuUpgradeClusterQueueName).WithSpec(kueueacv1beta2.ClusterQueueSpec().WithStopPolicy(kueuev1beta2.None))
	_, err = test.Client().Kueue().KueueV1beta2().ClusterQueues().Apply(test.Ctx(), clusterQueuePatch, metav1.ApplyOptions{FieldManager: "application/apply-patch", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Enabled ClusterQueue %s by setting StopPolicy to None", gpuUpgradeClusterQueueName)

	test.Eventually(KueueWorkloads(test, gpuUpgradeNamespaceName), TestTimeoutLong).Should(
		ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("GPU workload failed to be admitted"))),
	)
	test.T().Logf("Kueue Workload for PyTorchJob %s/%s is admitted", gpuUpgradeNamespaceName, gpuUpgradePyTorchJobName)

	test.Eventually(PyTorchJob(test, gpuUpgradeNamespaceName, gpuUpgradePyTorchJobName), TestTimeoutLong).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	test.Eventually(PyTorchJob(test, gpuUpgradeNamespaceName, gpuUpgradePyTorchJobName), TestTimeoutLong).
		Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PyTorchJob %s/%s completed successfully after upgrade", gpuUpgradeNamespaceName, gpuUpgradePyTorchJobName)
}

func createGpuUpgradePyTorchJob(test Test, namespace, localQueueName string) *kftov1.PyTorchJob {
	_, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Get(test.Ctx(), gpuUpgradePyTorchJobName, metav1.GetOptions{})
	if err == nil {
		err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Delete(test.Ctx(), gpuUpgradePyTorchJobName, metav1.DeleteOptions{})
		test.Expect(err).NotTo(HaveOccurred())
		test.Eventually(PyTorchJobs(test, namespace), TestTimeoutShort).Should(BeEmpty())
	} else if !errors.IsNotFound(err) {
		test.T().Fatalf("Error retrieving PyTorchJob with name `%s`: %v", gpuUpgradePyTorchJobName, err)
	}

	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: kftov1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuUpgradePyTorchJobName,
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
						Spec: corev1.PodSpec{
							Tolerations: []corev1.Toleration{
								{
									Key:      "nvidia.com/gpu",
									Operator: corev1.TolerationOpExists,
								},
							},
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           GetTrainingCudaPyTorch251Image(test),
									ImagePullPolicy: corev1.PullIfNotPresent,
									Command: []string{
										"python",
										"-c",
										"import torch; assert torch.cuda.is_available(); print('CUDA OK')",
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:                        resource.MustParse("2"),
											corev1.ResourceMemory:                     resource.MustParse("6Gi"),
											corev1.ResourceName(NVIDIA.ResourceLabel): resource.MustParse("1"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceName(NVIDIA.ResourceLabel): resource.MustParse("1"),
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

	tuningJob, err = test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PyTorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

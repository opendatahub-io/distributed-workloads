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
	"strings"
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"
	kueuev1beta2 "sigs.k8s.io/kueue/apis/kueue/v1beta2"
	kueueacv1beta2 "sigs.k8s.io/kueue/client-go/applyconfiguration/kueue/v1beta2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

var (
	upgradeNamespaceName = "test-trainer-upgrade"
	resourceFlavorName   = "rf-trainer-upgrade"
	clusterQueueName     = "cq-trainer-upgrade"
	localQueueName       = "lq-trainer-upgrade"
	upgradeTrainJobName  = "trainjob-upgrade"

	// Specific runtime upgrade test variables
	specificRuntimeNamespaceName  = "test-trainer-upgrade-specific"
	specificRuntimeResourceFlavor = "rf-trainer-upgrade-specific"
	specificRuntimeClusterQueue   = "cq-trainer-upgrade-specific"
	specificRuntimeLocalQueue     = "lq-trainer-upgrade-specific"
	specificRuntimeTrainJobName   = "trainjob-upgrade-specific"
	specificRuntimeConfigMapName  = "specific-runtime-upgrade"
	specificRuntimeConfigMapKey   = "runtime-name"

	// Custom runtime upgrade test variables
	customRuntimeNamespaceName  = "test-trainer-upgrade-custom-rt"
	customRuntimeResourceFlavor = "rf-trainer-upgrade-custom-rt"
	customRuntimeClusterQueue   = "cq-trainer-upgrade-custom-rt"
	customRuntimeLocalQueue     = "lq-trainer-upgrade-custom-rt"
	customRuntimeTrainJobName   = "trainjob-upgrade-custom-rt"
	customRuntimeCTRName        = "custom-upgrade-runtime"
)

func TestSetupUpgradeTrainJob(t *testing.T) {
	// Skip due to issue RHOAIENG-48867
	t.Skip("Skip due to issue RHOAIENG-48867")
	//Tags(t, PreUpgrade)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	// Create a namespace with Kueue label
	CreateOrGetTestNamespaceWithName(test, upgradeNamespaceName, WithKueueManaged())
	test.T().Logf("Created Kueue-managed namespace: %s", upgradeNamespaceName)

	// Create Kueue resources with StopPolicy
	resourceFlavor := kueueacv1beta2.ResourceFlavor(resourceFlavorName)
	appliedResourceFlavor, err := test.Client().Kueue().KueueV1beta2().ResourceFlavors().Apply(test.Ctx(), resourceFlavor, metav1.ApplyOptions{FieldManager: "setup-TrainJob", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ResourceFlavor %s successfully", appliedResourceFlavor.Name)

	clusterQueue := kueueacv1beta2.ClusterQueue(clusterQueueName).WithSpec(
		kueueacv1beta2.ClusterQueueSpec().
			WithNamespaceSelector(&metav1ac.LabelSelectorApplyConfiguration{}).
			WithResourceGroups(
				kueueacv1beta2.ResourceGroup().WithCoveredResources(
					corev1.ResourceName("cpu"), corev1.ResourceName("memory"),
				).WithFlavors(
					kueueacv1beta2.FlavorQuotas().
						WithName(kueuev1beta2.ResourceFlavorReference(resourceFlavorName)).
						WithResources(
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceCPU).WithNominalQuota(resource.MustParse("8")),
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceMemory).WithNominalQuota(resource.MustParse("18Gi")),
						),
				),
			).
			WithStopPolicy(kueuev1beta2.Hold),
	)
	appliedClusterQueue, err := test.Client().Kueue().KueueV1beta2().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "setup-TrainJob", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ClusterQueue %s with StopPolicy=Hold successfully", appliedClusterQueue.Name)

	localQueue := kueueacv1beta2.LocalQueue(localQueueName, upgradeNamespaceName).
		WithAnnotations(map[string]string{"kueue.x-k8s.io/default-queue": "true"}).
		WithSpec(
			kueueacv1beta2.LocalQueueSpec().WithClusterQueue(kueuev1beta2.ClusterQueueReference(clusterQueueName)),
		)
	appliedLocalQueue, err := test.Client().Kueue().KueueV1beta2().LocalQueues(upgradeNamespaceName).Apply(test.Ctx(), localQueue, metav1.ApplyOptions{FieldManager: "setup-TrainJob", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue LocalQueue %s/%s successfully", appliedLocalQueue.Namespace, appliedLocalQueue.Name)

	// Create TrainJob
	trainJob := createUpgradeTrainJob(test, upgradeNamespaceName, appliedLocalQueue.Name, upgradeTrainJobName, trainerutils.DefaultClusterTrainingRuntimeCUDA)

	// Verify Kueue Workload is Inadmissible
	var workloadName string
	test.Eventually(KueueWorkloads(test, upgradeNamespaceName), TestTimeoutShort).Should(
		ContainElement(WithTransform(func(w *kueuev1beta2.Workload) bool {
			inadmissible, _ := KueueWorkloadInadmissible(w)
			if inadmissible {
				workloadName = w.Name
			}
			return inadmissible
		}, BeTrue())),
	)
	test.T().Logf("Kueue Workload '%s' is Inadmissible", workloadName)

	// Make sure the TrainJob is suspended, waiting for ClusterQueue to be enabled
	test.Eventually(TrainJob(test, trainJob.Namespace, upgradeTrainJobName), TestTimeoutShort).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s is suspended, waiting for ClusterQueue to be enabled after upgrade", trainJob.Namespace, upgradeTrainJobName)
}

func TestRunUpgradeTrainJob(t *testing.T) {
	t.Skip("Skip due to issue RHOAIENG-48867")
	// Skip due to issue RHOAIENG-48867
	//Tags(t, PostUpgrade)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)
	namespace := GetNamespaceWithName(test, upgradeNamespaceName)

	defer test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(test.Ctx(), resourceFlavorName, metav1.DeleteOptions{})
	defer test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(test.Ctx(), clusterQueueName, metav1.DeleteOptions{})
	defer DeleteTestNamespace(test, namespace)

	// Enable ClusterQueue to process waiting TrainJob
	clusterQueue := kueueacv1beta2.ClusterQueue(clusterQueueName).WithSpec(kueueacv1beta2.ClusterQueueSpec().WithStopPolicy(kueuev1beta2.None))
	_, err := test.Client().Kueue().KueueV1beta2().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "application/apply-patch", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Enabled ClusterQueue %s by setting StopPolicy to None", clusterQueueName)

	// Verify Kueue Workload is admitted first
	test.Eventually(KueueWorkloads(test, upgradeNamespaceName), TestTimeoutLong).Should(
		ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("Workload failed to be admitted"))),
	)
	test.T().Logf("Kueue Workload for TrainJob %s/%s is admitted", upgradeNamespaceName, upgradeTrainJobName)

	// TrainJob should be unsuspended now
	test.Eventually(TrainJob(test, upgradeNamespaceName, upgradeTrainJobName), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionFalse)))
	test.T().Logf("TrainJob %s/%s is now running", upgradeNamespaceName, upgradeTrainJobName)

	// Make sure the TrainJob completes successfully
	test.Eventually(TrainJob(test, upgradeNamespaceName, upgradeTrainJobName), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s completed successfully after upgrade", upgradeNamespaceName, upgradeTrainJobName)
}

// This test verifies backward compatibility of the Trainer operator with older/specific runtimes that may be replaced during upgrades.
func TestSetupSpecificRuntimeUpgradeTrainJob(t *testing.T) {
	t.Skip("Skip due to issue RHOAIENG-48867")
	//Tags(t, PreUpgrade)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	// Find a specific ClusterTrainingRuntime
	specificRuntime := findSpecificRuntime(test)
	if specificRuntime == "" {
		test.T().Skip("No specific ClusterTrainingRuntime found, skipping test ...")
		return
	}
	test.T().Logf("Using specific ClusterTrainingRuntime: %s", specificRuntime)

	// Create namespace with Kueue label
	CreateOrGetTestNamespaceWithName(test, specificRuntimeNamespaceName, WithKueueManaged())
	test.T().Logf("Created Kueue-managed namespace: %s", specificRuntimeNamespaceName)

	// Store the runtime name in ConfigMap for post-upgrade verification
	storeSpecificRuntimeInConfigMap(test, specificRuntime)

	// Create Kueue resources with StopPolicy=Hold
	resourceFlavor := kueueacv1beta2.ResourceFlavor(specificRuntimeResourceFlavor)
	appliedResourceFlavor, err := test.Client().Kueue().KueueV1beta2().ResourceFlavors().Apply(test.Ctx(), resourceFlavor, metav1.ApplyOptions{FieldManager: "setup-specific-runtime", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ResourceFlavor %s successfully", appliedResourceFlavor.Name)

	clusterQueue := kueueacv1beta2.ClusterQueue(specificRuntimeClusterQueue).WithSpec(
		kueueacv1beta2.ClusterQueueSpec().
			WithNamespaceSelector(&metav1ac.LabelSelectorApplyConfiguration{}).
			WithResourceGroups(
				kueueacv1beta2.ResourceGroup().WithCoveredResources(
					corev1.ResourceName("cpu"), corev1.ResourceName("memory"),
				).WithFlavors(
					kueueacv1beta2.FlavorQuotas().
						WithName(kueuev1beta2.ResourceFlavorReference(specificRuntimeResourceFlavor)).
						WithResources(
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceCPU).WithNominalQuota(resource.MustParse("8")),
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceMemory).WithNominalQuota(resource.MustParse("18Gi")),
						),
				),
			).
			WithStopPolicy(kueuev1beta2.Hold),
	)
	appliedClusterQueue, err := test.Client().Kueue().KueueV1beta2().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "setup-specific-runtime", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ClusterQueue %s with StopPolicy=Hold successfully", appliedClusterQueue.Name)

	localQueue := kueueacv1beta2.LocalQueue(specificRuntimeLocalQueue, specificRuntimeNamespaceName).
		WithAnnotations(map[string]string{"kueue.x-k8s.io/default-queue": "true"}).
		WithSpec(
			kueueacv1beta2.LocalQueueSpec().WithClusterQueue(kueuev1beta2.ClusterQueueReference(specificRuntimeClusterQueue)),
		)
	appliedLocalQueue, err := test.Client().Kueue().KueueV1beta2().LocalQueues(specificRuntimeNamespaceName).Apply(test.Ctx(), localQueue, metav1.ApplyOptions{FieldManager: "setup-specific-runtime", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue LocalQueue %s/%s successfully", appliedLocalQueue.Namespace, appliedLocalQueue.Name)

	// Create TrainJob using the specific runtime
	trainJob := createUpgradeTrainJob(test, specificRuntimeNamespaceName, appliedLocalQueue.Name, specificRuntimeTrainJobName, specificRuntime)

	// Verify Kueue Workload is Inadmissible
	var workloadName string
	test.Eventually(KueueWorkloads(test, specificRuntimeNamespaceName), TestTimeoutShort).Should(
		ContainElement(WithTransform(func(w *kueuev1beta2.Workload) bool {
			inadmissible, _ := KueueWorkloadInadmissible(w)
			if inadmissible {
				workloadName = w.Name
			}
			return inadmissible
		}, BeTrue())),
	)
	test.T().Logf("Kueue Workload '%s' is Inadmissible", workloadName)

	// Verify TrainJob is suspended
	test.Eventually(TrainJob(test, trainJob.Namespace, specificRuntimeTrainJobName), TestTimeoutShort).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s using runtime %s is suspended, waiting for upgrade", trainJob.Namespace, specificRuntimeTrainJobName, specificRuntime)
}

// TestRunSpecificRuntimeUpgradeTrainJob verifies that a TrainJob using a specific cluster training runtime
// ClusterTrainingRuntime still works after upgrade, even if that runtime was replaced.
func TestRunSpecificRuntimeUpgradeTrainJob(t *testing.T) {
	t.Skip("Skip due to issue RHOAIENG-48867")
	//Tags(t, PostUpgrade)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	// Get the specific runtime name from ConfigMap
	specificRuntime := getSpecificRuntimeFromConfigMap(test)
	if specificRuntime == "" {
		test.T().Skip("No specific runtime stored in ConfigMap, skipping post-upgrade verification ...")
		return
	}
	test.T().Logf("Verifying upgrade for TrainJob using specific runtime: %s", specificRuntime)

	namespace := GetNamespaceWithName(test, specificRuntimeNamespaceName)

	defer func() {
		_ = test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(test.Ctx(), specificRuntimeResourceFlavor, metav1.DeleteOptions{})
		_ = test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(test.Ctx(), specificRuntimeClusterQueue, metav1.DeleteOptions{})
		_ = test.Client().Core().CoreV1().ConfigMaps(specificRuntimeNamespaceName).Delete(test.Ctx(), specificRuntimeConfigMapName, metav1.DeleteOptions{})
		DeleteTestNamespace(test, namespace)
	}()

	// Verify the ClusterTrainingRuntime still exists
	_, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Get(test.Ctx(), specificRuntime, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			test.T().Logf("ClusterTrainingRuntime %s was removed during upgrade (expected for versioned runtimes)", specificRuntime)
		} else {
			test.Expect(err).NotTo(HaveOccurred(), "Unexpected error checking ClusterTrainingRuntime")
		}
	} else {
		test.T().Logf("ClusterTrainingRuntime %s still exists after upgrade", specificRuntime)
	}

	// Verify TrainJob is still suspended
	trainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(specificRuntimeNamespaceName).Get(test.Ctx(), specificRuntimeTrainJobName, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "TrainJob should exist after upgrade")
	test.T().Logf("TrainJob %s/%s exists after upgrade with RuntimeRef: %s", trainJob.Namespace, trainJob.Name, trainJob.Spec.RuntimeRef.Name)

	// Enable ClusterQueue to process the TrainJob
	clusterQueue := kueueacv1beta2.ClusterQueue(specificRuntimeClusterQueue).WithSpec(kueueacv1beta2.ClusterQueueSpec().WithStopPolicy(kueuev1beta2.None))
	_, err = test.Client().Kueue().KueueV1beta2().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "application/apply-patch", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Enabled ClusterQueue %s by setting StopPolicy to None", specificRuntimeClusterQueue)

	// Verify Kueue Workload is admitted first
	test.Eventually(KueueWorkloads(test, specificRuntimeNamespaceName), TestTimeoutLong).Should(
		ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("Workload failed to be admitted"))),
	)
	test.T().Logf("Kueue Workload for TrainJob %s/%s is admitted", specificRuntimeNamespaceName, specificRuntimeTrainJobName)

	// TrainJob should be unsuspended now
	test.Eventually(TrainJob(test, specificRuntimeNamespaceName, specificRuntimeTrainJobName), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionFalse)))
	test.T().Logf("TrainJob %s/%s is now running with specific runtime %s", specificRuntimeNamespaceName, specificRuntimeTrainJobName, specificRuntime)

	// Wait for TrainJob to complete - this verifies the old runtime still works after upgrade
	test.Eventually(TrainJob(test, specificRuntimeNamespaceName, specificRuntimeTrainJobName), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s using specific runtime %s completed successfully after upgrade", specificRuntimeNamespaceName, specificRuntimeTrainJobName, specificRuntime)

}

// TestSetupCustomRuntimeUpgradeTrainJob creates a custom ClusterTrainingRuntime (using the image
// from the default CUDA runtime) and a suspended TrainJob referencing it. Since this CTR is
// user-created, its spec won't change during upgrade, avoiding the immutable JobSet field issue
// from RHOAIENG-48867.
func TestSetupCustomRuntimeUpgradeTrainJob(t *testing.T) {
	Tags(t, PreUpgrade)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	// Get image from the default CUDA runtime to use in our custom CTR
	image, err := trainerutils.GetImageFromClusterTrainingRuntime(test, trainerutils.DefaultClusterTrainingRuntimeCUDA)
	test.Expect(err).NotTo(HaveOccurred())

	// Create custom ClusterTrainingRuntime
	createCustomClusterTrainingRuntime(test, image)

	// Create Kueue-managed namespace
	CreateOrGetTestNamespaceWithName(test, customRuntimeNamespaceName, WithKueueManaged())
	test.T().Logf("Created Kueue-managed namespace: %s", customRuntimeNamespaceName)

	// Create Kueue resources with StopPolicy=Hold
	resourceFlavor := kueueacv1beta2.ResourceFlavor(customRuntimeResourceFlavor)
	appliedResourceFlavor, err := test.Client().Kueue().KueueV1beta2().ResourceFlavors().Apply(test.Ctx(), resourceFlavor, metav1.ApplyOptions{FieldManager: "setup-custom-runtime", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ResourceFlavor %s successfully", appliedResourceFlavor.Name)

	clusterQueue := kueueacv1beta2.ClusterQueue(customRuntimeClusterQueue).WithSpec(
		kueueacv1beta2.ClusterQueueSpec().
			WithNamespaceSelector(&metav1ac.LabelSelectorApplyConfiguration{}).
			WithResourceGroups(
				kueueacv1beta2.ResourceGroup().WithCoveredResources(
					corev1.ResourceName("cpu"), corev1.ResourceName("memory"),
				).WithFlavors(
					kueueacv1beta2.FlavorQuotas().
						WithName(kueuev1beta2.ResourceFlavorReference(customRuntimeResourceFlavor)).
						WithResources(
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceCPU).WithNominalQuota(resource.MustParse("8")),
							kueueacv1beta2.ResourceQuota().WithName(corev1.ResourceMemory).WithNominalQuota(resource.MustParse("18Gi")),
						),
				),
			).
			WithStopPolicy(kueuev1beta2.Hold),
	)
	appliedClusterQueue, err := test.Client().Kueue().KueueV1beta2().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "setup-custom-runtime", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue ClusterQueue %s with StopPolicy=Hold successfully", appliedClusterQueue.Name)

	localQueue := kueueacv1beta2.LocalQueue(customRuntimeLocalQueue, customRuntimeNamespaceName).
		WithAnnotations(map[string]string{"kueue.x-k8s.io/default-queue": "true"}).
		WithSpec(
			kueueacv1beta2.LocalQueueSpec().WithClusterQueue(kueuev1beta2.ClusterQueueReference(customRuntimeClusterQueue)),
		)
	appliedLocalQueue, err := test.Client().Kueue().KueueV1beta2().LocalQueues(customRuntimeNamespaceName).Apply(test.Ctx(), localQueue, metav1.ApplyOptions{FieldManager: "setup-custom-runtime", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Applied Kueue LocalQueue %s/%s successfully", appliedLocalQueue.Namespace, appliedLocalQueue.Name)

	// Create TrainJob using the custom CTR
	trainJob := createUpgradeTrainJob(test, customRuntimeNamespaceName, appliedLocalQueue.Name, customRuntimeTrainJobName, customRuntimeCTRName)

	// Verify Kueue Workload is Inadmissible
	var workloadName string
	test.Eventually(KueueWorkloads(test, customRuntimeNamespaceName), TestTimeoutShort).Should(
		ContainElement(WithTransform(func(w *kueuev1beta2.Workload) bool {
			inadmissible, _ := KueueWorkloadInadmissible(w)
			if inadmissible {
				workloadName = w.Name
			}
			return inadmissible
		}, BeTrue())),
	)
	test.T().Logf("Kueue Workload '%s' is Inadmissible", workloadName)

	// Verify TrainJob is suspended
	test.Eventually(TrainJob(test, trainJob.Namespace, customRuntimeTrainJobName), TestTimeoutShort).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s using custom runtime %s is suspended, waiting for upgrade", trainJob.Namespace, customRuntimeTrainJobName, customRuntimeCTRName)
}

func TestRunCustomRuntimeUpgradeTrainJob(t *testing.T) {
	Tags(t, PostUpgrade)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	namespace := GetNamespaceWithName(test, customRuntimeNamespaceName)

	defer func() {
		_ = test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(test.Ctx(), customRuntimeResourceFlavor, metav1.DeleteOptions{})
		_ = test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(test.Ctx(), customRuntimeClusterQueue, metav1.DeleteOptions{})
		_ = test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Delete(test.Ctx(), customRuntimeCTRName, metav1.DeleteOptions{})
		DeleteTestNamespace(test, namespace)
	}()

	// Verify custom CTR still exists after upgrade
	_, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Get(test.Ctx(), customRuntimeCTRName, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "Custom ClusterTrainingRuntime should exist after upgrade")
	test.T().Logf("Custom ClusterTrainingRuntime %s is preserved after upgrade", customRuntimeCTRName)

	// Enable ClusterQueue to process the TrainJob
	clusterQueue := kueueacv1beta2.ClusterQueue(customRuntimeClusterQueue).WithSpec(kueueacv1beta2.ClusterQueueSpec().WithStopPolicy(kueuev1beta2.None))
	_, err = test.Client().Kueue().KueueV1beta2().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "application/apply-patch", Force: true})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Enabled ClusterQueue %s by setting StopPolicy to None", customRuntimeClusterQueue)

	// Verify Kueue Workload is admitted
	test.Eventually(KueueWorkloads(test, customRuntimeNamespaceName), TestTimeoutLong).Should(
		ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("Workload failed to be admitted"))),
	)
	test.T().Logf("Kueue Workload for TrainJob %s/%s is admitted", customRuntimeNamespaceName, customRuntimeTrainJobName)

	// TrainJob should be unsuspended
	test.Eventually(TrainJob(test, customRuntimeNamespaceName, customRuntimeTrainJobName), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionFalse)))
	test.T().Logf("TrainJob %s/%s is now running", customRuntimeNamespaceName, customRuntimeTrainJobName)

	// Wait for TrainJob to complete
	test.Eventually(TrainJob(test, customRuntimeNamespaceName, customRuntimeTrainJobName), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s/%s using custom runtime %s completed successfully after upgrade", customRuntimeNamespaceName, customRuntimeTrainJobName, customRuntimeCTRName)
}

// Helper functions

func createUpgradeTrainJob(test Test, namespace, localQueueName, jobName, runtimeName string) *trainerv1alpha1.TrainJob {
	// Delete existing TrainJob if present
	_, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Get(test.Ctx(), jobName, metav1.GetOptions{})
	if err == nil {
		err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Delete(test.Ctx(), jobName, metav1.DeleteOptions{})
		test.Expect(err).NotTo(HaveOccurred())
		test.Eventually(TrainJobs(test, namespace), TestTimeoutShort).Should(BeEmpty())
	} else if !errors.IsNotFound(err) {
		test.T().Fatalf("Error retrieving TrainJob with name `%s`: %v", jobName, err)
	}

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			Name: jobName,
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": localQueueName,
			},
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: runtimeName,
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command: []string{
					"python",
					"-c",
					"import torch; print(f'PyTorch version: {torch.__version__}'); import time; time.sleep(5); print('Training completed successfully')",
				},
			},
		},
	}

	trainJob, err = test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(test.Ctx(), trainJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created TrainJob %s/%s with runtime %s", trainJob.Namespace, trainJob.Name, runtimeName)

	return trainJob
}

func findSpecificRuntime(test Test) string {
	runtimes, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "Failed to list ClusterTrainingRuntimes")

	var specificRuntimes []string
	for _, runtime := range runtimes.Items {
		if !trainerutils.IsDefaultRuntime(runtime.Name) {
			specificRuntimes = append(specificRuntimes, runtime.Name)
		}
	}

	if len(specificRuntimes) == 0 {
		return ""
	}

	test.T().Logf("Available specific ClusterTrainingRuntimes: [%s]", strings.Join(specificRuntimes, ", "))

	// Return the first one found
	return specificRuntimes[0]
}

// storeSpecificRuntimeInConfigMap stores the specific runtime name for post-upgrade verification
func storeSpecificRuntimeInConfigMap(test Test, runtimeName string) {
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      specificRuntimeConfigMapName,
			Namespace: specificRuntimeNamespaceName,
		},
		Data: map[string]string{
			specificRuntimeConfigMapKey: runtimeName,
		},
	}

	// Delete existing ConfigMap if present
	_ = test.Client().Core().CoreV1().ConfigMaps(specificRuntimeNamespaceName).Delete(test.Ctx(), specificRuntimeConfigMapName, metav1.DeleteOptions{})

	_, err := test.Client().Core().CoreV1().ConfigMaps(specificRuntimeNamespaceName).Create(test.Ctx(), configMap, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred(), "Failed to create ConfigMap for specific runtime")
	test.T().Logf("Stored specific runtime name in ConfigMap %s/%s: %s", specificRuntimeNamespaceName, specificRuntimeConfigMapName, runtimeName)
}

// getSpecificRuntimeFromConfigMap retrieves the specific runtime name from ConfigMap
func getSpecificRuntimeFromConfigMap(test Test) string {
	configMap, err := test.Client().Core().CoreV1().ConfigMaps(specificRuntimeNamespaceName).Get(test.Ctx(), specificRuntimeConfigMapName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			test.T().Log("ConfigMap for specific runtime not found, pre-upgrade test may not have run")
			return ""
		}
		test.Expect(err).NotTo(HaveOccurred(), "Failed to get ConfigMap for specific runtime")
	}

	runtimeName, ok := configMap.Data[specificRuntimeConfigMapKey]
	if !ok || runtimeName == "" {
		test.T().Log("No specific runtime name found in ConfigMap")
		return ""
	}

	return runtimeName
}

func createCustomClusterTrainingRuntime(test Test, image string) {
	_, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Get(
		test.Ctx(), customRuntimeCTRName, metav1.GetOptions{})
	if err == nil {
		err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Delete(
			test.Ctx(), customRuntimeCTRName, metav1.DeleteOptions{})
		test.Expect(err).NotTo(HaveOccurred())
		test.Eventually(func() bool {
			_, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Get(
				test.Ctx(), customRuntimeCTRName, metav1.GetOptions{})
			return errors.IsNotFound(err)
		}, TestTimeoutShort).Should(BeTrue())
	} else if !errors.IsNotFound(err) {
		test.T().Fatalf("Error retrieving ClusterTrainingRuntime: %v", err)
	}

	ctr := &trainerv1alpha1.ClusterTrainingRuntime{
		ObjectMeta: metav1.ObjectMeta{
			Name: customRuntimeCTRName,
		},
		Spec: trainerv1alpha1.TrainingRuntimeSpec{
			Template: trainerv1alpha1.JobSetTemplateSpec{
				Spec: jobsetv1alpha2.JobSetSpec{
					ReplicatedJobs: []jobsetv1alpha2.ReplicatedJob{
						{
							Name:     "node",
							Replicas: 1,
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
											RestartPolicy: corev1.RestartPolicyNever,
											Containers: []corev1.Container{
												{
													Name:            "node",
													Image:           image,
													ImagePullPolicy: corev1.PullIfNotPresent,
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

	_, err = test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Create(
		test.Ctx(), ctr, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created custom ClusterTrainingRuntime %s with image %s", customRuntimeCTRName, image)
}

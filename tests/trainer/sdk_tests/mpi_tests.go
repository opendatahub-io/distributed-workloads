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

package sdk_tests

import (
	"fmt"
	"os"
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"

	common "github.com/opendatahub-io/distributed-workloads/tests/common"
	support "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	mpiNotebookName       = "mpi.ipynb"
	mpiNotebookPath       = "resources/" + mpiNotebookName
	mpiScriptName         = "fashion_mnist_mpi.py"
	mpiTrainingScriptPath = "resources/" + mpiScriptName
)

func RunOpenMPICudaKueueDistributedTraining(t *testing.T, accelerator support.Accelerator) {
	runOpenMPICudaDistributedTraining(t, accelerator, true)
}

func runOpenMPICudaDistributedTraining(t *testing.T, accelerator support.Accelerator, useKueue bool) {
	test := support.With(t)

	namespace := newMPITestNamespace(test, useKueue)

	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	trainerutils.CreateUserClusterRoleBindingForTrainerRuntimes(test, userName)

	var localQueueName string
	var cleanupKueue func()
	if useKueue {
		localQueueName, cleanupKueue = setupOpenMPIGpuKueue(test, namespace.Name, accelerator)
		defer cleanupKueue()
	}

	notebookContent, err := os.ReadFile(mpiNotebookPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", mpiNotebookPath))

	installScript, err := os.ReadFile(installScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", installScriptPath))

	mpiTrainingScript, err := os.ReadFile(mpiTrainingScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read MPI training script: %s", mpiTrainingScriptPath))

	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{
		mpiNotebookName:       notebookContent,
		installKubeflowScript: installScript,
		mpiScriptName:         mpiTrainingScript,
	})

	storageClass, err := support.GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	rwxPvc := support.CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		"20Gi",
		support.AccessModes(corev1.ReadWriteMany),
		support.StorageClassName(storageClass.Name),
	)

	sdkInstallExports := buildKubeflowInstallExports()
	queueExport := ""
	if useKueue {
		queueExport = fmt.Sprintf("export KUEUE_QUEUE_NAME=%s; ", shellQuote(localQueueName))
	}

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export IPYTHONDIR='/tmp/.ipython'; "+
			"export OPENSHIFT_API_URL=%s; "+
			"export NOTEBOOK_USER_TOKEN=%s; "+
			"export NOTEBOOK_NAMESPACE=%s; "+
			"export NOTEBOOK_CONFIGMAP_NAME=%s; "+
			"export TRAINING_RUNTIME=%s; "+
			"export GPU_TYPE=%s; "+
			"%s"+
			"%s"+
			"python -m pip install --quiet --no-cache-dir --break-system-packages papermill && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		shellQuote(support.GetOpenShiftApiUrl(test)),
		shellQuote(userToken),
		shellQuote(namespace.Name),
		shellQuote(cm.Name),
		shellQuote(trainerutils.DefaultClusterTrainingRuntimeOpenMPICUDA),
		shellQuote(acceleratorGPUType(accelerator)),
		queueExport,
		sdkInstallExports,
		installKubeflowScript,
		mpiNotebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

	common.CreateNotebook(
		test,
		namespace,
		userToken,
		command,
		cm.Name,
		mpiNotebookName,
		0,
		rwxPvc,
		common.ContainerSizeMedium,
		common.GetRecommendedNotebookImageFromImageStream(test, common.NotebookImageStreamTrainingHubCUDA),
	)

	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), support.TestTimeoutLong).Should(HaveLen(0))
	}()

	if useKueue {
		test.T().Logf("Verifying SDK-submitted OpenMPI TrainJob has custom queue label: %s", localQueueName)
		test.Eventually(support.TrainJobs(test, namespace.Name), support.TestTimeoutDouble).Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(func(job trainerv1alpha1.TrainJob) string {
					return job.Labels["kueue.x-k8s.io/queue-name"]
				}, Equal(localQueueName))),
			),
		)

		test.T().Log("Verifying Kueue Workloads: Notebook on default queue, OpenMPI TrainJob on custom queue...")
		test.Eventually(support.KueueWorkloads(test, namespace.Name), support.TestTimeoutDouble).Should(
			And(
				HaveLen(2),
				ContainElement(WithTransform(func(w *kueuev1beta1.Workload) string {
					return w.Spec.QueueName
				}, Equal(support.KueueDefaultQueueName))),
				ContainElement(
					And(
						WithTransform(func(w *kueuev1beta1.Workload) string {
							return w.Spec.QueueName
						}, Equal(localQueueName)),
						WithTransform(support.KueueWorkloadAdmitted, BeTrue()),
					),
				),
			),
		)
	}

	podName, containerName := trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	err = trainerutils.PollNotebookLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")
}

func newMPITestNamespace(test support.Test, useKueue bool) *corev1.Namespace {
	test.T().Helper()
	if useKueue {
		namespace := test.NewTestNamespace(support.WithKueueManaged())
		test.T().Logf("Created Kueue-managed namespace: %s", namespace.Name)
		return namespace
	}

	return test.NewTestNamespace()
}

func setupOpenMPIGpuKueue(test support.Test, namespaceName string, accelerator support.Accelerator) (string, func()) {
	test.T().Helper()

	resourceFlavor := support.CreateKueueResourceFlavor(test, kueuev1beta1.ResourceFlavorSpec{
		NodeLabels: map[string]string{
			accelerator.ResourceLabel + ".present": "true",
		},
	})
	clusterQueue := support.CreateKueueClusterQueue(test, kueuev1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"kubernetes.io/metadata.name": namespaceName,
			},
		},
		ResourceGroups: []kueuev1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{
					corev1.ResourceCPU,
					corev1.ResourceMemory,
					corev1.ResourceName(accelerator.ResourceLabel),
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
								Name:         corev1.ResourceName(accelerator.ResourceLabel),
								NominalQuota: resource.MustParse("2"),
							},
						},
					},
				},
			},
		},
	})
	localQueue := support.CreateKueueLocalQueue(test, namespaceName, clusterQueue.Name)
	test.T().Logf("Created custom LocalQueue %s for OpenMPI SDK TrainJob", localQueue.Name)
	cleanup := func() {
		if err := test.Client().Kueue().KueueV1beta1().LocalQueues(namespaceName).Delete(test.Ctx(), localQueue.Name, metav1.DeleteOptions{}); err != nil {
			test.T().Logf("failed to delete LocalQueue %s: %v", localQueue.Name, err)
		}
		if err := test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{}); err != nil {
			test.T().Logf("failed to delete ClusterQueue %s: %v", clusterQueue.Name, err)
		}
		if err := test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{}); err != nil {
			test.T().Logf("failed to delete ResourceFlavor %s: %v", resourceFlavor.Name, err)
		}
	}
	return localQueue.Name, cleanup
}

func acceleratorGPUType(accelerator support.Accelerator) string {
	switch accelerator.ResourceLabel {
	case support.AMD.ResourceLabel:
		return "amd"
	case support.NVIDIA.ResourceLabel:
		return "nvidia"
	default:
		return accelerator.Type
	}
}

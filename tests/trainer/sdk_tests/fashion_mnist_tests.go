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
	notebookName          = "mnist.ipynb"
	notebookPath          = "resources/" + notebookName
	installScriptPath     = "resources/disconnected_env/install_kubeflow.py"
	installKubeflowScript = "install_kubeflow.py"
)

// CPU Only - Distributed Training
func RunFashionMnistCpuDistributedTraining(t *testing.T) {
	test := support.With(t)

	// Create a new test namespace
	namespace := test.NewTestNamespace()

	// Ensure Notebook ServiceAccount exists (no extra RBAC)
	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	// RBACs setup
	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	// ClusterRoleBinding for cluster-scoped resources (ClusterTrainingRuntimes) - minimal get/list/watch access
	trainerutils.CreateUserClusterRoleBindingForTrainerRuntimes(test, userName)

	// Create ConfigMap with notebook and kubeflow install script
	nb, err := os.ReadFile(notebookPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", notebookPath))
	installScript, err := os.ReadFile(installScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", installScriptPath))
	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{
		notebookName:          nb,
		installKubeflowScript: installScript,
	})

	// Build command with parameters and pinned deps, and print definitive status line to logs
	endpoint, endpointOK := support.GetStorageBucketDefaultEndpoint()
	accessKey, _ := support.GetStorageBucketAccessKeyId()
	secretKey, _ := support.GetStorageBucketSecretKey()
	bucket, bucketOK := support.GetStorageBucketName()
	prefix, _ := support.GetStorageBucketMnistDir()
	if !endpointOK {
		endpoint = ""
	}
	if !bucketOK {
		bucket = ""
	}
	// Create RWX PVC for shared dataset and pass the claim name to the notebook
	storageClass, err := support.GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	rwxPvc := support.CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		"20Gi",
		support.AccessModes(corev1.ReadWriteMany),
		support.StorageClassName(storageClass.Name),
	)

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export OPENSHIFT_API_URL='%s'; export NOTEBOOK_TOKEN='%s'; "+
			"export NOTEBOOK_NAMESPACE='%s'; "+
			"export SHARED_PVC_NAME='%s'; "+
			"export AWS_DEFAULT_ENDPOINT='%s'; export AWS_ACCESS_KEY_ID='%s'; "+
			"export AWS_SECRET_ACCESS_KEY='%s'; export AWS_STORAGE_BUCKET='%s'; "+
			"export AWS_STORAGE_BUCKET_MNIST_DIR='%s'; "+
			"export TRAINING_RUNTIME='%s'; "+
			"export GPU_TYPE='cpu'; "+
			"python -m pip install --quiet --no-cache-dir ipykernel papermill boto3==1.34.162 && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		support.GetOpenShiftApiUrl(test), userToken, namespace.Name, rwxPvc.Name,
		endpoint, accessKey, secretKey, bucket, prefix,
		trainerutils.DefaultClusterTrainingRuntime,
		installKubeflowScript,
		notebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

	// Create Notebook CR using the RWX PVC
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, notebookName, 0, rwxPvc, common.ContainerSizeSmall)

	// Cleanup - use longer timeout due to large runtime images
	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), support.TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Wait for the Notebook Pod and get pod/container names
	podName, containerName := trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	// Poll logs to check if the notebook execution completed successfully
	err = trainerutils.PollNotebookLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")

}

// Kueue Integration - CPU Only - Distributed Training
func RunFashionMnistKueueCpuDistributedTraining(t *testing.T) {
	test := support.With(t)

	// Create a Kueue-managed namespace
	namespace := test.NewTestNamespace(support.WithKueueManaged())
	test.T().Logf("Created Kueue-managed namespace: %s", namespace.Name)

	// Ensure Notebook ServiceAccount exists (no extra RBAC)
	trainerutils.EnsureNotebookServiceAccount(t, test, namespace.Name)

	// RBACs setup
	userName := common.GetNotebookUserName(test)
	userToken := common.GenerateNotebookUserToken(test)
	support.CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")
	// ClusterRoleBinding for cluster-scoped resources (ClusterTrainingRuntimes) - minimal get/list/watch access
	trainerutils.CreateUserClusterRoleBindingForTrainerRuntimes(test, userName)

	// Create Kueue resources
	resourceFlavor := support.CreateKueueResourceFlavor(test, kueuev1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})

	cqSpec := kueuev1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []kueuev1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceCPU, corev1.ResourceMemory},
				Flavors: []kueuev1beta1.FlavorQuotas{
					{
						Name: kueuev1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []kueuev1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("8"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("36Gi"),
							},
						},
					},
				},
			},
		},
	}

	clusterQueue := support.CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})

	// Note: a default LocalQueue (named "default") is auto-created in Kueue-managed namespaces for the Notebook CR
	// Custom LocalQueue for the TrainJob — demonstrates explicit local queue assignment via the SDK
	customLocalQueue := support.CreateKueueLocalQueue(test, namespace.Name, clusterQueue.Name)
	test.T().Logf("Created custom LocalQueue %s for TrainJob", customLocalQueue.Name)

	// Create ConfigMap with notebook and kubeflow install script
	nb, err := os.ReadFile(notebookPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read notebook: %s", notebookPath))
	installScript, err := os.ReadFile(installScriptPath)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("failed to read install script: %s", installScriptPath))
	cm := support.CreateConfigMap(test, namespace.Name, map[string][]byte{
		notebookName:          nb,
		installKubeflowScript: installScript,
	})

	// Build command with parameters and pinned deps, and print definitive status line to logs
	endpoint, endpointOK := support.GetStorageBucketDefaultEndpoint()
	accessKey, _ := support.GetStorageBucketAccessKeyId()
	secretKey, _ := support.GetStorageBucketSecretKey()
	bucket, bucketOK := support.GetStorageBucketName()
	prefix, _ := support.GetStorageBucketMnistDir()
	if !endpointOK {
		endpoint = ""
	}
	if !bucketOK {
		bucket = ""
	}
	// Create RWX PVC for shared dataset and pass the claim name to the notebook
	storageClass, err := support.GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	rwxPvc := support.CreatePersistentVolumeClaim(
		test,
		namespace.Name,
		"20Gi",
		support.AccessModes(corev1.ReadWriteMany),
		support.StorageClassName(storageClass.Name),
	)

	shellCmd := fmt.Sprintf(
		"set -e; "+
			"export OPENSHIFT_API_URL='%s'; export NOTEBOOK_TOKEN='%s'; "+
			"export NOTEBOOK_NAMESPACE='%s'; "+
			"export SHARED_PVC_NAME='%s'; "+
			"export AWS_DEFAULT_ENDPOINT='%s'; export AWS_ACCESS_KEY_ID='%s'; "+
			"export AWS_SECRET_ACCESS_KEY='%s'; export AWS_STORAGE_BUCKET='%s'; "+
			"export AWS_STORAGE_BUCKET_MNIST_DIR='%s'; "+
			"export TRAINING_RUNTIME='%s'; "+
			"export GPU_TYPE='cpu'; "+
			"export KUEUE_QUEUE_NAME='%s'; "+
			"python -m pip install --quiet --no-cache-dir ipykernel papermill boto3==1.34.162 && "+
			"python /opt/app-root/notebooks/%s && "+
			"if python -m papermill -k python3 /opt/app-root/notebooks/%s /opt/app-root/src/out.ipynb --log-output; "+
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
		support.GetOpenShiftApiUrl(test), userToken, namespace.Name, rwxPvc.Name,
		endpoint, accessKey, secretKey, bucket, prefix,
		trainerutils.DefaultClusterTrainingRuntime,
		customLocalQueue.Name,
		installKubeflowScript,
		notebookName,
	)
	command := []string{"/bin/sh", "-c", shellCmd}

	// Create Notebook CR using the RWX PVC
	common.CreateNotebook(test, namespace, userToken, command, cm.Name, notebookName, 0, rwxPvc, common.ContainerSizeSmall)

	// Cleanup - use longer timeout due to large runtime images
	defer func() {
		common.DeleteNotebook(test, namespace)
		test.Eventually(common.Notebooks(test, namespace), support.TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Verify TrainJob is created with the custom local queue-name label
	test.T().Logf("Verifying SDK-submitted TrainJob has custom queue label: %s", customLocalQueue.Name)
	test.Eventually(support.TrainJobs(test, namespace.Name), support.TestTimeoutDouble).Should(
		And(
			HaveLen(1),
			ContainElement(WithTransform(func(job trainerv1alpha1.TrainJob) string {
				return job.Labels["kueue.x-k8s.io/queue-name"]
			}, Equal(customLocalQueue.Name))),
		),
	)
	test.T().Logf("SDK-submitted TrainJob has kueue label: kueue.x-k8s.io/queue-name=%s", customLocalQueue.Name)

	// Verify Kueue Workloads: one for the Notebook (default queue) and one for the TrainJob (custom queue)
	test.T().Log("Verifying Kueue Workloads: Notebook on default queue, TrainJob on custom queue...")
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
					}, Equal(customLocalQueue.Name)),
					WithTransform(support.KueueWorkloadAdmitted, BeTrue()),
				),
			),
		),
	)
	test.T().Log("Kueue Workload admitted successfully for SDK-submitted TrainJob")

	// Wait for the Notebook Pod and get pod/container names
	podName, containerName := trainerutils.WaitForNotebookPodRunning(test, namespace.Name)

	// Poll logs to check if the notebook execution completed successfully
	err = trainerutils.PollNotebookLogsForStatus(test, namespace.Name, podName, containerName, support.TestTimeoutDouble)
	test.Expect(err).ShouldNot(HaveOccurred(), "Notebook execution reported FAILURE")
}

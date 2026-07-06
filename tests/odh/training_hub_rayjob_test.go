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

package odh

import (
	"fmt"
	"strings"
	"testing"

	. "github.com/onsi/gomega"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

func TestGrpoRayJobSingleNodeMultiGpu(t *testing.T) {
	Tags(t, ExamplesCuda, MultiGpu(NVIDIA, 2))
	trainingHubRayJob(t, "grpo", 2)
}

func TestSftRayJobSingleGpu(t *testing.T) {
	Tags(t, ExamplesCuda)
	trainingHubRayJob(t, "sft", 1)
}

func TestOsftRayJobSingleGpu(t *testing.T) {
	Tags(t, ExamplesCuda)
	trainingHubRayJob(t, "osft", 1)
}

func TestLoraRayJobSingleGpu(t *testing.T) {
	Tags(t, ExamplesCuda)
	trainingHubRayJob(t, "lora", 1)
}

func trainingHubRayJob(t *testing.T, algorithm string, numGpus int) {
	test := With(t)

	namespace := test.NewTestNamespace()

	ensureNotebookServiceAccount(test, namespace.Name)

	userName := GetNotebookUserName(test)
	userToken := GetNotebookUserToken(test)
	CreateUserRoleBindingWithClusterRole(test, userName, namespace.Name, "admin")

	rayTrainingHubImage := GetRayTrainingHubImage()

	jupyterNotebookConfigMapFileName := "training_hub_rayjob.ipynb"
	scriptFileName := fmt.Sprintf("%s_train.py", algorithm)
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		jupyterNotebookConfigMapFileName: readFile(test, "resources/training_hub_rayjob.ipynb"),
		scriptFileName:                   readFile(test, fmt.Sprintf("resources/%s_train.py", algorithm)),
	})

	storageClass, err := GetRWXStorageClass(test)
	test.Expect(err).NotTo(HaveOccurred(), "Failed to find an RWX supporting StorageClass")
	notebookPVC := CreatePersistentVolumeClaim(test, namespace.Name, "20Gi",
		AccessModes(corev1.ReadWriteMany), StorageClassName(storageClass.Name))

	s3Endpoint, _ := GetStorageBucketDefaultEndpoint()
	s3AccessKey, _ := GetStorageBucketAccessKeyId()
	s3SecretKey, _ := GetStorageBucketSecretKey()
	s3Bucket, _ := GetStorageBucketName()
	s3Prefix, _ := GetStorageBucketRayTrainingHubDir()

	notebookCommand := getTrainingHubNotebookCommand(
		rayTrainingHubImage, config.Name, algorithm, notebookPVC.Name,
		s3Endpoint, s3AccessKey, s3SecretKey, s3Bucket, s3Prefix,
	)

	CreateNotebook(test, namespace, userToken, notebookCommand, config.Name,
		jupyterNotebookConfigMapFileName, numGpus, notebookPVC, ContainerSizeMedium,
		GetRecommendedNotebookImageFromImageStream(test, NotebookImageStreamDataScience))

	defer func() {
		DeleteNotebook(test, namespace)
		test.Eventually(Notebooks(test, namespace), TestTimeoutGpuProvisioning).Should(HaveLen(0))
	}()

	// Wait for the RayCluster to be created and running
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutGpuProvisioning).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(RayClusterState, Equal(rayv1.Ready))),
			),
		)

	// Poll notebook logs for definitive status using the shared helper.
	err = trainerutils.PollNotebookLogsForStatus(test, namespace.Name,
		NOTEBOOK_POD_NAME, NOTEBOOK_CONTAINER_NAME, TestTimeoutGpuProvisioning)
	test.Expect(err).ShouldNot(HaveOccurred(),
		fmt.Sprintf("%s notebook execution reported FAILURE", algorithm))

	// Verify the RayJob CR completed successfully
	rayJobs := GetRayJobs(test, namespace.Name)
	test.Expect(rayJobs).To(HaveLen(1), "Expected exactly one RayJob")
	test.Expect(RayJobStatus(rayJobs[0])).To(Equal(rayv1.JobStatusSucceeded),
		fmt.Sprintf("%s RayJob did not succeed", algorithm))

	// Best-effort: verify training completion message in submitter pod logs.
	// The submitter pod may already be cleaned up if shutdownAfterJobFinishes is true.
	rayJobName := rayJobs[0].Name
	submitterPods := GetPods(test, namespace.Name, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("ray.io/rayjob=%s,batch.kubernetes.io/job-name", rayJobName),
	})
	if len(submitterPods) > 0 {
		submitterLogs := GetPodLog(test, namespace.Name, submitterPods[0].Name, corev1.PodLogOptions{
			Container: "ray-job-submitter",
		})
		test.Expect(submitterLogs).To(ContainSubstring("training completed"),
			fmt.Sprintf("%s training did not complete - missing completion message in submitter pod logs", algorithm))
	} else {
		test.T().Logf("Submitter pod already cleaned up — skipping log verification (RayJob CR status confirmed success)")
	}

	test.T().Logf("Waiting for RayCluster cleanup...")
	test.Eventually(RayClusters(test, namespace.Name), TestTimeoutGpuProvisioning).
		Should(BeEmpty())
}

func getTrainingHubNotebookCommand(rayImage, scriptConfigMap, algorithm, pvcName,
	s3Endpoint, s3AccessKey, s3SecretKey, s3Bucket, s3Prefix string) []string {

	s3Exports := ""
	if s3Endpoint != "" && s3Bucket != "" {
		s3Exports = fmt.Sprintf(
			"export AWS_DEFAULT_ENDPOINT=%s; export AWS_ACCESS_KEY_ID=%s; "+
				"export AWS_SECRET_ACCESS_KEY=%s; export AWS_STORAGE_BUCKET=%s; "+
				"export AWS_STORAGE_BUCKET_RAY_TRAINING_HUB_DIR=%s; ",
			trainingHubShellQuote(s3Endpoint), trainingHubShellQuote(s3AccessKey),
			trainingHubShellQuote(s3SecretKey), trainingHubShellQuote(s3Bucket),
			trainingHubShellQuote(s3Prefix),
		)
	}

	return []string{
		"/bin/sh",
		"-c",
		s3Exports +
			"pip install papermill huggingface_hub datasets s3fs && " +
			"if papermill /opt/app-root/notebooks/{{.NotebookConfigMapFileName}}" +
			" /opt/app-root/src/mcad-out.ipynb -p namespace {{.Namespace}} -p ray_image " + rayImage +
			fmt.Sprintf(" -p script_configmap %s -p algorithm %s -p pvc_name %s", scriptConfigMap, algorithm, pvcName) +
			" -p openshift_api_url {{.OpenShiftApiUrl}} -p kubernetes_user_bearer_token {{.KubernetesUserBearerToken}}" +
			" -p num_gpus {{ .NumGpus }} --log-output; " +
			"then echo 'NOTEBOOK_STATUS: SUCCESS'; else echo 'NOTEBOOK_STATUS: FAILURE'; fi; sleep infinity",
	}
}

func trainingHubShellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

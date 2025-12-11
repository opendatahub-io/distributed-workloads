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

package odh

import (
	"embed"
	"net/url"
	"os"

	gonanoid "github.com/matoous/go-nanoid/v2"
	gomega "github.com/onsi/gomega"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

//go:embed resources/*
var files embed.FS

func readFile(t support.Test, fileName string) []byte {
	t.T().Helper()
	file, err := files.ReadFile(fileName)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return file
}

func ReadFileExt(t support.Test, fileName string) []byte {
	t.T().Helper()
	file, err := os.ReadFile(fileName)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return file
}

func getNotebookCommand(rayImage string) []string {
	return []string{
		"/bin/sh",
		"-c",
		"pip install papermill && papermill /opt/app-root/notebooks/{{.NotebookConfigMapFileName}}" +
			" /opt/app-root/src/mcad-out.ipynb -p namespace {{.Namespace}} -p ray_image " + rayImage + " " +
			" -p openshift_api_url {{.OpenShiftApiUrl}} -p kubernetes_user_bearer_token {{.KubernetesUserBearerToken}}" +
			" -p num_gpus {{ .NumGpus }} --log-output && sleep infinity",
	}
}

func GetDashboardUrl(test support.Test, namespace *corev1.Namespace, rayCluster *rayv1.RayCluster) string {
	dashboardName := "ray-dashboard-" + rayCluster.Name
	route := support.GetRoute(test, namespace.Name, dashboardName)
	hostname := route.Status.Ingress[0].Host
	dashboardUrl, _ := url.Parse("https://" + hostname)
	test.T().Logf("Ray-dashboard route : %s\n", dashboardUrl.String())

	return dashboardUrl.String()
}

func GetTestJobId(test support.Test, rayClient support.RayClusterClient) string {
	allJobsData, err := rayClient.ListJobs()
	test.Expect(err).ToNot(gomega.HaveOccurred())
	test.Expect(allJobsData).NotTo(gomega.BeEmpty())

	jobID := allJobsData[0].SubmissionID
	test.T().Logf("Ray job has been successfully submitted to the raycluster with Submission-ID : %s\n", jobID)

	return jobID
}

// EnsureNotebookServiceAccount ensures the Notebook ServiceAccount exists in the target namespace.
// This avoids webhook/controller failures when creating the Notebook CR.
func ensureNotebookServiceAccount(test support.Test, namespace string) {
	test.T().Helper()
	saName := "jupyter-nb-kube-3aadmin"
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: namespace}}
	_, err := test.Client().Core().CoreV1().ServiceAccounts(namespace).Create(test.Ctx(), sa, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		test.T().Fatalf("Failed to create ServiceAccount %s/%s: %v", namespace, saName, err)
	}
}

// Adds a unique suffix to the provided string
func uniqueSuffix(prefix string) string {
	suffix := gonanoid.MustGenerate("1234567890abcdef", 4)
	return prefix + "-" + suffix
}

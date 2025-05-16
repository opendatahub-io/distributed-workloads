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
	awv1beta2 "github.com/project-codeflare/appwrapper/api/v1beta2"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

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

func GetDashboardUrl(test support.Test, namespace *v1.Namespace, rayCluster *rayv1.RayCluster) string {
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

// Adds a unique suffix to the provided string
func uniqueSuffix(prefix string) string {
	suffix := gonanoid.MustGenerate("1234567890abcdef", 4)
	return prefix + "-" + suffix
}

func newAppWrapperWithRayCluster(awName string, rcName string, namespace string) *awv1beta2.AppWrapper {
	return &awv1beta2.AppWrapper{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "workload.codeflare.dev/v1beta2",
			Kind:       "AppWrapper",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      awName,
			Namespace: namespace,
		},
		Spec: awv1beta2.AppWrapperSpec{
			Components: []awv1beta2.AppWrapperComponent{
				{
					Template: runtime.RawExtension{
						Object: &rayv1.RayCluster{
							TypeMeta: metav1.TypeMeta{
								APIVersion: "ray.io/v1",
								Kind:       "RayCluster",
							},
							ObjectMeta: metav1.ObjectMeta{
								Name:      rcName,
								Namespace: namespace,
							},
							Spec: rayv1.RayClusterSpec{
								HeadGroupSpec: rayv1.HeadGroupSpec{
									RayStartParams: map[string]string{},
									Template: v1.PodTemplateSpec{
										Spec: v1.PodSpec{
											Containers: []v1.Container{
												{
													Name:  "ray-head",
													Image: "rayproject/ray:latest",
													Ports: []v1.ContainerPort{
														{
															Name:          "redis",
															ContainerPort: 6379,
														},
													},
												},
											},
										},
									},
								},
								WorkerGroupSpecs: []rayv1.WorkerGroupSpec{
									{
										GroupName:      "workers",
										Replicas:       support.Ptr(int32(1)),
										RayStartParams: map[string]string{},
										Template: v1.PodTemplateSpec{
											Spec: v1.PodSpec{
												Containers: []v1.Container{
													{
														Name:  "ray-worker",
														Image: "rayproject/ray:latest",
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
}

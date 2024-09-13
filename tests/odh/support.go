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

	gomega "github.com/onsi/gomega"
	"github.com/project-codeflare/codeflare-common/support"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	v1 "k8s.io/api/core/v1"
)

//go:embed resources/*
var files embed.FS

func ReadFile(t support.Test, fileName string) []byte {
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

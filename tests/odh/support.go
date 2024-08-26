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
	"net/http"
	"net/url"
	"os"

	. "github.com/onsi/gomega"
	gomega "github.com/onsi/gomega"
	"github.com/project-codeflare/codeflare-common/support"
	. "github.com/project-codeflare/codeflare-common/support"
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

func GetDashboardUrl(test support.Test, namespace *v1.Namespace, rayCluster *rayv1.RayCluster) *url.URL {
	dashboardName := "ray-dashboard-" + rayCluster.Name
	test.T().Logf("Raycluster created : %s\n", rayCluster.Name)
	route := GetRoute(test, namespace.Name, dashboardName)
	hostname := route.Status.Ingress[0].Host
	dashboardUrl, _ := url.Parse("https://" + hostname)
	test.T().Logf("Ray-dashboard route : %s\n", dashboardUrl.String())

	return dashboardUrl
}

func GetTestJobId(test Test, rayClient RayClusterClient, hostName string) string {
	listJobsReq, err := http.NewRequest("GET", "https://"+hostName+"/api/jobs/", nil)
	if err != nil {
		test.T().Errorf("failed to do get request: %s\n", err)
	}
	listJobsReq.Header.Add("Authorization", "Bearer "+test.Config().BearerToken)

	allJobsData, err := rayClient.GetJobs()
	test.Expect(err).ToNot(HaveOccurred())

	jobID := (*allJobsData)[0].SubmissionID
	if len(*allJobsData) > 0 {
		test.T().Logf("Ray job has been successfully submitted to the raycluster with Submission-ID : %s\n", jobID)
	}
	return jobID
}

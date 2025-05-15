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

package support

import (
	"github.com/onsi/gomega"
)

func GetRayJobAPIDetails(t Test, rayClient RayClusterClient, jobID string) *RayJobDetailsResponse {
	t.T().Helper()
	return RayJobAPIDetails(t, rayClient, jobID)(t)
}

func WriteRayJobAPILogs(t Test, rayClient RayClusterClient, jobID string) {
	t.T().Helper()
	jobLogs, err := rayClient.GetJobLogs(jobID)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	WriteToOutputDir(t, "ray-job-log-"+jobID, Log, []byte(jobLogs.Logs))
}

func RayJobAPIDetails(t Test, rayClient RayClusterClient, jobID string) func(g gomega.Gomega) *RayJobDetailsResponse {
	return func(g gomega.Gomega) *RayJobDetailsResponse {
		jobDetails, err := rayClient.GetJobDetails(jobID)
		t.Expect(err).NotTo(gomega.HaveOccurred())
		return jobDetails
	}
}

func GetRayJobAPIDetailsStatus(jobDetails *RayJobDetailsResponse) string {
	return jobDetails.Status
}

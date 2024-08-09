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
	"crypto/tls"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	. "github.com/onsi/gomega"
	gomega "github.com/onsi/gomega"
	"github.com/project-codeflare/codeflare-common/support"
	. "github.com/project-codeflare/codeflare-common/support"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func ReadJobLogs(test support.Test, namespace *v1.Namespace) string {
	rayClusters, err := test.Client().Ray().RayV1().RayClusters(namespace.Name).List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.Expect(len(rayClusters.Items)).To(BeNumerically(">", 0))

	rayCluster := rayClusters.Items[0]
	dashboardName := "ray-dashboard-" + rayCluster.Name
	fmt.Printf("Raycluster created : %s\n", rayCluster.Name)
	route := GetRoute(test, namespace.Name, dashboardName)
	hostname := route.Status.Ingress[0].Host

	// Wait for expected HTTP code
	fmt.Printf("Waiting for Route %s/%s to be available...\n", route.Namespace, route.Name)
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		Proxy:           http.ProxyFromEnvironment,
	}
	client := &http.Client{Transport: tr}
	req, err := http.NewRequest("GET", "https://"+hostname+"/api/jobs/", nil)
	if err != nil {
		test.T().Fatal(err)
	}
	req.Header.Add("Authorization", "Bearer "+test.Config().BearerToken)

	resp, err := client.Do(req)
	test.Expect(err).ToNot(HaveOccurred())
	test.Expect(resp.StatusCode).ToNot(Equal(503))
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	test.Expect(err).ToNot(HaveOccurred())

	var resp_json []map[string]interface{}
	err = json.Unmarshal(body, &resp_json)
	test.Expect(err).ToNot(HaveOccurred())
	if len(resp_json) > 0 {
		fmt.Printf("Job is submitted in the raycluster!\nSubmission-ID : %s\n", resp_json[0]["submission_id"])
	}

	var status string
	var prevStatus string
	fmt.Printf("Waiting for job to be Succeeded...\n")
	for status != "SUCCEEDED" {
		resp, err := client.Do(req)
		test.Expect(err).ToNot(HaveOccurred())
		body, err := io.ReadAll(resp.Body)
		test.Expect(err).ToNot(HaveOccurred())
		var result []map[string]interface{}
		if err := json.Unmarshal(body, &result); err != nil {
			time.Sleep(2 * time.Second)
			break
		}
		if status, ok := result[0]["status"].(string); ok {
			if prevStatus != status && status != "SUCCEEDED" {
				fmt.Printf("JobStatus : %s...\n", status)
				prevStatus = status
			}
			if status == "SUCCEEDED" {
				fmt.Printf("JobStatus : %s\n", status)
				prevStatus = status
				break
			}
			prevStatus = status
		} else {
			test.T().Logf("Status key not found or not a string")
		}
		time.Sleep(3 * time.Second)
	}
	if prevStatus != "SUCCEEDED" {
		fmt.Printf("Job failed!")
	}
	return prevStatus
}

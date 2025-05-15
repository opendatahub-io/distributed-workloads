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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

type RayJobSetup struct {
	EntryPoint string         `json:"entrypoint"`
	RuntimeEnv map[string]any `json:"runtime_env"`
}

type RayJobResponse struct {
	JobID        string `json:"job_id"`
	SubmissionID string `json:"submission_id"`
}

type RayJobDetailsResponse struct {
	JobID        string `json:"job_id"`
	SubmissionID string `json:"submission_id"`
	Status       string `json:"status"`
}

type RayJobLogsResponse struct {
	Logs string `json:"logs"`
}

type RayClusterClientConfig struct {
	Address string
	Client  *http.Client
}

var _ RayClusterClient = (*rayClusterClient)(nil)

type rayClusterClient struct {
	config RayClusterClientConfig
}

type RayClusterClient interface {
	CreateJob(job *RayJobSetup) (*RayJobResponse, error)
	GetJobDetails(jobID string) (*RayJobDetailsResponse, error)
	GetJobLogs(jobID string) (*RayJobLogsResponse, error)
	ListJobs() ([]RayJobDetailsResponse, error)
}

func NewRayClusterClient(config RayClusterClientConfig) (RayClusterClient, error) {
	endpoint, err := url.Parse(config.Address)
	if err != nil {
		return nil, fmt.Errorf("invalid dashboard endpoint address: %s", endpoint)
	}

	rayClusterApiClient := &rayClusterClient{config}
	return rayClusterApiClient, nil
}

func (client *rayClusterClient) CreateJob(job *RayJobSetup) (response *RayJobResponse, err error) {
	marshalled, err := json.Marshal(job)
	if err != nil {
		return
	}

	createJobURL := client.config.Address + "/api/jobs/"

	resp, err := client.config.Client.Post(createJobURL, "application/json", bytes.NewReader(marshalled))
	if err != nil {
		return
	}

	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("incorrect response code: %d for creating Ray Job, response body: %s", resp.StatusCode, respData)
	}

	err = json.Unmarshal(respData, &response)
	return
}

func (client *rayClusterClient) ListJobs() (response []RayJobDetailsResponse, err error) {
	getAllJobsDetailsURL := client.config.Address + "/api/jobs/"

	resp, err := client.config.Client.Get(getAllJobsDetailsURL)
	if err != nil {
		return
	}

	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("incorrect response code: %d for retrieving Ray Job details, response body: %s", resp.StatusCode, respData)
	}

	err = json.Unmarshal(respData, &response)
	return
}

func (client *rayClusterClient) GetJobDetails(jobID string) (response *RayJobDetailsResponse, err error) {
	getJobDetailsURL := client.config.Address + "/api/jobs/" + jobID

	resp, err := client.config.Client.Get(getJobDetailsURL)
	if err != nil {
		return
	}

	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("incorrect response code: %d for retrieving Ray Job details, response body: %s", resp.StatusCode, respData)
	}

	err = json.Unmarshal(respData, &response)
	return
}

func (client *rayClusterClient) GetJobLogs(jobID string) (response *RayJobLogsResponse, err error) {
	getJobLogsURL := client.config.Address + "/api/jobs/" + jobID + "/logs"

	resp, err := client.config.Client.Get(getJobLogsURL)
	if err != nil {
		return
	}

	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("incorrect response code: %d for retrieving Ray Job logs, response body: %s", resp.StatusCode, respData)
	}

	err = json.Unmarshal(respData, &response)
	return
}

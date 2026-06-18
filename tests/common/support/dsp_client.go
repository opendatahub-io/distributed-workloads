/*
Copyright 2026.

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
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

const defaultPollInterval = 30 * time.Second

type DspClient struct {
	baseURL      string
	token        string
	httpClient   *http.Client
	PollInterval time.Duration
}

func NewDspClient(routeURL, bearerToken string) *DspClient {
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //nolint:gosec
		Proxy:           http.ProxyFromEnvironment,
	}
	return &DspClient{
		baseURL:      strings.TrimRight(routeURL, "/"),
		token:        bearerToken,
		httpClient:   &http.Client{Transport: tr, Timeout: 30 * time.Second},
		PollInterval: defaultPollInterval,
	}
}

func (c *DspClient) do(req *http.Request) (*http.Response, error) {
	req.Header.Set("Authorization", "Bearer "+c.token)
	return c.httpClient.Do(req)
}

func (c *DspClient) UploadPipeline(pipelineYAML []byte, name string) (string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, err := w.CreateFormFile("uploadfile", name+".yaml")
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}
	if _, err = fw.Write(pipelineYAML); err != nil {
		return "", fmt.Errorf("write pipeline YAML: %w", err)
	}
	_ = w.WriteField("name", name)
	w.Close()

	url := c.baseURL + "/apis/v2beta1/pipelines/upload"
	req, err := http.NewRequest(http.MethodPost, url, &buf)
	if err != nil {
		return "", fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.do(req)
	if err != nil {
		return "", fmt.Errorf("upload pipeline: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("upload pipeline: HTTP %d: %s", resp.StatusCode, body)
	}

	var result struct {
		PipelineID string `json:"pipeline_id"`
	}
	if err = json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("parse upload response: %w — body: %s", err, body)
	}
	return result.PipelineID, nil
}

func (c *DspClient) CreateRun(pipelineID, experimentID, runName string, params map[string]interface{}) (string, error) {
	payload := map[string]interface{}{
		"display_name": runName,
		"pipeline_version_reference": map[string]string{
			"pipeline_id": pipelineID,
		},
		"runtime_config": map[string]interface{}{
			"parameters": params,
		},
	}
	if experimentID != "" {
		payload["experiment_id"] = experimentID
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal run payload: %w", err)
	}

	url := c.baseURL + "/apis/v2beta1/runs"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.do(req)
	if err != nil {
		return "", fmt.Errorf("create run: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("create run: HTTP %d: %s", resp.StatusCode, respBody)
	}

	var result struct {
		RunID string `json:"run_id"`
	}
	if err = json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse create run response: %w — body: %s", err, respBody)
	}
	return result.RunID, nil
}

// WaitForRunCompletion polls the run until it reaches a terminal state.
// Returns nil on SUCCEEDED; returns an error on FAILED, SKIPPED, or timeout.
func (c *DspClient) WaitForRunCompletion(t Test, runID string, timeout time.Duration) error {
	t.T().Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		state, err := c.getRunState(runID)
		if err != nil {
			return err
		}
		t.T().Logf("DSP run %s state: %s", runID, state)
		switch state {
		case "SUCCEEDED":
			return nil
		case "FAILED", "SKIPPED", "CANCELED":
			return fmt.Errorf("run %s reached terminal state: %s", runID, state)
		}
		time.Sleep(c.PollInterval)
	}
	return fmt.Errorf("run %s did not complete within %s", runID, timeout)
}

func (c *DspClient) getRunState(runID string) (string, error) {
	url := c.baseURL + "/apis/v2beta1/runs/" + runID
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("new request: %w", err)
	}
	resp, err := c.do(req)
	if err != nil {
		return "", fmt.Errorf("get run state: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("get run: HTTP %d: %s", resp.StatusCode, body)
	}

	var result struct {
		State string `json:"state"`
	}
	if err = json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("parse run response: %w", err)
	}
	return result.State, nil
}

// GetRunDetails returns the full JSON response for a run, useful for post-run assertions.
func (c *DspClient) GetRunDetails(runID string) (map[string]interface{}, error) {
	url := c.baseURL + "/apis/v2beta1/runs/" + runID
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	resp, err := c.do(req)
	if err != nil {
		return nil, fmt.Errorf("get run details: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("get run details: HTTP %d: %s", resp.StatusCode, body)
	}

	var result map[string]interface{}
	if err = json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("parse run details: %w", err)
	}
	return result, nil
}

func (c *DspClient) DeletePipeline(pipelineID string) error {
	url := c.baseURL + "/apis/v2beta1/pipelines/" + pipelineID
	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	resp, err := c.do(req)
	if err != nil {
		return fmt.Errorf("delete pipeline: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("delete pipeline: HTTP %d: %s", resp.StatusCode, body)
	}
	return nil
}

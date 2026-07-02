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
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/onsi/gomega"
)

func TestNewDspClient(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	c := NewDspClient("https://example.com", "token123")
	g.Expect(c).NotTo(gomega.BeNil())
	g.Expect(c.baseURL).To(gomega.Equal("https://example.com"))
	g.Expect(c.token).To(gomega.Equal("token123"))
}

func TestNewDspClientTrimsTrailingSlash(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	c := NewDspClient("https://example.com/", "tok")
	g.Expect(c.baseURL).To(gomega.Equal("https://example.com"))
}

func TestUploadPipelineReturnsID(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.Expect(r.Method).To(gomega.Equal(http.MethodPost))
		g.Expect(r.URL.Path).To(gomega.Equal("/apis/v2beta1/pipelines/upload"))
		g.Expect(r.Header.Get("Authorization")).To(gomega.Equal("Bearer tok"))
		g.Expect(r.Header.Get("Content-Type")).To(gomega.ContainSubstring("multipart/form-data"))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"pipeline_id":"pipe-abc","display_name":"test-sft"}`))
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	id, err := c.UploadPipeline([]byte("apiVersion: v1"), "test-sft")
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(id).To(gomega.Equal("pipe-abc"))
}

func TestUploadPipelineHTTPError(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	_, err := c.UploadPipeline([]byte("yaml"), "test")
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("HTTP 500"))
}

func TestCreateRunReturnsID(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.Expect(r.Method).To(gomega.Equal(http.MethodPost))
		g.Expect(r.URL.Path).To(gomega.Equal("/apis/v2beta1/runs"))
		g.Expect(r.Header.Get("Content-Type")).To(gomega.Equal("application/json"))

		body, _ := io.ReadAll(r.Body)
		var payload map[string]interface{}
		g.Expect(json.Unmarshal(body, &payload)).To(gomega.Succeed())
		g.Expect(payload["display_name"]).To(gomega.Equal("e2e-sft-run"))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"run_id":"run-123"}`))
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	params := map[string]interface{}{"model": "Qwen/Qwen2.5-1.5B-Instruct"}
	id, err := c.CreateRun("pipe-abc", "", "e2e-sft-run", params)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(id).To(gomega.Equal("run-123"))
}

func TestCreateRunWithExperimentID(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var payload map[string]interface{}
		g.Expect(json.Unmarshal(body, &payload)).To(gomega.Succeed())
		g.Expect(payload["experiment_id"]).To(gomega.Equal("exp-1"))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"run_id":"run-456"}`))
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	id, err := c.CreateRun("pipe-abc", "exp-1", "run", nil)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(id).To(gomega.Equal("run-456"))
}

func TestWaitForRunCompletionSucceeded(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.Expect(r.URL.Path).To(gomega.Equal("/apis/v2beta1/runs/run-1"))
		callCount++
		state := "RUNNING"
		if callCount >= 2 {
			state = "SUCCEEDED"
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"run_id":"run-1","state":"` + state + `"}`))
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	c.PollInterval = 10 * time.Millisecond
	test := With(t)
	err := c.WaitForRunCompletion(test, "run-1", 5*time.Second)
	g.Expect(err).NotTo(gomega.HaveOccurred())
}

func TestWaitForRunCompletionFailed(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"run_id":"run-2","state":"FAILED"}`))
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	c.PollInterval = 10 * time.Millisecond
	test := With(t)
	err := c.WaitForRunCompletion(test, "run-2", 5*time.Second)
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("FAILED"))
}

func TestGetRunDetails(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.Expect(r.URL.Path).To(gomega.Equal("/apis/v2beta1/runs/run-1"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"run_id":"run-1","state":"SUCCEEDED","pipeline_spec":{"root":{}}}`))
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	details, err := c.GetRunDetails("run-1")
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(details["state"]).To(gomega.Equal("SUCCEEDED"))
	g.Expect(details["run_id"]).To(gomega.Equal("run-1"))
}

func TestDeletePipeline(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.Expect(r.Method).To(gomega.Equal(http.MethodDelete))
		g.Expect(r.URL.Path).To(gomega.Equal("/apis/v2beta1/pipelines/pipe-abc"))
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	err := c.DeletePipeline("pipe-abc")
	g.Expect(err).NotTo(gomega.HaveOccurred())
}

func TestDeletePipelineHTTPError(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"not found"}`))
	}))
	defer srv.Close()

	c := NewDspClient(srv.URL, "tok")
	err := c.DeletePipeline("nonexistent")
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("HTTP 404"))
}

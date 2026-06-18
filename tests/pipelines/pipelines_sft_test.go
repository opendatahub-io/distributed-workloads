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

package pipelines

import (
	"fmt"
	"testing"

	. "github.com/onsi/gomega"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	"github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	sftPipelineYAMLPath = "resources/sft_pipeline.yaml"
)

func TestSftPipelineRun(t *testing.T) {
	Tags(t, Tier1, Gpu(support.NVIDIA))
	test := support.With(t)

	dspURL := GetDspRouteURL(test)
	dspToken := GetDspBearerToken(test)
	client := support.NewDspClient(dspURL, dspToken)

	pipelineYAML := readFile(test, sftPipelineYAMLPath)

	pipelineID, err := client.UploadPipeline(pipelineYAML, "e2e-sft-pipeline")
	test.Expect(err).NotTo(HaveOccurred(), "failed to upload SFT pipeline")

	t.Cleanup(func() {
		if err := client.DeletePipeline(pipelineID); err != nil {
			t.Logf("warning: failed to delete pipeline %s: %v", pipelineID, err)
		}
	})

	dataURI := buildSftDataURI(test)

	params := map[string]interface{}{
		"phase_01_dataset_man_data_uri": dataURI,
		"phase_01_dataset_opt_subset":   "100",
		"phase_02_train_man_model":      "Qwen/Qwen2.5-1.5B-Instruct",
		"phase_02_train_man_epochs":     "1",
		"phase_02_train_man_workers":    "1",
		"phase_02_train_man_gpu":        "1",
		"phase_02_train_opt_runtime":    "training-hub",
		"phase_03_eval_man_tasks":       "[]",
		"phase_04_registry_man_address": "",
	}

	runID, err := client.CreateRun(pipelineID, "", "e2e-sft-run", params)
	test.Expect(err).NotTo(HaveOccurred(), "failed to create SFT pipeline run")

	t.Logf("SFT pipeline run created: %s", runID)

	err = client.WaitForRunCompletion(test, runID, PipelineRunTimeout)
	test.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("SFT pipeline run %s did not succeed", runID))
}

func buildSftDataURI(test support.Test) string {
	endpoint, endpointOK := support.GetStorageBucketDefaultEndpoint()
	bucket, bucketOK := support.GetStorageBucketName()
	prefix, prefixOK := support.GetStorageBucketSftDir()

	if endpointOK && bucketOK && prefixOK && endpoint != "" && bucket != "" && prefix != "" {
		return fmt.Sprintf("s3://%s/%s", bucket, prefix)
	}

	test.T().Log("S3 not fully configured; falling back to HuggingFace dataset URI")
	return "hf://ibm/merlinite-9b-lab-processed"
}

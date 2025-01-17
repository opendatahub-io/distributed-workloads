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

package kfto

import (
	"embed"
	"time"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusmodel "github.com/prometheus/common/model"

	corev1 "k8s.io/api/core/v1"
)

type Gpu struct {
	ResourceLabel                 string
	PrometheusGpuUtilizationLabel string
}

var (
	NVIDIA = Gpu{ResourceLabel: "nvidia.com/gpu", PrometheusGpuUtilizationLabel: "DCGM_FI_DEV_GPU_UTIL"}
	AMD    = Gpu{ResourceLabel: "amd.com/gpu"}
	CPU    = Gpu{ResourceLabel: "cpu"}
)

//go:embed resources/*
var files embed.FS

func ReadFile(t Test, fileName string) []byte {
	t.T().Helper()
	file, err := files.ReadFile(fileName)
	t.Expect(err).NotTo(HaveOccurred())
	return file
}

func OpenShiftPrometheusGpuUtil(test Test, pod corev1.Pod, gpu Gpu) func(g Gomega) prometheusmodel.Vector {
	return func(g Gomega) prometheusmodel.Vector {
		prometheusApiClient := GetOpenShiftPrometheusApiClient(test)
		result, warnings, err := prometheusApiClient.Query(test.Ctx(), gpu.PrometheusGpuUtilizationLabel, time.Now(), prometheusapiv1.WithTimeout(5*time.Second))
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(warnings).Should(HaveLen(0))

		var util prometheusmodel.Vector
		for _, sample := range result.(prometheusmodel.Vector) {
			if string(sample.Metric["exported_namespace"]) == pod.GetNamespace() && string(sample.Metric["exported_pod"]) == pod.GetName() {
				util = append(util, sample)
			}
		}

		return util
	}
}

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

	gonanoid "github.com/matoous/go-nanoid/v2"
	"github.com/onsi/gomega"
	. "github.com/onsi/gomega"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusmodel "github.com/prometheus/common/model"

	corev1 "k8s.io/api/core/v1"

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

func OpenShiftPrometheusGpuUtil(test support.Test, pod corev1.Pod, gpu support.Accelerator) func(g Gomega) prometheusmodel.Vector {
	return func(g Gomega) prometheusmodel.Vector {
		prometheusApiClient := support.GetOpenShiftPrometheusApiClient(test)
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

type compare[T any] func(T, T) bool

func upsert[T any](items []T, item T, predicate compare[T]) []T {
	for i, t := range items {
		if predicate(t, item) {
			items[i] = item
			return items
		}
	}
	return append(items, item)
}

func withEnvVarName(name string) compare[corev1.EnvVar] {
	return func(e1, e2 corev1.EnvVar) bool {
		return e1.Name == name
	}
}

// Adds a unique suffix to the provided string
func uniqueSuffix(prefix string) string {
	suffix := gonanoid.MustGenerate("1234567890abcdef", 4)
	return prefix + "-" + suffix
}

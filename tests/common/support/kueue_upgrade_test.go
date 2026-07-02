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
	"testing"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	kueuev1beta2 "sigs.k8s.io/kueue/apis/kueue/v1beta2"
)

func TestAddUpgradeResourceBaseline(t *testing.T) {
	test := NewTest(t)

	data := map[string]string{}
	spec := kueuev1beta2.ClusterQueueSpec{
		ResourceGroups: []kueuev1beta2.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName(NVIDIA.ResourceLabel)},
				Flavors: []kueuev1beta2.FlavorQuotas{
					{
						Resources: []kueuev1beta2.ResourceQuota{
							{
								Name:         corev1.ResourceName(NVIDIA.ResourceLabel),
								NominalQuota: resource.MustParse("1"),
							},
						},
					},
				},
			},
		},
	}

	err := AddUpgradeResourceBaseline(data, "generation-key", "spec-key", 3, spec)
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.Expect(data["generation-key"]).To(gomega.Equal("3"))

	var restored kueuev1beta2.ClusterQueueSpec
	err = json.Unmarshal([]byte(data["spec-key"]), &restored)
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.Expect(restored.ResourceGroups).To(gomega.HaveLen(1))
	test.Expect(restored.ResourceGroups[0].Flavors[0].Resources[0].NominalQuota).To(gomega.Equal(resource.MustParse("1")))
}

func TestClusterQueueNominalGPUQuota(t *testing.T) {
	test := NewTest(t)

	clusterQueue := &kueuev1beta2.ClusterQueue{
		Spec: kueuev1beta2.ClusterQueueSpec{
			ResourceGroups: []kueuev1beta2.ResourceGroup{
				{
					Flavors: []kueuev1beta2.FlavorQuotas{
						{
							Resources: []kueuev1beta2.ResourceQuota{
								{
									Name:         corev1.ResourceName(NVIDIA.ResourceLabel),
									NominalQuota: resource.MustParse("2"),
								},
							},
						},
					},
				},
			},
		},
	}

	quota := ClusterQueueNominalGPUQuota(clusterQueue, NVIDIA.ResourceLabel)
	test.Expect(quota).To(gomega.Equal(resource.MustParse("2")))
	test.Expect(ClusterQueueNominalGPUQuota(clusterQueue, "missing")).To(gomega.Equal(resource.Quantity{}))
}

func TestVerifyUpgradeResourceSpecIntegrity(t *testing.T) {
	test := NewTest(t)
	configMap := &corev1.ConfigMap{
		Data: map[string]string{
			"gen-key":              "5",
			"spec-key":             `{"nodeLabels":{"nvidia.com/gpu.present":"true"}}`,
			UpgradeRHOAIVersionKey: "3.4.0",
		},
	}
	spec := kueuev1beta2.ResourceFlavorSpec{
		NodeLabels: map[string]string{
			"nvidia.com/gpu.present": "true",
		},
	}

	VerifyUpgradeResourceSpecIntegrity(test, "ResourceFlavor", 5, spec, configMap, "gen-key", "spec-key")
}

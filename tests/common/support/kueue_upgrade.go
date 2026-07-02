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
	"fmt"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kueuev1beta2 "sigs.k8s.io/kueue/apis/kueue/v1beta2"
)

const UpgradeRHOAIVersionKey = "rhoai-version"

func AddUpgradeResourceBaseline(data map[string]string, genKey, specKey string, generation int64, spec interface{}) error {
	specJSON, err := json.Marshal(spec)
	if err != nil {
		return err
	}
	data[genKey] = fmt.Sprintf("%d", generation)
	data[specKey] = string(specJSON)
	return nil
}

func StoreUpgradeBaseline(test Test, namespace, configMapName string, data map[string]string) {
	test.T().Helper()

	data[UpgradeRHOAIVersionKey] = GetRHOAIVersionFromDSCI(test)
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: namespace,
		},
		Data: data,
	}
	_ = test.Client().Core().CoreV1().ConfigMaps(namespace).Delete(test.Ctx(), configMapName, metav1.DeleteOptions{})
	_, err := test.Client().Core().CoreV1().ConfigMaps(namespace).Create(test.Ctx(), configMap, metav1.CreateOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.T().Logf("Stored upgrade baseline in ConfigMap %s/%s", namespace, configMapName)
}

func VerifyUpgradeResourceSpecIntegrity(test Test, resourceName string, generation int64, spec interface{},
	configMap *corev1.ConfigMap, genKey, specKey string) {
	test.T().Helper()

	expectedGen := configMap.Data[genKey]
	actualGen := fmt.Sprintf("%d", generation)
	if actualGen != expectedGen {
		currentSpecJSON, _ := json.Marshal(spec)
		test.T().Logf("%s generation changed during upgrade (%s to %s)", resourceName, expectedGen, actualGen)
		test.T().Logf("Pre-upgrade %s spec: %s", resourceName, configMap.Data[specKey])
		test.T().Logf("Post-upgrade %s spec: %s", resourceName, currentSpecJSON)
	}
	test.Expect(actualGen).To(gomega.Equal(expectedGen),
		"%s spec should be unchanged after upgrade (generation %s, expected %s)", resourceName, actualGen, expectedGen)
	test.T().Logf("%s generation unchanged after upgrade: %s", resourceName, actualGen)
}

func ClusterQueueNominalGPUQuota(clusterQueue *kueuev1beta2.ClusterQueue, gpuResourceLabel string) resource.Quantity {
	for _, resourceGroup := range clusterQueue.Spec.ResourceGroups {
		for _, flavor := range resourceGroup.Flavors {
			for _, quota := range flavor.Resources {
				if string(quota.Name) == gpuResourceLabel {
					return quota.NominalQuota
				}
			}
		}
	}
	return resource.Quantity{}
}

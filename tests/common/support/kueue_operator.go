/*
Copyright 2025.

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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kueueoperatorv1 "github.com/openshift/kueue-operator/pkg/apis/kueueoperator/v1"
)

func GetKueueCR(t Test, name string) (*kueueoperatorv1.Kueue, error) {
	t.T().Helper()
	return t.Client().KueueOperator().KueueV1().Kueues().Get(t.Ctx(), name, metav1.GetOptions{})
}

func KueueCRExists(t Test, name string) func(g gomega.Gomega) bool {
	return func(g gomega.Gomega) bool {
		_, err := t.Client().KueueOperator().KueueV1().Kueues().Get(t.Ctx(), name, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return false
		}
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return true
	}
}

func KueueCR(t Test, name string) func(g gomega.Gomega) *kueueoperatorv1.Kueue {
	return func(g gomega.Gomega) *kueueoperatorv1.Kueue {
		kueue, err := GetKueueCR(t, name)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return kueue
	}
}

func KueueCRConditionAvailable(kueue *kueueoperatorv1.Kueue) metav1.ConditionStatus {
	return getKueueCRCondition(kueue, "Available")
}

func KueueCRConditionCertManagerAvailable(kueue *kueueoperatorv1.Kueue) metav1.ConditionStatus {
	return getKueueCRCondition(kueue, "CertManagerAvailable")
}

func getKueueCRCondition(kueue *kueueoperatorv1.Kueue, conditionType string) metav1.ConditionStatus {
	if kueue == nil {
		return metav1.ConditionUnknown
	}
	for _, condition := range kueue.Status.Conditions {
		if condition.Type == conditionType {
			return metav1.ConditionStatus(condition.Status)
		}
	}
	return metav1.ConditionUnknown
}

func KueueCRFrameworks(kueue *kueueoperatorv1.Kueue) []string {
	if kueue == nil {
		return nil
	}
	var frameworks []string
	for _, f := range kueue.Spec.Config.Integrations.Frameworks {
		frameworks = append(frameworks, string(f))
	}
	return frameworks
}

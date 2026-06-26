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
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	kueueoperatorv1 "github.com/openshift/kueue-operator/pkg/apis/kueueoperator/v1"
)

const (
	KueueCRName         = "cluster"
	PyTorchJobFramework = "PyTorchJob"
	TrainJobFramework   = "TrainJob"
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

func SetupKueue(test Test, initialKueueState string, expectedFrameworks ...string) {
	test.T().Cleanup(func() {
		if test.T().Failed() {
			StoreKueueDiagnostics(test)
		}
	})

	if initialKueueState == "Unmanaged" {
		test.T().Log("SetupKueue: Kueue managementState was already Unmanaged, next verify status of 'Kueue CR'")
		VerifyKueueReady(test, expectedFrameworks...)
		return
	}

	test.T().Log("SetupKueue: Setting kueue to Unmanaged managementState in DataScienceCluster...")
	err := SetComponentState(test, DefaultDSCName, "kueue", StateUnmanaged, 2*time.Minute)
	test.Expect(err).NotTo(gomega.HaveOccurred(), "Should be able to set DSC kueue to Unmanaged")

	// Verify kueue status is Unmanaged and KueueReady condition is True
	test.Eventually(DSCResource(test, DefaultDSCName), TestTimeoutShort).Should(gomega.And(
		gomega.WithTransform(func(dsc *unstructured.Unstructured) string {
			return ComponentStatusManagementState(dsc, "kueue")
		}, gomega.Equal("Unmanaged")),
		gomega.WithTransform(func(dsc *unstructured.Unstructured) string {
			return ComponentConditionStatus(dsc, "KueueReady")
		}, gomega.Equal("True")),
	))
	test.T().Log("SetupKueue: Kueue is set to Unmanaged managementState successfully")

	VerifyKueueReady(test, expectedFrameworks...)
}

func StoreKueueDiagnostics(t Test) {
	t.T().Helper()
	t.T().Log("Collecting Kueue diagnostics...")

	storeKueueCRState(t)
	storeDSCState(t)
	storeKueueOperatorLogs(t)
}

func storeKueueCRState(t Test) {
	t.T().Helper()
	kueueCR, err := GetKueueCR(t, KueueCRName)
	if err != nil {
		t.T().Logf("Failed to get Kueue CR for diagnostics: %v", err)
		return
	}
	data, err := json.MarshalIndent(kueueCR, "", "  ")
	if err != nil {
		t.T().Logf("Failed to marshal Kueue CR: %v", err)
		return
	}
	WriteToOutputDir(t, "kueue-cr-state", Log, data)
	t.T().Log("Stored Kueue CR state")
}

func storeDSCState(t Test) {
	t.T().Helper()
	dsc, err := GetDSC(t, DefaultDSCName)
	if err != nil {
		t.T().Logf("Failed to get DSC for diagnostics: %v", err)
		return
	}
	data, err := json.MarshalIndent(dsc.Object, "", "  ")
	if err != nil {
		t.T().Logf("Failed to marshal DSC: %v", err)
		return
	}
	WriteToOutputDir(t, "dsc-state", Log, data)
	t.T().Log("Stored DSC state")
}

func storeKueueOperatorLogs(t Test) {
	t.T().Helper()

	namespace, pods := findKueueOperatorPods(t)
	if len(pods) == 0 {
		t.T().Log("No Kueue operator pods found")
		return
	}

	tailLines := int64(500)
	for _, pod := range pods {
		for _, container := range pod.Spec.Containers {
			options := corev1.PodLogOptions{Container: container.Name, TailLines: &tailLines}
			stream, err := t.Client().Core().CoreV1().Pods(namespace).GetLogs(pod.Name, &options).Stream(t.Ctx())
			if err != nil {
				t.T().Logf("Failed to get logs for %s/%s: %v", pod.Name, container.Name, err)
				continue
			}
			data, readErr := io.ReadAll(stream)
			_ = stream.Close()
			if readErr != nil {
				t.T().Logf("Failed to read logs for %s/%s: %v", pod.Name, container.Name, readErr)
				continue
			}
			fileName := fmt.Sprintf("kueue-operator-%s-%s", pod.Name, container.Name)
			WriteToOutputDir(t, fileName, Log, data)
			t.T().Logf("Stored Kueue operator logs for %s/%s", pod.Name, container.Name)
		}
	}
}

func findKueueOperatorPods(t Test) (string, []corev1.Pod) {
	t.T().Helper()
	selector := "control-plane=controller-manager,app.kubernetes.io/name=kueue"
	for _, namespace := range []string{"openshift-kueue-operator", "kueue-system"} {
		pods := GetPods(t, namespace, metav1.ListOptions{LabelSelector: selector})
		if len(pods) > 0 {
			return namespace, pods
		}
	}
	return "", nil
}

func VerifyKueueReady(test Test, expectedFrameworks ...string) {
	test.Eventually(KueueCRExists(test, KueueCRName), TestTimeoutMedium).Should(
		gomega.BeTrue(),
		"Kueue CR should be created when kueue is set to Unmanaged in DataScienceCluster",
	)
	test.T().Logf("Kueue CR '%s' exists", KueueCRName)

	test.T().Log("Waiting for Kueue CR to be ready...")
	test.Eventually(KueueCR(test, KueueCRName), TestTimeoutLong).Should(
		gomega.And(
			gomega.WithTransform(KueueCRConditionAvailable, gomega.Equal(metav1.ConditionTrue)),
			gomega.WithTransform(KueueCRConditionCertManagerAvailable, gomega.Equal(metav1.ConditionTrue)),
		),
	)
	test.T().Log("Kueue CR is ready")

	for _, framework := range expectedFrameworks {
		test.T().Logf("Verifying %s framework is present in Kueue CR...", framework)
		test.Eventually(KueueCR(test, KueueCRName), TestTimeoutShort).Should(
			gomega.WithTransform(KueueCRFrameworks, gomega.ContainElement(framework)),
		)
		test.T().Logf("%s framework is present in Kueue CR", framework)
	}
}

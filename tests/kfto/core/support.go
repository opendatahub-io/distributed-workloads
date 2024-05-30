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

package core

import (
	"embed"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
)

//go:embed *.json
var files embed.FS

func ReadFile(t Test, fileName string) []byte {
	t.T().Helper()
	file, err := files.ReadFile(fileName)
	t.Expect(err).NotTo(HaveOccurred())
	return file
}

func PytorchJob(t Test, namespace, name string) func(g Gomega) *kftov1.PyTorchJob {
	return func(g Gomega) *kftov1.PyTorchJob {
		job, err := t.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		return job
	}
}

func PytorchJobConditionRunning(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PytorchJobCondition(job, kftov1.JobRunning)
}

func PytorchJobConditionSucceeded(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PytorchJobCondition(job, kftov1.JobSucceeded)
}

func PytorchJobConditionSuspended(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PytorchJobCondition(job, kftov1.JobSuspended)
}

func PytorchJobCondition(job *kftov1.PyTorchJob, conditionType kftov1.JobConditionType) corev1.ConditionStatus {
	for _, condition := range job.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status
		}
	}
	return corev1.ConditionUnknown
}

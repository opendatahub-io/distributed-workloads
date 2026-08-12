/*
Copyright 2024.

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
	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func PyTorchJob(t Test, namespace, name string) func(g gomega.Gomega) *kftov1.PyTorchJob {
	return func(g gomega.Gomega) *kftov1.PyTorchJob {
		job, err := t.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return job
	}
}

func PyTorchJobs(t Test, namespace string) func(g gomega.Gomega) []kftov1.PyTorchJob {
	return func(g gomega.Gomega) []kftov1.PyTorchJob {
		jobs, err := t.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).List(t.Ctx(), metav1.ListOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return jobs.Items
	}
}

func PyTorchJobConditionRunning(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PyTorchJobCondition(job, kftov1.JobRunning)
}

func PyTorchJobConditionSucceeded(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PyTorchJobCondition(job, kftov1.JobSucceeded)
}

func PyTorchJobConditionSuspended(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PyTorchJobCondition(job, kftov1.JobSuspended)
}

func PyTorchJobConditionFailed(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PyTorchJobCondition(job, kftov1.JobFailed)
}

func PyTorchJobCondition(job *kftov1.PyTorchJob, conditionType kftov1.JobConditionType) corev1.ConditionStatus {
	for _, condition := range job.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status
		}
	}
	return corev1.ConditionUnknown
}

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
	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TrainJob(t Test, namespace, name string) func(g gomega.Gomega) *trainerv1alpha1.TrainJob {
	return func(g gomega.Gomega) *trainerv1alpha1.TrainJob {
		job, err := t.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return job
	}
}

func TrainJobs(t Test, namespace string) func(g gomega.Gomega) []trainerv1alpha1.TrainJob {
	return func(g gomega.Gomega) []trainerv1alpha1.TrainJob {
		jobs, err := t.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).List(t.Ctx(), metav1.ListOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return jobs.Items
	}
}

func TrainJobConditionComplete(job *trainerv1alpha1.TrainJob) metav1.ConditionStatus {
	return TrainJobCondition(job, trainerv1alpha1.TrainJobComplete)
}

func TrainJobConditionFailed(job *trainerv1alpha1.TrainJob) metav1.ConditionStatus {
	return TrainJobCondition(job, trainerv1alpha1.TrainJobFailed)
}

func TrainJobConditionSuspended(job *trainerv1alpha1.TrainJob) metav1.ConditionStatus {
	return TrainJobCondition(job, trainerv1alpha1.TrainJobSuspended)
}

func TrainJobCondition(job *trainerv1alpha1.TrainJob, conditionType string) metav1.ConditionStatus {
	for _, condition := range job.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status
		}
	}
	return metav1.ConditionUnknown
}

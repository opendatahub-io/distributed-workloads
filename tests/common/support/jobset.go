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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"
)

func JobSets(t Test, namespace string) func(g gomega.Gomega) []*jobsetv1alpha2.JobSet {
	return func(g gomega.Gomega) []*jobsetv1alpha2.JobSet {
		jobsets, err := t.Client().JobSet().JobsetV1alpha2().JobSets(namespace).List(t.Ctx(), metav1.ListOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())

		jobsetsp := []*jobsetv1alpha2.JobSet{}
		for _, v := range jobsets.Items {
			jobsetsp = append(jobsetsp, &v)
		}

		return jobsetsp
	}
}

func SingleJobSet(t Test, namespace string) func(g gomega.Gomega) *jobsetv1alpha2.JobSet {
	return func(g gomega.Gomega) *jobsetv1alpha2.JobSet {
		jobsets := JobSets(t, namespace)(g)
		g.Expect(jobsets).To(gomega.HaveLen(1))
		return jobsets[0]
	}
}

func JobSetReplicatedJobsCount(jobset *jobsetv1alpha2.JobSet) int {
	if jobset == nil {
		return 0
	}
	return len(jobset.Spec.ReplicatedJobs)
}

func JobSetCondition(jobset *jobsetv1alpha2.JobSet, conditionType jobsetv1alpha2.JobSetConditionType) metav1.ConditionStatus {
	if jobset == nil {
		return metav1.ConditionUnknown
	}
	for _, condition := range jobset.Status.Conditions {
		if string(condition.Type) == string(conditionType) {
			return condition.Status
		}
	}
	return metav1.ConditionUnknown
}

func JobSetConditionFailed(jobset *jobsetv1alpha2.JobSet) metav1.ConditionStatus {
	return JobSetCondition(jobset, jobsetv1alpha2.JobSetFailed)
}

func JobSetConditionCompleted(jobset *jobsetv1alpha2.JobSet) metav1.ConditionStatus {
	return JobSetCondition(jobset, jobsetv1alpha2.JobSetCompleted)
}

func JobSetFailureMessage(jobset *jobsetv1alpha2.JobSet) string {
	if jobset == nil {
		return ""
	}
	for _, condition := range jobset.Status.Conditions {
		if string(condition.Type) == string(jobsetv1alpha2.JobSetFailed) && condition.Status == metav1.ConditionTrue {
			return condition.Message
		}
	}
	return ""
}

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
	"strings"

	batchv1 "k8s.io/api/batch/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func GetJobByNamePattern(test Test, namespace, pattern string) (*batchv1.Job, error) {
	test.T().Helper()

	jobs, err := test.Client().Core().BatchV1().Jobs(namespace).List(test.Ctx(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	for i := range jobs.Items {
		job := &jobs.Items[i]
		if strings.Contains(job.Name, pattern) {
			return job, nil
		}
	}

	return nil, nil
}

func GetAllJobs(test Test, namespace string) ([]batchv1.Job, error) {
	test.T().Helper()

	jobs, err := test.Client().Core().BatchV1().Jobs(namespace).List(test.Ctx(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	return jobs.Items, nil
}

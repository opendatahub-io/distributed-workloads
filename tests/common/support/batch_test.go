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

package support

import (
	"testing"

	"github.com/onsi/gomega"

	batchv1 "k8s.io/api/batch/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetJob(t *testing.T) {

	test := NewTest(t)

	Job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-job-1",
			Namespace: "my-namespace",
		},
	}

	test.client.Core().BatchV1().Jobs("my-namespace").Create(test.ctx, Job, metav1.CreateOptions{})

	// Call the Job function using the fake client
	jobs := GetJob(test, "my-namespace", "my-job-1")

	test.Expect(jobs.Name).To(gomega.Equal("my-job-1"))
	test.Expect(jobs.Namespace).To(gomega.Equal("my-namespace"))

}

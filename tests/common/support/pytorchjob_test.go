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
	"testing"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPyTorchJob(t *testing.T) {

	test := NewTest(t)

	job := &kftov1.PyTorchJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-job-1",
			Namespace: "my-namespace",
		},
	}

	test.client.Kubeflow().KubeflowV1().PyTorchJobs("my-namespace").Create(test.ctx, job, metav1.CreateOptions{})

	pyTorchJob := PyTorchJob(test, "my-namespace", "my-job-1")(test)
	test.Expect(pyTorchJob.Name).To(gomega.Equal("my-job-1"))
	test.Expect(pyTorchJob.Namespace).To(gomega.Equal("my-namespace"))
}

func TestPyTorchJobs(t *testing.T) {

	test := NewTest(t)

	jobOne := &kftov1.PyTorchJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-job-1",
			Namespace: "my-namespace",
		},
	}
	jobTwo := &kftov1.PyTorchJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-job-2",
			Namespace: "my-namespace",
		},
	}

	test.client.Kubeflow().KubeflowV1().PyTorchJobs("my-namespace").Create(test.ctx, jobOne, metav1.CreateOptions{})
	test.client.Kubeflow().KubeflowV1().PyTorchJobs("my-namespace").Create(test.ctx, jobTwo, metav1.CreateOptions{})

	pyTorchJobs := PyTorchJobs(test, "my-namespace")(test)

	test.Expect(len(pyTorchJobs)).To(gomega.Equal(2))
	test.Expect(pyTorchJobs[0].Name).To(gomega.Equal("my-job-1"))
	test.Expect(pyTorchJobs[0].Namespace).To(gomega.Equal("my-namespace"))
	test.Expect(pyTorchJobs[1].Name).To(gomega.Equal("my-job-2"))
	test.Expect(pyTorchJobs[1].Namespace).To(gomega.Equal("my-namespace"))
}

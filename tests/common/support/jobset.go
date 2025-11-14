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
	. "github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	jobsetGroup   = "jobset.x-k8s.io"
	jobsetVersion = "v1alpha2"
)

var jobsetOperatorGVR = schema.GroupVersionResource{
	Group:    jobsetGroup,
	Version:  jobsetVersion,
	Resource: "jobsets",
}

func GetSingleJobSet(test Test, namespace string) (*unstructured.Unstructured, error) {
	test.T().Helper()

	jobsets, err := test.Client().Dynamic().Resource(jobsetOperatorGVR).Namespace(namespace).List(
		test.Ctx(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	test.Expect(jobsets.Items).To(HaveLen(1), "Expected exactly one JobSet in namespace")
	return &jobsets.Items[0], nil
}

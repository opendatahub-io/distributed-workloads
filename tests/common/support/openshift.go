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
	"github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/api/errors"
)

func IsOpenShift(test Test) bool {
	test.T().Helper()
	_, err := test.Client().Core().Discovery().ServerResourcesForGroupVersion("image.openshift.io/v1")
	if err != nil && errors.IsNotFound(err) {
		return false
	}
	test.Expect(err).NotTo(gomega.HaveOccurred())
	return true
}

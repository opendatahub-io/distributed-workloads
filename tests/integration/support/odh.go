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
	cfosupport "github.com/project-codeflare/codeflare-operator/test/support"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func GetODHCodeFlareImageStreamTag(test cfosupport.Test) string {
	test.T().Helper()

	cfis, err := test.Client().Image().ImageV1().ImageStreams(GetOpenDataHubNamespace()).Get(test.Ctx(), "codeflare-notebook", metav1.GetOptions{})
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.Expect(cfis.Spec.Tags).To(gomega.HaveLen(1))
	return cfis.Spec.Tags[0].Name
}

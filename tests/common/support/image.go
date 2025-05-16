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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	imagev1 "github.com/openshift/api/image/v1"
)

func GetImageStream(t Test, namespace string, name string) *imagev1.ImageStream {
	t.T().Helper()

	is, err := t.Client().Image().ImageV1().ImageStreams(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())

	return is
}

func GetImageStreamTag(t Test, namespace string, name string) *imagev1.ImageStreamTag {
	t.T().Helper()

	istag, err := t.Client().Image().ImageV1().ImageStreamTags(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())

	return istag
}

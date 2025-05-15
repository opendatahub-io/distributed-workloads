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
	"testing"

	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	imagev1 "github.com/openshift/api/image/v1"
)

func TestGetImageStream(t *testing.T) {

	test := NewTest(t)

	ImageStream := &imagev1.ImageStream{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-imagestream-1",
			Namespace: "my-namespace",
		},
	}

	test.client.Image().ImageV1().ImageStreams("my-namespace").Create(test.ctx, ImageStream, metav1.CreateOptions{})

	image := GetImageStream(test, "my-namespace", "my-imagestream-1")

	test.Expect(image.Name).To(gomega.Equal("my-imagestream-1"))
	test.Expect(image.Namespace).To(gomega.Equal("my-namespace"))
}

func TestGetImageStreamTag(t *testing.T) {

	test := NewTest(t)

	imageStreamTag := &imagev1.ImageStreamTag{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-imagestreamtag",
			Namespace: "my-namespace",
		},
	}

	test.client.Image().ImageV1().ImageStreamTags("my-namespace").Create(test.ctx, imageStreamTag, metav1.CreateOptions{})

	imageTag := GetImageStreamTag(test, "my-namespace", "my-imagestreamtag")

	test.Expect(imageTag.Name).To(gomega.Equal("my-imagestreamtag"))
	test.Expect(imageTag.Namespace).To(gomega.Equal("my-namespace"))
}

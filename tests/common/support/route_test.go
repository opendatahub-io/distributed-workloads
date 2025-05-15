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

	routev1 "github.com/openshift/api/route/v1"
)

func TestGetRoute(t *testing.T) {

	test := NewTest(t)

	route := &routev1.Route{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-1",
			Namespace: "my-namespace",
		},
	}

	test.client.Route().RouteV1().Routes("my-namespace").Create(test.ctx, route, metav1.CreateOptions{})

	routes := GetRoute(test, "my-namespace", "test-1")

	test.Expect(routes.Name).To(gomega.Equal("test-1"))
	test.Expect(routes.Namespace).To(gomega.Equal("my-namespace"))

}

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
	"os"
	"testing"

	. "github.com/onsi/gomega"
)

func TestCreateOrGetTestNamespaceCreatingNamespace(t *testing.T) {
	test := NewTest(t)

	namespace := test.CreateOrGetTestNamespace()

	test.Expect(namespace).NotTo(BeNil())
	test.Expect(namespace.GenerateName).To(Equal("test-ns-"))
}

func TestCreateOrGetTestNamespaceGettingExistingNamespace(t *testing.T) {
	test := NewTest(t)

	CreateTestNamespaceWithName(test, "test-namespace")
	os.Setenv(testNamespaceNameEnvVar, "test-namespace")
	defer os.Unsetenv(testNamespaceNameEnvVar)

	namespace := test.CreateOrGetTestNamespace()

	test.Expect(namespace).NotTo(BeNil())
	test.Expect(namespace.Name).To(Equal("test-namespace"))
}

func TestCreateOrGetTestNamespaceGettingNonExistingNamespace(t *testing.T) {
	test := NewTest(t)

	os.Setenv(testNamespaceNameEnvVar, "non-existing-namespace")
	defer os.Unsetenv(testNamespaceNameEnvVar)

	namespace := test.CreateOrGetTestNamespace()

	test.Expect(namespace).NotTo(BeNil())
	test.Expect(namespace.Name).To(Equal("non-existing-namespace"))
}

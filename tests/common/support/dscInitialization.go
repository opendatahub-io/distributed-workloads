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
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const DefaultDSCIName = "default-dsci"

var DsciGVR = schema.GroupVersionResource{
	Group:    "dscinitialization.opendatahub.io",
	Version:  "v2",
	Resource: "dscinitializations",
}

func GetDSCI(test Test, name string) (*unstructured.Unstructured, error) {
	return test.Client().Dynamic().Resource(DsciGVR).Get(test.Ctx(), name, metav1.GetOptions{})
}

func GetApplicationsNamespaceFromDSCI(test Test, dsciName string) (string, error) {
	dsci, err := GetDSCI(test, dsciName)
	if err != nil {
		return "", err
	}
	namespace, found, err := unstructured.NestedString(dsci.Object, "spec", "applicationsNamespace")
	if err != nil {
		return "", fmt.Errorf("failed to get applicationsNamespace from DSCI %s: %w", dsciName, err)
	}
	if !found {
		return "", fmt.Errorf("applicationsNamespace field not found in DSCI %s", dsciName)
	}
	return namespace, nil
}

/*
Copyright 2026.

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
	"strings"

	"github.com/operator-framework/api/pkg/operators/v1alpha1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// FindCSVByPrefix finds a CSV in the given namespace that starts with the given prefix
func FindCSVByPrefix(test Test, namespace, prefix string) (*v1alpha1.ClusterServiceVersion, error) {
	csvList, err := test.Client().OLM().OperatorsV1alpha1().ClusterServiceVersions(namespace).List(
		test.Ctx(), metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list CSVs in namespace %s: %w", namespace, err)
	}

	var matchingCSVs []v1alpha1.ClusterServiceVersion
	for _, csv := range csvList.Items {
		if strings.HasPrefix(csv.Name, prefix) {
			matchingCSVs = append(matchingCSVs, csv)
		}
	}

	switch len(matchingCSVs) {
	case 0:
		return nil, fmt.Errorf("no CSV found with prefix %s in namespace %s", prefix, namespace)
	case 1:
		return &matchingCSVs[0], nil
	default:
		var names []string
		for _, csv := range matchingCSVs {
			names = append(names, csv.Name)
		}
		return nil, fmt.Errorf("multiple CSVs found with prefix %s in namespace %s: %v", prefix, namespace, names)
	}
}

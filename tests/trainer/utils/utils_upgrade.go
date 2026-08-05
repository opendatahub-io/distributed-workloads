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

package trainer

import "strings"

// specMutationExpectedPaths lists from→to minor version pairs where Trainer API changes
// are known to mutate existing TrainJob/TrainingRuntime specs during upgrade.
var specMutationExpectedPaths = [][2]string{
	// kubeflow/trainer#3309: PodTemplateOverrides → RuntimePatches
	//	{"3.4", "3.5"},
}

// IsSpecMutationExpected returns true if the upgrade path from→to is known to mutate specs.
// Versions are compared by major.minor only.
func IsSpecMutationExpected(fromVersion, toVersion string) bool {
	fromMinor := majorMinor(fromVersion)
	toMinor := majorMinor(toVersion)
	if fromMinor == "" || toMinor == "" {
		return false
	}
	for _, pair := range specMutationExpectedPaths {
		if pair[0] == fromMinor && pair[1] == toMinor {
			return true
		}
	}
	return false
}

func majorMinor(version string) string {
	parts := strings.SplitN(version, ".", 3)
	if len(parts) < 2 {
		return ""
	}
	return parts[0] + "." + parts[1]
}

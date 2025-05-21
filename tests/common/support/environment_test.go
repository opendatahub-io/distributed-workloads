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
	"os"
	"testing"

	"github.com/onsi/gomega"
)

func TestGetRayVersion(t *testing.T) {

	g := gomega.NewGomegaWithT(t)
	// Set the environment variable.
	os.Setenv(TestRayVersion, "1.4.5")

	// Get the version.
	version := GetRayVersion()

	// Assert that the version is correct.

	g.Expect(version).To(gomega.Equal("1.4.5"), "Expected version 1.4.5, but got %s", version)

}

func TestGetRayImage(t *testing.T) {

	g := gomega.NewGomegaWithT(t)
	// Set the environment variable.
	os.Setenv(TestRayImage, "ray/ray:latest")

	// Get the image.
	image := GetRayImage()

	// Assert that the image is correct.

	g.Expect(image).To(gomega.Equal("ray/ray:latest"), "Expected image ray/ray:latest, but got %s", image)

}

func TestGetTrainingImage(t *testing.T) {

	g := gomega.NewGomegaWithT(t)
	// Set the environment variable.
	os.Setenv(TestTrainingCudaPyTorch251Image, "training/training:latest")

	// Get the image.
	image := GetTrainingCudaPyTorch251Image()
	// Assert that the image is correct.

	g.Expect(image).To(gomega.Equal("training/training:latest"), "Expected image training/training:latest, but got %s", image)

}

func TestGetClusterType(t *testing.T) {

	g := gomega.NewGomegaWithT(t)

	tests := []struct {
		name        string
		envVarValue string
		expected    ClusterType
	}{
		{
			name:        "OSD cluster",
			envVarValue: "OSD",
			expected:    OsdCluster,
		},
		{
			name:        "OCP cluster",
			envVarValue: "OCP",
			expected:    OcpCluster,
		},
		{
			name:        "Hypershift cluster",
			envVarValue: "HYPERSHIFT",
			expected:    HypershiftCluster,
		},
		{
			name:        "KIND cluster",
			envVarValue: "KIND",
			expected:    KindCluster,
		},
	}
	ttt := With(t)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			os.Setenv(ClusterTypeEnvVar, tt.envVarValue)
			actual := GetClusterType(ttt)

			g.Expect(actual).To(
				gomega.Equal(tt.expected),
				"Expected GetClusterType() to return %v, but got %v", tt.expected, actual,
			)

		})
	}
}

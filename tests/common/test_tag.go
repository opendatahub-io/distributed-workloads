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

package common

import (
	"fmt"
	"testing"

	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func Tags(t *testing.T, tags ...func(test Test) (runTest bool, skipReason string)) {
	test := With(t)
	for _, tag := range tags {
		runTest, skipReason := tag(test)
		if !runTest {
			test.T().Skip(skipReason)
		}
	}
}

// Test tag list

var Smoke = func(test Test) (runTest bool, skipReason string) {
	return testTier(test, tierSmoke)
}

var Sanity = func(test Test) (runTest bool, skipReason string) {
	return testTier(test, tierSanity)
}

var Tier1 = func(test Test) (runTest bool, skipReason string) {
	return testTier(test, tier1)
}

var Tier2 = func(test Test) (runTest bool, skipReason string) {
	return testTier(test, tier2)
}

var Tier3 = func(test Test) (runTest bool, skipReason string) {
	return testTier(test, tier3)
}

var PreUpgrade = func(test Test) (runTest bool, skipReason string) {
	return mandatoryTestTier(test, preUpgrade)
}

var PostUpgrade = func(test Test) (runTest bool, skipReason string) {
	return mandatoryTestTier(test, postUpgrade)
}

var KftoCuda = func(test Test) (runTest bool, skipReason string) {
	return testTier(test, kftoCuda)
}

var KftoRocm = func(test Test) (runTest bool, skipReason string) {
	return testTier(test, kftoRocm)
}

func Gpu(accelerator Accelerator) func(test Test) (runTest bool, skipReason string) {
	return func(test Test) (runTest bool, skipReason string) {
		return isGpuCountAvailableForNodes(test, 1, accelerator.ResourceLabel, 1)
	}
}

func MultiGpu(accelerator Accelerator, numberOfGpus int) func(test Test) (runTest bool, skipReason string) {
	return func(test Test) (runTest bool, skipReason string) {
		return isGpuCountAvailableForNodes(test, 1, accelerator.ResourceLabel, numberOfGpus)
	}
}

func MultiNode(numberOfNodes int) func(test Test) (runTest bool, skipReason string) {
	return func(test Test) (runTest bool, skipReason string) {
		nodes, err := test.Client().Core().CoreV1().Nodes().List(test.Ctx(), v1.ListOptions{LabelSelector: "!node-role.kubernetes.io/infra,node-role.kubernetes.io/worker"})
		test.Expect(err).NotTo(HaveOccurred())

		if len(nodes.Items) < numberOfNodes {
			return false, fmt.Sprintf("Detected number of nodes is %d, which is lower than expected %d.", len(nodes.Items), numberOfNodes)
		}
		return true, ""
	}
}

func MultiNodeGpu(numberOfNodes int, accelerator Accelerator) func(test Test) (runTest bool, skipReason string) {
	return func(test Test) (runTest bool, skipReason string) {
		return isGpuCountAvailableForNodes(test, numberOfNodes, accelerator.ResourceLabel, 1)
	}
}

func MultiNodeMultiGpu(numberOfNodes int, accelerator Accelerator, numberOfGpus int) func(test Test) (runTest bool, skipReason string) {
	return func(test Test) (runTest bool, skipReason string) {
		return isGpuCountAvailableForNodes(test, numberOfNodes, accelerator.ResourceLabel, numberOfGpus)
	}
}

// util functions

func testTier(test Test, expectedTestTier string) (runTest bool, skipReason string) {
	actualTestTier, found := GetTestTier(test)
	if !found || actualTestTier == expectedTestTier {
		return true, ""
	}
	return false, fmt.Sprintf("Test tier '%s' doesn't match expected tier '%s'", actualTestTier, expectedTestTier)
}

func mandatoryTestTier(test Test, expectedTestTier string) (runTest bool, skipReason string) {
	actualTestTier, found := GetTestTier(test)
	if found && actualTestTier == expectedTestTier {
		return true, ""
	}
	return false, fmt.Sprintf("Test tier '%s' doesn't match expected tier '%s'", actualTestTier, expectedTestTier)
}

func isGpuCountAvailableForNodes(test Test, expectedNodes int, gpuResourceName string, expectedGpus int) (runTest bool, skipReason string) {
	nodes, err := test.Client().Core().CoreV1().Nodes().List(test.Ctx(), v1.ListOptions{LabelSelector: "!node-role.kubernetes.io/infra,node-role.kubernetes.io/worker"})
	test.Expect(err).NotTo(HaveOccurred())

	var gpuNodes []corev1.Node
	for _, node := range nodes.Items {
		if node.Status.Allocatable.Name(corev1.ResourceName(gpuResourceName), resource.DecimalSI).Value() != 0 {
			gpuNodes = append(gpuNodes, node)
		}
	}

	if len(gpuNodes) < expectedNodes {
		return false, fmt.Sprintf("Detected number of nodes with resource '%s' is %d, which is lower than expected %d.", gpuResourceName, len(gpuNodes), expectedNodes)
	}

	for _, gpuNode := range gpuNodes {
		gpuCount := int(gpuNode.Status.Allocatable.Name(corev1.ResourceName(gpuResourceName), resource.DecimalSI).Value())

		if gpuCount < expectedGpus {
			return false, fmt.Sprintf("Detected number of GPUs for nodes with resource '%s' is %d, which is lower than expected %d.", gpuResourceName, gpuCount, expectedGpus)
		}
	}
	return true, ""
}

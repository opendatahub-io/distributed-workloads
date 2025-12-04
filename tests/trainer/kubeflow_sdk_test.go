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

package trainer

import (
	"testing"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	support "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	sdktests "github.com/opendatahub-io/distributed-workloads/tests/trainer/sdk_tests"
)

func TestKubeflowSdkSanity(t *testing.T) {
	Tags(t, Sanity)
	sdktests.RunFashionMnistCpuDistributedTraining(t)
}

// TestOsftTrainingHubMultiNodeMultiGPU tests OSFT training using TrainingHubTrainer
func TestOsftTrainingHubMultiNodeMultiGPU(t *testing.T) {
	Tags(t, KftoCuda, MultiNodeMultiGpu(2, support.NVIDIA, 1)) // TODO: may need to be updated once https://issues.redhat.com/browse/RHOAIENG-30719 and https://issues.redhat.com/browse/RHOAIENG-24552 are resolved
	sdktests.RunOsftTrainingHubMultiGpuDistributedTraining(t)
}

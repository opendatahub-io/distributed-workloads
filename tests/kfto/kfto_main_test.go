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

package kfto

import (
	"fmt"
	"os"
	"testing"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

var initialKueueState string

func TestMain(m *testing.M) {
	initialKueueState = CaptureComponentState(DefaultDSCName, "kueue")
	fmt.Printf("Initial Kueue managementState: %s\n", initialKueueState)

	code := m.Run()

	if initialKueueState != "Unmanaged" {
		if err := TearDownComponent(DefaultDSCName, "kueue"); err != nil {
			fmt.Printf("TearDown: Failed to set Kueue to Removed: %v\n", err)
		}
	} else {
		fmt.Println("TearDown: Skipping Kueue teardown as initial managementState was Unmanaged")
	}

	os.Exit(code)
}

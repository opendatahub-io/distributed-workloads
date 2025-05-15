/*
Copyright 2023.

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
	"os"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	ApplyOptions = metav1.ApplyOptions{FieldManager: "dw-test", Force: true}

	TestTimeoutShort           = 1 * time.Minute
	TestTimeoutMedium          = 2 * time.Minute
	TestTimeoutLong            = 5 * time.Minute
	TestTimeoutDouble          = 20 * time.Minute
	TestTimeoutGpuProvisioning = 30 * time.Minute
)

func init() {
	if value, ok := os.LookupEnv("TEST_TIMEOUT_SHORT"); ok {
		if duration, err := time.ParseDuration(value); err == nil {
			TestTimeoutShort = duration
		} else {
			fmt.Printf("Error parsing TEST_TIMEOUT_SHORT. Using default value: %s", TestTimeoutShort)
		}
	}
	if value, ok := os.LookupEnv("TEST_TIMEOUT_MEDIUM"); ok {
		if duration, err := time.ParseDuration(value); err == nil {
			TestTimeoutMedium = duration
		} else {
			fmt.Printf("Error parsing TEST_TIMEOUT_MEDIUM. Using default value: %s", TestTimeoutMedium)
		}
	}
	if value, ok := os.LookupEnv("TEST_TIMEOUT_LONG"); ok {
		if duration, err := time.ParseDuration(value); err == nil {
			TestTimeoutLong = duration
		} else {
			fmt.Printf("Error parsing TEST_TIMEOUT_LONG. Using default value: %s", TestTimeoutLong)
		}
	}
	if value, ok := os.LookupEnv("TEST_TIMEOUT_GPU_PROVISIONING"); ok {
		if duration, err := time.ParseDuration(value); err == nil {
			TestTimeoutGpuProvisioning = duration
		} else {
			fmt.Printf("Error parsing TEST_TIMEOUT_GPU_PROVISIONING. Using default value: %s", TestTimeoutGpuProvisioning)
		}
	}

	// Gomega settings
	gomega.SetDefaultEventuallyTimeout(TestTimeoutShort)
	gomega.SetDefaultEventuallyPollingInterval(1 * time.Second)
	gomega.SetDefaultConsistentlyDuration(30 * time.Second)
	gomega.SetDefaultConsistentlyPollingInterval(1 * time.Second)
	// Disable object truncation on test results
	format.MaxLength = 0
}

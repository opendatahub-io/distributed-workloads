/*
Copyright 2024

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
	"os"
)

const (
	// The environment variable for FMS HF Tuning image to be tested
	fmsHfTuningImageEnvVar = "FMS_HF_TUNING_IMAGE"
	// The environment variable referring to image containing bloom-560m model
	bloomModelImageEnvVar = "BLOOM_MODEL_IMAGE"
)

func GetFmsHfTuningImage() string {
	return lookupEnvOrDefault(fmsHfTuningImageEnvVar, "quay.io/modh/fms-hf-tuning:d0bd35b0297c28b87ee6caa32d5966d77587591f")
}

func GetBloomModelImage() string {
	return lookupEnvOrDefault(bloomModelImageEnvVar, "quay.io/ksuta/bloom-560m:0.0.1")
}

func lookupEnvOrDefault(key, value string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return value
}

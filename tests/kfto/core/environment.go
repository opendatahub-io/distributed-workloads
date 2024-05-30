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

package core

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
	return lookupEnvOrDefault(fmsHfTuningImageEnvVar, "quay.io/modh/fms-hf-tuning:bd8bf628cd739c7a201a976bc3c1096785353f1a")
}

func GetBloomModelImage() string {
	return lookupEnvOrDefault(bloomModelImageEnvVar, "quay.io/ksuta/bloom-560m@sha256:f6db02bb7b5d09a8d698c04994d747bfb9e581bbb4c07d00290244d207623733")
}

func lookupEnvOrDefault(key, value string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return value
}

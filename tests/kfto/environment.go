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
	// The environment variable referring to image containing bloom-560m model
	bloomModelImageEnvVar = "BLOOM_MODEL_IMAGE"
	// The environment variable referring to image containing Stanford Alpaca dataset
	alpacaDatasetImageEnvVar = "ALPACA_DATASET_IMAGE"
)

func GetBloomModelImage() string {
	return lookupEnvOrDefault(bloomModelImageEnvVar, "quay.io/ksuta/bloom-560m@sha256:f6db02bb7b5d09a8d698c04994d747bfb9e581bbb4c07d00290244d207623733")
}

func GetAlpacaDatasetImage() string {
	return lookupEnvOrDefault(alpacaDatasetImageEnvVar, "quay.io/ksuta/alpaca-dataset@sha256:2e90f631180c7b2c916f9569b914b336b612e8ae86efad82546adc5c9fcbbb8d")
}

func lookupEnvOrDefault(key, value string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return value
}

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

package common

import (
	"flag"
	"os"
	"slices"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	// The environment variable for namespace where ODH is installed to.
	odhNamespaceEnvVar = "ODH_NAMESPACE"
	// Name of the authenticated Notebook user
	notebookUserName = "NOTEBOOK_USER_NAME"
	// Token of the authenticated Notebook user
	notebookUserToken = "NOTEBOOK_USER_TOKEN"
	// Password of the authenticated Notebook user
	notebookUserPassword = "NOTEBOOK_USER_PASSWORD"
	// Image of the Notebook
	notebookImage = "NOTEBOOK_IMAGE"
	// Test tier to be invoked
	testTierEnvVar = "TEST_TIER"
	// The environment variable for HuggingFace token to download models which require authentication
	huggingfaceTokenEnvVar = "HF_TOKEN"
)

const (
	tierSmoke   = "Smoke"
	tierSanity  = "Sanity"
	tier1       = "Tier1"
	tier2       = "Tier2"
	tier3       = "Tier3"
	preUpgrade  = "Pre-Upgrade"
	postUpgrade = "Post-Upgrade"
	kftoCuda    = "KFTO-CUDA"
	kftoRocm    = "KFTO-ROCm"
)

var testTiers = []string{tierSmoke, tierSanity, tier1, tier2, tier3, preUpgrade, postUpgrade, kftoCuda, kftoRocm}

var testTierParam string

func GetOpenDataHubNamespace(t Test) string {
	ns, ok := os.LookupEnv(odhNamespaceEnvVar)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify namespace where ODH is installed to.", odhNamespaceEnvVar)
	}
	return ns
}

func GetNotebookUserName(t Test) string {
	name, ok := os.LookupEnv(notebookUserName)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify name of the authenticated Notebook user.", notebookUserName)
	}
	return name
}

func GetNotebookUserToken(t Test) string {
	token, ok := os.LookupEnv(notebookUserToken)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify token of the authenticated Notebook user.", notebookUserToken)
	}
	return token
}

func GetNotebookUserPassword(t Test) string {
	password, ok := os.LookupEnv(notebookUserPassword)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify token of the authenticated Notebook password.", notebookUserPassword)
	}
	return password
}

func GetNotebookImage(t Test) string {
	notebook_image, ok := os.LookupEnv(notebookImage)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify image of the Notebook.", notebookImage)
	}
	return notebook_image
}

func GetTestTier(t Test) (string, bool) {
	tt := lookupEnvOrDefault(testTierEnvVar, testTierParam)
	if tt != "" {
		if slices.Contains(testTiers, tt) {
			return tt, true
		}
		t.T().Fatalf("Environment variable %s is defined and contains invalid value: '%s'. Valid values are: %v", testTierEnvVar, tt, testTiers)
	}
	return "", false
}

func GetHuggingFaceToken(t Test) string {
	t.T().Helper()
	token, ok := os.LookupEnv(huggingfaceTokenEnvVar)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify HuggingFace token to download models.", huggingfaceTokenEnvVar)
	}
	return token
}

func init() {
	flag.StringVar(&testTierParam, "testTier", "", "Test tier")
}

func lookupEnvOrDefault(key, value string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return value
}

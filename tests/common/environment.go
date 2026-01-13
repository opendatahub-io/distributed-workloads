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
	"os/exec"
	"slices"
	"strings"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	// The environment variable referring to image simulating sleep condition in container
	sleepImageEnvVar = "SLEEP_IMAGE"
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
	tierSmoke    = "Smoke"
	tierSanity   = "Sanity"
	tier1        = "Tier1"
	tier2        = "Tier2"
	tier3        = "Tier3"
	preUpgrade   = "Pre-Upgrade"
	postUpgrade  = "Post-Upgrade"
	kftoCuda     = "KFTO-CUDA"
	kftoRocm     = "KFTO-ROCm"
	examplesCuda = "Examples-CUDA"
	examplesRocm = "Examples-ROCm"
)

var testTiers = []string{tierSmoke, tierSanity, tier1, tier2, tier3, preUpgrade, postUpgrade, kftoCuda, kftoRocm, examplesCuda, examplesRocm}

var testTierParam string

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

// GenerateNotebookUserToken generates an OpenShift token using oc login with username and password
func GenerateNotebookUserToken(t Test) string {
	userName := GetNotebookUserName(t)
	password := GetNotebookUserPassword(t)

	// Use own kubeconfig file to retrieve user token to keep it separated from main test credentials
	tempFile, err := os.CreateTemp("", "custom-kubeconfig-")
	if err != nil {
		t.T().Fatalf("Failed to create temp kubeconfig file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// Login by oc CLI using username and password
	cmd := exec.Command("oc", "login", "-u", userName, "-p", password, GetOpenShiftApiUrl(t), "--insecure-skip-tls-verify=true", "--kubeconfig="+tempFile.Name())
	out, err := cmd.Output()
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			t.T().Logf("Error running 'oc login' command: %v\n", exitError)
			t.T().Logf("Output: %s\n", out)
			t.T().Logf("Error output: %s\n", exitError.Stderr)
		} else {
			t.T().Logf("Error running 'oc login' command: %v\n", err)
		}
		t.T().FailNow()
	}

	// Use oc CLI to retrieve user token from kubeconfig
	cmd = exec.Command("oc", "whoami", "--show-token", "--kubeconfig="+tempFile.Name())
	out, err = cmd.Output()
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			t.T().Logf("Error running 'oc whoami' command: %v\n", exitError)
			t.T().Logf("Output: %s\n", out)
			t.T().Logf("Error output: %s\n", exitError.Stderr)
		} else {
			t.T().Logf("Error running 'oc whoami' command: %v\n", err)
		}
		t.T().FailNow()
	}

	return strings.TrimSpace(string(out))
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

func GetSleepImage() string {
	return lookupEnvOrDefault(sleepImageEnvVar, "gcr.io/k8s-staging-perf-tests/sleep@sha256:8d91ddf9f145b66475efda1a1b52269be542292891b5de2a7fad944052bab6ea")
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

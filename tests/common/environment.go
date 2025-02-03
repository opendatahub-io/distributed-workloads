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
	"os"

	. "github.com/project-codeflare/codeflare-common/support"
)

const (
	// The environment variable for namespace where ODH is installed to.
	odhNamespaceEnvVar = "ODH_NAMESPACE"
	// Name of the authenticated Notebook user
	notebookUserName = "NOTEBOOK_USER_NAME"
	// Token of the authenticated Notebook user
	notebookUserToken = "NOTEBOOK_USER_TOKEN"
	// Image of the Notebook
	notebookImage = "NOTEBOOK_IMAGE"
)

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

func GetNotebookImage(t Test) string {
	notebook_image, ok := os.LookupEnv(notebookImage)
	if !ok {
		t.T().Fatalf("Expected environment variable %s not found, please use this environment variable to specify image of the Notebook.", notebookImage)
	}
	return notebook_image
}

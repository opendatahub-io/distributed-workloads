package sdk_tests

import (
	"os"
	"strings"
)

// buildKubeflowInstallExports builds a shell-export prefix that is injected into
// notebook pod commands before running install_kubeflow.py.
//
// Why this exists:
//   - SDK tests start from host-side go test processes, but kubeflow installation
//     happens inside notebook containers.
//   - Host environment variables are not automatically available in the notebook
//     process, so we explicitly export the selected install vars.
//
// Selection precedence:
// 1. KUBEFLOW_GIT_URL: install from git (sets KUBEFLOW_INSTALL_FROM_GIT=true)
// 2. KUBEFLOW_REQUIRED_VERSION: install that version from index/default index
// 3. Neither set: sets KUBEFLOW_SKIP_INSTALL=true to use the SDK baked into the notebook image
//
// Index behavior:
//   - If KUBEFLOW_PYPI_INDEX_URL is set, we export it
//     for install_kubeflow.py to use as the package index.
func buildKubeflowInstallExports() string {
	gitURL := strings.TrimSpace(os.Getenv("KUBEFLOW_GIT_URL"))
	version := strings.TrimSpace(os.Getenv("KUBEFLOW_REQUIRED_VERSION"))
	indexURL := strings.TrimSpace(os.Getenv("KUBEFLOW_PYPI_INDEX_URL"))

	var exports strings.Builder
	if gitURL != "" {
		exports.WriteString("export KUBEFLOW_INSTALL_FROM_GIT='true'; ")
		exports.WriteString("export KUBEFLOW_GIT_URL=" + shellQuote(gitURL) + "; ")
	} else if version != "" {
		exports.WriteString("export KUBEFLOW_REQUIRED_VERSION=" + shellQuote(version) + "; ")
	} else {
		exports.WriteString("export KUBEFLOW_SKIP_INSTALL='true'; ")
	}

	if indexURL != "" {
		exports.WriteString("export KUBEFLOW_PYPI_INDEX_URL=" + shellQuote(indexURL) + "; ")
	}
	return exports.String()
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

package callbacktests

import (
	"os"
	"strings"
)

const (
	installScriptPath     = "resources/disconnected_env/install_kubeflow.py"
	installKubeflowScript = "install_kubeflow.py"
)

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

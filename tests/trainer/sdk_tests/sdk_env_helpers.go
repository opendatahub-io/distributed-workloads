package sdk_tests

import (
	"os"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// BuildKubeflowInstallEnv builds the environment used by install_kubeflow.py.
//
// Why this exists:
//   - SDK tests start from host-side go test processes, but kubeflow installation
//     happens inside notebook containers.
//   - Host environment variables are not automatically available in the notebook
//     process, so we explicitly pass the selected install vars to the container.
//
// Selection precedence:
// 1. KUBEFLOW_GIT_URL: install from git (sets KUBEFLOW_INSTALL_FROM_GIT=true)
// 2. KUBEFLOW_REQUIRED_VERSION: install that version from index/default index
// 3. Neither set: sets KUBEFLOW_SKIP_INSTALL=true to use the SDK baked into the notebook image
//
// Index behavior:
//   - If KUBEFLOW_PYPI_INDEX_URL is set, we pass it
//     for install_kubeflow.py to use as the package index.
func BuildKubeflowInstallEnv() []corev1.EnvVar {
	gitURL := strings.TrimSpace(os.Getenv("KUBEFLOW_GIT_URL"))
	version := strings.TrimSpace(os.Getenv("KUBEFLOW_REQUIRED_VERSION"))
	indexURL := strings.TrimSpace(os.Getenv("KUBEFLOW_PYPI_INDEX_URL"))

	var env []corev1.EnvVar
	if gitURL != "" {
		env = append(env,
			corev1.EnvVar{Name: "KUBEFLOW_INSTALL_FROM_GIT", Value: "true"},
			corev1.EnvVar{Name: "KUBEFLOW_GIT_URL", Value: gitURL},
		)
	} else if version != "" {
		env = append(env, corev1.EnvVar{Name: "KUBEFLOW_REQUIRED_VERSION", Value: version})
	} else {
		env = append(env, corev1.EnvVar{Name: "KUBEFLOW_SKIP_INSTALL", Value: "true"})
	}

	if indexURL != "" {
		env = append(env, corev1.EnvVar{Name: "KUBEFLOW_PYPI_INDEX_URL", Value: indexURL})
	}
	return env
}

func ShellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func buildKubeflowInstallEnv() []corev1.EnvVar { return BuildKubeflowInstallEnv() }

func shellQuote(value string) string { return ShellQuote(value) }

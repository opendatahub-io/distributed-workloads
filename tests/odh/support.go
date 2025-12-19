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

package odh

import (
	"embed"
	"net/url"
	"os"
	"time"

	gonanoid "github.com/matoous/go-nanoid/v2"
	gomega "github.com/onsi/gomega"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	routev1 "github.com/openshift/api/route/v1"

	"github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

//go:embed resources/*
var files embed.FS

func readFile(t support.Test, fileName string) []byte {
	t.T().Helper()
	file, err := files.ReadFile(fileName)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return file
}

func ReadFileExt(t support.Test, fileName string) []byte {
	t.T().Helper()
	file, err := os.ReadFile(fileName)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return file
}

func getNotebookCommand(rayImage string) []string {
	return []string{
		"/bin/sh",
		"-c",
		"pip install papermill && papermill /opt/app-root/notebooks/{{.NotebookConfigMapFileName}}" +
			" /opt/app-root/src/mcad-out.ipynb -p namespace {{.Namespace}} -p ray_image " + rayImage + " " +
			" -p openshift_api_url {{.OpenShiftApiUrl}} -p kubernetes_user_bearer_token {{.KubernetesUserBearerToken}}" +
			" -p num_gpus {{ .NumGpus }} --log-output && sleep infinity",
	}
}

// GetDashboardUrl attempts to create a service and route for external dashboard access.
// Returns the dashboard URL if successful, or empty string if it fails.
// This function is best-effort and will NOT fail the test on errors.
func GetDashboardUrl(test support.Test, namespace *corev1.Namespace, rayCluster *rayv1.RayCluster) string {
	// The kuberay operator creates a headless service which doesn't work well with routes.
	// Create a non-headless service and an edge-terminated route for direct dashboard access.
	// This bypasses OAuth authentication for testing purposes - the Ray dashboard on port 8265
	// doesn't require authentication when accessed directly.

	dashboardServiceName := rayCluster.Name + "-dashboard-svc"
	dashboardRouteName := rayCluster.Name + "-dashboard-route"

	// Create a non-headless ClusterIP service for the Ray dashboard (port 8265)
	dashboardService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dashboardServiceName,
			Namespace: namespace.Name,
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeClusterIP,
			Selector: map[string]string{
				"ray.io/cluster":   rayCluster.Name,
				"ray.io/node-type": "head",
			},
			Ports: []corev1.ServicePort{
				{
					Name:       "dashboard",
					Port:       8265,
					TargetPort: intstr.FromInt(8265),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	_, err := test.Client().Core().CoreV1().Services(namespace.Name).Create(test.Ctx(), dashboardService, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		test.T().Logf("Warning: Failed to create dashboard service: %v", err)
		return ""
	}
	test.T().Logf("Created Dashboard Service %s/%s successfully", namespace.Name, dashboardServiceName)

	// Create a plain HTTP route (no TLS) to the dashboard service
	// Using plain HTTP to avoid TLS termination issues that were causing timeouts
	dashboardRoute := &routev1.Route{
		TypeMeta: metav1.TypeMeta{
			APIVersion: routev1.SchemeGroupVersion.String(),
			Kind:       "Route",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      dashboardRouteName,
			Namespace: namespace.Name,
		},
		Spec: routev1.RouteSpec{
			To: routev1.RouteTargetReference{
				Kind: "Service",
				Name: dashboardServiceName,
			},
			Port: &routev1.RoutePort{
				TargetPort: intstr.FromString("dashboard"),
			},
			// No TLS - use plain HTTP to avoid TLS termination issues
		},
	}

	_, err = test.Client().Route().RouteV1().Routes(namespace.Name).Create(test.Ctx(), dashboardRoute, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		test.T().Logf("Warning: Failed to create dashboard route: %v", err)
		return ""
	}
	test.T().Logf("Created Dashboard Route %s/%s successfully", namespace.Name, dashboardRouteName)

	// Wait for the route to be admitted (with manual polling, no assertions)
	test.T().Logf("Waiting for Dashboard route %s/%s to be admitted...", namespace.Name, dashboardRouteName)
	routeAdmitted := false
	for i := 0; i < 30; i++ { // 30 seconds max
		time.Sleep(1 * time.Second)
		route, err := test.Client().Route().RouteV1().Routes(namespace.Name).Get(test.Ctx(), dashboardRouteName, metav1.GetOptions{})
		if err != nil {
			continue
		}
		for _, cond := range route.Status.Ingress {
			for _, c := range cond.Conditions {
				if c.Type == routev1.RouteAdmitted && c.Status == corev1.ConditionTrue {
					routeAdmitted = true
					break
				}
			}
		}
		if routeAdmitted {
			break
		}
	}
	if !routeAdmitted {
		test.T().Logf("Warning: Dashboard route was not admitted within timeout")
		return ""
	}

	route, err := test.Client().Route().RouteV1().Routes(namespace.Name).Get(test.Ctx(), dashboardRouteName, metav1.GetOptions{})
	if err != nil || len(route.Status.Ingress) == 0 {
		test.T().Logf("Warning: Could not get route ingress info")
		return ""
	}
	hostname := route.Status.Ingress[0].Host
	dashboardUrl, _ := url.Parse("http://" + hostname)
	test.T().Logf("Ray-dashboard route: %s\n", dashboardUrl.String())

	// Log head pod info for debugging
	pods, _ := test.Client().Core().CoreV1().Pods(namespace.Name).List(test.Ctx(), metav1.ListOptions{
		LabelSelector: "ray.io/node-type=head",
	})
	if pods != nil && len(pods.Items) > 0 {
		for _, pod := range pods.Items {
			test.T().Logf("Found head pod: %s, labels: %v, phase: %s", pod.Name, pod.Labels, pod.Status.Phase)
		}
	}

	// Wait for service endpoints (manual polling, no assertions)
	test.T().Logf("Waiting for Dashboard service %s/%s to have endpoints...", namespace.Name, dashboardServiceName)
	endpointsReady := false
	for i := 0; i < 60; i++ { // 60 seconds max
		time.Sleep(1 * time.Second)
		endpoints, err := test.Client().Core().CoreV1().Endpoints(namespace.Name).Get(test.Ctx(), dashboardServiceName, metav1.GetOptions{})
		if err != nil {
			continue
		}
		for _, subset := range endpoints.Subsets {
			if len(subset.Addresses) > 0 {
				test.T().Logf("Service has %d ready endpoints", len(subset.Addresses))
				endpointsReady = true
				break
			}
		}
		if endpointsReady {
			break
		}
	}
	if !endpointsReady {
		test.T().Logf("Warning: Dashboard service endpoints not ready within timeout")
		return ""
	}

	// Return the URL - we'll check if dashboard is actually responding in TryMonitorRayJob
	// This avoids blocking here on 503 errors since external access is unreliable
	return dashboardUrl.String()
}

func GetTestJobId(test support.Test, rayClient support.RayClusterClient) string {
	allJobsData, err := rayClient.ListJobs()
	test.Expect(err).ToNot(gomega.HaveOccurred())
	test.Expect(allJobsData).NotTo(gomega.BeEmpty())

	jobID := allJobsData[0].SubmissionID
	test.T().Logf("Ray job has been successfully submitted to the raycluster with Submission-ID : %s\n", jobID)

	return jobID
}

// EnsureNotebookServiceAccount ensures the Notebook ServiceAccount exists in the target namespace.
// This avoids webhook/controller failures when creating the Notebook CR.
func ensureNotebookServiceAccount(test support.Test, namespace string) {
	test.T().Helper()
	saName := "jupyter-nb-kube-3aadmin"
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: namespace}}
	_, err := test.Client().Core().CoreV1().ServiceAccounts(namespace).Create(test.Ctx(), sa, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		test.T().Fatalf("Failed to create ServiceAccount %s/%s: %v", namespace, saName, err)
	}
}

// Adds a unique suffix to the provided string
func uniqueSuffix(prefix string) string {
	suffix := gonanoid.MustGenerate("1234567890abcdef", 4)
	return prefix + "-" + suffix
}

// TryMonitorRayJob attempts to monitor the Ray job via the external dashboard.
// This is best-effort - if it succeeds, it returns the job status and logs it.
// If it fails, it logs a warning and returns "", false so the test can continue
// with the fallback (waiting for RayCluster deletion).
// Note: External dashboard access is often unreliable (503 errors, timeouts).
// The notebook uses internal service access which is more reliable.
func TryMonitorRayJob(test support.Test, namespace *corev1.Namespace, rayClusterName string) (string, bool) {
	test.T().Logf("Attempting to monitor Ray job via external dashboard (best-effort)...")

	// Get the RayCluster
	rayCluster, err := test.Client().Ray().RayV1().RayClusters(namespace.Name).Get(test.Ctx(), rayClusterName, metav1.GetOptions{})
	if err != nil {
		test.T().Logf("Warning: Could not get RayCluster %s: %v", rayClusterName, err)
		return "", false
	}

	// Try to get dashboard URL (this creates service and route)
	dashboardUrl := GetDashboardUrl(test, namespace, rayCluster)
	if dashboardUrl == "" {
		test.T().Logf("Warning: Could not get dashboard URL")
		return "", false
	}

	// Create Ray client
	rayClient := support.GetRayClusterClient(test, dashboardUrl, test.Config().BearerToken)

	// Wait for job to exist - this also serves as a dashboard connectivity check
	// Use a shorter timeout since we have a fallback
	var jobs []support.RayJobDetailsResponse
	jobFound := false
	consecutiveErrors := 0
	maxConsecutiveErrors := 30 // Give up after 30 consecutive errors (external access is unreliable)

	test.T().Logf("Waiting for Ray job to appear via external dashboard...")
	for i := 0; i < 120; i++ { // 2 minutes max wait for job to appear
		time.Sleep(1 * time.Second)
		jobs, err = rayClient.ListJobs()
		if err != nil {
			consecutiveErrors++
			if consecutiveErrors <= 5 || consecutiveErrors%10 == 0 {
				test.T().Logf("Error listing jobs (%d consecutive): %v", consecutiveErrors, err)
			}
			if consecutiveErrors >= maxConsecutiveErrors {
				test.T().Logf("Warning: Too many consecutive errors accessing external dashboard, giving up")
				return "", false
			}
			continue
		}
		consecutiveErrors = 0 // Reset on success
		if len(jobs) > 0 {
			jobFound = true
			break
		}
	}
	if !jobFound {
		test.T().Logf("Warning: Ray job not found via external dashboard after 2 minutes")
		return "", false
	}

	// Get job ID
	jobID := jobs[0].SubmissionID
	if jobID == "" {
		test.T().Logf("Warning: Could not get job ID")
		return "", false
	}
	test.T().Logf("Found Ray job with ID: %s", jobID)

	// Wait for the job to complete (SUCCEEDED or FAILED)
	var rayJobStatus string
	consecutiveErrors = 0
	test.T().Logf("Monitoring job status via external dashboard...")
	for i := 0; i < 600; i++ { // 10 minutes max
		time.Sleep(1 * time.Second)
		resp, err := rayClient.GetJobDetails(jobID)
		if err != nil {
			consecutiveErrors++
			if consecutiveErrors <= 5 || consecutiveErrors%10 == 0 {
				test.T().Logf("Error getting job details (%d consecutive): %v", consecutiveErrors, err)
			}
			if consecutiveErrors >= maxConsecutiveErrors {
				test.T().Logf("Warning: Too many consecutive errors, giving up on external monitoring")
				return "", false
			}
			continue
		}
		consecutiveErrors = 0 // Reset on success
		rayJobStatusVal := resp.Status
		if rayJobStatusVal == "SUCCEEDED" || rayJobStatusVal == "FAILED" {
			test.T().Logf("Job completed with status: %s", rayJobStatusVal)
			rayJobStatus = rayJobStatusVal
			break
		}
		if rayJobStatus != rayJobStatusVal {
			test.T().Logf("Job status: %s", rayJobStatusVal)
			rayJobStatus = rayJobStatusVal
		}
	}

	if rayJobStatus == "" {
		test.T().Logf("Warning: Job did not complete within timeout via external dashboard")
		return "", false
	}

	// Note: We skip WriteRayJobAPILogs here because:
	// 1. The RayCluster is often already deleted by the notebook by the time we try to get logs
	// 2. The function uses assertions that can fail the test even with recover()
	// 3. Pod logs are already collected by the test framework on failure

	return rayJobStatus, true
}

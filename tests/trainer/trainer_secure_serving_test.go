package trainer

import (
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	trainerControllerDeployment = "kubeflow-trainer-controller-manager"
	trainerControllerService    = "kubeflow-trainer-controller-manager"
	trainerMetricsPort          = int32(8443)
	maxMetricsResponseBytes     = 10 * 1024 * 1024
	tlsProfileTransitionTimeout = 3 * time.Minute
	tlsProfileLeaseDuration     = 15 * time.Minute
	metricsStartupTimeout       = 2 * time.Minute
)

var (
	openShiftAPIServerGVR = schema.GroupVersionResource{
		Group: "config.openshift.io", Version: "v1", Resource: "apiservers",
	}
)

func TestTrainerSecureServing(t *testing.T) {
	Tags(t, Tier2)
	test := With(t)
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())

	service, deployment := trainerMetricsResources(test, applicationsNamespace)
	serviceName := service.GetName()
	pod := trainerControllerPod(test, applicationsNamespace, deployment)
	test.T().Logf("Checking Trainer metrics service %s/%s", applicationsNamespace, serviceName)
	// client-go port-forwarding does not resolve Services, so use a Ready pod selected by the Service.
	checkTrainerMetricsEndpoint(test, applicationsNamespace, pod.Name)
	test.T().Logf("Validated Trainer HTTPS metrics authentication for service %s/%s via pod %s", applicationsNamespace, serviceName, pod.Name)
}

func TestTrainerPrometheusScrape(t *testing.T) {
	Tags(t, Tier2)
	test := With(t)
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())

	trainerMetricsResources(test, applicationsNamespace)
	prometheus := GetOpenShiftPrometheusApiClient(test)
	test.T().Logf("Waiting for Prometheus to discover Trainer ServiceMonitor target %s/%s", applicationsNamespace, trainerControllerService)

	var target prometheusapiv1.ActiveTarget
	test.Eventually(func(g Gomega, ctx context.Context) {
		result, targetErr := prometheus.Targets(ctx)
		g.Expect(targetErr).NotTo(HaveOccurred())
		found := false
		for _, candidate := range result.Active {
			if string(candidate.Labels["namespace"]) != applicationsNamespace ||
				string(candidate.Labels["service"]) != trainerControllerService {
				continue
			}
			target = candidate
			found = true
			break
		}
		g.Expect(found).To(BeTrue(), "Trainer ServiceMonitor target was not discovered by Prometheus")
		g.Expect(target.Health).To(Equal(prometheusapiv1.HealthGood))
		g.Expect(target.LastError).To(BeEmpty())
	}, 5*time.Minute, 10*time.Second).WithContext(test.Ctx()).Should(Succeed())
	test.T().Logf("Prometheus target is healthy: scrapeURL=%s", target.ScrapeURL)

	targetSelector := fmt.Sprintf(
		"job=%q,instance=%q,namespace=%q,service=%q",
		target.Labels["job"], target.Labels["instance"], target.Labels["namespace"], target.Labels["service"],
	)
	test.Eventually(func(g Gomega, ctx context.Context) {
		value, warnings, queryErr := prometheus.Query(ctx, fmt.Sprintf("up{%s}", targetSelector), time.Now())
		g.Expect(queryErr).NotTo(HaveOccurred())
		g.Expect(warnings).To(BeEmpty())
		up, ok := value.(model.Vector)
		g.Expect(ok).To(BeTrue())
		g.Expect(up).NotTo(BeEmpty())
		for _, sample := range up {
			g.Expect(float64(sample.Value)).To(Equal(float64(1)))
		}

		value, warnings, queryErr = prometheus.Query(ctx, fmt.Sprintf("certwatcher_read_certificate_total{%s}", targetSelector), time.Now())
		g.Expect(queryErr).NotTo(HaveOccurred())
		g.Expect(warnings).To(BeEmpty())
		metrics, ok := value.(model.Vector)
		g.Expect(ok).To(BeTrue())
		g.Expect(metrics).NotTo(BeEmpty(), "Trainer certificate watcher metric was not scraped")
	}, 5*time.Minute, 10*time.Second).WithContext(test.Ctx()).Should(Succeed())
	test.T().Logf("Prometheus returned up=1 and certwatcher metrics for Trainer target")
}

func TestTrainerTLSProfileWatcher(t *testing.T) {
	Tags(t, Tier2)
	test := With(t)
	apiServer, err := test.Client().Dynamic().Resource(openShiftAPIServerGVR).Get(
		test.Ctx(), "cluster", metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		t.Skip("OpenShift APIServer is unavailable; TLS profile watcher is not applicable")
	}
	test.Expect(err).NotTo(HaveOccurred())
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())
	releaseTLSProfileLock := acquireTLSProfileLock(test, applicationsNamespace)
	test.T().Cleanup(releaseTLSProfileLock)

	originalProfile, found, err := unstructured.NestedFieldCopy(apiServer.Object, "spec", "tlsSecurityProfile")
	test.Expect(err).NotTo(HaveOccurred())
	if !found {
		t.Skip("OpenShift APIServer has no tlsSecurityProfile to switch")
	}
	originalProfileMap, ok := originalProfile.(map[string]interface{})
	test.Expect(ok).To(BeTrue())
	originalType, found, err := unstructured.NestedString(originalProfileMap, "type")
	test.Expect(err).NotTo(HaveOccurred())
	if !found || originalType == "" {
		t.Skip("OpenShift APIServer tlsSecurityProfile has no type")
	}
	newProfile := "Intermediate"
	if originalType == newProfile {
		newProfile = "Old"
	}
	newProfileSpec := tlsSecurityProfile(newProfile)
	test.T().Logf("TLS profile watcher test will transition APIServer profile %s -> %s", originalType, newProfile)

	test.T().Cleanup(func() {
		test.T().Logf("Restoring APIServer TLS profile to %s", originalType)
		restoreCtx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		resource := test.Client().Dynamic().Resource(openShiftAPIServerGVR)
		var lastErr error
		restoreErr := wait.PollUntilContextTimeout(
			restoreCtx, 5*time.Second, 10*time.Minute, true,
			func(ctx context.Context) (bool, error) {
				current, getErr := resource.Get(ctx, "cluster", metav1.GetOptions{})
				if getErr != nil {
					lastErr = getErr
					return false, nil
				}
				if setErr := unstructured.SetNestedField(current.Object, originalProfileMap, "spec", "tlsSecurityProfile"); setErr != nil {
					return false, setErr
				}
				_, updateErr := resource.Update(ctx, current, metav1.UpdateOptions{})
				if updateErr != nil {
					lastErr = updateErr
					return false, nil
				}
				return true, nil
			},
		)
		if restoreErr != nil {
			if lastErr != nil {
				test.T().Errorf("failed to restore OpenShift APIServer TLS profile: %v", lastErr)
			} else {
				test.T().Errorf("failed to restore OpenShift APIServer TLS profile: %v", restoreErr)
			}
		} else {
			test.T().Logf("Restored APIServer TLS profile to %s", originalType)
			recoveryErr := wait.PollUntilContextTimeout(
				restoreCtx, 5*time.Second, 10*time.Minute, true,
				func(ctx context.Context) (bool, error) {
					current, getErr := resource.Get(ctx, "cluster", metav1.GetOptions{})
					if getErr != nil {
						return false, nil
					}
					currentType, _, typeErr := unstructured.NestedString(current.Object, "spec", "tlsSecurityProfile", "type")
					if typeErr != nil || currentType != originalType {
						return false, nil
					}
					deployment := getTrainerDeployment(test, applicationsNamespace)
					return trainerControllerReady(test, applicationsNamespace, deployment, ctx)
				},
			)
			if recoveryErr != nil {
				test.Expect(recoveryErr).NotTo(HaveOccurred(), "failed waiting for Trainer recovery after restoring TLS profile")
			} else {
				test.T().Logf("Trainer recovered after restoring APIServer TLS profile")
			}
		}
	})

	deployment := getTrainerDeployment(test, applicationsNamespace)
	beforePod := trainerControllerPod(test, applicationsNamespace, deployment)
	beforeRestartCount := trainerControllerRestartCount(beforePod)
	test.T().Logf("Updating APIServer TLS profile; Trainer pod=%s restartCount=%d", beforePod.Name, beforeRestartCount)
	resource := test.Client().Dynamic().Resource(openShiftAPIServerGVR)
	var lastErr error
	updateErr := wait.PollUntilContextTimeout(
		test.Ctx(), 5*time.Second, tlsProfileTransitionTimeout, true,
		func(ctx context.Context) (bool, error) {
			current, getErr := resource.Get(ctx, "cluster", metav1.GetOptions{})
			if getErr != nil {
				lastErr = getErr
				return false, nil
			}
			if setErr := unstructured.SetNestedField(current.Object, newProfileSpec, "spec", "tlsSecurityProfile"); setErr != nil {
				return false, setErr
			}
			_, updateErr := resource.Update(ctx, current, metav1.UpdateOptions{})
			if updateErr != nil {
				lastErr = updateErr
				return false, nil
			}
			return true, nil
		},
	)
	if updateErr != nil {
		if lastErr != nil {
			test.T().Fatalf("failed to update OpenShift APIServer TLS profile: %v", lastErr)
		}
		test.Expect(updateErr).NotTo(HaveOccurred())
	}
	profileUpdatedAt := time.Now()
	test.T().Logf("APIServer TLS profile updated to %s; waiting for Trainer restart", newProfile)

	waitForTrainerControllerRestart(test, applicationsNamespace, deployment, beforePod, beforeRestartCount, profileUpdatedAt)

	// The controller should remain healthy after the profile transition.
	test.Eventually(func(g Gomega, ctx context.Context) {
		updated := getTrainerDeployment(test, applicationsNamespace)
		g.Expect(deploymentAvailable(updated)).To(BeTrue())
	}, tlsProfileTransitionTimeout, 5*time.Second).WithContext(test.Ctx()).Should(Succeed())
	test.T().Logf("Trainer controller recovered after TLS profile transition")
}

func tlsSecurityProfile(profileType string) map[string]interface{} {
	profile := map[string]interface{}{"type": profileType}
	if profileType == "Old" {
		profile["old"] = map[string]interface{}{}
	} else {
		profile["intermediate"] = map[string]interface{}{}
	}
	return profile
}

func trainerMetricsResources(test Test, namespace string) (*unstructured.Unstructured, *unstructured.Unstructured) {
	deployment := getTrainerDeployment(test, namespace)
	service, err := test.Client().Dynamic().Resource(schema.GroupVersionResource{
		Version: "v1", Resource: "services",
	}).Namespace(namespace).Get(test.Ctx(), trainerControllerService, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	ports, found, err := unstructured.NestedSlice(service.Object, "spec", "ports")
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(found).To(BeTrue())
	for _, rawPort := range ports {
		port, ok := rawPort.(map[string]interface{})
		if !ok {
			continue
		}
		portNumber, ok := port["port"].(int64)
		if ok && int32(portNumber) == trainerMetricsPort {
			return service, deployment
		}
	}
	test.T().Fatalf("Trainer metrics Service %s/%s does not expose port %d", namespace, trainerControllerService, trainerMetricsPort)
	return nil, nil
}

func getTrainerDeployment(test Test, namespace string) *unstructured.Unstructured {
	deployment, err := test.Client().Dynamic().Resource(schema.GroupVersionResource{
		Group: "apps", Version: "v1", Resource: "deployments",
	}).Namespace(namespace).Get(test.Ctx(), trainerControllerDeployment, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	return deployment
}

func trainerControllerPod(test Test, namespace string, deployment *unstructured.Unstructured) *corev1.Pod {
	selector, _, err := unstructured.NestedStringMap(deployment.Object, "spec", "selector", "matchLabels")
	test.Expect(err).NotTo(HaveOccurred())
	var readyPod *corev1.Pod
	test.Eventually(func(g Gomega, ctx context.Context) {
		readyPod = nil
		pods, listErr := test.Client().Core().CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: labelsToSelector(selector),
		})
		g.Expect(listErr).NotTo(HaveOccurred())
		for i := range pods.Items {
			if pods.Items[i].Status.Phase == corev1.PodRunning && podReady(&pods.Items[i]) {
				readyPod = &pods.Items[i]
				return
			}
		}
		g.Expect(readyPod).NotTo(BeNil(), "Trainer controller pod is not ready")
	}, tlsProfileTransitionTimeout, 5*time.Second).WithContext(test.Ctx()).Should(Succeed())
	test.T().Logf("Trainer controller pod %s is Ready (restartCount=%d)", readyPod.Name, trainerControllerRestartCount(readyPod))
	return readyPod
}

func podReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func waitForTrainerControllerRestart(test Test, namespace string, deployment *unstructured.Unstructured, beforePod *corev1.Pod, beforeRestartCount int32, profileUpdatedAt time.Time) {
	selector, _, err := unstructured.NestedStringMap(deployment.Object, "spec", "selector", "matchLabels")
	test.Expect(err).NotTo(HaveOccurred())
	test.Eventually(func(g Gomega, ctx context.Context) {
		pods, listErr := test.Client().Core().CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: labelsToSelector(selector),
		})
		g.Expect(listErr).NotTo(HaveOccurred())
		for i := range pods.Items {
			pod := &pods.Items[i]
			if pod.UID != beforePod.UID && podReady(pod) && pod.CreationTimestamp.Time.After(profileUpdatedAt.Add(-30*time.Second)) {
				test.T().Logf("Trainer pod was replaced after TLS profile change: %s -> %s", beforePod.Name, pod.Name)
				return
			}
			if pod.UID == beforePod.UID && podReady(pod) && trainerControllerRestartCount(pod) > beforeRestartCount {
				test.T().Logf("Trainer container restarted in pod %s: restartCount %d -> %d", pod.Name, beforeRestartCount, trainerControllerRestartCount(pod))
				return
			}
		}
		g.Expect(false).To(BeTrue(), "Trainer controller did not restart after TLS profile change")
	}, tlsProfileTransitionTimeout, 5*time.Second).WithContext(test.Ctx()).Should(Succeed())
}

func trainerControllerRestartCount(pod *corev1.Pod) int32 {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == "manager" {
			return status.RestartCount
		}
	}
	return 0
}

func trainerControllerReady(test Test, namespace string, deployment *unstructured.Unstructured, ctx context.Context) (bool, error) {
	selector, _, err := unstructured.NestedStringMap(deployment.Object, "spec", "selector", "matchLabels")
	if err != nil {
		return false, err
	}
	pods, err := test.Client().Core().CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelsToSelector(selector),
	})
	if err != nil {
		return false, err
	}
	for i := range pods.Items {
		if pods.Items[i].Status.Phase == corev1.PodRunning && podReady(&pods.Items[i]) {
			return true, nil
		}
	}
	return false, nil
}

func acquireTLSProfileLock(test Test, namespace string) func() {
	hostname, err := os.Hostname()
	test.Expect(err).NotTo(HaveOccurred())
	identity := fmt.Sprintf("%s-%d", hostname, os.Getpid())
	leases := test.Client().Core().CoordinationV1().Leases(namespace)
	durationSeconds := int32(tlsProfileLeaseDuration / time.Second)

	acquireErr := wait.PollUntilContextTimeout(
		test.Ctx(), 5*time.Second, 10*time.Minute, true,
		func(ctx context.Context) (bool, error) {
			lease, getErr := leases.Get(ctx, "trainer-tls-profile-e2e", metav1.GetOptions{})
			if apierrors.IsNotFound(getErr) {
				now := metav1.NewMicroTime(time.Now())
				_, createErr := leases.Create(ctx, &coordinationv1.Lease{
					ObjectMeta: metav1.ObjectMeta{Name: "trainer-tls-profile-e2e"},
					Spec: coordinationv1.LeaseSpec{
						HolderIdentity:       &identity,
						LeaseDurationSeconds: &durationSeconds,
						AcquireTime:          &now,
						RenewTime:            &now,
					},
				}, metav1.CreateOptions{})
				if apierrors.IsAlreadyExists(createErr) {
					return false, nil
				}
				return createErr == nil, createErr
			}
			if getErr != nil {
				return false, getErr
			}
			if lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != identity &&
				(lease.Spec.RenewTime == nil || lease.Spec.RenewTime.Time.Add(tlsProfileLeaseDuration).After(time.Now())) {
				return false, nil
			}
			now := metav1.NewMicroTime(time.Now())
			lease.Spec.HolderIdentity = &identity
			lease.Spec.LeaseDurationSeconds = &durationSeconds
			lease.Spec.AcquireTime = &now
			lease.Spec.RenewTime = &now
			_, updateErr := leases.Update(ctx, lease, metav1.UpdateOptions{})
			if apierrors.IsConflict(updateErr) {
				return false, nil
			}
			return updateErr == nil, updateErr
		},
	)
	test.Expect(acquireErr).NotTo(HaveOccurred())
	test.T().Logf("Acquired TLS profile test lock %s", identity)

	return func() {
		lease, getErr := leases.Get(context.Background(), "trainer-tls-profile-e2e", metav1.GetOptions{})
		if apierrors.IsNotFound(getErr) {
			return
		}
		if getErr != nil || lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != identity {
			test.T().Logf("TLS profile test lock was not held during release: %v", getErr)
			return
		}
		if deleteErr := leases.Delete(context.Background(), "trainer-tls-profile-e2e", metav1.DeleteOptions{}); deleteErr != nil && !apierrors.IsNotFound(deleteErr) {
			test.T().Logf("failed to release TLS profile test lock: %v", deleteErr)
		}
	}
}

func labelsToSelector(labels map[string]string) string {
	items := make([]string, 0, len(labels))
	for key, value := range labels {
		items = append(items, key+"="+value)
	}
	return strings.Join(items, ",")
}

func checkTrainerMetricsEndpoint(test Test, namespace, podName string) {
	metricsURL, stopPortForward := startTrainerMetricsPortForward(test, namespace, podName)
	defer stopPortForward()
	client := &http.Client{
		Transport: &http.Transport{
			// The endpoint is reached through a local port-forward; TLS and
			// ServiceMonitor configuration are validated by the scrape test.
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12}, //nolint:gosec
		},
		Timeout: 30 * time.Second,
	}

	request, err := http.NewRequestWithContext(test.Ctx(), http.MethodGet, metricsURL, nil)
	test.Expect(err).NotTo(HaveOccurred())
	response := eventuallyMetricsRequest(test, client, request)
	test.Expect(response.StatusCode).To(Equal(http.StatusUnauthorized))
	_ = response.Body.Close()

	request, err = http.NewRequestWithContext(test.Ctx(), http.MethodGet, metricsURL, nil)
	test.Expect(err).NotTo(HaveOccurred())
	request.Header.Set("Authorization", "Bearer invalid-token")
	response = eventuallyMetricsRequest(test, client, request)
	test.Expect(response.StatusCode).To(Equal(http.StatusForbidden))
	_ = response.Body.Close()

	request, err = http.NewRequestWithContext(test.Ctx(), http.MethodGet, metricsURL, nil)
	test.Expect(err).NotTo(HaveOccurred())
	prometheusServiceAccount, err := test.Client().Core().CoreV1().ServiceAccounts("openshift-monitoring").Get(
		test.Ctx(), "prometheus-k8s", metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	request.Header.Set("Authorization", "Bearer "+CreateToken(test, "openshift-monitoring", prometheusServiceAccount))
	response = eventuallyMetricsRequest(test, client, request)
	defer response.Body.Close()
	test.Expect(response.StatusCode).To(Equal(http.StatusOK))
	body, err := io.ReadAll(io.LimitReader(response.Body, maxMetricsResponseBytes+1))
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(len(body)).To(BeNumerically("<=", maxMetricsResponseBytes))
	test.Expect(string(body)).To(ContainSubstring("# HELP"))
}

func eventuallyMetricsRequest(test Test, client *http.Client, request *http.Request) *http.Response {
	var response *http.Response
	test.Eventually(func(g Gomega, ctx context.Context) {
		if response != nil {
			_ = response.Body.Close()
		}
		currentRequest := request.Clone(ctx)
		var err error
		response, err = client.Do(currentRequest)
		g.Expect(err).NotTo(HaveOccurred())
	}, metricsStartupTimeout, 2*time.Second).WithContext(test.Ctx()).Should(Succeed())
	return response
}

func startTrainerMetricsPortForward(test Test, namespace, podName string) (string, func()) {
	transport, upgrader, err := spdy.RoundTripperFor(test.Config())
	test.Expect(err).NotTo(HaveOccurred())

	portForwardURL := fmt.Sprintf(
		"%s/api/v1/namespaces/%s/pods/%s/portforward",
		strings.TrimRight(test.Config().Host, "/"), namespace, podName,
	)
	parsedPortForwardURL, err := url.Parse(portForwardURL)
	test.Expect(err).NotTo(HaveOccurred())
	dialer := spdy.NewDialer(
		upgrader,
		&http.Client{Transport: transport},
		http.MethodPost,
		parsedPortForwardURL,
	)
	stopChan := make(chan struct{})
	readyChan := make(chan struct{})
	var stdout, stderr bytes.Buffer
	forwarder, err := portforward.New(
		dialer,
		[]string{fmt.Sprintf(":%d", trainerMetricsPort)},
		stopChan,
		readyChan,
		&stdout,
		&stderr,
	)
	test.Expect(err).NotTo(HaveOccurred())

	forwardErr := make(chan error, 1)
	go func() {
		forwardErr <- forwarder.ForwardPorts()
	}()
	select {
	case <-readyChan:
	case err := <-forwardErr:
		test.T().Fatalf("failed to start metrics port-forward: %v: %s", err, stderr.String())
	case <-time.After(30 * time.Second):
		close(stopChan)
		test.T().Fatalf("timed out starting metrics port-forward: %s", stderr.String())
	}
	ports, err := forwarder.GetPorts()
	if err != nil || len(ports) == 0 {
		test.T().Fatalf("failed to determine local metrics port: %v", err)
	}
	localPort := ports[0].Local

	return fmt.Sprintf("https://127.0.0.1:%d/metrics", localPort), func() {
		close(stopChan)
		select {
		case <-forwardErr:
		case <-time.After(10 * time.Second):
			test.T().Logf("timed out stopping metrics port-forward: %s", stderr.String())
		}
	}
}

func deploymentAvailable(deployment *unstructured.Unstructured) bool {
	conditions, _, _ := unstructured.NestedSlice(deployment.Object, "status", "conditions")
	for _, rawCondition := range conditions {
		condition := rawCondition.(map[string]interface{})
		if condition["type"] == "Available" && condition["status"] == "True" {
			return true
		}
	}
	return false
}

package trainer

import (
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	trainerControllerService = "kubeflow-trainer-controller-manager"
	trainerMetricsPort       = int32(8443)
	maxMetricsResponseBytes  = 10 * 1024 * 1024
	metricsStartupTimeout    = 2 * time.Minute
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
	unauthorizedServiceAccount, err := test.Client().Core().CoreV1().ServiceAccounts(namespace).Get(
		test.Ctx(), "default", metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	request.Header.Set("Authorization", "Bearer "+CreateToken(test, namespace, unauthorizedServiceAccount))
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

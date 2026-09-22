package trainer

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
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
	rbacv1 "k8s.io/api/rbac/v1"
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
	trainerServiceMonitor       = "kubeflow-trainer-controller-manager-metrics-monitor"
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
	serviceMonitorGVR = schema.GroupVersionResource{
		Group: "monitoring.coreos.com", Version: "v1", Resource: "servicemonitors",
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
	expectedCertSecretName := mountedMetricsCertificateSecret(test, applicationsNamespace, pod)

	test.T().Logf("Checking Trainer metrics service %s/%s using Ready pod %s (restartCount=%d)", applicationsNamespace, serviceName, pod.Name, trainerControllerRestartCount(pod))
	certSecretName, serverName := checkTrainerServiceMonitor(test, applicationsNamespace, service, expectedCertSecretName)
	certSecret := EventuallySecret(test, applicationsNamespace, certSecretName)
	test.Expect(certSecret.Data).To(HaveKey("tls.crt"))
	test.Expect(certSecret.Data).To(HaveKey("tls.key"))
	test.Expect(certSecret.Data).To(HaveKey("ca.crt"))
	checkMetricsCertificate(test, certSecret.Data["tls.crt"], serverName)
	test.T().Logf("Validated metrics certificate Secret %s/%s and server name %q", applicationsNamespace, certSecretName, serverName)

	test.Expect(pod.Spec.ServiceAccountName).NotTo(BeEmpty())
	test.Expect(podUsesSecret(pod, certSecretName)).To(BeTrue(), "metrics certificate secret is not mounted")

	checkTrainerRBAC(test, applicationsNamespace, pod.Spec.ServiceAccountName)
	checkPrometheusRBAC(test)
	checkTrainerMetricsEndpoint(test, applicationsNamespace, pod.Name, certSecret.Data["ca.crt"], serverName)
	test.T().Logf("Validated authenticated HTTPS metrics endpoint on pod %s", pod.Name)
}

func TestTrainerPrometheusScrape(t *testing.T) {
	Tags(t, Tier2)
	test := With(t)
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())

	service, _ := trainerMetricsResources(test, applicationsNamespace)
	serviceName := service.GetName()
	prometheus := GetOpenShiftPrometheusApiClient(test)
	test.T().Logf("Waiting for Prometheus to discover Trainer ServiceMonitor target %s/%s", applicationsNamespace, serviceName)

	var target prometheusapiv1.ActiveTarget
	test.Eventually(func(g Gomega, ctx context.Context) {
		result, targetErr := prometheus.Targets(ctx)
		g.Expect(targetErr).NotTo(HaveOccurred())
		found := false
		for _, candidate := range result.Active {
			if string(candidate.Labels["namespace"]) != applicationsNamespace ||
				string(candidate.Labels["service"]) != serviceName {
				continue
			}
			target = candidate
			found = true
			break
		}
		g.Expect(found).To(BeTrue(), "Trainer ServiceMonitor target was not discovered by Prometheus")
		g.Expect(target.Health).To(Equal(prometheusapiv1.HealthGood))
		g.Expect(target.LastError).To(BeEmpty())
		g.Expect(target.ScrapeURL).To(HavePrefix("https://"))
		g.Expect(target.ScrapeURL).To(ContainSubstring(":8443"))
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
	if !RunTrainerTLSProfileWatcherE2E() {
		t.Skipf("set %s=true to run the cluster-wide TLS profile mutation test", TrainerTLSProfileWatcherE2E)
	}
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())
	releaseTLSProfileLock := acquireTLSProfileLock(test, applicationsNamespace)
	defer releaseTLSProfileLock()

	apiServer, err := test.Client().Dynamic().Resource(openShiftAPIServerGVR).Get(
		test.Ctx(), "cluster", metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		t.Skip("OpenShift APIServer is unavailable; TLS profile watcher is not applicable")
	}
	test.Expect(err).NotTo(HaveOccurred())

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

	defer func() {
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
				if apierrors.IsConflict(updateErr) {
					lastErr = updateErr
					return false, nil
				}
				return updateErr == nil, updateErr
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
				test.T().Errorf("failed waiting for Trainer recovery after restoring TLS profile: %v", recoveryErr)
			} else {
				test.T().Logf("Trainer recovered after restoring APIServer TLS profile")
			}
		}
	}()

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
			if apierrors.IsConflict(updateErr) {
				lastErr = updateErr
				return false, nil
			}
			return updateErr == nil, updateErr
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

func podUsesSecret(pod *corev1.Pod, name string) bool {
	for _, volume := range pod.Spec.Volumes {
		if volume.Secret != nil && volume.Secret.SecretName == name {
			return true
		}
	}
	return false
}

func EventuallySecret(test Test, namespace, name string) *corev1.Secret {
	var secret *corev1.Secret
	test.Eventually(func(g Gomega, ctx context.Context) {
		var err error
		secret, err = test.Client().Core().CoreV1().Secrets(namespace).Get(ctx, name, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(secret.Data).To(HaveKey("tls.crt"))
	}, 5*time.Minute, 5*time.Second).WithContext(test.Ctx()).Should(Succeed())
	return secret
}

func checkTrainerRBAC(test Test, namespace, serviceAccountName string) {
	bindings, err := test.Client().Core().RbacV1().ClusterRoleBindings().List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	roles := test.Client().Core().RbacV1().ClusterRoles()
	for _, binding := range bindings.Items {
		if !hasServiceAccount(binding.Subjects, namespace, serviceAccountName) {
			continue
		}
		role, err := roles.Get(test.Ctx(), binding.RoleRef.Name, metav1.GetOptions{})
		test.Expect(err).NotTo(HaveOccurred())
		if hasRule(role.Rules, "authentication.k8s.io", "tokenreviews") &&
			hasRule(role.Rules, "authorization.k8s.io", "subjectaccessreviews") {
			return
		}
	}
	test.T().Fatalf("Trainer service account %s/%s lacks TokenReview and SubjectAccessReview permissions", namespace, serviceAccountName)
}

func hasServiceAccount(subjects []rbacv1.Subject, namespace, name string) bool {
	for _, subject := range subjects {
		if subject.Kind == "ServiceAccount" && subject.Namespace == namespace && subject.Name == name {
			return true
		}
	}
	return false
}

func hasRule(rules []rbacv1.PolicyRule, apiGroup, resource string) bool {
	for _, rule := range rules {
		if contains(rule.APIGroups, apiGroup) && contains(rule.Resources, resource) && contains(rule.Verbs, "create") {
			return true
		}
	}
	return false
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted || value == "*" {
			return true
		}
	}
	return false
}

func checkTrainerServiceMonitor(test Test, namespace string, service *unstructured.Unstructured, expectedCertSecretName string) (string, string) {
	monitors, err := test.Client().Dynamic().Resource(serviceMonitorGVR).Namespace(namespace).List(
		test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	serviceLabels := service.GetLabels()
	for _, monitor := range monitors.Items {
		if monitor.GetName() != trainerServiceMonitor {
			continue
		}
		selector, _, selectorErr := unstructured.NestedStringMap(monitor.Object, "spec", "selector", "matchLabels")
		test.Expect(selectorErr).NotTo(HaveOccurred())
		test.Expect(selectorMatches(serviceLabels, selector)).To(BeTrue(), "ServiceMonitor does not select the Trainer metrics Service")

		endpoints, found, endpointErr := unstructured.NestedSlice(monitor.Object, "spec", "endpoints")
		test.Expect(endpointErr).NotTo(HaveOccurred())
		test.Expect(found).To(BeTrue())
		for _, rawEndpoint := range endpoints {
			endpoint, ok := rawEndpoint.(map[string]interface{})
			if !ok {
				continue
			}
			port, _, _ := unstructured.NestedString(endpoint, "port")
			if port != "monitoring-port" {
				continue
			}
			scheme, _, _ := unstructured.NestedString(endpoint, "scheme")
			test.Expect(scheme).To(Equal("https"))
			bearerTokenFile, _, _ := unstructured.NestedString(endpoint, "bearerTokenFile")
			test.Expect(bearerTokenFile).To(Equal("/var/run/secrets/kubernetes.io/serviceaccount/token"))
			caName, _, _ := unstructured.NestedString(endpoint, "tlsConfig", "ca", "secret", "name")
			test.Expect(caName).To(Equal(expectedCertSecretName))
			caKey, _, _ := unstructured.NestedString(endpoint, "tlsConfig", "ca", "secret", "key")
			test.Expect(caKey).To(Equal("ca.crt"))
			serverName, _, _ := unstructured.NestedString(endpoint, "tlsConfig", "serverName")
			test.Expect(serverName).To(Equal(fmt.Sprintf("%s.%s.svc", trainerControllerService, namespace)))
			return caName, serverName
		}
		test.T().Fatalf("HTTPS ServiceMonitor endpoint for Trainer was not found")
	}
	test.T().Fatalf("Trainer ServiceMonitor %s was not found", trainerServiceMonitor)
	return "", ""
}

func mountedMetricsCertificateSecret(test Test, namespace string, pod *corev1.Pod) string {
	for _, volume := range pod.Spec.Volumes {
		if volume.Secret == nil {
			continue
		}
		secret, err := test.Client().Core().CoreV1().Secrets(namespace).Get(
			test.Ctx(), volume.Secret.SecretName, metav1.GetOptions{})
		test.Expect(err).NotTo(HaveOccurred())
		if secret.Data["tls.crt"] != nil && secret.Data["tls.key"] != nil && secret.Data["ca.crt"] != nil {
			return volume.Secret.SecretName
		}
	}
	test.T().Fatalf("Trainer metrics certificate secret is not mounted in pod %s", pod.Name)
	return ""
}

func checkMetricsCertificate(test Test, encodedCertificate []byte, serverName string) {
	block, _ := pem.Decode(encodedCertificate)
	test.Expect(block).NotTo(BeNil())
	certificate, err := x509.ParseCertificate(block.Bytes)
	test.Expect(err).NotTo(HaveOccurred())
	now := time.Now()
	test.Expect(certificate.NotBefore.Before(now)).To(BeTrue())
	test.Expect(certificate.NotAfter.After(now)).To(BeTrue())
	test.Expect(certificate.VerifyHostname(serverName)).NotTo(HaveOccurred())
}

func checkPrometheusRBAC(test Test) {
	bindings, err := test.Client().Core().RbacV1().ClusterRoleBindings().List(test.Ctx(), metav1.ListOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	roles := test.Client().Core().RbacV1().ClusterRoles()
	for _, binding := range bindings.Items {
		if !hasServiceAccount(binding.Subjects, "openshift-monitoring", "prometheus-k8s") {
			continue
		}
		role, roleErr := roles.Get(test.Ctx(), binding.RoleRef.Name, metav1.GetOptions{})
		test.Expect(roleErr).NotTo(HaveOccurred())
		for _, rule := range role.Rules {
			if contains(rule.NonResourceURLs, "/metrics") && contains(rule.Verbs, "get") {
				return
			}
		}
	}
	test.T().Fatalf("Prometheus service account lacks GET permission for /metrics")
}

func selectorMatches(actual, expected map[string]string) bool {
	for key, value := range expected {
		if actual[key] != value {
			return false
		}
	}
	return true
}

func checkTrainerMetricsEndpoint(test Test, namespace, podName string, caPEM []byte, serverName string) {
	metricsURL, stopPortForward := startTrainerMetricsPortForward(test, namespace, podName)
	defer stopPortForward()
	rootCAs := x509.NewCertPool()
	test.Expect(rootCAs.AppendCertsFromPEM(caPEM)).To(BeTrue(), "metrics CA certificate is invalid")
	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{RootCAs: rootCAs, ServerName: serverName, MinVersion: tls.VersionTLS12},
		},
		Timeout: 30 * time.Second,
	}

	request, err := http.NewRequestWithContext(test.Ctx(), http.MethodGet, metricsURL, nil)
	test.Expect(err).NotTo(HaveOccurred())
	response := eventuallyMetricsRequest(test, client, request)
	test.Expect(response.StatusCode).To(Or(Equal(http.StatusUnauthorized), Equal(http.StatusForbidden)))
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

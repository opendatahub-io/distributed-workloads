/*
Copyright 2026.

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

package trainer

import (
	"context"
	"fmt"
	"net"
	"regexp"
	"strconv"
	"testing"
	"time"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	trainerNetpolCurlImage       = "curlimages/curl:8.12.1"
	trainerNetpolCurlContainer   = "curl"
	trainerNetpolConnectTimeout  = "5"
	trainerNetpolMaxTimeout      = "10"
	trainerNetpolWorkloadPort    = int32(18080)
	trainerNetpolReadinessMarker = "TRAINER_NETPOL_LOCAL_HTTP_READY"
)

var curlConnectTimePattern = regexp.MustCompile(`CURL_TIMING time_connect=([0-9]+(?:\.[0-9]+)?)`)

func TestTrainerNetworkPolicyWebhook(t *testing.T) {
	Tags(t, Tier3)
	runTrainerControllerNetworkPolicyTest(t, "webhook", "https", 9443, "/")
}

func TestTrainerNetworkPolicyMetrics(t *testing.T) {
	Tags(t, Tier3)
	runTrainerControllerNetworkPolicyTest(t, "metrics", "https", 8443, "/metrics")
}

func TestTrainerNetworkPolicyHealth(t *testing.T) {
	Tags(t, Tier3)
	runTrainerControllerNetworkPolicyTest(t, "health", "http", 8081, "/healthz")
}

func TestTrainerNetworkPolicyStatusServer(t *testing.T) {
	Tags(t, Tier3)
	runTrainerControllerNetworkPolicyTest(t, "status-server", "https", 10443, "/")
}

func TestTrainerNetworkPolicyUnpublished(t *testing.T) {
	Tags(t, Tier3)
	runTrainerControllerNetworkPolicyTest(t, "unpublished", "https", 31415, "/")
}

func TestTrainerNetworkPolicyWorkload(t *testing.T) {
	Tags(t, Tier3)
	test := With(t)

	workloadNamespace := test.NewTestNamespace().Name
	sourceNamespace := test.NewTestNamespace().Name
	image, err := trainerutils.GetImageFromClusterTrainingRuntime(test, trainerutils.DefaultClusterTrainingRuntimeCPU)
	test.Expect(err).NotTo(HaveOccurred(), "unable to resolve the CPU ClusterTrainingRuntime image")

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "trainer-netpol-workload-",
			Namespace:    workloadNamespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{Name: trainerutils.DefaultClusterTrainingRuntimeCPU},
			Trainer: &trainerv1alpha1.Trainer{
				Image:    Ptr(image),
				Command:  []string{"python", "-c", trainerNetpolHTTPServerCommand()},
				NumNodes: Ptr(int32(1)),
				ResourcesPerNode: &corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("100m"),
						corev1.ResourceMemory: resource.MustParse("128Mi"),
					},
				},
			},
		},
	}
	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(workloadNamespace).Create(
		test.Ctx(), trainJob, metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created CPU TrainJob %s/%s using runtime %q", workloadNamespace, createdTrainJob.Name, trainerutils.DefaultClusterTrainingRuntimeCPU)

	selector := "jobset.sigs.k8s.io/jobset-name=" + createdTrainJob.Name + ",jobset.sigs.k8s.io/replicatedjob-name=node"
	var workloadPod *corev1.Pod
	test.Eventually(func(g Gomega, ctx context.Context) {
		pods, listErr := test.Client().Core().CoreV1().Pods(workloadNamespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
		g.Expect(listErr).NotTo(HaveOccurred())
		for i := range pods.Items {
			candidate := &pods.Items[i]
			if candidate.Status.Phase == corev1.PodRunning && podReady(candidate) && candidate.Status.PodIP != "" {
				workloadPod = candidate.DeepCopy()
				return
			}
		}
		g.Expect(workloadPod).NotTo(BeNil(), "TrainJob %s has no Ready pod with a pod IP; observed pods: %+v", createdTrainJob.Name, pods.Items)
	}, TestTimeoutLong, 3*time.Second).WithContext(test.Ctx()).Should(Succeed())

	containerName := trainerNodeContainer(test, workloadPod)
	test.Eventually(PodLog(test, workloadNamespace, workloadPod.Name, corev1.PodLogOptions{Container: containerName}), TestTimeoutShort, time.Second).
		Should(ContainSubstring(trainerNetpolReadinessMarker), "TrainJob HTTP listener did not pass its in-pod HTTP readiness request")
	test.T().Logf("TrainJob pod %s/%s completed its local HTTP readiness request on port %d", workloadNamespace, workloadPod.Name, trainerNetpolWorkloadPort)

	logNetworkPolicies(test, sourceNamespace, workloadNamespace)
	probeTrainerNetworkPolicy(test, sourceNamespace, networkPolicyTarget{
		Name:   workloadPod.Namespace + "/" + workloadPod.Name,
		IP:     workloadPod.Status.PodIP,
		Port:   trainerNetpolWorkloadPort,
		Scheme: "http",
		Path:   "/",
	})
}

func runTrainerControllerNetworkPolicyTest(t *testing.T, endpoint, scheme string, port int32, path string) {
	test := With(t)
	sourceNamespace := test.NewTestNamespace().Name
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())
	deployment := getTrainerDeployment(test, applicationsNamespace)
	targetPods := readyTrainerControllerPods(test, applicationsNamespace, deployment)
	logNetworkPolicies(test, sourceNamespace, applicationsNamespace)

	for _, pod := range targetPods {
		test.T().Logf("Probing Trainer %s endpoint at %s/%s pod IP %s port %d", endpoint, applicationsNamespace, pod.Name, pod.Status.PodIP, port)
		probeTrainerNetworkPolicy(test, sourceNamespace, networkPolicyTarget{
			Name:   applicationsNamespace + "/" + pod.Name,
			IP:     pod.Status.PodIP,
			Port:   port,
			Scheme: scheme,
			Path:   path,
		})
	}
}

func readyTrainerControllerPods(test Test, namespace string, deployment *unstructured.Unstructured) []corev1.Pod {
	test.T().Helper()
	selector, _, err := unstructured.NestedStringMap(deployment.Object, "spec", "selector", "matchLabels")
	test.Expect(err).NotTo(HaveOccurred())
	desiredReplicas, found, err := unstructured.NestedInt64(deployment.Object, "spec", "replicas")
	test.Expect(err).NotTo(HaveOccurred())
	if !found {
		desiredReplicas = 1
	}
	test.Expect(desiredReplicas).To(BeNumerically(">", 0), "Trainer deployment has no desired replicas")

	var readyPods []corev1.Pod
	test.Eventually(func(g Gomega, ctx context.Context) {
		pods, listErr := test.Client().Core().CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: labelsToSelector(selector),
		})
		g.Expect(listErr).NotTo(HaveOccurred())
		readyPods = readyPods[:0]
		for i := range pods.Items {
			pod := &pods.Items[i]
			if pod.Status.Phase == corev1.PodRunning && podReady(pod) && pod.Status.PodIP != "" {
				readyPods = append(readyPods, *pod.DeepCopy())
			}
		}
		g.Expect(readyPods).To(HaveLen(int(desiredReplicas)), "expected every Trainer controller replica to be Ready with a pod IP; observed pods: %+v", pods.Items)
	}, TestTimeoutLong, 3*time.Second).WithContext(test.Ctx()).Should(Succeed())
	return readyPods
}

type networkPolicyTarget struct {
	Name   string
	IP     string
	Port   int32
	Scheme string
	Path   string
}

func probeTrainerNetworkPolicy(test Test, sourceNamespace string, target networkPolicyTarget) {
	test.T().Helper()
	address := net.JoinHostPort(target.IP, strconv.Itoa(int(target.Port)))
	url := fmt.Sprintf("%s://%s%s", target.Scheme, address, target.Path)
	args := []string{
		"--silent", "--show-error", "--verbose", "--noproxy", "*",
		"--connect-timeout", trainerNetpolConnectTimeout,
		"--max-time", trainerNetpolMaxTimeout,
		"--output", "/dev/null",
		"--write-out", "\nCURL_TIMING time_connect=%{time_connect} time_total=%{time_total}\n",
	}
	if target.Scheme == "https" {
		args = append(args, "--insecure")
	}
	args = append(args, url)

	probe := CreatePod(test, sourceNamespace, &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "trainer-netpol-curl-"},
		Spec: corev1.PodSpec{
			RestartPolicy:                corev1.RestartPolicyNever,
			AutomountServiceAccountToken: Ptr(false),
			EnableServiceLinks:           Ptr(false),
			SecurityContext: &corev1.PodSecurityContext{
				SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault},
			},
			Containers: []corev1.Container{{
				Name:            trainerNetpolCurlContainer,
				Image:           trainerNetpolCurlImage,
				ImagePullPolicy: corev1.PullIfNotPresent,
				Command:         []string{"curl"},
				Args:            args,
				SecurityContext: &corev1.SecurityContext{
					RunAsNonRoot:             Ptr(true),
					AllowPrivilegeEscalation: Ptr(false),
					Capabilities:             &corev1.Capabilities{Drop: []corev1.Capability{"ALL"}},
				},
			}},
		},
	})
	test.T().Logf("Probing %s from namespace %s at %s", target.Name, sourceNamespace, url)

	var terminated *corev1.ContainerStateTerminated
	var observedPod *corev1.Pod
	test.Eventually(func(g Gomega, ctx context.Context) {
		pod, getErr := test.Client().Core().CoreV1().Pods(sourceNamespace).Get(ctx, probe.Name, metav1.GetOptions{})
		g.Expect(getErr).NotTo(HaveOccurred())
		observedPod = pod
		for i := range pod.Status.ContainerStatuses {
			status := &pod.Status.ContainerStatuses[i]
			if status.Name == trainerNetpolCurlContainer {
				terminated = status.State.Terminated
				break
			}
		}
		g.Expect(terminated).NotTo(BeNil(), "curl container has not terminated; pod status: %+v", pod.Status)
	}, TestTimeoutLong, 2*time.Second).WithContext(test.Ctx()).Should(Succeed())

	output := GetPodLog(test, sourceNamespace, probe.Name, corev1.PodLogOptions{Container: trainerNetpolCurlContainer})
	test.T().Logf("curl probe pod %s/%s terminated with exit code %d; logs:\n%s", sourceNamespace, probe.Name, terminated.ExitCode, output)
	test.Expect(observedPod.Status.Phase).To(Equal(corev1.PodFailed), "one-shot curl pod should fail with curl's timeout exit status")
	test.Expect(terminated.ExitCode).To(Equal(int32(28)), "expected curl to fail because the TCP connection timed out")
	test.Expect(output).NotTo(ContainSubstring("Connected to"), "curl connected before timing out")

	timing := curlConnectTimePattern.FindStringSubmatch(output)
	test.Expect(timing).To(HaveLen(2), "curl did not report time_connect in its output")
	connectTime, parseErr := strconv.ParseFloat(timing[1], 64)
	test.Expect(parseErr).NotTo(HaveOccurred())
	test.Expect(connectTime).To(Equal(float64(0)), "curl established a connection before timing out")
}

func logNetworkPolicies(test Test, namespaces ...string) {
	test.T().Helper()
	for _, namespace := range namespaces {
		policies, err := test.Client().Core().NetworkingV1().NetworkPolicies(namespace).List(test.Ctx(), metav1.ListOptions{})
		test.Expect(err).NotTo(HaveOccurred())
		if len(policies.Items) == 0 {
			test.T().Logf("No NetworkPolicies found in namespace %s", namespace)
			continue
		}
		for i := range policies.Items {
			policy := &policies.Items[i]
			test.T().Logf("Existing NetworkPolicy %s/%s: podSelector=%+v policyTypes=%v ingress=%+v egress=%+v",
				policy.Namespace, policy.Name, policy.Spec.PodSelector, policy.Spec.PolicyTypes, policy.Spec.Ingress, policy.Spec.Egress)
		}
	}
}

func trainerNetpolHTTPServerCommand() string {
	return fmt.Sprintf(
		"from http.server import HTTPServer, SimpleHTTPRequestHandler; import threading, urllib.request; "+
			"server = HTTPServer(('0.0.0.0', %d), SimpleHTTPRequestHandler); "+
			"threading.Thread(target=server.serve_forever, daemon=True).start(); "+
			"urllib.request.urlopen('http://127.0.0.1:%d/', timeout=2).read(); "+
			"print(%q, flush=True); threading.Event().wait()",
		trainerNetpolWorkloadPort, trainerNetpolWorkloadPort, trainerNetpolReadinessMarker,
	)
}

func trainerNodeContainer(test Test, pod *corev1.Pod) string {
	test.T().Helper()
	for _, container := range pod.Spec.Containers {
		if container.Name == "node" {
			return container.Name
		}
	}
	test.T().Fatalf("TrainJob pod %s/%s has no trainer node container named %q", pod.Namespace, pod.Name, "node")
	return ""
}

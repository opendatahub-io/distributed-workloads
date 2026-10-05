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
	"strconv"
	"strings"
	"testing"
	"time"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	trainerNetpolCurlImage       = "registry.access.redhat.com/ubi9/ubi-minimal:9.8-1790754119@sha256:eba570d04193d1523a8576b1c0ff00e681c9edb1a41d4742559b6e3ff457601e"
	trainerNetpolCurlContainer   = "curl"
	trainerNetpolWorkloadPort    = int32(18080)
	trainerNetpolReadinessMarker = "TRAINER_NETPOL_LOCAL_HTTP_READY"
)

func TestTrainerNetworkPolicyBlocksIngressFromOtherNamespacesToMetrics(t *testing.T) {
	Tags(t, Tier2)
	metricsPort := controllerPortByName(t, "metrics")
	runTrainerControllerNetworkPolicyTest(t, "https", metricsPort)
}

func TestTrainerNetworkPolicyBlocksIngressFromOtherNamespacesToHealth(t *testing.T) {
	Tags(t, Tier2)
	healthPort := controllerPortByName(t, "health")
	runTrainerControllerNetworkPolicyTest(t, "http", healthPort)
}

func TestTrainerNetworkPolicyBlocksIngressFromOtherNamespacesToStatusServer(t *testing.T) {
	Tags(t, Tier2)
	statusServerPort := controllerPortByName(t, "status-server")
	runTrainerControllerNetworkPolicyTest(t, "https", statusServerPort)
}

// Probe an undeclared port to verify ingress is blocked beyond the named ports.
func TestTrainerNetworkPolicyBlocksIngressFromOtherNamespacesToUnpublishedPort(t *testing.T) {
	Tags(t, Tier2)
	unpublishedPort := int32(31415)
	runTrainerControllerNetworkPolicyTest(t, "https", unpublishedPort)
}

func TestTrainerNetworkPolicyBlocksIngressFromOtherNamespacesToWorkloadPort(t *testing.T) {
	Tags(t, Tier2)
	test := With(t)

	workloadNamespace := test.NewTestNamespace().Name
	sourceNamespace := test.NewTestNamespace().Name
	runTrainerWorkloadNetworkPolicyTest(t, sourceNamespace, workloadNamespace)
}

func TestTrainerNetworkPolicyBlocksIngressFromOtherWorkloadsToWorkloadPort(t *testing.T) {
	Tags(t, Tier2)
	test := With(t)

	workloadNamespace := test.NewTestNamespace().Name
	sourceNamespace := workloadNamespace
	runTrainerWorkloadNetworkPolicyTest(t, sourceNamespace, workloadNamespace)
}

// runTrainerControllerNetworkPolicyTest tests that a pod running in another namespace cannot reach the trainer controller
// on port.
func runTrainerControllerNetworkPolicyTest(t *testing.T, scheme string, port int32) {
	t.Helper()
	test := With(t)
	sourceNamespace := test.NewTestNamespace().Name
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())
	deployment := trainerutils.GetTrainerControllerDeployment(test, applicationsNamespace)
	pod := trainerutils.TrainerControllerPod(test, applicationsNamespace, deployment, TestTimeoutLong)
	assertNetworkConnectionTimesOut(test, sourceNamespace, pod, scheme, port)
}

// runTrainerWorkloadNetworkPolicyTest checks that a pod in sourceNamespace cannot reach a workload created in workloadNamespace
func runTrainerWorkloadNetworkPolicyTest(t *testing.T, sourceNamespace, workloadNamespace string) {
	test := With(t)

	image, err := trainerutils.GetImageFromClusterTrainingRuntime(test, trainerutils.DefaultClusterTrainingRuntimeCPU)
	test.Expect(err).NotTo(HaveOccurred(), "unable to resolve the CPU ClusterTrainingRuntime image")

	// create a TrainJob that exposers a server listening on trainerNetpolWorkloadPort
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
			if candidate.Status.Phase == corev1.PodRunning && trainerutils.PodReady(candidate) && candidate.Status.PodIP != "" {
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

	assertNetworkConnectionTimesOut(test, sourceNamespace, workloadPod, "http", trainerNetpolWorkloadPort)
}

func controllerPortByName(t *testing.T, name string) int32 {
	t.Helper()
	test := With(t)
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())
	deployment := trainerutils.GetTrainerControllerDeployment(test, applicationsNamespace)
	pod := trainerutils.TrainerControllerPod(test, applicationsNamespace, deployment, TestTimeoutLong)
	for _, container := range pod.Spec.Containers {
		for _, port := range container.Ports {
			if port.Name == name {
				return port.ContainerPort
			}
		}
	}
	test.T().Fatalf("controller pod %s has no port named %q", pod.Name, name)
	return 0
}

// assert that a pod from sourceNamespace cannot reach targetPod on targetPort.
func assertNetworkConnectionTimesOut(test Test, sourceNamespace string, targetPod *corev1.Pod, scheme string, targetPort int32) {
	test.T().Helper()
	address := net.JoinHostPort(targetPod.Status.PodIP, strconv.Itoa(int(targetPort)))
	url := fmt.Sprintf("%s://%s", scheme, address)
	output, exitCode := runNetworkPolicyCurlProbe(test, sourceNamespace, url)
	test.Expect(exitCode).To(Equal(int32(28)), "curl logs: %s", output)
	test.Expect(output).To(ContainSubstring("CURL_TIMING time_connect=0.000000 "),
		"expected curl to time out before establishing a TCP connection; logs: %s", output)
}

func runNetworkPolicyCurlProbe(test Test, sourceNamespace, url string) (string, int32) {
	test.T().Helper()
	args := []string{
		"--silent", "--show-error", "--verbose", "--noproxy", "*",
		"--connect-timeout", "5",
		"--max-time", "10",
		"--output", "/dev/null",
		"--write-out", "\nCURL_TIMING time_connect=%{time_connect} time_total=%{time_total}\n",
	}
	if strings.HasPrefix(url, "https://") {
		args = append(args, "--insecure")
	}
	args = append(args, url)

	probe := CreatePod(test, sourceNamespace, &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "trainer-netpol-curl-"},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{{
				Name:    trainerNetpolCurlContainer,
				Image:   trainerNetpolCurlImage,
				Command: []string{"curl"},
				Args:    args,
			}},
		},
	})

	var terminated *corev1.ContainerStateTerminated
	test.Eventually(func(g Gomega, ctx context.Context) {
		pod, getErr := test.Client().Core().CoreV1().Pods(sourceNamespace).Get(ctx, probe.Name, metav1.GetOptions{})
		g.Expect(getErr).NotTo(HaveOccurred())
		terminated = nil
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
	test.T().Logf("curl probe pod %s/%s target=%s exit=%d; logs:\n%s", sourceNamespace, probe.Name, url, terminated.ExitCode, output)
	return output, terminated.ExitCode
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

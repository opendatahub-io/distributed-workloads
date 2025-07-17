package kfto

import (
	"context"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	openshiftclient "github.com/openshift/client-go/config/clientset/versioned"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestKftoSmoke(t *testing.T) {
	Tags(t, Smoke)
	runSmoke(t, "kubeflow-training-operator", "odh-training-operator")
}

// runSmoke runs a smoke test for a given deployment and expected image name.
func runSmoke(t *testing.T, deploymentName string, expectedImage string) {
	test := With(t)
	namespace := GetOpenDataHubNamespace(test)

	test.T().Logf("Waiting for %s deployment to be available ...", deploymentName)
	test.Eventually(func(g Gomega, ctx context.Context) {
		deployment, err := test.Client().Core().AppsV1().Deployments(namespace).Get(
			ctx, deploymentName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred())

		status := ConditionStatus(appsv1.DeploymentAvailable)(deployment)
		g.Expect(status).To(Equal(corev1.ConditionTrue), "deployment %s not available", deploymentName)
	}, 5*time.Minute, 5*time.Second).WithContext(test.Ctx()).Should(Succeed())

	test.T().Logf("%s deployment is available", deploymentName)

	// Determine registry based on cluster environment
	configClient, err := openshiftclient.NewForConfig(test.Config())
	test.Expect(err).NotTo(HaveOccurred())

	infra, err := configClient.ConfigV1().Infrastructures().Get(test.Ctx(), "cluster", metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	envType := infra.Labels["hypershift.openshift.io/managed"]
	registryName := "registry.redhat.io"
	if envType == "true" {
		registryName = "quay.io"
	}

	test.T().Logf("Verifying %s container image is referred from expected registry ...", deploymentName)

	// List all running pods in the namespace
	podList := GetPods(test, namespace, metav1.ListOptions{
		FieldSelector: "status.phase=Running",
	})

	// Filter pods whose name starts with the prefix - deployment Name
	var matchedPods []corev1.Pod
	for _, pod := range podList {
		if strings.HasPrefix(pod.Name, deploymentName+"-") {
			matchedPods = append(matchedPods, pod)
		}
	}

	if len(matchedPods) != 1 {
		var podNames []string
		for _, pod := range matchedPods {
			podNames = append(podNames, pod.Name)
		}
		test.T().Logf("Found pods matching prefix '%s-': %v", deploymentName, podNames)
		test.T().Errorf("Expected exactly one pod, found %d", len(matchedPods))
		test.T().FailNow()
	}

	containerImage := matchedPods[0].Spec.Containers[0].Image
	test.Expect(containerImage).To(ContainSubstring(registryName + "/rhoai/" + expectedImage))
	test.T().Logf("%s container image is referred from %s", deploymentName, registryName)
}

package kfto

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

var (
	initialTrainingOperatorState string
	initialKueueState            string
)

func TestMain(m *testing.M) {
	var code int
	var setupFailed bool

	// Capture initial TrainingOperator state before running any tests
	initialTrainingOperatorState = CaptureComponentState(DefaultDSCName, "trainingoperator")
	fmt.Printf("Initial TrainingOperator managementState: %s\n", initialTrainingOperatorState)

	// Setup TrainingOperator to Managed if not already
	if initialTrainingOperatorState != "Managed" {
		if err := SetupComponent(DefaultDSCName, "trainingoperator", StateManaged); err != nil {
			fmt.Printf("Setup failed: %v\n", err)
			fmt.Println("Skipping test execution due to setup failure ...")
			setupFailed = true
			code = 1
		}
	} else {
		fmt.Println("Setup: Skipping TrainingOperator setup as it is already set to Managed in DataScienceCluster")
	}

	// Capture initial Kueue state before running any tests
	initialKueueState = CaptureComponentState(DefaultDSCName, "kueue")
	fmt.Printf("Initial Kueue managementState: %s\n", initialKueueState)

	// Run all tests only if setup succeeded
	if !setupFailed {
		code = m.Run()
	}

	// TearDown TrainingOperator: Only set to Removed if it was not already Managed before tests
	if initialTrainingOperatorState != "Managed" {
		if err := TearDownComponent(DefaultDSCName, "trainingoperator"); err != nil {
			fmt.Printf("TearDown: Failed to set TrainingOperator to Removed in DataScienceCluster: %v\n", err)
		}
	} else {
		fmt.Println("TearDown: Skipping TrainingOperator teardown as Initial TrainingOperator managementState was Managed in DataScienceCluster")
	}

	// TearDown Kueue: Only set to Removed if it was not already Unmanaged before tests
	if initialKueueState != "Unmanaged" {
		if err := TearDownComponent(DefaultDSCName, "kueue"); err != nil {
			fmt.Printf("TearDown: Failed to set Kueue to Removed: %v\n", err)
		}
	} else {
		fmt.Println("TearDown: Skipping Kueue teardown as Initial Kueue managementState was Unmanaged in DataScienceCluster")
	}

	os.Exit(code)
}

func TestKftoSmoke(t *testing.T) {
	Tags(t, Smoke)
	runSmoke(t, "kubeflow-training-operator", "odh-training-operator")
}

// runSmoke runs a smoke test for a given deployment and expected image name.
func runSmoke(t *testing.T, deploymentName string, expectedImage string) {
	test := With(t)
	namespace, err := GetApplicationsNamespaceFromDSCI(test, DefaultDSCIName)
	test.Expect(err).NotTo(HaveOccurred())

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
	registryName := GetExpectedRegistry(test)

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

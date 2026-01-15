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
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	defaultDSCName = "default-dsc"
)

var initialTrainingOperatorState string

func TestMain(m *testing.M) {
	var code int
	var setupFailed bool

	// Capture initial TrainingOperator state before running any tests
	initialTrainingOperatorState = captureComponentState("trainingoperator")
	fmt.Printf("Initial TrainingOperator managementState: %s\n", initialTrainingOperatorState)

	// Setup TrainingOperator to Managed if not already
	if initialTrainingOperatorState != "Managed" {
		if err := setupTrainingOperator(); err != nil {
			fmt.Printf("Setup failed: %v\n", err)
			fmt.Println("Skipping test execution due to setup failure ...")
			setupFailed = true
			code = 1
		}
	} else {
		fmt.Println("Setup: Skipping TrainingOperator setup as it is already set to Managed in DataScienceCluster")
	}

	// Run all tests only if setup succeeded
	if !setupFailed {
		code = m.Run()
	}

	// TearDown TrainingOperator: Only set to Removed if it was not already Managed before tests
	if initialTrainingOperatorState != "Managed" {
		if err := tearDownComponent("trainingoperator"); err != nil {
			fmt.Printf("TearDown: Failed to set TrainingOperator to Removed in DataScienceCluster: %v\n", err)
		}
	} else {
		fmt.Println("TearDown: Skipping TrainingOperator teardown as Initial TrainingOperator managementState was Managed in DataScienceCluster")
	}

	os.Exit(code)
}

func createDynamicClient() (dynamic.Interface, error) {
	cfg, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		clientcmd.NewDefaultClientConfigLoadingRules(),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig: %w", err)
	}

	dynamicClient, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %w", err)
	}

	return dynamicClient, nil
}

func captureComponentState(component string) string {
	dynamicClient, err := createDynamicClient()
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
		return ""
	}

	dsc, err := dynamicClient.Resource(DscGVR).Get(context.Background(), defaultDSCName, metav1.GetOptions{})
	if err != nil {
		fmt.Printf("Warning: Failed to get DSC: %v\n", err)
		return ""
	}

	return ComponentStatusManagementState(dsc, component)
}

func setupTrainingOperator() error {
	dynamicClient, err := createDynamicClient()
	if err != nil {
		return err
	}

	fmt.Println("Setup: Setting trainingoperator managementState to Managed in DataScienceCluster...")
	err = SetComponentStateAndWait(dynamicClient, context.Background(), defaultDSCName, "trainingoperator", StateManaged, 2*time.Minute)
	if err != nil {
		return err
	}

	fmt.Println("Setup: TrainingOperator is set to Managed managementState successfully")
	return nil
}

func tearDownComponent(component string) error {
	dynamicClient, err := createDynamicClient()
	if err != nil {
		return fmt.Errorf("TearDown: %w", err)
	}

	fmt.Printf("TearDown: Setting %s managementState to Removed in DataScienceCluster...\n", component)
	err = SetComponentStateAndWait(dynamicClient, context.Background(), defaultDSCName, component, StateRemoved, 2*time.Minute)
	if err != nil {
		return fmt.Errorf("TearDown: failed to set %s to Removed: %w", component, err)
	}

	fmt.Printf("TearDown: %s is set to Removed managementState successfully\n", component)
	return nil
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

/*
Copyright 2025.

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

package support

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/gomega"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
)

// DSC constants
const (
	DefaultDSCName = "default-dsc"

	// Phase values
	PhaseReady = "Ready"

	// Condition status values
	ConditionTrue  = "True"
	ConditionFalse = "False"

	// Condition reason values
	ReasonRemoved = "Removed"

	// Management state values
	StateManaged   = "Managed"
	StateRemoved   = "Removed"
	StateUnmanaged = "Unmanaged"
)

var DscGVR = schema.GroupVersionResource{
	Group:    "datasciencecluster.opendatahub.io",
	Version:  "v2",
	Resource: "datascienceclusters",
}

func getDSC(dynamicClient dynamic.Interface, ctx context.Context, name string) (*unstructured.Unstructured, error) {
	return dynamicClient.Resource(DscGVR).Get(ctx, name, metav1.GetOptions{})
}

func updateDSC(dynamicClient dynamic.Interface, ctx context.Context, dsc *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	return dynamicClient.Resource(DscGVR).Update(ctx, dsc, metav1.UpdateOptions{})
}

func getDSCPhase(dsc *unstructured.Unstructured) string {
	phase, found, err := unstructured.NestedString(dsc.Object, "status", "phase")
	if err != nil || !found {
		return ""
	}
	return phase
}

func waitForDSCReady(dynamicClient dynamic.Interface, ctx context.Context, dscName string) error {
	var phase string

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("%s: timed out waiting for DSC to be Ready - current state: %s", dscName, phase)
		default:
			dsc, err := getDSC(dynamicClient, ctx, dscName)
			if err != nil {
				return err
			}
			phase = getDSCPhase(dsc)
			if phase == PhaseReady {
				return nil
			}
			time.Sleep(5 * time.Second)
		}
	}
}

type ConditionDetails struct {
	Type    string
	Status  string
	Reason  string
	Message string
}

func componentReadyCondition(component string) string {
	if len(component) == 0 {
		return ""
	}
	return strings.ToUpper(component[:1]) + component[1:] + "Ready"
}

func getConditionDetails(dsc *unstructured.Unstructured, conditionType string) ConditionDetails {
	details := ConditionDetails{Type: conditionType}
	conditions, found, err := unstructured.NestedSlice(dsc.Object, "status", "conditions")
	if err != nil || !found {
		return details
	}
	for _, c := range conditions {
		condition, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		conditionTypeValue, ok := condition["type"].(string)
		if !ok {
			continue
		}
		if strings.EqualFold(conditionTypeValue, conditionType) {
			if status, ok := condition["status"].(string); ok {
				details.Status = status
			}
			if reason, ok := condition["reason"].(string); ok {
				details.Reason = reason
			}
			if message, ok := condition["message"].(string); ok {
				details.Message = message
			}
			return details
		}
	}
	return details
}

func setComponentState(dynamicClient dynamic.Interface, ctx context.Context, dscName, component, state string) error {
	// Retry up to 5 times to handle optimistic locking conflicts
	maxRetries := 5
	for i := 0; i < maxRetries; i++ {
		dsc, err := getDSC(dynamicClient, ctx, dscName)
		if err != nil {
			return err
		}

		err = unstructured.SetNestedField(dsc.Object, state, "spec", "components", component, "managementState")
		if err != nil {
			return err
		}

		_, err = updateDSC(dynamicClient, ctx, dsc)
		if err == nil {
			return nil
		}

		if apierrors.IsConflict(err) {
			time.Sleep(500 * time.Millisecond)
			continue
		}

		return err
	}
	return fmt.Errorf("failed to set component %s state after %d retries due to conflicts", component, maxRetries)
}

func waitForComponentReady(dynamicClient dynamic.Interface, ctx context.Context, dscName, component string) error {
	conditionType := componentReadyCondition(component)
	var condition ConditionDetails

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("%s: timed out waiting for component to be ready - status: %s, reason: %s, message: %s",
				conditionType, condition.Status, condition.Reason, condition.Message)
		default:
			dsc, err := getDSC(dynamicClient, ctx, dscName)
			if err != nil {
				return err
			}
			condition = getConditionDetails(dsc, conditionType)

			if condition.Status == ConditionTrue {
				return nil
			}

			time.Sleep(5 * time.Second)
		}
	}
}

func waitForComponentRemoved(dynamicClient dynamic.Interface, ctx context.Context, dscName, component string) error {
	conditionType := componentReadyCondition(component)
	var condition ConditionDetails

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("%s: timed out waiting for component to be removed - status: %s, reason: %s, message: %s",
				conditionType, condition.Status, condition.Reason, condition.Message)
		default:
			dsc, err := getDSC(dynamicClient, ctx, dscName)
			if err != nil {
				return err
			}
			condition = getConditionDetails(dsc, conditionType)

			if condition.Status == ConditionFalse && condition.Reason == ReasonRemoved {
				return nil
			}

			time.Sleep(5 * time.Second)
		}
	}
}

func SetComponentStateAndWait(dynamicClient dynamic.Interface, ctx context.Context, dscName, component, state string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if err := setComponentState(dynamicClient, ctx, dscName, component, state); err != nil {
		return err
	}

	if state == StateManaged || state == StateUnmanaged {
		if err := waitForComponentReady(dynamicClient, ctx, dscName, component); err != nil {
			return err
		}
	}

	if state == StateRemoved {
		if err := waitForComponentRemoved(dynamicClient, ctx, dscName, component); err != nil {
			return err
		}
	}

	return waitForDSCReady(dynamicClient, ctx, dscName)
}

func GetDSC(test Test, name string) (*unstructured.Unstructured, error) {
	return getDSC(test.Client().Dynamic(), test.Ctx(), name)
}

func SetComponentState(test Test, dscName, component, state string, timeout time.Duration) error {
	return SetComponentStateAndWait(test.Client().Dynamic(), test.Ctx(), dscName, component, state, timeout)
}

func DSCResource(test Test, name string) func(g gomega.Gomega) *unstructured.Unstructured {
	return func(g gomega.Gomega) *unstructured.Unstructured {
		dsc, err := GetDSC(test, name)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return dsc
	}
}

func ComponentStatusManagementState(dsc *unstructured.Unstructured, component string) string {
	state, found, err := unstructured.NestedString(dsc.Object, "status", "components", component, "managementState")
	if err != nil || !found {
		return ""
	}
	return state
}

func ComponentConditionStatus(dsc *unstructured.Unstructured, conditionType string) string {
	conditions, found, err := unstructured.NestedSlice(dsc.Object, "status", "conditions")
	if err != nil || !found {
		return ""
	}
	for _, c := range conditions {
		condition, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		conditionTypeValue, ok := condition["type"].(string)
		if !ok {
			continue
		}
		if strings.EqualFold(conditionTypeValue, conditionType) {
			if status, ok := condition["status"].(string); ok {
				return status
			}
		}
	}
	return ""
}

func CreateDynamicClient() (dynamic.Interface, error) {
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

func CaptureComponentState(dscName, component string) string {
	dynamicClient, err := CreateDynamicClient()
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
		return ""
	}

	dsc, err := dynamicClient.Resource(DscGVR).Get(context.Background(), dscName, metav1.GetOptions{})
	if err != nil {
		fmt.Printf("Warning: Failed to get DSC: %v\n", err)
		return ""
	}

	return ComponentStatusManagementState(dsc, component)
}

func SetupComponent(dscName, component, desiredState string) error {
	dynamicClient, err := CreateDynamicClient()
	if err != nil {
		return err
	}

	fmt.Printf("Setup: Setting %s managementState to %s in DataScienceCluster...\n", component, desiredState)
	err = SetComponentStateAndWait(dynamicClient, context.Background(), dscName, component, desiredState, 2*time.Minute)
	if err != nil {
		return err
	}

	fmt.Printf("Setup: %s is set to %s managementState successfully\n", component, desiredState)
	return nil
}

func TearDownComponent(dscName, component string) error {
	dynamicClient, err := CreateDynamicClient()
	if err != nil {
		return fmt.Errorf("TearDown: %w", err)
	}

	fmt.Printf("TearDown: Setting %s managementState to Removed in DataScienceCluster...\n", component)
	err = SetComponentStateAndWait(dynamicClient, context.Background(), dscName, component, StateRemoved, 2*time.Minute)
	if err != nil {
		return fmt.Errorf("TearDown: failed to set %s to Removed: %w", component, err)
	}

	fmt.Printf("TearDown: %s is set to Removed managementState successfully\n", component)
	return nil
}

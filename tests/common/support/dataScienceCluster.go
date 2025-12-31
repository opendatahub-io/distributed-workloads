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
)

// DSC constants
const (
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

func GetDSCPhase(dsc *unstructured.Unstructured) string {
	phase, found, err := unstructured.NestedString(dsc.Object, "status", "phase")
	if err != nil || !found {
		return ""
	}
	return phase
}

func WaitForDSCReady(dynamicClient dynamic.Interface, ctx context.Context, dscName string) error {
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
			phase = GetDSCPhase(dsc)
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

func GetConditionDetails(dsc *unstructured.Unstructured, conditionType string) ConditionDetails {
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
		if condition["type"] == conditionType {
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

func WaitForComponentReady(dynamicClient dynamic.Interface, ctx context.Context, dscName, component string) error {
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
			condition = GetConditionDetails(dsc, conditionType)

			if condition.Status == ConditionTrue {
				return nil
			}

			time.Sleep(5 * time.Second)
		}
	}
}

func WaitForComponentRemoved(dynamicClient dynamic.Interface, ctx context.Context, dscName, component string) error {
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
			condition = GetConditionDetails(dsc, conditionType)

			if condition.Status == "False" && condition.Reason == ReasonRemoved {
				return nil
			}

			time.Sleep(5 * time.Second)
		}
	}
}

// SetComponentStateAndWait sets a component's managementState and waits for completion.
// It creates a context with timeout that enforces the deadline across all operations.
func SetComponentStateAndWait(dynamicClient dynamic.Interface, ctx context.Context, dscName, component, state string, timeout time.Duration) error {
	// Create a context with timeout - this will automatically enforce the deadline
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if err := setComponentState(dynamicClient, ctx, dscName, component, state); err != nil {
		return err
	}

	if state == StateManaged {
		if err := WaitForComponentReady(dynamicClient, ctx, dscName, component); err != nil {
			return err
		}
	}

	if state == StateRemoved {
		if err := WaitForComponentRemoved(dynamicClient, ctx, dscName, component); err != nil {
			return err
		}
	}

	return WaitForDSCReady(dynamicClient, ctx, dscName)
}

func GetDSC(test Test, name string) (*unstructured.Unstructured, error) {
	return getDSC(test.Client().Dynamic(), test.Ctx(), name)
}

func SetComponentToUnmanaged(test Test, dscName string, component string) error {
	return setComponentState(test.Client().Dynamic(), test.Ctx(), dscName, component, StateUnmanaged)
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
		if condition["type"] == conditionType {
			if status, ok := condition["status"].(string); ok {
				return status
			}
		}
	}
	return ""
}

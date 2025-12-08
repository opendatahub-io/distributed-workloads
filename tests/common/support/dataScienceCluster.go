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
	"time"

	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
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

func setComponentState(dynamicClient dynamic.Interface, ctx context.Context, dscName, component, state string) error {
	dsc, err := getDSC(dynamicClient, ctx, dscName)
	if err != nil {
		return err
	}

	err = unstructured.SetNestedField(dsc.Object, state, "spec", "components", component, "managementState")
	if err != nil {
		return err
	}

	_, err = updateDSC(dynamicClient, ctx, dsc)
	return err
}

func WaitForComponentState(dynamicClient dynamic.Interface, ctx context.Context, dscName, component, expectedState string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		dsc, err := getDSC(dynamicClient, ctx, dscName)
		if err != nil {
			return err
		}
		currentState := ComponentStatusManagementState(dsc, component)
		if currentState == expectedState {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("timeout waiting for component %s to reach state %s", component, expectedState)
}

func SetComponentStateAndWait(dynamicClient dynamic.Interface, ctx context.Context, dscName, component, state string, timeout time.Duration) error {
	if err := setComponentState(dynamicClient, ctx, dscName, component, state); err != nil {
		return err
	}
	return WaitForComponentState(dynamicClient, ctx, dscName, component, state, timeout)
}

func GetDSC(test Test, name string) (*unstructured.Unstructured, error) {
	return getDSC(test.Client().Dynamic(), test.Ctx(), name)
}

func SetComponentToUnmanaged(test Test, dscName string, component string) error {
	return setComponentState(test.Client().Dynamic(), test.Ctx(), dscName, component, "Unmanaged")
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

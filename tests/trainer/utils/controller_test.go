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
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func controllerPortDeployment() *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"metadata": map[string]interface{}{"name": TrainerControllerDeployment, "namespace": "applications"},
		"spec": map[string]interface{}{
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"name":  "sidecar",
							"ports": []interface{}{map[string]interface{}{"name": "unpublished", "containerPort": int64(31415)}},
						},
						map[string]interface{}{
							"name":  "manager",
							"ports": []interface{}{map[string]interface{}{"name": "webhook", "containerPort": int64(19443)}},
						},
					},
				},
			},
		},
	}}
}

func TestTrainerControllerPortUsesDeploymentPort(t *testing.T) {
	port, err := TrainerControllerPort(controllerPortDeployment(), "webhook")
	if err != nil || port != 19443 {
		t.Fatalf("expected deployment port 19443, got port=%d err=%v", port, err)
	}
}

func TestTrainerControllerPortRejectsMissingManagerPort(t *testing.T) {
	if _, err := TrainerControllerPort(controllerPortDeployment(), "unpublished"); err == nil {
		t.Fatal("expected an error for a port declared only on a sidecar")
	}
}

func TestPodReadyRequiresTrueCondition(t *testing.T) {
	pod := &corev1.Pod{}
	if PodReady(pod) {
		t.Fatal("pod without a Ready condition must not be ready")
	}
	pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}}
	if PodReady(pod) {
		t.Fatal("pod with Ready=False must not be ready")
	}
	pod.Status.Conditions[0].Status = corev1.ConditionTrue
	if !PodReady(pod) {
		t.Fatal("pod with Ready=True must be ready")
	}
}

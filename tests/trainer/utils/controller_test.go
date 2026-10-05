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
)

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

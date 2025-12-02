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

package trainer

import (
	"fmt"
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	jobsetv1alpha2 "sigs.k8s.io/jobset/api/jobset/v1alpha2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

// EnsureNotebookServiceAccount ensures the Notebook ServiceAccount exists in the target namespace.
// This avoids webhook/controller failures when creating the Notebook CR.
func EnsureNotebookServiceAccount(t *testing.T, test Test, namespace string) {
	t.Helper()
	saName := "jupyter-nb-kube-3aadmin"
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: namespace}}
	_, err := test.Client().Core().CoreV1().ServiceAccounts(namespace).Create(test.Ctx(), sa, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("Failed to create ServiceAccount %s/%s: %v", namespace, saName, err)
	}
}

// CreateClusterTrainingRuntime creates a ClusterTrainingRuntime for multi-node training.
// It handles cleanup automatically via t.T().Cleanup().
func CreateClusterTrainingRuntime(t *testing.T, test Test, name string, numNodes, numGPUsPerNode int32, image string) *trainerv1alpha1.ClusterTrainingRuntime {
	t.Helper()

	runtime := &trainerv1alpha1.ClusterTrainingRuntime{
		TypeMeta: metav1.TypeMeta{
			APIVersion: trainerv1alpha1.SchemeGroupVersion.String(),
			Kind:       "ClusterTrainingRuntime",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"trainer.kubeflow.org/framework": "training-hub",
			},
		},
		Spec: trainerv1alpha1.TrainingRuntimeSpec{
			MLPolicy: &trainerv1alpha1.MLPolicy{
				NumNodes: &numNodes,
				MLPolicySource: trainerv1alpha1.MLPolicySource{
					Torch: &trainerv1alpha1.TorchMLPolicySource{
						NumProcPerNode: &intstr.IntOrString{
							Type:   intstr.Int,
							IntVal: numGPUsPerNode,
						},
					},
				},
			},
			Template: trainerv1alpha1.JobSetTemplateSpec{
				Spec: jobsetv1alpha2.JobSetSpec{
					ReplicatedJobs: []jobsetv1alpha2.ReplicatedJob{
						{
							Name:     "node",
							Replicas: 1, // Must always be 1; numNodes is controlled by MLPolicy.NumNodes
							Template: batchv1.JobTemplateSpec{
								ObjectMeta: metav1.ObjectMeta{
									Labels: map[string]string{
										"trainer.kubeflow.org/trainjob-ancestor-step": "trainer",
									},
								},
								Spec: batchv1.JobSpec{
									Template: corev1.PodTemplateSpec{
										Spec: corev1.PodSpec{
											Containers: []corev1.Container{
												{
													Name:  "node",
													Image: image,
													Resources: corev1.ResourceRequirements{
														Limits: corev1.ResourceList{
															"nvidia.com/gpu": resource.MustParse(fmt.Sprintf("%d", numGPUsPerNode)),
														},
														Requests: corev1.ResourceList{
															"nvidia.com/gpu": resource.MustParse(fmt.Sprintf("%d", numGPUsPerNode)),
														},
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	// Create the ClusterTrainingRuntime
	createdRuntime, err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Create(test.Ctx(), runtime, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("Failed to create ClusterTrainingRuntime %s: %v", name, err)
	}

	// If it already exists, get it instead
	if apierrors.IsAlreadyExists(err) {
		createdRuntime, err = test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Get(test.Ctx(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get existing ClusterTrainingRuntime %s: %v", name, err)
		}
		t.Logf("ClusterTrainingRuntime %s already exists, using existing one", name)
	} else {
		t.Logf("Created ClusterTrainingRuntime %s successfully", name)
	}

	// Register cleanup to delete the runtime after the test
	t.Cleanup(func() {
		err := test.Client().Trainer().TrainerV1alpha1().ClusterTrainingRuntimes().Delete(test.Ctx(), name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			t.Logf("Warning: Failed to delete ClusterTrainingRuntime %s: %v", name, err)
		}
	})

	return createdRuntime
}

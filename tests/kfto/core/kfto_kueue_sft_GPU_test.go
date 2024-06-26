/*
Copyright 2024.

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

package core

import (
	"testing"
	"time"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusmodel "github.com/prometheus/common/model"
)

func TestMultiGpuPytorchjobWithSFTtrainer(t *testing.T) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Create a ConfigMap with configuration
	configData := map[string][]byte{
		"config.json": ReadFile(test, "config_GPU.json"),
	}
	config := CreateConfigMap(test, namespace.Name, configData)

	// Create Kueue resources utilizing GPU
	rfSpec := kueuev1beta1.ResourceFlavorSpec{
		NodeLabels: map[string]string{"nvidia.com/gpu.present": "true"},
	}
	resourceFlavor := CreateKueueResourceFlavor(test, rfSpec)
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := kueuev1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []kueuev1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory"), corev1.ResourceName("nvidia.com/gpu")},
				Flavors: []kueuev1beta1.FlavorQuotas{
					{
						Name: kueuev1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []kueuev1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("2"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("5Gi"),
							},
							{
								Name:         corev1.ResourceName("nvidia.com/gpu"),
								NominalQuota: resource.MustParse("2"),
							},
						},
					},
				},
			},
		},
	}
	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, namespace.Name, clusterQueue.Name)

	// Create training PyTorch job
	tuningJob := createAlpacaPyTorchJob(test, namespace.Name, localQueue.Name, *config)

	// Make sure the Kueue Workload is admitted
	test.Eventually(KueueWorkloads(test, namespace.Name), TestTimeoutLong).
		Should(
			And(
				HaveLen(1),
				ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrueBecause("Workload failed to be admitted"))),
			),
		)

	// Make sure the PyTorch job is running
	test.Eventually(PytorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutLong).
		Should(WithTransform(PytorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	if IsOpenShift(test) {
		// Check that both GPUs were utilized recently
		// That itself doesn't guarantee that PyTorchJob generated the load in GPU, but is the best we can achieve for now
		test.Eventually(openShiftPrometheusGpuUtil(test), TestTimeoutMedium).
			Should(
				And(
					HaveLen(2),
					HaveEach(
						// Check that both GPUs were utilized on more than 90%
						HaveField("Value", BeNumerically(">", 90)),
					),
				),
			)
	}

	// Make sure the PyTorch job succeed
	test.Eventually(PytorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutLong).Should(WithTransform(PytorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)
}

func createAlpacaPyTorchJob(test Test, namespace, localQueueName string, config corev1.ConfigMap) *kftov1.PyTorchJob {
	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kfto-sft-",
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": localQueueName,
			},
		},
		Spec: kftov1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kftov1.ReplicaType]*kftov1.ReplicaSpec{
				"Master": {
					Replicas:      Ptr(int32(1)),
					RestartPolicy: "OnFailure",
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Tolerations: []corev1.Toleration{
								{
									Key:      "nvidia.com/gpu",
									Operator: corev1.TolerationOpExists,
								},
							},
							InitContainers: []corev1.Container{
								{
									Name:            "copy-model",
									Image:           GetBloomModelImage(),
									ImagePullPolicy: corev1.PullIfNotPresent,
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
									},
									Command: []string{"/bin/sh", "-c"},
									Args:    []string{"mkdir /tmp/model; cp -r /models/bloom-560m /tmp/model"},
								},
								{
									Name:            "copy-dataset",
									Image:           GetAlpacaDatasetImage(),
									ImagePullPolicy: corev1.PullIfNotPresent,
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
									},
									Command: []string{"/bin/sh", "-c"},
									Args:    []string{"mkdir /tmp/dataset; cp /dataset/alpaca_data_tenth.json /tmp/dataset/alpaca_data.json"},
								},
							},
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           GetFmsHfTuningImage(test),
									ImagePullPolicy: corev1.PullIfNotPresent,
									Env: []corev1.EnvVar{
										{
											Name:  "SFT_TRAINER_CONFIG_JSON_PATH",
											Value: "/etc/config/config.json",
										},
										{
											Name:  "HF_HOME",
											Value: "/tmp/huggingface",
										},
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "config-volume",
											MountPath: "/etc/config",
										},
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("2"),
											corev1.ResourceMemory: resource.MustParse("5Gi"),
											"nvidia.com/gpu":      resource.MustParse("2"),
										},
										Limits: corev1.ResourceList{
											"nvidia.com/gpu": resource.MustParse("2"),
										},
									},
									SecurityContext: &corev1.SecurityContext{
										RunAsNonRoot:           Ptr(true),
										ReadOnlyRootFilesystem: Ptr(true),
									},
								},
							},
							Volumes: []corev1.Volume{
								{
									Name: "config-volume",
									VolumeSource: corev1.VolumeSource{
										ConfigMap: &corev1.ConfigMapVolumeSource{
											LocalObjectReference: corev1.LocalObjectReference{
												Name: config.Name,
											},
										},
									},
								},
								{
									Name: "tmp-volume",
									VolumeSource: corev1.VolumeSource{
										EmptyDir: &corev1.EmptyDirVolumeSource{},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	tuningJob, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

func openShiftPrometheusGpuUtil(test Test) func(g Gomega) prometheusmodel.Vector {
	return func(g Gomega) prometheusmodel.Vector {
		prometheusApiClient := GetOpenShiftPrometheusApiClient(test)
		result, warnings, err := prometheusApiClient.Query(test.Ctx(), "DCGM_FI_DEV_GPU_UTIL", time.Now(), prometheusapiv1.WithTimeout(5*time.Second))
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(warnings).Should(HaveLen(0))
		return result.(prometheusmodel.Vector)
	}
}

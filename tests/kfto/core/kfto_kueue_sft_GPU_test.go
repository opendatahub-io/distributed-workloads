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
	"fmt"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusmodel "github.com/prometheus/common/model"
)

var numberOfGpus = 8

func TestMultiGpuPytorchjobGranite20bCodeInstruct(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_granite_20b_code_instruct.json")
}

func TestMultiGpuPytorchjobGranite34bCodeInstructLoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_granite_34b_code_instruct_lora.json")
}

func TestMultiGpuPytorchjobLlama213bChatHf(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_llama2_13b_chat_hf.json")
}

func TestMultiGpuPytorchjobLlama213bChatHfLoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_llama2_13b_chat_hf_lora.json")
}

func TestMultiGpuPytorchjobMetaLlama318b(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_meta_llama3_1_8b.json")
}

func TestMultiGpuPytorchjobMetaLlama38bInstruct(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_meta_llama3_8b_instruct.json")
}

func TestMultiGpuPytorchjobMetaLlama3170bLoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_meta_llama3_1_70b_lora.json")
}

func TestMultiGpuPytorchjobMetaLlama370bInstructLoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_meta_llama3_70b_instruct_lora.json")
}

func TestMultiGpuPytorchjobMistral7bv03(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_mistral_7b_v03.json")
}

func TestMultiGpuPytorchjobMixtral8x7bv01(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_mixtral_8x7b_v01.json")
}

func TestMultiGpuPytorchjobMixtral8x7bInstructv01LoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_mixtral_8x7b_instruct_v01_lora.json")
}

func TestMultiGpuPytorchjobMerlinite7b(t *testing.T) {
	runMultiGpuPytorchjob(t, "config_merlinite_7b.json")
}

func runMultiGpuPytorchjob(t *testing.T, modelConfigFile string) {
	test := With(t)

	namespace := GetMultiGpuNamespace(test)

	// Create a ConfigMap with configuration
	configData := map[string][]byte{
		"config.json": ReadFile(test, modelConfigFile),
	}
	config := CreateConfigMap(test, namespace, configData)
	defer test.Client().Core().CoreV1().ConfigMaps(namespace).Delete(test.Ctx(), config.Name, *metav1.NewDeleteOptions(0))

	// Create training PyTorch job
	tuningJob := createAlpacaPyTorchJob(test, namespace, *config)
	defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Delete(test.Ctx(), tuningJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the PyTorch job is running
	test.Eventually(PytorchJob(test, namespace, tuningJob.Name), TestTimeoutLong).
		Should(WithTransform(PytorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	if IsOpenShift(test) {
		// Check that GPUs were utilized recently
		// That itself doesn't guarantee that PyTorchJob generated the load in GPU, but is the best we can achieve for now
		test.Eventually(openShiftPrometheusGpuUtil(test, namespace), 60*time.Minute).
			Should(
				And(
					HaveLen(numberOfGpus),
					ContainElement(
						// Check that at lest some GPU was utilized on more than 50%
						HaveField("Value", BeNumerically(">", 50)),
					),
				),
			)
	}

	// Make sure the PyTorch job succeed
	test.Eventually(PytorchJob(test, namespace, tuningJob.Name), 60*time.Minute).Should(WithTransform(PytorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)
}

func createAlpacaPyTorchJob(test Test, namespace string, config corev1.ConfigMap) *kftov1.PyTorchJob {
	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kfto-sft-",
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
									Name:            "copy-dataset",
									Image:           GetAlpacaDatasetImage(),
									ImagePullPolicy: corev1.PullIfNotPresent,
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "scratch-volume",
											MountPath: "/mnt/scratch",
										},
									},
									Command: []string{"/bin/sh", "-c"},
									Args:    []string{"mkdir /mnt/scratch/dataset; cp /dataset/alpaca_data_hundredth.json /mnt/scratch/dataset/alpaca_data.json"},
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
											Value: "/mnt/scratch/huggingface-home",
										},
										{
											Name:  "HF_TOKEN",
											Value: GetHuggingFaceToken(test),
										},
										{
											Name:  "TMPDIR",
											Value: "/mnt/scratch",
										},
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "config-volume",
											MountPath: "/etc/config",
										},
										{
											Name:      "scratch-volume",
											MountPath: "/mnt/scratch",
										},
										{
											Name:      "output-volume",
											MountPath: "/mnt/output",
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("2"),
											corev1.ResourceMemory: resource.MustParse("10Gi"),
											"nvidia.com/gpu":      resource.MustParse(fmt.Sprint(numberOfGpus)),
										},
										Limits: corev1.ResourceList{
											"nvidia.com/gpu": resource.MustParse(fmt.Sprint(numberOfGpus)),
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
									Name: "scratch-volume",
									VolumeSource: corev1.VolumeSource{
										Ephemeral: &corev1.EphemeralVolumeSource{
											VolumeClaimTemplate: &corev1.PersistentVolumeClaimTemplate{
												Spec: corev1.PersistentVolumeClaimSpec{
													AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
													Resources: corev1.VolumeResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceStorage: resource.MustParse("500Gi"),
														},
													},
													VolumeMode: Ptr(corev1.PersistentVolumeFilesystem),
												},
											},
										},
									},
								},
								{
									Name: "output-volume",
									VolumeSource: corev1.VolumeSource{
										Ephemeral: &corev1.EphemeralVolumeSource{
											VolumeClaimTemplate: &corev1.PersistentVolumeClaimTemplate{
												Spec: corev1.PersistentVolumeClaimSpec{
													AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
													Resources: corev1.VolumeResourceRequirements{
														Requests: corev1.ResourceList{
															corev1.ResourceStorage: resource.MustParse("500Gi"),
														},
													},
													VolumeMode: Ptr(corev1.PersistentVolumeFilesystem),
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

	tuningJob, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}

func openShiftPrometheusGpuUtil(test Test, namespace string) func(g Gomega) prometheusmodel.Vector {
	return func(g Gomega) prometheusmodel.Vector {
		prometheusApiClient := GetOpenShiftPrometheusApiClient(test)
		result, warnings, err := prometheusApiClient.Query(test.Ctx(), "DCGM_FI_DEV_GPU_UTIL", time.Now(), prometheusapiv1.WithTimeout(5*time.Second))
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(warnings).Should(HaveLen(0))

		var util prometheusmodel.Vector
		for _, sample := range result.(prometheusmodel.Vector) {
			if string(sample.Metric["exported_namespace"]) == namespace {
				util = append(util, sample)
			}
		}

		return util
	}
}

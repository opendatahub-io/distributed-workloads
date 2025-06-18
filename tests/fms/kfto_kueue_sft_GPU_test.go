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

package fms

import (
	"fmt"
	"testing"
	"time"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusmodel "github.com/prometheus/common/model"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	"github.com/opendatahub-io/distributed-workloads/tests/kfto"
)

func TestMultiGpuPytorchjobAllamBeta13bChatGptq(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_allam_beta_13b_chat_gptq.json", 2, mountModelVolumeIntoMaster)
}

func TestMultiGpuPytorchjobGranite8bCodeInstructGptq(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_granite_8b_code_instruct_gptq.json", 2, mountModelVolumeIntoMaster)
}

func TestMultiGpuPytorchjobGranite20bCodeInstruct(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_granite_20b_code_instruct.json", 4)
}

func TestMultiGpuPytorchjobGranite34bCodeBaseGptq(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_granite_34b_code_base_gptq.json", 2, mountModelVolumeIntoMaster)
}

func TestMultiGpuPytorchjobGranite34bCodeInstructLoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_granite_34b_code_instruct_lora.json", 4)
}

func TestMultiGpuPytorchjobMetaLlama318b(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_meta_llama3_1_8b.json", 2)
}

func TestMultiGpuPytorchjobMetaLlama38bInstruct(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_meta_llama3_8b_instruct.json", 2)
}

func TestMultiGpuPytorchjobMetaLlama370bInstructGptqBlue(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_meta_llama3_70b_instruct_gptq_blue.json", 2, mountModelVolumeIntoMaster)
}

func TestMultiGpuPytorchjobMetaLlama31405bGptq(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_meta_llama3_1_405b_gptq.json", 8, mountModelVolumeIntoMaster)
}

func TestMultiGpuPytorchjobMetaLlama3170bLoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_meta_llama3_1_70b_lora.json", 4)
}

func TestMultiGpuPytorchjobMetaLlama370bInstructLoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_meta_llama3_70b_instruct_lora.json", 4)
}

func TestMultiGpuPytorchjobMistral7bv03Gptq(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_mistral_7b_v03_gptq.json", 2, mountModelVolumeIntoMaster)
}
func TestMultiGpuPytorchjobMistral7bv03(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_mistral_7b_v03.json", 2)
}

func TestMultiGpuPytorchjobMixtral8x7bv01(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_mixtral_8x7b_v01.json", 8)
}

func TestMultiGpuPytorchjobMixtral8x7bInstructv01Gptq(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_mixtral_8x7b_instruct_v01_gptq.json", 2, mountModelVolumeIntoMaster)
}

func TestMultiGpuPytorchjobMixtral8x7bInstructv01LoRa(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_mixtral_8x7b_instruct_v01_lora.json", 4)
}

func TestMultiGpuPytorchjobMerlinite7b(t *testing.T) {
	runMultiGpuPytorchjob(t, "resources/config_merlinite_7b.json", 2)
}

func runMultiGpuPytorchjob(t *testing.T, modelConfigFile string, numberOfGpus int, options ...Option[*kftov1.PyTorchJob]) {
	test := With(t)

	namespace := test.CreateOrGetTestNamespace().Name

	// Create a ConfigMap with configuration
	configData := map[string][]byte{
		"config.json": ReadFile(test, modelConfigFile),
	}
	config := CreateConfigMap(test, namespace, configData)
	defer test.Client().Core().CoreV1().ConfigMaps(namespace).Delete(test.Ctx(), config.Name, *metav1.NewDeleteOptions(0))

	// Create PVC for trained model
	outputPvc := CreatePersistentVolumeClaim(test, namespace, "200Gi", AccessModes(corev1.ReadWriteOnce))
	defer test.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Delete(test.Ctx(), outputPvc.Name, metav1.DeleteOptions{})

	// Create training PyTorch job
	tuningJob := createAlpacaPyTorchJob(test, namespace, *config, numberOfGpus, outputPvc.Name, options...)
	defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Delete(test.Ctx(), tuningJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the PyTorch job is running
	test.Eventually(PyTorchJob(test, namespace, tuningJob.Name), TestTimeoutLong).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

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
	test.Eventually(PyTorchJob(test, namespace, tuningJob.Name), 60*time.Minute).Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)

	_, bucketUploadSet := GetStorageBucketUploadName()
	if bucketUploadSet {
		uploadToS3(test, namespace, outputPvc.Name, "model")
	}
}

func createAlpacaPyTorchJob(test Test, namespace string, config corev1.ConfigMap, numberOfGpus int, outputPvc string, options ...Option[*kftov1.PyTorchJob]) *kftov1.PyTorchJob {
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
									Image:           kfto.GetAlpacaDatasetImage(),
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
											Name:  "HOME",
											Value: "/mnt/scratch/home",
										},
										{
											Name:  "TRITON_HOME",
											Value: "/mnt/scratch/triton-home",
										},
										{
											Name:  "TRITON_CACHE_DIR",
											Value: "/mnt/scratch/triton-cache",
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
															corev1.ResourceStorage: resource.MustParse("2000Gi"),
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
										PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
											ClaimName: outputPvc,
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

	for _, option := range options {
		test.Expect(option.ApplyTo(tuningJob)).To(Succeed())
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

var mountModelVolumeIntoMaster = ErrorOption[*kftov1.PyTorchJob](func(to *kftov1.PyTorchJob) error {
	pvcName, err := GetGptqModelPvcName()
	if err != nil {
		return err
	}

	modelVolume := corev1.Volume{
		Name: "model-volume",
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	}

	to.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Volumes = append(to.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Volumes, modelVolume)

	modelVolumeMount := corev1.VolumeMount{
		Name:      "model-volume",
		MountPath: "/mnt/model",
	}
	to.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Containers[0].VolumeMounts = append(to.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Containers[0].VolumeMounts, modelVolumeMount)
	return nil
})

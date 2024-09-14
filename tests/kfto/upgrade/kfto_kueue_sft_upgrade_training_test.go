/*
Copyright 2023.

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

package kfto

import (
	"testing"

	. "github.com/onsi/gomega"
	kftocore "github.com/opendatahub-io/distributed-workloads/tests/kfto/core"
	. "github.com/project-codeflare/codeflare-common/support"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"
	kueueacv1beta1 "sigs.k8s.io/kueue/client-go/applyconfiguration/kueue/v1beta1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
)

var (
	namespaceName      = "test-kfto-upgrade"
	resourceFlavorName = "rf-upgrade"
	clusterQueueName   = "cq-upgrade"
	pyTorchJobName     = "pytorch-upgrade"
)

func TestSetupPytorchjob(t *testing.T) {
	test := With(t)

	// Create a namespace
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespaceName,
		},
	}
	_, err := test.Client().Core().CoreV1().Namespaces().Create(test.Ctx(), namespace, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	// Create a ConfigMap with training dataset and configuration
	configData := map[string][]byte{
		"config.json":                   kftocore.ReadFile(test, "config.json"),
		"twitter_complaints_small.json": kftocore.ReadFile(test, "twitter_complaints_small.json"),
	}
	config := CreateConfigMap(test, namespaceName, configData)

	// Create Kueue resources
	resourceFlavor := &kueuev1beta1.ResourceFlavor{
		ObjectMeta: metav1.ObjectMeta{
			Name: resourceFlavorName,
		},
	}
	resourceFlavor, err = test.Client().Kueue().KueueV1beta1().ResourceFlavors().Create(test.Ctx(), resourceFlavor, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	clusterQueue := &kueuev1beta1.ClusterQueue{
		ObjectMeta: metav1.ObjectMeta{
			Name: clusterQueueName,
		},
		Spec: kueuev1beta1.ClusterQueueSpec{
			NamespaceSelector: &metav1.LabelSelector{},
			ResourceGroups: []kueuev1beta1.ResourceGroup{
				{
					CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory")},
					Flavors: []kueuev1beta1.FlavorQuotas{
						{
							Name: kueuev1beta1.ResourceFlavorReference(resourceFlavor.Name),
							Resources: []kueuev1beta1.ResourceQuota{
								{
									Name:         corev1.ResourceCPU,
									NominalQuota: resource.MustParse("8"),
								},
								{
									Name:         corev1.ResourceMemory,
									NominalQuota: resource.MustParse("12Gi"),
								},
							},
						},
					},
				},
			},
			StopPolicy: Ptr(kueuev1beta1.Hold),
		},
	}
	clusterQueue, err = test.Client().Kueue().KueueV1beta1().ClusterQueues().Create(test.Ctx(), clusterQueue, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())

	localQueue := CreateKueueLocalQueue(test, namespaceName, clusterQueue.Name, AsDefaultQueue)

	// Create training PyTorch job
	tuningJob := createPyTorchJob(test, namespaceName, localQueue.Name, *config)

	// Make sure the PyTorch job is suspended, waiting for ClusterQueue to be enabled
	test.Eventually(kftocore.PytorchJob(test, tuningJob.Namespace, pyTorchJobName), TestTimeoutShort).
		Should(WithTransform(kftocore.PytorchJobConditionSuspended, Equal(corev1.ConditionTrue)))
}

func TestRunPytorchjob(t *testing.T) {
	test := With(t)
	namespace := GetNamespaceWithName(test, namespaceName)

	// Cleanup everything in the end
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavorName, metav1.DeleteOptions{})
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueueName, metav1.DeleteOptions{})
	defer DeleteTestNamespace(test, namespace)

	// Enable ClusterQueue to process waiting PyTorchJob
	clusterQueue := kueueacv1beta1.ClusterQueue(clusterQueueName).WithSpec(kueueacv1beta1.ClusterQueueSpec().WithStopPolicy(kueuev1beta1.None))
	_, err := test.Client().Kueue().KueueV1beta1().ClusterQueues().Apply(test.Ctx(), clusterQueue, metav1.ApplyOptions{FieldManager: "application/apply-patch", Force: true})
	test.Expect(err).NotTo(HaveOccurred())

	// PyTorch job should be started now
	test.Eventually(kftocore.PytorchJob(test, namespaceName, pyTorchJobName), TestTimeoutLong).
		Should(WithTransform(kftocore.PytorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Make sure the PyTorch job succeed
	test.Eventually(kftocore.PytorchJob(test, namespaceName, pyTorchJobName), TestTimeoutLong).
		Should(WithTransform(kftocore.PytorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
}

func createPyTorchJob(test Test, namespace, localQueueName string, config corev1.ConfigMap) *kftov1.PyTorchJob {
	tuningJob := &kftov1.PyTorchJob{
		ObjectMeta: metav1.ObjectMeta{
			Name: pyTorchJobName,
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
							InitContainers: []corev1.Container{
								{
									Name:            "copy-model",
									Image:           kftocore.GetBloomModelImage(),
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
							},
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           kftocore.GetFmsHfTuningImage(test),
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
											corev1.ResourceMemory: resource.MustParse("7Gi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("2"),
											corev1.ResourceMemory: resource.MustParse("7Gi"),
										},
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
											Items: []corev1.KeyToPath{
												{
													Key:  "config.json",
													Path: "config.json",
												},
												{
													Key:  "twitter_complaints_small.json",
													Path: "twitter_complaints_small.json",
												},
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

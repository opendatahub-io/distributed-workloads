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

package odh

import (
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestInstructlabTrainingOnRhoai(t *testing.T) {
	instructlabDistributedTrainingOnRhoai(t, 1)
}

func instructlabDistributedTrainingOnRhoai(t *testing.T, numGpus int) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Download standalone script used for running instructlab distributed training on RHOAI
	standaloneFilePath := "resources/standalone.py"
	cmd := exec.Command("curl", "-L", "-o", standaloneFilePath, "https://github.com/redhat-et/ilab-on-ocp/raw/refs/heads/main/standalone/standalone.py")
	err := cmd.Run()
	if err != nil {
		test.T().Logf(err.Error())
		test.Expect(err).ToNot(HaveOccurred())
	}
	test.T().Logf("File '%s' downloaded sucessfully", standaloneFilePath)

	// Create configmap to store standalone script and mount in workbench pod
	fileContent, err := os.ReadFile(standaloneFilePath)
	configMap := map[string][]byte{
		"standalone.py": fileContent,
	}
	createdCM := CreateConfigMap(test, namespace.Name, configMap)
	defer test.Client().Core().CoreV1().ConfigMaps(namespace.Name).Delete(test.Ctx(), createdCM.Name, metav1.DeleteOptions{})

	// Create Service account
	createdSA := CreateServiceAccount(test, namespace.Name)
	defer test.Client().Core().CoreV1().ServiceAccounts(namespace.Name).Delete(test.Ctx(), createdSA.Name, metav1.DeleteOptions{})

	// Create cluster role
	policyRules := []rbacv1.PolicyRule{
		{
			APIGroups: []string{""},
			Resources: []string{"pods", "pods/log", "services", "secrets", "jobs", "persistentvolumes", "persistentvolumeclaims"},
			Verbs: []string{
				"get", "list", "create", "watch", "delete", "update", "patch",
			},
		},
	}
	createdCR := CreateClusterRole(test, policyRules)
	defer test.Client().Core().RbacV1().ClusterRoles().Delete(test.Ctx(), createdCR.Name, metav1.DeleteOptions{})

	// Create cluster binding
	clusterRoleBinding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-crb-",
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      createdSA.Name,
				Namespace: namespace.Name,
			},
		},
		RoleRef: rbacv1.RoleRef{
			Kind: "ClusterRole",
			Name: createdCR.Name,
		},
	}
	createdCRB, err := test.Client().Core().RbacV1().ClusterRoleBindings().Create(test.Ctx(), clusterRoleBinding, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Created ClusterRoleBinding %s successfully", createdCRB.Name)
	defer test.Client().Core().RbacV1().ClusterRoleBindings().Delete(test.Ctx(), createdCRB.Name, metav1.DeleteOptions{})

	// Get S3 bucket credentials from environment variables
	s3BucketName, s3BucketNameExists := GetStorageBucketName()
	s3AccessKeyId, _ := GetStorageBucketAccessKeyId()
	s3SecretAccessKey, _ := GetStorageBucketSecretKey()
	s3DefaultRegion, _ := GetStorageBucketDefaultRegion()
	s3BucketDefaultEndpoint, _ := GetStorageBucketDefaultEndpoint()
	s3BucketDataKey, s3BucketDataKeyExists := GetStorageBucketDataKey()
	s3BucketVerifyTls, _ := GetStorageBucketVerifyTls()

	if !s3BucketNameExists {
		test.T().Logf("Please provide S3 bucket credentials to download SDG data from..")
	}
	if !s3BucketDataKeyExists {
		test.T().Logf("Please provide S3 bucket credentials to download SDG data from..")
	}

	// Create secret to store S3 bucket credentials to mount it in workbench pod
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-secret-",
			Namespace:    namespace.Name,
		},
		Type: corev1.SecretTypeOpaque,
		StringData: map[string]string{
			"bucket":     s3BucketName,
			"access_key": s3AccessKeyId,
			"secret_key": s3SecretAccessKey,
			"data_key":   s3BucketDataKey,
			"endpoint":   s3BucketDefaultEndpoint,
			"region":     s3DefaultRegion,
			"verify_tls": s3BucketVerifyTls,
		},
	}
	createdSecret, err := test.Client().Core().CoreV1().Secrets(namespace.Name).Create(test.Ctx(), secret, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Secret '%s' created successfully\n", createdSecret.Name)

	// Create KFP-server configmap
	kfpConfigmap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kfp-model-server",
			Namespace: namespace.Name,
		},
		Data: map[string]string{
			"endpoint": "https://mistral-7b-instruct-v02-sallyom.apps.ocp-beta-test.nerc.mghpcc.org/v1",
			"model":    "mistral-7b-instruct-v02",
		},
	}

	createdKfpCM, err := test.Client().Core().CoreV1().ConfigMaps(namespace.Name).Create(test.Ctx(), kfpConfigmap, metav1.CreateOptions{})
	test.T().Logf("Created %s configmap successfully", createdKfpCM.Name)
	test.Expect(err).ToNot(HaveOccurred())
	defer test.Client().Core().CoreV1().ConfigMaps(namespace.Name).Delete(test.Ctx(), createdKfpCM.Name, metav1.DeleteOptions{})

	// Create KFP-model-server secret
	kfpSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kfp-model-server",
			Namespace: namespace.Name,
		},
		Type: corev1.SecretTypeOpaque,
		StringData: map[string]string{
			"api_key": "ksdadcad",
		},
	}
	_, err = test.Client().Core().CoreV1().Secrets(namespace.Name).Create(test.Ctx(), kfpSecret, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Created %s secret successfully", kfpSecret.Name)
	defer test.Client().Core().CoreV1().Secrets(namespace.Name).Delete(test.Ctx(), kfpSecret.Name, metav1.DeleteOptions{})

	// Create pod resource using workbench image to run standalone script
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-workbench-pod-",
			Namespace:    namespace.Name,
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: createdSA.Name,
			Containers: []corev1.Container{
				{
					Name:  "workbench-container",
					Image: "quay.io/opendatahub/workbench-images@sha256:7f26f5f2bec4184af15acd95f29b3450526c5c28c386b6cb694fbe82d71d0b41",
					SecurityContext: &corev1.SecurityContext{
						AllowPrivilegeEscalation: Ptr(false),
						Capabilities: &corev1.Capabilities{
							Drop: []corev1.Capability{"ALL"},
						},
						SeccompProfile: &corev1.SeccompProfile{
							Type: corev1.SeccompProfileTypeRuntimeDefault,
						},
					},
					Env: []corev1.EnvVar{
						{
							Name: "SDG_OBJECT_STORE_ENDPOINT",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: createdSecret.Name,
									},
									Key: "endpoint",
								},
							},
						},
						{
							Name: "SDG_OBJECT_STORE_BUCKET",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: createdSecret.Name,
									},
									Key: "bucket",
								},
							},
						},
						{
							Name: "SDG_OBJECT_STORE_ACCESS_KEY",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: createdSecret.Name,
									},
									Key: "access_key",
								},
							},
						},
						{
							Name: "SDG_OBJECT_STORE_SECRET_KEY",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: createdSecret.Name,
									},
									Key: "secret_key",
								},
							},
						},
						{
							Name: "SDG_OBJECT_STORE_REGION",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: createdSecret.Name,
									},
									Key: "region",
								},
							},
						},
						{
							Name: "SDG_OBJECT_STORE_DATA_KEY",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: createdSecret.Name,
									},
									Key: "data_key",
								},
							},
						},
						{
							Name: "SDG_OBJECT_STORE_VERIFY_TLS",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: createdSecret.Name,
									},
									Key: "verify_tls",
								},
							},
						},
					},
					VolumeMounts: []corev1.VolumeMount{
						{
							Name:      "script-volume",
							MountPath: "/home/standalone.py",
							SubPath:   "standalone.py",
						},
					},
					Command: []string{
						"python3", "/home/standalone.py", "run",
						"--namespace", namespace.Name,
						"--judge-serving-model-endpoint", "http://serving.kubeflow.svc.cluster.local:8080/v1",
						"--judge-serving-model-name", "prometheus-eval/prometheus-8x7b-v2.0",
						"--judge-serving-model-api-key", "dummy-value",
						"--nproc-per-node", strconv.Itoa(numGpus),
						"--storage-class", "nfs",
						"--sdg-object-store-secret", createdSecret.Name,
						"--training-1-epoch-num", strconv.Itoa(1),
						"--training-2-epoch-num", strconv.Itoa(1),
						"--force-pull",
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "script-volume",
					VolumeSource: corev1.VolumeSource{
						ConfigMap: &corev1.ConfigMapVolumeSource{
							LocalObjectReference: corev1.LocalObjectReference{Name: createdCM.Name},
						},
					},
				},
			},
		},
	}
	createdPod, err := test.Client().Core().CoreV1().Pods(namespace.Name).Create(test.Ctx(), pod, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Pod '%s' created successfully\n", createdPod.Name)

	// Wait until workbench pod status becomes succeeded - timeout in 60mins
	var workbenchPod *corev1.Pod
	test.Eventually(func() corev1.PodPhase {
		workbenchPod, err = test.Client().Core().CoreV1().Pods(namespace.Name).Get(test.Ctx(), createdPod.Name, metav1.GetOptions{})
		test.Expect(err).To(BeNil())
		return workbenchPod.Status.Phase
	}, 60*time.Minute, 2*time.Second).Should(Equal(corev1.PodSucceeded))

	// Clean up downloaded files
	err = os.Remove("resources/standalone.py")
	test.Expect(err).ToNot(HaveOccurred())

}

func GetStorageBucketDataKey() (string, bool) {
	data_key, exists := os.LookupEnv("SDG_OBJECT_STORE_DATA_KEY")
	return data_key, exists
}

func GetStorageBucketVerifyTls() (string, bool) {
	data_key, exists := os.LookupEnv("SDG_OBJECT_STORE_VERIFY_TLS")
	return data_key, exists
}

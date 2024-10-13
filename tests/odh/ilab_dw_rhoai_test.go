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
	"context"
	"fmt"
	"os"
	"os/exec"
	"testing"
	"time"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestInstructlabTrainingOnRhoai(t *testing.T) {
	instructlabDistributedTrainingOnRhoai(t, 0)
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
		return
	}
	test.T().Logf("File '%s' downloaded sucessfully", standaloneFilePath)

	// Create configmap to store standalone script and mount in workbench pod
	fileContent, err := os.ReadFile(standaloneFilePath)
	configMap := map[string][]byte{
		"standalone.py": fileContent,
	}

	createdConfigMap := CreateConfigMap(test, namespace.Name, configMap)

	// Create Service account
	serviceAccount := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-sa-",
			Namespace:    namespace.Name,
		},
	}
	createdSA, err := test.Client().Core().CoreV1().ServiceAccounts(namespace.Name).Create(context.TODO(), serviceAccount, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Service account '%s' created successfully\n", createdSA.Name)

	// Create cluster role
	clusterRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-cr-",
		},

		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"pods", "services", "secrets", "jobs", "persistentvolumes", "persistentvolumeclaims"},
				Verbs: []string{
					"get", "list", "create", "watch", "delete", "update", "patch",
				},
			},
		},
	}
	createdCR, err := test.Client().Core().RbacV1().ClusterRoles().Create(context.TODO(), clusterRole, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Cluster role '%s' created successfully\n", createdCR.Name)

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
	createdCRB, err := test.Client().Core().RbacV1().ClusterRoleBindings().Create(context.TODO(), clusterRoleBinding, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Cluster role binding '%s' created successfully\n", createdCRB.Name)

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
	createdSecret, err := test.Client().Core().CoreV1().Secrets(namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Cluster role binding '%s' created successfully\n", createdSecret.Name)

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
						AllowPrivilegeEscalation: BoolPtr(false),
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
						"--judge-serving-endpoint", "http://serving.kubeflow.svc.cluster.local:8080/v1",
						"--judge-serving-model-name", "prometheus-eval/prometheus-8x7b-v2.0",
						"--judge-serving-model-api-key", "dummy-value",
						"--nproc-per-node", string(numGpus),
						"--storage-class", "managed-nfs-storage",
						"--sdg-object-store-secret", createdSecret.Name,
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "script-volume",
					VolumeSource: corev1.VolumeSource{
						ConfigMap: &corev1.ConfigMapVolumeSource{
							LocalObjectReference: corev1.LocalObjectReference{Name: createdConfigMap.Name},
						},
					},
				},
			},
		},
	}
	createdPod, err := test.Client().Core().CoreV1().Pods(namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	test.Expect(err).ToNot(HaveOccurred())
	test.T().Logf("Pod '%s' created successfully\n", createdPod.Name)

	time.Sleep(30) // interrupt test here to inspect pod logs and operations in detail

	// Wait for 30 mins for workbench pod status to be succeeded
	var workbenchPod *corev1.Pod
	test.Eventually(func() corev1.PodPhase {
		workbenchPod, err = test.Client().Core().CoreV1().Pods(namespace.Name).Get(context.TODO(), createdPod.Name, metav1.GetOptions{})
		test.Expect(err).To(BeNil())
		return workbenchPod.Status.Phase
	}, 30*time.Minute, 2*time.Second).Should(Equal(corev1.PodSucceeded))

	// cleaup all resources created if pod doesn't succeed in given time
	defer func() {
		fmt.Println("Pod did not succeed, cleaning up resources..")
		// Delete created workbench pod
		err = test.Client().Core().CoreV1().Pods(namespace.Name).Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
		test.Expect(err).To(BeNil())
		// Delete created cluster role binding
		err = test.Client().Core().RbacV1().ClusterRoleBindings().Delete(context.TODO(), createdCRB.Name, metav1.DeleteOptions{})
		test.Expect(err).To(BeNil())
		// Delete created cluster role
		err = test.Client().Core().RbacV1().ClusterRoles().Delete(context.TODO(), createdCR.Name, metav1.DeleteOptions{})
		test.Expect(err).To(BeNil())
		// Delete created service account
		err = test.Client().Core().CoreV1().ServiceAccounts(namespace.Name).Delete(context.TODO(), createdSA.Name, metav1.DeleteOptions{})
		test.Expect(err).To(BeNil())
	}()

	// Clean up downloaded files
	err = os.Remove("resources/standalone.py")
	test.Expect(err).ToNot(HaveOccurred())

}

func BoolPtr(b bool) *bool {
	return &b
}

func GetStorageBucketDataKey() (string, bool) {
	data_key, exists := os.LookupEnv("SDG_OBJECT_STORE_DATA_KEY")
	return data_key, exists
}

func GetStorageBucketVerifyTls() (string, bool) {
	data_key, exists := os.LookupEnv("SDG_OBJECT_STORE_VERIFY_TLS")
	fmt.Println(data_key)
	return data_key, exists
}

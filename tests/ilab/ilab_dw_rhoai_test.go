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
	"fmt"
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

// setting some defaults in case not provided.
const (
	ILAB_RHELAI_WORKBENCH_IMAGE = "quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.11-20241004-609ffb8"
	ILAB_RHELAI_STORAGE_CLASS   = "nfs-csi"
)

func TestInstructlabTrainingOnRhoai(t *testing.T) {
	instructlabDistributedTrainingOnRhoai(t, 1)
}

func instructlabDistributedTrainingOnRhoai(t *testing.T, numGpus int) {
	test := With(t)

	// Pre-requisites :

	rhelaiWorkbenchImage, rhelaiWorkbenchImageExists := GetRhelaiWorkbenchImage()
	if !rhelaiWorkbenchImageExists {
		rhelaiWorkbenchImage = ILAB_RHELAI_WORKBENCH_IMAGE

		test.T().Logf("RHELAI workbench image is not provided as environment variable. Using workbench image: %s", ILAB_RHELAI_WORKBENCH_IMAGE)
	}

	standaloneScriptURL, standaloneScriptURLExists := GetStandaloneScriptURL()
	if !standaloneScriptURLExists {
		test.T().Skip("The standalones script URL is not provided")
	}

	// Get S3 bucket credentials using environment variables
	s3BucketName, s3BucketNameExists := GetStorageBucketName()
	s3AccessKeyId, _ := GetStorageBucketAccessKeyId()
	s3SecretAccessKey, _ := GetStorageBucketSecretKey()
	s3DefaultRegion, _ := GetStorageBucketDefaultRegion()
	s3BucketDefaultEndpoint, _ := GetStorageBucketDefaultEndpoint()
	s3BucketDataKey, s3BucketDataKeyExists := GetStorageBucketDataKey()
	s3BucketVerifyTls, _ := GetStorageBucketVerifyTls()

	if !s3BucketNameExists {
		test.T().Skip("AWS_STORAGE_BUCKET Bucket name is required.")
	}
	if !s3BucketDataKeyExists {
		test.T().Skip("SDG_OBJECT_STORE_DATA_KEY is required to download required data to start training.")
	}

	ilabStorageClassName, ilabStorageClassNameExists := GetStorageClassName()
	if !ilabStorageClassNameExists {
		ilabStorageClassName = ILAB_RHELAI_STORAGE_CLASS

		test.T().Logf("Storage class is not provided. Using default %s", ilabStorageClassName)
	}

	// Create a namespace
	test_namespace, test_namespace_exists := GetTestNamespace()
	var namespace *corev1.Namespace

	if !test_namespace_exists {
		namespace = test.NewTestNamespace()
	} else {
		_, namespace_exists_err := test.Client().Core().CoreV1().Namespaces().Get(test.Ctx(), test_namespace, metav1.GetOptions{})

		if namespace_exists_err != nil {

			test.T().Logf("%s namespace doesn't exists. Creating ...", test_namespace)
			namespace = CreateTestNamespaceWithName(test, test_namespace)

		} else {
			namespace = GetNamespaceWithName(test, test_namespace)
			test.T().Logf("Using the namespace name which is provided using environment variable..")
		}
	}

	defer test.Client().Core().CoreV1().Namespaces().Delete(test.Ctx(), namespace.Name, metav1.DeleteOptions{})

	// Download standalone script used for running instructlab distributed training on RHOAI
	standaloneFilePath := "resources/standalone.py"
	cmd := exec.Command("curl", "-L", "-o", standaloneFilePath, standaloneScriptURL)
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
	test_sa, test_sa_exists := GetTestServiceAccount()
	var createdSA *corev1.ServiceAccount
	if !test_sa_exists {
		test.T().Logf("The service account name is not provided using environment variable..")
		createdSA = CreateServiceAccount(test, namespace.Name)
	} else {
		createdSA, err = test.Client().Core().CoreV1().ServiceAccounts(namespace.Name).Get(test.Ctx(), test_sa, metav1.GetOptions{})
		if err != nil {
			test.T().Skip("The service-account name provided using environment variable doesn't exists..")
			createdSA = CreateServiceAccountWithName(test, namespace.Name, test_sa)
		}
	}
	defer test.Client().Core().CoreV1().ServiceAccounts(namespace.Name).Delete(test.Ctx(), createdSA.Name, metav1.DeleteOptions{})

	// Create cluster role
	policyRules := []rbacv1.PolicyRule{
		{
			APIGroups: []string{""},
			Resources: []string{"pods/log"},
			Verbs: []string{
				"get", "list",
			},
		},
		{
			APIGroups: []string{"batch"},
			Resources: []string{"jobs"},
			Verbs: []string{
				"get", "list", "create", "watch",
			},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"pods"},
			Verbs: []string{
				"get", "list", "create", "watch",
			},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"secrets"},
			Verbs: []string{
				"get", "create",
			},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"configmaps"},
			Verbs: []string{
				"get", "create",
			},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"persistentvolumes", "persistentvolumeclaims"},
			Verbs: []string{
				"list", "create",
			},
		},
		{
			APIGroups: []string{"kubeflow.org"},
			Resources: []string{"pytorchjobs"},
			Verbs: []string{
				"get", "list", "create", "watch",
			},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"events"},
			Verbs: []string{
				"get", "list", "watch",
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

	judgeServingModelSecret := CreateJudgeServingModelSecret(test, namespace.Name)

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
					Image: rhelaiWorkbenchImage,
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
						{
							Name:  "JUDGE_SERVING_MODEL_SECRET",
							Value: judgeServingModelSecret.Name,
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
						"--judge-serving-model-secret", judgeServingModelSecret.Name,
						"--nproc-per-node", strconv.Itoa(numGpus),
						"--storage-class", ilabStorageClassName,
						"--sdg-object-store-secret", createdSecret.Name,
						// "--training-1-epoch-num", strconv.Itoa(1),
						// "--training-2-epoch-num", strconv.Itoa(1),
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

	// Wait until workbench pod status becomes succeeded - timeout in 3 hrs
	var workbenchPod *corev1.Pod
	test.Eventually(func() corev1.PodPhase {
		workbenchPod, err = test.Client().Core().CoreV1().Pods(namespace.Name).Get(test.Ctx(), createdPod.Name, metav1.GetOptions{})
		test.Expect(err).To(BeNil())
		return workbenchPod.Status.Phase
	}, 180*time.Minute, 2*time.Second).Should(Equal(corev1.PodSucceeded))

	// Clean up downloaded files
	err = os.Remove("resources/standalone.py")
	test.Expect(err).ToNot(HaveOccurred())
}

func CreateJudgeServingModelSecret(test Test, namespace string) *corev1.Secret {
	// judge model details like endpoint, api-key, model-name, ca certs, ...etc should be provided via k8s secret
	// we need the secret name so the standalone.py script can fetch the details from that secret.
	// Get Judge model server credentials using environment variables
	judgeServingModelApiKeyEnvVar := "JUDGE_API_KEY"
	judgeServingModelNameEnvVar := "JUDGE_NAME"
	judgeServingModelEndpointEnvVar := "JUDGE_ENDPOINT"
	judgeServingCaCertEnvVar := "JUDGE_CA_CERT"
	judgeServingCaCertCmKeyEnvVar := "JUDGE_CA_CERT_CM_KEY"
	judgeServingCaCertFromOpenShiftEnvVar := "JUDGE_CA_CERT_FROM_OPENSHIFT"
	judgeServingModelApiKey, judgeServingModelApiKeyExists := os.LookupEnv(judgeServingModelApiKeyEnvVar)
	judgeServingModelName, judgeServingModelNameExists := os.LookupEnv(judgeServingModelNameEnvVar)
	judgeServingModelEndpoint, judgeServingModelEndpointExists := os.LookupEnv(judgeServingModelEndpointEnvVar)
	judgeServingCaCertFromOpenShift, judgeServingCaCertFromOpenShiftExists := os.LookupEnv(judgeServingCaCertFromOpenShiftEnvVar)

	test.Expect(judgeServingModelApiKeyExists).To(BeTrue(), fmt.Sprintf("please provide judge serving model api key using env variable %s", judgeServingModelApiKeyEnvVar))
	test.Expect(judgeServingModelNameExists).To(BeTrue(), fmt.Sprintf("please provide judge serving model name using env variable %s", judgeServingModelNameEnvVar))
	test.Expect(judgeServingModelEndpointExists).To(BeTrue(), fmt.Sprintf("please provide judge serving model endpoint using env variable %s", judgeServingModelEndpointEnvVar))

	judgeServingDetails := map[string]string{
		judgeServingModelApiKeyEnvVar:   judgeServingModelApiKey,
		judgeServingModelEndpointEnvVar: judgeServingModelEndpoint,
		judgeServingModelNameEnvVar:     judgeServingModelName,
	}

	if judgeServingCaCertFromOpenShiftExists && judgeServingCaCertFromOpenShift == "true" {
		test.T().Logf("Using OpenShift CA as Judge CA certificate")
		judgeServingDetails[judgeServingCaCertEnvVar] = "kube-root-ca.crt"
		judgeServingDetails[judgeServingCaCertCmKeyEnvVar] = "ca.crt"
	} else {
		test.T().Logf("Env variable '%s' not defined or not set to `true`, Judge CA certificate ConfigMap is not provided", judgeServingCaCertFromOpenShiftEnvVar)
	}

	judgeServingModelSecret := CreateSecret(test, namespace, judgeServingDetails)
	return judgeServingModelSecret
}

func GetRhelaiWorkbenchImage() (string, bool) {
	data_key, exists := os.LookupEnv("RHELAI_WORKBENCH_IMAGE")
	return data_key, exists
}

func GetStorageBucketDataKey() (string, bool) {
	data_key, exists := os.LookupEnv("SDG_OBJECT_STORE_DATA_KEY")
	return data_key, exists
}

func GetStorageBucketVerifyTls() (string, bool) {
	data_key, exists := os.LookupEnv("SDG_OBJECT_STORE_VERIFY_TLS")
	return data_key, exists
}

// GetStorageClassName name of the storage class to use for testing, default is nfs-csi
func GetStorageClassName() (string, bool) {
	data_key, exists := os.LookupEnv("TEST_ILAB_STORAGE_CLASS_NAME")
	return data_key, exists
}

func GetTestNamespace() (string, bool) {
	data_key, exists := os.LookupEnv("TEST_NAMESPACE")
	return data_key, exists
}

func GetTestServiceAccount() (string, bool) {
	data_key, exists := os.LookupEnv("TEST_SERVICE_ACCOUNT")
	return data_key, exists
}

func GetStandaloneScriptURL() (string, bool) {
	data_key, exists := os.LookupEnv("STANDALONE_SCRIPT_URL")
	return data_key, exists
}

func CreateServiceAccountWithName(t Test, namespace string, name string) *corev1.ServiceAccount {
	t.T().Helper()

	serviceAccount := &corev1.ServiceAccount{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ServiceAccount",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
	serviceAccount, err := t.Client().Core().CoreV1().ServiceAccounts(namespace).Create(t.Ctx(), serviceAccount, metav1.CreateOptions{})
	t.Expect(err).NotTo(HaveOccurred())
	t.T().Logf("Created ServiceAccount %s/%s successfully", serviceAccount.Namespace, serviceAccount.Name)

	return serviceAccount
}

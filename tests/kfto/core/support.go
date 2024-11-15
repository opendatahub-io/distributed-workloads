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

package core

import (
	"embed"
	"fmt"
	"time"

	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
)

//go:embed *.json
var files embed.FS

func ReadFile(t Test, fileName string) []byte {
	t.T().Helper()
	file, err := files.ReadFile(fileName)
	t.Expect(err).NotTo(HaveOccurred())
	return file
}

func PyTorchJob(t Test, namespace, name string) func(g Gomega) *kftov1.PyTorchJob {
	return func(g Gomega) *kftov1.PyTorchJob {
		job, err := t.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		return job
	}
}

func PyTorchJobConditionRunning(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PyTorchJobCondition(job, kftov1.JobRunning)
}

func PyTorchJobConditionSucceeded(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PyTorchJobCondition(job, kftov1.JobSucceeded)
}

func PyTorchJobConditionSuspended(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PyTorchJobCondition(job, kftov1.JobSuspended)
}

func PyTorchJobConditionFailed(job *kftov1.PyTorchJob) corev1.ConditionStatus {
	return PyTorchJobCondition(job, kftov1.JobFailed)
}

func PyTorchJobCondition(job *kftov1.PyTorchJob, conditionType kftov1.JobConditionType) corev1.ConditionStatus {
	for _, condition := range job.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status
		}
	}
	return corev1.ConditionUnknown
}

func GetOrCreateTestNamespace(t Test) string {
	namespaceName, exists := GetTestNamespaceName()
	if exists {
		// Make sure the namespace really exists
		_, err := t.Client().Core().CoreV1().Namespaces().Get(t.Ctx(), namespaceName, metav1.GetOptions{})
		t.Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Namespace %s declared by env variable but not found", namespaceName))
	} else {
		t.T().Logf("Namespace name not specified, creating temporary namespace for test")
		namespaceName = t.NewTestNamespace().Name
		t.T().Logf("Created temporary Namespace '%s' successfully", namespaceName)
	}
	return namespaceName
}

func uploadToS3(test Test, namespace string, pvcName string, storedAssetsPath string) {
	defaultEndpoint, found := GetStorageBucketDefaultEndpoint()
	test.Expect(found).To(BeTrue(), "Storage bucket default endpoint needs to be specified for S3 upload")
	accessKeyId, found := GetStorageBucketAccessKeyId()
	test.Expect(found).To(BeTrue(), "Storage bucket access key id needs to be specified for S3 upload")
	secretKey, found := GetStorageBucketSecretKey()
	test.Expect(found).To(BeTrue(), "Storage bucket secret key needs to be specified for S3 upload")
	bucketName, found := GetStorageBucketName()
	test.Expect(found).To(BeTrue(), "Storage bucket name needs to be specified for S3 upload")
	bucketPath := GetStorageBucketModelPath()

	// Create Secret with AWS/Minio access credentials
	secretData := map[string]string{
		"S3ENDPOINT":  defaultEndpoint,
		"ACCESSKEYID": accessKeyId,
		"SECRETKEY":   secretKey,
	}
	secret := CreateSecret(test, namespace, secretData)
	defer test.Client().Core().CoreV1().Secrets(namespace).Delete(test.Ctx(), secret.Name, metav1.DeleteOptions{})

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "upload-",
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: Ptr(int32(0)),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:    "s3",
							Image:   GetMinioCliImage(),
							Command: []string{"/bin/sh", "-c"},
							Args:    []string{fmt.Sprintf("mc alias set mys3 $S3ENDPOINT $ACCESSKEYID $SECRETKEY; mc cp --recursive /mnt/%s mys3/%s/%s", storedAssetsPath, bucketName, bucketPath)},
							EnvFrom: []corev1.EnvFromSource{
								{
									SecretRef: &corev1.SecretEnvSource{
										LocalObjectReference: corev1.LocalObjectReference{Name: secret.Name},
									},
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "mounted-volume",
									MountPath: "/mnt",
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "mounted-volume",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: pvcName,
								},
							},
						},
					},
				},
			},
		},
	}
	job, err := test.Client().Core().BatchV1().Jobs(namespace).Create(test.Ctx(), job, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created Job %s/%s successfully", job.Namespace, job.Name)
	defer test.Client().Core().BatchV1().Jobs(namespace).Delete(test.Ctx(), job.Name, metav1.DeleteOptions{PropagationPolicy: Ptr(metav1.DeletePropagationBackground)})

	test.Eventually(Job(test, namespace, job.Name), 60*time.Minute).Should(
		Or(
			WithTransform(JobConditionCompleted, Equal(corev1.ConditionTrue)),
			WithTransform(JobConditionFailed, Equal(corev1.ConditionTrue)),
		),
	)
	test.Expect(GetJob(test, namespace, job.Name)).To(WithTransform(JobConditionCompleted, Equal(corev1.ConditionTrue)), "Job uploading content to S3 bucket failed")
}

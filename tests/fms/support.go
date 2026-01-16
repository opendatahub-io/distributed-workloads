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

package fms

import (
	"embed"
	"fmt"
	"time"

	. "github.com/onsi/gomega"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

//go:embed resources/*
var files embed.FS

func ReadFile(t Test, fileName string) []byte {
	t.T().Helper()
	file, err := files.ReadFile(fileName)
	t.Expect(err).NotTo(HaveOccurred())
	return file
}

func UploadToS3(test Test, namespace string, pvcName string, storedAssetsPath string) {
	defaultEndpoint, found := GetStorageBucketDefaultEndpoint()
	test.Expect(found).To(BeTrue(), "Storage bucket default endpoint needs to be specified for S3 upload")
	accessKeyId, found := GetStorageBucketAccessKeyId()
	test.Expect(found).To(BeTrue(), "Storage bucket access key id needs to be specified for S3 upload")
	secretKey, found := GetStorageBucketSecretKey()
	test.Expect(found).To(BeTrue(), "Storage bucket secret key needs to be specified for S3 upload")
	bucketName, found := GetStorageBucketUploadName()
	test.Expect(found).To(BeTrue(), "Storage bucket name needs to be specified for S3 upload")
	bucketPath := GetStorageBucketUploadModelPath()

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
							Args:    []string{fmt.Sprintf("mc alias set --insecure mys3 $S3ENDPOINT $ACCESSKEYID $SECRETKEY; mc cp --recursive --insecure /mnt/%s mys3/%s/%s", storedAssetsPath, bucketName, bucketPath)},
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

func DownloadFromS3(test Test, namespace string, pvcName string, storedAssetsPath string) {
	defaultEndpoint, found := GetStorageBucketDefaultEndpoint()
	test.Expect(found).To(BeTrue(), "Storage bucket default endpoint needs to be specified for download from S3")
	accessKeyId, found := GetStorageBucketAccessKeyId()
	test.Expect(found).To(BeTrue(), "Storage bucket access key id needs to be specified for download from S3")
	secretKey, found := GetStorageBucketSecretKey()
	test.Expect(found).To(BeTrue(), "Storage bucket secret key needs to be specified for download from S3")
	bucketName, found := GetStorageBucketDownloadName()
	test.Expect(found).To(BeTrue(), "Storage bucket name needs to be specified for download from S3")
	bucketPath := GetStorageBucketDownloadModelPath()

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
			GenerateName: "download-",
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
							Args:    []string{fmt.Sprintf("mc alias set --insecure mys3 $S3ENDPOINT $ACCESSKEYID $SECRETKEY; mc cp --recursive --insecure mys3/%s/%s /mnt/%s", bucketName, bucketPath, storedAssetsPath)},
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
	test.Expect(GetJob(test, namespace, job.Name)).To(WithTransform(JobConditionCompleted, Equal(corev1.ConditionTrue)), "Job downloading content from S3 bucket failed")
}

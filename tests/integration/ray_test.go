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

package integration

import (
	"encoding/base64"
	"net/http"
	"net/url"
	"testing"

	. "github.com/onsi/gomega"
	support "github.com/project-codeflare/codeflare-operator/test/support"
	rayv1alpha1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1alpha1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	routev1 "github.com/openshift/api/route/v1"
)

func TestRayCluster(t *testing.T) {
	test := support.With(t)
	test.T().Parallel()

	// This test is unstable. It seems that RayJob CR sometimes trigger 2 jobs in Ray, causing confusion in KubeRay operator.
	// Needs to be checked with newer KubeRay version. If still unstable then it needs to be reported.
	test.T().Skip("Requires https://github.com/opendatahub-io/distributed-workloads/issues/65")

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Create MNIST training script
	mnist := createMnistConfigMap(test, namespace.Name)

	// Create Ray cluster
	rayCluster := createRayCluster(test, namespace.Name, mnist)

	rayJob := &rayv1alpha1.RayJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rayv1alpha1.GroupVersion.String(),
			Kind:       "RayJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mnist",
			Namespace: namespace.Name,
		},
		Spec: rayv1alpha1.RayJobSpec{
			Entrypoint: "python /home/ray/jobs/mnist.py",
			RuntimeEnv: base64.StdEncoding.EncodeToString([]byte(`
{
  "pip": [
    "pytorch_lightning==1.5.10",
    "torchmetrics==0.9.1",
    "torchvision==0.12.0"
  ],
  "env_vars": {
  }
}
`)),
			ClusterSelector: map[string]string{
				support.RayJobDefaultClusterSelectorKey: rayCluster.Name,
			},
			ShutdownAfterJobFinishes: false,
		},
	}
	rayJob, err := test.Client().Ray().RayV1alpha1().RayJobs(namespace.Name).Create(test.Ctx(), rayJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created RayJob %s/%s successfully", rayJob.Namespace, rayJob.Name)

	// Retrieving the job logs once it has completed or timed out
	// Create a route to expose the Ray cluster API
	dashboardRoute := &routev1.Route{
		TypeMeta: metav1.TypeMeta{
			APIVersion: routev1.SchemeGroupVersion.String(),
			Kind:       "Route",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ray",
			Namespace: namespace.Name,
		},
		Spec: routev1.RouteSpec{
			To: routev1.RouteTargetReference{
				Name: "raycluster-head-svc",
			},
			Port: &routev1.RoutePort{
				TargetPort: intstr.FromString("dashboard"),
			},
		},
	}
	_, err = test.Client().Route().RouteV1().Routes(namespace.Name).Create(test.Ctx(), dashboardRoute, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created Route %s/%s successfully", dashboardRoute.Namespace, dashboardRoute.Name)

	test.T().Logf("Waiting for Route %s/%s to be available", dashboardRoute.Namespace, dashboardRoute.Name)
	test.Eventually(support.Route(test, dashboardRoute.Namespace, dashboardRoute.Name), support.TestTimeoutLong).
		Should(WithTransform(support.ConditionStatus(routev1.RouteAdmitted), Equal(corev1.ConditionTrue)))

	// Retrieve dashboard hostname
	dashboard, err := test.Client().Route().RouteV1().Routes(namespace.Name).Get(test.Ctx(), dashboardRoute.Name, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	dashboardHostname := dashboard.Status.Ingress[0].Host

	rayClient := support.NewRayClusterClient(url.URL{Scheme: "http", Host: dashboardHostname})
	defer support.WriteRayJobLogs(test, rayClient, rayJob.Namespace, rayJob.Name)

	test.T().Logf("Waiting for RayJob %s/%s to complete", rayJob.Namespace, rayJob.Name)
	test.Eventually(support.RayJob(test, rayJob.Namespace, rayJob.Name), support.TestTimeoutLong).
		Should(WithTransform(support.RayJobStatus, Satisfy(rayv1alpha1.IsJobTerminal)))

	// Assert the Ray job has completed successfully
	test.Expect(support.GetRayJob(test, rayJob.Namespace, rayJob.Name)).
		To(WithTransform(support.RayJobStatus, Equal(rayv1alpha1.JobStatusSucceeded)))
}

func TestRayJobSubmissionRest(t *testing.T) {
	test := support.With(t)
	test.T().Parallel()

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Create MNIST training script
	mnist := createMnistConfigMap(test, namespace.Name)

	// Create Ray cluster
	createRayCluster(test, namespace.Name, mnist)

	// Create a route to expose the Ray cluster API
	dashboardRoute := &routev1.Route{
		TypeMeta: metav1.TypeMeta{
			APIVersion: routev1.SchemeGroupVersion.String(),
			Kind:       "Route",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ray",
			Namespace: namespace.Name,
		},
		Spec: routev1.RouteSpec{
			To: routev1.RouteTargetReference{
				Name: "raycluster-head-svc",
			},
			Port: &routev1.RoutePort{
				TargetPort: intstr.FromString("dashboard"),
			},
		},
	}
	_, err := test.Client().Route().RouteV1().Routes(namespace.Name).Create(test.Ctx(), dashboardRoute, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created Route %s/%s successfully", dashboardRoute.Namespace, dashboardRoute.Name)

	test.T().Logf("Waiting for Route %s/%s to be available", dashboardRoute.Namespace, dashboardRoute.Name)
	test.Eventually(support.Route(test, dashboardRoute.Namespace, dashboardRoute.Name), support.TestTimeoutLong).
		Should(WithTransform(support.ConditionStatus(routev1.RouteAdmitted), Equal(corev1.ConditionTrue)))

	// Retrieve dashboard hostname
	dashboard, err := test.Client().Route().RouteV1().Routes(namespace.Name).Get(test.Ctx(), dashboardRoute.Name, metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	dashboardHostname := dashboard.Status.Ingress[0].Host

	// Wait for 200 reply from dashboard route
	test.Eventually(func() int {
		resp, _ := http.Get("http://" + dashboardHostname)
		return resp.StatusCode
	}, support.TestTimeoutLong).Should(Equal(200))

	rayClient := support.NewRayClusterClient(url.URL{Scheme: "http", Host: dashboardHostname})

	// Create Ray Job using REST API
	job := support.RayJobSetup{
		EntryPoint: "python /home/ray/jobs/mnist.py",
		RuntimeEnv: map[string]any{
			"pip": []string{
				"pytorch_lightning==1.5.10",
				"torchmetrics==0.9.1",
				"torchvision==0.12.0",
			},
		},
	}
	jobResponse, err := rayClient.CreateJob(&job)
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Ray Job %s submitted successfully", jobResponse.JobID)

	// Retrieving the job logs once it has completed or timed out
	defer support.WriteRayJobAPILogs(test, rayClient, jobResponse.JobID)

	test.T().Logf("Waiting for Job %s to finish", jobResponse.JobID)
	test.Eventually(support.RayJobAPIDetails(test, rayClient, jobResponse.JobID), support.TestTimeoutLong).
		Should(
			Or(
				WithTransform(support.GetRayJobAPIDetailsStatus, Equal("SUCCEEDED")),
				WithTransform(support.GetRayJobAPIDetailsStatus, Equal("STOPPED")),
				WithTransform(support.GetRayJobAPIDetailsStatus, Equal("FAILED")),
			))

	// Assert the job has completed successfully
	test.Expect(support.GetRayJobAPIDetails(test, rayClient, jobResponse.JobID)).
		To(WithTransform(support.GetRayJobAPIDetailsStatus, Equal("SUCCEEDED")))
}

func createMnistConfigMap(test support.Test, namespace string) (mnist *corev1.ConfigMap) {
	mnist = &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mnist",
			Namespace: namespace,
		},
		BinaryData: map[string][]byte{
			"mnist.py": ReadFile(test, "resources/mnist.py"),
		},
		Immutable: support.Ptr(true),
	}
	mnist, err := test.Client().Core().CoreV1().ConfigMaps(namespace).Create(test.Ctx(), mnist, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created ConfigMap %s/%s successfully", mnist.Namespace, mnist.Name)
	return
}

func createRayCluster(test support.Test, namespace string, mnist *corev1.ConfigMap) (rayCluster *rayv1alpha1.RayCluster) {
	// RayCluster, CR taken from https://github.com/project-codeflare/codeflare-operator/blob/main/test/e2e/mnist_rayjob_mcad_raycluster_test.go
	rayCluster = &rayv1alpha1.RayCluster{
		TypeMeta: metav1.TypeMeta{
			APIVersion: rayv1alpha1.GroupVersion.String(),
			Kind:       "RayCluster",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "raycluster",
			Namespace: namespace,
		},
		Spec: rayv1alpha1.RayClusterSpec{
			RayVersion: support.GetRayVersion(),
			HeadGroupSpec: rayv1alpha1.HeadGroupSpec{
				RayStartParams: map[string]string{
					"dashboard-host": "0.0.0.0",
				},
				Template: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:  "ray-head",
								Image: support.GetRayImage(),
								Ports: []corev1.ContainerPort{
									{
										ContainerPort: 6379,
										Name:          "gcs",
									},
									{
										ContainerPort: 8265,
										Name:          "dashboard",
									},
									{
										ContainerPort: 10001,
										Name:          "client",
									},
								},
								Lifecycle: &corev1.Lifecycle{
									PreStop: &corev1.LifecycleHandler{
										Exec: &corev1.ExecAction{
											Command: []string{"/bin/sh", "-c", "ray stop"},
										},
									},
								},
								Resources: corev1.ResourceRequirements{
									Requests: corev1.ResourceList{
										corev1.ResourceCPU:    resource.MustParse("250m"),
										corev1.ResourceMemory: resource.MustParse("512Mi"),
									},
									Limits: corev1.ResourceList{
										corev1.ResourceCPU:    resource.MustParse("1"),
										corev1.ResourceMemory: resource.MustParse("1G"),
									},
								},
								VolumeMounts: []corev1.VolumeMount{
									{
										Name:      "mnist",
										MountPath: "/home/ray/jobs",
									},
								},
							},
						},
						Volumes: []corev1.Volume{
							{
								Name: "mnist",
								VolumeSource: corev1.VolumeSource{
									ConfigMap: &corev1.ConfigMapVolumeSource{
										LocalObjectReference: corev1.LocalObjectReference{
											Name: mnist.Name,
										},
									},
								},
							},
						},
					},
				},
			},
			WorkerGroupSpecs: []rayv1alpha1.WorkerGroupSpec{
				{
					Replicas:       support.Ptr(int32(1)),
					MinReplicas:    support.Ptr(int32(1)),
					MaxReplicas:    support.Ptr(int32(2)),
					GroupName:      "small-group",
					RayStartParams: map[string]string{},
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{
								{
									Name:    "init-myservice",
									Image:   "busybox:1.28",
									Command: []string{"sh", "-c", "until nslookup $RAY_IP.$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace).svc.cluster.local; do echo waiting for myservice; sleep 2; done"},
								},
							},
							Containers: []corev1.Container{
								{
									Name:  "ray-worker",
									Image: support.GetRayImage(),
									Lifecycle: &corev1.Lifecycle{
										PreStop: &corev1.LifecycleHandler{
											Exec: &corev1.ExecAction{
												Command: []string{"/bin/sh", "-c", "ray stop"},
											},
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("250m"),
											corev1.ResourceMemory: resource.MustParse("256Mi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("1"),
											corev1.ResourceMemory: resource.MustParse("512Mi"),
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

	rayCluster, err := test.Client().Ray().RayV1alpha1().RayClusters(namespace).Create(test.Ctx(), rayCluster, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created RayCluster %s/%s successfully", rayCluster.Namespace, rayCluster.Name)

	test.T().Logf("Waiting for RayCluster %s/%s to complete", rayCluster.Namespace, rayCluster.Name)
	test.Eventually(support.RayCluster(test, rayCluster.Namespace, rayCluster.Name), support.TestTimeoutLong).
		Should(WithTransform(support.RayClusterState, Equal(rayv1alpha1.Ready)))
	return
}

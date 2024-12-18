package kfto

import (
	"testing"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPyTorchJobFailureWithCuda(t *testing.T) {
	runFailedPyTorchJobTest(t, GetCudaTrainingImage())
}

func TestPyTorchJobFailureWithROCm(t *testing.T) {
	runFailedPyTorchJobTest(t, GetROCmTrainingImage())
}

func runFailedPyTorchJobTest(t *testing.T, image string) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	// Create training PyTorch job
	tuningJob := createFailedPyTorchJob(test, namespace.Name, image)

	// Make sure the PyTorch job is failed
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionFailed, Equal(corev1.ConditionTrue)))
}

func createFailedPyTorchJob(test Test, namespace string, baseImage string) *kftov1.PyTorchJob {
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
					RestartPolicy: "Never",
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           baseImage,
									Command:         []string{"python", "-c", "raise Exception('Test failure')"},
									ImagePullPolicy: corev1.PullIfNotPresent,
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("1"),
											corev1.ResourceMemory: resource.MustParse("1Gi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("1"),
											corev1.ResourceMemory: resource.MustParse("1Gi"),
										},
									},
									SecurityContext: &corev1.SecurityContext{
										RunAsNonRoot:           Ptr(true),
										ReadOnlyRootFilesystem: Ptr(true),
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

	return tuningJob
}

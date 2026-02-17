package kfto

import (
	"testing"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

func TestPyTorchJobFailureWithCudaPyTorch241(t *testing.T) {
	Tags(t, Tier1)
	runFailedPyTorchJobTest(t, GetTrainingCudaPyTorch241Image())
}

func TestPyTorchJobFailureWithCudaPyTorch251(t *testing.T) {
	Tags(t, Tier1)
	runFailedPyTorchJobTest(t, GetTrainingCudaPyTorch251Image())
}

func TestPyTorchJobFailureWithROCmPyTorch241(t *testing.T) {
	Tags(t, Tier1)
	runFailedPyTorchJobTest(t, GetTrainingROCmPyTorch241Image())
}

func TestPyTorchJobFailureWithROCmPyTorch251(t *testing.T) {
	Tags(t, Tier1)
	runFailedPyTorchJobTest(t, GetTrainingROCmPyTorch251Image())
}

func runFailedPyTorchJobTest(t *testing.T, image string) {
	test := With(t)

	SetupKueue(test, initialKueueState, PyTorchJobFramework)

	// Create a namespace with Kueue labeled
	namespace := test.NewTestNamespace(WithKueueManaged())
	test.T().Logf("Created Kueue-managed namespace: %s", namespace.Name)

	// Create Kueue resources
	resourceFlavor := CreateKueueResourceFlavor(test, v1beta1.ResourceFlavorSpec{})
	defer test.Client().Kueue().KueueV1beta1().ResourceFlavors().Delete(test.Ctx(), resourceFlavor.Name, metav1.DeleteOptions{})
	cqSpec := v1beta1.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		ResourceGroups: []v1beta1.ResourceGroup{
			{
				CoveredResources: []corev1.ResourceName{corev1.ResourceName("cpu"), corev1.ResourceName("memory")},
				Flavors: []v1beta1.FlavorQuotas{
					{
						Name: v1beta1.ResourceFlavorReference(resourceFlavor.Name),
						Resources: []v1beta1.ResourceQuota{
							{
								Name:         corev1.ResourceCPU,
								NominalQuota: resource.MustParse("8"),
							},
							{
								Name:         corev1.ResourceMemory,
								NominalQuota: resource.MustParse("18Gi"),
							},
						},
					},
				},
			},
		},
	}

	clusterQueue := CreateKueueClusterQueue(test, cqSpec)
	defer test.Client().Kueue().KueueV1beta1().ClusterQueues().Delete(test.Ctx(), clusterQueue.Name, metav1.DeleteOptions{})
	localQueue := CreateKueueLocalQueue(test, namespace.Name, clusterQueue.Name, AsDefaultQueue)

	// Create training PyTorch job
	tuningJob := createFailedPyTorchJob(test, namespace.Name, image, localQueue)

	// Make sure the PyTorch job is failed
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionFailed, Equal(corev1.ConditionTrue)))
}

func createFailedPyTorchJob(test Test, namespace string, baseImage string, localQueue *v1beta1.LocalQueue) *kftov1.PyTorchJob {
	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kfto-sft-",
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": localQueue.Name,
			},
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

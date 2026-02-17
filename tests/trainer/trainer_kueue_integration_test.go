/*
Copyright 2025.

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

package trainer

import (
	"fmt"
	"os"
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kueuev1beta1 "sigs.k8s.io/kueue/apis/kueue/v1beta1"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

var initialKueueState string

func TestMain(m *testing.M) {
	// Capture initial Kueue state before running any tests
	initialKueueState = CaptureComponentState(DefaultDSCName, "kueue")
	fmt.Printf("Initial Kueue managementState: %s\n", initialKueueState)

	// Run all tests only if setup succeeded
	m.Run()

	// TearDown Kueue: Only set to Removed if it was not already Unmanaged before tests
	if initialKueueState != "Unmanaged" {
		if err := TearDownComponent(DefaultDSCName, "kueue"); err != nil {
			fmt.Printf("TearDown: Failed to set Kueue to Removed: %v\n", err)
		}
	} else {
		fmt.Println("TearDown: Skipping Kueue teardown as Initial Kueue managementState was Unmanaged in DataScienceCluster")
	}

	os.Exit(0)
}

func TestKueueDefaultLocalQueueLabelInjection(t *testing.T) {
	Tags(t, Sanity)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	//Create a namespace with Kueue label
	namespace := test.NewTestNamespace(WithKueueManaged()).Name
	test.T().Logf("Created Kueue-managed namespace: %s", namespace)

	//Verify default LocalQueue is auto-created after namespace creation
	test.T().Log("Verifying default LocalQueue is auto-created after namespace creation ...")
	test.Eventually(func(g Gomega) {
		lq, err := test.Client().Kueue().KueueV1beta1().LocalQueues(namespace).Get(
			test.Ctx(),
			"default",
			metav1.GetOptions{},
		)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(string(lq.Spec.ClusterQueue)).To(Equal("default"))
	}, TestTimeoutShort).Should(Succeed())
	test.T().Log("LocalQueue 'default' exists and points to ClusterQueue 'default'")
	test.T().Log("LocalQueue 'default' is created automatically triggered by kueue label in namespace")

	//Create a TrainJob without kueue label
	test.T().Log("Creating TrainJob without kueue.x-k8s.io/queue-name label...")
	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-unlabeled-trainjob-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: trainerutils.DefaultClusterTrainingRuntime,
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command: []string{"echo", "test"},
			},
		},
	}

	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "TrainJob should be created successfully")
	test.T().Logf("Created TrainJob: %s", createdTrainJob.Name)

	//Verify TrainJob got the default localqueue label injected by the mutating webhook
	test.T().Log("Verifying default localqueue label was injected...")
	queueLabel, exists := createdTrainJob.Labels["kueue.x-k8s.io/queue-name"]
	test.Expect(exists).To(BeTrue(), "TrainJob should have kueue.x-k8s.io/queue-name label")
	test.Expect(queueLabel).To(Equal("default"), "Local queue label should be 'default'")
	test.T().Logf("TrainJob has kueue label: kueue.x-k8s.io/queue-name=%s", queueLabel)

	//Verify a Workload is created with the default localqueue
	test.T().Log("Verifying Workload is created with default localqueue...")
	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutShort).Should(
		And(
			HaveLen(1),
			ContainElement(WithTransform(func(w *kueuev1beta1.Workload) string {
				return w.Spec.QueueName
			}, Equal("default"))),
		),
	)
	test.T().Log("Workload created with localqueueName: default")

	//Verify Workload is admitted
	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutMedium).Should(
		ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrue())),
	)
	workloads := GetKueueWorkloads(test, namespace)
	test.T().Logf("Workload '%s' is admitted", workloads[0].Name)

	//Verify TrainJob completes successfully
	test.Eventually(TrainJob(test, namespace, createdTrainJob.Name), TestTimeoutLong).
		Should(WithTransform(TrainJobConditionComplete, Equal(metav1.ConditionTrue)))
	test.T().Logf("TrainJob %s completed successfully", createdTrainJob.Name)

	test.T().Log("Default localqueue label injection and admission verified successfully !!!")
}

func TestKueueWorkloadPreemptionSuspendsTrainJob(t *testing.T) {
	Tags(t, Sanity)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	// Create a namespace with Kueue label
	namespace := test.NewTestNamespace(WithKueueManaged()).Name
	test.T().Logf("Created Kueue-managed namespace: %s", namespace)

	// Wait for default LocalQueue to be created
	test.T().Log("Waiting for default LocalQueue to be created...")
	test.Eventually(func(g Gomega) {
		_, err := test.Client().Kueue().KueueV1beta1().LocalQueues(namespace).Get(
			test.Ctx(),
			"default",
			metav1.GetOptions{},
		)
		g.Expect(err).NotTo(HaveOccurred())
	}, TestTimeoutShort).Should(Succeed())

	// Create a TrainJob with a sleep so user can preempt it
	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-preemption-trainjob-",
			Namespace:    namespace,
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: trainerutils.DefaultClusterTrainingRuntime,
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command: []string{"sleep", "120"},
			},
		},
	}

	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created TrainJob: %s", createdTrainJob.Name)

	// Wait for Workload to be created and admitted
	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutMedium).Should(
		And(
			HaveLen(1),
			ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrue())),
		),
	)
	workload := GetKueueWorkloads(test, namespace)[0]
	test.T().Logf("Workload '%s' is admitted", workload.Name)

	// Verify TrainJob is running
	test.Eventually(TrainJob(test, namespace, createdTrainJob.Name), TestTimeoutShort).Should(
		WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionFalse)),
	)
	test.T().Logf("TrainJob '%s' is running", createdTrainJob.Name)

	// Preempt the workload
	test.T().Logf("User is preempting workload '%s' now ...", workload.Name)
	workload.Spec.Active = Ptr(false)
	_, err = test.Client().Kueue().KueueV1beta1().Workloads(namespace).Update(
		test.Ctx(),
		workload,
		metav1.UpdateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Workload '%s' is preempted", workload.Name)

	// Verify Workload is evicted
	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutShort).Should(
		ContainElement(WithTransform(KueueWorkloadEvicted, BeTrue())),
	)
	test.T().Logf("Workload '%s' is evicted", workload.Name)

	// Verify TrainJob is suspended
	test.Eventually(TrainJob(test, namespace, createdTrainJob.Name), TestTimeoutMedium).Should(
		WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionTrue)),
	)
	test.T().Logf("TrainJob '%s' is suspended", createdTrainJob.Name)

	test.T().Logf("Workload preemption successfully suspended TrainJob '%s'!", createdTrainJob.Name)
}

func TestKueueWorkloadInadmissibleWithNonExistentLocalQueue(t *testing.T) {
	Tags(t, Sanity)
	test := With(t)
	SetupKueue(test, initialKueueState, TrainJobFramework)

	// Create a namespace with Kueue label
	namespace := test.NewTestNamespace(WithKueueManaged()).Name
	test.T().Logf("Created Kueue-managed namespace: %s", namespace)

	nonExistentQueue := "non-existent-queue"

	// Create a TrainJob referencing a LocalQueue that doesn't exist
	test.T().Logf("Creating TrainJob with non-existent LocalQueue...")

	trainJob := &trainerv1alpha1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-non-existent-localqueue-trainjob-",
			Namespace:    namespace,
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name": nonExistentQueue,
			},
		},
		Spec: trainerv1alpha1.TrainJobSpec{
			RuntimeRef: trainerv1alpha1.RuntimeRef{
				Name: trainerutils.DefaultClusterTrainingRuntime,
			},
			Trainer: &trainerv1alpha1.Trainer{
				Command: []string{"echo", "test"},
			},
		},
	}

	createdTrainJob, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Create(
		test.Ctx(),
		trainJob,
		metav1.CreateOptions{},
	)
	test.Expect(err).NotTo(HaveOccurred(), "TrainJob should be created")
	test.T().Logf("TrainJob created: %s", createdTrainJob.Name)

	// Verify Workload is created but marked as Inadmissible
	test.T().Log("Verifying Workload is Inadmissible due to non-existent LocalQueue...")
	var inadmissibleMsg string
	var workloadName string
	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutShort).Should(
		And(
			HaveLen(1),
			ContainElement(WithTransform(func(w *kueuev1beta1.Workload) bool {
				inadmissible, msg := KueueWorkloadInadmissible(w)
				inadmissibleMsg = msg
				workloadName = w.Name
				return inadmissible
			}, BeTrue())),
		),
	)
	test.T().Logf("Workload '%s' is Inadmissible as %s", workloadName, inadmissibleMsg)

	// Verify TrainJob is suspended
	test.Eventually(TrainJob(test, namespace, createdTrainJob.Name), TestTimeoutShort).Should(
		WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionTrue)),
	)
	test.T().Logf("TrainJob '%s' is suspended as expected", createdTrainJob.Name)

	test.T().Log("Non-existent LocalQueue causes Workload Inadmissible - verified successfully!")
}

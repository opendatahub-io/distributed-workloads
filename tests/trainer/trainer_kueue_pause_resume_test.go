/*
Copyright 2026.

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
	"encoding/json"
	"fmt"
	"testing"

	trainerv1alpha1 "github.com/kubeflow/trainer/v2/pkg/apis/trainer/v1alpha1"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	kueuev1beta2 "sigs.k8s.io/kueue/apis/kueue/v1beta2"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

// Covers Kueue #14836 and midstream Trainer #240: admission and suspension
// must preserve user runtime patches without treating serialization as mutation.
func TestKueueTrainJobPauseResumePreservesRuntimePatches(t *testing.T) {
	Tags(t, Tier1)
	test := With(t)
	job := newKueuePauseResumeJob(test)
	job.waitRunning(test)
	job.expectConfiguration(test)

	job.pause(test)
	job.expectConfiguration(test)
	job.setActive(test, true)
	job.waitRunning(test)
	job.expectConfiguration(test)
}

// RHOAIENG-88520: Trainer #240 retains trainer immutability, including when
// suspended. This deliberately does not assume the scaling support requested
// by RHOAIENG-88584 exists.
func TestKueueTrainJobRejectsNodeScaling(t *testing.T) {
	Tags(t, Tier2)
	test := With(t)
	job := newKueuePauseResumeJob(test)
	job.waitRunning(test)
	job.expectScalingRejected(test)

	job.pause(test)
	job.expectScalingRejected(test)
	job.setActive(test, true)
	job.waitRunning(test)
}

type kueuePauseResumeJob struct {
	namespace string
	name      string
	workload  string
}

func newKueuePauseResumeJob(test Test) kueuePauseResumeJob {
	test.T().Helper()
	SetupKueue(test, initialKueueState, TrainJobFramework)
	namespace := test.NewTestNamespace(WithKueueManaged()).Name
	flavor := CreateKueueResourceFlavor(test, kueuev1beta2.ResourceFlavorSpec{
		// A real scheduling directive forces Kueue to write its runtime patch.
		NodeLabels: map[string]string{"kubernetes.io/os": "linux"},
	})
	test.T().Cleanup(func() {
		test.Expect(test.Client().Kueue().KueueV1beta2().ResourceFlavors().Delete(
			test.Ctx(), flavor.Name, metav1.DeleteOptions{})).To(Succeed())
	})
	queue := CreateKueueClusterQueue(test, kueuev1beta2.ClusterQueueSpec{
		NamespaceSelector: &metav1.LabelSelector{},
		StopPolicy:        Ptr(kueuev1beta2.StopPolicy("Hold")),
		ResourceGroups: []kueuev1beta2.ResourceGroup{{
			CoveredResources: []corev1.ResourceName{corev1.ResourceCPU, corev1.ResourceMemory},
			Flavors: []kueuev1beta2.FlavorQuotas{{
				Name: kueuev1beta2.ResourceFlavorReference(flavor.Name),
				Resources: []kueuev1beta2.ResourceQuota{
					{Name: corev1.ResourceCPU, NominalQuota: resource.MustParse("2")},
					{Name: corev1.ResourceMemory, NominalQuota: resource.MustParse("2Gi")},
				},
			}},
		}},
	})
	test.T().Cleanup(func() {
		test.Expect(test.Client().Kueue().KueueV1beta2().ClusterQueues().Delete(
			test.Ctx(), queue.Name, metav1.DeleteOptions{})).To(Succeed())
	})
	localQueue := CreateKueueLocalQueue(test, namespace, queue.Name)

	// Use raw JSON to retain explicit omitempty zero values, as SDK clients do.
	// A typed Go create would remove readOnly:false and env.value:"", masking
	// the CEL regression before Kueue ever processes the object.
	var job unstructured.Unstructured
	test.Expect(json.Unmarshal([]byte(`{
		"apiVersion":"trainer.kubeflow.org/v1alpha1", "kind":"TrainJob",
		"spec":{
			"suspend":true,
			"trainer":{
				"numNodes":1, "command":["sleep","3600"],
				"env":[{"name":"REGRESSION_EMPTY_VALUE","value":""}],
				"resourcesPerNode":{"requests":{"cpu":"100m","memory":"128Mi"}}
			},
			"runtimePatches":[{
				"manager":"regression-test",
				"trainingRuntimeSpec":{"template":{"spec":{"replicatedJobs":[{
					"name":"node", "template":{"spec":{"template":{"spec":{
						"volumes":[{"name":"regression-data","emptyDir":{}}],
						"containers":[{"name":"node","volumeMounts":[{
							"name":"regression-data","mountPath":"/tmp/regression-data","readOnly":false
						}]}]
					}}}}
				}]}}}
			}]
		}
	}`), &job.Object)).To(Succeed())
	job.SetGenerateName("kueue-pause-resume-")
	job.SetNamespace(namespace)
	job.SetLabels(map[string]string{"kueue.x-k8s.io/queue-name": localQueue.Name})
	test.Expect(unstructured.SetNestedField(job.Object, trainerutils.DefaultClusterTrainingRuntimeCPU,
		"spec", "runtimeRef", "name")).To(Succeed())
	jobs := test.Client().Dynamic().Resource(trainerv1alpha1.SchemeGroupVersion.WithResource("trainjobs")).Namespace(namespace)
	created, err := jobs.Create(test.Ctx(), &job, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	stored, err := jobs.Get(test.Ctx(), created.GetName(), metav1.GetOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	// Match slices rather than indexing untrusted API data so malformed fixtures
	// fail with an assertion, not a panic.
	patches, found, err := unstructured.NestedSlice(stored.Object, "spec", "runtimePatches")
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(found).To(BeTrue())
	mountMatcher := ContainElement(And(
		HaveKeyWithValue("name", "regression-data"), HaveKeyWithValue("readOnly", false)))
	containerMatcher := ContainElement(And(
		HaveKeyWithValue("name", "node"), HaveKeyWithValue("volumeMounts", mountMatcher)))
	podTemplateMatcher := HaveKeyWithValue("spec", HaveKeyWithValue("containers", containerMatcher))
	jobTemplateMatcher := HaveKeyWithValue("spec", HaveKeyWithValue("template", podTemplateMatcher))
	replicatedJobMatcher := ContainElement(And(
		HaveKeyWithValue("name", "node"), HaveKeyWithValue("template", jobTemplateMatcher)))
	runtimeMatcher := HaveKeyWithValue("template", HaveKeyWithValue("spec",
		HaveKeyWithValue("replicatedJobs", replicatedJobMatcher)))
	test.Expect(patches).To(ContainElement(And(
		HaveKeyWithValue("manager", "regression-test"), HaveKeyWithValue("trainingRuntimeSpec", runtimeMatcher))),
		"the stored fixture must retain explicit readOnly:false before admission")
	env, found, err := unstructured.NestedSlice(stored.Object, "spec", "trainer", "env")
	test.Expect(err).NotTo(HaveOccurred())
	test.Expect(found).To(BeTrue())
	test.Expect(env).To(ContainElement(HaveKeyWithValue("value", "")))

	test.Eventually(KueueWorkloads(test, namespace), TestTimeoutMedium).Should(HaveLen(1))
	workload := GetKueueWorkloads(test, namespace)[0]
	test.Expect(KueueWorkloadAdmitted(workload)).To(BeFalse())
	_, err = test.Client().Kueue().KueueV1beta2().ClusterQueues().Patch(test.Ctx(), queue.Name,
		types.MergePatchType, []byte(`{"spec":{"stopPolicy":"None"}}`), metav1.PatchOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	return kueuePauseResumeJob{namespace: namespace, name: created.GetName(), workload: workload.Name}
}

func (job kueuePauseResumeJob) waitRunning(test Test) {
	test.T().Helper()
	test.Eventually(KueueWorkloads(test, job.namespace), TestTimeoutMedium).Should(
		ContainElement(WithTransform(KueueWorkloadAdmitted, BeTrue())))
	test.Eventually(TrainJob(test, job.namespace, job.name), TestTimeoutMedium).Should(
		WithTransform(func(j *trainerv1alpha1.TrainJob) bool {
			return j.Spec.Suspend != nil && !*j.Spec.Suspend
		}, BeTrue()))
	test.Eventually(Pods(test, job.namespace, job.podSelector()), TestTimeoutLong).Should(
		And(HaveLen(1), ContainElement(WithTransform(podPhase, Equal(corev1.PodRunning)))))
	test.Expect(SingleJobSet(test, job.namespace)(test).Spec.ReplicatedJobs).To(ContainElement(And(
		HaveField("Name", "node"),
		HaveField("Template.Spec.Template.Spec.NodeSelector", HaveKeyWithValue("kubernetes.io/os", "linux")),
	)))
}

func (job kueuePauseResumeJob) podSelector() metav1.ListOptions {
	return metav1.ListOptions{LabelSelector: "jobset.sigs.k8s.io/jobset-name=" + job.name}
}

func (job kueuePauseResumeJob) setActive(test Test, active bool) {
	test.T().Helper()
	_, err := test.Client().Kueue().KueueV1beta2().Workloads(job.namespace).Patch(test.Ctx(), job.workload,
		types.MergePatchType, []byte(fmt.Sprintf(`{"spec":{"active":%t}}`, active)), metav1.PatchOptions{})
	test.Expect(err).NotTo(HaveOccurred())
}

func (job kueuePauseResumeJob) pause(test Test) {
	test.T().Helper()
	job.setActive(test, false)
	test.Eventually(KueueWorkloads(test, job.namespace), TestTimeoutMedium).Should(
		ContainElement(WithTransform(KueueWorkloadEvicted, BeTrue())))
	test.Eventually(TrainJob(test, job.namespace, job.name), TestTimeoutMedium).Should(
		WithTransform(TrainJobConditionSuspended, Equal(metav1.ConditionTrue)))
	test.Eventually(Pods(test, job.namespace, job.podSelector()), TestTimeoutMedium).Should(BeEmpty())
}

func (job kueuePauseResumeJob) expectScalingRejected(test Test) {
	test.T().Helper()
	_, err := test.Client().Trainer().TrainerV1alpha1().TrainJobs(job.namespace).Patch(test.Ctx(), job.name,
		types.MergePatchType, []byte(`{"spec":{"trainer":{"numNodes":2}}}`), metav1.PatchOptions{})
	test.Expect(err).To(HaveOccurred())
	test.Expect(apierrors.IsForbidden(err)).To(BeTrue(), "expected Trainer webhook rejection, got: %v", err)
	test.Expect(err.Error()).To(And(ContainSubstring("spec.trainer"), ContainSubstring("field is immutable")))
	current := TrainJob(test, job.namespace, job.name)(test)
	test.Expect(current.Spec.Trainer).NotTo(BeNil())
	test.Expect(current.Spec.Trainer.NumNodes).To(Equal(Ptr(int32(1))))
}

func (job kueuePauseResumeJob) expectConfiguration(test Test) {
	test.T().Helper()
	current := TrainJob(test, job.namespace, job.name)(test)
	test.Expect(current.Spec.Trainer).NotTo(BeNil())
	test.Expect(current.Spec.Trainer.NumNodes).To(Equal(Ptr(int32(1))))
	test.Expect(current.Spec.Trainer.Env).To(ContainElement(corev1.EnvVar{Name: "REGRESSION_EMPTY_VALUE"}))
	test.Expect(current.Spec.RuntimePatches).To(ContainElement(And(
		HaveField("Manager", "regression-test"),
		HaveField("TrainingRuntimeSpec.Template.Spec.ReplicatedJobs", ContainElement(And(
			HaveField("Name", "node"),
			HaveField("Template.Spec.Template.Spec.Volumes", ContainElement(corev1.Volume{
				Name: "regression-data", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
			})),
			HaveField("Template.Spec.Template.Spec.Containers", ContainElement(And(
				HaveField("Name", "node"),
				HaveField("VolumeMounts", ContainElement(corev1.VolumeMount{
					Name: "regression-data", MountPath: "/tmp/regression-data",
				})),
			))),
		))),
	)))
	jobSet := SingleJobSet(test, job.namespace)(test)
	test.Expect(jobSet.Spec.ReplicatedJobs).To(ContainElement(And(
		HaveField("Name", "node"),
		HaveField("Template.Spec.Template.Spec.Volumes", ContainElement(HaveField("Name", "regression-data"))),
		HaveField("Template.Spec.Template.Spec.Containers", ContainElement(And(
			HaveField("Name", "node"),
			HaveField("VolumeMounts", ContainElement(HaveField("MountPath", "/tmp/regression-data"))),
		))),
	)))
}

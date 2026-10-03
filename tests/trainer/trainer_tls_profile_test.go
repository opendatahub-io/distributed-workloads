package trainer

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	. "github.com/onsi/gomega"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"

	. "github.com/opendatahub-io/distributed-workloads/tests/common"
	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
	trainerutils "github.com/opendatahub-io/distributed-workloads/tests/trainer/utils"
)

const (
	tlsProfileLeaseName         = "trainer-tls-profile-e2e"
	tlsProfileLeaseNamespace    = "openshift-config"
	tlsProfileTransitionTimeout = 3 * time.Minute
	tlsProfileLeaseDuration     = 15 * time.Minute
)

var (
	openShiftAPIServerGVR = schema.GroupVersionResource{
		Group: "config.openshift.io", Version: "v1", Resource: "apiservers",
	}
)

func TestTrainerTLSProfileWatcher(t *testing.T) {
	Tags(t, Tier3)
	test := With(t)
	apiServer, err := test.Client().Dynamic().Resource(openShiftAPIServerGVR).Get(
		test.Ctx(), "cluster", metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		t.Skip("OpenShift APIServer is unavailable; TLS profile watcher is not applicable")
	}
	test.Expect(err).NotTo(HaveOccurred())
	applicationsNamespace, err := GetApplicationsNamespace(test)
	test.Expect(err).NotTo(HaveOccurred())
	tlsProfileLock := acquireTLSProfileLock(test)
	test.T().Cleanup(func() { tlsProfileLock.Release(test) })

	originalProfile, found, err := unstructured.NestedFieldCopy(apiServer.Object, "spec", "tlsSecurityProfile")
	test.Expect(err).NotTo(HaveOccurred())
	if !found {
		t.Skip("OpenShift APIServer has no tlsSecurityProfile to switch")
	}
	originalProfileMap, ok := originalProfile.(map[string]interface{})
	test.Expect(ok).To(BeTrue())
	originalType, found, err := unstructured.NestedString(originalProfileMap, "type")
	test.Expect(err).NotTo(HaveOccurred())
	if !found || originalType == "" {
		t.Skip("OpenShift APIServer tlsSecurityProfile has no type")
	}
	newProfile := "Intermediate"
	if originalType == newProfile {
		newProfile = "Old"
	}
	newProfileSpec := tlsSecurityProfile(newProfile)
	test.T().Logf("TLS profile watcher test will transition APIServer profile %s -> %s", originalType, newProfile)

	restoreTLSProfile := func() {
		if err := tlsProfileLock.EnsureHeld(context.Background()); err != nil {
			test.T().Errorf("cannot restore APIServer TLS profile after losing test lock: %v", err)
			return
		}
		test.T().Logf("Restoring APIServer TLS profile to %s", originalType)
		restoreCtx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		resource := test.Client().Dynamic().Resource(openShiftAPIServerGVR)
		restoreDeployment := trainerutils.GetTrainerControllerDeployment(test, applicationsNamespace)
		restoreBeforePod := trainerutils.TrainerControllerPod(test, applicationsNamespace, restoreDeployment, tlsProfileTransitionTimeout)
		restoreBeforePodUIDs := trainerControllerPodUIDs(test, applicationsNamespace, restoreDeployment)
		restoreBeforeRestartCount := trainerControllerRestartCount(restoreBeforePod)
		var restoreProfileUpdateStartedAt time.Time
		var lastErr error
		restoreErr := wait.PollUntilContextTimeout(
			restoreCtx, 5*time.Second, 10*time.Minute, true,
			func(ctx context.Context) (bool, error) {
				current, getErr := resource.Get(ctx, "cluster", metav1.GetOptions{})
				if getErr != nil {
					lastErr = getErr
					return false, nil
				}
				if setErr := unstructured.SetNestedField(current.Object, originalProfileMap, "spec", "tlsSecurityProfile"); setErr != nil {
					return false, setErr
				}
				restoreProfileUpdateStartedAt = time.Now()
				_, updateErr := resource.Update(ctx, current, metav1.UpdateOptions{})
				if updateErr != nil {
					lastErr = updateErr
					return false, nil
				}
				return true, nil
			},
		)
		if restoreErr != nil {
			if lastErr != nil {
				test.T().Errorf("failed to restore OpenShift APIServer TLS profile: %v", lastErr)
			} else {
				test.T().Errorf("failed to restore OpenShift APIServer TLS profile: %v", restoreErr)
			}
		} else {
			test.T().Logf("Restored APIServer TLS profile to %s", originalType)
			waitForTrainerControllerRestart(test, applicationsNamespace, restoreDeployment, restoreBeforePod, restoreBeforePodUIDs, restoreBeforeRestartCount, restoreProfileUpdateStartedAt)
			recoveryErr := wait.PollUntilContextTimeout(
				restoreCtx, 5*time.Second, 10*time.Minute, true,
				func(ctx context.Context) (bool, error) {
					if err := tlsProfileLock.EnsureHeld(ctx); err != nil {
						return false, err
					}
					current, getErr := resource.Get(ctx, "cluster", metav1.GetOptions{})
					if getErr != nil {
						return false, nil
					}
					currentType, _, typeErr := unstructured.NestedString(current.Object, "spec", "tlsSecurityProfile", "type")
					if typeErr != nil || currentType != originalType {
						return false, nil
					}
					deployment := trainerutils.GetTrainerControllerDeployment(test, applicationsNamespace)
					return trainerControllerReady(test, applicationsNamespace, deployment, ctx)
				},
			)
			if recoveryErr != nil {
				test.Expect(recoveryErr).NotTo(HaveOccurred(), "failed waiting for Trainer recovery after restoring TLS profile")
			} else {
				test.T().Logf("Trainer recovered after restoring APIServer TLS profile")
			}
		}
	}

	deployment := trainerutils.GetTrainerControllerDeployment(test, applicationsNamespace)
	beforePod := trainerutils.TrainerControllerPod(test, applicationsNamespace, deployment, tlsProfileTransitionTimeout)
	beforePodUIDs := trainerControllerPodUIDs(test, applicationsNamespace, deployment)
	beforeRestartCount := trainerControllerRestartCount(beforePod)
	test.T().Logf("Updating APIServer TLS profile; Trainer pod=%s restartCount=%d", beforePod.Name, beforeRestartCount)
	resource := test.Client().Dynamic().Resource(openShiftAPIServerGVR)
	var profileUpdateStartedAt time.Time
	var lastErr error
	updateErr := wait.PollUntilContextTimeout(
		test.Ctx(), 5*time.Second, tlsProfileTransitionTimeout, true,
		func(ctx context.Context) (bool, error) {
			if err := tlsProfileLock.EnsureHeld(ctx); err != nil {
				return false, err
			}
			current, getErr := resource.Get(ctx, "cluster", metav1.GetOptions{})
			if getErr != nil {
				lastErr = getErr
				return false, nil
			}
			if setErr := unstructured.SetNestedField(current.Object, newProfileSpec, "spec", "tlsSecurityProfile"); setErr != nil {
				return false, setErr
			}
			profileUpdateStartedAt = time.Now()
			_, updateErr := resource.Update(ctx, current, metav1.UpdateOptions{})
			if updateErr != nil {
				lastErr = updateErr
				return false, nil
			}
			return true, nil
		},
	)
	if updateErr != nil {
		if lastErr != nil {
			test.T().Fatalf("failed to update OpenShift APIServer TLS profile: %v", lastErr)
		}
		test.Expect(updateErr).NotTo(HaveOccurred())
	}
	test.T().Cleanup(restoreTLSProfile)
	test.T().Logf("APIServer TLS profile updated to %s; waiting for Trainer restart", newProfile)

	waitForTrainerControllerRestart(test, applicationsNamespace, deployment, beforePod, beforePodUIDs, beforeRestartCount, profileUpdateStartedAt)

	// The controller should remain healthy after the profile transition.
	test.Eventually(func(g Gomega, ctx context.Context) {
		updated := trainerutils.GetTrainerControllerDeployment(test, applicationsNamespace)
		g.Expect(deploymentAvailable(updated)).To(BeTrue())
	}, tlsProfileTransitionTimeout, 5*time.Second).WithContext(test.Ctx()).Should(Succeed())
	test.T().Logf("Trainer controller recovered after TLS profile transition")
}

func tlsSecurityProfile(profileType string) map[string]interface{} {
	profile := map[string]interface{}{"type": profileType}
	if profileType == "Old" {
		profile["old"] = map[string]interface{}{}
	} else {
		profile["intermediate"] = map[string]interface{}{}
	}
	return profile
}

func trainerControllerPodUIDs(test Test, namespace string, deployment *unstructured.Unstructured) map[string]struct{} {
	selector := trainerutils.TrainerControllerSelector(test, deployment)
	pods := GetPods(test, namespace, metav1.ListOptions{
		LabelSelector: selector,
	})
	ids := make(map[string]struct{}, len(pods))
	for _, pod := range pods {
		ids[string(pod.UID)] = struct{}{}
	}
	return ids
}

func waitForTrainerControllerRestart(test Test, namespace string, deployment *unstructured.Unstructured, beforePod *corev1.Pod, beforePodUIDs map[string]struct{}, beforeRestartCount int32, profileUpdatedAt time.Time) {
	selector := trainerutils.TrainerControllerSelector(test, deployment)
	test.Eventually(func(g Gomega, ctx context.Context) {
		pods, listErr := test.Client().Core().CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: selector,
		})
		g.Expect(listErr).NotTo(HaveOccurred())
		for i := range pods.Items {
			pod := &pods.Items[i]
			if _, existed := beforePodUIDs[string(pod.UID)]; !existed && trainerutils.PodReady(pod) && pod.CreationTimestamp.Time.After(profileUpdatedAt) {
				test.T().Logf("Trainer pod was replaced after TLS profile change: %s -> %s", beforePod.Name, pod.Name)
				return
			}
			if pod.UID == beforePod.UID && trainerutils.PodReady(pod) && trainerControllerRestartCount(pod) > beforeRestartCount {
				test.T().Logf("Trainer container restarted in pod %s: restartCount %d -> %d", pod.Name, beforeRestartCount, trainerControllerRestartCount(pod))
				return
			}
		}
		g.Expect(false).To(BeTrue(), "Trainer controller did not restart after TLS profile change")
	}, tlsProfileTransitionTimeout, 5*time.Second).WithContext(test.Ctx()).Should(Succeed())
}

func trainerControllerRestartCount(pod *corev1.Pod) int32 {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == "manager" {
			return status.RestartCount
		}
	}
	return 0
}

func trainerControllerReady(test Test, namespace string, deployment *unstructured.Unstructured, ctx context.Context) (bool, error) {
	selector := trainerutils.TrainerControllerSelector(test, deployment)
	pods, err := test.Client().Core().CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector,
	})
	if err != nil {
		return false, err
	}
	for i := range pods.Items {
		if pods.Items[i].Status.Phase == corev1.PodRunning && trainerutils.PodReady(&pods.Items[i]) {
			return true, nil
		}
	}
	return false, nil
}

type tlsProfileLock struct {
	leases interface {
		Get(context.Context, string, metav1.GetOptions) (*coordinationv1.Lease, error)
		Update(context.Context, *coordinationv1.Lease, metav1.UpdateOptions) (*coordinationv1.Lease, error)
		Delete(context.Context, string, metav1.DeleteOptions) error
		Create(context.Context, *coordinationv1.Lease, metav1.CreateOptions) (*coordinationv1.Lease, error)
	}
	identity string
	stop     chan struct{}
	done     chan struct{}
	lost     chan struct{}
	lostOnce sync.Once
}

func (lock *tlsProfileLock) markLost() {
	lock.lostOnce.Do(func() { close(lock.lost) })
}

func (lock *tlsProfileLock) isLost() bool {
	select {
	case <-lock.lost:
		return true
	default:
		return false
	}
}

func (lock *tlsProfileLock) EnsureHeld(ctx context.Context) error {
	select {
	case <-lock.lost:
		return fmt.Errorf("TLS profile test lock was lost")
	default:
	}
	lease, err := lock.leases.Get(ctx, tlsProfileLeaseName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != lock.identity {
		lock.markLost()
		return fmt.Errorf("TLS profile test lock is held by another runner")
	}
	return nil
}

func (lock *tlsProfileLock) renew(ctx context.Context) error {
	lease, err := lock.leases.Get(ctx, tlsProfileLeaseName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != lock.identity {
		lock.markLost()
		return fmt.Errorf("TLS profile test lock is held by another runner")
	}
	now := metav1.NewMicroTime(time.Now())
	lease.Spec.RenewTime = &now
	_, err = lock.leases.Update(ctx, lease, metav1.UpdateOptions{})
	return err
}

func (lock *tlsProfileLock) Release(test Test) {
	close(lock.stop)
	<-lock.done
	if err := lock.EnsureHeld(context.Background()); err != nil {
		test.T().Logf("TLS profile test lock was not held during release: %v", err)
		return
	}
	if err := lock.leases.Delete(context.Background(), tlsProfileLeaseName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		test.T().Logf("failed to release TLS profile test lock: %v", err)
	}
}

func acquireTLSProfileLock(test Test) *tlsProfileLock {
	hostname, err := os.Hostname()
	test.Expect(err).NotTo(HaveOccurred())
	identity := fmt.Sprintf("%s-%d", hostname, os.Getpid())
	leases := test.Client().Core().CoordinationV1().Leases(tlsProfileLeaseNamespace)
	durationSeconds := int32(tlsProfileLeaseDuration / time.Second)

	acquireErr := wait.PollUntilContextTimeout(
		test.Ctx(), 5*time.Second, 10*time.Minute, true,
		func(ctx context.Context) (bool, error) {
			lease, getErr := leases.Get(ctx, tlsProfileLeaseName, metav1.GetOptions{})
			if apierrors.IsNotFound(getErr) {
				now := metav1.NewMicroTime(time.Now())
				_, createErr := leases.Create(ctx, &coordinationv1.Lease{
					ObjectMeta: metav1.ObjectMeta{Name: tlsProfileLeaseName},
					Spec: coordinationv1.LeaseSpec{
						HolderIdentity:       &identity,
						LeaseDurationSeconds: &durationSeconds,
						AcquireTime:          &now,
						RenewTime:            &now,
					},
				}, metav1.CreateOptions{})
				if apierrors.IsAlreadyExists(createErr) {
					return false, nil
				}
				return createErr == nil, createErr
			}
			if getErr != nil {
				return false, getErr
			}
			if lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != identity &&
				(lease.Spec.RenewTime == nil || lease.Spec.RenewTime.Time.Add(tlsProfileLeaseDuration).After(time.Now())) {
				return false, nil
			}
			now := metav1.NewMicroTime(time.Now())
			lease.Spec.HolderIdentity = &identity
			lease.Spec.LeaseDurationSeconds = &durationSeconds
			lease.Spec.AcquireTime = &now
			lease.Spec.RenewTime = &now
			_, updateErr := leases.Update(ctx, lease, metav1.UpdateOptions{})
			if apierrors.IsConflict(updateErr) {
				return false, nil
			}
			return updateErr == nil, updateErr
		},
	)
	test.Expect(acquireErr).NotTo(HaveOccurred())
	test.T().Logf("Acquired TLS profile test lock %s", identity)

	lock := &tlsProfileLock{
		leases:   leases,
		identity: identity,
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
		lost:     make(chan struct{}, 1),
	}
	go func() {
		defer close(lock.done)
		ticker := time.NewTicker(tlsProfileLeaseDuration / 3)
		defer ticker.Stop()
		for {
			select {
			case <-lock.stop:
				return
			case <-ticker.C:
				if renewErr := lock.renew(context.Background()); renewErr != nil {
					if lock.isLost() || apierrors.IsConflict(renewErr) || apierrors.IsNotFound(renewErr) || apierrors.IsForbidden(renewErr) {
						lock.markLost()
						test.T().Logf("lost TLS profile test lock while renewing: %v", renewErr)
						return
					}
					test.T().Logf("transient TLS profile test lock renewal failure; will retry: %v", renewErr)
				}
			}
		}
	}()
	return lock
}

func deploymentAvailable(deployment *unstructured.Unstructured) bool {
	conditions, _, _ := unstructured.NestedSlice(deployment.Object, "status", "conditions")
	for _, rawCondition := range conditions {
		condition := rawCondition.(map[string]interface{})
		if condition["type"] == "Available" && condition["status"] == "True" {
			return true
		}
	}
	return false
}

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

package support

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"reflect"
	"time"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func CreateConfigMap(t Test, namespace string, content map[string][]byte) *corev1.ConfigMap {
	t.T().Helper()

	configMap := &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "config-",
			Namespace:    namespace,
		},
		BinaryData: content,
		Immutable:  Ptr(true),
	}

	configMap, err := t.Client().Core().CoreV1().ConfigMaps(namespace).Create(t.Ctx(), configMap, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created ConfigMap %s/%s successfully", configMap.Namespace, configMap.Name)

	return configMap
}

func CreateSecret(t Test, namespace string, content map[string]string) *corev1.Secret {
	t.T().Helper()

	secret := &corev1.Secret{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "Secret",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "secret-",
			Namespace:    namespace,
		},
		StringData: content,
		Immutable:  Ptr(true),
	}

	secret, err := t.Client().Core().CoreV1().Secrets(namespace).Create(t.Ctx(), secret, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created Secret %s/%s successfully", secret.Namespace, secret.Name)

	return secret
}

func CreateSecretBinary(t Test, namespace string, content map[string][]byte) *corev1.Secret {
	t.T().Helper()

	secret := &corev1.Secret{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "Secret",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "secret-",
			Namespace:    namespace,
		},
		Data:      content,
		Immutable: Ptr(true),
	}

	secret, err := t.Client().Core().CoreV1().Secrets(namespace).Create(t.Ctx(), secret, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created binary Secret %s/%s successfully", secret.Namespace, secret.Name)

	return secret
}

func Raw(t Test, obj runtime.Object) runtime.RawExtension {
	t.T().Helper()
	data, err := json.Marshal(obj)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return runtime.RawExtension{
		Raw: data,
	}
}

func Pods(t Test, namespace string, options metav1.ListOptions) func(g gomega.Gomega) []corev1.Pod {
	return func(g gomega.Gomega) []corev1.Pod {
		pods, err := t.Client().Core().CoreV1().Pods(namespace).List(t.Ctx(), options)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return pods.Items
	}
}

func GetPods(t Test, namespace string, options metav1.ListOptions) []corev1.Pod {
	t.T().Helper()
	return Pods(t, namespace, options)(t)
}

func PodLog(t Test, namespace, name string, options corev1.PodLogOptions) func(g gomega.Gomega) string {
	return func(g gomega.Gomega) string {
		stream, err := t.Client().Core().CoreV1().Pods(namespace).GetLogs(name, &options).Stream(t.Ctx())
		g.Expect(err).NotTo(gomega.HaveOccurred())

		defer func() {
			g.Expect(stream.Close()).To(gomega.Succeed())
		}()

		bytes, err := io.ReadAll(stream)
		g.Expect(err).NotTo(gomega.HaveOccurred())

		return string(bytes)
	}
}

func GetPodLog(t Test, namespace, name string, options corev1.PodLogOptions) string {
	t.T().Helper()
	return PodLog(t, namespace, name, options)(t)
}

func storeAllPodLogs(t Test, namespace *corev1.Namespace) {
	t.T().Helper()

	pods, err := t.Client().Core().CoreV1().Pods(namespace.Name).List(t.Ctx(), metav1.ListOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())

	for _, pod := range pods.Items {
		for _, container := range pod.Spec.Containers {
			t.T().Logf("Retrieving Pod Container %s/%s/%s logs", pod.Namespace, pod.Name, container.Name)
			storeContainerLog(t, namespace, pod.Name, container.Name)
		}
	}
}

func storeContainerLog(t Test, namespace *corev1.Namespace, podName, containerName string) {
	t.T().Helper()

	options := corev1.PodLogOptions{Container: containerName}
	stream, err := t.Client().Core().CoreV1().Pods(namespace.Name).GetLogs(podName, &options).Stream(t.Ctx())
	if err != nil {
		t.T().Logf("Failed to retrieve logs for Pod Container %s/%s/%s, logs cannot be stored", namespace.Name, podName, containerName)
		return
	}

	defer func() {
		t.Expect(stream.Close()).To(gomega.Succeed())
	}()

	bytes, err := io.ReadAll(stream)
	t.Expect(err).NotTo(gomega.HaveOccurred())

	containerLogFileName := "pod-" + podName + "-" + containerName
	WriteToOutputDir(t, containerLogFileName, Log, bytes)
}

const maxCapturedLogBytes int64 = 10 << 20 // 10 MiB per container snapshot

func startPeriodicPodLogCapture(t Test, namespace *corev1.Namespace) context.CancelFunc {
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	client := t.Client().Core().CoreV1()

	writeFile := func(filePath string, data []byte) {
		if err := os.WriteFile(filePath, data, 0o600); err != nil {
			t.T().Logf("periodic pod capture: failed writing %s: %v", filePath, err)
		}
	}

	fetchLog := func(outputDir, podName, containerName string, previous bool) {
		tailLines := int64(10000)
		options := corev1.PodLogOptions{Container: containerName, Previous: previous, TailLines: &tailLines}
		stream, err := client.Pods(namespace.Name).GetLogs(podName, &options).Stream(ctx)
		if err != nil {
			return
		}
		defer func() { _ = stream.Close() }()

		data, err := io.ReadAll(io.LimitReader(stream, maxCapturedLogBytes))
		if err != nil || len(data) == 0 {
			return
		}

		suffix := ""
		if previous {
			suffix = "-previous"
		}
		writeFile(path.Join(outputDir, "pod-"+podName+"-"+containerName+suffix+".log"), data)
	}

	go func() {
		defer close(done)
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				outputDir := t.OutputDir()

				pods, err := client.Pods(namespace.Name).List(ctx, metav1.ListOptions{})
				if err != nil {
					continue
				}

				for _, pod := range pods.Items {
					for _, container := range pod.Spec.InitContainers {
						fetchLog(outputDir, pod.Name, container.Name, false)
						fetchLog(outputDir, pod.Name, container.Name, true)
					}

					for _, container := range pod.Spec.Containers {
						fetchLog(outputDir, pod.Name, container.Name, false)
						fetchLog(outputDir, pod.Name, container.Name, true)
					}

					status := map[string]any{
						"phase":             pod.Status.Phase,
						"containerStatuses": pod.Status.ContainerStatuses,
						"initStatuses":      pod.Status.InitContainerStatuses,
					}
					if data, err := json.MarshalIndent(status, "", "  "); err == nil {
						writeFile(path.Join(outputDir, "pod-"+pod.Name+"-status.log"), data)
					}
				}
			}
		}
	}()

	return func() {
		cancel()
		<-done
	}
}

func CreateServiceAccount(t Test, namespace string) *corev1.ServiceAccount {
	t.T().Helper()

	serviceAccount := &corev1.ServiceAccount{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "ServiceAccount",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "sa-",
			Namespace:    namespace,
		},
	}
	serviceAccount, err := t.Client().Core().CoreV1().ServiceAccounts(namespace).Create(t.Ctx(), serviceAccount, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created ServiceAccount %s/%s successfully", serviceAccount.Namespace, serviceAccount.Name)

	return serviceAccount
}

func ServiceAccount(t Test, namespace, name string) func(g gomega.Gomega) *corev1.ServiceAccount {
	return func(g gomega.Gomega) *corev1.ServiceAccount {
		sa, err := t.Client().Core().CoreV1().ServiceAccounts(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return sa
	}
}

func GetServiceAccount(t Test, namespace, name string) *corev1.ServiceAccount {
	t.T().Helper()
	return ServiceAccount(t, namespace, name)(t)
}

func ServiceAccounts(t Test, namespace string) func(g gomega.Gomega) []*corev1.ServiceAccount {
	return func(g gomega.Gomega) []*corev1.ServiceAccount {
		sas, err := t.Client().Core().CoreV1().ServiceAccounts(namespace).List(t.Ctx(), metav1.ListOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())

		sasp := []*corev1.ServiceAccount{}
		for _, v := range sas.Items {
			sasp = append(sasp, &v)
		}

		return sasp
	}
}

func GetServiceAccounts(t Test, namespace string) []*corev1.ServiceAccount {
	t.T().Helper()
	return ServiceAccounts(t, namespace)(t)
}

type PVCOption Option[*corev1.PersistentVolumeClaim]

func StorageClassName(name string) PVCOption {
	return ErrorOption[*corev1.PersistentVolumeClaim](func(pvc *corev1.PersistentVolumeClaim) error {
		pvc.Spec.StorageClassName = &name
		return nil
	})
}

func AccessModes(accessModes ...corev1.PersistentVolumeAccessMode) PVCOption {
	return ErrorOption[*corev1.PersistentVolumeClaim](func(pvc *corev1.PersistentVolumeClaim) error {
		pvc.Spec.AccessModes = accessModes
		return nil
	})
}

func CreatePersistentVolumeClaim(t Test, namespace string, storageSize string, opts ...PVCOption) *corev1.PersistentVolumeClaim {
	t.T().Helper()

	pvc := &corev1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    namespace,
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			// AccessModes and StorageClassName will be set by applying options
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse(storageSize),
				},
			},
		},
	}

	// Apply all provided options
	for _, opt := range opts {
		if err := opt.ApplyTo(pvc); err != nil {
			t.T().Fatalf("Error applying PVC option: %v", err)
		}
	}

	pvc, err := t.Client().Core().CoreV1().PersistentVolumeClaims(namespace).Create(t.Ctx(), pvc, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created PersistentVolumeClaim %s/%s successfully", pvc.Namespace, pvc.Name)

	return pvc
}

func GetNodes(t Test) []corev1.Node {
	t.T().Helper()
	nodes, err := t.Client().Core().CoreV1().Nodes().List(t.Ctx(), metav1.ListOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return nodes.Items
}

func GetNodeInternalIP(t Test, node corev1.Node) (IP string) {
	t.T().Helper()

	for _, address := range node.Status.Addresses {
		if address.Type == "InternalIP" {
			IP = address.Address
		}
	}
	t.Expect(IP).Should(gomega.Not(gomega.BeEmpty()), "Node internal IP address not found")

	return
}

func ResourceName(obj any) (string, error) {
	value := reflect.ValueOf(obj)
	if value.Kind() != reflect.Struct {
		return "", fmt.Errorf("input must be a struct")
	}
	return value.FieldByName("Name").String(), nil
}

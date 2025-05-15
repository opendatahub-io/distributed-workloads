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
	"os"
	"path"
	"sync"
	"testing"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
)

type Test interface {
	T() *testing.T
	Ctx() context.Context
	Client() Client
	Config() *rest.Config
	OutputDir() string

	gomega.Gomega

	NewTestNamespace(...Option[*corev1.Namespace]) *corev1.Namespace
	CreateOrGetTestNamespace(...Option[*corev1.Namespace]) *corev1.Namespace
}

type Option[T any] interface {
	ApplyTo(to T) error
}

type ErrorOption[T any] func(to T) error

// nolint: unused
// To be removed when the false-positivity is fixed.
func (o ErrorOption[T]) ApplyTo(to T) error {
	return o(to)
}

var _ Option[any] = ErrorOption[any](nil)

func With(t *testing.T) Test {
	return WithConfig(t, nil)
}

func WithConfig(t *testing.T, cfg *rest.Config) Test {
	t.Helper()
	ctx := context.Background()

	if deadline, ok := t.Deadline(); ok {
		withDeadline, cancel := context.WithDeadline(ctx, deadline)
		t.Cleanup(cancel)
		ctx = withDeadline
	}

	return &T{
		WithT: gomega.NewWithT(t),
		t:     t,
		ctx:   ctx,
		cfg:   cfg,
	}
}

func WithNamespaceName(name string) Option[*corev1.Namespace] {
	return ErrorOption[*corev1.Namespace](func(ns *corev1.Namespace) error {
		ns.Name = name
		return nil
	})
}

type T struct {
	*gomega.WithT
	t *testing.T
	// nolint: containedctx
	ctx       context.Context
	client    Client
	cfg       *rest.Config
	outputDir string
	once      struct {
		client    sync.Once
		outputDir sync.Once
	}
}

func (t *T) T() *testing.T {
	return t.t
}

func (t *T) Ctx() context.Context {
	return t.ctx
}

func (t *T) Client() Client {
	t.T().Helper()
	t.once.client.Do(func() {
		if t.client == nil {
			c, cfg, err := newTestClient(t.cfg)
			if err != nil {
				t.T().Fatalf("Error creating client: %v", err)
			}
			t.client = c
			t.cfg = cfg
		}
	})
	return t.client
}

func (t *T) Config() *rest.Config {
	t.T().Helper()
	// Invoke Client() function to initialize client
	t.Client()
	return t.cfg
}

func (t *T) OutputDir() string {
	t.T().Helper()
	t.once.outputDir.Do(func() {
		if parent, ok := os.LookupEnv(CodeFlareTestOutputDir); ok {
			if !path.IsAbs(parent) {
				if cwd, err := os.Getwd(); err == nil {
					// best effort to output the parent absolute path
					parent = path.Join(cwd, parent)
				}
			}
			t.T().Logf("Creating output directory in parent directory: %s", parent)
			dir, err := os.MkdirTemp(parent, t.T().Name())
			if err != nil {
				t.T().Fatalf("Error creating output directory: %v", err)
			}
			t.outputDir = dir
		} else {
			t.T().Logf("Creating ephemeral output directory as %s env variable is unset", CodeFlareTestOutputDir)
			t.outputDir = t.T().TempDir()
		}
		t.T().Logf("Output directory has been created at: %s", t.outputDir)
	})
	return t.outputDir
}

func (t *T) NewTestNamespace(options ...Option[*corev1.Namespace]) *corev1.Namespace {
	t.T().Helper()
	namespace := createTestNamespace(t, options...)
	t.T().Cleanup(func() {
		storeAllPodLogs(t, namespace)
		storeEvents(t, namespace)
		deleteTestNamespace(t, namespace)
	})
	return namespace
}

func (t *T) CreateOrGetTestNamespace(options ...Option[*corev1.Namespace]) *corev1.Namespace {
	t.T().Helper()

	testNamespaceName, testNamespaceNameExists := GetTestNamespaceName()

	if testNamespaceNameExists {
		// Verify that the namespace really exists and return it, create it if doesn't exist yet
		namespace, err := t.Client().Core().CoreV1().Namespaces().Get(t.Ctx(), testNamespaceName, metav1.GetOptions{})
		if err == nil {
			t.T().Logf("Using the namespace name which is provided using environment variable..")
			return namespace
		} else if errors.IsNotFound(err) {
			t.T().Logf("%s namespace doesn't exists. Creating ...", testNamespaceName)
			return CreateTestNamespaceWithName(t, testNamespaceName, options...)
		} else {
			t.T().Fatalf("Error retrieving namespace with name `%s`: %v", testNamespaceName, err)
		}
	}
	return t.NewTestNamespace(options...)
}

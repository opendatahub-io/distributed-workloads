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

package support

import (
	"testing"

	fakekfto "github.com/kubeflow/training-operator/pkg/client/clientset/versioned/fake"
	fakeray "github.com/ray-project/kuberay/ray-operator/pkg/client/clientset/versioned/fake"

	"k8s.io/apimachinery/pkg/runtime"
	fakeDynamic "k8s.io/client-go/dynamic/fake"
	fakeCore "k8s.io/client-go/kubernetes/fake"
	fakekueue "sigs.k8s.io/kueue/client-go/clientset/versioned/fake"

	fakeimage "github.com/openshift/client-go/image/clientset/versioned/fake"
	fakeMachine "github.com/openshift/client-go/machine/clientset/versioned/fake"
	fakeroute "github.com/openshift/client-go/route/clientset/versioned/fake"
)

func NewTest(t *testing.T) *T {
	fakeCoreClient := fakeCore.NewSimpleClientset()
	fakemachineClient := fakeMachine.NewSimpleClientset()
	fakeimageClient := fakeimage.NewSimpleClientset()
	fakerouteClient := fakeroute.NewSimpleClientset()
	fakerayClient := fakeray.NewSimpleClientset()
	fakekueueClient := fakekueue.NewSimpleClientset()
	fakeKftoClient := fakekfto.NewSimpleClientset()
	fakedynamicClient := fakeDynamic.NewSimpleDynamicClient(runtime.NewScheme())

	test := With(t).(*T)
	test.client = &testClient{
		core:     fakeCoreClient,
		machine:  fakemachineClient,
		image:    fakeimageClient,
		route:    fakerouteClient,
		ray:      fakerayClient,
		kueue:    fakekueueClient,
		kubeflow: fakeKftoClient,
		dynamic:  fakedynamicClient,
	}
	return test
}

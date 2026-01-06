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
	trainerclient "github.com/kubeflow/trainer/v2/pkg/client/clientset/versioned"
	kubeflowclient "github.com/kubeflow/training-operator/pkg/client/clientset/versioned"
	olmclient "github.com/operator-framework/operator-lifecycle-manager/pkg/api/client/clientset/versioned"
	rayclient "github.com/ray-project/kuberay/ray-operator/pkg/client/clientset/versioned"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	storageclient "k8s.io/client-go/kubernetes/typed/storage/v1"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	jobsetclient "sigs.k8s.io/jobset/client-go/clientset/versioned"
	kueueclient "sigs.k8s.io/kueue/client-go/clientset/versioned"

	imagev1 "github.com/openshift/client-go/image/clientset/versioned"
	machinev1 "github.com/openshift/client-go/machine/clientset/versioned"
	routev1 "github.com/openshift/client-go/route/clientset/versioned"
	kueueoperatorclient "github.com/openshift/kueue-operator/pkg/generated/clientset/versioned"
	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
)

type Client interface {
	Core() kubernetes.Interface
	Trainer() trainerclient.Interface
	Kubeflow() kubeflowclient.Interface
	JobSet() jobsetclient.Interface
	Kueue() kueueclient.Interface
	KueueOperator() kueueoperatorclient.Interface
	Machine() machinev1.Interface
	Route() routev1.Interface
	Image() imagev1.Interface
	Ray() rayclient.Interface
	Dynamic() dynamic.Interface
	Storage() storageclient.StorageV1Interface
	OLM() olmclient.Interface
}

type testClient struct {
	core          kubernetes.Interface
	trainer       trainerclient.Interface
	kubeflow      kubeflowclient.Interface
	jobset        jobsetclient.Interface
	kueue         kueueclient.Interface
	kueueOperator kueueoperatorclient.Interface
	machine       machinev1.Interface
	route         routev1.Interface
	image         imagev1.Interface
	ray           rayclient.Interface
	dynamic       dynamic.Interface
	storage       storageclient.StorageV1Interface
	olm           olmclient.Interface
}

var _ Client = (*testClient)(nil)

func (t *testClient) Core() kubernetes.Interface {
	return t.core
}

func (t *testClient) Trainer() trainerclient.Interface {
	return t.trainer
}

func (t *testClient) Kubeflow() kubeflowclient.Interface {
	return t.kubeflow
}

func (t *testClient) JobSet() jobsetclient.Interface {
	return t.jobset
}

func (t *testClient) Kueue() kueueclient.Interface {
	return t.kueue
}

func (t *testClient) KueueOperator() kueueoperatorclient.Interface {
	return t.kueueOperator
}

func (t *testClient) Machine() machinev1.Interface {
	return t.machine
}

func (t *testClient) Route() routev1.Interface {
	return t.route
}

func (t *testClient) Image() imagev1.Interface {
	return t.image
}

func (t *testClient) Ray() rayclient.Interface {
	return t.ray
}

func (t *testClient) Dynamic() dynamic.Interface {
	return t.dynamic
}

func (t *testClient) Storage() storageclient.StorageV1Interface {
	return t.storage
}

func (t *testClient) OLM() olmclient.Interface {
	return t.olm
}

func newTestClient(cfg *rest.Config) (Client, *rest.Config, error) {
	var err error
	if cfg == nil {
		cfg, err = clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			clientcmd.NewDefaultClientConfigLoadingRules(),
			&clientcmd.ConfigOverrides{},
		).ClientConfig()
		if err != nil {
			return nil, nil, err
		}
	}

	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	trainerClient, err := trainerclient.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	kubeflowClient, err := kubeflowclient.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	jobsetClient, err := jobsetclient.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	kueueClient, err := kueueclient.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	kueueOperatorClient, err := kueueoperatorclient.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	machineClient, err := machinev1.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	routeClient, err := routev1.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	imageClient, err := imagev1.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	rayClient, err := rayclient.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	dynamicClient, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	storageClient, err := storageclient.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	olmClient, err := olmclient.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}

	return &testClient{
		core:          kubeClient,
		trainer:       trainerClient,
		kubeflow:      kubeflowClient,
		jobset:        jobsetClient,
		kueue:         kueueClient,
		kueueOperator: kueueOperatorClient,
		machine:       machineClient,
		route:         routeClient,
		image:         imageClient,
		ray:           rayClient,
		dynamic:       dynamicClient,
		storage:       storageClient,
		olm:           olmClient,
	}, cfg, nil
}

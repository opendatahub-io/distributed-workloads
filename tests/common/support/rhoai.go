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

package support

import (
	"fmt"

	"github.com/onsi/gomega"
	"github.com/operator-framework/api/pkg/operators/v1alpha1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// OdhCsvNamePrefix is the prefix for ODH operator CSV
	OdhCsvNamePrefix = "opendatahub-operator"
	// OdhApplicationsNamespace is the namespace for ODH applications
	OdhApplicationsNamespace = "opendatahub"
	// RhoaiCsvNamePrefix is the prefix for RHOAI operator CSV
	RhoaiCsvNamePrefix = "rhods-operator"
	// RhoaiApplicationsNamespace is the namespace for RHOAI applications
	RhoaiApplicationsNamespace = "redhat-ods-applications"
)

type Product struct {
	ApplicationsNamespace string
	CsvNamePrefix         string
}

var (
	// ODH represents the Open Data Hub product configuration
	ODH = Product{
		ApplicationsNamespace: OdhApplicationsNamespace,
		CsvNamePrefix:         OdhCsvNamePrefix,
	}
	// RHOAI represents the Red Hat OpenShift AI product configuration
	RHOAI = Product{
		ApplicationsNamespace: RhoaiApplicationsNamespace,
		CsvNamePrefix:         RhoaiCsvNamePrefix,
	}

	products = []Product{ODH, RHOAI}
)

// GetProduct returns the product configuration based on the applications namespace.
// Use GetBuildType to identify the build: ODH nightlies can use RHOAI namespaces and CSV names.
func GetProduct(test Test) (*Product, error) {
	test.T().Helper()

	dsciApplicationsNamespace, err := GetApplicationsNamespace(test)
	if err != nil {
		return nil, err
	}

	for _, product := range products {
		if product.ApplicationsNamespace == dsciApplicationsNamespace {
			return &product, nil
		}
	}

	return nil, fmt.Errorf("no product found for applications namespace %s", dsciApplicationsNamespace)
}

// BuildType identifies the distribution selected by the operator subscription.
type BuildType string

const (
	ODHBuild   BuildType = "ODH"
	RHOAIBuild BuildType = "RHOAI"
)

// GetBuildType identifies the build from the operator subscription channel:
// odh-stable is ODH; all other channels are assumed to be RHOAI.
func GetBuildType(test Test) (BuildType, error) {
	test.T().Helper()

	subscriptions, err := test.Client().OLM().OperatorsV1alpha1().Subscriptions(metav1.NamespaceAll).List(
		test.Ctx(), metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to list operator subscriptions: %w", err)
	}

	var subscription *v1alpha1.Subscription
	for i := range subscriptions.Items {
		sub := &subscriptions.Items[i]
		if sub.Spec == nil || (sub.Spec.Package != OdhCsvNamePrefix && sub.Spec.Package != RhoaiCsvNamePrefix) {
			continue
		}
		if subscription != nil {
			return "", fmt.Errorf("multiple ODH/RHOAI subscriptions found: %s/%s and %s/%s",
				subscription.Namespace, subscription.Name, sub.Namespace, sub.Name)
		}
		subscription = sub
	}
	if subscription == nil {
		return "", fmt.Errorf("no ODH/RHOAI operator subscription found")
	}
	channel := subscription.Spec.Channel
	buildType := RHOAIBuild
	if channel == "odh-stable" {
		buildType = ODHBuild
	}
	test.T().Logf("Build installed in cluster: %s; channel=%q; installedCSV=%q",
		buildType, channel, subscription.Status.InstalledCSV)
	return buildType, nil
}

// IsRhoai reports whether the installed operator is a RHOAI build, failing the
// test if the build cannot be identified.
func IsRhoai(test Test) bool {
	test.T().Helper()

	buildType, err := GetBuildType(test)
	test.Expect(err).NotTo(gomega.HaveOccurred(), "Failed to identify installed ODH/RHOAI build")
	return buildType == RHOAIBuild
}

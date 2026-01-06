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

// GetProduct returns the product configuration based on the applications namespace
func GetProduct(test Test) (*Product, error) {
	test.T().Helper()

	dsciApplicationsNamespace, err := GetApplicationsNamespaceFromDSCI(test, DefaultDSCIName)
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

func IsRhoai(test Test) bool {
	test.T().Helper()

	product, err := GetProduct(test)
	if err != nil {
		return false
	}

	return *product == RHOAI
}

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
	"crypto/tls"
	"net/http"

	. "github.com/onsi/gomega"
	prometheusapi "github.com/prometheus/client_golang/api"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusconfig "github.com/prometheus/common/config"
)

var prometheusApiClient prometheusapiv1.API

func GetOpenShiftPrometheusApiClient(t Test) prometheusapiv1.API {
	if prometheusApiClient == nil {
		prometheusOpenShiftRoute := GetRoute(t, "openshift-monitoring", "prometheus-k8s")

		// Skip TLS check to work on clusters with insecure certificates too
		// Functionality intended just for testing purpose, DO NOT USE IN PRODUCTION
		tr := &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			Proxy:           http.ProxyFromEnvironment,
		}
		client, err := prometheusapi.NewClient(prometheusapi.Config{
			Address: "https://" + prometheusOpenShiftRoute.Status.Ingress[0].Host,
			Client:  &http.Client{Transport: prometheusconfig.NewAuthorizationCredentialsRoundTripper("Bearer", prometheusconfig.NewInlineSecret(t.Config().BearerToken), tr)},
		})
		t.Expect(err).NotTo(HaveOccurred())

		prometheusApiClient = prometheusapiv1.NewAPI(client)
	}

	return prometheusApiClient
}

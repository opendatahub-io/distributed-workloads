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
	"crypto/x509"
	"net/http"

	. "github.com/onsi/gomega"
	prometheusapi "github.com/prometheus/client_golang/api"
	prometheusapiv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	prometheusconfig "github.com/prometheus/common/config"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var prometheusApiClient prometheusapiv1.API

func GetOpenShiftPrometheusApiClient(t Test) prometheusapiv1.API {
	if prometheusApiClient == nil {
		prometheusOpenShiftRoute := GetRoute(t, "openshift-monitoring", "prometheus-k8s")
		routeHost := prometheusOpenShiftRoute.Status.Ingress[0].Host
		routerCA, err := t.Client().Core().CoreV1().Secrets("openshift-ingress-operator").Get(
			t.Ctx(), "router-ca", metav1.GetOptions{})
		t.Expect(err).NotTo(HaveOccurred())
		// Keep the system roots because the ingress controller may use a
		// custom/public certificate (for example, Let's Encrypt) instead of
		// the internal router CA. Append the router CA as well for clusters
		// that use the default OpenShift ingress certificate.
		rootCAs, err := x509.SystemCertPool()
		if err != nil || rootCAs == nil {
			rootCAs = x509.NewCertPool()
		}
		t.Expect(rootCAs.AppendCertsFromPEM(routerCA.Data["tls.crt"])).To(BeTrue())

		tr := &http.Transport{
			TLSClientConfig: &tls.Config{RootCAs: rootCAs, ServerName: routeHost},
			Proxy:           http.ProxyFromEnvironment,
		}
		client, err := prometheusapi.NewClient(prometheusapi.Config{
			Address: "https://" + routeHost,
			Client:  &http.Client{Transport: prometheusconfig.NewAuthorizationCredentialsRoundTripper("Bearer", prometheusconfig.NewInlineSecret(t.Config().BearerToken), tr)},
		})
		t.Expect(err).NotTo(HaveOccurred())

		prometheusApiClient = prometheusapiv1.NewAPI(client)
	}

	return prometheusApiClient
}

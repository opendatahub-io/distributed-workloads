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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var prometheusApiClient prometheusapiv1.API

func GetOpenShiftPrometheusApiClient(t Test) prometheusapiv1.API {
	if prometheusApiClient == nil {
		prometheusOpenShiftRoute := GetRoute(t, "openshift-monitoring", "prometheus-k8s")
		routeHost := prometheusOpenShiftRoute.Status.Ingress[0].Host
		bearerToken := t.Config().BearerToken
		if bearerToken == "" {
			// Konflux provisions kubeconfig credentials with a username and
			// password, so there is no user bearer token in rest.Config. Use the
			// Prometheus service account token as the fallback for the protected
			// Prometheus route.
			prometheusServiceAccount, err := t.Client().Core().CoreV1().ServiceAccounts("openshift-monitoring").Get(
				t.Ctx(), "prometheus-k8s", metav1.GetOptions{})
			t.Expect(err).NotTo(HaveOccurred())
			bearerToken = CreateToken(t, "openshift-monitoring", prometheusServiceAccount)
		}
		routerCA, err := t.Client().Core().CoreV1().Secrets("openshift-ingress-operator").Get(
			t.Ctx(), "router-ca", metav1.GetOptions{})
		t.Expect(err).NotTo(HaveOccurred())
		// Keep the system roots because the ingress controller may use a
		// public certificate (for example, Let's Encrypt). The
		// default-ingress-cert ConfigMap contains the CA bundle for custom
		// ingress certificates used by managed OpenShift clusters. The
		// router-ca Secret covers clusters using the operator-generated
		// default ingress certificate.
		rootCAs, err := x509.SystemCertPool()
		if err != nil || rootCAs == nil {
			rootCAs = x509.NewCertPool()
		}

		activeIngressCA, err := t.Client().Core().CoreV1().ConfigMaps("openshift-config-managed").Get(
			t.Ctx(), "default-ingress-cert", metav1.GetOptions{})
		if err == nil {
			activeIngressCABundle := activeIngressCA.Data["ca-bundle.crt"]
			if activeIngressCABundle != "" {
				t.Expect(rootCAs.AppendCertsFromPEM([]byte(activeIngressCABundle))).To(BeTrue())
			}
		} else if !apierrors.IsNotFound(err) {
			t.Expect(err).NotTo(HaveOccurred())
		}

		t.Expect(rootCAs.AppendCertsFromPEM(routerCA.Data["tls.crt"])).To(BeTrue())

		tr := &http.Transport{
			TLSClientConfig: &tls.Config{RootCAs: rootCAs, ServerName: routeHost},
			Proxy:           http.ProxyFromEnvironment,
		}
		client, err := prometheusapi.NewClient(prometheusapi.Config{
			Address: "https://" + routeHost,
			Client:  &http.Client{Transport: prometheusconfig.NewAuthorizationCredentialsRoundTripper("Bearer", prometheusconfig.NewInlineSecret(bearerToken), tr)},
		})
		t.Expect(err).NotTo(HaveOccurred())

		prometheusApiClient = prometheusapiv1.NewAPI(client)
	}

	return prometheusApiClient
}

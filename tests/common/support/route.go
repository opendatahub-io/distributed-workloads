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
	"net/http"
	"net/url"

	"github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	routev1 "github.com/openshift/api/route/v1"
)

func Route(t Test, namespace, name string) func(g gomega.Gomega) *routev1.Route {
	return func(g gomega.Gomega) *routev1.Route {
		route, err := t.Client().Route().RouteV1().Routes(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return route
	}
}

func GetRoute(t Test, namespace, name string) *routev1.Route {
	t.T().Helper()
	return Route(t, namespace, name)(t)
}

func ExposeServiceByRoute(t Test, name string, namespace string, serviceName string, servicePort string) url.URL {
	r := &routev1.Route{
		TypeMeta: metav1.TypeMeta{
			APIVersion: routev1.SchemeGroupVersion.String(),
			Kind:       "Route",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: routev1.RouteSpec{
			To: routev1.RouteTargetReference{
				Name: serviceName,
			},
			Port: &routev1.RoutePort{
				TargetPort: intstr.FromString(servicePort),
			},
		},
	}

	_, err := t.Client().Route().RouteV1().Routes(r.Namespace).Create(t.Ctx(), r, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created Route %s/%s successfully", r.Namespace, r.Name)

	t.T().Logf("Waiting for Route %s/%s to be available", r.Namespace, r.Name)
	t.Eventually(Route(t, r.Namespace, r.Name), TestTimeoutLong).
		Should(gomega.WithTransform(ConditionStatus(routev1.RouteAdmitted), gomega.Equal(corev1.ConditionTrue)))

	// Retrieve hostname
	r, err = t.Client().Route().RouteV1().Routes(r.Namespace).Get(t.Ctx(), r.Name, metav1.GetOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	hostname := r.Status.Ingress[0].Host

	// Wait for expected HTTP code
	t.Eventually(func() (int, error) {
		resp, err := http.Get("http://" + hostname)
		if err != nil {
			return -1, err
		}
		return resp.StatusCode, nil
	}, TestTimeoutLong).Should(gomega.Not(gomega.Equal(503)))

	r = GetRoute(t, r.Namespace, r.Name)
	routeURL := url.URL{
		Scheme: "http",
		Host:   r.Status.Ingress[0].Host,
	}

	return routeURL
}

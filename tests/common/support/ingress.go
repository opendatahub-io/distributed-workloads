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
	"net/url"

	"github.com/onsi/gomega"

	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func Ingress(t Test, namespace, name string) func(g gomega.Gomega) *networkingv1.Ingress {
	return func(g gomega.Gomega) *networkingv1.Ingress {
		ingress, err := t.Client().Core().NetworkingV1().Ingresses(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return ingress
	}
}

func GetIngress(t Test, namespace, name string) *networkingv1.Ingress {
	t.T().Helper()
	return Ingress(t, namespace, name)(t)
}

func LoadBalancerIngresses(ingress *networkingv1.Ingress) []networkingv1.IngressLoadBalancerIngress {
	return ingress.Status.LoadBalancer.Ingress
}

func ExposeServiceByIngress(t Test, name string, namespace string, serviceName string, servicePort string) url.URL {
	ingress := &networkingv1.Ingress{
		TypeMeta: metav1.TypeMeta{
			APIVersion: networkingv1.SchemeGroupVersion.String(),
			Kind:       "Ingress",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Annotations: map[string]string{
				"nginx.ingress.kubernetes.io/use-regex":      "true",
				"nginx.ingress.kubernetes.io/rewrite-target": "/$2",
			},
		},
		Spec: networkingv1.IngressSpec{
			Rules: []networkingv1.IngressRule{
				networkingv1.IngressRule{
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								networkingv1.HTTPIngressPath{
									Path:     "/" + name + "(/|$)(.*)",
									PathType: Ptr(networkingv1.PathTypePrefix),
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: serviceName,
											Port: networkingv1.ServiceBackendPort{
												Name: servicePort,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	_, err := t.Client().Core().NetworkingV1().Ingresses(ingress.Namespace).Create(t.Ctx(), ingress, metav1.CreateOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	t.T().Logf("Created Ingress %s/%s successfully", ingress.Namespace, ingress.Name)

	t.T().Logf("Waiting for Ingress %s/%s to be admitted", ingress.Namespace, ingress.Name)
	t.Eventually(Ingress(t, ingress.Namespace, ingress.Name), TestTimeoutMedium).
		Should(gomega.WithTransform(LoadBalancerIngresses, gomega.HaveLen(1)))

	ingress = GetIngress(t, ingress.Namespace, ingress.Name)

	ingressURL := url.URL{
		Scheme: "http",
		Host:   ingress.Status.LoadBalancer.Ingress[0].IP,
		Path:   name,
	}
	return ingressURL
}

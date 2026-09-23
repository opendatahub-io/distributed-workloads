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
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/onsi/gomega"
	"github.com/operator-framework/api/pkg/operators/v1alpha1"
	olmclient "github.com/operator-framework/operator-lifecycle-manager/pkg/api/client/clientset/versioned"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	restfake "k8s.io/client-go/rest/fake"
)

func TestGetBuildTypeFromSubscription(t *testing.T) {
	// ODH nightlies can use RHOAI namespaces; detection needs only the subscription.
	t.Setenv(ApplicationsNamespaceEnvVar, RhoaiApplicationsNamespace)
	for _, tc := range []struct {
		name      string
		channel   string
		buildType BuildType
	}{
		{"ODH nightly", "odh-stable", ODHBuild},
		{"RHOAI EA", "beta", RHOAIBuild},
		{"RHOAI GA", "stable-3.4", RHOAIBuild},
		{"RHOAI GA multi-digit version", "stable-2.25", RHOAIBuild},
	} {
		t.Run(tc.name, func(t *testing.T) {
			test := NewTest(t)
			sub := buildSubscription(tc.channel)
			// Ignore unrelated and incomplete subscriptions, regardless of name.
			unrelated := v1alpha1.Subscription{
				ObjectMeta: metav1.ObjectMeta{Name: "rhods-operator-unrelated"},
				Spec:       &v1alpha1.SubscriptionSpec{Package: "other-operator"},
			}
			mockSubscriptions(test, []v1alpha1.Subscription{unrelated, {}, sub}, false)

			buildType, err := GetBuildType(test)
			test.Expect(err).NotTo(gomega.HaveOccurred())
			test.Expect(buildType).To(gomega.Equal(tc.buildType))
			test.Expect(IsRhoai(test)).To(gomega.Equal(tc.buildType == RHOAIBuild))
		})
	}
}

func TestGetBuildTypeErrors(t *testing.T) {
	sub := buildSubscription("odh-stable")
	other := sub.DeepCopy()
	other.Namespace = "other-operators"
	for _, tc := range []struct {
		name          string
		subscriptions []v1alpha1.Subscription
		forbidden     bool
		wantError     string
	}{
		{"no subscription", nil, false, "no ODH/RHOAI operator subscription"},
		{"ambiguous subscriptions", []v1alpha1.Subscription{sub, *other}, false, "multiple ODH/RHOAI subscriptions"},
		{"API forbidden", nil, true, "failed to list operator subscriptions"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			test := NewTest(t)
			mockSubscriptions(test, tc.subscriptions, tc.forbidden)

			buildType, err := GetBuildType(test)
			test.Expect(err).To(gomega.MatchError(gomega.ContainSubstring(tc.wantError)))
			test.Expect(buildType).To(gomega.BeEmpty())
		})
	}
}

func buildSubscription(channel string) v1alpha1.Subscription {
	return v1alpha1.Subscription{
		ObjectMeta: metav1.ObjectMeta{Name: "rhoai-operator-dev", Namespace: "redhat-ods-operator"},
		Spec:       &v1alpha1.SubscriptionSpec{Package: RhoaiCsvNamePrefix, Channel: channel},
	}
}

// Exercise the real OLM client without contacting a cluster.
func mockSubscriptions(test *T, subscriptions []v1alpha1.Subscription, forbidden bool) {
	test.T().Helper()

	httpClient := restfake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		test.Expect(req.Method).To(gomega.Equal(http.MethodGet))
		test.Expect(req.URL.Path).To(gomega.Equal("/apis/operators.coreos.com/v1alpha1/subscriptions"))
		var body any = &v1alpha1.SubscriptionList{Items: subscriptions}
		statusCode := http.StatusOK
		if forbidden {
			statusCode = http.StatusForbidden
			body = &metav1.Status{Status: metav1.StatusFailure, Reason: metav1.StatusReasonForbidden, Code: http.StatusForbidden}
		}
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		return &http.Response{
			StatusCode: statusCode,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(string(data))),
		}, nil
	})
	client, err := olmclient.NewForConfigAndClient(&rest.Config{Host: "https://cluster.example"}, httpClient)
	test.Expect(err).NotTo(gomega.HaveOccurred())
	test.client.(*testClient).olm = client
}

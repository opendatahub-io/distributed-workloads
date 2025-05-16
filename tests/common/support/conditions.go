/*
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package support

import (
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"

	routev1 "github.com/openshift/api/route/v1"
)

type conditionType interface {
	~string
}

func ConditionStatus[T conditionType](conditionType T) func(any) corev1.ConditionStatus {
	return func(object any) corev1.ConditionStatus {
		switch o := object.(type) {

		case *batchv1.Job:
			if c := getJobCondition(o.Status.Conditions, batchv1.JobConditionType(conditionType)); c != nil {
				return c.Status
			}
		case *appsv1.Deployment:
			if c := getDeploymentCondition(o.Status.Conditions, appsv1.DeploymentConditionType(conditionType)); c != nil {
				return c.Status
			}
		case *routev1.Route:
			if len(o.Status.Ingress) == 0 {
				// Route is not initialized yet
				break
			}
			if c := getRouteCondition(o.Status.Ingress[0].Conditions, routev1.RouteIngressConditionType(conditionType)); c != nil {
				return c.Status
			}
		}

		return corev1.ConditionUnknown
	}
}

// TODO: to be replaced with a generic version once common struct fields of a type set can be used.
// See https://github.com/golang/go/issues/48522
func getJobCondition(conditions []batchv1.JobCondition, conditionType batchv1.JobConditionType) *batchv1.JobCondition {
	for _, c := range conditions {
		if c.Type == conditionType {
			return &c
		}
	}
	return nil
}

func getDeploymentCondition(conditions []appsv1.DeploymentCondition, conditionType appsv1.DeploymentConditionType) *appsv1.DeploymentCondition {
	for _, c := range conditions {
		if c.Type == conditionType {
			return &c
		}
	}
	return nil
}

func getRouteCondition(conditions []routev1.RouteIngressCondition, conditionType routev1.RouteIngressConditionType) *routev1.RouteIngressCondition {
	for _, c := range conditions {
		if c.Type == conditionType {
			return &c
		}
	}
	return nil
}

/*
Copyright 2025.

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
	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	machinev1beta1 "github.com/openshift/api/machine/v1beta1"
)

func GetMachineSets(t Test) ([]machinev1beta1.MachineSet, error) {
	ms, err := t.Client().Machine().MachineV1beta1().MachineSets("openshift-machine-api").List(t.Ctx(), metav1.ListOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return ms.Items, err
}

func Machines(t Test, machineSetName string) func(g gomega.Gomega) []machinev1beta1.Machine {
	return func(g gomega.Gomega) []machinev1beta1.Machine {
		machine, err := t.Client().Machine().MachineV1beta1().Machines("openshift-machine-api").List(t.Ctx(), metav1.ListOptions{LabelSelector: "machine.openshift.io/cluster-api-machineset=" + machineSetName})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return machine.Items
	}
}

func GetMachines(t Test, machineSetName string) []machinev1beta1.Machine {
	t.T().Helper()
	return Machines(t, machineSetName)(t)
}

func MachineSetId(machineSet machinev1beta1.MachineSet) string {
	return machineSet.Name
}

func MachineSet(t Test, namespace string, machineSetName string) func(g gomega.Gomega) *machinev1beta1.MachineSet {
	return func(g gomega.Gomega) *machinev1beta1.MachineSet {
		machineset, err := t.Client().Machine().MachineV1beta1().MachineSets(namespace).Get(t.Ctx(), machineSetName, metav1.GetOptions{})
		g.Expect(err).NotTo(gomega.HaveOccurred())
		return machineset
	}
}

func MachineSetReplicas(machineSet *machinev1beta1.MachineSet) *int32 {
	return machineSet.Spec.Replicas
}

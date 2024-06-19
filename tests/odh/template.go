/*
Copyright 2024.

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

package odh

import (
	"bytes"
	"html/template"

	"github.com/onsi/gomega"
	"github.com/project-codeflare/codeflare-common/support"
)

func ParseTemplate(t support.Test, inputTemplate []byte, props interface{}) []byte {
	t.T().Helper()

	// Parse input template
	parsedTemplate, err := template.New("template").Parse(string(inputTemplate))
	t.Expect(err).NotTo(gomega.HaveOccurred())

	// Filter template and store results to the buffer
	buffer := new(bytes.Buffer)
	err = parsedTemplate.Execute(buffer, props)
	t.Expect(err).NotTo(gomega.HaveOccurred())

	return buffer.Bytes()
}

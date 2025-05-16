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
	"testing"
)

func TestGetDefaultEventValueIfNull(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"World", "World"},
	}

	for _, test := range tests {
		actual := getDefaultEventValueIfNull(test.input)
		if actual != test.expected {
			t.Errorf("getDefaultEventValueIfNull(%s) = %s; expected %s", test.input, actual, test.expected)
		}
	}
}

func TestGetWhitespaceStr(t *testing.T) {
	tests := []struct {
		size     int
		expected string
	}{
		{0, ""},
		{1, " "},
		{5, "     "},
		{10, "          "},
	}

	for _, test := range tests {
		actual := getWhitespaceStr(test.size)
		if actual != test.expected {
			t.Errorf("getWhitespaceStr(%d) = %s; expected %s", test.size, actual, test.expected)
		}
	}
}

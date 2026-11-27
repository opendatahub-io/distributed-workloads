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
	"time"
)

const timingSchemaVersion = "dw-timing/v1"

// TimingEvent is the machine-readable representation of one measured test
// phase. The log prefix makes these records easy to extract from CI output,
// while the JSON payload keeps the format extensible for later resource and
// accelerator metadata.
type TimingEvent struct {
	Schema          string            `json:"schema"`
	Test            string            `json:"test"`
	Phase           string            `json:"phase"`
	StartedAt       time.Time         `json:"started_at"`
	FinishedAt      time.Time         `json:"finished_at"`
	DurationSeconds float64           `json:"duration_seconds"`
	Namespace       string            `json:"namespace,omitempty"`
	Resource        string            `json:"resource,omitempty"`
	Attributes      map[string]string `json:"attributes,omitempty"`
}

// StartTiming starts a phase timer and returns a function that emits the
// completed event. Call the returned function after the phase succeeds. The
// helper intentionally logs through testing.T so the records are preserved by
// the existing Go test and JUnit/CI artifact flow.
func StartTiming(test Test, phase, namespace, resource string, attributes map[string]string) func() {
	test.T().Helper()
	startedAt := time.Now().UTC()

	return func() {
		test.T().Helper()
		finishedAt := time.Now().UTC()
		event := TimingEvent{
			Schema:          timingSchemaVersion,
			Test:            test.T().Name(),
			Phase:           phase,
			StartedAt:       startedAt,
			FinishedAt:      finishedAt,
			DurationSeconds: finishedAt.Sub(startedAt).Seconds(),
			Namespace:       namespace,
			Resource:        resource,
			Attributes:      attributes,
		}

		payload, err := json.Marshal(event)
		if err != nil {
			test.T().Logf("DW_TIMING_ERROR phase=%s error=%v", phase, err)
			return
		}
		test.T().Logf("DW_TIMING %s", payload)
	}
}
